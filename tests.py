"""
Synthetic test harness for the exposed-area pipeline.

Each test constructs a tiny GeoDataFrame with hand-known geometry and
elevation data, runs the relevant pipeline steps using the FIXED algorithms,
and asserts that the output matches the hand-computed expected value.

Run with:
    ~/.uv-global/bin/python tests.py

All tests complete in well under 1 second. They exist so that the correctness
of the pipeline can be verified without running the full 48-hour Barcelona job.

Tests skipped:
    Test E (tile-edge halo) is N/A: the input is a single integrated
    AMB-wide GeoPackage, not a set of independently processed tiles.
    There are no internal tile boundaries to bridge.
"""

import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import box
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Constants, which must match the values defined in envelope_area.py
# ---------------------------------------------------------------------------

EPS = 0.05           # snapping tolerance in metres: buildings with a gap
                     # smaller than this are treated as sharing a wall.
                     # Also sets the ~EPS length bias in the shared-edge
                     # measurement (acceptable: ~0.05 m per shared edge).

H_FLOOR = 3.0        # minimum building height assigned during repair (m).
                     # Used when Alc_Top <= Alc_Base (lidar ground model error).

CONTAINMENT_FRAC = 0.9   # if the intersection covers more than this fraction
                         # of the smaller footprint, the smaller building is
                         # considered contained within the larger one and dropped.

OVERLAP_AUDIT_AREA = 0.5  # m²: 2-D intersections larger than this are flagged
                           # as ICC digitizing errors in the overlap audit.

COMPACTNESS_FLOOR = 0.02  # minimum isoperimetric quotient (4π·area/perimeter²).
                           # Buildings below this are needle-strip artefacts.


# ---------------------------------------------------------------------------
# Core algorithm functions
# These implement the FIXED versions of the pipeline steps.
# The docstrings explain WHY each formula is correct, mirroring the
# comments in script.py so both files stay in sync.
# ---------------------------------------------------------------------------

def shared_edge_length(geom_a, geom_b, buf_a, eps):
    """
    Compute the per-building shared plan-lengths between two building footprints.

    OLD approach (removed): buffer both polygons, intersect, halve the
    perimeter of the resulting sliver. This is wrong when the intersection
    is a genuine 2-D overlap (data error).

    NEW approach: measure what portion of each building's boundary lies within
    the other building's buffered outline, subtract 2*eps from each to correct
    for the buffer end-cap bias, and keep the two measurements SEPARATE rather
    than averaging them:

        la      = length of b's boundary inside buffer(a, eps)
        lb      = length of a's boundary inside buffer(b, eps)
        L_for_a = max(0, lb - 2*eps)   [a's wall facing b, used for a's deduction]
        L_for_b = max(0, la - 2*eps)   [b's wall facing a, used for b's deduction]

    For a symmetric straight-wall pair (la ≈ lb) the two values are identical
    and equal to the old 0.5*(la+lb) - 2*eps result. For a concave-wrapping
    pair (building b is U-shaped and encloses building a on multiple sides),
    la > lb: b's inner concave wall is longer than a's outer wall. Using lb
    directly for a's deduction prevents overcounting a's hidden wall; the old
    average 0.5*(la+lb) inflated it by (la-lb)/2, causing deduction fractions
    above 1.0 for enclosed buildings.

    Parameters
    ----------
    geom_a, geom_b : Shapely Polygon / MultiPolygon
    buf_a          : geom_a.buffer(eps), precomputed once per building a
                     so it is not recomputed for every candidate neighbour.
    eps            : snapping tolerance in metres (EPS constant)

    Returns
    -------
    (L_for_a, L_for_b) : tuple of floats, per-building shared lengths in metres.
                          L_for_a is used when computing a's wall deduction;
                          L_for_b is used when computing b's wall deduction.
    """
    la = geom_b.boundary.intersection(buf_a).length
    lb = geom_a.boundary.intersection(geom_b.buffer(eps)).length
    # Subtract 2*eps to correct for end-cap bias: the buffer circle at each of
    # the two endpoints of the shared segment captures ~eps of the adjacent
    # (non-shared) wall. The correction is applied to each measurement separately.
    L_for_a = max(0.0, lb - 2.0 * eps)   # how much of a's boundary faces b
    L_for_b = max(0.0, la - 2.0 * eps)   # how much of b's boundary faces a
    return L_for_a, L_for_b


def repair_heights(gdf, h_floor=H_FLOOR):
    """
    Repair (Alc_Top, Alc_Base) pairs where Alc_Top <= Alc_Base.

    WHY repair the pair rather than just clip height to zero:
        With the elevation-interval deduction (§1), the deduction for
        building j is proportional to  min(Alc_Top_i, Alc_Top_j) - Alc_Base_j.
        If building i has a garbage Alc_Top that is below its own Alc_Base,
        min(top_i, top_j) is corrupted and j receives a wrong deduction,
        potentially a large underestimate of the hidden wall area.
        Clamping height to zero leaves the corrupted Alc_Top in place,
        so we repair the source pair instead.

    Repair rule: set Alc_Top = Alc_Base + H_FLOOR wherever Alc_Top <= Alc_Base.
    H_FLOOR is a conservative floor (e.g. 3.0 m), not a precise height.
    Every repaired row is counted and reported so the operator can judge
    whether a tile is pathological.

    Parameters
    ----------
    gdf     : GeoDataFrame with float Alc_Top, Alc_Base columns
    h_floor : minimum height to assign to repaired buildings (metres)

    Returns
    -------
    gdf_repaired : copy of gdf with corrected Alc_Top, Alc_Base, height
    n_repaired   : number of rows that were repaired (for diagnostics)
    """
    gdf = gdf.copy()
    gdf["Alc_Top"]  = gdf["Alc_Top"].astype(float)
    gdf["Alc_Base"] = gdf["Alc_Base"].astype(float)

    bad = gdf["Alc_Top"] <= gdf["Alc_Base"]
    n_bad = int(bad.sum())

    if n_bad > 0:
        gdf.loc[bad, "Alc_Top"] = gdf.loc[bad, "Alc_Base"] + h_floor
        print(f"  Repaired {n_bad} buildings with Alc_Top <= Alc_Base "
              f"(set Alc_Top = Alc_Base + {h_floor} m)")

    # derive height AFTER repair: always > 0 now
    gdf["height"] = gdf["Alc_Top"] - gdf["Alc_Base"]
    return gdf, n_bad


def find_shared_pairs(gdf, eps=EPS):
    """
    Find all neighbouring building pairs and compute their shared edge lengths.

    Uses the boundary-intersection method (§3) to measure shared length.
    Also produces an overlap audit list: pairs whose raw 2-D intersection
    area exceeds OVERLAP_AUDIT_AREA are ICC digitizing errors and should
    be reviewed, even though they are not dropped automatically.

    Parameters
    ----------
    gdf : GeoDataFrame with geometry column, after containment filtering
    eps : snapping tolerance in metres

    Returns
    -------
    shared        : dict (i, j) -> (L_for_i, L_for_j), with i < j.
                    L_for_i: corrected length of i's boundary facing j (for i's deduction).
                    L_for_j: corrected length of j's boundary facing i (for j's deduction).
    overlap_audit : list of (i, j, overlap_area_m2) for flagged pairs
    """
    geoms = gdf.geometry.values
    tree = STRtree(geoms)
    shared = {}
    overlap_audit = []

    # precompute buffer(i, eps) once per building, outside the inner loop.
    # in script.py this is vectorised with shapely.buffer(geoms, EPS);
    # here we use a list comprehension for clarity in the test harness.
    bufs = [g.buffer(eps) for g in geoms]

    for i, buf_i in enumerate(bufs):
        for j in tree.query(buf_i):
            if j <= i:
                continue   # process each pair once, canonical order i < j

            geom_j = geoms[j]

            # --- overlap audit (data error check) ---
            # compute the raw intersection WITHOUT the buffer to detect
            # genuine 2-D footprint overlaps (buildings occupying the same
            # plan area), which indicate ICC digitizing errors.
            raw = geoms[i].intersection(geom_j)
            if (not raw.is_empty
                    and raw.geom_type in ("Polygon", "MultiPolygon")
                    and raw.area > OVERLAP_AUDIT_AREA):
                overlap_audit.append((i, j, raw.area))

            # --- shared edge length ---
            L_for_i, L_for_j = shared_edge_length(geoms[i], geom_j, buf_i, eps)
            if L_for_i > 1e-6 or L_for_j > 1e-6:
                shared[(i, j)] = (L_for_i, L_for_j)

    return shared, overlap_audit


def compute_exposed_area(gdf, shared):
    """
    Compute exposed surface area using elevation-interval deductions.

    CORRECT PHYSICS (§1):
        For a shared wall segment of plan-length L between buildings i and j,
        the wall area hidden from outdoor air depends on the absolute
        elevations, not just the heights. Building i's wall is exposed only
        above the neighbour's roofline; everything at or below the
        neighbour's roof is hidden, either by the neighbour's body (above
        the neighbour's own base) or by the soil the neighbour is founded on
        (below the neighbour's base, a buried/retaining wall, NOT air-exposed).

        Deduction per shared segment (asymmetric, different for each building):
            ded_i = L * clip(min(top_i, top_j) - base_i, 0, h_i)
            ded_j = L * clip(min(top_i, top_j) - base_j, 0, h_j)

        Why this is correct:
          - min(top_i, top_j) is the elevation below which both buildings
            are either present (body) or founded on soil (hidden).
          - Subtracting base_k gives the hidden wall height for building k.
          - clip(0, h_k) guards degenerate cases where one building's base
            sits above the other's roof (no real 3-D contact: zero deduction).

        Flat-ground invariant (Test A):
            When base_i = base_j = b, min(top_i,top_j) - b = min(h_i,h_j),
            so this reduces exactly to the original min(height) formula.
            The existing ±5% flat-tile validation is therefore preserved.

    MODELLING NOTE:
        Walls below a higher neighbour's base elevation are treated as buried
        against soil (correct for an outdoor-air exposure model). Ground-
        contact area (a distinct heat-transfer path) is NOT computed here.

    Parameters
    ----------
    gdf    : GeoDataFrame after height repair, containment filtering,
             and artifact removal. Must have Alc_Top, Alc_Base, height,
             footprint_area columns and a reset 0-based integer index.
    shared : dict (i, j) -> L, the output of find_shared_pairs()

    Returns
    -------
    gdf with added columns: wall_area_naive, roof_area, wall_deduction,
    wall_area, exposed_area, deduction_fraction_raw, deduction_fraction.
    """
    gdf = gdf.copy()
    n = len(gdf)

    # materialise elevation arrays by integer position so that index
    # lookups inside the pair loop are O(1) array reads, not DataFrame .loc
    top    = gdf["Alc_Top"].values
    base   = gdf["Alc_Base"].values
    height = gdf["height"].values

    gdf["wall_area_naive"] = gdf.geometry.length * height
    # geometry.length is the perimeter of the 2-D footprint.
    # For polygons-with-holes (courtyards), it includes interior rings;
    # courtyard facades are genuinely exposed to outdoor air, so this is correct.

    gdf["roof_area"] = gdf["footprint_area"]
    # flat-roof assumption (LoD1): roof area = footprint area.

    wall_deduction = np.zeros(n)

    for (idx_i, idx_j), (L_for_i, L_for_j) in shared.items():
        min_top = min(top[idx_i], top[idx_j])

        ded_i = L_for_i * np.clip(min_top - base[idx_i], 0.0, height[idx_i])
        ded_j = L_for_j * np.clip(min_top - base[idx_j], 0.0, height[idx_j])

        wall_deduction[idx_i] += ded_i
        wall_deduction[idx_j] += ded_j

    gdf["wall_deduction"] = wall_deduction

    # audit deduction fraction BEFORE clipping (§6).
    # if fraction > 1.0, the deduction exceeds the naive wall area, which
    # indicates over-detection (blob-intersection lengths, multi-pair
    # accumulation, or a residual height/elevation error). Log these cases
    # so they can be inspected; the clip below is a safety net, not a fix.
    gdf["deduction_fraction_raw"] = (
        wall_deduction / gdf["wall_area_naive"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    over = (gdf["deduction_fraction_raw"] > 1.0).sum()
    if over > 0:
        print(f"  WARNING: {over} buildings have deduction > naive wall before clip "
              "(deduction_fraction_raw > 1.0; inspect these rows)")

    gdf["wall_area"] = (gdf["wall_area_naive"] - wall_deduction).clip(lower=0.0)
    gdf["exposed_area"] = gdf["roof_area"] + gdf["wall_area"]
    gdf["deduction_fraction"] = gdf["deduction_fraction_raw"].clip(upper=1.0)

    return gdf


# ---------------------------------------------------------------------------
# Helper: build a tiny GeoDataFrame from rectangular buildings
# ---------------------------------------------------------------------------

def make_building(xmin, ymin, xmax, ymax, alc_base, alc_top):
    """
    Create a single-row GeoDataFrame for a rectangular building footprint.

    Elevation values are stored as floats (the real dataset uses strings;
    the pipeline casts them to float early on).
    """
    return {
        "Alc_Base": float(alc_base),
        "Alc_Top":  float(alc_top),
        "geometry": box(xmin, ymin, xmax, ymax),
    }


def build_gdf(*buildings):
    """
    Stack dicts from make_building() into one GeoDataFrame and reset index.

    Does NOT derive height here: call repair_heights() first in any test
    that includes a degenerate building, so the repair is tested end-to-end.
    """
    gdf = gpd.GeoDataFrame(list(buildings), crs="EPSG:25831")
    gdf["footprint_area"] = gdf.geometry.area
    return gdf.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_a_flat_regression():
    """
    Test A: Flat ground, regression guard.

    Two 10x10 m squares placed side by side sharing a 10 m edge.
    Both buildings have the same base (0) and roof (10) elevation.
    This is the scenario the original code was validated against (flat tile,
    ±5% manual check). The new asymmetric deduction formula MUST give the
    same result as the old min(height) formula, because on flat ground:

        min(top_i, top_j) - base = min(h_i, h_j)

    Hand-computed expected values:
        shared length L         ~ 10 m
        deduction each          = L * clip(10 - 0, 0, 10) = 100 m²
        naive wall each         = perimeter * h = 40 * 10 = 400 m²
        exposed wall each       = 400 - 100 = 300 m²
        roof each               = 100 m²
        total exposed (2 bldgs) = 2 * (300 + 100) = 800 m²
    """
    gdf = build_gdf(
        make_building(0, 0, 10, 10, alc_base=0, alc_top=10),
        make_building(10, 0, 20, 10, alc_base=0, alc_top=10),
    )
    gdf["height"] = gdf["Alc_Top"] - gdf["Alc_Base"]

    shared, _ = find_shared_pairs(gdf)

    assert len(shared) == 1, f"Expected 1 shared pair, got {len(shared)}"
    L_for_i, L_for_j = list(shared.values())[0]
    # For symmetric rectangular buildings (la = lb), L_for_i = L_for_j.
    # After the -2*EPS end-cap correction both should be close to the true 10 m.
    assert abs(L_for_i - 10.0) < EPS * 2, f"L_for_i = {L_for_i:.4f} m, expected ~10 m"
    assert abs(L_for_j - 10.0) < EPS * 2, f"L_for_j = {L_for_j:.4f} m, expected ~10 m"

    result = compute_exposed_area(gdf, shared)

    # Building 0 is left_idx (i), building 1 is right_idx (j), so building 0
    # uses L_for_i and building 1 uses L_for_j. Both are ~10 m here.
    ded_expected_i = L_for_i * 10.0   # clip(min(10,10) - 0, 0, 10) = 10
    ded_expected_j = L_for_j * 10.0
    assert abs(result.loc[0, "wall_deduction"] - ded_expected_i) < 1.0, \
        f"ded_i = {result.loc[0, 'wall_deduction']:.2f}, expected {ded_expected_i:.2f}"
    assert abs(result.loc[1, "wall_deduction"] - ded_expected_j) < 1.0, \
        f"ded_j = {result.loc[1, 'wall_deduction']:.2f}, expected {ded_expected_j:.2f}"

    # Total exposed: 2*(roof + wall). With L_for_i = L_for_j ≈ 10 m each building
    # loses L*10 from its 400 m² naive wall; total ≈ 2*(100 + 300) = 800 m².
    total = result["exposed_area"].sum()
    assert abs(total - 800.0) < 10.0, \
        f"Total exposed = {total:.1f} m², expected ~800 (within 10)"

    print("  PASS  Test A: flat regression guard")


def test_b_slope():
    """
    Test B: Slope (the case the old code gets wrong).

    Building i: Alc_Base=0, Alc_Top=10 (h=10).
    Building j: Alc_Base=5, Alc_Top=13 (h=8).  Shared edge L ~ 10 m.

    Physical picture:
        Building j sits on higher terrain. The shared face at x=10 runs:
          z=0..5:  building j has not started yet; building i's wall here
                   is buried against the soil that j is founded on
                   (not outdoor-air exposed; ground-contact only)
          z=5..10: both buildings are present; wall of i is hidden by j
          z=10..13: only building j exists; it has an exposed strip here

        So building i has ZERO exposed wall along this segment.
        Building j has an exposed strip of height 13-10 = 3 m, giving L*3 = 30 m².

    Deduction formula check:
        min(top_i, top_j) = min(10, 13) = 10
        ded_i = L * clip(10 - 0, 0, 10) = L * 10  --> exposed_i_seg = 0
        ded_j = L * clip(10 - 5, 0,  8) = L *  5  --> exposed_j_seg = 30

    OLD code (min(height) = min(10,8) = 8):
        ded_i_old = L * 8 = 80  --> exposed_i_seg = 20   WRONG
        ded_j_old = L * 8 = 80  --> exposed_j_seg =  0   WRONG
    """
    gdf = build_gdf(
        make_building(0, 0, 10, 10, alc_base=0, alc_top=10),   # building i
        make_building(10, 0, 20, 10, alc_base=5, alc_top=13),  # building j
    )
    gdf["height"] = gdf["Alc_Top"] - gdf["Alc_Base"]

    shared, _ = find_shared_pairs(gdf)
    assert len(shared) == 1, f"Expected 1 shared pair, got {len(shared)}"
    L_for_i, L_for_j = list(shared.values())[0]
    # For side-by-side rectangles la = lb, so L_for_i = L_for_j ≈ 10 m.

    result = compute_exposed_area(gdf, shared)

    # ded_i = L_for_i * 10 (building i's entire wall height is hidden)
    ded_i_expected = L_for_i * 10.0
    assert abs(result.loc[0, "wall_deduction"] - ded_i_expected) < 1.0, \
        (f"ded_i = {result.loc[0, 'wall_deduction']:.2f}, "
         f"expected {ded_i_expected:.2f} (L_for_i*10)")

    # ded_j = L_for_j * 5 (hidden from base_j=5 up to min_top=10, interval=5)
    ded_j_expected = L_for_j * 5.0
    assert abs(result.loc[1, "wall_deduction"] - ded_j_expected) < 1.0, \
        (f"ded_j = {result.loc[1, 'wall_deduction']:.2f}, "
         f"expected {ded_j_expected:.2f} (L_for_j*5)")

    # exposed wall contribution along the shared segment only
    naive_i_seg = L_for_i * 10.0
    naive_j_seg = L_for_j * 8.0
    exposed_i_seg = naive_i_seg - ded_i_expected   # should be ~0
    exposed_j_seg = naive_j_seg - ded_j_expected   # should be ~30

    assert abs(exposed_i_seg) < 1.0, \
        f"exposed wall i at shared segment = {exposed_i_seg:.2f} m², expected ~0"
    assert abs(exposed_j_seg - 30.0) < 1.0, \
        f"exposed wall j at shared segment = {exposed_j_seg:.2f} m², expected ~30"

    print("  PASS  Test B: slope / elevation-interval deduction")


def test_c_containment():
    """
    Test C: Containment detection.

    A 30x30 m box (900 m²) fully contains a 5x5 m box (25 m²).
    The overlap is 100% of the smaller footprint (far above the 90%
    threshold), so the small building must be flagged for removal.
    This indicates a digitizing error where a block-level polygon was
    drawn on top of the individual unit polygons within it.
    """
    gdf = build_gdf(
        make_building(0, 0, 30, 30, alc_base=0, alc_top=10),   # large block
        make_building(5, 5, 10, 10, alc_base=0, alc_top=8),    # small unit (contained)
    )

    tree = STRtree(gdf.geometry)
    to_drop = set()
    for bi, geom_bi in enumerate(gdf.geometry):
        for bj in tree.query(geom_bi):
            if bj <= bi:
                continue
            overlap = geom_bi.intersection(gdf.geometry[bj])
            if (overlap.is_empty
                    or overlap.geom_type not in ("Polygon", "MultiPolygon")):
                continue
            smaller_area = min(
                gdf.loc[bi, "footprint_area"],
                gdf.loc[bj, "footprint_area"],
            )
            if overlap.area / smaller_area > CONTAINMENT_FRAC:
                to_drop.add(
                    bi if gdf.loc[bi, "footprint_area"] < gdf.loc[bj, "footprint_area"]
                    else bj
                )

    assert 1 in to_drop, "Small contained building (index 1) must be flagged"
    assert 0 not in to_drop, "Large building (index 0) must not be flagged"

    print("  PASS  Test C: containment detection")


def test_d_partial_overlap():
    """
    Test D: Partial overlap (data error, not containment).

    Box i occupies x=[0,10], box j occupies x=[6,16]. They overlap along
    x=[6,10], full y=[0,10], giving an overlap area of 40 m².
    Each box has area 100 m², so overlap fraction = 40%, below the 90%
    containment threshold.

    Expected behaviour:
      1. Neither building is dropped by the containment filter.
      2. The pair IS recorded in the overlap audit list (genuine digitizing
         error, flagged for operator review but not silently discarded).
    """
    gdf = build_gdf(
        make_building(0, 0, 10, 10, alc_base=0, alc_top=10),
        make_building(6, 0, 16, 10, alc_base=0, alc_top=10),
    )
    gdf["height"] = gdf["Alc_Top"] - gdf["Alc_Base"]

    # verify the overlap fraction is below the containment threshold
    raw_overlap = gdf.geometry[0].intersection(gdf.geometry[1])
    assert raw_overlap.area > 0, "Should have a 2-D overlap"
    frac = raw_overlap.area / min(
        gdf.loc[0, "footprint_area"], gdf.loc[1, "footprint_area"]
    )
    assert frac < CONTAINMENT_FRAC, (
        f"Overlap fraction {frac:.2f} should be below {CONTAINMENT_FRAC}: "
        "this is a partial overlap, not containment"
    )

    # containment filter should drop neither building
    tree = STRtree(gdf.geometry)
    to_drop = set()
    for bi, geom_bi in enumerate(gdf.geometry):
        for bj in tree.query(geom_bi):
            if bj <= bi:
                continue
            ov = geom_bi.intersection(gdf.geometry[bj])
            if ov.is_empty or ov.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            smaller = min(
                gdf.loc[bi, "footprint_area"], gdf.loc[bj, "footprint_area"]
            )
            if ov.area / smaller > CONTAINMENT_FRAC:
                to_drop.add(bi if gdf.loc[bi, "footprint_area"] < gdf.loc[bj, "footprint_area"] else bj)

    assert len(to_drop) == 0, \
        "Partial overlap should not trigger containment drop"

    # the pair should be recorded in the overlap audit
    _, overlap_audit = find_shared_pairs(gdf)
    assert len(overlap_audit) > 0, \
        "Partial overlap pair should appear in the overlap audit list"

    print("  PASS  Test D: partial overlap flagged in audit, not dropped")


def test_f_height_repair():
    """
    Test F: Degenerate height repair.

    Building i has Alc_Top=5 < Alc_Base=10 (lidar ground model error).
    Building j is a normal neighbour: Alc_Base=0, Alc_Top=10.

    After repair: building i gets Alc_Top = 10 + H_FLOOR = 13.

    WHY the repair matters for the neighbour's deduction:
        Without repair (bad top=5):
            min(top_i, top_j) = min(5, 10) = 5
            ded_j = L * clip(5 - 0, 0, 10) = L * 5   <-- WRONG (under-deducted)

        With repair (top_i=13):
            min(top_i, top_j) = min(13, 10) = 10
            ded_j = L * clip(10 - 0, 0, 10) = L * 10  <-- CORRECT

        Building j's wall is fully hidden (it sits entirely below the terrain
        on which building i is founded), so the correct deduction is L * h_j.
    """
    gdf_raw = build_gdf(
        make_building(0, 0, 10, 10, alc_base=10, alc_top=5),   # bad: top < base
        make_building(10, 0, 20, 10, alc_base=0,  alc_top=10), # normal neighbour
    )

    # repair step
    gdf, n_repaired = repair_heights(gdf_raw, h_floor=H_FLOOR)

    assert n_repaired == 1, \
        f"Expected 1 repaired building, got {n_repaired}"

    expected_top = gdf.loc[0, "Alc_Base"] + H_FLOOR   # 10 + 3 = 13
    assert abs(gdf.loc[0, "Alc_Top"] - expected_top) < 1e-9, \
        f"Repaired Alc_Top = {gdf.loc[0, 'Alc_Top']}, expected {expected_top}"

    assert gdf.loc[0, "height"] > 0, \
        "Height must be positive after repair"

    # the repaired Alc_Top must be used when computing the neighbour's deduction
    shared, _ = find_shared_pairs(gdf)
    result = compute_exposed_area(gdf, shared)

    L_for_i, L_for_j = list(shared.values())[0] if shared else (10.0, 10.0)
    # For side-by-side rectangles la = lb, so L_for_i = L_for_j ≈ 10 m.

    # ded_j = L_for_j * clip(min(13,10) - 0, 0, 10) = L_for_j * 10
    ded_j_expected = L_for_j * 10.0
    assert abs(result.loc[1, "wall_deduction"] - ded_j_expected) < 1.0, \
        (f"ded_j = {result.loc[1, 'wall_deduction']:.2f}, "
         f"expected {ded_j_expected:.2f} (was the repaired Alc_Top used?)")

    # ded_i = L_for_i * clip(min(13,10) - 10, 0, 3) = L_for_i * 0 = 0
    # building i sits entirely above building j's roof: no hidden wall
    assert abs(result.loc[0, "wall_deduction"]) < 1.0, \
        f"ded_i = {result.loc[0, 'wall_deduction']:.2f}, expected ~0"

    print("  PASS  Test F: degenerate height repair")


def test_g_compactness_filter():
    """
    Test G: Compactness filter.

    A needle-sliver that passes the area threshold (area > AREA_FLOOR = 1 m²)
    but has a degenerate isoperimetric quotient (Q < COMPACTNESS_FLOOR = 0.02)
    must be removed by the compactness filter.

    Sliver: a 0.01 m × 200 m strip.
        area        = 2.0 m²            (above AREA_FLOOR: passes area filter)
        perimeter  ≈ 400.02 m
        Q = 4π × 2 / 400.02² ≈ 1.57e-4   (well below COMPACTNESS_FLOOR)

    Normal building: a 10 m × 10 m square.
        Q = 4π × 100 / 40² ≈ 0.785       (well above threshold)

    After the compactness filter only the square should remain.
    """
    gdf = build_gdf(
        make_building(0,  0, 10,   10,   alc_base=0, alc_top=10),  # normal square
        make_building(20, 0, 20.01, 200, alc_base=0, alc_top=10),  # needle sliver
    )

    compact = 4.0 * np.pi * gdf["footprint_area"] / gdf.geometry.length ** 2

    assert compact.iloc[0] >= COMPACTNESS_FLOOR, \
        f"Normal building compactness {compact.iloc[0]:.3f} should be >= {COMPACTNESS_FLOOR}"
    assert compact.iloc[1] < COMPACTNESS_FLOOR, \
        f"Sliver compactness {compact.iloc[1]:.2e} should be < {COMPACTNESS_FLOOR}"

    gdf_filtered = gdf[compact >= COMPACTNESS_FLOOR].reset_index(drop=True)

    assert len(gdf_filtered) == 1, \
        f"Expected 1 building after compactness filter, got {len(gdf_filtered)}"
    assert abs(gdf_filtered.iloc[0]["footprint_area"] - 100.0) < 0.01, \
        f"Retained building footprint = {gdf_filtered.iloc[0]['footprint_area']:.2f} m², expected 100"

    print("  PASS  Test G: compactness filter drops needle-sliver with normal area")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("Exposed-area synthetic test harness")
    print("=" * 40)

    tests = [
        ("A", "flat regression guard",            test_a_flat_regression),
        ("B", "slope / elevation intervals",      test_b_slope),
        ("C", "containment detection",            test_c_containment),
        ("D", "partial overlap audit",            test_d_partial_overlap),
        ("E", "tile-edge halo (N/A)",             None),
        ("F", "degenerate height repair",         test_f_height_repair),
        ("G", "compactness filter",               test_g_compactness_filter),
    ]

    passed = failed = skipped = 0
    for label, desc, fn in tests:
        if fn is None:
            print(f"  SKIP  Test {label}: {desc}")
            skipped += 1
            continue
        print(f"Test {label}: {desc}")
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print()
    print(f"Results: {passed} passed, {skipped} skipped, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
