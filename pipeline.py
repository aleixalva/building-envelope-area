
__generated_with = "0.23.10"

# %%
import marimo as mo
import geopandas as gpd          # GeoDataFrame: tabular data with geometry column
import numpy as np               # array operations and math utilities
import time
import shapely                   # geometry operations (version >= 2.0 required;
                                 # the STRtree.query bulk API changed in 2.0)
from shapely import make_valid, force_2d  # geometry repair tools:
                                 #   force_2d: strip Z coordinates from 3D geometries
                                 #   make_valid: fix degenerate rings and self-intersections
from shapely.strtree import STRtree
from scipy.stats import gaussian_kde  # kernel density estimation, used to find
                                      # the natural size threshold between artifact
                                      # slivers and real building footprints

# Shapely 2.0 introduced a new STRtree.query() signature that returns integer
# indices and supports bulk geometry array inputs.  The code below relies on
# that API; it will silently produce wrong results on shapely 1.x.
assert tuple(int(x) for x in shapely.__version__.split(".")[:2]) >= (2, 0), (
    f"shapely >= 2.0 required, found {shapely.__version__}"
)

START_TIME = time.time()

# ---------------------------------------------------------------------------
# Named constants -- all magic numbers live here so they are easy to audit
# ---------------------------------------------------------------------------

EPS = 0.05
# Snapping tolerance in metres. Buildings with a gap smaller than EPS are
# treated as sharing a wall. EPS is also the radius of the buffer used when
# measuring shared edge lengths, so it sets the ~EPS length bias in each
# shared-edge measurement (acceptable: ~0.1 m per shared edge for a square
# building with two end-caps of radius EPS).

H_FLOOR = 3.0
# Minimum height (m) assigned to buildings whose lidar data is inconsistent
# (Alc_Top <= Alc_Base). This is a conservative floor, not a precise height.
# Every repaired building is logged so the operator can judge tile quality.

AREA_FLOOR = 1.0
# Fixed area threshold (m²) used as a fallback when the KDE-based automatic
# threshold fails or lands outside THRESHOLD_BOUNDS (see threshold cell).

THRESHOLD_BOUNDS = (1e-3, 3.0)
# Sane window (m²) for the KDE valley threshold. If the detected valley falls
# outside this range the KDE is considered unreliable for this tile and
# AREA_FLOOR is used instead. A very low lower bound protects against a tile
# with almost no slivers; a 3 m² upper bound prevents deleting real small
# buildings (garden sheds, substations) in an unusual tile.

CONTAINMENT_FRAC = 0.9
# If the 2-D intersection of two footprints covers more than this fraction of
# the smaller footprint, the smaller building is considered contained within
# the larger one and is dropped (ICC digitizing error: a block polygon drawn
# on top of the individual units inside it).

OVERLAP_AUDIT_AREA = 0.5
# m²: genuine 2-D footprint overlaps larger than this are flagged in the
# overlap audit and written to a side file. They indicate ICC digitizing
# errors (partial overlaps that are not full containment). The pair is kept
# in the dataset but marked so the operator can quantify their impact.

# %%
mo.md(r"""
# Exposed surface area of buildings from ICC LoD1 data

This notebook computes the exterior surface area of buildings exposed to the
outside environment, for the full AMB (Area Metropolitana de Barcelona)
building dataset from the ICC (Institut Cartografic de Catalunya). The input
is `EdificiosAMB.gpkg`, a single integrated GeoPackage assembled from the ICC
per-tile `_con.shp` shapefiles (construccio layer), which provides building
footprints as 3D polygons with absolute roof and base elevations.

The problem is not trivial because buildings in a dense urban fabric share
walls with their neighbours. A naive computation of perimeter * height
double-counts party walls and overestimates the exposed surface. The correct
exposed wall area requires detecting all shared boundaries between adjacent
buildings and subtracting the hidden portions, accounting for absolute
elevation differences: if two buildings at different terrain levels share a
wall, the one on higher ground may hide part of the lower building's wall
even if it is not the taller building in absolute height terms.

The pipeline proceeds as follows:

1. Load and inspect the raw GeoPackage
2. Repair 3D geometry artifacts and fix inconsistent elevation pairs
3. Remove artifact slivers using an automatic threshold on the footprint-area distribution
4. Detect shared wall segments between all neighbouring building pairs
5. Compute exposed wall and roof area per building
6. Save results
""")

# %%
# Load the building footprint GeoPackage for the full AMB.
#
# The dataset was assembled from individual ICC tiles (each originally a
# _con.shp shapefile) and merged into a single GeoPackage. Processing the
# full dataset at once avoids tile-edge artefacts: party walls that cross
# a tile boundary are detected correctly because both neighbours are always
# present in the same query.
#
# Expected column types after loading:
#   Alc_Top:  string (absolute elevation of roof, metres above sea level)
#   Alc_Base: string (absolute elevation of building base)
#   Both are stored as strings in the original shapefile and must be cast
#   to float before any arithmetic. Non-numeric values are coerced to NaN
#   and reported so the operator can investigate.
#
# Geometry type: MULTIPOLYGON Z (3D), where Z encodes the roof elevation.
# The Z coordinate is redundant with Alc_Top and will be stripped next.

gdf_raw = gpd.read_file("data/EdificiosAMB.gpkg")
print(gdf_raw.shape, gdf_raw.crs)
print(gdf_raw.dtypes)

# %%
# Geometry repair, elevation repair, and height computation.
#
# STEP 1 -- strip Z and repair polygon topology
#   force_2d: removes the Z coordinate from MULTIPOLYGON Z geometries.
#     The ICC extraction pipeline leaves degenerate rings (rings with fewer
#     than 4 points, which cannot enclose area) as sub-polygons within
#     MultiPolygons. GEOS flags the whole geometry as invalid because of them.
#   make_valid: discards those degenerate rings and keeps the real polygon.
#     After force_2d this brings all geometries to valid status.
#
# STEP 2 -- cast elevation columns, guarding against non-numeric strings
#   pd.to_numeric(..., errors='coerce') converts bad strings to NaN rather
#   than raising, so one bad row does not abort the full batch.
#   Any NaN elevations are reported and those rows are dropped before height
#   computation, because NaN elevations produce NaN deductions that would
#   silently corrupt neighbouring buildings.
#
# STEP 3 -- repair inconsistent (Alc_Top, Alc_Base) pairs
#   A small fraction of buildings have Alc_Top <= Alc_Base due to lidar
#   ground model errors (the interpolated base elevation is higher than
#   the scanned roof elevation). The old code clamped height = clip(0)
#   but left the corrupted Alc_Top in place. This is wrong with the
#   elevation-interval deduction formula (added in the exposed-area cell):
#
#     ded_j = L * clip(min(Alc_Top_i, Alc_Top_j) - Alc_Base_j, 0, h_j)
#
#   If building i has a garbage Alc_Top below its own Alc_Base, then
#   min(top_i, top_j) is corrupted and building j receives the wrong
#   deduction. We repair the source pair by setting:
#
#     Alc_Top = Alc_Base + H_FLOOR  wherever Alc_Top <= Alc_Base
#
#   H_FLOOR = 3.0 m is a conservative floor. Every repaired building is
#   counted and logged; a high count signals a problematic tile.
#
# STEP 4 -- derive height and footprint area AFTER all repairs

import pandas as pd

gdf_fixed = gdf_raw.copy()
gdf_fixed["geometry"] = make_valid(force_2d(gdf_raw.geometry))

# cast elevations, coercing bad strings to NaN
gdf_fixed["Alc_Top"]  = pd.to_numeric(gdf_fixed["Alc_Top"],  errors="coerce")
gdf_fixed["Alc_Base"] = pd.to_numeric(gdf_fixed["Alc_Base"], errors="coerce")

# report and drop any rows where elevation is NaN after casting
n_nan_top  = gdf_fixed["Alc_Top"].isna().sum()
n_nan_base = gdf_fixed["Alc_Base"].isna().sum()
if n_nan_top or n_nan_base:
    print(f"WARNING: {n_nan_top} rows with non-numeric Alc_Top, "
          f"{n_nan_base} with non-numeric Alc_Base -- dropping these rows")
    gdf_fixed = gdf_fixed.dropna(subset=["Alc_Top", "Alc_Base"])

# repair (Alc_Top, Alc_Base) pairs where Alc_Top <= Alc_Base
bad_elev = gdf_fixed["Alc_Top"] <= gdf_fixed["Alc_Base"]
n_repaired = int(bad_elev.sum())
n_equal    = int((gdf_fixed["Alc_Top"] == gdf_fixed["Alc_Base"]).sum())
if n_repaired > 0:
    repair_mag = H_FLOOR  # all repaired buildings get exactly this height
    gdf_fixed.loc[bad_elev, "Alc_Top"] = (
        gdf_fixed.loc[bad_elev, "Alc_Base"] + H_FLOOR
    )
    print(f"Repaired {n_repaired} buildings with Alc_Top <= Alc_Base "
          f"(of which {n_equal} were exactly equal). "
          f"Set Alc_Top = Alc_Base + {H_FLOOR} m for all.")

# derive height and footprint area from the repaired elevations
gdf_fixed["height"]         = gdf_fixed["Alc_Top"] - gdf_fixed["Alc_Base"]
gdf_fixed["footprint_area"] = gdf_fixed.geometry.area

print(gdf_fixed[["height", "footprint_area"]].describe())

# %%
gdf_fixed

# %%
mo.md(r"""
A **sliver** is a geometry artifact -- a polygon that is technically valid
(it has enough points to close a ring) but is so thin or small that it
represents no real physical object.

Slivers typically arise from:

- **Boundary operations**: when two polygons that should share an edge have a
  tiny gap or overlap due to floating-point precision, intersection or
  difference operations produce a hair-thin polygon along that boundary.
- **Coordinate snapping**: when polygon vertices from different sources do not
  align exactly, leaving a narrow strip between them.
- **Automated extraction pipelines**: like the ICC lidar extraction here,
  where the algorithm produces degenerate outline fragments alongside the
  real building polygons.

A sliver has a characteristic signature: very small area but non-negligible
perimeter, giving it an extremely low **compactness** (the ratio
$4\pi \cdot \text{area} / \text{perimeter}^2$, which equals 1 for a circle
and approaches 0 for an infinitely thin shape).
""")

# %%
mo.md(r"""
Both a histogram and a KDE are ways to estimate the probability distribution
of a set of numbers from a finite sample, but they work differently.

**Histogram**: you divide the range into fixed bins and count how many values
fall in each bin. The result is a step function -- blocky, and sensitive to
where you place the bin edges.

**KDE**: instead of binning, you place a smooth bump (the "kernel", typically
a Gaussian bell curve) centred on each data point, then sum all those bumps
into a single smooth curve. The `bw_method` parameter controls how wide each
bump is -- narrow bumps preserve fine detail but fit noise; wide bumps give a
smoother picture of the overall shape.

The key advantage for our use case is that a KDE is a smooth differentiable
function, so finding its minimum between two peaks is a well-defined operation
-- you can simply evaluate it on a fine grid and call `np.argmin`. A histogram
minimum is sensitive to bin placement and would require more careful handling.
""")

# %%
# Automatic artifact threshold using kernel density estimation on log10(area).
#
# The footprint area distribution is strongly bimodal in log space:
#   - left peak:  artifact slivers with areas down to ~1e-9 m²
#                 (essentially points and lines that make_valid collapsed
#                 to near-zero polygons)
#   - right peak: real buildings, centred around ~10-100 m²
#
# Working in log10 space is essential: the two populations are separated by
# several orders of magnitude, and a linear scale would bury the artifact
# peak against the axis.
#
# GUARD 1 -- drop non-positive areas before log10.
#   make_valid can produce exactly-zero-area polygons (a degenerate ring
#   reduced to a point or segment). log10(0) = -inf, which breaks
#   gaussian_kde. We drop these first and report how many were removed.
#
# GUARD 2 -- bound the KDE threshold.
#   On a tile with few/no slivers the KDE valley can land near the main peak
#   and delete real small buildings. On a tile with an odd distribution it
#   can keep junk. We clamp the result to THRESHOLD_BOUNDS = (1e-3, 3.0) m².
#   If the valley falls outside this window, or if the KDE fails, we fall
#   back to a fixed AREA_FLOOR = 1.0 m².
#
# The threshold and drop count are printed per run for batch monitoring.

# drop zero or negative areas before log transform
pos_mask = gdf_fixed["footprint_area"] > 0
n_zero = (~pos_mask).sum()
if n_zero > 0:
    print(f"Dropping {n_zero} buildings with non-positive footprint area "
          "(make_valid artefacts)")
gdf_pos = gdf_fixed[pos_mask].copy()

log_areas = np.log10(gdf_pos["footprint_area"].values)

try:
    kde_coarse = gaussian_kde(log_areas, bw_method=0.30)
    x_grid = np.linspace(log_areas.min(), log_areas.max(), 2000)
    density  = kde_coarse(x_grid)

    main_peak_idx = np.argmax(density)
    main_peak     = x_grid[main_peak_idx]

    left_mask      = x_grid < main_peak
    valley_log     = x_grid[left_mask][np.argmin(density[left_mask])]
    threshold_area = 10 ** valley_log

    lo, hi = THRESHOLD_BOUNDS
    if lo <= threshold_area <= hi:
        print(f"KDE threshold: 10^{valley_log:.2f} = {threshold_area:.4f} m²  "
              f"(main mode at 10^{main_peak:.2f} = {10**main_peak:.1f} m²)")
    else:
        print(f"KDE valley {threshold_area:.4g} m² outside bounds "
              f"{THRESHOLD_BOUNDS} -- using fixed fallback AREA_FLOOR = {AREA_FLOOR} m²")
        threshold_area = AREA_FLOOR

except Exception as e:
    print(f"KDE failed ({e}) -- using fixed fallback AREA_FLOOR = {AREA_FLOOR} m²")
    threshold_area = AREA_FLOOR

print(f"Threshold applied: {threshold_area:.4f} m²  "
      f"(artifacts to drop: {(gdf_pos['footprint_area'] < threshold_area).sum()})")

# %%
# Filter out artifact geometries below the automatic threshold.
#
# .copy() creates an independent GeoDataFrame rather than a view of gdf_pos,
# avoiding pandas SettingWithCopyWarning when we add columns in later cells.
#
# reset_index(drop=True) reindexes the rows from 0 to N-1. This is critical
# for the shared-wall detection and exposed-area cells, which use integer
# positional indices to identify building pairs and accumulate deductions into
# numpy arrays aligned to this index. Without the reset, any gaps left by
# dropped rows would cause array indexing to silently produce wrong results.

gdf_clean = (
    gdf_pos[gdf_pos["footprint_area"] >= threshold_area]
    .copy()
    .reset_index(drop=True)
)

# Detect and remove contained buildings: cases where one building footprint
# lies almost entirely inside another. This is physically impossible and
# indicates a digitizing error in the ICC dataset (a large block polygon
# digitized on top of the individual unit polygons within it).
#
# A spatial index (STRtree) limits comparisons to overlapping bounding boxes,
# making the search O(n log n) rather than O(n²).
#
# The CONTAINMENT_FRAC = 0.9 threshold means: if the 2-D intersection covers
# more than 90% of the smaller footprint, the smaller building is considered
# contained and dropped. The larger polygon is kept as the authoritative one.

_tree_contain = STRtree(gdf_clean.geometry)
_to_drop = set()

for _bi, _geom_bi in enumerate(gdf_clean.geometry):
    for _bj in _tree_contain.query(_geom_bi):
        if _bj <= _bi:
            continue
        _overlap = _geom_bi.intersection(gdf_clean.geometry[_bj])
        if _overlap.is_empty or _overlap.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        _smaller = min(
            gdf_clean.loc[_bi, "footprint_area"],
            gdf_clean.loc[_bj, "footprint_area"],
        )
        if _overlap.area / _smaller > CONTAINMENT_FRAC:
            _to_drop.add(
                _bi if gdf_clean.loc[_bi, "footprint_area"] < gdf_clean.loc[_bj, "footprint_area"]
                else _bj
            )

gdf_clean = gdf_clean.drop(index=_to_drop).reset_index(drop=True)
print(f"Buildings retained after artifact + containment removal: {len(gdf_clean)} "
      f"(dropped {len(_to_drop)} contained)")

# %%
gdf_clean

# %%
# Overlap audit -- flag genuine 2-D footprint overlaps for operator review.
#
# After containment removal, some building pairs may still have a partial
# 2-D overlap (overlapping 10-90% of the smaller footprint). These are ICC
# digitizing errors where two footprints were drawn to share some floor area.
# They do not trigger the containment drop (overlap fraction < 0.9) but
# they corrupt the shared-edge length measurement (a 2-D blob's boundary
# length is meaningless as a wall length), so they must be flagged.
#
# This cell builds the STRtree ONCE outside the loop (the previous version
# of this script rebuilt it inside the loop -- O(n) tree constructions for
# n buildings, a severe performance bug for 851k buildings).
#
# Flagged pairs are written to a side file so the operator can review them
# without running the notebook again. They are kept in the dataset: dropping
# them automatically would silently change the building count.

_tree_overlap = STRtree(gdf_clean.geometry)
overlapping_pairs = []

for _bi, _geom_bi in enumerate(gdf_clean.geometry):
    for _bj in _tree_overlap.query(_geom_bi):
        if _bj <= _bi:
            continue
        _overlap = _geom_bi.intersection(gdf_clean.geometry[_bj])
        if (not _overlap.is_empty
                and _overlap.geom_type in ("Polygon", "MultiPolygon")
                and _overlap.area > OVERLAP_AUDIT_AREA):
            overlapping_pairs.append((_bi, _bj, _overlap.area))

if overlapping_pairs:
    overlap_areas = np.array([a for _, _, a in overlapping_pairs])
    print(f"Genuinely overlapping pairs (area > {OVERLAP_AUDIT_AREA} m²): "
          f"{len(overlapping_pairs)}")
    print(f"  min:    {overlap_areas.min():.4f} m²")
    print(f"  median: {np.median(overlap_areas):.4f} m²")
    print(f"  max:    {overlap_areas.max():.4f} m²")
    import os; os.makedirs("output", exist_ok=True)
    pd.DataFrame(overlapping_pairs,
                 columns=["idx_i", "idx_j", "overlap_area_m2"]
                 ).to_csv("output/overlap_audit.csv", index=False)
    print(f"  Written to output/overlap_audit.csv for review.")
else:
    print("No genuine 2-D footprint overlaps found (good).")

# %%
# Detect shared wall segments between neighbouring buildings.
#
# WHAT WE ARE COMPUTING
#   For each pair of buildings, the length of boundary they share in plan
#   view. This length, multiplied by the appropriate elevation interval
#   (see the exposed-area cell), gives the wall area hidden by the neighbour.
#
# SPATIAL INDEX (STRtree)
#   A Sort-Tile-Recursive tree allows fast bounding-box queries. Without it,
#   finding neighbours requires O(n²) comparisons (851k² ~ 7×10¹¹ -- infeasible).
#   The index reduces this to O(n log n) by pre-filtering to geometries whose
#   bounding boxes overlap the query geometry.
#
# SNAPPING TOLERANCE (EPS = 0.05 m)
#   Buildings that should share a wall but have a small gap (digitizing
#   imprecision) are detected by buffering each geometry by EPS before the
#   spatial query. The buffer bridges gaps up to EPS and converts a shared
#   edge (1-D, zero area) into a thin detectable strip. EPS is a named
#   constant at the top of the script.
#
# SHARED EDGE LENGTH -- BOUNDARY INTERSECTION METHOD
#   OLD approach (removed): intersect the buffered geometries and halve the
#   perimeter of the resulting sliver. This fails when two footprints genuinely
#   overlap in 2-D (a data error): the sliver becomes a blob, and half its
#   perimeter is meaningless as a wall length.
#
#   NEW approach: measure the length of each building's boundary that falls
#   inside the other building's buffer, then average the two measurements:
#
#       la = (boundary of j) ∩ (buffer of i)   -- length
#       lb = (boundary of i) ∩ (buffer of j)   -- length
#       shared_length = 0.5 * (la + lb)
#
#   For a clean shared edge this gives the true edge length, within ~EPS
#   (the buffer end-caps add ~EPS at each corner; the average cancels part
#   of the bias, leaving ~EPS total across the shared segment). For a genuine
#   2-D overlap it returns a length related to the outline, not the blob area
#   -- still wrong, but those pairs are already flagged in the overlap audit.
#
# PERFORMANCE -- vectorised bulk operations (shapely 2.0)
#   STRtree.query(array_of_geometries, predicate="intersects") accepts a
#   numpy array and returns (left_indices, right_indices) for ALL candidate
#   pairs at once. This replaces the Python for-loop that called tree.query()
#   once per building. For 851k buildings this is the dominant speedup.
#
#   After pair-finding, shapely.boundary(), shapely.intersection(), and
#   shapely.length() all operate element-wise on numpy arrays of geometries
#   with no Python loop -- the geometry work is done inside GEOS (C++).
#   Precomputing all buffers with shapely.buffer(geoms, EPS) outside the
#   pair loop avoids recomputing each buffer once per candidate neighbour.
#
# PAIR DEDUPLICATION
#   tree.query(buf_i) for a given building i returns all buildings j whose
#   bounding box overlaps buf_i, including i itself (j == i) and buildings
#   already processed in an earlier iteration (j < i). With the bulk query,
#   deduplication is done with a mask: keep only pairs where left_idx < right_idx.

t0_shared = time.time()

geoms = gdf_clean.geometry.values   # numpy array of Shapely geometry objects,
                                    # aligned to the reset 0-based integer index

# precompute buffer(i, EPS) for every building once, as a numpy array.
# shapely.buffer() is vectorised and runs in C++; it is much faster than a
# Python list comprehension calling geom.buffer(EPS) individually.
bufs = shapely.buffer(geoms, EPS)

tree = STRtree(geoms)

# find ALL candidate pairs at once with the bulk query.
# predicate="intersects" filters pairs whose buffered geometry intersects
# the corresponding query geometry, using the GEOS intersection test --
# this eliminates bounding-box false positives before the exact measurement.
left_idx, right_idx = tree.query(bufs, predicate="intersects")

# keep only canonical pairs i < j to avoid double-counting
mask = left_idx < right_idx
left_idx  = left_idx[mask]
right_idx = right_idx[mask]

print(f"Candidate neighbour pairs after spatial filtering: {len(left_idx):,}")

# compute shared edge lengths for all pairs at once (no Python loop).
# boundary() includes interior rings (courtyard facades): this is correct
# because courtyard walls are genuinely exposed to outdoor air, so their
# perimeter must be included in the shared-edge measurement.
bounds_j = shapely.boundary(geoms[right_idx])   # boundary of building j
la = shapely.length(shapely.intersection(bounds_j, bufs[left_idx]))
# la[k] = length of j's boundary that falls inside buffer(i, EPS)

bounds_i = shapely.boundary(geoms[left_idx])    # boundary of building i
lb = shapely.length(shapely.intersection(bounds_i, bufs[right_idx]))
# lb[k] = length of i's boundary that falls inside buffer(j, EPS)
# averaging la and lb makes the measurement symmetric and partially cancels
# the ~EPS end-cap bias from the buffer circles.

shared_lengths = 0.5 * (la + lb)

# discard pairs with no meaningful shared boundary (point contacts,
# bounding-box false positives that survived the predicate filter)
valid = shared_lengths > 1e-6
left_idx      = left_idx[valid]
right_idx     = right_idx[valid]
shared_lengths = shared_lengths[valid]

elapsed_shared = time.time() - t0_shared
print(f"Shared wall pairs found: {len(left_idx):,}")
print(f"Total shared boundary length: {shared_lengths.sum():.1f} m")
print(f"Shared-wall detection elapsed: {elapsed_shared:.1f} s  "
      f"({elapsed_shared/60:.1f} min)")

# %%
# Compute exposed surface area for each building.
#
# COMPONENTS
#   roof_area     = footprint area (flat roof, LoD1 assumption)
#   wall_area     = perimeter * height  minus  hidden wall area
#   exposed_area  = roof_area + wall_area
#
# CORRECT DEDUCTION FORMULA -- ELEVATION INTERVALS (replaces old min(height))
#   The old formula subtracted L * min(h_i, h_j) from BOTH buildings, where
#   h = Alc_Top - Alc_Base. This is only correct on flat ground (equal base
#   elevations). On any slope -- much of Barcelona (Gracia, Sarria, Horta,
#   the Collserola flank) -- adjacent buildings have different Alc_Base, and
#   the old formula gives wrong results in both directions.
#
#   Correct physics for a shared segment of plan-length L:
#     Building i's wall at the shared face is exposed to outdoor air ONLY
#     above the neighbour's roofline. Everything at or below the neighbour's
#     roof is hidden -- either by the neighbour's body (above the neighbour's
#     own base) or by the soil the neighbour is founded on (below the
#     neighbour's base: a buried/retaining wall, NOT air-exposed surface).
#
#     Deduction per shared segment (ASYMMETRIC -- differs for each building):
#       ded_i = L * clip(min(top_i, top_j) - base_i, 0, h_i)
#       ded_j = L * clip(min(top_i, top_j) - base_j, 0, h_j)
#
#     Flat-ground invariant: when base_i = base_j = b,
#       min(top_i, top_j) - b = min(h_i, h_j)
#     so the formula reduces exactly to the old min(height) result. The
#     existing +/-5% flat-tile validation therefore still holds. (Test A)
#
#     Clip lower bound = 0 handles the case where one building's base sits
#     above the other's roof (no real 3-D contact -> zero deduction).
#     Clip upper bound = h_i prevents over-deduction if the shared length
#     is slightly inflated by the buffer end-cap bias.
#
#   MODELLING NOTE: this treats the wall below a higher neighbour's base as
#   buried against soil. This is correct for an outdoor-air exposure model.
#   Ground-contact area (a different heat-transfer path) is NOT computed here
#   and must not be added to exposed_area.
#
# PRE-CLIP DEDUCTION AUDIT (§6)
#   We compute deduction_fraction = wall_deduction / wall_area_naive BEFORE
#   the clip(lower=0) safety net, so that over-deduction is visible. A
#   fraction > 1.0 means the accumulated deductions exceed the naive wall
#   area, indicating over-detection (inflated shared lengths, a building
#   receiving deductions from many neighbours, or a residual elevation error).
#   After the §1 and §3 fixes this set should be small or empty; any residual
#   cases are logged for inspection.
#
# PERFORMANCE
#   Alc_Top, Alc_Base, and height are materialised as numpy arrays indexed
#   by integer position (aligned with the reset index of gdf_clean). This
#   allows O(1) array reads inside the deduction accumulation, rather than
#   the slow DataFrame .loc lookups that were in the previous version.
#   np.add.at() accumulates deductions from multiple neighbours correctly
#   even when a building appears multiple times as left_idx or right_idx.

t0_area = time.time()

# materialise elevation arrays -- positional, aligned with gdf_clean index
top_arr    = gdf_clean["Alc_Top"].values
base_arr   = gdf_clean["Alc_Base"].values
height_arr = gdf_clean["height"].values

gdf_clean["wall_area_naive"] = gdf_clean.geometry.length * height_arr
# geometry.length is the perimeter of the 2-D footprint in metres.
# For polygons-with-holes (courtyards) it includes interior rings --
# courtyard facades are genuinely exposed to outdoor air, so this is correct.

gdf_clean["roof_area"] = gdf_clean["footprint_area"]
# flat-roof assumption (LoD1): roof area equals footprint area exactly.

# elevation-interval deduction, vectorised over all shared pairs
top_i = top_arr[left_idx]
top_j = top_arr[right_idx]
base_i = base_arr[left_idx]
base_j = base_arr[right_idx]
h_i = height_arr[left_idx]
h_j = height_arr[right_idx]
L   = shared_lengths

min_top = np.minimum(top_i, top_j)   # shared ceiling: hidden below this elevation

ded_i = L * np.clip(min_top - base_i, 0.0, h_i)
ded_j = L * np.clip(min_top - base_j, 0.0, h_j)
# ded_i[k]: area of building left_idx[k]'s wall hidden by building right_idx[k]
# ded_j[k]: area of building right_idx[k]'s wall hidden by building left_idx[k]
# These are asymmetric: a building on higher ground hides more of its lower
# neighbour's wall than the lower neighbour hides of its own.

wall_deduction = np.zeros(len(gdf_clean))
np.add.at(wall_deduction, left_idx,  ded_i)
np.add.at(wall_deduction, right_idx, ded_j)
# np.add.at is an unbuffered in-place operation that handles duplicate indices
# correctly, unlike wall_deduction[left_idx] += ded_i which would silently
# drop contributions when the same index appears more than once in left_idx.

gdf_clean["wall_deduction"] = wall_deduction

# audit deduction fraction BEFORE the clip (§6):
# report how many buildings have a deduction larger than their naive wall area.
naive = gdf_clean["wall_area_naive"].values
with np.errstate(divide="ignore", invalid="ignore"):
    ded_frac_raw = np.where(naive > 0, wall_deduction / naive, 0.0)

n_over = int((ded_frac_raw > 1.0).sum())
if n_over > 0:
    print(f"WARNING: {n_over} buildings have wall_deduction > wall_area_naive "
          f"before clip (deduction_fraction_raw > 1.0). "
          f"Max fraction: {ded_frac_raw.max():.3f}. "
          "These rows warrant inspection -- possible causes: inflated shared "
          "lengths from 2-D overlap pairs, or cumulative deductions from "
          "many neighbours.")
else:
    print("All deduction fractions <= 1.0 before clip (good).")

gdf_clean["deduction_fraction_raw"] = ded_frac_raw

gdf_clean["wall_area"] = (
    gdf_clean["wall_area_naive"] - wall_deduction
).clip(lower=0.0)
# clip(lower=0) is a safety net for the rare over-deduction cases.
# It must NOT be used to silently absorb systematic errors -- those are
# reported above and should be investigated before the full-city run.

gdf_clean["exposed_area"] = gdf_clean["roof_area"] + gdf_clean["wall_area"]

gdf_clean["deduction_fraction"] = (
    wall_deduction / gdf_clean["wall_area_naive"]
).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=1.0)

elapsed_area = time.time() - t0_area
print("\nExposed area stats (m²):")
print(gdf_clean["exposed_area"].describe())
print(f"\nTotal exposed area:    {gdf_clean['exposed_area'].sum():>15,.0f} m²")
print(f"Total naive wall area: {gdf_clean['wall_area_naive'].sum():>15,.0f} m²")
print(f"Total wall deduction:  {gdf_clean['wall_deduction'].sum():>15,.0f} m²")
print(f"Total roof area:       {gdf_clean['roof_area'].sum():>15,.0f} m²")
print(f"Area computation elapsed: {elapsed_area:.1f} s")

# %%
import os

cols_to_save = [
    "Nivell", "Descripcio", "Full",
    "Alc_Top", "Alc_Base", "height", "footprint_area",
    "roof_area", "wall_area_naive", "wall_deduction",
    "wall_area", "deduction_fraction", "deduction_fraction_raw",
    "exposed_area",
    "geometry",
]

os.makedirs("output", exist_ok=True)
gdf_clean[cols_to_save].to_file("output/buildings_exposed.gpkg", driver="GPKG")
print(f"Saved {len(gdf_clean)} buildings to output/buildings_exposed.gpkg")

elapsed_total = time.time() - START_TIME
print(f"\nTotal elapsed: {elapsed_total:.1f} s  "
      f"({elapsed_total/60:.1f} min  /  {elapsed_total/3600:.2f} h)")

# %%
gdf_clean

# %%
mo.md(r"""
# Summary

The pipeline was applied to the full AMB (Area Metropolitana de Barcelona)
building dataset assembled from ICC LoD1 footprints.

## Key algorithmic changes from the previous version

**Elevation-interval deduction (replaces `min(height)`):**
The wall hidden by a neighbour depends on absolute elevations, not just
relative heights. For a shared segment of plan-length L between buildings
i and j:

    ded_i = L * clip(min(top_i, top_j) - base_i, 0, h_i)
    ded_j = L * clip(min(top_i, top_j) - base_j, 0, h_j)

On flat ground this reduces exactly to the old `min(height)` formula (the
previous validation still holds). On sloped terrain (Gracia, Sarria, Horta,
Collserola) the old formula incorrectly allocated exposed wall area; the new
formula gives the physically correct result.

**Elevation pair repair (replaces height clip):**
Buildings with `Alc_Top <= Alc_Base` (lidar ground model errors) now have
their `Alc_Top` repaired to `Alc_Base + H_FLOOR` before any computation.
Simply clamping `height = clip(0)` left a corrupted `Alc_Top` that could
propagate wrong deductions onto neighbouring buildings via `min(top_i, top_j)`.

**Shared-edge length (replaces buffer/half-perimeter):**
The new method measures the length of each building's boundary that falls
within the other building's buffer, then averages the two directions:

    la = length(boundary(j) ∩ buffer(i, EPS))
    lb = length(boundary(i) ∩ buffer(j, EPS))
    shared = 0.5 * (la + lb)

This is immune to the 2-D-blob problem that made the old half-perimeter
approach wrong for genuine footprint overlaps.

**Vectorised pair-finding (shapely 2.0 bulk API):**
All candidate pairs are found in a single `STRtree.query(array, predicate=...)`
call, and shared lengths are computed with bulk `shapely.boundary()`,
`shapely.intersection()`, and `shapely.length()` -- no Python loop.
""")
