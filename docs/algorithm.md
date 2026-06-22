# Algorithm and Methodology

This document gives a full technical account of the exposed-surface-area
pipeline: every formula, every design decision, and every known limitation.
It is the place to look when the code comments alone are not enough.

---

## 1. Input data

The pipeline consumes a single GeoPackage of ICC LoD1 building footprints for
the Barcelona Metropolitan Area (AMB). The file contains 851,505 rows, one per
building polygon, in EPSG:25831 (ETRS89 / UTM zone 31N).

Each row carries two elevation attributes stored as character strings:

| Column | Description |
|--------|-------------|
| `Alc_Top` | Absolute elevation of the roof (metres above sea level) |
| `Alc_Base` | Absolute elevation of the ground at the building base |

Both are strings in the raw ICC export and are cast to float early in the
pipeline; non-numeric values are coerced to `NaN` and those rows are dropped
with a warning.

The geometry type is `MULTIPOLYGON Z`, where the Z coordinate encodes the roof
elevation. The Z coordinate is redundant with `Alc_Top` and is stripped
immediately (see Step 1 below).

---

## 2. What the pipeline computes

For each building, the exposed surface area is the total area of the building
envelope in contact with outdoor air:

```
exposed_area = roof_area + wall_area
```

where:

```
wall_area = wall_area_naive - wall_deduction
         = perimeter * height  -  (sum of hidden wall segments)
```

A wall segment is hidden when it is shared with a neighbour: it faces the
interior of the block, not the street. Correctly identifying and subtracting
these hidden segments is the core difficulty, complicated by the fact that
adjacent buildings often sit on different terrain levels (sloped terrain).

---

## 3. Step 1: Geometry and elevation repair

### 3a. Topology repair

The ICC extraction pipeline occasionally produces degenerate sub-polygons
within `MULTIPOLYGON Z` geometries: rings with fewer than four vertices that
cannot enclose any area. GEOS flags the whole geometry as topologically invalid
because of them.

Two Shapely operations fix this:

- `force_2d`: removes the Z coordinate, converting `MULTIPOLYGON Z` to
  `MULTIPOLYGON`. At this stage, degenerate 3-D rings become degenerate 2-D
  rings (still invalid).
- `make_valid`: discards degenerate rings and keeps the real footprint polygon.

After both operations, all geometries are valid 2-D polygons.

### 3b. Elevation pair repair

A small fraction of buildings have `Alc_Top <= Alc_Base` due to lidar ground
model errors: the interpolated base elevation is higher than the scanned roof
elevation. This can happen at the edge of a lidar strip, on steep terrain, or
due to interpolation artefacts.

The naive fix (clip the derived height to zero) is wrong for the following
reason. The deduction formula (see Step 5 below) requires the raw `Alc_Top`
value of every neighbour:

```
ded_j = L * clip( min(Alc_Top_i, Alc_Top_j) - Alc_Base_j,  0,  h_j )
```

If building i has a garbage `Alc_Top` below its own `Alc_Base`, then
`min(Alc_Top_i, Alc_Top_j)` is corrupted and building j receives a wrong
deduction, possibly a large underestimate. Clamping the derived height to zero
but leaving the corrupted `Alc_Top` in place does not fix this.

The correct repair sets:

```
Alc_Top = Alc_Base + H_FLOOR   wherever Alc_Top <= Alc_Base
```

`H_FLOOR = 3.0 m` is a conservative floor, not a precise height. Every
repaired building is counted in the report so the operator can assess tile
quality; a high count signals a problematic dataset or lidar strip.

---

## 4. Step 2: Artefact sliver removal

### 4a. KDE threshold

A sliver is a geometrically valid polygon (it closes a ring) that represents
no real physical object. Slivers arise from boundary operations, coordinate
snapping, and automated extraction pipelines. They are characterised by very
small area and non-negligible perimeter.

The area distribution of building footprints is strongly bimodal in log10
space:

- Left peak: artefact slivers with areas down to ~1e-9 m² (machine precision
  degenerate polygons that `make_valid` reduces to finite but tiny area).
- Right peak: real buildings, centred around 10 to 100 m² in the AMB dataset.

Working in log10 space is essential: the two populations are separated by
several orders of magnitude and a linear scale buries the sliver peak entirely.

A KDE (kernel density estimate) is fitted to the log10-area distribution with
`bw_method = 0.30`. This deliberately coarse bandwidth prevents spurious local
minima within the sliver cluster that would arise with a fine bandwidth. The
valley (minimum density between the two peaks) is found by evaluating the KDE
on a grid of 2,000 points and calling `numpy.argmin` on the left side of the
main peak. The corresponding area is the threshold below which buildings are
considered artefacts.

The threshold is bounded to `THRESHOLD_BOUNDS = (1e-3, 3.0)` m². If the
detected valley falls outside this window (for instance because the sliver
population is so small or so extreme that the KDE does not form a clear
secondary peak), the threshold falls back to `AREA_FLOOR = 1.0 m²`.

On the full 851k-building dataset, the KDE typically finds a clear valley
around 0.5 to 2 m². On small tiles (e.g. the 500 x 500 m test tile), the
sliver population clusters near machine precision (~1e-10 m²) without forming
a detectable secondary peak, so the fallback is triggered. The fallback value
of 1.0 m² is a safe choice: the 25th percentile of real building areas in the
test tile is ~7 m², so the fallback does not remove any real buildings.

### 4b. Containment removal

After sliver removal, some building pairs remain where one footprint lies almost
entirely inside the other. This is a digitising error in the ICC data: a
block-level polygon drawn on top of the individual unit polygons within it.

A contained building is identified when the 2-D intersection area of two
polygons exceeds `CONTAINMENT_FRAC = 0.9` (90 %) of the smaller footprint
area. The smaller building is dropped; the larger is kept as the authoritative
polygon. The STRtree limits the search to overlapping bounding boxes, keeping
the detection at O(n log n).

---

## 5. Step 3: Overlap audit

After containment removal, some pairs may still have a partial 2-D overlap
(10 to 90 % of the smaller footprint). These are ICC digitising errors where
two footprints genuinely share some floor area. They do not qualify for the
containment drop but they corrupt the shared-edge length measurement (see
Step 4 below), because the boundary-intersection formula returns a length
related to the boundary of the overlapping blob, not the true shared wall.

Any pair whose 2-D intersection area exceeds `OVERLAP_AUDIT_AREA = 0.5 m²` is
written to `output/overlap_audit.csv` for manual review. The pair is kept in
the dataset: automatic removal would change the building count without operator
awareness. The pipeline warns about over-deducted buildings (deduction fraction
above 1.0) in the report; genuine overlap pairs are a contributing cause.

---

## 6. Step 4: Shared-wall detection

### 6a. Spatial index

Finding all neighbouring building pairs by brute force requires O(n²)
comparisons (851,505² ~ 7 x 10^11 for the full dataset, infeasible). A
Sort-Tile-Recursive tree (`STRtree`) reduces this to O(n log n) by
pre-filtering to pairs whose bounding boxes overlap.

The tree is queried in a single bulk call:

```python
left_idx, right_idx = tree.query(bufs, predicate="intersects")
```

where `bufs` is an array of `buffer(geom_i, EPS)` geometries, one per
building. This replaces the original per-building Python loop over the tree,
which was the dominant performance bottleneck in the old pipeline.

### 6b. Snapping tolerance (EPS)

Buildings that should share a wall but have a small gap due to digitising
imprecision are detected by buffering each geometry by `EPS = 0.05 m` before
the spatial query. The buffer bridges gaps up to EPS wide, converting a shared
edge (geometrically 1-D, zero area) into a thin strip detectable by an
intersection test.

EPS = 0.05 m (5 cm) is chosen to bridge typical digitising imprecision in the
ICC dataset without picking up buildings separated by a real alley.

### 6c. Shared-edge length measurement

For each candidate pair (i, j), the shared plan-length is measured as the
average of two boundary-intersection lengths:

```
la = length( boundary(j) ∩ buffer(i, EPS) )
lb = length( boundary(i) ∩ buffer(j, EPS) )
shared_length = max( 0,  0.5 * (la + lb) - 2 * EPS )
```

`la` is the length of building j's boundary that falls inside the EPS-expanded
polygon of building i. For a clean shared edge, this captures the shared
segment itself. Averaging with `lb` (the symmetric measurement from building
i's perspective) was intended to cancel end-cap bias, but does not fully do so
(see Section 6d below).

This formula replaces the old approach of intersecting the two buffered
polygons and halving the perimeter of the resulting sliver. That approach
breaks whenever two footprints genuinely overlap in 2-D (a data error): the
sliver becomes a blob, and half its perimeter is meaningless as a wall length.
The boundary-intersection formula is robust to 2-D overlaps: it returns a
length related to the boundary of the overlapping region, which is still wrong,
but the pair is already flagged in the overlap audit.

The `boundary()` operation includes interior rings (courtyards). Courtyard
facades are genuinely exposed to outdoor air, so including them is physically
correct.

### 6d. End-cap correction (the -2*EPS term)

The buffer end-caps introduce a systematic positive bias in both `la` and `lb`.
At each endpoint of a shared segment, the buffer circle of building i overhangs
the corner and captures approximately EPS of the adjacent (non-shared) wall of
building j. Both `la` and `lb` carry this same bias (the corner is shared by
both buildings), so averaging does not cancel it. The total bias per pair is
approximately 2*EPS (one EPS per endpoint, two endpoints per segment).

Subtracting 2*EPS restores the true edge length:

```
shared_length = max( 0,  0.5 * (la + lb) - 2 * EPS )
```

This correction was confirmed empirically on the test suite: a 10 m test edge
measured at ~10.10 m without the correction and ~10.00 m with it.

**Trade-off.** The correction assumes both endpoints of the shared segment are
at building corners, where each endpoint contributes ~EPS of end-cap arc. At a
T-junction (where one building's wall ends mid-segment of the other), one
endpoint contributes ~EPS but the other contributes ~0, so the correction
over-subtracts by ~EPS for that pair. T-junctions are common in dense
Barcelona block fabric. However, the systematic positive bias (which, without
the correction, pushes 12.6 % of buildings into the wall-area clip-to-zero
trap) is a larger and more predictable error than the occasional T-junction
under-correction, so the correction is applied.

Very short walls (shorter than 2*EPS = 0.10 m) are zeroed by the `max(0,...)`
clamp. A 10 cm shared wall contributes negligible area and is considered a
point contact.

---

## 7. Step 5: Exposed surface area

### 7a. Naive wall area

```
wall_area_naive = perimeter * height
```

`perimeter` is the 2-D plan perimeter of the footprint (`geometry.length` in
Shapely), including interior rings for buildings with courtyards. `height` is
`Alc_Top - Alc_Base` after repair.

### 7b. The elevation-interval deduction formula

For a shared segment of plan-length L between buildings i and j, the
deductions are asymmetric:

```
ded_i = L * clip( min(Alc_Top_i, Alc_Top_j) - Alc_Base_i,  0,  h_i )
ded_j = L * clip( min(Alc_Top_i, Alc_Top_j) - Alc_Base_j,  0,  h_j )
```

The physical reasoning:

- `min(Alc_Top_i, Alc_Top_j)` is the elevation below which both buildings are
  present (at or above their respective bases) or the ground on which they are
  founded is present. Below this elevation, building i's wall is not exposed
  to outdoor air: it faces either the body of building j (between `Alc_Base_j`
  and `min_top`) or the soil on which j is founded (below `Alc_Base_j`). Both
  are non-air surfaces for building i's heat-transfer budget.
- Subtracting `Alc_Base_i` gives the hidden wall height for building i.
- `clip(0, h_i)` handles two degenerate cases: (a) the lower bound guards
  against one building's base sitting above the other's roof (no real 3-D
  contact, zero deduction); (b) the upper bound prevents over-deduction from
  slightly inflated shared lengths (buffer end-cap residual).

This is asymmetric: the two deductions are generally different, because the
buildings sit at different elevations.

**Modelling note.** Walls below a higher neighbour's base elevation are treated
as buried against soil. This is correct for an outdoor-air exposure model.
Ground-contact area (a distinct heat-transfer path, relevant for ground-source
heat exchange) is not computed here.

### 7c. Flat-ground invariant

On flat ground (`Alc_Base_i = Alc_Base_j = b`):

```
min(Alc_Top_i, Alc_Top_j) - b = min(h_i, h_j)
```

The formula reduces to `ded_i = ded_j = L * min(h_i, h_j)`, which is the
classical formula used in the original pipeline. Flat-tile results from the
old pipeline should therefore agree with the new pipeline within ~5 %.
Differences on sloped tiles (much of Barcelona: Gracia, Sarria, Horta, the
Collserola flank) are expected, because the old formula was wrong there.

### 7d. Accumulation with numpy.add.at

Deductions from multiple neighbours accumulate per building:

```python
np.add.at(wall_deduction, left_idx,  ded_i)
np.add.at(wall_deduction, right_idx, ded_j)
```

`numpy.add.at` is used rather than `wall_deduction[left_idx] += ded_i`
because the latter silently drops duplicate indices (if building i appears
more than once in `left_idx`, only the last contribution is kept). `add.at`
is an unbuffered in-place operation that handles repeated indices correctly.

### 7e. Pre-clip audit and over-deduction

The deduction fraction is computed before clipping:

```
deduction_fraction_raw = wall_deduction / wall_area_naive
```

A value above 1.0 means the accumulated deductions exceed the naive wall area.
This can happen because:

1. Buffer end-cap residual (the -2*EPS correction reduces but does not
   eliminate this for T-junctions and buildings with many neighbours).
2. The 58 genuine 2-D overlap pairs flagged in the overlap audit, whose
   shared lengths are inflated by the boundary-of-blob effect.
3. Cumulative accumulation in buildings that are surrounded on all sides by
   taller neighbours: each neighbour contributes a valid partial deduction, and
   the sum can legitimately exceed the naive wall area for very small buildings
   in dense blocks.

The clip to zero (`wall_area = max(0, wall_area_naive - wall_deduction)`) is
the safety net. For a building with no truly exposed wall (fully embedded in a
block), the correct physical answer is wall_area = 0, and the clip achieves
this. For buildings where the over-deduction is a measurement artefact rather
than a physical truth, the clip is a slight over-correction, but the error is
small: after the -2*EPS correction, the maximum over-deduction in the test
tile is 4.9 % of buildings, with a maximum raw fraction of 1.049.

---

## 8. Named constants

All parameters are defined as named constants at the top of `envelope_area.py` so
they can be audited and adjusted in one place.

| Constant | Value | Meaning |
|----------|-------|---------|
| `EPS` | 0.05 m | Snapping tolerance; also the radius of the buffer used in shared-edge detection |
| `H_FLOOR` | 3.0 m | Minimum height assigned to buildings with `Alc_Top <= Alc_Base` |
| `AREA_FLOOR` | 1.0 m² | Fixed area threshold used as KDE fallback |
| `THRESHOLD_BOUNDS` | (1e-3, 3.0) m² | Sane range for the KDE valley threshold |
| `CONTAINMENT_FRAC` | 0.9 | Fraction of smaller footprint that triggers containment removal |
| `OVERLAP_AUDIT_AREA` | 0.5 m² | Minimum 2-D overlap area to flag in the audit CSV |

---

## 9. Performance

The dominant cost in the original pipeline was the per-building Python loop
over the STRtree and the per-pair scalar `.loc` lookups. The corrected pipeline
eliminates both:

- **Pair finding:** a single `STRtree.query(bufs, predicate="intersects")` call
  returns all candidate pairs as numpy index arrays, with no Python loop.
- **Geometry operations:** `shapely.boundary()`, `shapely.intersection()`, and
  `shapely.length()` operate element-wise on numpy geometry arrays, running
  inside GEOS (C++) without Python overhead.
- **Deduction accumulation:** `numpy.add.at()` accumulates all pair deductions
  in a single vectorised pass.
- **Index alignment:** after `reset_index(drop=True)`, the positional indices
  of the GeoDataFrame align with the numpy arrays, enabling O(1) element
  access without `.loc` lookups.

On a 500 x 500 m tile with 17,067 raw buildings (16,240 after cleaning), the
shared-wall detection completes in ~2 seconds. The original loop-based
implementation on the full 851k-building dataset required ~48 hours; the
vectorised implementation is expected to complete in well under one hour.

---

## 10. Known limitations

**Buffer end-cap residual.** The -2*EPS correction removes the end-cap bias for
corner-to-corner shared walls but slightly over-corrects for T-junctions (see
Section 6d). A more exact correction would require detecting the type of each
endpoint, which is geometrically expensive and not implemented.

**Flat-roof assumption.** The pipeline uses `roof_area = footprint_area` (LoD1
flat-roof model). Pitched roofs, mansard roofs, and setbacks are not modelled.

**No ground-contact area.** Walls below a higher neighbour's base elevation are
treated as buried against soil and excluded from the exposed area. The
ground-contact surface (relevant for ground-source heat exchange) is not
computed.

**Genuine 2-D overlaps.** The 58 overlap pairs in the test tile (and a
proportional number in the full dataset) have slightly inflated shared lengths
that the correction does not fully fix. Their effect on total area is small but
non-zero; they are flagged in `overlap_audit.csv` for manual review.

**KDE fallback on small tiles.** The KDE valley-detection relies on a
bimodal distribution in log10-area space. On small tiles where the sliver
population is sparse or clusters at machine precision without forming a
detectable secondary peak, the fallback to `AREA_FLOOR = 1.0 m²` is
triggered. This is correct behaviour but worth noting when interpreting the
report's "Sliver threshold" line.

**Single integrated dataset assumed.** The pipeline processes the full AMB as a
single GeoPackage. Buildings at the outer boundary of the AMB have no
neighbours beyond the boundary, so their outer walls are correctly reported as
fully exposed. No tile-edge halo correction is applied (and none is needed
given the single integrated input).
