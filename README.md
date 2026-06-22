# Building Envelope Area: Barcelona Metropolitan Area

Computes the exterior surface area of buildings exposed to outdoor air (roof
and walls, minus shared party walls) from the ICC LoD1 building dataset for
the Barcelona Metropolitan Area (AMB). The result is a per-building estimate
of the envelope area directly in contact with the outside environment, which
is a key input for urban energy modelling, heat island analysis, and building
physics studies.

---

## Background

The **ICC (Institut Cartogràfic de Catalunya)** distributes building footprints
as part of its topographic base (BT5M). The LoD1 (Level of Detail 1)
representation models each building as a simple vertical extrusion of its 2-D
footprint up to a flat roof. Two elevation attributes are available per
building:

| Attribute | Meaning |
|-----------|---------|
| `Alc_Top` | Absolute roof elevation, metres above sea level |
| `Alc_Base` | Absolute ground elevation at the building base |

Building height is derived as `Alc_Top - Alc_Base`. Because both values are
absolute elevations, buildings on sloped terrain have different base
elevations even when they share a wall, a subtlety that the pipeline handles
correctly (see [Algorithm](#algorithm) below).

---

## The problem

A naive estimate of exposed wall area (perimeter multiplied by height)
double-counts every party wall, because both buildings on either side of the
shared face would each claim the full wall. In a dense urban fabric such as
Barcelona, where continuous block construction means that most wall surface is
shared, this naive estimate overstates the true exposed area by a factor of 3
to 4.

The correct approach is to:
1. detect all shared wall segments between adjacent buildings, and
2. subtract the hidden portion from each building's wall area budget.

The hidden portion depends on the **absolute elevations** of both buildings,
not just their relative heights. Consider two buildings on a hillside sharing
a 10 m-long face: if building A has its roof at z = 10 m and building B (on
higher ground) has its base at z = 5 m and roof at z = 13 m, then the entire
wall of A is hidden (buried behind B's foundation and body), while B retains
an exposed strip from z = 10 m to z = 13 m. Using `min(height)` (the
approach in the original version of this pipeline) gives the wrong answer in
both directions.

---

## Algorithm

### 1. Geometry repair

The ICC geometries are stored as `MULTIPOLYGON Z` with occasional degenerate
rings (collapsed to points or line segments in 3-D). `force_2d` strips the Z
coordinate, and `make_valid` discards the degenerate sub-polygons. Buildings
whose `Alc_Top <= Alc_Base` (lidar ground model errors) are repaired by
setting `Alc_Top = Alc_Base + H_FLOOR` (3 m default) rather than simply
clamping the derived height, because the raw `Alc_Top` value propagates into
the neighbour's deduction formula.

### 2. Sliver removal

The area distribution in log10 space is bimodal: a left peak of near-zero
artefact slivers and a right peak of real buildings. A KDE fit with a coarse
bandwidth (0.30 in log space) locates the valley automatically. The threshold
is clamped to a configurable window (10^-3 to 3 m²) and falls back to a fixed
1 m² floor if the KDE fails or the valley is implausible. Footprints fully
contained within another (ICC digitising errors) are also dropped.

### 3. Shared-wall detection

A Shapely 2.0 `STRtree` is queried in bulk with all buffered geometries at
once (vectorised, no Python loop). For each candidate pair the shared
plan-length is measured as the average of two boundary-intersection lengths,
with an end-cap correction of -2*EPS to remove the systematic bias introduced
by the buffer circles at segment endpoints:

```
la = length( boundary(j) ∩ buffer(i, EPS) )
lb = length( boundary(i) ∩ buffer(j, EPS) )
shared_length = max( 0,  0.5 * (la + lb) - 2 * EPS )
```

This is robust to the half-perimeter problem of the original approach, which
broke when two footprints genuinely overlapped in 2-D (a digitising error).
Genuine 2-D overlaps are flagged in `output/overlap_audit.csv` for review.

### 4. Deduction and exposed area

For each shared segment of plan-length L between buildings i and j, the
deductions are **asymmetric**:

```
ded_i = L * clip( min(Alc_Top_i, Alc_Top_j) - Alc_Base_i,  0,  h_i )
ded_j = L * clip( min(Alc_Top_i, Alc_Top_j) - Alc_Base_j,  0,  h_j )
```

On flat ground (equal base elevations) this reduces to the classical
`min(height)` formula, preserving consistency with earlier validated results.
On sloped terrain it gives the physically correct answer: the wall below the
higher neighbour's base is modelled as buried against soil and therefore not
exposed to outdoor air.

Deductions are accumulated with `numpy.add.at` over all pairs simultaneously
(vectorised, no Python loop). The deduction fraction is audited before
clipping so that any over-deduction (a sign of remaining data errors) is
visible in the output.

For full technical details, design decisions, and known limitations see
[docs/algorithm.md](docs/algorithm.md).

---

## Requirements

```
geopandas >= 1.0
shapely >= 2.0      # bulk array API required
numpy
scipy
pandas
```

Install with:

```bash
pip install geopandas shapely numpy scipy pandas
```

---

## Data

Place the ICC building GeoPackage at:

```
data/buildings.gpkg
```

The file is not included in this repository (571 MB). It can be obtained from
the ICC download service. The `data/` directory is listed in `.gitignore`.

---

## Usage

```bash
python envelope_area.py [--input FILE] [--output DIR]
```

Default input: `data/buildings.gpkg`. Default output: `output/`.

Run with explicit paths:

```bash
python envelope_area.py --input /path/to/buildings.gpkg --output /path/to/results/
```

Results are written to `<output>/buildings_exposed.gpkg`. A summary report is
printed to the terminal and written to `<output>/report.txt`. An overlap audit
(pairs of buildings with genuine 2-D footprint overlaps, indicating ICC
digitising errors) is written to `<output>/overlap_audit.csv` when such pairs
are found.

Run the synthetic test suite before any production run:

```bash
python tests.py
```

All tests complete in under one second and verify the core algorithms
against hand-computed expected values without requiring the full dataset.

---

## Output

`output/buildings_exposed.gpkg` contains one row per building with the
following added columns:

| Column | Description |
|--------|-------------|
| `height` | Building height in metres (`Alc_Top - Alc_Base`, after repair) |
| `footprint_area` | 2-D polygon area (m²) |
| `roof_area` | Exposed roof area = `footprint_area` (flat-roof assumption) |
| `wall_area_naive` | Perimeter × height, as if the building stood in isolation |
| `wall_deduction` | Total wall area hidden by adjacent buildings |
| `wall_area` | Exposed wall area = `wall_area_naive - wall_deduction` |
| `exposed_area` | Total exposed area = `roof_area + wall_area` |
| `deduction_fraction` | `wall_deduction / wall_area_naive` (0 = fully isolated, 1 = fully embedded) |
| `deduction_fraction_raw` | Same ratio before the clip(0) safety net; values > 1 indicate over-detection |

---

## Reference results (full AMB, previous pipeline version)

The original pipeline produced the following totals on the full 851,505-building
dataset. These serve as a regression baseline; the corrected pipeline will
differ on sloped tiles (where the old `min(height)` formula was wrong) but
should agree closely on flat tiles.

| Quantity | Value |
|----------|-------|
| Buildings processed | 850,136 |
| Total roof area | 44,440,670 m² |
| Total naive wall area | 363,900,900 m² |
| Total wall deduction | 257,686,000 m² |
| Total exposed area | 152,386,300 m² |

---

## Author

Aleix Alva, aleix.alva@upc.edu
