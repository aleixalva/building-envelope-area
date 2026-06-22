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
once (vectorised, no Python loop). For each candidate pair, two boundary-
intersection lengths are measured and corrected for buffer end-cap bias:

```
la = length( boundary(j) ∩ buffer(i, EPS) )
lb = length( boundary(i) ∩ buffer(j, EPS) )
L_for_i = max( 0, lb - 2*EPS )   [used for building i's deduction]
L_for_j = max( 0, la - 2*EPS )   [used for building j's deduction]
```

The two lengths are kept separate (asymmetric) rather than averaged. For a
symmetric straight-wall pair (`la = lb`) the result is identical to averaging.
For a concave-wrapping pair (a U-shaped building enclosing a smaller one),
`la != lb`; using each measurement directly for its own building prevents the
inflation in the enclosed building's deduction that averaging would cause.
Genuine 2-D footprint overlaps are flagged in `output/overlap_audit.csv` for
review.

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

## Setup

All commands below are meant to be typed into a terminal.

### Step 1 — Install uv

`uv` is a tool that installs Python and all required libraries automatically,
in an isolated environment that cannot interfere with anything else on your
computer. Install it once with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After the installer finishes, close and reopen your terminal so the `uv`
command becomes available. You can verify the installation with:

```bash
uv --version
```

### Step 2 — Get the dataset

The ICC building GeoPackage (`EdificiosAMB.gpkg`, 571 MB) is not included in
this repository. Obtain it from the ICC download service and place it
anywhere on your system.

### Step 3 — Create the environment

Open a terminal, navigate to the repository folder, and run:

```bash
uv sync
```

This does three things automatically:
1. Downloads Python 3.14 if it is not already on your machine.
2. Creates a self-contained environment in a folder called `.venv/` inside the
   repository.
3. Installs every library at exactly the version used for the original run
   (recorded in `uv.lock`).

The first run takes a couple of minutes while packages are downloaded;
subsequent runs are nearly instant because everything is cached.

---

## Usage

### Verify the installation first

Before processing the full dataset, run the test suite to confirm the
environment is working correctly:

```bash
uv run python tests.py
```

All tests finish in under one second and check the core calculations against
hand-computed values. You should see no errors.

The prefix `uv run` tells the shell to use the project's own environment in
`.venv/` rather than any other Python installation that may be on your system.
Use it for every command below.

### Run the pipeline

```bash
uv run python envelope_area.py --input /path/to/buildings.gpkg --output /path/to/results/
```

The `--output` folder will be created if it does not exist. If you omit either
flag, the script falls back to `data/buildings.gpkg` for input and `output/`
for output, both relative to the repository folder.

The pipeline prints a progress summary to the terminal as it runs and writes
three files when it finishes:

| File | Contents |
|------|----------|
| `output/buildings_exposed.gpkg` | Main result — one row per building (see [Output](#output)) |
| `output/report.txt` | The same summary that was printed to the terminal |
| `output/overlap_audit.csv` | Pairs of buildings with genuine 2-D footprint overlaps (ICC digitising errors); absent if none were found |


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

## Author

Aleix Alva, aleix.alva@upc.edu
