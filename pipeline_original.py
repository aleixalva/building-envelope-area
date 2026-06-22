
__generated_with = "0.23.10"

# %%
import marimo as mo
import geopandas as gpd          # GeoDataFrame: tabular data with geometry column
import numpy as np               # array operations and math utilities
import time

START_TIME = time.time()

from shapely import make_valid, force_2d  # geometry repair tools:
                                 #   force_2d: strip Z coordinates from 3D geometries
                                 #   make_valid: fix degenerate rings and self-intersections
from scipy.stats import gaussian_kde  # kernel density estimation, used to find
                                      # the natural size threshold between artifact
                                      # slivers and real building footprints
from shapely.strtree import STRtree

# %%
mo.md(r"""
# Exposed surface area of buildings from ICC LoD1 data

This notebook computes the exterior surface area of buildings exposed to the
outside environment, for a single tile of the ICC (Institut Cartografic de
Catalunya) topographic dataset. The input is the `_con.shp` shapefile
(construccio layer), which provides building footprints as 3D polygons with
absolute roof and base elevations.

The problem is not trivial because buildings in a dense urban fabric share
walls with their neighbors. A naive computation of perimeter * height
double-counts party walls and overestimates the exposed surface. The correct
exposed wall area requires detecting all shared boundaries between adjacent
buildings and subtracting the hidden portions, accounting for height
differences: if two buildings of different heights share a wall, the taller
one retains an exposed strip above the shorter neighbor.

The pipeline proceeds as follows:

1. Load and inspect the raw shapefile
2. Repair 3D geometry artifacts and compute building heights
3. Remove artifact slivers using an automatic threshold on the footprint area distribution
4. Detect shared wall segments between all neighboring building pairs
5. Compute exposed wall and roof area per building
6. Visualise results interactively
""")

# %%
# Load the building footprint shapefile for one ICC tile.
# The ICC (Institut Cartografic de Catalunya) topographic dataset uses the
# naming convention <tile_id>_con.shp where "con" stands for "construccio"
# (construction), the layer containing building polygons.
#
# Expected output:
#   CRS: EPSG:25831 (ETRS89 / UTM zone 31N), the standard for Catalonia.
#   Coordinates are in metres, so area and length computations are direct.
#
# Columns of interest:
#   Alc_Top:  absolute elevation of the roof (metres above sea level)
#   Alc_Base: absolute elevation of the ground at the building base
#   Both are stored as strings in the shapefile and need casting later.
#
# Geometry type: MULTIPOLYGON Z (3D), where Z encodes the roof elevation.
# The Z coordinate is redundant with Alc_Top and will be stripped in cell 3.

gdf_raw = gpd.read_file("data/EdificiosAMB.gpkg")
print(gdf_raw.shape, gdf_raw.crs)  # print shape (rows and columns, and Coordinate Reference System [CRS])
print(gdf_raw.dtypes)  # print data types

# %%
# Geometry repair and height computation.
#
# force_2d: strips the Z coordinate from MULTIPOLYGON Z geometries.
#   The Z values are redundant with Alc_Top and cause GEOS (the underlying
#   geometry library) to report all 6788 polygons as invalid with the error
#   "too few points in geometry component". This is an issue with
#   3D geometries that have degenerate rings (rings with fewer than 4 points
#   that collapse to a point or line when projected to 2D).
# GEOS requires a valid ring to have at least 4 points -- 3 unique vertices plus the closing repeat.
# A "degenerate ring" is one with fewer than 4 points, which means it cannot enclose any area
# -- it collapses to a point (1 point), a line segment (2 points),
# or an unclosed triangle (3 points without the closing repeat).
# In this dataset, the ICC extraction pipeline left some of these degenerate rings as sub-polygons within MultiPolygons
# -- artifacts of the automated building outline extraction from lidar.
# They are physically meaningless (zero area) and GEOS flags the whole geometry
# as invalid because of them. make_valid simply discards them and keeps the real polygon parts.
#
# make_valid: discards those degenerate rings and repairs the polygons.
#   After force_2d this brings all geometries to valid status. The geometry
#   type changes from MULTIPOLYGON to POLYGON for most buildings (6770/6788),
#   with 18 remaining as MULTIPOLYGON (L-shapes, courtyards, etc).
#
# Alc_Top and Alc_Base are absolute elevations in metres above sea level,
#   stored as strings in the shapefile. They are cast to float here.
#
# height = Alc_Top - Alc_Base: the building height in metres.
#   A small number of buildings have negative values due to lidar ground model
#   errors where the interpolated base elevation exceeds the roof elevation.
#   These are clamped to zero with clip(lower=0.0) -- they have valid
#   footprints but contribute no wall area.
#
# footprint_area: the 2D polygon area in m², used in the next cell to
#   separate real buildings from artifact slivers.

gdf_fixed = gdf_raw.copy()  # get a new copy
gdf_fixed["geometry"] = make_valid(force_2d(gdf_raw.geometry))  # strip z and discard degenerate rings
gdf_fixed["Alc_Top"]  = gdf_fixed["Alc_Top"].astype(float)  # cast string to float
gdf_fixed["Alc_Base"] = gdf_fixed["Alc_Base"].astype(float)  # from string to float
gdf_fixed["height"]   = (gdf_fixed["Alc_Top"] - gdf_fixed["Alc_Base"]).clip(lower=0.0)  # get the height of buildings
gdf_fixed["footprint_area"] = gdf_fixed.geometry.area  # column with 2D polygon area
print(gdf_fixed[["height", "footprint_area"]].describe())

# %%
gdf_fixed

# %%
mo.md(r"""
A **sliver** is a geometry artifact -- a polygon that is technically valid (it has enough points to close a ring) but is so thin or small that it represents no real physical object.

Slivers typically arise from:

- **Boundary operations**: when two polygons that should share an edge have a tiny gap or overlap due to floating point precision, intersection or difference operations produce a hair-thin polygon along that boundary
- **Coordinate snapping**: when polygon vertices from different sources don't align exactly, leaving a narrow strip between them
- **Automated extraction pipelines**: like the ICC lidar extraction here, where the algorithm produces degenerate outline fragments alongside the real building polygons

A sliver has a characteristic signature: very small area but non-negligible perimeter, giving it an extremely low **compactness** (the ratio $4\pi$*area/perimeter^2, which equals 1 for a circle and approaches 0 for an infinitely thin shape).
""")

# %%
mo.md(r"""
Both a histogram and a KDE are ways to estimate the probability distribution of a set of numbers from a finite sample, but they work differently.

**Histogram**: you divide the range into fixed bins and count how many values fall in each bin. The result is a step function -- blocky, and sensitive to where you place the bin edges. Two histograms of the same data with different bin widths or offsets can look quite different.

**KDE**: instead of binning, you place a smooth bump (the "kernel", typically a Gaussian bell curve) centered on each data point, then sum all those bumps into a single smooth curve. The `bw_method` parameter controls how wide each bump is -- narrow bumps preserve fine detail but fit noise; wide bumps give a smoother picture of the overall shape.

Visually: if you have 6788 data points, a KDE is like replacing each point with a small Gaussian and adding them all up. Where points are dense the bumps pile up into a tall peak; where points are sparse the bumps barely overlap and the curve stays low.

The key advantage for our use case is that a KDE is a smooth differentiable function, so finding its minimum between two peaks is a well-defined mathematical operation -- you can simply evaluate it on a fine grid and call `np.argmin`. A histogram minimum is sensitive to bin placement and would require more careful handling.
""")

# %%
# Automatic artifact threshold using kernel density estimation on log10(area).
#
# The footprint area distribution is strongly bimodal in log space:
#   - left peak:  artifact slivers produced by the geometry repair in cell 3,
#                 with areas down to ~1e-9 m² (essentially points and lines
#                 that make_valid reduced to near-zero polygons)
#   - right peak: real buildings, centered around ~10 m² for this dense tile
#
# Working in log10 space is essential because the two populations are separated
# by several orders of magnitude -- a linear scale would bury the artifact
# peak against the axis.
#
# gaussian_kde fits a smooth continuous density to the log-area histogram.
# bw_method=0.30 is a deliberately coarse bandwidth: a fine bandwidth (e.g.
# 0.05) fits the noise within the artifact cluster and produces many spurious
# local minima, making it impossible to find the valley automatically.
#
# The threshold is the minimum of the KDE to the left of the main (right) peak.
# This is more robust than searching for the valley between two detected peaks,
# which fails when the artifact peak is too small to register as a clear mode.
#
# The threshold found for this tile (~0.003 m²) should adapt automatically to
# other ICC tiles where the artifact population may differ in size or position.
# The valley between the two humps was found at log10(area) = -2.54.
# This means: 10^(-2.54) = 0.0029 m²

log_areas = np.log10(gdf_fixed["footprint_area"].values) # areas in logarithmic scale
kde_coarse = gaussian_kde(log_areas, bw_method=0.30) # create the kde
# bw_method=0.30 means the standard deviation of each Gaussian bump is 0.30 units in the space where the KDE is computed

x_grid = np.linspace(log_areas.min(), log_areas.max(), 2000)
density_coarse = kde_coarse(x_grid)

main_peak_idx = np.argmax(density_coarse)
main_peak = x_grid[main_peak_idx]

left_of_main = x_grid < main_peak  # mask
threshold_log  = x_grid[left_of_main][np.argmin(density_coarse[left_of_main])] # min within mask
threshold_area = 10 ** threshold_log  # undo the logarithm to recover the area

print(f"Main mode:  10^{main_peak:.2f} = {10**main_peak:.2f} m²")
print(f"Threshold:  10^{threshold_log:.2f} = {threshold_area:.4f} m²")
print(f"Artifacts to drop: {(gdf_fixed['footprint_area'] < threshold_area).sum()}")

# %%
# Filter out artifact geometries below the automatic threshold
#
# .copy() creates an independent GeoDataFrame rather than a view of gdf_fixed,
# avoiding pandas SettingWithCopyWarning when we add columns in later cells.
#
# reset_index(drop=True) reindexes the rows from 0 to N-1. This is important
# for cell 6, where the shared wall detection uses integer positional indices
# to identify building pairs and accumulate deductions into a numpy array.
# Without this reset, the index would have gaps (the dropped artifact rows)
# and array indexing would silently produce wrong results.

# Filter out artifact geometries below the automatic threshold from the previous cell.
# .copy() creates an independent GeoDataFrame rather than a view of gdf_fixed,
# avoiding pandas SettingWithCopyWarning when we add columns in later cells.
# reset_index(drop=True) reindexes the rows from 0 to N-1. This is important
# for the shared wall detection cell, where integer positional indices
# identify building pairs and accumulate deductions into a numpy array.
# Without this reset, the index would have gaps (the dropped artifact rows)
# and array indexing would silently produce wrong results.
gdf_clean = gdf_fixed[gdf_fixed["footprint_area"] >= threshold_area].copy().reset_index(drop=True)

# Detect and remove contained buildings: cases where one building footprint
# lies almost entirely inside another, which is physically impossible and
# indicates a digitizing error in the ICC dataset (a large block polygon
# digitized on top of the individual unit polygons within it).
# A spatial index is built to avoid O(n^2) comparisons.
_tree5 = STRtree(gdf_clean.geometry)
_to_drop = set()
for _bi, _geom_bi in enumerate(gdf_clean.geometry):
    for _bj in _tree5.query(_geom_bi):
        if _bj <= _bi:
            continue
        # compute the exact 2D intersection of the two footprints
        _overlap = _geom_bi.intersection(gdf_clean.geometry[_bj])
        # only 2D overlaps (Polygon/MultiPolygon) indicate containment;
        # linear intersections are shared edges and are handled in the wall detection cell
        if _overlap.is_empty or _overlap.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        _smaller = min(gdf_clean.loc[_bi, "footprint_area"], gdf_clean.loc[_bj, "footprint_area"])
        # if the overlap covers more than 90% of the smaller footprint,
        # the smaller building is considered contained within the larger one
        if _overlap.area / _smaller > 0.9:
            # drop the smaller of the two -- the larger polygon is the authoritative one
            _to_drop.add(_bi if gdf_clean.loc[_bi, "footprint_area"] < gdf_clean.loc[_bj, "footprint_area"] else _bj)

gdf_clean = gdf_clean.drop(index=_to_drop).reset_index(drop=True)
print(f"Buildings retained: {len(gdf_clean)} (dropped {len(_to_drop)} contained)")

# %%
gdf_clean

# %%
# check overlapping buildings
overlapping_pairs = []
for _bi, _geom_bi in enumerate(gdf_clean.geometry):
    for _bj in STRtree(gdf_clean.geometry).query(_geom_bi):
        if _bj <= _bi:
            continue
        _overlap = _geom_bi.intersection(gdf_clean.geometry[_bj])
        if not _overlap.is_empty and _overlap.geom_type in ("Polygon", "MultiPolygon"):
            if _overlap.area > 1e-4:
                overlapping_pairs.append((_bi, _bj, _overlap.area))

overlap_areas = np.array([_a for _, _, _a in overlapping_pairs])
print(f"Genuinely overlapping pairs: {len(overlapping_pairs)}")
print(f"  min:    {overlap_areas.min():.6f} m²")
print(f"  median: {np.median(overlap_areas):.6f} m²")
print(f"  mean:   {overlap_areas.mean():.6f} m²")
print(f"  max:    {overlap_areas.max():.4f} m²")
print(f"  > 0.01 m²: {(overlap_areas > 0.01).sum()}")
print(f"  > 0.10 m²: {(overlap_areas > 0.10).sum()}")
print(f"  > 1.00 m²: {(overlap_areas > 1.00).sum()}")

# %%
# Detect shared wall segments between neighboring buildings.
#
# The core question is: for each pair of buildings, what length of boundary
# do they share? This length, multiplied by the smaller of the two heights,
# gives the wall area that must be subtracted from both buildings.
#
# STRtree is a spatial index (Sort-Tile-Recursive tree) that allows fast
# bounding-box queries. Without it, finding neighbors would require comparing
# every building against every other building -- O(n^2) = 6497^2 ~ 42 million
# comparisons. The spatial index reduces this to O(n log n) by first filtering
# to only geometries whose bounding boxes overlap.
#
# The buffer of eps=0.05 m serves two purposes:
#   1. Snapping tolerance: buildings that should share a wall but have a small
#      gap (due to digitizing imprecision) will have their buffered geometries
#      overlap, correctly detecting the shared boundary.
#   2. Converting a shared edge (1D, zero area) into a thin sliver (2D) that
#      intersection operations can reliably detect. Two polygons sharing an
#      exact edge produce a LineString intersection; with the buffer they
#      produce a thin Polygon whose boundary encodes the shared length.
#
# The condition j <= i ensures each pair (i, j) is processed only once,
# storing results with i < j as the canonical order.
#
# The intersection of two buffered polygons can be:
#   LineString / MultiLineString: the clean case -- two polygons sharing an
#     edge or nearly touching. The length is directly the shared wall length.
#   Polygon / MultiPolygon: two footprints that genuinely overlap in 2D,
#     meaning the buildings share not just a wall but a volume. The shared
#     wall length is approximated as half the overlap perimeter -- dividing
#     by 2 because the perimeter of the overlap region traces the boundary
#     twice (once for each building's contribution).
#   GeometryCollection: a mix of the above, which can occur at complex
#     junctions where several buildings meet at a point. Only the linear
#     parts are summed; point intersections (0D) contribute no wall area.
#
# The threshold L > 1e-6 m discards point-like intersections that survived
# as degenerate LineStrings with near-zero length.

tree = STRtree(gdf_clean.geometry)

eps = 0.05  # metres -- snapping tolerance
# buffering expands the polygon outward by eps metres in all directions, producing a slightly larger polygon.
# If two buildings have a gap smaller than eps, their expanded polygons overlap -- the gap is bridged.
# If two buildings share an exact edge, each expanded polygon crosses into the other's territory by eps, producing a thin rectangular sliver of width 2*eps along the shared edge.

shared = {}  # (i, j) -> shared boundary length in metres, with i < j

for i, geom_i in enumerate(gdf_clean.geometry):
    # i is the integer position (0 to N-1), geom_i is the Shapely geometry object

    buffered_i = geom_i.buffer(eps)
    # expand building i outward by eps=0.05 m in all directions
    # i.e. radially -- the buffer expands the polygon outward by exactly eps metres
    # measured perpendicularly from each point on the boundary, in every direction simultaneously.
    # The result is that straight edges shift outward by eps and corners become rounded arcs.
    # The consequence for a rectangular building footprint:
    # the four straight walls each shift outward by eps,
    # and the four corners become quarter-circle arcs of radius eps.
    # The buffered polygon is slightly larger and has rounded corners.

    candidate_idx = tree.query(buffered_i)
    # ask the spatial index: which buildings have a bounding box that overlaps
    # buffered_i? returns a list of integer positions -- a cheap pre-filter.
    # some candidates will not actually intersect once checked precisely below.

    for j in candidate_idx:
        if j <= i:
            continue
        # skip building i itself (j==i) and pairs we already processed (j<i),
        # ensuring each pair is stored once with i < j

        geom_j = gdf_clean.geometry[j]
        # retrieve the actual geometry of the candidate building

        intersection = buffered_i.intersection(geom_j.buffer(eps))
        # buffer only geom_i, not geom_j, to avoid double-counting at segment ends
        intersection = buffered_i.intersection(geom_j)
        # compute the exact geometric intersection of the two expanded polygons.
        # both are buffered so that gaps up to 2*eps are bridged and exact
        # shared edges produce a detectable sliver rather than a 1D line.

        if intersection.is_empty:
            continue
        # the bounding boxes overlapped (candidate_idx said so) but the actual
        # geometries do not -- discard this false positive from the spatial index

        if intersection.geom_type in ("LineString", "MultiLineString"):
            L = intersection.length
            # the intersection is a line -- this happens when one buffered polygon
            # just grazes the boundary of the other without producing a 2D overlap.
            # length gives the shared wall length directly.

        elif intersection.geom_type in ("Polygon", "MultiPolygon"):
            L = intersection.length / 2
            # the intersection is a 2D sliver -- the typical case when two buildings
            # share a wall or have a small gap. The sliver perimeter traces the
            # shared boundary twice (once on each side), so we halve it.

        elif intersection.geom_type == "GeometryCollection":
            L = sum(g.length for g in intersection.geoms
                    if g.geom_type in ("LineString", "MultiLineString"))
            # a mix of geometry types, occurring at complex junctions where several
            # buildings meet. only the linear parts contribute wall length;
            # any polygon parts are small enough to ignore here.

        else:
            continue
            # point intersection (two buildings touching at a single corner) --
            # no shared wall area, discard.

        if L > 1e-6:
            shared[(i, j)] = L
            # store the pair only if the shared length is physically meaningful.
            # 1e-6 m filters out degenerate near-zero lengths from floating point
            # arithmetic without discarding any real shared boundaries.

print(f"Shared wall pairs found: {len(shared)}")
print(f"Total shared boundary length: {sum(shared.values()):.1f} m")

# %%
# Compute exposed surface area for each building.
#
# The total exposed surface has two components:
#   roof_area:  the horizontal top surface, equal to the footprint area.
#               This assumes flat roofs, which is standard for LoD1 models
#               (Level of Detail 1 -- the ICC dataset represents buildings as
#               simple extruded footprints with no roof geometry).
#   wall_area:  the vertical lateral surface, computed as naive perimeter
#               times height, minus the area hidden by adjacent buildings.
#
# wall_area_naive = perimeter * height
#   This is the total wall area as if the building stood in isolation.
#   geometry.length returns the perimeter of the 2D footprint in metres.
#
# The deduction logic handles the 3D case correctly:
#   If building A (height h_A) and building B (height h_B) share a wall
#   segment of length L, the hidden wall height is min(h_A, h_B) -- the
#   portion where both buildings are present. The taller building retains
#   the exposed strip above the shorter neighbor: L * |h_A - h_B|, which
#   is automatically accounted for because we only subtract L * min(h_A, h_B)
#   from each building rather than L * h_i.
#
#   wall_deduction accumulates all deductions for each building by iterating
#   over the shared dict. It is a plain numpy array indexed by
#   the same integer positions as gdf_clean (guaranteed by reset_index).
#
# clip(lower=0.0) guards against floating point negatives that could arise
#   when the buffer-based shared length slightly overestimates the true
#   shared boundary, producing a deduction marginally larger than the naive
#   wall area for very small buildings.

gdf_clean["wall_area_naive"] = gdf_clean.geometry.length * gdf_clean["height"]
# perimeter * height gives the total wall area as if the building stood alone,
# with no neighbors. geometry.length is the perimeter of the 2D footprint in metres.

gdf_clean["roof_area"] = gdf_clean["footprint_area"]
# flat roof assumption (LoD1): roof area equals the footprint area exactly.

wall_deduction = np.zeros(len(gdf_clean))
# accumulator array, one entry per building, initialised to zero.
# will store the total wall area hidden by adjacent buildings.

for (idx_i, idx_j), shared_length in shared.items():
    # iterate over all neighboring pairs found before.
    # idx_i, idx_j are the integer positions of the two buildings in gdf_clean.
    # shared_length is the length of their common boundary in metres.

    h_i = gdf_clean.loc[idx_i, "height"]
    h_j = gdf_clean.loc[idx_j, "height"]
    # retrieve the height of each building in the pair.

    h_shared = min(h_i, h_j)
    # the hidden wall height is the smaller of the two heights: the portion
    # where both buildings are present and block each other.
    # the taller building retains an exposed strip of height |h_i - h_j|
    # above the shorter neighbor -- this is automatically preserved because
    # we only subtract shared_length * h_shared, not shared_length * h_i.

    wall_deduction[idx_i] += shared_length * h_shared
    wall_deduction[idx_j] += shared_length * h_shared
    # add the hidden wall area to both buildings in the pair.
    # a building can accumulate deductions from multiple neighbors,
    # hence += rather than =.

gdf_clean["wall_deduction"] = wall_deduction
# store the total deduction for each building as a column.

gdf_clean["wall_area"] = (gdf_clean["wall_area_naive"] - gdf_clean["wall_deduction"]).clip(lower=0.0)

# actual exposed wall area = naive wall area minus hidden portions.
# clip(lower=0.0) prevents negative values from floating point overestimation
# of shared lengths in the buffer-based detection.

gdf_clean["exposed_area"] = gdf_clean["roof_area"] + gdf_clean["wall_area"]
# total exposed surface = roof + exposed walls.

gdf_clean["deduction_fraction"] = (gdf_clean["wall_deduction"] / gdf_clean["wall_area_naive"]).fillna(0.0)

print("Exposed area stats (m²):")
print(gdf_clean["exposed_area"].describe())
print(f"\nTotal exposed area:      {gdf_clean['exposed_area'].sum():.0f} m²")
print(f"Total naive wall area:   {gdf_clean['wall_area_naive'].sum():.0f} m²")
print(f"Total wall deduction:    {gdf_clean['wall_deduction'].sum():.0f} m²")
print(f"Total roof area:         {gdf_clean['roof_area'].sum():.0f} m²")

# %%
import os

cols_to_save = ["Nivell", "Descripcio", "Full", "Alc_Top", "Alc_Base",
                "height", "footprint_area", "roof_area",
                "wall_area_naive", "wall_deduction", "wall_area","deduction_fraction", "exposed_area",
                "geometry"]

os.makedirs("output", exist_ok=True)
gdf_clean[cols_to_save].to_file("output/buildings_exposed.gpkg", driver="GPKG")
print(f"Saved {len(gdf_clean)} buildings to output/buildings_exposed.gpkg")

elapsed = time.time() - START_TIME

print(f"Elapsed: {elapsed:.1f} s")
print(f"Elapsed: {elapsed/60:.1f} min")
print(f"Elapsed: {elapsed/3600:.2f} h")

# %%
gdf_clean

# %%
mo.md(r"""
# Summary

The pipeline was applied to tile 421-03-29, covering a dense urban area of
approximately 700 x 470 m in the Barcelona metropolitan region.

## Data cleaning

The raw shapefile contained 6788 buildings, all flagged as geometrically
invalid due to degenerate rings in the 3D MultiPolygon representation. After
stripping the Z coordinate and repairing with `make_valid`, 291 artifact
slivers were removed using an automatic threshold derived from the bimodal
distribution of footprint areas in log10 space. The clean dataset contains
6497 buildings.

## Shared wall detection

A buffer-based intersection approach with a snapping tolerance of 0.05 m
detected 16067 shared wall pairs, with a total shared boundary length of
68655 m. The spatial index (STRtree) kept the computation tractable for a
dataset of this size.

## Exposed surface areas

| Quantity | Value |
|---|---|
| Total roof area | 214,566 m² |
| Total naive wall area | 2,662,376 m² |
| Total wall deduction | 2,061,630 m² |
| Total exposed wall area | 600,746 m² |
| Total exposed area | 825,919 m² |

The wall deduction represents 77% of the naive wall area, consistent with the
dense continuous block structure of the tile where most wall surface is party
wall. Buildings with deduction fraction equal to 1.0 (fully embedded,
zero exposed wall) are present and physically meaningful -- small structures
completely enclosed within a block.
""")
