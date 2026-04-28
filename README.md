# DielectricDescriptors

This repository contains descriptor analysis code for ceria-doped zirconia structures, with the goal of capturing configurational variance in dielectric tensor behavior.

## Dataset

- `DielectricDescriptors/training_set/` contains POSCAR files for enumerated Ce/Zr/O structures.
- Each POSCAR is one unique doped structure and uses Ce/Zr substitution on the zirconia lattice.
- Most structures have a single Ce atom per cell, so local site descriptors are based on the Ce oxygen environment and periodic Ce/Zr cluster arrangements.

## Scripts

### `analyze_ce_ce_distance.py`

- Computes Ce-based distance descriptors across all structures.
- Extracts:
  - periodic minimum Ce–Ce distance proxy
  - nearest Ce–Zr distance
  - nearest Ce–O distance
- Saves plot: `ce_descriptor_plot.png`

### `analyze_ce_site_symmetry.py`

- Computes a local Ce site symmetry score from the Ce–O neighbor shell.
- Derives descriptors including:
  - number of nearest O neighbors
  - mean Ce–O distance
  - Ce–O distance standard deviation
  - angular dispersion around Ce
  - symmetry score
- Saves plots:
  - `ce_site_symmetry.png`
  - `ce_coord_vs_symmetry.png`
  - `ce_angle_descriptor.png`

### `cluster_correlations.py`

- Computes cluster correlation vectors for Ce/Zr occupancy.
- Uses +1 for Ce and -1 for Zr occupancy.
- Calculates pair, triplet, and quadruplet correlation values for each structure.
- Saves plots:
  - `ce_zr_cluster_order_2.png`
  - `ce_zr_cluster_order_3.png`
  - `ce_zr_cluster_order_4.png`

### `cluster_correlation_heatmaps.py`

- Produces literature-style heatmaps of cluster correlations.
- Generates structure-by-cluster-type matrices for orders 2, 3, and 4.
- Saves plots:
  - `ce_zr_cluster_heatmap_order_2.png`
  - `ce_zr_cluster_heatmap_order_3.png`
  - `ce_zr_cluster_heatmap_order_4.png`
- Generates summary file:
  - `ce_zr_cluster_heatmap_summary.txt`

## Findings so far

- There are 16 unique structures in `training_set`.
- The local Ce symmetry analysis identified 14 distinct Ce environments across these structures.
- Local descriptors are based on Ce–O coordination and bond-angle dispersion.
- Cluster correlation analysis captures Ce/Zr occupancy patterns and provides pair/triplet/quadruplet descriptors for configurational sampling.

## Usage

Install dependencies:

```bash
python -m pip install numpy matplotlib
```

Run the analysis scripts from the repository root:

```bash
python DielectricDescriptors/analyze_ce_ce_distance.py
python DielectricDescriptors/analyze_ce_site_symmetry.py
python DielectricDescriptors/cluster_correlations.py
python DielectricDescriptors/cluster_correlation_heatmaps.py
```

## Notes

- The current descriptor workflow is exploratory and intended to identify representative structures for dielectric tensor sampling.
- `cluster_correlation_heatmaps.py` is the preferred visualization for comparing how different structures occupy Ce/Zr clusters in the enumerated set.
- Future work can extend this repository by adding dielectric tensor data, filtering representative structures, and correlating descriptors to actual VASP results.
