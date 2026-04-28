# Repository Guidelines

## Project Structure & Module Organization

This repository contains Python analysis scripts for Ce/Zr/O POSCAR structures and generated descriptor plots.

- `training_set/`: input VASP POSCAR files used by all analysis scripts.
- `analyze_ce_ce_distance.py`: Ce-centered distance descriptors.
- `analyze_ce_site_symmetry.py`: Ce-O coordination, angular dispersion, and site-symmetry descriptors.
- `cluster_correlations.py`: pair, triplet, and quadruplet Ce/Zr cluster-correlation vectors.
- `cluster_correlation_heatmaps.py`: heatmap visualizations and cluster-type summaries.
- `*.png` and `ce_zr_cluster_heatmap_summary.txt`: generated analysis outputs.
- `Literature/`: supporting papers and a short literature index.

There is currently no package layout or dedicated test directory.

## Build, Test, and Development Commands

Install runtime dependencies:

```bash
python -m pip install numpy matplotlib
```

Run analyses from the repository root:

```bash
python analyze_ce_ce_distance.py
python analyze_ce_site_symmetry.py
python cluster_correlations.py
python cluster_correlation_heatmaps.py
```

Check syntax for all scripts:

```bash
python -m py_compile analyze_ce_ce_distance.py analyze_ce_site_symmetry.py cluster_correlations.py cluster_correlation_heatmaps.py
```

## Coding Style & Naming Conventions

Use Python 3, 4-space indentation, and clear snake_case names for functions, variables, and output files. Keep scripts self-contained unless shared code becomes substantial enough to justify a helper module. Prefer explicit filenames such as `ce_zr_cluster_order_2.png` and preserve numeric POSCAR naming patterns like `POSCAR.1`.

Avoid committing generated caches such as `__pycache__/` or `.pyc` files.

## Testing Guidelines

No formal test framework is configured. Before committing, run `python -m py_compile ...` and at least one representative analysis script. When changing parsing, periodic-distance logic, or cluster labeling, run all four scripts and verify that expected plots and summaries are regenerated without errors.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or descriptive messages, for example `Rename training data folder` and `Added relevant literature`. Keep commits focused on one logical change.

Pull requests should include:

- A brief summary of the descriptor or workflow change.
- Any regenerated output files.
- Notes on commands run for verification.
- Screenshots or references to changed plots when visual outputs are affected.

## Data & Literature Notes

Treat `training_set/` as the canonical input dataset. If adding structures, document the source or enumeration criteria in `README.md`. Keep literature PDFs in `Literature/` and update `Literature/README.md` with why each paper is relevant.
