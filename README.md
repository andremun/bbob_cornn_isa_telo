# BBOB-CORNN Instance Space Analysis

Code for the instance space analysis (ISA) comparing the CORNN neural network
training benchmark suite with the BBOB large-scale noiseless benchmark suite.

---

## Repository structure

```
cornn_telo/
├── cornn/
│   ├── __init__.py
│   └── config.py                    # All constants and paths (single source of truth)
├── cornn_config.m                   # MATLAB mirror of cornn/config.py
├── bbob_collect_raw_data.py         # Evaluate BBOB functions on Sobol grids
├── bbob_run_nevergrad.py            # BBOB nevergrad optimiser runs
├── bbob_run_adam.py                 # BBOB finite-difference Adam runs
├── bbob_run_pflacco.py              # BBOB ELA feature computation
├── cornn_collect_raw_data.py        # Evaluate CORNN loss landscapes on Sobol grids
├── cornn_run_nevergrad.py           # CORNN nevergrad optimiser runs
├── cornn_run_adam.py                # CORNN finite-difference Adam runs
├── cornn_run_pflacco.py             # CORNN ELA feature computation
├── shared_collect_input_samples.m   # Sobol grid generation (run once)
├── shared_consolidate_raw_data.m    # AUC computation for both suites
├── shared_generate_instance_space.m # Full ISA pipeline (PCA, t-SNE, TRACE, figures)
├── TRACE.m                          # Footprint estimation toolbox
├── scriptfcn.m                      # ISA helper functions
├── tblvertcat.m                     # Table vertical concatenation utility
├── daviolinplot.m                   # Violin plot utility
├── KNNRegressor.m                   # KNN regression model (MATILDA)
├── slurm/
│   ├── run_collect_raw_data.sh      # Stage 1: raw landscape data (1404 tasks)
│   ├── run_collect_pflacco.sh       # Stage 2: ELA features (684 tasks)
│   ├── run_all_performance.sh       # All performance data (7020 tasks)
├── migrate_data.sh                  # One-off: migrate data from old layout
├── suite_largescale.c               # Modified COCO source (dims 41, 261, 481)
├── requirements.txt                 # Python 3.10 environment
└── README.md
```

---

## Data layout

All data lives under a single root directory (default: `~/punim0320/bbob_cornn_isa/`
on the cluster, `D:/bbob_cornn_isa/` on Windows). Override via the environment
variable `BBOB_CORNN_ROOT`.

```
bbob_cornn_isa/
├── input/           Sobol grids  X_D{dim}_S100_R{sid}.csv
├── bbob/
│   ├── raw/         BBOB landscape evaluations
│   ├── ela/         BBOB ELA features
│   ├── nevergrad/   BBOB nevergrad runs
│   ├── adam/        BBOB Adam runs
│   └── meta/        bbob_fopt.csv
├── cornn/
│   ├── raw/         CORNN loss landscape evaluations
│   ├── ela/         CORNN ELA features
│   ├── nevergrad/   CORNN nevergrad runs
│   └── adam/        CORNN Adam runs
└── isa/             MATLAB post-processing output
```

---

## Installation

### BBOB (custom build)

The standard `cocoex` package does not support dimensions {41, 261, 481}.
Build COCO v2.6.100 from source with the modified `suite_largescale.c`
(provided in this repository):

```bash
git clone https://github.com/numbbo/coco.git
cd coco
# Replace build/suite_largescale.c with the version in this repository
python do.py run-python
pip install code-experiments/build/python
```

### MATLAB toolboxes

The following MATILDA toolbox scripts are bundled directly in this repository:
`TRACE.m`, `scriptfcn.m`, `KNNRegressor.m`. No separate MATILDA installation
is needed.

The t-SNE implementation used in `shared_generate_instance_space.m` is loaded
via `addpath('.\tSNE_matlab\')`. Download the Barnes-Hut t-SNE toolbox by
van der Maaten from https://lvdmaaten.github.io/tsne/ and place it in a
`tSNE_matlab/` subdirectory alongside the scripts, or replace the `tsne_d`
call with MATLAB's built-in `tsne` if the Statistics Toolbox is available.

### Python environment

This project requires Python 3.10. All dependencies, including `pflacco`
(which requires `numpy~=1.24.3`) and `nevergrad`, are compatible with
Python 3.10.

```bash
pip install -r requirements.txt
```

---

## Execution order

### Step 1 — Generate Sobol input grids (MATLAB, run once)

```matlab
shared_collect_input_samples   % generates input/ X_D{dim}_S100_R{sid}.csv files
```

### Step 2 — Collect raw landscape data (cluster)

```bash
sbatch slurm/run_collect_raw_data.sh
```

### Step 3 — Compute ELA features (cluster, after Step 2)

```bash
sbatch --dependency=afterok:<STEP2_JOBID> slurm/run_collect_pflacco.sh
```

### Step 4 — Run optimiser performance experiments (cluster)

```bash
sbatch slurm/run_all_performance.sh
```

### Step 5 — Post-processing (MATLAB)

```matlab
shared_consolidate_raw_data        % computes AUC, processes ELA
shared_generate_instance_space     % ISA pipeline, figures, tables
```

---

## Data migration

If you have data from the previous `TELO_DATA` layout, migrate it with:

```bash
bash migrate_data.sh --dry-run   # preview
bash migrate_data.sh             # execute
```

---

## Algorithm portfolio

| Algorithm | Type | Notes |
|---|---|---|
| CMA-ES | Evolution strategy | via nevergrad |
| PSO | Particle swarm | via nevergrad |
| TwoPointsDE | Differential evolution | via nevergrad |
| RandomSearch | Baseline | via nevergrad |
| Adam | Gradient-based | finite-difference, forward differences |

---

## Canonical filename patterns

| File type | Pattern |
|---|---|
| Sobol grid | `X_D{dim}_S100_R{sid}.csv` |
| BBOB raw | `F{fid}_D{dim}_I{iid}_S100_R{sid}.csv` |
| CORNN raw | `F_{fcn}_{arch}_S100_R{sid}.csv` |
| BBOB ELA | `ELA_F{fid}_D{dim}_S100_R{sid}.csv` |
| CORNN ELA | `ELA_F_{fcn}_{arch}_S100_R{sid}.csv` |
| BBOB nevergrad | `{problem_id}_{algo}_R{run}.csv` |
| BBOB Adam | `{problem_id}_Adam_R{run}.csv` |
| CORNN nevergrad | `F_{fcn}_{arch}_{algo}_R{run}.csv` |
| CORNN Adam | `F_{fcn}_{arch}_Adam_R{run}.csv` |

---

## License

PolyForm Noncommercial License 1.0.0.
Copyright (c) 2026 Mario Andres Munoz Acosta, University of Melbourne.
