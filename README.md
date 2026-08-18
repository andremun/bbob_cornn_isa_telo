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
├── CONFIGURATION.md                 # Configuration guide
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
│   └── run_all_performance.sh       # All performance data (7020 tasks)
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
cp suite_largescale.c coco/build/suite_largescale.c
cd coco
python do.py run-python
pip install code-experiments/build/python
```

`suite_largescale.c` is provided in the root of this repository.

### MATLAB toolboxes

The following MATILDA toolbox scripts are bundled directly in this repository:
`TRACE.m`, `scriptfcn.m`, `KNNRegressor.m`. No separate MATILDA installation
is needed.

The `tsne` call in `shared_generate_instance_space.m` uses MATLAB's built-in
t-SNE (Statistics and Machine Learning Toolbox, R2017b or later). The
`tsne_d` call for feature clustering uses the same toolbox via a
distance-matrix input (`tsne(D, 'Distance', 'precomputed')`). No external
t-SNE toolbox is required.

Required MATLAB toolboxes: Statistics and Machine Learning Toolbox.

### Python environment

This project requires Python 3.10. All dependencies, including `pflacco`
(which requires `numpy~=1.24.3`) and `nevergrad`, are compatible with
Python 3.10.

```bash
pip install -r requirements.txt
```

---

## Sample mode (local verification)

To run the full pipeline locally without a cluster — for example, to verify
each step produces the expected output — use sample mode:

```bash
export SAMPLE_MODE=1
```

In sample mode the scripts restrict to 2 BBOB functions (f1, f8) × 3 instances
× dimension 41 × 1 replicate, and 1 CORNN function × 1 architecture. The full
pipeline completes in approximately 10 minutes on a standard laptop.

For MATLAB post-processing in sample mode:
```matlab
setenv('SAMPLE_MODE', '1');
shared_consolidate_raw_data
shared_generate_instance_space
```

Intermediate data (Sobol grids, ELA features, AUC tables) for dimension 41
are available on Figshare (see the paper for the DOI), allowing Steps 1–4
to be skipped entirely for post-processing verification.

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

Individual sections of `shared_generate_instance_space.m` can be run
independently using the section flags at the top of the script (e.g. set
`RUN_FOOTPRINTS = false` to skip footprint estimation).

---

## Verification: expected outputs

| Step | Expected output | Quick check (full mode) | Quick check (sample mode) |
|---|---|---|---|
| 1 — Sobol grids | CSV files in `input/` | `ls input/ \| wc -l` → 15 | → 3 |
| 2 — BBOB raw | CSV files in `bbob/raw/` | `ls bbob/raw/ \| wc -l` → 5400 | → 6 |
| 3 — ELA features | CSV files in `bbob/ela/` | `ls bbob/ela/ \| wc -l` → 360 | → 2 |
| 4 — Performance | CSV files across `bbob/` and `cornn/` | `ls bbob/nevergrad/ \| wc -l` → 129600 | → 180 |
| 5 — MATLAB | Files in `isa/` | `BBOB_CORNN_metadata.csv` exists | same |

---

## Configuration

See [CONFIGURATION.md](CONFIGURATION.md) for a full description of all
tuneable parameters, data paths, algorithm portfolio options, and
instructions for adapting the pipeline to a different benchmark or machine.

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

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{Malan2026cornn,
  title   = {An Instance Space Analysis of Neural Network Training as a
             Black-Box Optimisation Problem},
  author  = {Malan, Katherine Mary and Mu{\~n}oz, Mario Andr{\'e}s},
  journal = {ACM Transactions on Evolutionary Learning and Optimization},
  year    = {2026},
  note    = {Accepted}
}

@misc{MunozData2026,
  author    = {Mu{\~n}oz, Mario Andr{\'e}s},
  title     = {{ISA} of Neural Network Training as a {BBO} Problem},
  year      = {2026},
  publisher = {FigShare},
  doi       = {10.26188/32609130},
  url       = {https://doi.org/10.26188/32609130}
}
```

---

## License

PolyForm Noncommercial License 1.0.0.
Copyright (c) 2026 Mario Andres Munoz Acosta, University of Melbourne.