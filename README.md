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
├── bbob_collect_meta.py             # Provenance: trigger COCO observer -> .dat logs
├── bbob_collect_fopt.m              # Provenance: extract fopt from COCO .dat logs
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
│   └── run_all_performance.sh       # All performance data (7020 tasks)
├── suite_largescale.c               # Modified COCO source (dims 41, 261, 481)
├── requirements.txt                 # Python 3.10 environment
├── CITATION.cff                     # Machine-readable citation metadata
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

The standard `cocoex` package (installed via `pip install coco-experiment`)
does not support dimensions {41, 261, 481}. A source build with a modified
`suite_largescale.c` (provided in the root of this repository) is required.

**Important — the upstream repository has been restructured.** The
`numbbo/coco` monorepo is now archived/outdated; the actively maintained
Python interface lives in a separate repository,
[`numbbo/coco-experiment`](https://github.com/numbbo/coco-experiment), with
a different build workflow than older instructions (`do.py run-python`)
describe. As of writing, the documented build process is:

```bash
git clone https://github.com/numbbo/coco-experiment.git
cd coco-experiment
# Locate the current path to suite_largescale.c under src/ and replace it
# with the version in this repository's root, e.g.:
find . -name suite_largescale.c
cp ../suite_largescale.c <path found above>

# Bundle the modified C sources into the language-specific build folders
python scripts/fabricate

# Build and install the Python bindings
cd build/python
pip install .
```

**Pin the exact version you build against.** The results in this paper
were produced with `cocoex` version `2.6.100-dev34+ga1bd588d` -- a
git-describe string indicating commit `a1bd588d`, 34 commits past the
`2.6.100` reference point. Check out this exact commit before building to
guarantee an identical build:

```bash
git checkout a1bd588d
```

To verify the version of an existing installation (`pip show
coco-experiment` may not resolve if the package was installed under a
different distribution name):

```bash
python -c "import cocoex; print(cocoex.__version__)"
```

**Compilation troubleshooting** (from upstream `DEVELOPMENT.md`):
- On macOS with ARM, use `arch -arm64 pip install .`.
- On older systems, you may need `CFLAGS="-std=c99" pip install .`.

### CORNN (neural network training benchmark)

CORNN (Malan & Cleghorn, 2022, *A Continuous Optimisation Benchmark Suite
from Neural Network Regression*, LNCS vol. 13398 / arXiv:2109.05606) is
Katherine Malan's benchmark suite and is **not** distributed on PyPI. It
must be cloned and installed in editable mode:

```bash
git clone https://github.com/CWCleghornAI/CORNN.git CORNN
cd CORNN
pip install -r requirements.txt
pip install -e .
```

The paper's methodology reports CORNN package v0.9; check out the
matching tag/commit if the repository provides one, or confirm the
installed version matches via `pip show CORNN`.

**Why the directory structure matters.** `CORNN/lib/CORNN.py` uses a
relative import (`import lib.Benchmark_Functions_2D_Definition`) that only
resolves when the current working directory *is* the cloned CORNN root.
This is why every CORNN-related script and SLURM job in this repository
must be run from inside that same directory — `CORNN_REPO_DIR` (see
"Running on a different machine or cluster" below) must point at the
cloned CORNN root itself, not merely at some venv folder. In practice this
means copying (or symlinking) this repository's `*.py` scripts, `cornn/`
package, and `cornn_config.m` into the CORNN root alongside its own `lib/`
directory. This is an artefact of CORNN's own import structure, not a
choice made in this repository.

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
× dimension 41 × 1 replicate, and 1 CORNN function × 1 architecture, with
runs reduced to 3 and the evaluation budget reduced to 500. Task dispatch is
bypassed entirely, so each bare `python <script>.py` command below processes
the whole sample subset in one go — no SLURM, no cluster, no environment
variables beyond `SAMPLE_MODE=1`:

```bash
export SAMPLE_MODE=1

python bbob_collect_raw_data.py    # both sample functions,        ~1 min
python bbob_run_pflacco.py         # both sample functions,        ~1 min
python bbob_run_nevergrad.py       # 4 algs x 2 fns, 3 runs each,  ~2 min
python bbob_run_adam.py            # 2 functions, 3 runs each,     ~1 min
python cornn_collect_raw_data.py   # first fn x first arch,        ~1 min
python cornn_run_pflacco.py        # first fn x first arch,        ~1 min
python cornn_run_nevergrad.py      # 4 algs, 3 runs each,          ~2 min
python cornn_run_adam.py           # 3 runs,                       ~1 min
```

The full pipeline completes in approximately 10 minutes on a standard
laptop, with no need to create the venv/repo directory layout the SLURM
scripts assume (see "Running on a different machine or cluster" below).
See [CONFIGURATION.md](CONFIGURATION.md) for exactly what each command
restricts and produces.

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

## Running on a different machine or cluster

The SLURM scripts in `slurm/` assume the authors' own directory layout: a
Python venv at `~/venvs/CORNN/` with this repository's scripts copied into
`~/venvs/CORNN/CORNN/` (required so `import lib.CORNN` resolves -- see
`CONFIGURATION.md`). Both paths are overridable without editing any script:

```bash
sbatch --export=CORNN_VENV_DIR=/path/to/your/venv,CORNN_REPO_DIR=/path/to/this/repo \
    slurm/run_collect_raw_data.sh
```

The `module load` lines are specific to SPARTAN (The University of Melbourne
HPC) and will not exist on a different cluster. Edit the `module purge` /
`module load` block near the top of each script in `slurm/` to match your
own environment's module names, or remove it entirely if you are using a
container or a pre-built environment where the packages in
`requirements.txt` are already installed and importable without modules.

If you are not using SLURM at all, skip the `slurm/` scripts entirely and
use the bare Python commands shown under each step below, or `SAMPLE_MODE`
for a fast local check (see "Sample mode" above).

---

## Execution order

Each step below prints a start banner, an `[OK]`/`[WARN]`/`[SKIP]` line for
every file it reads or writes, and an end-of-stage summary count -- the
console log always shows how far a run progressed and where its output
landed, even on failure.

### Step 0 — Obtain `bbob_fopt.csv` (required before Step 5)

`shared_consolidate_raw_data.m` needs the known optimal value for every
BBOB function/instance/dimension combination to compute residuals; these
are not derivable from the raw landscape data alone. Place this file at:

```
bbob_cornn_isa/bbob/meta/bbob_fopt.csv
```

This file ships as part of the paper's data release (see
[Citation](#citation) for the Figshare DOI) rather than being produced by
any script in this pipeline. If you are setting up a fresh environment,
obtain it from there and copy it into place before running Step 5.

**Full provenance chain (optional, not part of the required pipeline).**
`bbob_collect_meta.py` triggers a COCO observer to write per-function
`.dat` logs, and `bbob_collect_fopt.m` parses those logs into
`bbob_fopt.csv`. Running both in sequence regenerates the file from
scratch instead of obtaining it from Figshare:

```bash
python bbob_collect_meta.py    # writes .dat logs under bbob/meta/
```
```matlab
bbob_collect_fopt              % parses .dat logs -> bbob_fopt.csv
```

Both scripts respect `SAMPLE_MODE`, so this chain also works for the fast
local sample-mode check described above.

### Step 1 — Generate Sobol input grids (MATLAB, run once)

```matlab
shared_collect_input_samples
```
**Output:** `input/X_D{dim}_S100_R{sid}.csv` — 15 files in full mode, 1 in sample mode.

### Step 2 — Collect raw landscape data (cluster)

```bash
sbatch slurm/run_collect_raw_data.sh
```
**Output:** `bbob/raw/F{fid}_D{dim}_I{iid}_S100_R{sid}.csv` (5400 files) and
`cornn/raw/F_{fcn}_{arch}_S100_R{sid}.csv` (1620 files).

**Without SLURM** — each array task runs one bare Python command; a single
task looks like `TASK_ID=1 python bbob_collect_raw_data.py`. The full stage
without a scheduler (same total work, run serially):
```bash
for i in $(seq 1 1080); do TASK_ID=$i python bbob_collect_raw_data.py;  done
for i in $(seq 1 324);  do TASK_ID=$i python cornn_collect_raw_data.py; done
```

### Step 3 — Compute ELA features (cluster, after Step 2)

```bash
sbatch --dependency=afterok:<STEP2_JOBID> slurm/run_collect_pflacco.sh
```
**Output:** `bbob/ela/ELA_F{fid}_D{dim}_S100_R{sid}.csv` (360 files) and
`cornn/ela/ELA_F_{fcn}_{arch}_S100_R{sid}.csv` (324 files).

**Without SLURM** (after Step 2 has produced `bbob/raw/` and `cornn/raw/`):
```bash
for i in $(seq 1 360); do TASK_ID=$i python bbob_run_pflacco.py;  done
for i in $(seq 1 324); do TASK_ID=$i python cornn_run_pflacco.py; done
```

### Step 4 — Run optimiser performance experiments (cluster)

```bash
sbatch slurm/run_all_performance.sh
```
**Output:** `bbob/nevergrad/`, `bbob/adam/`, `cornn/nevergrad/`,
`cornn/adam/` — one CSV per (algorithm, instance, run); 129,600 + 32,400 +
38,880 + 9,720 files respectively.

**Without SLURM** — this is the full-scale dataset (the same total compute
as the cluster job, just serial, so this will take a long time on a single
machine; see "Sample mode" above for a fast local check instead):
```bash
for i in $(seq 1 4320); do TASK_ID=$i python bbob_run_nevergrad.py;  done
for i in $(seq 1 1080); do TASK_ID=$i python bbob_run_adam.py;       done
for i in $(seq 1 1296); do TASK_ID=$i python cornn_run_nevergrad.py; done
for i in $(seq 1 324);  do TASK_ID=$i python cornn_run_adam.py;      done
```

### Step 5 — Post-processing (MATLAB)

```matlab
shared_consolidate_raw_data
shared_generate_instance_space
```
**Output of `shared_consolidate_raw_data`:** `isa/BBOB_area_under_the_curve.csv`,
`isa/CORNN_area_under_the_curve.csv`, `isa/BBOB_CORNN_pflacco.csv`,
`isa/ecdf_per_algorithm.png`, `isa/target_difficulty_by_dimension.png`.

**Output of `shared_generate_instance_space`:** `isa/BBOB_CORNN_metadata.csv`,
`isa/finds_targets.csv`, `isa/rho_features_axes.csv`,
`isa/footprint_summary.csv`, `isa/train_test_distance.csv`, and all `*.png`
figures (projection, feature, footprint, and portfolio plots).

Individual sections of `shared_generate_instance_space.m` can be run
independently using the `cfg.run.*` flags set in `cornn_config.m`, each
overridable via an environment variable, e.g.
`setenv('RUN_FOOTPRINTS', '0')` to skip footprint estimation. See
[CONFIGURATION.md](CONFIGURATION.md) for the full flag list.

---

## Verification: expected outputs

| Step | Expected output | Quick check (full mode) | Quick check (sample mode) |
|---|---|---|---|
| 1 — Sobol grids | CSV files in `input/` | `ls input/ \| wc -l` → 15 (3 dims × 5 reps) | → 1 (1 dim × 1 rep) |
| 2 — BBOB raw | CSV files in `bbob/raw/` | `ls bbob/raw/ \| wc -l` → 5400 (24×15×3×5) | → 6 (2×3×1×1) |
| 3 — ELA features | CSV files in `bbob/ela/` | `ls bbob/ela/ \| wc -l` → 360 (3×5×24) | → 2 (1×1×2) |
| 4 — Performance | CSV files in `bbob/nevergrad/` | `ls bbob/nevergrad/ \| wc -l` → 129600 (4×24×15×3×30) | → 72 (4×2×3×1×3) |
| 5 — MATLAB | Files in `isa/` | `BBOB_CORNN_metadata.csv` exists | same |

Counts are (functions or dims) × (instances or archs) × (dims) × (reps or
runs), matching the loop order in each script -- see
[CONFIGURATION.md](CONFIGURATION.md) for the full breakdown, including the
matching CORNN counts.

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

A machine-readable [CITATION.cff](CITATION.cff) is also provided at the
repository root; GitHub uses this to render a "Cite this repository"
button automatically.

---

## License

PolyForm Noncommercial License 1.0.0.
Copyright (c) 2026 Mario Andres Munoz Acosta, University of Melbourne.
