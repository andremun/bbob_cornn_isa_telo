# BBOB-CORNN Instance Space Analysis

Code for the instance space analysis (ISA) comparing the CORNN neural network
training benchmark suite with the BBOB large-scale noiseless benchmark suite.

---

## Repository structure

```
bbob_cornn_isa/
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
├── tsne_d.m                         # t-SNE on a precomputed distance matrix (van der Maaten)
├── tsne_p.m                         # t-SNE on a precomputed affinity matrix (van der Maaten)
├── d2p.m                            # Distance -> Gaussian affinity conversion (van der Maaten)
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

The standard `cocoex` package (installed with `pip install coco-experiment`)
does not support dimensions {41, 261, 481}. This repository needs a source
build with a modified `suite_largescale.c`, provided in the repository
root.

**Build prerequisites.** The build needs a C compiler. Per upstream's
`env.yaml`, it also needs `pip>=22.3`, `numpy>=1.24`, `cython>=0.29`, and
`hatchling>=1.26.3` before you run `pip install .`. Build isolation does
not always install these on its own:

```bash
pip install "numpy>=1.24" "cython>=0.29" "hatchling>=1.26.3"
```

**Option A — exact reproduction (recommended).** The paper's results used
`cocoex` version `2.6.100-dev34+ga1bd588d`. This git-describe string
identifies commit
[`a1bd588dd06a27e248aaa976c018bae759a296a0`](https://github.com/numbbo/coco/commit/a1bd588dd06a27e248aaa976c018bae759a296a0)
in the old `numbbo/coco` monorepo, not in `numbbo/coco-experiment`. These
two repositories have separate histories, and `numbbo/coco-experiment`
does not contain this commit. `git checkout a1bd588d` fails there with an
unknown revision error. `numbbo/coco` is archived and read-only, but you
can still clone it. At this commit, it already uses the `scripts/fabricate`
build shown below, so this historical snapshot still builds today:

```bash
git clone https://github.com/numbbo/coco.git
cd coco
git checkout a1bd588dd06a27e248aaa976c018bae759a296a0

# Replace suite_largescale.c with the version in this repository's root
cp ../suite_largescale.c code-experiments/src/suite_largescale.c

# Bundle the modified C sources into the language-specific build folders
python scripts/fabricate

# Build and install the Python bindings
cd code-experiments/build/python
pip install .
```

**Option B — general install.** If you cannot clone the archived
`numbbo/coco` repository, use the actively maintained
[`numbbo/coco-experiment`](https://github.com/numbbo/coco-experiment)
repository instead. It uses the same build workflow, laid out at the
repository root instead of under `code-experiments/`. This path does not
reproduce the exact pinned version above, since that commit does not
exist there. The BBOB problem definitions stay the same regardless of
this version difference:

```bash
git clone https://github.com/numbbo/coco-experiment.git
cd coco-experiment

# Locate the current path to suite_largescale.c under src/ and replace it
# with the version in this repository's root, e.g.:
find . -name suite_largescale.c
cp ../suite_largescale.c <path found above>

python scripts/fabricate
cd build/python
pip install .
```

To check the version of an existing installation, run this command.
`pip show coco-experiment` may not resolve if the package was installed
under a different distribution name:

```bash
python -c "import cocoex; print(cocoex.__version__)"
```

**Compilation troubleshooting** (from upstream `DEVELOPMENT.md`):
- On macOS with ARM, use `arch -arm64 pip install .`.
- On older systems, use `CFLAGS="-std=c99" pip install .`.

### CORNN (neural network training benchmark)

CORNN (Malan and Cleghorn, 2022, *A Continuous Optimisation Benchmark Suite
from Neural Network Regression*, LNCS vol. 13398 / arXiv:2109.05606) is
a neural network training benchmark suite. It is **not** on PyPI, so you
must clone and install it in editable mode.

**Clone it outside this repository**, not into a `CORNN` subfolder here.
This repository already has a `cornn/` package directory. On a
case-insensitive filesystem, the default on Windows and macOS, `CORNN`
and `cornn` collide, and the clone fails:

```bash
cd ..                      # or any directory outside this repository
git clone https://github.com/CWCleghornAI/CORNN.git CORNN
```

**Do not run `pip install -e .` or `pip install -r requirements.txt` inside
the cloned repository.** Both commands are broken and unnecessary:
- CORNN's `setup.py` declares `packages=['CORNN']`, but the actual code
  lives under `lib/`, not a `CORNN/` package directory. `pip install -e .`
  fails immediately with the error `package directory 'CORNN' does not
  exist`.
- CORNN's `requirements.txt` pins `torch==1.9.0`, `numpy==1.20.1`, and
  `pandas==1.2.3`. These versions no longer have wheels for Python 3.10 on
  common platforms, and they conflict with the newer versions this
  repository's own `requirements.txt` already installs.

Every script in this repository imports CORNN with `import lib.CORNN`, a
plain filesystem-relative import. This needs no installation step at all.
This repository's own `requirements.txt` (see "Python environment" below)
already provides compatible `torch`, `numpy`, and `pandas` versions. You
need nothing further inside the CORNN clone.

The paper's methodology reports CORNN package v0.9. Check out the
matching tag or commit if the repository provides one.

**Why the directory structure matters.** `CORNN/lib/CORNN.py` uses a
relative import, `import lib.Benchmark_Functions_2D_Definition`, that only
resolves when the current working directory is the cloned CORNN root.
This is why every CORNN-related script and SLURM job in this repository
must run from inside that same directory. `CORNN_REPO_DIR` (see "Running
on a different machine or cluster" below) must point at the cloned CORNN
root itself, not at a venv folder. In practice, copy or symlink this
repository's `*.py` scripts, `cornn/` package, and `cornn_config.m` into
the CORNN root, alongside its own `lib/` directory. This comes from
CORNN's own import structure, not from a choice made in this repository.

### MATLAB toolboxes

This repository bundles the following MATILDA toolbox scripts directly:
`TRACE.m`, `scriptfcn.m`, `KNNRegressor.m`. It needs no separate MATILDA
installation.

The main PCA and t-SNE projection in `shared_generate_instance_space.m`
uses MATLAB's built-in `tsne` (Statistics and Machine Learning Toolbox,
R2017b or later). The feature-clustering step needs t-SNE on a
*precomputed* pairwise distance matrix instead. That call uses
[Laurens van der Maaten's original t-SNE implementation](https://lvdmaaten.github.io/tsne/):
`tsne_d`, `tsne_p`, and `d2p`, bundled directly in this repository, not
MATLAB's own `tsne`. These are different implementations, not
interchangeable substitutes. MATLAB's `tsne` does not reliably support a
`'Distance','precomputed'` input across releases. As of R2026a,
`'precomputed'` is not a supported `Distance` value at all, and even
where it does run, it does not reproduce `tsne_d`'s output. Use `tsne_d`
for that call.

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

Sample mode runs the full pipeline locally, without a cluster. Use it to
check that each step produces the expected output:

```bash
export SAMPLE_MODE=1
```

Sample mode restricts the scripts to 2 BBOB functions (f1, f8), 3
instances, dimension 41, and 1 replicate. It also restricts CORNN to 1
function and 1 architecture, with 3 runs and an evaluation budget of 500.
Sample mode bypasses task dispatch entirely. Each bare `python
<script>.py` command below then processes the whole sample subset in one
go. This needs no SLURM, no cluster, and no environment variable beyond
`SAMPLE_MODE=1`:

```matlab
setenv('SAMPLE_MODE', '1');
shared_collect_input_samples       % Step 1 -- must run first, writes input/
```

```bash
export SAMPLE_MODE=1

python bbob_collect_raw_data.py    # 2 fns x 3 instances,               ~1 min
python bbob_run_pflacco.py         # 2 fns x 3 instances,               ~1 min
python bbob_run_nevergrad.py       # 4 algs x 2 fns x 3 instances x 3 runs, ~2 min
python bbob_run_adam.py            # 2 fns x 3 instances x 3 runs,      ~1 min
```

Run the four `cornn_*.py` commands below with the CORNN repository
(cloned in Installation above) as the current working directory. CORNN's
own relative import only resolves from there. First copy or symlink this
repository's `cornn_*.py` scripts, `cornn/` package, and `cornn_config.m`
into the CORNN root. Then run:

```bash
export SAMPLE_MODE=1

python cornn_collect_raw_data.py   # first fn x first arch,             ~1 min
python cornn_run_pflacco.py        # first fn x first arch,             ~1 min
python cornn_run_nevergrad.py      # 4 algs x 3 runs,                   ~2 min
python cornn_run_adam.py           # 3 runs,                            ~1 min
```

The full pipeline completes in about 10 minutes on a standard laptop. It
needs no venv or repository directory layout, unlike the layout the
SLURM scripts assume (see "Running on a different machine or cluster"
below). See [CONFIGURATION.md](CONFIGURATION.md) for what each command
restricts and produces.

**`bbob/meta/bbob_fopt.csv` must exist before the MATLAB step** (see
"Step 0" below). If you test sample mode without the Figshare data
release, this file does not yet exist. Generate it locally instead. Both
scripts below respect `SAMPLE_MODE`:

```bash
export SAMPLE_MODE=1
python bbob_collect_meta.py        # writes .dat logs under bbob/meta/obs_logs/
```
```matlab
setenv('SAMPLE_MODE', '1');
bbob_collect_fopt                  % parses .dat logs -> bbob/meta/bbob_fopt.csv
```

Then run MATLAB post-processing in sample mode:
```matlab
setenv('SAMPLE_MODE', '1');
shared_consolidate_raw_data
shared_generate_instance_space
```

Figshare provides intermediate data for dimension 41: Sobol grids, ELA
features, and AUC tables (see the paper for the DOI). This lets you skip
Steps 1-4, including `bbob_fopt.csv` above, for post-processing
verification alone.

---

## Running on a different machine or cluster

The SLURM scripts in `slurm/` assume one default directory layout: a
Python venv at `~/venvs/CORNN/`, with this repository's scripts copied
into `~/venvs/CORNN/CORNN/`. This layout lets `import lib.CORNN` resolve
(see `CONFIGURATION.md`). You can override both paths without editing
any script:

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
BBOB function, instance, and dimension combination, to compute residuals.
The raw landscape data alone does not give you this value. Place this
file at:

```
bbob_cornn_isa/bbob/meta/bbob_fopt.csv
```

This file ships with the paper's data release (see
[Citation](#citation) for the Figshare DOI). No script in this pipeline
produces it. If you set up a fresh environment, get it from there and
copy it into place before you run Step 5.

**No Figshare access, for example on a first local sample-mode check?**
Regenerate the file from scratch instead. `bbob_collect_meta.py` triggers
a COCO observer that writes per-function `.dat` logs under
`bbob/meta/obs_logs/`. `bbob_collect_fopt.m` then parses those logs into
`bbob/meta/bbob_fopt.csv`. If you skip Figshare, this step is required,
not optional. Run both, in order, before Step 5:

```bash
python bbob_collect_meta.py    # writes .dat logs under bbob/meta/obs_logs/
```
```matlab
bbob_collect_fopt              % parses .dat logs -> bbob/meta/bbob_fopt.csv
```

Both scripts respect `SAMPLE_MODE`, so this chain also works for the fast
local sample-mode check described above (see "Sample mode" above for the
exact commands).

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
`cornn/ela/ELA_F_{fcn}_{arch}_S100_R{sid}.csv` (1620 files — one per
(fcn, arch) task **times 5 replicates**, since `cornn_run_pflacco.py`
writes a separate file per `sid` inside each task; this differs from the
BBOB ELA case, where each of the 360 tasks already corresponds to one
`(dim, sid, fid)` triple and writes exactly one file).

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
| 2 — CORNN raw | CSV files in `cornn/raw/` | `ls cornn/raw/ \| wc -l` → 1620 (54×6×5) | → 1 (1×1×1) |
| 3 — BBOB ELA | CSV files in `bbob/ela/` | `ls bbob/ela/ \| wc -l` → 360 (3×5×24) | → 2 (1×1×2) |
| 3 — CORNN ELA | CSV files in `cornn/ela/` | `ls cornn/ela/ \| wc -l` → 1620 (54×6×5 — one file per replicate, not per task; see Step 3 above) | → 1 (1×1×1) |
| 4 — BBOB performance | CSV files in `bbob/nevergrad/` | `ls bbob/nevergrad/ \| wc -l` → 129600 (4×24×15×3×30) | → 72 (4×2×3×1×3) |
| 4 — CORNN performance | CSV files in `cornn/nevergrad/` | `ls cornn/nevergrad/ \| wc -l` → 38880 (4×54×6×30) | → 12 (4×1×1×3) |
| 5 — MATLAB | Files in `isa/` | `BBOB_CORNN_metadata.csv` exists | same |

Counts are (functions or dims) × (instances or archs) × (dims) × (reps or
runs), matching the loop order in each script. `bbob/adam/` and
`cornn/adam/` are not separately checked here (same loop structure as
their `nevergrad` counterparts, one row per run) — see Step 4's own
Output line above for their full-mode counts. See
[CONFIGURATION.md](CONFIGURATION.md) for the sample-mode parameter
values (which functions/instances/architectures are selected, etc.)
behind these counts.

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
