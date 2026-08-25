# Configuration Guide

All tuneable parameters and path definitions live in two files that mirror
each other:

- **`cornn/config.py`** — read by all Python scripts
- **`cornn_config.m`** — read by all MATLAB scripts via `cfg = cornn_config()`

No other script contains hardcoded paths or magic numbers. To adapt the
pipeline to a different machine, benchmark configuration, or algorithm
portfolio, edit only these two files.

---

## Data root directory

The single root directory from which all subdirectories are derived.

| Setting | Default (cluster) | Default (Windows) | Override |
|---|---|---|---|
| Root data dir | `~/punim0320/bbob_cornn_isa/` | `D:/bbob_cornn_isa/` | `BBOB_CORNN_ROOT` |

**To use a different root directory:**
```bash
export BBOB_CORNN_ROOT=/path/to/your/data
```
```matlab
setenv('BBOB_CORNN_ROOT', '/path/to/your/data');
cfg = cornn_config();
```

All subdirectories (`input/`, `bbob/raw/`, `cornn/ela/`, etc.) are created
automatically under the root when any script first runs.

Individual subdirectories can also be overridden independently:

| Subdirectory | Environment variable |
|---|---|
| Sobol grids | `BBOB_CORNN_INPUT_DIR` |
| BBOB raw evaluations | `BBOB_RAW_DIR` |
| BBOB ELA features | `BBOB_ELA_DIR` |
| BBOB nevergrad runs | `BBOB_NG_DIR` |
| BBOB Adam runs | `BBOB_ADAM_DIR` |
| CORNN raw evaluations | `CORNN_RAW_DIR` |
| CORNN ELA features | `CORNN_ELA_DIR` |
| CORNN nevergrad runs | `CORNN_NG_DIR` |
| CORNN Adam runs | `CORNN_ADAM_DIR` |
| MATLAB post-processing output | `BBOB_CORNN_ISA_DIR` |

---

## Benchmark dimensions

Controls which problem dimensions are used across both suites.

| Parameter | Default | Notes |
|---|---|---|
| `DIMENSIONS` | `[41, 261, 481]` | Must match CORNN architecture sizes |

To run on a subset of dimensions (e.g. only 41D):
```python
# cornn/config.py
DIMENSIONS = [41]
```
```matlab
% cornn_config.m
cfg.dimensions = 41;
```
Note: changing dimensions requires rebuilding the COCO suite with a matching
`suite_largescale.c` and regenerating Sobol grids.

---

## Sobol input grids

| Parameter | Default | Notes |
|---|---|---|
| `N_REPLICATES` | `5` | Number of independent Sobol samples per instance |
| `SAMPLE_SIZE` | `100` | Points per dimension; total points = `SAMPLE_SIZE × D` |

Increasing `SAMPLE_SIZE` improves ELA feature stability at the cost of
compute time. The value `100 × D` follows best practice from Renau et al.
(2020). Changing either parameter requires regenerating input grids
(`shared_collect_input_samples.m`) and all downstream raw data.

---

## BBOB suite

| Parameter | Default | Notes |
|---|---|---|
| `N_BBOB_FUNCTIONS` | `24` | All 24 noiseless BBOB functions |
| `N_BBOB_INSTANCES` | `15` | Instances 1–15 per function |
| `BBOB_SUITE` | `"bbob-largescale"` | COCO suite name |

To restrict to a subset of BBOB functions, modify `BBOB_SETTINGS`:
```python
# cornn/config.py — e.g. separable functions only (f1-f5)
BBOB_SETTINGS = "function_indices:1-5 dimensions:41,261,481"
N_BBOB_FUNCTIONS = 5
```

---

## Optimiser runs

| Parameter | Default | Notes |
|---|---|---|
| `BUDGET` | `5000` | Function evaluations per run |
| `N_RUNS` | `30` | Independent runs per instance |
| `BOUNDS_LO` | `-5.0` | Search space lower bound |
| `BOUNDS_HI` | `5.0` | Search space upper bound |

---

## Algorithm portfolio

| Parameter | Default | Notes |
|---|---|---|
| `ALGORITHM_NAMES` | `["CMA", "PSO", "RandomSearch", "TwoPointsDE"]` | nevergrad algorithms |

These names must match valid nevergrad registry keys
(`ng.optimizers.registry`). To add or remove an algorithm:

```python
# cornn/config.py
ALGORITHM_NAMES = ["CMA", "PSO", "RandomSearch", "TwoPointsDE", "DE"]
```

Adam is always included as the fifth algorithm via the dedicated
`bbob_run_adam.py` / `cornn_run_adam.py` scripts. It is not part of
`ALGORITHM_NAMES` because it uses a separate finite-difference
implementation rather than nevergrad.

After changing the portfolio, update `shared_consolidate_raw_data.m` —
specifically the `algorithm_header` cell array, which must list all
algorithm names in the same order used during data collection.

---

## Performance targets

| Parameter | Default | Notes |
|---|---|---|
| `TARGET_MIN` | `-1.0` | log₁₀ of tightest target (10⁻¹) |
| `TARGET_MAX` | `1.0` | log₁₀ of loosest target (10¹) |
| `TARGET_STEP` | `0.1` | Step size in log₁₀ scale |
| `EPSILON` | `0.0` | Acceptability threshold (AUC > 0) |

The 21 targets span `[10⁻¹, 10¹]` in steps of 0.1 on a log₁₀ scale,
representing precision levels relative to the known optimum. Targets above
10 were found empirically to be trivially achieved; targets below 10⁻¹
were rarely achieved at these dimensions.

---

## Adam hyper-parameters

| Parameter | Default | Notes |
|---|---|---|
| `ADAM_LR` | `1e-3` | Learning rate |
| `ADAM_BETA1` | `0.9` | First moment decay |
| `ADAM_BETA2` | `0.999` | Second moment decay |
| `ADAM_EPS` | `1e-8` | Denominator stabiliser |
| `FD_EPS` | `1e-3` | Finite-difference step size |

Forward differences are used (`d+1` evaluations per gradient step). The
remaining budget after the last full gradient step is exhausted by
evaluating `f(x_final)` repeatedly, so all runs produce exactly `BUDGET`
rows.

---

## Task dispatch

Every runner script processes one (algorithm, function, ...) combination
per invocation by default, selected by `cornn.config.get_task_id()`:

```python
def get_task_id() -> int:
    # Reads TASK_ID first, then SLURM_ARRAY_TASK_ID, then defaults to 1.
```

`TASK_ID` is set by the SLURM dispatch scripts in `slurm/` (see
`README.md`); `SLURM_ARRAY_TASK_ID` is set automatically by a standalone
SLURM array submission. Running a script directly with neither variable
set (a plain local `python bbob_run_nevergrad.py`) processes task 1 only
-- the first combination in that script's enumeration order.

## Sample mode

Restricts all data collection to a small subset for local verification
without a cluster.

| Parameter | Default | Notes |
|---|---|---|
| `SAMPLE_MODE` | `False` | Set via `SAMPLE_MODE=1` env var |
| `SAMPLE_BBOB_FUNCTIONS` | `[1, 8]` | f1 (sphere) and f8 (Rosenbrock) |
| `SAMPLE_BBOB_INSTANCES` | `[1, 2, 3]` | First 3 instances only |
| `SAMPLE_DIMENSIONS` | `[41]` | Lowest dimension only |
| `SAMPLE_REPLICATES` | `[1]` | Single Sobol replicate |
| `SAMPLE_RUNS` | `3` | Optimiser runs per instance (vs 30 normally) |
| `SAMPLE_BUDGET` | `500` | Function evaluations per run (vs 5000 normally) |

**Mechanism.** In every runner script, `SAMPLE_MODE` does two things:
1. `cornn/config.py` restricts `DIMENSIONS`, `BBOB_SETTINGS`, `BBOB_INSTANCES`,
   `N_REPLICATES`, `N_RUNS`, and `BUDGET` to the sample values above -- this
   happens automatically for every script that imports them, with no
   per-script changes needed.
2. Each runner script normally processes exactly one (algorithm, function, ...)
   combination per invocation, selected by `TASK_ID` / `SLURM_ARRAY_TASK_ID`
   (see "Task dispatch" above). When `SAMPLE_MODE=1`, this dispatch is
   bypassed entirely and the script processes **every** combination in the
   (small) sample subset within a single invocation, so one bare command
   produces complete output.

For CORNN scripts, the sample subset is further restricted to the first
function and first architecture returned by `CORNN.get_benchmark_functions()`
/ `CORNN.get_NN_models()`, with all algorithms still exercised (to verify
the full portfolio).

```bash
export SAMPLE_MODE=1
python bbob_collect_raw_data.py    # both sample functions, ~1 min
python bbob_run_pflacco.py         # both sample functions, ~1 min
python bbob_run_nevergrad.py       # all 4 algorithms x 2 functions, 3 runs each, ~2 min
python bbob_run_adam.py            # 2 functions, 3 runs each, ~1 min
python cornn_collect_raw_data.py   # first function x first architecture, ~1 min
python cornn_run_pflacco.py        # first function x first architecture, ~1 min
python cornn_run_nevergrad.py      # all 4 algorithms, 3 runs each, ~2 min
python cornn_run_adam.py           # 3 runs, ~1 min
```

**Caveat.** `bbob_run_pflacco.py` iterates over the explicit
`BBOB_FUNCTION_IDS` / `BBOB_INSTANCE_IDS` lists from `cornn/config.py`
rather than `range(1, N+1)`, because the sample function subset `{1, 8}`
is not a contiguous range. Any future script that enumerates BBOB
functions or instances manually (not via a `cocoex.Suite` object) should
do the same, or it will silently process the wrong functions in sample
mode.

---

## Data flow between pipeline stages

```
shared_collect_input_samples.m
  └─ writes ──► input/X_D{dim}_S100_R{sid}.csv
                    │
          ┌─────────┤
          ▼         ▼
bbob_collect_raw_data.py    cornn_collect_raw_data.py
  └─ writes ──► bbob/raw/   └─ writes ──► cornn/raw/
                    │                          │
          ┌─────────┘              ┌───────────┘
          ▼                        ▼
bbob_run_pflacco.py         cornn_run_pflacco.py
  └─ writes ──► bbob/ela/   └─ writes ──► cornn/ela/

bbob_run_nevergrad.py  bbob_run_adam.py
  └─ writes ──► bbob/nevergrad/  bbob/adam/

cornn_run_nevergrad.py  cornn_run_adam.py
  └─ writes ──► cornn/nevergrad/  cornn/adam/

shared_consolidate_raw_data.m
  reads ◄── bbob/{nevergrad,adam}/   cornn/{nevergrad,adam}/
  reads ◄── bbob/ela/                cornn/ela/
  reads ◄── bbob/meta/bbob_fopt.csv
  writes ──► isa/BBOB_area_under_the_curve.csv
  writes ──► isa/CORNN_area_under_the_curve.csv
  writes ──► isa/BBOB_CORNN_pflacco.csv

shared_generate_instance_space.m
  reads ◄── isa/BBOB_area_under_the_curve.csv
  reads ◄── isa/CORNN_area_under_the_curve.csv
  reads ◄── isa/BBOB_CORNN_pflacco.csv
  writes ──► isa/BBOB_CORNN_metadata.csv
  writes ──► isa/*.png  (all figures)
```

---

## Post-processing sections

`shared_generate_instance_space.m` is divided into independently
executable sections controlled by `cfg.run.*` flags, set in
`cornn_config.m` and each overridable via an environment variable of the
same name:

| Flag | Env var override | Controls | Output files |
|---|---|---|---|
| `cfg.run.load_data`  | `RUN_LOAD_DATA`  | Join ELA features with AUC data | `BBOB_CORNN_metadata.csv` |
| `cfg.run.projection` | `RUN_PROJECTION` | PCA + t-SNE 2D embedding | (in-memory `Z`) |
| `cfg.run.features`   | `RUN_FEATURES`   | Feature clustering and correlation | (printed to console) |
| `cfg.run.models`     | `RUN_MODELS`     | KNN prediction models | (in-memory `Yhat`, `Ycv`) |
| `cfg.run.footprints` | `RUN_FOOTPRINTS` | TRACE footprint estimation | (in-memory `trace_outputs`) |
| `cfg.run.figures`    | `RUN_FIGURES`    | All scatter, violin, and footprint plots | `*.png` in `isa/` |
| `cfg.run.tables`     | `RUN_TABLES`     | `finds_targets`, `rho_features_axes`, `footprint_summary`, `train_test_distance` | `*.csv` in `isa/` |

The flags live in configuration (not hardcoded in the script) so they can
be set from the command line without editing any file, e.g.:

```matlab
setenv('RUN_FIGURES', '0');   % skip figure generation this run
shared_generate_instance_space
```

Note: `cfg.run.models` and `cfg.run.footprints` depend on
`cfg.run.projection` having been run in the same MATLAB session (they
reuse `Z` from the workspace). `cfg.run.figures` and `cfg.run.tables`
depend on all preceding sections. Every section prints an `[OK]`/`[SKIP]`
line so it is always clear from the console log which sections ran.

---

## Extending to a new benchmark suite

To add a third benchmark suite alongside BBOB and CORNN:

1. Add new subdirectories to `cornn/config.py` and `cornn_config.m`
   (e.g. `NEW_RAW_DIR`, `NEW_ELA_DIR`, `NEW_NG_DIR`).
2. Write a `new_collect_raw_data.py` and `new_run_pflacco.py` following
   the same structure as the CORNN equivalents.
3. Extend `shared_consolidate_raw_data.m` with a new consolidation block
   and AUC loop, appending results to `area_under_the_curve_new`.
4. Extend `shared_generate_instance_space.m` — add new group membership
   vectors (following the `isbo041`, `isnet1` pattern) and extend the
   `groups` and `group_names` arrays.
