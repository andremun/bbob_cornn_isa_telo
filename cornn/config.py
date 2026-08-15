"""
cornn/config.py
---------------
Central configuration for the BBOB-CORNN instance space analysis project.
All tuneable constants and path definitions live here; no magic numbers or
hardcoded paths elsewhere.

Machine configurations
----------------------
                Spartan HPC (cluster)       Windows (local)
RAM             varies                      64 GB
Data path       ~/punim0320/                D:/bbob_cornn_isa/
                bbob_cornn_isa/

Two configurations, detected automatically at import time:
  1. IS_CLUSTER  (Linux, no GPU, SLURM present)
  2. IS_WINDOWS  (local development / post-processing)

Override any path via environment variables (see Path block below).
"""

# Copyright (c) 2026 Mario Andres Munoz Acosta
# School of Computing and Information Systems
# The University of Melbourne
#
# Date: May 2026
#
# This software is licensed under the PolyForm Noncommercial License 1.0.0.
# You may use, copy, modify, and distribute this software for any
# non-commercial purpose. Commercial use is prohibited.
# Full license text: https://polyformproject.org/licenses/noncommercial/1.0.0
#
# Required Notice: Copyright (c) 2026 Mario Andres Munoz Acosta
#                  School of Computing and Information Systems
#                  The University of Melbourne

import os
import platform
from pathlib import Path

# =============================================================================
# PLATFORM DETECTION
# =============================================================================

IS_WINDOWS = platform.system() == "Windows"
IS_CLUSTER = (
    platform.system() == "Linux"
    and os.environ.get("SLURM_JOB_ID") is not None
)
IS_LINUX   = platform.system() == "Linux"

# =============================================================================
# ROOT DATA DIRECTORY
# =============================================================================

def _default_root() -> Path:
    if IS_WINDOWS:
        return Path("D:/bbob_cornn_isa")
    else:
        # Cluster or any other Linux (local VM, etc.)
        return Path.home() / "punim0320" / "bbob_cornn_isa"

ROOT_DATA_DIR: Path = Path(
    os.environ.get("BBOB_CORNN_ROOT", str(_default_root()))
)

# =============================================================================
# SUBDIRECTORY LAYOUT  (single source of truth)
# =============================================================================

INPUT_DIR       = ROOT_DATA_DIR / "input"           # Sobol grids

BBOB_RAW_DIR    = ROOT_DATA_DIR / "bbob"  / "raw"   # landscape evaluations
BBOB_ELA_DIR    = ROOT_DATA_DIR / "bbob"  / "ela"   # pflacco features
BBOB_NG_DIR     = ROOT_DATA_DIR / "bbob"  / "nevergrad"
BBOB_ADAM_DIR   = ROOT_DATA_DIR / "bbob"  / "adam"

CORNN_RAW_DIR   = ROOT_DATA_DIR / "cornn" / "raw"
CORNN_ELA_DIR   = ROOT_DATA_DIR / "cornn" / "ela"
CORNN_NG_DIR    = ROOT_DATA_DIR / "cornn" / "nevergrad"
CORNN_ADAM_DIR  = ROOT_DATA_DIR / "cornn" / "adam"

ISA_DIR         = ROOT_DATA_DIR / "isa"             # MATLAB post-processing output

# Allow individual directory overrides via environment variables
INPUT_DIR      = Path(os.environ.get("BBOB_CORNN_INPUT_DIR",    str(INPUT_DIR)))
BBOB_RAW_DIR   = Path(os.environ.get("BBOB_RAW_DIR",            str(BBOB_RAW_DIR)))
BBOB_ELA_DIR   = Path(os.environ.get("BBOB_ELA_DIR",            str(BBOB_ELA_DIR)))
BBOB_NG_DIR    = Path(os.environ.get("BBOB_NG_DIR",             str(BBOB_NG_DIR)))
BBOB_ADAM_DIR  = Path(os.environ.get("BBOB_ADAM_DIR",           str(BBOB_ADAM_DIR)))
CORNN_RAW_DIR  = Path(os.environ.get("CORNN_RAW_DIR",           str(CORNN_RAW_DIR)))
CORNN_ELA_DIR  = Path(os.environ.get("CORNN_ELA_DIR",           str(CORNN_ELA_DIR)))
CORNN_NG_DIR   = Path(os.environ.get("CORNN_NG_DIR",            str(CORNN_NG_DIR)))
CORNN_ADAM_DIR = Path(os.environ.get("CORNN_ADAM_DIR",          str(CORNN_ADAM_DIR)))
ISA_DIR        = Path(os.environ.get("BBOB_CORNN_ISA_DIR",      str(ISA_DIR)))


def make_dirs() -> None:
    """Create all output directories. Call once at the start of any script."""
    for d in [INPUT_DIR, BBOB_RAW_DIR, BBOB_ELA_DIR, BBOB_NG_DIR, BBOB_ADAM_DIR,
              CORNN_RAW_DIR, CORNN_ELA_DIR, CORNN_NG_DIR, CORNN_ADAM_DIR, ISA_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# EXPERIMENT CONSTANTS  (paper-authoritative values)
# =============================================================================

# --- Problem dimensions (BBOB large-scale and CORNN architectures) -----------
DIMENSIONS      = [41, 261, 481]

# --- Sobol grid ---------------------------------------------------------------
N_REPLICATES    = 5
SAMPLE_SIZE     = 100           # points per dimension (total = SAMPLE_SIZE * D)

# --- BBOB suite ---------------------------------------------------------------
BBOB_SUITE      = "bbob-largescale"
BBOB_INSTANCES  = "instances:1-15"
BBOB_SETTINGS   = "function_indices:1-24 dimensions:41,261,481"
N_BBOB_FUNCTIONS  = 24
N_BBOB_INSTANCES  = 15

# --- Optimiser runs -----------------------------------------------------------
BUDGET          = 5_000
N_RUNS          = 30
BOUNDS_LO       = -5.0
BOUNDS_HI       =  5.0

# --- Algorithm portfolio ------------------------------------------------------
ALGORITHM_NAMES = ["CMA", "PSO", "RandomSearch", "TwoPointsDE"]

# --- Performance targets (fixed-target, log10 scale) -------------------------
TARGET_MIN      = -1.0
TARGET_MAX      =  1.0
TARGET_STEP     =  0.1
# targets = [10^-1.0, 10^-0.9, ..., 10^0.9, 10^1.0]  (21 values)
import numpy as np
TARGETS = np.round(
    np.arange(TARGET_MIN, TARGET_MAX + TARGET_STEP / 2, TARGET_STEP), 1
).tolist()
N_TARGETS       = len(TARGETS)  # 21

# --- Acceptability threshold --------------------------------------------------
EPSILON         = 0.0

# --- Adam hyper-parameters (finite-difference) --------------------------------
ADAM_LR         = 1e-3
ADAM_BETA1      = 0.9
ADAM_BETA2      = 0.999
ADAM_EPS        = 1e-8      # denominator stabiliser
FD_EPS          = 1e-3      # finite-difference step size

# =============================================================================
# SAMPLE MODE  (local reproducibility and badge verification)
# =============================================================================
# Set SAMPLE_MODE=1 to restrict data collection to a small subset that runs
# the full pipeline locally in a few minutes without a cluster.
#
# Sample subset:
#   BBOB:  functions 1 and 8, instances 1-3, dimension 41 only, 1 replicate
#   CORNN: first function x first architecture only, dimension 41, 1 replicate
#
# Usage:
#   SAMPLE_MODE=1 python bbob_collect_raw_data.py

SAMPLE_MODE = os.environ.get("SAMPLE_MODE", "0") == "1"

SAMPLE_BBOB_FUNCTIONS  = [1, 8]
SAMPLE_BBOB_INSTANCES  = [1, 2, 3]
SAMPLE_DIMENSIONS      = [41]
SAMPLE_REPLICATES      = [1]

# =============================================================================
# TASK ID  (SLURM array or local fallback)
# =============================================================================

def get_task_id() -> int:
    """
    Return the current task index.
    Reads TASK_ID first (set by the dispatcher script), then falls back to
    SLURM_ARRAY_TASK_ID (standalone array submission), then defaults to 1
    for local runs.
    """
    for key in ("TASK_ID", "SLURM_ARRAY_TASK_ID"):
        val = os.environ.get(key)
        if val is not None:
            try:
                return int(val)
            except ValueError:
                pass
    return 1
