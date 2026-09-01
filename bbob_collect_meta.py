"""
bbob_collect_meta.py
---------------------
Trigger a COCO observer to write per-(function, dimension) .dat log files
containing the known optimal value (fopt) for every BBOB instance.

This script does NOT compute or write bbob_fopt.csv itself -- it only
evaluates each problem once to make COCO's observer flush a .dat log to
disk. Extraction of fopt from those logs into a single CSV is done by
bbob_collect_fopt.m (MATLAB), which expects the observer's default
per-function output layout:

    {BBOB_META_DIR}/data_f{fid}/bbobexp_f{fid}_DIM{dim}.dat

Together, this script and bbob_collect_fopt.m form the full provenance
chain for bbob_fopt.csv. Most users should obtain bbob_fopt.csv directly
from the paper's Figshare data release instead of running this chain --
see README.md, "Step 0".

Runs in a single serial invocation (no SLURM/TASK_ID dispatch): a single
observed evaluation per problem is cheap, so even the full 1080-problem
suite completes quickly.
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

import os, sys

try:
    import cocoex
except Exception as e:
    raise RuntimeError(
        "Failed to import cocoex (COCO/BBOB interface). "
        "See README.md, Installation > BBOB (custom build)."
    ) from e

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS,
    BBOB_META_DIR, SAMPLE_MODE, make_dirs,
)

make_dirs()

suite    = cocoex.Suite(BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS)
observer = cocoex.Observer(BBOB_SUITE, f"result_folder:{BBOB_META_DIR}")

mode_msg = "[INFO] SAMPLE_MODE active: only sample functions/instances/dims logged" \
    if SAMPLE_MODE else "[INFO] Full-scale run: all functions/instances/dims logged"
print(mode_msg)
print(f"[INFO] Writing .dat logs under: {BBOB_META_DIR}")

n_ok = 0
n_failed = 0
current_fid = None

for problem in suite:
    fid = problem.id_function
    if fid != current_fid:
        if current_fid is not None:
            print(f"[OK] Function f{current_fid}: done")
        current_fid = fid

    try:
        problem_obs = problem.observe_with(observer)
        x0 = problem_obs.initial_solution
        problem_obs(x0)
        problem_obs.free()
        n_ok += 1
    except Exception as e:
        print(f"[ERROR] Logging failed for {problem.id}: {e}")
        n_failed += 1

if current_fid is not None:
    print(f"[OK] Function f{current_fid}: done")

print(f"\n[OK] Observer logging complete: {n_ok} problems logged, {n_failed} failed")
print(f"[OK] Next step: run bbob_collect_fopt.m to extract bbob_fopt.csv from these logs")
