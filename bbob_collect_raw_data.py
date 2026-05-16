"""
collect_raw_data_bbob.py
------------------------
Evaluate BBOB large-scale functions on the pre-generated Sobol input grids
and write one Y-file per (fid, iid, sid) triple.

SLURM: one task per problem (24 fns x 15 inst x 3 dims = 1080 tasks).
       Task index is read via cornn.config.get_task_id().
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
import pandas as pd
import cocoex

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS,
    INPUT_DIR, BBOB_RAW_DIR,
    N_REPLICATES, SAMPLE_SIZE,
    get_task_id, make_dirs,
)

make_dirs()

suite         = cocoex.Suite(BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS)
task_id       = get_task_id()
current_index = 0

for problem in suite:
    current_index += 1
    if current_index != task_id:
        continue

    dim = problem.dimension
    fid = problem.id_function
    iid = problem.id_instance

    for sid in range(1, N_REPLICATES + 1):
        y_path = BBOB_RAW_DIR / f"F{fid}_D{dim}_I{iid}_S{SAMPLE_SIZE}_R{sid}.csv"
        if y_path.is_file():
            continue

        x_path = INPUT_DIR / f"X_D{dim}_S{SAMPLE_SIZE}_R{sid}.csv"
        if not x_path.is_file():
            print(f"[WARN] Missing: {x_path}. Skipping.")
            continue

        try:
            # header=None: MATLAB writematrix produces no header row.
            # Scale from [-1, 1] to BBOB domain [-5, 5].
            x_data = pd.read_csv(x_path, header=None).to_numpy() * 5.0
        except Exception as e:
            print(f"[ERROR] Read failed '{x_path}': {e}. Skipping.")
            continue

        y_vals = []
        for i, x in enumerate(x_data):
            try:
                y_vals.append(problem(x))
            except Exception as e:
                print(f"[ERROR] Evaluation failed at row {i}: {e}")
                y_vals.append(float("nan"))

        pd.DataFrame(y_vals, columns=["Y"]).to_csv(y_path, index=False)
        print(f"[OK] {y_path}")

    break  # one problem per task
