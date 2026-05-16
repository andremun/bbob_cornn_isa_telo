"""
run_bbob_nevergrad.py
---------------------
Run nevergrad optimisers on the BBOB large-scale suite.

SLURM: one task per (algorithm, problem) pair
       (4 algs x 24 fns x 15 inst x 3 dims = 4320 tasks).
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
import nevergrad as ng
import cocoex

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS,
    BBOB_NG_DIR, ALGORITHM_NAMES,
    BUDGET, N_RUNS, BOUNDS_LO, BOUNDS_HI,
    get_task_id, make_dirs,
)

make_dirs()

suite         = cocoex.Suite(BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS)
task_id       = get_task_id()
current_index = 0

for name in ALGORITHM_NAMES:
    for problem in suite:
        current_index += 1
        if current_index != task_id:
            continue

        for run in range(N_RUNS):
            out_path = BBOB_NG_DIR / f"{problem.id}_{name}_R{run}.csv"
            if out_path.is_file():
                continue

            y_hist = []

            def myproblem(x):
                y = problem(x)
                y_hist.append(y)
                return y

            # Fresh parametrization per run ensures independence between runs.
            parametrization = ng.p.Array(
                shape=(problem.dimension,)
            ).set_bounds(lower=BOUNDS_LO, upper=BOUNDS_HI)
            optim = ng.optimizers.registry[name](
                parametrization=parametrization, budget=BUDGET
            )
            optim.minimize(myproblem)
            pd.DataFrame(y_hist, columns=["Fval"]).to_csv(out_path, index=False)
            print(out_path)

        break  # one (name, problem) pair per task
