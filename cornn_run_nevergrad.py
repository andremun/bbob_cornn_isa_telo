"""
run_cornn_nevergrad.py
----------------------
Run nevergrad optimisers on the CORNN benchmark suite.

SLURM: one task per (algorithm, fcn, arch) triple
       (4 algs x 54 fns x 6 archs = 1296 tasks).

SAMPLE_MODE: restricts to the first function x first architecture, then
ignores task dispatch and processes every algorithm against that single
(fcn, arch) pair in a single invocation.
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
import lib.CORNN as CORNN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    CORNN_NG_DIR, ALGORITHM_NAMES,
    BUDGET, N_RUNS, BOUNDS_LO, BOUNDS_HI,
    SAMPLE_MODE, get_task_id, make_dirs,
)

make_dirs()

function_dictionary        = CORNN.get_benchmark_functions()
neural_network_dictionary  = CORNN.get_NN_models()

if SAMPLE_MODE:
    first_fcn  = next(iter(function_dictionary))
    first_arch = next(iter(neural_network_dictionary))
    function_dictionary       = {first_fcn: function_dictionary[first_fcn]}
    neural_network_dictionary = {first_arch: neural_network_dictionary[first_arch]}
    print(f"[INFO] SAMPLE_MODE active: restricted to function={first_fcn!r}, "
          f"architecture={first_arch!r}; processing all algorithms")

task_id                    = get_task_id()
current_index              = 0

for name in ALGORITHM_NAMES:
    for fcn in function_dictionary:
        training_data, test_data = CORNN.get_scaled_function_data(function_dictionary[fcn])

        for nn_architecture in neural_network_dictionary:
            current_index += 1
            if not SAMPLE_MODE and current_index != task_id:
                continue

            nn_arch  = neural_network_dictionary[nn_architecture]()
            bench    = CORNN.NN_Benchmark(training_data, test_data, nn_arch)
            fcn_tag  = fcn.replace(" ", "_")
            arch_tag = nn_architecture.replace(" ", "_")

            for run in range(N_RUNS):
                out_path = CORNN_NG_DIR / f"F_{fcn_tag}_{arch_tag}_{name}_R{run}.csv"
                if out_path.is_file():
                    print(f"[SKIP] {out_path} already exists")
                    continue

                y_hist_train, y_hist_test = [], []

                def myproblem_train(x):
                    y_train = bench.training_set_evaluation(x)
                    y_test  = bench.testing_set_evaluation(x)
                    y_hist_train.append(y_train)
                    y_hist_test.append(y_test)
                    return y_train

                # Fresh parametrization per run ensures independence.
                parametrization = ng.p.Array(
                    shape=(bench.get_weight_count(),)
                ).set_bounds(lower=BOUNDS_LO, upper=BOUNDS_HI)
                optim = ng.optimizers.registry[name](
                    parametrization=parametrization, budget=BUDGET
                )
                optim.minimize(myproblem_train)
                pd.DataFrame({
                    "training_loss": y_hist_train,
                    "testing_loss":  y_hist_test,
                }).to_csv(out_path, index=False)
                print(f"[OK] Wrote {out_path}")

            if not SAMPLE_MODE:
                break  # one (name, fcn, arch) per task
        else:
            continue
        break
    else:
        continue
    if not SAMPLE_MODE:
        break
