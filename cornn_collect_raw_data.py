"""
collect_raw_data_cornn.py
-------------------------
Evaluate CORNN loss landscapes on the pre-generated Sobol input grids.
Writes one Y-file per (fcn, nn_architecture, sid) triple.

SLURM: one task per (fcn, arch) pair (54 fns x 6 archs = 324 tasks).
       Task index is read via cornn.config.get_task_id().

SAMPLE_MODE: restricts to the first function x first architecture, then
ignores task dispatch and processes it in a single invocation.
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
import lib.CORNN as CORNN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    INPUT_DIR, CORNN_RAW_DIR,
    N_REPLICATES, SAMPLE_SIZE,
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
          f"architecture={first_arch!r}")

task_id                    = get_task_id()
current_index              = 0

for fcn in function_dictionary:
    training_data, test_data = CORNN.get_scaled_function_data(function_dictionary[fcn])

    for nn_architecture in neural_network_dictionary:
        current_index += 1
        if not SAMPLE_MODE and current_index != task_id:
            continue

        nn_arch  = neural_network_dictionary[nn_architecture]()
        bench    = CORNN.NN_Benchmark(training_data, test_data, nn_arch)
        dim      = bench.get_weight_count()

        fcn_tag  = fcn.replace(" ", "_")
        arch_tag = nn_architecture.replace(" ", "_")

        for sid in range(1, N_REPLICATES + 1):
            y_path = CORNN_RAW_DIR / f"F_{fcn_tag}_{arch_tag}_S{SAMPLE_SIZE}_R{sid}.csv"
            if y_path.is_file():
                print(f"[SKIP] {y_path} already exists")
                continue

            x_path = INPUT_DIR / f"X_D{dim}_S{SAMPLE_SIZE}_R{sid}.csv"
            if not x_path.is_file():
                print(f"[WARN] Missing: {x_path}. Skipping.")
                continue

            try:
                # header=None: MATLAB writematrix produces no header row.
                # Scale from [-1, 1] to CORNN domain [-5, 5].
                x_data = pd.read_csv(x_path, header=None).to_numpy() * 5.0
            except Exception as e:
                print(f"[ERROR] Read failed '{x_path}': {e}. Skipping.")
                continue

            tr_loss, te_loss = [], []
            for i, x in enumerate(x_data):
                try:
                    tr_loss.append(bench.training_set_evaluation(x))
                except Exception as e:
                    print(f"[ERROR] training eval failed row {i}: {e}")
                    tr_loss.append(float("nan"))
                try:
                    te_loss.append(bench.testing_set_evaluation(x))
                except Exception as e:
                    print(f"[ERROR] testing eval failed row {i}: {e}")
                    te_loss.append(float("nan"))

            pd.DataFrame({"training_loss": tr_loss, "testing_loss": te_loss}).to_csv(
                y_path, index=False
            )
            print(f"[OK] Wrote {y_path}")

        if not SAMPLE_MODE:
            break  # one (fcn, arch) pair per task
    else:
        continue
    if not SAMPLE_MODE:
        break
