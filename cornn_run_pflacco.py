"""
run_cornn_pflacco.py
--------------------
Compute ELA/pflacco features for all CORNN instances.

SLURM: one task per (fcn, arch) pair (54 fns x 6 archs = 324 tasks).
       sid (5 replicates) is an inner loop within each task.

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
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pflacco.classical_ela_features import (
    calculate_ela_distribution, calculate_ela_level, calculate_dispersion,
    calculate_information_content, calculate_nbc, calculate_pca,
)
try:
    from pflacco.classical_ela_features import calculate_ela_meta
    HAS_ELA_META = True
except ImportError:
    HAS_ELA_META = False
from pflacco.misc_features import calculate_fitness_distance_correlation
import lib.CORNN as CORNN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    INPUT_DIR, CORNN_RAW_DIR, CORNN_ELA_DIR,
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
    for nn_architecture in neural_network_dictionary:
        current_index += 1
        if not SAMPLE_MODE and current_index != task_id:
            continue

        # Derive dimension from the architecture without instantiating
        nn_arch  = neural_network_dictionary[nn_architecture]()
        bench    = CORNN.NN_Benchmark(
            *CORNN.get_scaled_function_data(function_dictionary[fcn]), nn_arch
        )
        dim      = bench.get_weight_count()
        fcn_tag  = fcn.replace(" ", "_")
        arch_tag = nn_architecture.replace(" ", "_")

        for sid in range(1, N_REPLICATES + 1):
            x_path = INPUT_DIR / f"X_D{dim}_S{SAMPLE_SIZE}_R{sid}.csv"
            if not x_path.is_file():
                print(f"[WARN] Missing: {x_path}. Skipping.")
                continue
            try:
                # header=None: MATLAB writematrix produces no header row.
                x_np = pd.read_csv(x_path, header=None).to_numpy()
            except Exception as e:
                print(f"[ERROR] Read X failed '{x_path}': {e}. Skipping.")
                continue

            y_path = CORNN_RAW_DIR / f"F_{fcn_tag}_{arch_tag}_S{SAMPLE_SIZE}_R{sid}.csv"
            if not y_path.is_file():
                print(f"[WARN] Missing Y: {y_path}. Skipping.")
                continue
            try:
                y_data = pd.read_csv(y_path)
                y_data[["training_loss"]] = StandardScaler().fit_transform(
                    y_data[["training_loss"]]
                )
                y_data[["testing_loss"]] = StandardScaler().fit_transform(
                    y_data[["testing_loss"]]
                )
            except Exception as e:
                print(f"[ERROR] Y processing failed '{y_path}': {e}. Skipping.")
                continue

            records = []
            for loss_type in ["training_loss", "testing_loss"]:
                y_np = y_data[loss_type].to_numpy()
                rec  = {
                    "function": f"F_{fcn_tag}_{arch_tag}_{loss_type}_S{SAMPLE_SIZE}_R{sid}"
                }
                for name, fn, kwargs in [
                    ("ela_meta",  calculate_ela_meta if (HAS_ELA_META and dim == 41) else None, {}),
                    ("ela_distr", calculate_ela_distribution, {}),
                    ("ela_level", calculate_ela_level, {}),
                    ("disp",      calculate_dispersion, {}),
                    ("ic",        calculate_information_content, {"seed": 0}),
                    ("nbc",       calculate_nbc, {}),
                    ("pca",       calculate_pca, {}),
                    ("fdc",       calculate_fitness_distance_correlation, {}),
                ]:
                    if fn is None:
                        continue
                    try:
                        rec.update(fn(x_np, y_np, **kwargs))
                    except Exception as e:
                        print(f"[ERROR] {name} failed {fcn_tag}_{arch_tag}_{loss_type}_R{sid}: {e}")
                records.append(rec)

            out_path = CORNN_ELA_DIR / (
                f"ELA_F_{fcn_tag}_{arch_tag}_S{SAMPLE_SIZE}_R{sid}.csv"
            )
            pd.DataFrame(records).to_csv(out_path, index=False)
            print(f"[OK] Wrote {out_path}")

        if not SAMPLE_MODE:
            break
    else:
        continue
    if not SAMPLE_MODE:
        break
