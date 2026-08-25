"""
run_bbob_pflacco.py
-------------------
Compute ELA/pflacco features for all BBOB large-scale instances.

SLURM: one task per (dim, sid, fid) triple (3 x 5 x 24 = 360 tasks).
       iid (15 instances) is an inner loop within each task.

SAMPLE_MODE: ignores task dispatch and processes every (dim, sid, fid)
combination in the small sample suite in a single invocation. Iterates
over the explicit BBOB_FUNCTION_IDS / BBOB_INSTANCE_IDS lists rather than
range(1, N+1), since the sample function subset {1, 8} is not a
contiguous range.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    DIMENSIONS, BBOB_FUNCTION_IDS, BBOB_INSTANCE_IDS,
    INPUT_DIR, BBOB_RAW_DIR, BBOB_ELA_DIR,
    N_REPLICATES, SAMPLE_SIZE,
    SAMPLE_MODE, get_task_id, make_dirs,
)

make_dirs()

task_id       = get_task_id()
current_index = 0

if SAMPLE_MODE:
    print(f"[INFO] SAMPLE_MODE active: processing all combinations in this "
          f"invocation, ignoring TASK_ID={task_id}")
    print(f"[INFO] Function IDs: {BBOB_FUNCTION_IDS} | Instance IDs: {BBOB_INSTANCE_IDS}")

for dim in DIMENSIONS:
    for sid in range(1, N_REPLICATES + 1):
        for fid in BBOB_FUNCTION_IDS:
            current_index += 1
            if not SAMPLE_MODE and current_index != task_id:
                continue

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

            records = []
            for iid in BBOB_INSTANCE_IDS:
                y_path = BBOB_RAW_DIR / f"F{fid}_D{dim}_I{iid}_S{SAMPLE_SIZE}_R{sid}.csv"
                if not y_path.is_file():
                    print(f"[WARN] Missing Y: {y_path}. Skipping.")
                    continue
                try:
                    y_raw = pd.read_csv(y_path)
                    y_raw[["Y"]] = StandardScaler().fit_transform(y_raw[["Y"]])
                    y_np = y_raw["Y"].to_numpy()
                except Exception as e:
                    print(f"[ERROR] Y processing failed '{y_path}': {e}. Skipping.")
                    continue

                rec = {"function": f"F{fid}_D{dim}_I{iid}_S{SAMPLE_SIZE}_R{sid}"}
                for name, fn, kwargs in [
                    ("ela_meta",   calculate_ela_meta if (HAS_ELA_META and dim == 41) else None, {}),
                    ("ela_distr",  calculate_ela_distribution, {}),
                    ("ela_level",  calculate_ela_level, {}),
                    ("disp",       calculate_dispersion, {}),
                    ("ic",         calculate_information_content, {"seed": 0}),
                    ("nbc",        calculate_nbc, {}),
                    ("pca",        calculate_pca, {}),
                    ("fdc",        calculate_fitness_distance_correlation, {}),
                ]:
                    if fn is None:
                        continue
                    try:
                        rec.update(fn(x_np, y_np, **kwargs))
                    except Exception as e:
                        print(f"[ERROR] {name} failed F{fid}_D{dim}_I{iid}_R{sid}: {e}")
                records.append(rec)

            out_path = BBOB_ELA_DIR / f"ELA_F{fid}_D{dim}_S{SAMPLE_SIZE}_R{sid}.csv"
            pd.DataFrame(records).to_csv(out_path, index=False)
            print(f"[OK] Wrote {out_path} ({len(records)} instance records)")

            if not SAMPLE_MODE:
                # Task dispatch selects exactly one (dim, sid, fid) triple;
                # nothing further to do in this invocation.
                sys.exit(0)
