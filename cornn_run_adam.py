"""
run_cornn_adam.py
-----------------
Run finite-difference Adam on the CORNN benchmark suite.

SLURM: one task per (fcn, arch) pair (54 fns x 6 archs = 324 tasks).
Forward differences: d+1 evals per gradient step on training loss.
Test loss recorded once per gradient step; not counted against budget.
Remaining budget after last full step is exhausted at the final point.

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
import lib.CORNN as CORNN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    CORNN_ADAM_DIR,
    BUDGET, N_RUNS, BOUNDS_LO, BOUNDS_HI,
    ADAM_LR, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, FD_EPS,
    SAMPLE_MODE, get_task_id, make_dirs,
)

make_dirs()


class FiniteDifferenceAdam:
    """
    Adam with forward finite differences on training loss.
    Test loss is evaluated once per step at the updated point (not counted
    against the budget). Remaining budget exhausted at the final point.
    """

    def __init__(self, dim, budget,
                 bounds=(BOUNDS_LO, BOUNDS_HI),
                 alpha=ADAM_LR, beta1=ADAM_BETA1, beta2=ADAM_BETA2,
                 eps_adam=ADAM_EPS, eps_fd=FD_EPS):
        self.dim      = dim
        self.budget   = budget
        self.lo, self.hi = bounds
        self.alpha    = alpha
        self.beta1    = beta1
        self.beta2    = beta2
        self.eps_adam = eps_adam
        self.eps_fd   = eps_fd

    def minimize(self, f_train, f_test, x0):
        x      = np.clip(x0.copy(), self.lo, self.hi)
        m      = np.zeros(self.dim)
        v      = np.zeros(self.dim)
        t      = 0
        evals  = 0
        tr_hist, te_hist = [], []
        cost   = self.dim + 1

        while evals + cost <= self.budget:
            f_0 = f_train(np.clip(x, self.lo, self.hi))
            tr_hist.append(f_0)
            evals += 1

            grad = np.zeros(self.dim)
            for i in range(self.dim):
                x_fwd   = x.copy(); x_fwd[i] += self.eps_fd
                f_fwd   = f_train(np.clip(x_fwd, self.lo, self.hi))
                tr_hist.append(f_fwd)
                grad[i] = (f_fwd - f_0) / self.eps_fd
                evals  += 1

            t      += 1
            m       = self.beta1 * m + (1.0 - self.beta1) * grad
            v       = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            m_hat   = m / (1.0 - self.beta1 ** t)
            v_hat   = v / (1.0 - self.beta2 ** t)
            x       = x - self.alpha * m_hat / (np.sqrt(v_hat) + self.eps_adam)
            x       = np.clip(x, self.lo, self.hi)

            # Test loss at updated weights (not counted against budget)
            te_hist.append(f_test(np.clip(x, self.lo, self.hi)))

        # Exhaust remaining budget at the final point
        f_tr_final = f_train(np.clip(x, self.lo, self.hi))
        f_te_final = f_test(np.clip(x, self.lo, self.hi))
        while evals < self.budget:
            tr_hist.append(f_tr_final)
            evals += 1

        # Align test history to training history length
        evals_per_step      = self.dim + 1
        te_hist_aligned     = []
        for step_val in te_hist:
            te_hist_aligned.extend([step_val] * evals_per_step)
        n = len(tr_hist)
        te_hist_aligned = (te_hist_aligned + [f_te_final] * n)[:n]

        return tr_hist, te_hist_aligned


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

        nn_arch   = neural_network_dictionary[nn_architecture]()
        bench     = CORNN.NN_Benchmark(training_data, test_data, nn_arch)
        dim       = bench.get_weight_count()
        fcn_tag   = fcn.replace(" ", "_")
        arch_tag  = nn_architecture.replace(" ", "_")
        optimizer = FiniteDifferenceAdam(dim=dim, budget=BUDGET)

        def f_train(x): return bench.training_set_evaluation(x)
        def f_test(x):  return bench.testing_set_evaluation(x)

        for run in range(N_RUNS):
            out_path = CORNN_ADAM_DIR / f"F_{fcn_tag}_{arch_tag}_Adam_R{run}.csv"
            if out_path.is_file():
                print(f"[SKIP] {out_path} already exists")
                continue

            rng = np.random.default_rng(run)
            x0  = rng.uniform(BOUNDS_LO, BOUNDS_HI, size=dim)
            tr_hist, te_hist = optimizer.minimize(f_train, f_test, x0)
            pd.DataFrame({
                "training_loss": tr_hist,
                "testing_loss":  te_hist,
            }).to_csv(out_path, index=False)
            print(f"[OK] Wrote {out_path}")

        if not SAMPLE_MODE:
            break
    else:
        continue
    if not SAMPLE_MODE:
        break
