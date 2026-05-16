"""
run_bbob_adam.py
----------------
Run finite-difference Adam on the BBOB large-scale suite.

SLURM: one task per problem (24 fns x 15 inst x 3 dims = 1080 tasks).
Forward differences: d+1 evals per gradient step.
Remaining budget after last full step is exhausted at the final point.
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
import cocoex

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cornn.config import (
    BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS,
    BBOB_ADAM_DIR,
    BUDGET, N_RUNS, BOUNDS_LO, BOUNDS_HI,
    ADAM_LR, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, FD_EPS,
    get_task_id, make_dirs,
)

make_dirs()


class FiniteDifferenceAdam:
    """
    Adam optimiser using forward finite differences for gradient estimation.

    One base evaluation f(x) is shared across all d perturbations, costing
    d+1 evaluations per gradient step. The optimiser stops when the budget
    would be exceeded; remaining evaluations are filled with f(x_final).
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

    def minimize(self, f, x0):
        x      = np.clip(x0.copy(), self.lo, self.hi)
        m      = np.zeros(self.dim)
        v      = np.zeros(self.dim)
        t      = 0
        evals  = 0
        hist   = []
        cost   = self.dim + 1   # forward differences: d+1 evals per step

        while evals + cost <= self.budget:
            f_0 = f(np.clip(x, self.lo, self.hi))
            hist.append(f_0)
            evals += 1

            grad = np.zeros(self.dim)
            for i in range(self.dim):
                x_fwd   = x.copy(); x_fwd[i] += self.eps_fd
                f_fwd   = f(np.clip(x_fwd, self.lo, self.hi))
                hist.append(f_fwd)
                grad[i] = (f_fwd - f_0) / self.eps_fd
                evals  += 1

            t      += 1
            m       = self.beta1 * m + (1.0 - self.beta1) * grad
            v       = self.beta2 * v + (1.0 - self.beta2) * grad ** 2
            m_hat   = m / (1.0 - self.beta1 ** t)
            v_hat   = v / (1.0 - self.beta2 ** t)
            x       = x - self.alpha * m_hat / (np.sqrt(v_hat) + self.eps_adam)
            x       = np.clip(x, self.lo, self.hi)

        # Exhaust remaining budget at the final point.
        f_final = f(np.clip(x, self.lo, self.hi))
        while evals < self.budget:
            hist.append(f_final)
            evals += 1

        return hist


suite         = cocoex.Suite(BBOB_SUITE, BBOB_INSTANCES, BBOB_SETTINGS)
task_id       = get_task_id()
current_index = 0

for problem in suite:
    current_index += 1
    if current_index != task_id:
        continue

    optimizer = FiniteDifferenceAdam(dim=problem.dimension, budget=BUDGET)

    for run in range(N_RUNS):
        out_path = BBOB_ADAM_DIR / f"{problem.id}_Adam_R{run}.csv"
        if out_path.is_file():
            continue

        rng = np.random.default_rng(run)
        x0  = rng.uniform(BOUNDS_LO, BOUNDS_HI, size=problem.dimension)
        hist = optimizer.minimize(problem, x0)
        pd.DataFrame(hist, columns=["Fval"]).to_csv(out_path, index=False)
        print(out_path)

    break  # one problem per task
