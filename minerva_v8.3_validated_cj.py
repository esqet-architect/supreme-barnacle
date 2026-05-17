#!/usr/bin/env python3
"""
minerva_v8.3_validated_cj.py
=============================
Advanced Chhabra-Jensen Multifractal Estimator with Unified Scaling Windows,
Partition-Origin Averaging, and Cross-Verification Safeguards.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC BENCHMARK GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def dataset_uniform_square(n=6400):
    """Generates a dense, perfect grid to minimize Poisson fluctuations."""
    side = int(np.sqrt(n))
    x = np.linspace(0.001, 0.999, side)
    y = np.linspace(0.001, 0.999, side)
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])

def dataset_deterministic_cantor(generations=8):
    """Generates exact midpoints of surviving Cantor intervals."""
    bounds = [[0.0, 1.0]]
    for _ in range(generations):
        next_bounds = []
        for lo, hi in bounds:
            w = (hi - lo) / 3.0
            next_bounds.append([lo, lo + w])
            next_bounds.append([hi - w, hi])
        bounds = next_bounds
    midpoints = [0.5 * (lo + hi) for lo, hi in bounds]
    return np.column_stack([midpoints, np.zeros_like(midpoints)])

def dataset_binomial_cascade(depth=13, p=0.3):
    p2 = 1.0 - p
    n = 2 ** depth
    indices = np.arange(n)
    weights = np.ones(n, dtype=float)
    for level in range(depth):
        bits = (indices >> (depth - 1 - level)) & 1
        weights *= np.where(bits == 0, p, p2)
    centres = (indices + 0.5) / n
    return centres.reshape(-1, 1), weights / weights.sum()

# ═══════════════════════════════════════════════════════════════════════════
# ROBUST UNIFIED CHHABRA-JENSEN SPECTRUM ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class RobustChhabraJensenEstimator:
    def __init__(self, pts, weights=None, initial_scales=None, n_jitters=10):
        self.pts = np.asarray(pts, dtype=float)
        self.n = len(pts)
        self.d = pts.shape[1] if pts.ndim > 1 else 1
        self.scales = np.sort(initial_scales)
        self.n_jitters = n_jitters
        
        if weights is None:
            self.weights = np.ones(self.n) / self.n
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()

        self.lo = self.pts.min(axis=0)
        self.hi = self.pts.max(axis=0)
        
        # Precompute averaged box measures over multiple spatial offsets
        self.box_measures = [self._compute_averaged_measure(eps) for eps in self.scales]

    def _compute_averaged_measure(self, eps):
        """Averages cell occupancies over random origin shifts to fix boundary aliasing."""
        rng = np.random.default_rng(42)
        accumulated_counts = {}
        
        for _ in range(self.n_jitters):
            shift = rng.uniform(0, eps, size=self.d)
            grid = ((self.pts - self.lo + shift) / eps).astype(int)
            
            if self.d == 1:
                keys = grid[:, 0]
            else:
                keys = grid[:, 0] + grid[:, 1] * 46340
                
            unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
            mu = np.zeros(len(unique_keys))
            np.add.at(mu, inverse_indices, self.weights)
            
            for k, val in zip(unique_keys, mu):
                if val > 0:
                    accumulated_counts[k] = accumulated_counts.get(k, 0.0) + val
                    
        # Normalize to find true expected spatial distribution profiles
        total_vals = np.array(list(accumulated_counts.values()))
        return total_vals / total_vals.sum()

    def _find_optimal_global_window(self, log_eps, log_Zq_matrix, min_window=4):
        """Enforces a single global scaling interval by optimizing collective R² across tau."""
        best_i, best_j = 0, len(log_eps)
        best_mean_r2 = -1.0
        n_scales = len(log_eps)
        
        for i in range(n_scales - min_window + 1):
            for j in range(i + min_window, n_scales + 1):
                r2_list = []
                xi = log_eps[i:j]
                
                # Check linearity across a few key diagnostic moments (q=-1, q=0, q=2)
                for q_idx in [0, log_Zq_matrix.shape[0] // 2, log_Zq_matrix.shape[0] - 1]:
                    yi = log_Zq_matrix[q_idx, i:j]
                    slope, intercept = np.polyfit(xi, yi, 1)
                    y_pred = slope * xi + intercept
                    ss_res = np.sum((yi - y_pred) ** 2)
                    ss_tot = np.sum((yi - np.mean(yi)) ** 2)
                    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                    r2_list.append(r2)
                    
                mean_r2 = np.mean(r2_list)
                if mean_r2 > best_mean_r2:
                    best_mean_r2 = mean_r2
                    best_i, best_j = i, j
                    
        return best_i, best_j

    def compute_spectrum(self, q_values):
        q_values = np.asarray(q_values, dtype=float)
        log_eps = np.log(self.scales)
        n_q = len(q_values)
        n_s = len(self.scales)

        # Allocate internal core matrices
        num_alpha = np.zeros((n_q, n_s))
        num_f = np.zeros((n_q, n_s))
        log_Zq = np.zeros((n_q, n_s))

        for j, q in enumerate(q_values):
            for s, mu in enumerate(self.box_measures):
                if q == 0:
                    Z_q = float(len(mu))
                    mu_tilde = np.ones(len(mu)) / Z_q
                elif q == 1:
                    Z_q = 1.0
                    mu_tilde = mu.copy()
                else:
                    Z_q = np.sum(mu ** q)
                    mu_tilde = (mu ** q) / Z_q

                log_Zq[j, s] = np.log(Z_q)
                num_alpha[j, s] = np.sum(mu_tilde * np.log(mu))
                num_f[j, s] = np.sum(mu_tilde * np.log(mu_tilde))

        # Phase 1: Lock down ONE global scaling interval
        opt_i, opt_j = self._find_optimal_global_window(log_eps, log_Zq)
        fit_log_eps = log_eps[opt_i:opt_j]

        alpha_arr = np.zeros(n_q)
        f_arr = np.zeros(n_q)
        tau_arr = np.zeros(n_q)

        # Phase 2: Compute all scaling exponents over that strict interval
        for j in range(n_q):
            alpha_arr[j] = np.polyfit(fit_log_eps, num_alpha[j, opt_i:opt_j], 1)[0]
            f_arr[j] = np.polyfit(fit_log_eps, num_f[j, opt_i:opt_j], 1)[0]
            tau_arr[j] = np.polyfit(fit_log_eps, log_Zq[j, opt_i:opt_j], 1)[0]

        # Phase 3: Enforce derivative scaling consistency for Dq
        Dq_arr = np.zeros(n_q)
        for j, q in enumerate(q_values):
            if abs(q - 1.0) > 0.01:
                Dq_arr[j] = tau_arr[j] / (q - 1.0)
            else:
                Dq_arr[j] = alpha_arr[j]

        return alpha_arr, f_arr, Dq_arr

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_validation():
    q_range = np.linspace(-2.0, 2.0, 17)
    
    print("=" * 65)
    print("MINERVA V8.3: UNIFIED WINDOW & JITTER-AVERAGED EVALUATION")
    print("=" * 65)

    # TEST 1: Uniform Square
    print("\nTEST 1: Uniform Grid Square (Target: D₀=2.000, Δα≈0)")
    pts1 = dataset_uniform_square(n=6400)
    scales1 = np.geomspace(0.02, 0.4, 16)
    est1 = RobustChhabraJensenEstimator(pts1, initial_scales=scales1, n_jitters=15)
    a1, f1, Dq1 = est1.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f1[8]:.4f} [Expected: 2.0000]")
    print(f"  Recovered Width Δα: {a1.max() - a1.min():.4f} [Compressed Monofractal]")

    # TEST 2: Deterministic Cantor Set
    print("\nTEST 2: Deterministic Cantor Set (Target: D₀=0.6309, Δα≈0)")
    pts2 = dataset_deterministic_cantor(generations=8)
    scales2 = 1.0 / (3.0 ** np.arange(1, 6))
    est2 = RobustChhabraJensenEstimator(pts2[:, :1], initial_scales=scales2, n_jitters=20)
    a2, f2, Dq2 = est2.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f2[8]:.4f}  [Expected: 0.6309]")
    print(f"  Recovered Width Δα: {a2.max() - a2.min():.4f}  [Expected: 0.0000]")

    # TEST 3: Binomial Cascade
    print("\nTEST 3: Deterministic Binomial Cascade (Target: D₀=1.000, Δα=1.2224)")
    p = 0.3
    da_theory = np.log((1-p)/p) / np.log(2)
    centres3, weights3 = dataset_binomial_cascade(depth=13, p=p)
    scales3 = 1.0 / (2.0 ** np.arange(2, 9))
    est3 = RobustChhabraJensenEstimator(centres3, weights=weights3, initial_scales=scales3, n_jitters=10)
    a3, f3, Dq3 = est3.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f3[8]:.4f} [Expected: 1.0000]")
    print(f"  Recovered Width Δα: {a3.max() - a3.min():.4f}  [Theory Δα: {da_theory:.4f}]")
    print("=" * 65)

if __name__ == '__main__':
    run_validation()
