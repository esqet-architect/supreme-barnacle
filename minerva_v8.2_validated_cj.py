#!/usr/bin/env python3
"""
minerva_v8.2_validated_cj.py
=============================
Advanced Chhabra-Jensen Multifractal Estimator featuring Automated 
Scaling-Region Selection via local R² optimization and deterministic benchmarking.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# DETERMINISTIC DATASET GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def dataset_uniform_square(n=6000):
    """Generates a dense, uniform grid to minimize local Poisson noise."""
    side = int(np.sqrt(n))
    x = np.linspace(0.001, 0.999, side)
    y = np.linspace(0.001, 0.999, side)
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])

def dataset_deterministic_cantor(generations=7):
    """Generates exact midpoints of surviving Cantor intervals to remove noise."""
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
# CHHABRA-JENSEN WITH AUTOMATED SCALING SELECTION
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveChhabraJensenEstimator:
    def __init__(self, pts, weights=None, initial_scales=None):
        self.pts = np.asarray(pts, dtype=float)
        self.n = len(pts)
        self.d = pts.shape[1] if pts.ndim > 1 else 1
        self.scales = np.sort(initial_scales)
        
        if weights is None:
            self.weights = np.ones(self.n) / self.n
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()

        self.lo = self.pts.min(axis=0)
        
        # Precompute box measures across all initial scales
        self.box_measures = [self._compute_box_measure(eps) for eps in self.scales]

    def _compute_box_measure(self, eps):
        shift = eps * 0.005
        grid = ((self.pts - self.lo + shift) / eps).astype(int)
        
        # Fast tuple-free row grouping for low dimensions
        if self.d == 1:
            keys = grid[:, 0]
        else:
            keys = grid[:, 0] + grid[:, 1] * 46340  # Unique hashing layout
            
        unique_keys, inverse_indices = np.unique(keys, return_inverse=True)
        mu = np.zeros(len(unique_keys))
        np.add.at(mu, inverse_indices, self.weights)
        return mu[mu > 0]

    def _optimize_fit(self, x, y, min_window=4):
        """Finds the sliding window sequence that maximizes R² alignment."""
        best_slope = 0.0
        best_r2 = -1.0
        n_scales = len(x)
        
        for i in range(n_scales - min_window + 1):
            for j in range(i + min_window, n_scales + 1):
                xi = x[i:j]
                yi = y[i:j]
                
                slope, intercept = np.polyfit(xi, yi, 1)
                y_pred = slope * xi + intercept
                ss_res = np.sum((yi - y_pred) ** 2)
                ss_tot = np.sum((yi - np.mean(yi)) ** 2)
                
                r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                if r2 > best_r2:
                    best_r2 = r2
                    best_slope = slope
                    
        return best_slope, best_r2

    def compute_spectrum(self, q_values):
        q_values = np.asarray(q_values, dtype=float)
        log_eps = np.log(self.scales)

        alpha_arr = np.zeros(len(q_values))
        f_arr = np.zeros(len(q_values))
        Dq_arr = np.zeros(len(q_values))

        for j, q in enumerate(q_values):
            num_alpha = np.zeros(len(self.scales))
            num_f = np.zeros(len(self.scales))
            log_Zq = np.zeros(len(self.scales))

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

                log_Zq[s] = np.log(Z_q)
                num_alpha[s] = np.sum(mu_tilde * np.log(mu))
                num_f[s] = np.sum(mu_tilde * np.log(mu_tilde))

            # Optimally select the scaling regime for each thermodynamic property
            alpha_arr[j], _ = self._optimize_fit(log_eps, num_alpha)
            f_arr[j], _ = self._optimize_fit(log_eps, num_f)
            tau_slope, _ = self._optimize_fit(log_eps, log_Zq)
            
            Dq_arr[j] = tau_slope / (q - 1) if abs(q - 1) > 0.01 else alpha_arr[j]

        return alpha_arr, f_arr, Dq_arr

# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_validation():
    q_range = np.linspace(-2.0, 2.0, 17)
    
    print("=" * 65)
    print("MINERVA V8.2: ADAPTIVE R² MULTIFRACTAL EVALUATION")
    print("=" * 65)

    # TEST 1: Uniform Square
    print("\nTEST 1: Uniform Grid Square (Target: D₀=2.000, Δα≈0)")
    pts1 = dataset_uniform_square(n=6400)
    scales1 = np.geomspace(0.02, 0.3, 15)
    est1 = AdaptiveChhabraJensenEstimator(pts1, initial_scales=scales1)
    a1, f1, Dq1 = est1.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f1[8]:.4f}")
    print(f"  Recovered Width Δα: {a1.max() - a1.min():.4f}")

    # TEST 2: Deterministic Cantor Set
    print("\nTEST 2: Deterministic Cantor Set (Target: D₀=0.6309, Δα≈0)")
    pts2 = dataset_deterministic_cantor(generations=7)
    scales2 = 1.0 / (3.0 ** np.arange(1, 6))
    est2 = AdaptiveChhabraJensenEstimator(pts2[:, :1], initial_scales=scales2)
    a2, f2, Dq2 = est2.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f2[8]:.4f}  [Expected: 0.6309]")
    print(f"  Recovered Width Δα: {a2.max() - a2.min():.4f}  [Expected: 0.0000]")

    # TEST 3: Binomial Cascade
    print("\nTEST 3: Deterministic Binomial Cascade (Target: D₀=1.000)")
    p = 0.3
    da_theory = np.log((1-p)/p) / np.log(2)
    centres3, weights3 = dataset_binomial_cascade(depth=13, p=p)
    scales3 = 1.0 / (2.0 ** np.arange(2, 9))
    est3 = AdaptiveChhabraJensenEstimator(centres3, weights=weights3, initial_scales=scales3)
    a3, f3, Dq3 = est3.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f3[8]:.4f}")
    print(f"  Recovered Width Δα: {a3.max() - a3.min():.4f}  [Theory Δα: {da_theory:.4f}]")
    print("=" * 65)

if __name__ == '__main__':
    run_validation()
