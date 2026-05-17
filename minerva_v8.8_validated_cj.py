#!/usr/bin/env python3
"""
minerva_v8.8_validated_cj.py
=============================
Thermodynamically Pure Chhabra-Jensen Multifractal Estimator.
Removes jitter loops entirely to prevent Jensen's Inequality bias.
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════
# DATASET GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def dataset_uniform_square(n=6400):
    """Generates a dense, uniform grid to verify perfect monofractal flatness."""
    side = int(np.sqrt(n))
    x = np.linspace(0.001, 0.999, side)
    y = np.linspace(0.001, 0.999, side)
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])

def dataset_deterministic_cantor(generations=8):
    """Generates midpoints of surviving Cantor intervals."""
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
    """Generates an asymmetric binomial cascade weight distribution."""
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
# PURE CHHABRA-JENSEN ENGINE (NO JITTERING)
# ═══════════════════════════════════════════════════════════════════════════

class PureChhabraJensenEstimator:
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

    def _get_pure_partition(self, eps):
        """Computes a clean spatial box distribution tied directly to origin minimums."""
        grid = ((self.pts - self.lo) / eps).astype(int)
        
        counts = {}
        for i in range(self.n):
            key = tuple(grid[i])
            counts[key] = counts.get(key, 0.0) + self.weights[i]
            
        vals = np.array(list(counts.values()), dtype=float)
        return vals / vals.sum()

    def _find_optimal_global_window(self, log_eps, log_Zq_matrix, min_window=4):
        """Locates the optimal structural scaling range maximizing overall R²."""
        best_i, best_j = 0, len(log_eps)
        best_mean_r2 = -1.0
        n_scales = len(log_eps)
        
        for i in range(n_scales - min_window + 1):
            for j in range(i + min_window, n_scales + 1):
                r2_list = []
                xi = log_eps[i:j]
                
                for q_idx in range(log_Zq_matrix.shape[0]):
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

        log_Zq_matrix = np.zeros((n_q, n_s))
        num_alpha_matrix = np.zeros((n_q, n_s))
        num_f_matrix = np.zeros((n_q, n_s))

        for s, eps in enumerate(self.scales):
            # Compute a single, structurally isolated spatial partition per scale
            mu = self._get_pure_partition(eps)
            
            for j, q in enumerate(q_values):
                if q == 0:
                    Z_q = float(len(mu))
                    mu_tilde = np.ones(len(mu)) / Z_q
                elif q == 1:
                    Z_q = 1.0
                    mu_tilde = mu.copy()
                else:
                    Z_q = np.sum(mu ** q)
                    mu_tilde = (mu ** q) / Z_q

                log_Zq_matrix[j, s] = np.log(Z_q)
                num_alpha_matrix[j, s] = np.sum(mu_tilde * np.log(mu))
                num_f_matrix[j, s] = np.sum(mu_tilde * np.log(mu_tilde))

        opt_i, opt_j = self._find_optimal_global_window(log_eps, log_Zq_matrix)
        fit_log_eps = log_eps[opt_i:opt_j]

        alpha_arr = np.zeros(n_q)
        f_arr = np.zeros(n_q)
        tau_arr = np.zeros(n_q)

        for j in range(n_q):
            alpha_arr[j] = np.polyfit(fit_log_eps, num_alpha_matrix[j, opt_i:opt_j], 1)[0]
            f_arr[j] = np.polyfit(fit_log_eps, num_f_matrix[j, opt_i:opt_j], 1)[0]
            tau_arr[j] = np.polyfit(fit_log_eps, log_Zq_matrix[j, opt_i:opt_j], 1)[0]

        Dq_arr = np.zeros(n_q)
        for j, q in enumerate(q_values):
            if abs(q - 1.0) > 0.01:
                Dq_arr[j] = tau_arr[j] / (q - 1.0)
            else:
                Dq_arr[j] = alpha_arr[j]

        return alpha_arr, f_arr, Dq_arr

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_validation():
    q_range = np.linspace(-1.5, 2.0, 15)
    print("=" * 65)
    print("MINERVA V8.8: PURE CHHABRA-JENSEN LOG-SPACE SCALING")
    print("=" * 65)

    # TEST 1: Uniform Square
    print("\nTEST 1: Uniform Grid Square (Target: D₀=2.000, Δα≈0)")
    pts1 = dataset_uniform_square(n=6400)
    scales1 = np.geomspace(0.04, 0.15, 14)
    est1 = PureChhabraJensenEstimator(pts1, initial_scales=scales1)
    a1, f1, Dq1 = est1.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f1[int(len(q_range)/2)]:.4f} [Expected: 2.0000]")
    print(f"  Recovered Width Δα: {a1.max() - a1.min():.4f}")

    # TEST 2: Deterministic Cantor Set
    print("\nTEST 2: Deterministic Cantor Set (Target: D₀=0.6309, Δα≈0)")
    pts2 = dataset_deterministic_cantor(generations=8)
    scales2 = 1.0 / (3.0 ** np.arange(1, 6))
    est2 = PureChhabraJensenEstimator(pts2[:, :1], initial_scales=scales2)
    a2, f2, Dq2 = est2.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f2[int(len(q_range)/2)]:.4f}  [Expected: 0.6309]")
    print(f"  Recovered Width Δα: {a2.max() - a2.min():.4f}  [Expected: 0.0000]")

    # TEST 3: Binomial Cascade
    print("\nTEST 3: Deterministic Binomial Cascade (Target: D₀=1.000)")
    p = 0.3
    centres3, weights3 = dataset_binomial_cascade(depth=13, p=p)
    scales3 = 1.0 / (2.0 ** np.arange(3, 10))
    est3 = PureChhabraJensenEstimator(centres3, weights=weights3, initial_scales=scales3)
    a3, f3, Dq3 = est3.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f3[int(len(q_range)/2)]:.4f} [Expected: 1.0000]")
    print(f"  Recovered Width Δα: {a3.max() - a3.min():.4f}")
    print("=" * 65)

    # PRODUCTION RUN: RAUZY POINT MATRIX
    rauzy_path = 'data/rauzy_points.npy'
    if os.path.exists(rauzy_path):
        print("\nPRODUCING SPECTRA: RAUZY POINTS LOCAL DISK ARRAY")
        print("-" * 65)
        pts_rauzy = np.load(rauzy_path)
        scales_rauzy = np.geomspace(0.05, 0.40, 14)
        q_rauzy = np.linspace(-1.5, 1.5, 13)
        
        est_rauzy = PureChhabraJensenEstimator(pts_rauzy, initial_scales=scales_rauzy)
        a_r, f_r, Dq_r = est_rauzy.compute_spectrum(q_rauzy)
        
        print(f"  Rauzy Extracted D₀ (q=0): {f_r[6]:.6f}")
        print(f"  Rauzy Extracted Width Δα: {a_r.max() - a_r.min():.6f}")
        print(f"  Full Dq Array (Unrounded):\n  {Dq_r}")
        print("=" * 65)
    else:
        print(f"\nRauzy production file not located at '{rauzy_path}'. Skipping production sweep.")
        print("=" * 65)

if __name__ == '__main__':
    run_validation()
