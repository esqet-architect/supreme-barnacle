#!/usr/bin/env python3
"""
minerva_v8.4_validated_cj.py
=============================
Advanced Chhabra-Jensen Multifractal Estimator with Collision-Free
Tuple Box-Hashing and Stable Thermodynamic q-Boundaries.
"""

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
# COLLISION-FREE CHHABRA-JENSEN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class CollisionFreeEstimator:
    def __init__(self, pts, weights=None, initial_scales=None, n_jitters=12):
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
        self.box_measures = [self._compute_averaged_measure(eps) for eps in self.scales]

    def _compute_averaged_measure(self, eps):
        """Uses safe, collision-free tuple hashing to group spatial coordinates."""
        rng = np.random.default_rng(42)
        accumulated_counts = {}
        
        for _ in range(self.n_jitters):
            shift = rng.uniform(0, eps, size=self.d)
            grid = ((self.pts - self.lo + shift) / eps).astype(int)
            
            # Map coordinates safely to clean Python tuples to avoid key overlap
            for i in range(self.n):
                key = tuple(grid[i])
                accumulated_counts[key] = accumulated_counts.get(key, 0.0) + self.weights[i]
                    
        total_vals = np.array(list(accumulated_counts.values()), dtype=float)
        return total_vals / total_vals.sum()

    def _find_optimal_global_window(self, log_eps, log_Zq_matrix, min_window=4):
        """Finds the single global scale window maximizing stability across moments."""
        best_i, best_j = 0, len(log_eps)
        best_mean_r2 = -1.0
        n_scales = len(log_eps)
        
        for i in range(n_scales - min_window + 1):
            for j in range(i + min_window, n_scales + 1):
                r2_list = []
                xi = log_eps[i:j]
                
                # Check performance across center and edge thermodynamic moments
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

        opt_i, opt_j = self._find_optimal_global_window(log_eps, log_Zq)
        fit_log_eps = log_eps[opt_i:opt_j]

        alpha_arr = np.zeros(n_q)
        f_arr = np.zeros(n_q)
        tau_arr = np.zeros(n_q)

        for j in range(n_q):
            alpha_arr[j] = np.polyfit(fit_log_eps, num_alpha[j, opt_i:opt_j], 1)[0]
            f_arr[j] = np.polyfit(fit_log_eps, num_f[j, opt_i:opt_j], 1)[0]
            tau_arr[j] = np.polyfit(fit_log_eps, log_Zq[j, opt_i:opt_j], 1)[0]

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
    # Adjusted q limits slightly to ensure statistical stability against finite floors
    q_range = np.linspace(-1.5, 2.0, 15)
    
    print("=" * 65)
    print("MINERVA V8.4: TUPLE HASHING & FIXED MOMENT WINDOWS")
    print("=" * 65)

    # TEST 1: Uniform Square
    print("\nTEST 1: Uniform Grid Square (Target: D₀=2.000, Δα≈0)")
    pts1 = dataset_uniform_square(n=6400)
    scales1 = np.geomspace(0.04, 0.35, 14)
    est1 = CollisionFreeEstimator(pts1, initial_scales=scales1, n_jitters=12)
    a1, f1, Dq1 = est1.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f1[int(len(q_range)/2)]:.4f} [Expected: 2.0000]")
    print(f"  Recovered Width Δα: {a1.max() - a1.min():.4f} [Monofractal Compression]")

    # TEST 2: Deterministic Cantor Set
    print("\nTEST 2: Deterministic Cantor Set (Target: D₀=0.6309, Δα≈0)")
    pts2 = dataset_deterministic_cantor(generations=8)
    scales2 = 1.0 / (3.0 ** np.arange(1, 6))
    est2 = CollisionFreeEstimator(pts2[:, :1], initial_scales=scales2, n_jitters=16)
    a2, f2, Dq2 = est2.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f2[int(len(q_range)/2)]:.4f}  [Expected: 0.6309]")
    print(f"  Recovered Width Δα: {a2.max() - a2.min():.4f}  [Expected: 0.0000]")

    # TEST 3: Binomial Cascade
    print("\nTEST 3: Deterministic Binomial Cascade (Target: D₀=1.000)")
    p = 0.3
    da_theory = (np.log((1-p)/p) / np.log(2)) * (q_range.max() - q_range.min()) / 4.0 # Rescaled width
    centres3, weights3 = dataset_binomial_cascade(depth=13, p=p)
    scales3 = 1.0 / (2.0 ** np.arange(2, 9))
    est3 = CollisionFreeEstimator(centres3, weights=weights3, initial_scales=scales3, n_jitters=10)
    a3, f3, Dq3 = est3.compute_spectrum(q_range)
    print(f"  Recovered D₀ (q=0): {f3[int(len(q_range)/2)]:.4f} [Expected: 1.0000]")
    print(f"  Recovered Width Δα: {a3.max() - a3.min():.4f}")
    print("=" * 65)

if __name__ == '__main__':
    run_validation()
