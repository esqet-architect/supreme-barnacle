#!/usr/bin/env python3
"""
MINERVA V7.9.2: ANISOTROPIC METRIC ADAPTATION & CONVERGENCE HARNESS
================================================================================
- Deploys Mahalanobis metric tensors to resolve severe manifold eccentricity.
- Integrates a sliding scale-window analyzer to track finite-size dimension drift.
- Preserves unconstrained, natural Rényi entropy hierarchies.
"""

import numpy as np
from scipy.linalg import pinvh, cholesky
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class AnisotropicManifoldProfiler:
    def __init__(self, coordinates):
        self.X = np.array(coordinates, dtype=float)
        self.n_samples, self.dims = self.X.shape
        self.X_transformed = self._apply_metric_tensor()

    def _apply_metric_tensor(self):
        """
        Computes the global covariance tensor, extracts the inverse metric space, 
        and transforms the coordinates to neutralize scaling anisotropy.
        """
        X_centered = self.X - np.mean(self.X, axis=0)
        cov = (X_centered.T @ X_centered) / (self.n_samples - 1)
        
        # Compute precision matrix (inverse covariance) securely using pseudo-inverse
        precision = pinvh(cov)
        
        # Use Cholesky decomposition to map coordinates into the normalized space
        # Transform: X_new = X_centered @ L, such that Cov(X_new) becomes Identity
        try:
            L = cholesky(precision, lower=False)
            return X_centered @ L
        except:
            # Fallback to standard eigenvalue scaling if Cholesky encounters numerical limits
            evals, evecs = np.linalg.eigh(cov)
            evals_inv = np.array([1.0 / np.sqrt(ev) if ev > 1e-11 else 0.0 for ev in evals])
            return X_centered @ (evecs * evals_inv)

    def analyze_sliding_windows(self, eps_min=0.01, eps_max=0.5, total_steps=12, window_size=5):
        """
        Evaluates partition scaling dimensions across a sliding window to track
        finite-scale convergence and identify structural exponent drift.
        """
        print(f"STAGE 1: Sliding Scale-Window Asymptotic Analysis (Size={window_size} steps)")
        eps_scales = np.logspace(np.log10(eps_min), np.log10(eps_max), total_steps)
        
        log_eps = []
        H_0, H_1, H_2 = [], [], []

        # Shift transformed space for safe multi-dimensional histogram binning
        X_bin = self.X_transformed - np.min(self.X_transformed, axis=0)

        for eps in eps_scales:
            bins = [np.arange(0, np.max(X_bin[:, d]) + eps, eps) for d in range(self.dims)]
            counts, _ = np.histogramdd(X_bin, bins=bins)
            
            probs = counts.flatten()[counts.flatten() > 0] / self.n_samples
            log_eps.append(np.log(eps))
            
            H_0.append(np.log(len(probs)))
            H_1.append(-np.sum(probs * np.log(probs)))
            H_2.append(-np.log(np.sum(probs**2)))

        print(f"   {'Window Scope':<18} | {'D_0':<8} | {'D_1':<8} | {'D_2':<8} | Status")
        print(f"   {'-'*72}")

        d0_records, d1_records, d2_records = [], [], []

        # Slide across the computed scale metrics
        for i in range(total_steps - window_size + 1):
            w_log_eps = log_eps[i : i + window_size]
            w_H0 = H_0[i : i + window_size]
            w_H1 = H_1[i : i + window_size]
            w_H2 = H_2[i : i + window_size]

            d0 = -float(np.polyfit(w_log_eps, w_H0, 1)[0])
            d1 = -float(np.polyfit(w_log_eps, w_H1, 1)[0])
            d2 = -float(np.polyfit(w_log_eps, w_H2, 1)[0])

            d0_records.append(d0)
            d1_records.append(d1)
            d2_records.append(d2)

            status = "✅" if d0 >= d1 >= d2 else "❌"
            scope_str = f"eps[{i}] -> eps[{i+window_size-1}]"
            print(f"   {scope_str:<18} | {d0:<8.4f} | {d1:<8.4f} | {d2:<8.4f} | {status}")

        print(f"   {'-'*72}")
        print(f"   * Exponent Variance Profile (Drift StdDev):")
        print(f"     ΔD_0: {np.std(d0_records):.5f} | ΔD_1: {np.std(d1_records):.5f} | ΔD_2: {np.std(d2_records):.5f}")
        return np.mean(d2_records)

    def run_anisotropic_gp_engine(self, r_min=0.01, r_max=0.5, num_r=12):
        """
        Computes the Grassberger-Procaccia correlation dimension inside the
        transformed Mahalanobis metric space to bypass alignment distortions.
        """
        print("\nSTAGE 2: Anisotropy-Adapted Grassberger-Procaccia Architecture")
        r_scales = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
        
        # Subsample to keep processing times light and efficient
        X_sample = self.X_transformed if self.n_samples <= 2000 else self.X_transformed[np.random.choice(self.n_samples, 2000, replace=False)]
        n_pts = len(X_sample)
        
        # Pairwise distance in the transformed symmetric space is mathematically equivalent to Mahalanobis
        pairwise_dists = pdist(X_sample, metric='euclidean')

        log_r, log_C = [], []
        for r in r_scales:
            pairs_inside = np.sum(pairwise_dists < r)
            correlation_sum = pairs_inside / (0.5 * n_pts * (n_pts - 1))
            if correlation_sum > 0:
                log_r.append(np.log(r))
                log_C.append(np.log(correlation_sum))

        D_2_gp = float(np.polyfit(log_r, log_C, 1)[0])
        print(f"   -> Anisotropy-Corrected GP D_2: {D_2_gp:.4f}")
        return D_2_gp

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.2: METRIC TENSOR ADAPTATION & ASYMPTOTIC ENGINE")
    print("=" * 95)

    # Re-generate the highly anisotropic validation attractor (Kappa ~ 95.85)
    np.random.seed(42)
    t = np.linspace(0, 50, 4000)
    x = np.sin(t) * np.exp(-0.001 * t)
    y = np.cos(np.sqrt(3) * t) * 0.1  # Highly compressed axis variance
    mock_data = np.column_stack([x, y])

    profiler = AnisotropicManifoldProfiler(mock_data)
    mean_partition_d2 = profiler.analyze_sliding_windows(eps_min=0.03, eps_max=0.35, total_steps=12, window_size=6)
    gp_d2 = profiler.run_anisotropic_gp_engine(r_min=0.03, r_max=0.35, num_r=12)
    
    print("─" * 95)
    print(f"📊 Final Inter-Estimator Discrepancy (Partition Mean D_2 vs GP D_2): {abs(mean_partition_d2 - gp_d2):.4f}")
    print("=" * 95)
