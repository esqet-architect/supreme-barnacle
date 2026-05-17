#!/usr/bin/env python3
"""
MINERVA V7.9.6: BOOTSTRAP UNCERTAINTY QUANTIFICATION & ESCORT FILTER
================================================================================
- Deploys a box-occupancy floor to suppress sparse boundary noise artifacts.
- Runs an automated bootstrap re-sampling loop to isolate negative-q variance.
- Tracks localized standard deviation drift profiles across the Rényi spectrum.
"""

import numpy as np
from scipy.linalg import pinvh, cholesky
import warnings
warnings.filterwarnings('ignore')

class BootstrapMultifractalEngine:
    def __init__(self, coordinates):
        self.X = np.array(coordinates, dtype=float)
        self.n_samples, self.dims = self.X.shape
        self.X_transformed = self._apply_metric_tensor()

    def _apply_metric_tensor(self):
        """Applies global covariance whitening to resolve geometric anisotropy."""
        X_centered = self.X - np.mean(self.X, axis=0)
        cov = (X_centered.T @ X_centered) / (self.n_samples - 1)
        precision = pinvh(cov)
        try:
            L = cholesky(precision, lower=False)
            return X_centered @ L
        except:
            evals, evecs = np.linalg.eigh(cov)
            evals_inv = np.array([1.0 / np.sqrt(ev) if ev > 1e-11 else 0.0 for ev in evals])
            return X_centered @ (evecs * evals_inv)

    def _run_single_cj_pass(self, data_matrix, q_values, eps_scales, min_count=3):
        """Executes a single direct Chhabra-Jensen estimation pass over an asset."""
        log_eps = np.log(eps_scales)
        X_bin = data_matrix - np.min(data_matrix, axis=0)
        
        box_probs_list = []
        for eps in eps_scales:
            bins = [np.arange(0, np.max(X_bin[:, d]) + eps, eps) for d in range(self.dims)]
            counts, _ = np.histogramdd(X_bin, bins=bins)
            flat_counts = counts.flatten()
            
            # CRITICAL CORRECTION: Drop ultra-sparse noise boxes that trigger negative-q blowups
            valid_counts = flat_counts[flat_counts >= min_count]
            
            if len(valid_counts) == 0:
                valid_counts = flat_counts[flat_counts > 0]
                
            probs = valid_counts / np.sum(valid_counts)
            box_probs_list.append(probs)

        alpha_pass = []
        f_pass = []

        for q in q_values:
            num_alpha = []
            num_f = []
            for probs in box_probs_list:
                Z_q = np.sum(probs**q)
                mu = (probs**q) / Z_q
                
                num_alpha.append(np.sum(mu * np.log(probs)))
                num_f.append(np.sum(mu * np.log(mu)))
                
            alpha_pass.append(np.polyfit(log_eps, num_alpha, 1)[0])
            f_pass.append(np.polyfit(log_eps, num_f, 1)[0])
            
        return np.array(alpha_pass), np.array(f_pass)

    def execute_bootstrap_validation(self, q_values, B=25, eps_min=0.12, eps_max=0.32, num_eps=10):
        """
        Runs B independent bootstrap cycles to extract standard deviation
        profiles, mapping out exactly where finite-size estimation degrades.
        """
        print(f"STAGE 1: Launching {B} Bootstrap Re-sampling Iterations...")
        eps_scales = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
        
        # Space arrays to collect data from each bootstrap trial
        all_alpha = np.zeros((B, len(q_values)))
        all_f = np.zeros((B, len(q_values)))

        for b in range(B):
            # Sample with replacement to create a distinct bootstrap replica
            indices = np.random.choice(self.n_samples, size=self.n_samples, replace=True)
            X_replica = self.X_transformed[indices]
            
            alpha_b, f_b = self._run_single_cj_pass(X_replica, q_values, eps_scales)
            all_alpha[b, :] = alpha_b
            all_f[b, :] = f_b

        # Calculate tracking averages and standard deviations across the runs
        mean_alpha = np.mean(all_alpha, axis=0)
        std_alpha = np.std(all_alpha, axis=0)
        mean_f = np.mean(all_f, axis=0)
        std_f = np.std(all_f, axis=0)

        print("\nSTAGE 2: Analyzing Uncertainty Metrics Across the Spectrum")
        print(f"\n   {'q_order':<8} | {'alpha (Mean)':<14} | {'σ_alpha':<8} | {'f(alpha) (Mean)':<15} | {'σ_f':<8} | Status")
        print(f"   {'-'*78}")
        
        for idx in range(len(q_values)):
            # Flag positions showing massive standard deviation or physical dimension inflation
            if std_f[idx] > 0.08 or mean_f[idx] > self.dims:
                status = "❌ UNSTABLE"
            else:
                status = "✅ STABLE"
            print(f"   {q_values[idx]:<8.2f} | {mean_alpha[idx]:<14.4f} | {std_alpha[idx]:<8.4f} | {mean_f[idx]:<15.4f} | {std_f[idx]:<8.4f} | {status}")
            
        print(f"   {'-'*78}")
        
        # Isolate the verified stable zone
        stable_qs = q_values[(std_f <= 0.08) & (mean_f <= self.dims)]
        if len(stable_qs) > 0:
            print(f"📊 Empirical Invariant Conclusion:\n   -> Physically Reliable Domain Range: q ∈ [{np.min(stable_qs)}, {np.max(stable_qs)}]")
        else:
            print("📊 Empirical Invariant Conclusion:\n   -> Warning: No fully stable scaling zone identified.")
            
        return mean_alpha, mean_f, std_f

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.6: BOOTSTRAP UNCERTAINTY QUANTIFICATION PLATFORM")
    print("=" * 95)

    # Recreate the highly anisotropic test attractor trajectory (Kappa ~ 95.85)
    np.random.seed(42)
    t = np.linspace(0, 50, 4000)
    x = np.sin(t) * np.exp(-0.0005 * t)
    y = np.cos(np.sqrt(3) * t) * 0.05
    mock_data = np.column_stack([x, y])

    # Scan moment spectrum from -3.0 to +3.0
    q_range = np.linspace(-3.0, 3.0, 13)

    harness = BootstrapMultifractalEngine(mock_data)
    harness.execute_bootstrap_validation(q_range, B=25, eps_min=0.12, eps_max=0.32, num_eps=10)
    print("=" * 95)
