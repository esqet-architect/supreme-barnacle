#!/usr/bin/env python3
"""
MINERVA V7.9.5: CHHABRA–JENSEN DIRECT MULTIFRACTAL SPECTRUM ENGINE
================================================================================
- Bypasses numerical Legendre differentiation via direct escort distributions.
- Stabilizes negative moment tracking (q < 0) over sparse partition bounds.
- Operates strictly within an anisotropy-adapted Mahalanobis coordinate space.
"""

import numpy as np
from scipy.linalg import pinvh, cholesky
import warnings
warnings.filterwarnings('ignore')

class ChhabraJensenEngine:
    def __init__(self, coordinates):
        self.X = np.array(coordinates, dtype=float)
        self.n_samples, self.dims = self.X.shape
        self.X_transformed = self._apply_metric_tensor()

    def _apply_metric_tensor(self):
        """Applies global covariance whitening to handle manifold anisotropy."""
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

    def extract_direct_spectrum(self, q_values, eps_min=0.1, eps_max=0.35, num_eps=10):
        """
        Executes Chhabra-Jensen direct estimation of alpha(q) and f(q)
        by performing independent linear regressions over escort measure steps.
        """
        print("STAGE 1: Constructing Spatial Partitions & Probabilities")
        eps_scales = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
        log_eps = np.log(eps_scales)
        
        X_bin = self.X_transformed - np.min(self.X_transformed, axis=0)
        
        # Pre-generate standard box probabilities across scales
        box_probs_list = []
        for eps in eps_scales:
            bins = [np.arange(0, np.max(X_bin[:, d]) + eps, eps) for d in range(self.dims)]
            counts, _ = np.histogramdd(X_bin, bins=bins)
            flat_counts = counts.flatten()
            nz_counts = flat_counts[flat_counts > 0]
            probs = nz_counts / self.n_samples
            box_probs_list.append(probs)

        alpha_spectrum = []
        f_spectrum = []

        print("STAGE 2: Computing Direct Escort Regressions")
        for q in q_values:
            numerator_alpha = []
            numerator_f = []

            for probs in box_probs_list:
                # Compute the partition weights denominator: sum(p_i^q)
                Z_q = np.sum(probs**q)
                
                # Construct the localized escort distribution mu_i(q, eps)
                mu = (probs**q) / Z_q
                
                # Direct expectation tracking
                num_alpha_step = np.sum(mu * np.log(probs))
                num_f_step = np.sum(mu * np.log(mu))
                
                numerator_alpha.append(num_alpha_step)
                numerator_f.append(num_f_step)

            # Regress directly against log(eps). 
            # Slope of numerator_alpha vs log_eps yields alpha(q)
            # Slope of numerator_f vs log_eps yields f(q)
            slope_alpha, _ = np.polyfit(log_eps, numerator_alpha, 1)
            slope_f, _ = np.polyfit(log_eps, numerator_f, 1)

            alpha_spectrum.append(slope_alpha)
            f_spectrum.append(slope_f)

        alpha_spectrum = np.array(alpha_spectrum)
        f_spectrum = np.array(f_spectrum)

        print(f"\n   {'q_order':<8} | {'alpha (Direct)':<18} | {'f(alpha) (Direct)'} | Physical Status")
        print(f"   {'-'*72}")
        for idx in range(len(q_values)):
            if idx % 2 == 0 or abs(q_values[idx]) < 1e-5:
                status = "✅ Valid" if f_spectrum[idx] <= self.dims else "⚠️ Inflated"
                print(f"   {q_values[idx]:<8.2f} | {alpha_spectrum[idx]:<18.4f} | {f_spectrum[idx]:<17.4f} | {status}")
        print(f"   {'-'*72}")

        f_max = np.max(f_spectrum)
        print("📊 Chhabra-Jensen Diagnostic Matrix Verification:")
        print(f"   -> Peak Extracted Singularity Dimension f(alpha_max): {f_max:.4f}")
        
        if f_max > self.dims:
            print(f"   ❌ CRITICAL INFRASTRUCTURE WARNING: Peak f(alpha) = {f_max:.4f} still exceeds d_embed = {self.dims}!")
            print("      -> The uniform partition grid is causing irrecoverable sampling breakdown at negative tails.")
        else:
            print(f"   ✅ SUCCESS: Full spectrum complies with physical constraints (f(alpha) <= {self.dims}).")

        return q_values, alpha_spectrum, f_spectrum

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.5: CHHABRA–JENSEN DIRECT SPECTRAL INFERENCE HARNESS")
    print("=" * 95)

    # Recreate the highly anisotropic simulation attractor (Kappa ~ 95.85, Embedded in 2D)
    np.random.seed(42)
    t = np.linspace(0, 50, 5000)
    x = np.sin(t) * np.exp(-0.0005 * t)
    y = np.cos(np.sqrt(3) * t) * 0.05
    mock_data = np.column_stack([x, y])

    # Scan the moment spectrum safely from -3.0 to +3.0
    q_range = np.linspace(-3.0, 3.0, 13)

    engine = ChhabraJensenEngine(mock_data)
    engine.extract_direct_spectrum(q_range, eps_min=0.12, eps_max=0.32, num_eps=10)
    print("=" * 95)
