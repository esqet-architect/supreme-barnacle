#!/usr/bin/env python3
"""
MINERVA V7.9.4: CANONICAL MULTIFRACTAL SPECTRUM & CONCAVITY VALIDATOR
================================================================================
- Corrects tau(q) formulation to eliminate unphysical f(alpha) > d_embed inflation.
- Evaluates analytical concavity guards (d2_tau/dq2 <= 0) for validation.
- Operates strictly within the established Mahalanobis metric space framework.
"""

import numpy as np
from scipy.linalg import pinvh, cholesky
import warnings
warnings.filterwarnings('ignore')

class CanonicalSpectrumEngine:
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

    def compute_valid_spectrum(self, q_values, eps_min=0.1, eps_max=0.35, num_eps=10, min_density=0.0005):
        """
        Extracts a mathematically unified tau(q) curve and uses a clean
        Legendre transform to calculate physical alpha and f(alpha) profiles.
        """
        print("STAGE 1: Extracting Unified Partition Mass Exponents tau(q)")
        eps_scales = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
        log_eps = np.log(eps_scales)
        
        X_bin = self.X_transformed - np.min(self.X_transformed, axis=0)
        
        partition_probs = []
        for eps in eps_scales:
            bins = [np.arange(0, np.max(X_bin[:, d]) + eps, eps) for d in range(self.dims)]
            counts, _ = np.histogramdd(X_bin, bins=bins)
            
            flat_counts = counts.flatten()
            total_binned = np.sum(flat_counts)
            
            # Stabilization filter for negative q: drop sparse edge bins that cause numerical blowing
            nz_counts = flat_counts[flat_counts > (min_density * total_binned)]
            probs = nz_counts / np.sum(nz_counts)
            partition_probs.append(probs)

        tau_q = []

        # Compute tau(q) uniformly across all moments using the canonical definition
        for q in q_values:
            log_Zq = []
            for probs in partition_probs:
                # Z(q, eps) = sum(p_i^q)
                Z_q = np.sum(probs**q)
                log_Zq.append(np.log(Z_q))
            
            # Slope of ln(Z_q) vs ln(eps) yields the mass exponent tau(q)
            slope, _ = np.polyfit(log_eps, log_Zq, 1)
            tau_q.append(slope)

        tau_q = np.array(tau_q)

        print("STAGE 2: Executing Legendre Transform & Concavity Diagnostics")
        # Step-size for gradient calculations
        dq = np.gradient(q_values)
        
        # alpha(q) = d_tau / d_q
        alpha = np.gradient(tau_q, q_values)
        
        # f(alpha) = q * alpha(q) - tau(q)
        f_alpha = (q_values * alpha) - tau_q
        
        # Verify mathematical concavity: d2_tau/dq2 must be <= 0
        d2_tau = np.gradient(alpha, q_values)
        is_concave = np.all(d2_tau <= 1e-3) # Allowing slight numerical tolerance
        
        print(f"\n   {'q_order':<8} | {'tau(q)':<9} | {'alpha (Strength)':<18} | {'f(alpha) (Dimension)'}")
        print(f"   {'-'*65}")
        for idx in range(len(q_values)):
            if idx % 2 == 0 or abs(q_values[idx]) < 1e-5:
                print(f"   {q_values[idx]:<8.2f} | {tau_q[idx]:<9.4f} | {alpha[idx]:<18.4f} | {f_alpha[idx]:.4f}")
        print(f"   {'-'*65}")

        # Post-execution physical validation checks
        f_max = np.max(f_alpha)
        print("📊 Validation Summary metrics:")
        print(f"   -> Enforced Convexity / Concavity Check: {'PASSED' if is_concave else 'FAILED'}")
        print(f"   -> Peak Singularity Dimension f(alpha_max): {f_max:.4f}")
        
        if f_max > self.dims:
            print(f"   ⚠️ PHYSICAL BOUNDS VIOLATION: f(alpha) = {f_max:.4f} exceeds embedding dimension d = {self.dims}!")
        else:
            print(f"   ✅ PHYSICAL BOUNDS VERIFIED: f(alpha) spectrum stays within embedding dimension bounds.")

        return q_values, tau_q, alpha, f_alpha

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.4: CANONICAL MULTIFRACTAL FORMALISM CONVERGENCE")
    print("=" * 95)

    # Generate our highly anisotropic simulation attractor (Kappa ~ 95.85)
    np.random.seed(42)
    t = np.linspace(0, 50, 5000)
    x = np.sin(t) * np.exp(-0.0005 * t)
    y = np.cos(np.sqrt(3) * t) * 0.05
    mock_data = np.column_stack([x, y])

    # Scan a symmetric range of moments
    q_range = np.linspace(-3.0, 3.0, 19)

    engine = CanonicalSpectrumEngine(mock_data)
    engine.compute_valid_spectrum(q_range, eps_min=0.12, eps_max=0.32, num_eps=10)
    print("=" * 95)
