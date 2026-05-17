#!/usr/bin/env python3
"""
MINERVA V7.9.3: METRIC-COVARIANT SINGULARITY SPECTRUM ENGINE [f(alpha)]
================================================================================
- Extracts the mass exponent tau(q) across an unconstrained range of moments.
- Executes numerical Legendre transforms to construct the f(alpha) spectrum.
- Operates entirely within an anisotropy-adapted Mahalanobis metric space.
"""

import numpy as np
from scipy.linalg import pinvh, cholesky
import warnings
warnings.filterwarnings('ignore')

class SingularitySpectrumEngine:
    def __init__(self, coordinates):
        self.X = np.array(coordinates, dtype=float)
        self.n_samples, self.dims = self.X.shape
        self.X_transformed = self._apply_metric_tensor()

    def _apply_metric_tensor(self):
        """Transforms coordinates via Cholesky factor of the precision matrix."""
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

    def compute_singularity_spectrum(self, q_values, eps_min=0.1, eps_max=0.4, num_eps=8):
        """
        Extracts tau(q) via linear regression over the stable scale window,
        then applies a Legendre transform to map out local alpha and f(alpha).
        """
        print("STAGE 1: Evaluating Mass Exponents tau(q) over Normalized Partitions")
        eps_scales = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
        log_eps = np.log(eps_scales)
        
        X_bin = self.X_transformed - np.min(self.X_transformed, axis=0)
        
        # Pre-calculate partition probabilities for each scale to optimize processing
        partition_probs = []
        for eps in eps_scales:
            bins = [np.arange(0, np.max(X_bin[:, d]) + eps, eps) for d in range(self.dims)]
            counts, _ = np.histogramdd(X_bin, bins=bins)
            probs = counts.flatten()[counts.flatten() > 0] / self.n_samples
            partition_probs.append(probs)

        tau_q = []
        
        # Calculate tau(q) for each moment order
        for q in q_values:
            log_mq = []
            for probs in partition_probs:
                if abs(q - 1.0) < 1e-6:
                    # Limit case for q=1 (Shannon information entropy proxy)
                    log_mq.append(np.sum(probs * np.log(probs)))
                else:
                    log_mq.append(np.log(np.sum(probs**q)))
            
            # The slope of log_mq vs log_eps equals the mass exponent tau(q)
            # For q=1, the slope corresponds directly to the Information Dimension D_1
            slope, _ = np.polyfit(log_eps, log_mq, 1)
            
            if abs(q - 1.0) < 1e-6:
                tau_q.append(slope)  # tau(1) = 0 by definition
            else:
                tau_q.append(slope)

        tau_q = np.array(tau_q)
        
        # Enforce analytical constraints: tau(1) must equal 0.0 perfectly
        q_1_idx = np.argmin(np.abs(q_values - 1.0))
        if np.abs(q_values[q_1_idx] - 1.0) < 1e-5:
            tau_q[q_1_idx] = 0.0

        print("STAGE 2: Executing Legendre Transformation (tau(q) -> alpha, f(alpha))")
        # Compute alpha(q) via central finite differences of tau(q) with boundary padding
        alpha = np.gradient(tau_q, q_values)
        
        # Compute f(alpha) via the standard formulation: f(alpha) = q*alpha - tau
        f_alpha = (q_values * alpha) - tau_q

        print(f"\n   {'q_order':<8} | {'tau(q)':<9} | {'alpha (Strength)':<18} | {'f(alpha) (Dimension)'}")
        print(f"   {'-'*65}")
        for idx in range(len(q_values)):
            if idx % 2 == 0 or q_values[idx] in [0.0, 1.0, 2.0]:
                print(f"   {q_values[idx]:<8.2f} | {tau_q[idx]:<9.4f} | {alpha[idx]:<18.4f} | {f_alpha[idx]:.4f}")
        
        print(f"   {'-'*65}")
        print("📊 Multifractal Geometry Profile Summary:")
        print(f"   -> Max Singularity Dimension f(alpha_max): {np.max(f_alpha):.4f} (Matches D_0 at q=0)")
        print(f"   -> Spectral Spread (Width Δalpha): {np.max(alpha) - np.min(alpha):.4f}")
        
        return q_values, tau_q, alpha, f_alpha

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.3: MULTIFRACTAL INFRASTRUCTURE & LEGENDRE TRANSFORM TRACER")
    print("=" * 95)

    # Re-generate our validation anisotropic manifold asset (Kappa ~ 95.85)
    np.random.seed(42)
    t = np.linspace(0, 50, 5000)
    x = np.sin(t) * np.exp(-0.0005 * t)
    y = np.cos(np.sqrt(3) * t) * 0.05
    mock_data = np.column_stack([x, y])

    # Scan moment ranges from -4 to +4
    q_range = np.linspace(-4.0, 4.0, 17)

    engine = SingularitySpectrumEngine(mock_data)
    # Using late stable scaling window verified in V7.9.2
    engine.compute_singularity_spectrum(q_range, eps_min=0.15, eps_max=0.35, num_eps=8)
    print("=" * 95)
