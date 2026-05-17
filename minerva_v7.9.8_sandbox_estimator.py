#!/usr/bin/env python3
"""
MINERVA V7.9.8: FIXED-RADIUS SANDBOX MULTIFRACTAL ESTIMATOR
================================================================================
- Bypasses box boundaries and tree smoothing via center-focused radial spheres.
- Preserves heterogeneous density profiles without equal-mass homogenization.
- Integrates Chhabra-Jensen direct escort dynamics with bootstrap verification loops.
"""

import numpy as np
from scipy.linalg import pinvh, cholesky
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

class SandboxMultifractalEngine:
    def __init__(self, coordinates):
        self.X = np.array(coordinates, dtype=float)
        self.n_samples, self.dims = self.X.shape
        self.X_transformed = self._apply_metric_tensor()

    def _apply_metric_tensor(self):
        """Applies global covariance whitening to isolate manifold anisotropy."""
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

    def _run_sandbox_cj_pass(self, data_matrix, q_values, num_centers=150, r_min_ratio=0.05, r_max_ratio=0.35, num_r=10):
        """Computes direct Chhabra-Jensen metrics using concentric mass spheres around random centers."""
        # Compute maximum spatial scale of the transformed set
        min_bounds = np.min(data_matrix, axis=0)
        max_bounds = np.max(data_matrix, axis=0)
        max_diameter = np.sqrt(np.sum((max_bounds - min_bounds) ** 2))
        
        radii = np.logspace(np.log10(r_min_ratio * max_diameter), np.log10(r_max_ratio * max_diameter), num_r)
        log_r = np.log(radii)
        
        # Select random center points from the trajectory
        center_indices = np.random.choice(len(data_matrix), size=num_centers, replace=False)
        centers = data_matrix[center_indices]
        
        # Calculate distance matrix from centers to all points
        dists = cdist(centers, data_matrix, metric='euclidean')
        
        # Build radial mass probabilities p_j(R)
        radial_probabilities = []
        for r in radii:
            counts_per_center = np.sum(dists <= r, axis=1)
            # Clip to prevent zero probabilities in extremely sparse regions
            counts_per_center = np.clip(counts_per_center, 1, None)
            probs = counts_per_center / len(data_matrix)
            radial_probabilities.append(probs)

        alpha_pass = []
        f_pass = []

        # Extract scaling exponents using direct escort distributions
        for q in q_values:
            num_alpha = []
            num_f = []
            for probs in radial_probabilities:
                Z_q = np.sum(probs**q)
                mu = (probs**q) / Z_q
                
                num_alpha.append(np.sum(mu * np.log(probs)))
                num_f.append(np.sum(mu * np.log(mu)))
                
            # Slopes against log(r) provide alpha(q) and f(q) values directly
            alpha_pass.append(np.polyfit(log_r, num_alpha, 1)[0])
            f_pass.append(np.polyfit(log_r, num_f, 1)[0])
            
        return np.array(alpha_pass), np.array(f_pass)

    def execute_sandbox_bootstrap(self, q_values, B=15, num_centers=150):
        """Runs validation bootstrap loops over sandbox radial distributions."""
        print(f"STAGE 1: Growing Concentric Sandbox Spheres Across {B} Bootstrap Loops...")
        
        all_alpha = np.zeros((B, len(q_values)))
        all_f = np.zeros((B, len(q_values)))

        for b in range(B):
            indices = np.random.choice(self.n_samples, size=self.n_samples, replace=True)
            X_replica = self.X_transformed[indices]
            
            alpha_b, f_b = self._run_sandbox_cj_pass(X_replica, q_values, num_centers=num_centers)
            all_alpha[b, :] = alpha_b
            all_f[b, :] = f_b

        mean_alpha = np.mean(all_alpha, axis=0)
        std_alpha = np.std(all_alpha, axis=0)
        mean_f = np.mean(all_f, axis=0)
        std_f = np.std(all_f, axis=0)

        print("\nSTAGE 2: Finalizing Sandbox Geometry Diagnostic Matrix")
        print(f"\n   {'q_order':<8} | {'alpha (Mean)':<14} | {'σ_alpha':<8} | {'f(alpha) (Mean)':<15} | {'σ_f':<8} | Status")
        print(f"   {'-'*78}")
        
        for idx in range(len(q_values)):
            if std_f[idx] > 0.08 or mean_f[idx] > self.dims:
                status = "❌ UNSTABLE"
            else:
                status = "✅ STABLE"
            print(f"   {q_values[idx]:<8.2f} | {mean_alpha[idx]:<14.4f} | {std_alpha[idx]:<8.4f} | {mean_f[idx]:<15.4f} | {std_f[idx]:<8.4f} | {status}")
            
        print(f"   {'-'*78}")
        
        stable_qs = q_values[(std_f <= 0.08) & (mean_f <= self.dims)]
        if len(stable_qs) > 0:
            print(f"📊 Sandbox Structural Conclusion:\n   -> Admissible Range: q ∈ [{np.min(stable_qs)}, {np.max(stable_qs)}]")
            print(f"   -> Extracted Central Dimension D_0 [q=0]: {mean_f[np.abs(q_values).argmin()]:.4f}")
            print(f"   -> Recovered Spectrum Width Δα: {np.max(mean_alpha) - np.min(mean_alpha):.4f}")
        else:
            print("📊 Sandbox Structural Conclusion:\n   -> Warning: No stable convergence zone identified.")
            
        return mean_alpha, mean_f

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.8: SANDBOX SPATIAL SAMPLING ENGINE")
    print("=" * 95)

    # Recreate the highly anisotropic target filament trajectory
    np.random.seed(42)
    t = np.linspace(0, 50, 3000)
    x = np.sin(t) * np.exp(-0.0005 * t)
    y = np.cos(np.sqrt(3) * t) * 0.04
    mock_data = np.column_stack([x, y])

    # Scan the multifractal testing spectrum
    q_range = np.linspace(-3.0, 3.0, 13)

    harness = SandboxMultifractalEngine(mock_data)
    harness.execute_sandbox_bootstrap(q_range, B=15, num_centers=150)
    print("=" * 95)
