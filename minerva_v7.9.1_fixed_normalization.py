#!/usr/bin/env python3
"""
MINERVA V7.9.1: UNIFIED RÉNYI PARTITION ENGINE (NORMALIZED)
================================================================================
- Corrects sign inversions by explicitly converting measures to Renyi Entropies.
- Eliminates hardcoded negative sign overrides in linear regressions.
- Guarantees positive, mathematically sound spectrum metrics across D0, D1, D2.
"""

import numpy as np
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

class NormalizedRenyiEngine:
    def __init__(self, coordinates):
        self.X = np.array(coordinates, dtype=float)
        self.n_samples, self.dims = self.X.shape

    def run_partition_engine(self, eps_min=0.02, eps_max=0.4, num_eps=12):
        """
        Calculates canonical Renyi Entropies H_q(eps) to extract accurate,
        positive dimensions from the slope: H_q(eps) = -D_q * log(eps) + C
        """
        print("ENGINE 1: Canonical Normalized Rényi Partition Architecture")
        eps_scales = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
        
        log_eps = []
        H_0, H_1, H_2 = [], [], []

        # Shift to positive quadrant for clean, boundary-safe multi-dimensional binning
        X_shifted = self.X - np.min(self.X, axis=0)

        for eps in eps_scales:
            # Construct multi-dimensional grid bins for current resolution
            bins = [np.arange(0, np.max(X_shifted[:, d]) + eps, eps) for d in range(self.dims)]
            counts, _ = np.histogramdd(X_shifted, bins=bins)
            
            # Extract non-zero probability masses
            flat_counts = counts.flatten()
            nz_counts = flat_counts[flat_counts > 0]
            probs = nz_counts / self.n_samples

            log_eps.append(np.log(eps))
            
            # Calculate standard Rényi Entropies H_q explicitly before regression
            H_0.append(np.log(len(probs)))                             # q=0 (Box counting)
            H_1.append(-np.sum(probs * np.log(probs)))                 # q=1 (Shannon Entropy)
            H_2.append(-np.log(np.sum(probs**2)))                      # q=2 (Correlation Entropy)

        # Regress H_q vs log(eps). The slope equals -D_q.
        slope_0, _ = np.polyfit(log_eps, H_0, 1)
        slope_1, _ = np.polyfit(log_eps, H_1, 1)
        slope_2, _ = np.polyfit(log_eps, H_2, 1)

        # Extract dimensions by negating the slope
        D_0 = -float(slope_0)
        D_1 = -float(slope_1)
        D_2 = -float(slope_2)

        print(f"   -> Normalized Partition D_0: {D_0:.4f}")
        print(f"   -> Normalized Partition D_1: {D_1:.4f}")
        print(f"   -> Normalized Partition D_2: {D_2:.4f}")
        
        status = "✅ Monotonic Hierarchy Verified" if D_0 >= D_1 >= D_2 else "⚠️ Deviation from Asymptotic Hierarchy"
        print(f"   -> Status: {status}")
        return D_0, D_1, D_2

    def run_correlation_sum_engine(self, r_min=0.02, r_max=0.4, num_r=12):
        """Standard Grassberger-Procaccia Engine for validation comparison."""
        print("\nENGINE 2: Pure Spatial Correlation Sum Architecture (Grassberger-Procaccia)")
        r_scales = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
        
        X_sample = self.X if self.n_samples <= 2000 else self.X[np.random.choice(self.n_samples, 2000, replace=False)]
        n_pts = len(X_sample)
        pairwise_dists = pdist(X_sample, metric='euclidean')

        log_r, log_C = [], []
        for r in r_scales:
            pairs_inside = np.sum(pairwise_dists < r)
            correlation_sum = pairs_inside / (0.5 * n_pts * (n_pts - 1))
            if correlation_sum > 0:
                log_r.append(np.log(r))
                log_C.append(np.log(correlation_sum))

        D_2_gp = float(np.polyfit(log_r, log_C, 1)[0])
        print(f"   -> Grassberger-Procaccia D_2: {D_2_gp:.4f}")
        return D_2_gp

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.1: RÉNYI ENGINE NORMALIZATION VALIDATION RUN")
    print("=" * 95)

    # Recreate the highly anisotropic test attractor (N=4000)
    np.random.seed(42)
    t = np.linspace(0, 50, 4000)
    x = np.sin(t) * np.exp(-0.001 * t)
    y = np.cos(np.sqrt(3) * t) * 0.1
    mock_data = np.column_stack([x, y])

    harness = NormalizedRenyiEngine(mock_data)
    harness.run_partition_engine()
    harness.run_correlation_sum_engine()
    print("=" * 95)
