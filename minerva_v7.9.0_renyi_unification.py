#!/usr/bin/env python3
"""
MINERVA V7.9.0: UNIFIED RÉNYI PARTITION ENGINE & COMPARATIVE ESTiMATOR
================================================================================
- Strips all ad-hoc additive corrections and manual monotonicity constraints.
- Framework 1: Pure Rényi Partition Engine derived from unified box measures p_i.
- Framework 2: Spatial Correlation Sum (Grassberger-Procaccia) vs kNN Entropy.
- Exposes raw, unvarnished dimensional ordering to evaluate true asymptotics.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class RenyiUnificationHarness:
    def __init__(self, X):
        self.X = np.array(X, dtype=float)
        self.n_samples, self.dims = self.X.shape

    def run_pure_renyi_partition_engine(self, eps_min=0.01, eps_max=0.5, num_eps=10):
        """
        Framework 1: Canonical Rényi Partition Engine
        Derives D0, D1, D2 directly from the unified probability distribution p_i(eps).
        Monotonicity (D0 >= D1 >= D2) is mathematically guaranteed by the Renyi definition.
        """
        print("ENGINE 1: Canonical Rényi Partition Architecture (Fixed-Radius Box)")
        eps_scales = np.logspace(np.log10(eps_min), np.log10(eps_max), num_eps)
        
        log_eps = []
        I_q0, I_q1, I_q2 = [], [], []

        # Shift coordinate space to strictly positive quadrant for clean binning
        X_min = np.min(self.X, axis=0)
        X_shifted = self.X - X_min

        for eps in eps_scales:
            # Generate multi-dimensional histogram bins based on current eps scale
            bins = [np.arange(0, np.max(X_shifted[:, d]) + eps, eps) for d in range(self.dims)]
            counts, _ = np.histogramdd(X_shifted, bins=bins)
            
            # Flatten to extract non-zero probability masses
            counts_flat = counts.flatten()
            nz_counts = counts_flat[counts_flat > 0]
            probs = nz_counts / self.n_samples

            log_eps.append(np.log(eps))
            
            # Q=0: Capacity / Box-Counting
            I_q0.append(np.log(len(probs)))
            
            # Q=1: Information Entropy (Shannon)
            I_q1.append(-np.sum(probs * np.log(probs)))
            
            # Q=2: Correlation (Sum of squares)
            I_q2.append(np.log(np.sum(probs**2)))

        # Regress slopes: I_q ~ (q-1) * D_q * log(eps) [or adjusted for q limits]
        # Slope of log(eps) vs I_q yields the dimension directly
        slope_0, _ = np.polyfit(log_eps, I_q0, 1)
        slope_1, _ = np.polyfit(log_eps, I_q1, 1)
        slope_2, _ = np.polyfit(log_eps, I_q2, 1)

        # In fixed-radius box counting, scaling is proportional to -log(eps)
        D_0 = -float(slope_0)
        D_1 = -float(slope_1)
        D_2 = -float(slope_2) / (2.0 - 1.0) # Generalized scaling factor: / (q - 1)

        print(f"   -> Box Partition D_0: {D_0:.4f}")
        print(f"   -> Box Partition D_1: {D_1:.4f}")
        print(f"   -> Box Partition D_2: {D_2:.4f}")
        print(f"   -> Status: {'✅ Monotonic Hierarchy Verified' if D_0 >= D_1 >= D_2 else '❌ Monotonicity Broken'}")
        return D_0, D_1, D_2

    def run_pure_correlation_sum_engine(self, r_min=0.01, r_max=0.5, num_r=10):
        """
        Framework 2: Canonical Grassberger-Procaccia Correlation Sum Engine
        Computes spatial correlation dimension D2 via exact pairwise distance integration:
        C(r) = (2 / (N*(N-1))) * sum( H(r - ||x_i - x_j||) )
        """
        print("\nENGINE 2: Pure Spatial Correlation Sum Architecture (Grassberger-Procaccia)")
        r_scales = np.logspace(np.log10(r_min), np.log10(r_max), num_r)
        
        # Subsample carefully if dataset size is massive to prevent memory thrashing
        X_sample = self.X if self.n_samples <= 2000 else self.X[np.random.choice(self.n_samples, 2000, replace=False)]
        n_pts = len(X_sample)
        
        # Compute full exact pairwise condensed distance vector
        pairwise_dists = pdist(X_sample, metric='euclidean')
        
        log_r = []
        log_C = []

        for r in r_scales:
            # Count pairs closer than radius r
            pairs_inside = np.sum(pairwise_dists < r)
            correlation_sum = pairs_inside / (0.5 * n_pts * (n_pts - 1))
            
            if correlation_sum > 0:
                log_r.append(np.log(r))
                log_C.append(np.log(correlation_sum))

        if len(log_C) < 3:
            print("   -> Correlation sum saturated or empty. Adjust r window.")
            return np.nan

        D_2_gp, _ = np.polyfit(log_r, log_C, 1)
        print(f"   -> Grassberger-Procaccia D_2: {float(D_2_gp):.4f}")
        return float(D_2_gp)

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.0: RIGOROUS UNCONSTRAINED RÉNYI ESTiMATOR UNIFICATION")
    print("=" * 95)

    # Re-generate the highly anisotropic validation test attractor (N=4000)
    np.random.seed(42)
    t = np.linspace(0, 50, 4000)
    x = np.sin(t) * np.exp(-0.001 * t)
    y = np.cos(np.sqrt(3) * t) * 0.1
    mock_data = np.column_stack([x, y])

    harness = RenyiUnificationHarness(mock_data)
    
    # Run the unconstrained comparative analysis
    harness.run_pure_renyi_partition_engine(eps_min=0.02, eps_max=0.4, num_eps=12)
    harness.run_pure_correlation_sum_engine(r_min=0.02, r_max=0.4, num_r=12)
    print("=" * 95)
