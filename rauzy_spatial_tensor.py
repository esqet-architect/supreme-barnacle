#!/usr/bin/env python3
"""
rauzy_spatial_tensor.py
=======================
Computes a true 2D spatial correlation tensor across point pairs to evaluate 
structural anisotropy without using 1D projection or grid binning.
"""

import os
import time
import json
import numpy as np

class SpatialCorrelationTensor:
    """Computes rotation-invariant geometric anisotropy using 2D pairwise tensors."""
    
    def __init__(self, data_path='data/rauzy_points.npy'):
        self.data_path = data_path
        self.points = None

    def load_data(self, max_samples=2000):
        """Loads point cloud and applies stable sub-sampling for performance optimization."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing dataset at {self.data_path}")
        
        full_pts = np.load(self.data_path)
        
        # Sub-sample deterministically to optimize O(N^2) pairwise calculation
        if len(full_pts) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(full_pts), size=max_samples, replace=False)
            self.points = full_pts[idx]
        else:
            self.points = full_pts
            
        return len(self.points)

    def compute_tensor(self, r_max):
        """
        Builds the 2D spatial correlation tensor from point pairs within distance r_max.
        Tensor components:
            T = [[Σ(Δx*Δx), Σ(Δx*Δy)],
                 [Σ(Δy*Δx), Σ(Δy*Δy)]]
        """
        pts = self.points
        n = len(pts)
        
        T = np.zeros((2, 2))
        pair_count = 0
        
        # Vectorized processing per point to extract local displacement vectors
        for i in range(n):
            # Compute distance vectors from point i to all subsequent points
            diffs = pts[i+1:] - pts[i]
            distances = np.linalg.norm(diffs, axis=1)
            
            # Filter pairs that reside within our target spatial neighborhood scale
            valid_diffs = diffs[distances <= r_max]
            pair_count += len(valid_diffs)
            
            if len(valid_diffs) > 0:
                # Accumulate outer product matrix components
                T[0, 0] += np.sum(valid_diffs[:, 0] ** 2)
                T[0, 1] += np.sum(valid_diffs[:, 0] * valid_diffs[:, 1])
                T[1, 1] += np.sum(valid_diffs[:, 1] ** 2)
                
        T[1, 0] = T[0, 1] # Symmetry condition
        
        if pair_count > 0:
            T /= pair_count # Normalize by total tracked pairs
            
        return T, pair_count

    def analyze_geometry(self):
        """Calculates tensor invariants, eigenvalues, and structural anisotropy ratios."""
        n_points = self.load_data()
        
        # Establish radius scale based on the system's global variance dimension
        global_std = np.mean(np.std(self.points, axis=0))
        target_radius = 0.25 * global_std
        
        start_time = time.time()
        tensor, pairs = self.compute_tensor(target_radius)
        duration = time.time() - start_time
        
        # Compute eigenvalues representing the principal geometric axes
        eigenvalues, _ = np.linalg.eigh(tensor)
        lambda_max = eigenvalues[1]
        lambda_min = eigenvalues[0]
        
        # Compute structural anisotropy ratio (invariant indicator)
        anisotropy_ratio = lambda_max / lambda_min if lambda_min > 0 else 1.0
        
        print("=" * 70)
        print("2D SPATIAL CORRELATION TENSOR REPORT")
        print("=" * 70)
        print(f"  Sample Sub-matrix : {n_points} points")
        print(f"  Neighborhood Scale: radius <= {target_radius:.4f}")
        print(f"  Analyzed Pairs    : {pairs} spatial relations identified")
        print(f"  Processing Speed  : {duration:.4f} seconds")
        print("-" * 70)
        print(f"  Tensor Matrix Components:")
        print(f"    [ {tensor[0,0]:10.6f}  {tensor[0,1]:10.6f} ]")
        print(f"    [ {tensor[1,0]:10.6f}  {tensor[1,1]:10.6f} ]")
        print("-" * 70)
        print(f"  Principal Eigenvalues (Axes) : Max = {lambda_max:.6f} | Min = {lambda_min:.6f}")
        print(f"  Invariant Anisotropy Ratio (R)  : {anisotropy_ratio:.4f}")
        print("=" * 70)
        
        # Save structural results to log files
        out_log = {
            "spatial_radius": target_radius,
            "tracked_pairs": pairs,
            "tensor_matrix": tensor.tolist(),
            "eigenvalues": [lambda_min, lambda_max],
            "anisotropy_ratio": anisotropy_ratio
        }
        with open('data/spatial_tensor_metrics.json', 'w') as f:
            json.dump(out_log, f, indent=4)

if __name__ == '__main__':
    analyzer = SpatialCorrelationTensor()
    analyzer.analyze_geometry()
