#!/usr/bin/env python3
"""
rauzy_spatial_tensor_kdtree.py
==============================
Implements KDTree-accelerated O(N log N) pair extraction and evaluates the 
canonical invariant anisotropy index A(r) on the target spatial point cloud.
"""

import os
import time
import json
import numpy as np
from scipy.spatial import cKDTree

class OptimizedStructureTensor:
    """Computes accelerated rotation-invariant anisotropy using spatial tree indexing."""
    
    def __init__(self, data_path='data/rauzy_points.npy'):
        self.data_path = data_path
        self.points = None

    def load_data(self):
        """Loads the full dataset without requiring sub-sampling limitations."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing source file at: {self.data_path}")
        self.points = np.load(self.data_path)
        return len(self.points)

    def compute_canonical_anisotropy(self, radius):
        """
        Uses a cKDTree index to prune pairs, accumulates the structure matrix,
        and derives the trace-normalized canonical anisotropy invariant index.
        """
        n_points = self.load_data()
        
        # Step 1: Initialize spatial index tree
        tree_start = time.time()
        spatial_tree = cKDTree(self.points)
        
        # Step 2: Prune candidate interactions to O(N log N) using neighborhood query
        pairs = spatial_tree.query_pairs(r=radius, output_type='ndarray')
        tree_duration = time.time() - tree_start
        
        pair_count = len(pairs)
        if pair_count == 0:
            return 0.0, pair_count, tree_duration, 0.0
            
        # Step 3: Vectorized extraction of coordinate displacement fields
        calc_start = time.time()
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        
        diffs = self.points[idx_j] - self.points[idx_i]
        
        # Step 4: Assemble outer product components
        T = np.zeros((2, 2))
        T[0, 0] = np.sum(diffs[:, 0] ** 2)
        T[0, 1] = np.sum(diffs[:, 0] * diffs[:, 1])
        T[1, 0] = T[0, 1]  # Enforce structural symmetry
        T[1, 1] = np.sum(diffs[:, 1] ** 2)
        
        # Normalize tensor by total pairs analyzed
        T /= pair_count
        
        # Step 5: Extract Canonical Geometric Invariants
        det_T = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
        trace_T = T[0, 0] + T[1, 1]
        
        if trace_T > 0:
            # Evaluate: A(r) = sqrt(1 - (4 * det) / (trace^2))
            val = 1.0 - (4.0 * det_T) / (trace_T ** 2)
            # Clip numerical precision residuals near boundaries
            anisotropy_index = np.sqrt(max(0.0, val))
        else:
            anisotropy_index = 0.0
            
        calc_duration = time.time() - calc_start
        return anisotropy_index, pair_count, tree_duration, calc_duration

    def execute_analysis(self):
        """Orchestrates calculation loop and displays high-precision performance metrics."""
        # Establish testing radius at 25% of the coordinate field standard deviation
        self.load_data()
        global_scale = np.mean(np.std(self.points, axis=0))
        target_radius = 0.25 * global_scale
        
        A_r, total_pairs, t_tree, t_calc = self.compute_canonical_anisotropy(target_radius)
        
        print("=" * 75)
        print("KDTREE-ACCELERATED CANONICAL STRUCTURE TENSOR")
        print("=" * 75)
        print(f"  Total Field Size    : {len(self.points)} Points (No Down-sampling)")
        print(f"  Evaluation Scale (r): {target_radius:.5f}")
        print(f"  Valid Spatial Pairs : {total_pairs} identified interactions")
        print("-" * 75)
        print(f"  Spatial Indexing Runtime  : {t_tree:.5f} seconds")
        print(f"  Vectorized Tensor Runtime : {t_calc:.5f} seconds")
        print(f"  Total Processing Lifetime : {t_tree + t_calc:.5f} seconds")
        print("-" * 75)
        print(f"  CANONICAL INVARIANT ANISOTROPY INDEX A(r) : {A_r:.6f}")
        print("=" * 75)
        
        # Preserve results to JSON storage layer
        output_metrics = {
            "parameters": {"scale_r": target_radius, "points": len(self.points)},
            "performance": {"indexing_time": t_tree, "calculation_time": t_calc},
            "invariants": {"canonical_anisotropy": A_r, "pair_count": total_pairs}
        }
        with open('data/kdtree_tensor_metrics.json', 'w') as f:
            json.dump(output_metrics, f, indent=4)

if __name__ == '__main__':
    runner = OptimizedStructureTensor()
    runner.execute_analysis()
