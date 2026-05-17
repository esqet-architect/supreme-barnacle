#!/usr/bin/env python3
"""
MINERVA V7.9.7: ADAPTIVE K-D TREE MULTIFRACTAL ESTIMATION ENGINE
================================================================================
- Replaces rigid Cartesian histogram grids with a hierarchical spatial tree.
- Uses adaptive cell bounding diameters as the dynamic scale variable (epsilon).
- Combines Chhabra-Jensen direct escort dynamics with bootstrap uncertainty loops.
"""

import numpy as np
from scipy.linalg import pinvh, cholesky
import warnings
warnings.filterwarnings('ignore')

class KDTreeNode:
    """Represents an adaptive bounding region in our multi-scale point cloud."""
    def __init__(self, points, depth=0, max_depth=6):
        self.points = points
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = True
        
        n_samples, dims = points.shape
        if n_samples == 0:
            self.bbox_diameter = 0.0
            return
            
        # Compute the spatial diameter (bounding box scale) of this specific cell
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        self.bbox_diameter = np.sqrt(np.sum((max_bounds - min_bounds) ** 2))
        
        # Recursively split the cell if we haven't hit the maximum target depth
        if depth < max_depth and n_samples > 10:
            self.is_leaf = False
            split_axis = depth % dims
            # Sort along the current alternating axis to find the median split point
            sorted_idx = np.argsort(points[:, split_axis])
            sorted_points = points[sorted_idx]
            mid = n_samples // 2
            
            self.left = KDTreeNode(sorted_points[:mid], depth + 1, max_depth)
            self.right = KDTreeNode(sorted_points[mid:], depth + 1, max_depth)

def collect_tree_level_statistics(node, target_depth, current_cells=None):
    """Traverses the tree to collect all bounding cells at a target hierarchical scale."""
    if current_cells is None:
        current_cells = []
    if node is None or len(node.points) == 0:
        return current_cells
        
    if node.depth == target_depth or (node.is_leaf and node.depth <= target_depth):
        current_cells.append(node)
        return current_cells
        
    collect_tree_level_statistics(node.left, target_depth, current_cells)
    collect_tree_level_statistics(node.right, target_depth, current_cells)
    return current_cells

class AdvancedMultifractalEngine:
    def __init__(self, coordinates):
        self.X = np.array(coordinates, dtype=float)
        self.n_samples, self.dims = self.X.shape
        self.X_transformed = self._apply_metric_tensor()

    def _apply_metric_tensor(self):
        """Applies global covariance whitening to handle manifold structural eccentricity."""
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

    def _run_tree_cj_pass(self, data_matrix, q_values, max_depth=6):
        """Builds an adaptive tree and resolves direct escort limits across depth levels."""
        root = KDTreeNode(data_matrix, depth=0, max_depth=max_depth)
        
        # Gather characteristics across tree depth levels (skipping top root layers)
        active_depths = range(2, max_depth + 1)
        effective_scales = []
        level_probabilities = []
        
        for d in active_depths:
            cells = collect_tree_level_statistics(root, d, current_cells=[])
            if len(cells) == 0:
                continue
                
            # Compute mass counts and scale characteristics for this level
            counts = np.array([len(c.points) for c in cells], dtype=float)
            diameters = np.array([c.bbox_diameter for c in cells if c.bbox_diameter > 0])
            
            if len(diameters) == 0:
                continue
                
            # The dynamic scale epsilon is the average size of the active sub-cells
            mean_epsilon = np.mean(diameters)
            probs = counts / np.sum(counts)
            
            effective_scales.append(mean_epsilon)
            level_probabilities.append(probs)

        log_eps = np.log(effective_scales)
        alpha_pass = []
        f_pass = []

        # Run Chhabra-Jensen direct regression profiles over the adaptive layers
        for q in q_values:
            num_alpha = []
            num_f = []
            for probs in level_probabilities:
                Z_q = np.sum(probs**q)
                mu = (probs**q) / Z_q
                
                num_alpha.append(np.sum(mu * np.log(probs)))
                num_f.append(np.sum(mu * np.log(mu)))
                
            alpha_pass.append(np.polyfit(log_eps, num_alpha, 1)[0])
            f_pass.append(np.polyfit(log_eps, num_f, 1)[0])
            
        return np.array(alpha_pass), np.array(f_pass)

    def execute_adaptive_bootstrap(self, q_values, B=15, max_depth=6):
        """Evaluates structural variance limits via bootstrap sweeps over the k-d tree."""
        print(f"STAGE 1: Building Hierarchical k-d Trees Across {B} Bootstrap Loops...")
        
        all_alpha = np.zeros((B, len(q_values)))
        all_f = np.zeros((B, len(q_values)))

        for b in range(B):
            indices = np.random.choice(self.n_samples, size=self.n_samples, replace=True)
            X_replica = self.X_transformed[indices]
            
            alpha_b, f_b = self._run_tree_cj_pass(X_replica, q_values, max_depth=max_depth)
            all_alpha[b, :] = alpha_b
            all_f[b, :] = f_b

        mean_alpha = np.mean(all_alpha, axis=0)
        std_alpha = np.std(all_alpha, axis=0)
        mean_f = np.mean(all_f, axis=0)
        std_f = np.std(all_f, axis=0)

        print("\nSTAGE 2: Finalizing Adaptive Hierarchy Diagnostic Matrix")
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
            print(f"📊 Adaptive Structural Conclusion:\n   -> Verified Admissible Range: q ∈ [{np.min(stable_qs)}, {np.max(stable_qs)}]")
        else:
            print("📊 Adaptive Structural Conclusion:\n   -> Warning: No stable zone converged.")
            
        return mean_alpha, mean_f

if __name__ == "__main__":
    print("=" * 95)
    print("MINERVA V7.9.7: ADAPTIVE K-D TREE INFERENCE ARCHITECTURE")
    print("=" * 95)

    # Generate a sample filamentary trajectory with high coordinate stretching
    np.random.seed(42)
    t = np.linspace(0, 50, 3000)
    x = np.sin(t) * np.exp(-0.0005 * t)
    y = np.cos(np.sqrt(3) * t) * 0.04
    mock_data = np.column_stack([x, y])

    # Analyze across a standard moment testing spectrum
    q_range = np.linspace(-3.0, 3.0, 13)

    harness = AdvancedMultifractalEngine(mock_data)
    harness.execute_bootstrap_warning = harness.execute_adaptive_bootstrap(q_range, B=15, max_depth=6)
    print("=" * 95)
