#!/usr/bin/env python3
"""
MINERVA V7.8 (DIAGNOSTIC ADD-ON): SPATIAL DENSITY & MANIFOLD OCCUPANCY ANALYSIS
================================================================================
- Measures micro-state coordinate density within closed stable manifolds.
- Computes Shannon Information Entropy to detect geometric sparse occupancy.
- Operates on the validated complex-pair preserving coordinate outputs.
"""

import numpy as np

def analyze_manifold_occupancy(intrinsic_points, grid_res=50):
    """
    Analyzes the uniform or sparse distribution of coordinates within the stable manifold.
    Supports both 1D (Fibonacci) and 2D (Tribonacci) intrinsic spaces.
    """
    n_pts, dims = intrinsic_points.shape
    print(f"\n📊 Running Micro-State Density Analysis Across {n_pts} Points ({dims}D Intrinsic Space)...")
    print("─" * 85)
    
    if dims == 1:
        # 1D Manifold Density (e.g., Fibonacci, Pisot Binary)
        counts, bin_edges = np.histogram(intrinsic_points[:, 0], bins=grid_res, density=True)
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(grid_res)
        sparsity_ratio = 1.0 - (len(probabilities) / grid_res)
        
        print(f"   -> Active Grid Cells: {len(probabilities)} / {grid_res}")
        print(f"   -> Structural Sparsity Ratio: {sparsity_ratio:.4f}")
        print(f"   -> Shannon Entropy: {shannon_entropy:.4f} bits (Max Possible: {max_entropy:.4f} bits)")
        
    elif dims == 2:
        # 2D Manifold Density (e.g., Tribonacci Rauzy Plane)
        counts, x_edges, y_edges = np.histogram2d(
            intrinsic_points[:, 0], intrinsic_points[:, 1], bins=grid_res
        )
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]
        shannon_entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(grid_res ** 2)
        sparsity_ratio = 1.0 - (len(probabilities) / (grid_res ** 2))
        
        print(f"   -> Active Grid Cells: {len(probabilities)} / {grid_res ** 2}")
        print(f"   -> Structural Sparsity Ratio: {sparsity_ratio:.4f}")
        print(f"   -> Shannon Entropy: {shannon_entropy:.4f} bits (Max Possible: {max_entropy:.4f} bits)")
    else:
        print(f"   [Notice] Density grid analysis omitted for high-dimensional space ({dims}D).")
        shannon_entropy, sparsity_ratio = 0.0, 0.0
        
    return shannon_entropy, sparsity_ratio

# Demonstration of execution using a mock closed 2D circular/toroidal trajectory
if __name__ == "__main__":
    # Simulating a mock 2D closed compact dataset to verify the analysis profile
    np.random.seed(42)
    mock_theta = np.linspace(0, 100, 1000)
    mock_intrinsic = np.column_stack([np.cos(mock_theta), np.sin(mock_theta)])
    
    entropy, sparsity = analyze_manifold_occupancy(mock_intrinsic, grid_res=20)
    print("─" * 85)
    print("Verification Script Loaded Successfully. Ready for full integration pipeline.")
