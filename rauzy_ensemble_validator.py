#!/usr/bin/env python3
"""
rauzy_ensemble_validator.py
==========================
High-performance vectorized version using optimized NumPy operations to compute 
2D rotational harmonic invariants without Python loop bottlenecks.
"""
import os
import json
import numpy as np
from scipy.spatial import cKDTree

def compute_harmonics(points, radii, alpha=2.0, epsilon=1e-6):
    num_points = len(points)
    tree = cKDTree(points)
    max_r = max(radii)
    
    # Extract all pairs as a NumPy array of index pairs
    pairs_idx = np.array(list(tree.query_pairs(max_r)))
    
    # Handle empty edge-case scenario
    if len(pairs_idx) == 0:
        return [(0.0, 0.0) for _ in radii]
        
    # Isolate coordinate arrays using advanced indexing (fully vectorized)
    p1 = points[pairs_idx[:, 0]]
    p2 = points[pairs_idx[:, 1]]
    
    # Compute distances and angles across the global spatial matrix
    delta = p2 - p1
    dist_sq = np.sum(delta**2, axis=1)
    dist = np.sqrt(dist_sq)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    
    # Pre-calculate angular components once for all scales
    exp_2itheta = np.exp(2j * theta)
    exp_4itheta = np.exp(4j * theta)
    
    scale_results = []
    
    # Process scales independently using ultra-fast vector math
    for r in radii:
        # Vectorized Riesz weight calculation across the pair matrix
        weights = (1.0 / ((dist_sq + epsilon) ** (alpha / 2.0))) * np.exp(-dist_sq / (r ** 2))
        
        sum_w = np.sum(weights)
        if sum_w == 0:
            scale_results.append((0.0, 0.0))
            continue
            
        # Compute exact phase magnitudes instantly
        mag_g2 = np.abs(np.sum(weights * exp_2itheta)) / sum_w
        mag_g4 = np.abs(np.sum(weights * exp_4itheta)) / sum_w
        
        scale_results.append((float(mag_g2), float(mag_g4)))
        
    return scale_results

def main():
    print("==============================================================================")
    print("HIGH-PERFORMANCE HARMONIC ENSEMBLE RUNNER (VECOTRIZED)")
    print("==============================================================================")
    
    np.random.seed(42)
    radii = [11.4333, 24.6323, 53.0687, 114.3330]
    num_samples = 2000
    num_mc_runs = 50
    
    # Generate mock anisotropic point set (Simulating raw data field geometry)
    auth_points = np.random.randn(num_samples, 2)
    auth_points[:, 0] *= 1.5  
    
    print("  [➔] Pre-calculating authentic harmonic baselines...")
    auth_metrics = compute_harmonics(auth_points, radii)
    
    # Allocate explicit matrix arrays for processing
    null_g2_matrix = np.zeros((num_mc_runs, len(radii)))
    null_g4_matrix = np.zeros((num_mc_runs, len(radii)))
    
    print(f"  [➔] Simulating occupancy mask null distributions ({num_mc_runs} runs)...")
    for run in range(num_mc_runs):
        if (run + 1) % 10 == 0 or run == 0:
            print(f"      Processing Monte Carlo trial {run + 1}/{num_mc_runs}...")
            
        null_points = np.random.randn(num_samples, 2)
        null_points[:, 0] *= 1.5 
        
        run_metrics = compute_harmonics(null_points, radii)
        for idx, r in enumerate(radii):
            null_g2_matrix[run, idx] = run_metrics[idx][0]
            null_g4_matrix[run, idx] = run_metrics[idx][1]
            
    # Compile output dictionaries
    json_output = {"scale_results": []}
    
    print("-" * 78)
    print(f"{'Scale (r)':>14} | {'g2 Auth':>9} | {'g2 Null μ':>9} | {'Z-Score g2':>10} || {'Z-Score g4':>10}")
    print("-" * 78)
    
    for idx, r in enumerate(radii):
        auth_g2 = auth_metrics[idx][0]
        auth_g4 = auth_metrics[idx][1]
        
        mu_g2 = np.mean(null_g2_matrix[:, idx])
        sigma_g2 = np.std(null_g2_matrix[:, idx])
        z_g2 = (auth_g2 - mu_g2) / sigma_g2 if sigma_g2 > 0 else 0.0
        
        mu_g4 = np.mean(null_g4_matrix[:, idx])
        sigma_g4 = np.std(null_g4_matrix[:, idx])
        z_g4 = (auth_g4 - mu_g4) / sigma_g4 if sigma_g4 > 0 else 0.0
        
        print(f"{r:14.4f} | {auth_g2:9.4f} | {mu_g2:9.4f} | {z_g2:10.3f} || {z_g4:10.3f}")
        
        json_output["scale_results"].append({
            "radius": r,
            "g2_stat": {
                "authentic": auth_g2,
                "null_mean": mu_g2,
                "null_sigma": sigma_g2,
                "z_score": z_g2
            },
            "g4_stat": {
                "authentic": auth_g4,
                "null_mean": mu_g4,
                "null_sigma": sigma_g4,
                "z_score": z_g4
            }
        })
        
    os.makedirs('data', exist_ok=True)
    with open('data/ensemble_statistical_metrics.json', 'w') as f:
        json.dump(json_output, f, indent=4)
        
    print("==============================================================================")
    print("[✓] Optimization successful. Analytical logs saved to json registry.")

if __name__ == '__main__':
    main()
