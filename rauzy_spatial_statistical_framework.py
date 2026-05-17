#!/usr/bin/env python3
"""
rauzy_spatial_statistical_framework.py
======================================
Implements an anisotropic pair correlation harmonic estimator g_m(r) and 
validates results against a support-preserving Convex Hull CSR null model.
"""

import os
import time
import json
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from matplotlib.path import Path

# ═══════════════════════════════════════════════════════════════════════════
# 1. NULL MODELS MODULE (Support-Preserving Geometry)
# ═══════════════════════════════════════════════════════════════════════════

def generate_convex_hull_csr(authentic_points, num_samples=2000):
    """Generates a random uniform point field constrained within the authentic Convex Hull."""
    hull = ConvexHull(authentic_points)
    hull_path = Path(authentic_points[hull.vertices])
    
    # Bounding box limits
    min_x, min_y = authentic_points.min(axis=0)
    max_x, max_y = authentic_points.max(axis=0)
    
    rng = np.random.default_rng(42)
    csr_points = []
    
    # Rejection sampling loop to enforce hard boundary support constraints
    while len(csr_points) < num_samples:
        candidate_batch = rng.uniform([min_x, min_y], [max_x, max_y], size=(num_samples * 2, 2))
        inside = hull_path.contains_points(candidate_batch)
        valid_candidates = candidate_batch[inside]
        
        csr_points.extend(valid_candidates)
        
    return np.array(csr_points[:num_samples])

# ═══════════════════════════════════════════════════════════════════════════
# 2. HARMONICS MODULE (Anisotropic Spatial Statistics)
# ═══════════════════════════════════════════════════════════════════════════

def compute_harmonic_correlations(points, radius_scales, alpha=2, eps=1e-6):
    """Decomposes the anisotropic pair correlation field into g2(r) and g4(r) harmonics."""
    tree = cKDTree(points)
    n = len(points)
    
    harmonic_profile = []
    
    for r in radius_scales:
        # Fast spatial neighborhood filtering via tree index
        pairs = tree.query_pairs(r=r, output_type='ndarray')
        pair_count = len(pairs)
        
        if pair_count == 0:
            harmonic_profile.append({"radius": r, "g2": 0.0, "g4": 0.0})
            continue
            
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]
        diffs = points[idx_j] - points[idx_i]
        
        d2 = np.sum(diffs**2, axis=1)
        
        # Continuous Riesz-style scale local kernel weighting
        w = 1.0 / ((d2 + eps)**(alpha / 2.0)) * np.exp(-d2 / (r**2))
        W = np.sum(w)
        
        if W > 0:
            # Extract spatial pair displacement angles
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            
            # Compute discrete realizations of the harmonic field fields
            g2_coef = np.abs(np.sum(w * np.exp(2j * angles))) / W
            g4_coef = np.abs(np.sum(w * np.exp(4j * angles))) / W
        else:
            g2_coef, g4_coef = 0.0, 0.0
            
        harmonic_profile.append({
            "radius": r,
            "g2": g2_coef,
            "g4": g4_coef
        })
        
    return harmonic_profile

# ═══════════════════════════════════════════════════════════════════════════
# 3. PIPELINE ORCHESTRATION LAYER
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Target data missing at {data_path}")
        return

    # Load data and instantiate stable evaluation configurations
    full_pts = np.load(data_path)
    sample_size = min(2000, len(full_pts))
    
    rng = np.random.default_rng(42)
    auth_sample = rng.choice(full_pts, size=sample_size, replace=False)
    
    print("=" * 78)
    print("HARMONIC ANISOTROPIC PAIR CORRELATION ENGINE")
    print("=" * 78)
    print(f"  Authentic Dataset Size: {len(full_pts)} points (Sample: {sample_size})")
    
    # Generate the boundary shape-preserving spatial control model
    print("  [➔] Compiling support-preserving Convex Hull CSR model...")
    null_sample = generate_convex_hull_csr(auth_sample, num_samples=sample_size)
    
    # Establish radial evaluation steps using system geometry limits
    global_std = np.mean(np.std(auth_sample, axis=0))
    test_radii = np.geomspace(0.05 * global_std, 0.8 * global_std, 5)
    
    # Run analysis across both fields
    print("  [➔] Processing harmonic decomposition over scales...")
    auth_harmonics = compute_harmonic_correlations(auth_sample, test_radii)
    null_harmonics = compute_harmonic_correlations(null_sample, test_radii)
    
    print("-" * 78)
    print(f"  {'Scale (r)':>10} | {'Auth |g2|':>12} | {'Null |g2|':>12} || {'Auth |g4|':>12} | {'Null |g4|':>12}")
    print("-" * 78)
    
    comparison_log = []
    for a, n in zip(auth_harmonics, null_harmonics):
        print(f"  {a['radius']:10.4f} | {a['g2']:12.4f} | {n['g2']:12.4f} || {a['g4']:12.4f} | {n['g4']:12.4f}")
        comparison_log.append({
            "radius": a['radius'],
            "authentic": {"g2": a['g2'], "g4": a['g4']},
            "null_csr": {"g2": n['g2'], "g4": n['g4']}
        })
        
    print("=" * 78)
    
    # Write archival log block to output directories
    os.makedirs('data', exist_ok=True)
    with open('data/harmonic_correlation_metrics.json', 'w') as f:
        json.dump({"scale_sweeps": comparison_log}, f, indent=4)
    print("  [✓] Complete scientific validation log preserved to json storage layer.")

if __name__ == '__main__':
    main()
