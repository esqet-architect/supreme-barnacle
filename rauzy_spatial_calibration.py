#!/usr/bin/env python3
"""
rauzy_spatial_calibration.py
============================
Implements a corrected geometric guard region filter and generates synthetic 
baseline calibration datasets to verify unconstrained spatial invariants.
"""

import os
import json
import numpy as np
from scipy.spatial import cKDTree

# ===========================================================================
# 1. SYNTHETIC CALIBRATION GENERATOR MODULE
# ===========================================================================

def generate_isotropic_disk(num_points=2000, radius=100.0):
    """Generates a perfectly isotropic uniform random point field within a disk."""
    rng = np.random.default_rng(42)
    r_vals = radius * np.sqrt(rng.uniform(0, 1, num_points))
    theta_vals = rng.uniform(0, 2 * np.pi, num_points)

    x = r_vals * np.cos(theta_vals)
    y = r_vals * np.sin(theta_vals)
    return np.column_stack((x, y))

def generate_anisotropic_line(num_points=2000, length=200.0, noise_std=2.0):
    """Generates an extremely directional point process along a 1D linear matrix axis."""
    rng = np.random.default_rng(42)
    linear_axis = rng.uniform(-length / 2, length / 2, num_points)
    perpendicular_noise = rng.normal(0, noise_std, num_points)

    # Orient the line at a fixed 45-degree angle to verify rotation tracking
    x = (linear_axis * np.cos(np.pi / 4)) - (perpendicular_noise * np.sin(np.pi / 4))
    y = (linear_axis * np.sin(np.pi / 4)) + (perpendicular_noise * np.cos(np.pi / 4))
    return np.column_stack((x, y))

# ===========================================================================
# 2. ACCELERATED HARMONIC ESTIMATOR WITH GEOMETRIC GUARD REGION
# ===========================================================================

def compute_harmonics_with_guard(points, radius, alpha=2, eps=1e-6):
    """
    Computes scale-dependent harmonics while applying a geometric guard zone.
    Points within distance 'radius' of the global boundary cannot act as pair origins.
    """
    tree = cKDTree(points)

    # Identify global bounding box boundaries
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)

    # Query all active pairs within search scale radius
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    if len(pairs) == 0:
        return 0.0, 0.0

    idx_i = pairs[:, 0]
    idx_j = pairs[:, 1]

    # Enforce Guard Region: Check bounds clearance for all origin points (i)
    origins = points[idx_i]
    dist_to_min = origins - min_coords
    dist_to_max = max_coords - origins

    # Extract minimum clearance for each coordinate dimension explicitly
    min_clearance_x = np.minimum(dist_to_min[:, 0], dist_to_max[:, 0])
    min_clearance_y = np.minimum(dist_to_min[:, 1], dist_to_max[:, 1])
    min_clearance = np.minimum(min_clearance_x, min_clearance_y)
    
    # Filter out points too close to any boundary edge
    valid_origin_mask = min_clearance >= radius

    filtered_idx_i = idx_i[valid_origin_mask]
    filtered_idx_j = idx_j[valid_origin_mask]

    if len(filtered_idx_i) == 0:
        return 0.0, 0.0

    # Vectorized continuous angular extraction
    diffs = points[filtered_idx_j] - points[filtered_idx_i]
    d2 = np.sum(diffs**2, axis=1)

    w = 1.0 / ((d2 + eps)**(alpha / 2.0)) * np.exp(-d2 / (radius**2))
    W = np.sum(w)

    if W > 0:
        angles = np.arctan2(diffs[:, 1], diffs[:, 0])
        g2 = np.abs(np.sum(w * np.exp(2j * angles))) / W
        g4 = np.abs(np.sum(w * np.exp(4j * angles))) / W
    else:
        g2, g4 = 0.0, 0.0

    return float(g2), float(g4)

# ===========================================================================
# 3. PIPELINE DIAGNOSTIC EXECUTION
# ===========================================================================

def run_calibration_suite():
    """Runs our upgraded guard-region pipeline on both baseline check datasets."""
    n_points = 2000
    test_radius = 15.0

    print("=" * 78)
    print("PIPELINE CALIBRATION SUITE & GEOMETRIC GUARD REGION FILTER")
    print("=" * 78)
    print(f"  Target Sample Fields : {n_points} Points each")
    print(f"  Evaluation Scale (r) : {test_radius:.2f} units (Enforced as Guard Buffer)")
    print("  [➔] Compiling synthetic calibration fields...")

    disk_pts = generate_isotropic_disk(num_points=n_points, radius=100.0)
    line_pts = generate_anisotropic_line(num_points=n_points, length=200.0, noise_std=1.5)

    print("  [➔] Evaluating isotropic control sample (Uniform Disk)...")
    disk_g2, disk_g4 = compute_harmonics_with_guard(disk_pts, test_radius)

    print("  [➔] Evaluating anisotropic control sample (1D Filament Line)...")
    line_g2, line_g4 = compute_harmonics_with_guard(line_pts, test_radius)

    print("-" * 78)
    print(f"  {'Dataset Context':<30} | {'Harmonic |g2|':>15} | {'Harmonic |g4|':>15}")
    print("-" * 78)
    print(f"  {'Uniform Isotropic Disk CSR':<30} | {disk_g2:15.5f} | {disk_g4:15.5f}")
    print(f"  {'Highly Anisotropic Linear Field':<30} | {line_g2:15.5f} | {line_g4:15.5f}")
    print("=" * 78)

    calibration_metrics = {
        "disk_control": {"g2": disk_g2, "g4": disk_g4},
        "line_control": {"g2": line_g2, "g4": line_g4}
    }
    os.makedirs('data', exist_ok=True)
    with open('data/pipeline_calibration_benchmarks.json', 'w') as f:
        json.dump(calibration_metrics, f, indent=4)
    print("  [✓] Spatial calibration logs archived to data folder storage layers.")

if __name__ == '__main__':
    run_calibration_suite()
