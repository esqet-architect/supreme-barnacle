#!/usr/bin/env python3
"""
rauzy_rotation_verifier.py
==========================
Explicitly verifies the rotational invariance property g_m(R_phi * X) = g_m(X)
by analyzing harmonic stability across a continuous angular transformation spectrum.
"""
import numpy as np
from scipy.spatial import cKDTree

def compute_harmonics_vectorized(points, radii, alpha=2.0, epsilon=1e-6):
    """Vectorized core harmonic extraction engine."""
    tree = cKDTree(points)
    max_r = max(radii)
    pairs_idx = np.array(list(tree.query_pairs(max_r)))
    
    if len(pairs_idx) == 0:
        return [(0.0, 0.0) for _ in radii]
        
    p1 = points[pairs_idx[:, 0]]
    p2 = points[pairs_idx[:, 1]]
    
    delta = p2 - p1
    dist_sq = np.sum(delta**2, axis=1)
    theta = np.arctan2(delta[:, 1], delta[:, 0])
    
    exp_2itheta = np.exp(2j * theta)
    exp_4itheta = np.exp(4j * theta)
    
    scale_results = []
    for r in radii:
        weights = (1.0 / ((dist_sq + epsilon) ** (alpha / 2.0))) * np.exp(-dist_sq / (r ** 2))
        sum_w = np.sum(weights)
        if sum_w == 0:
            scale_results.append((0.0, 0.0))
            continue
            
        mag_g2 = np.abs(np.sum(weights * exp_2itheta)) / sum_w
        mag_g4 = np.abs(np.sum(weights * exp_4itheta)) / sum_w
        scale_results.append((mag_g2, mag_g4))
        
    return scale_results

def rotate_points(points, angle_deg):
    """Applies a rigid 2D Euclidean rotation matrix to the point cloud spatial array."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return points @ R.T

def main():
    print("==============================================================================")
    print("ROTATIONAL INVARIANCE VERIFICATION SUITE")
    print("==============================================================================")
    
    np.random.seed(42)
    radii = [24.6323, 114.3330] # Representative micro and macro evaluation scales
    num_samples = 1500
    
    # Generate an elongated point cloud mimicking your authentic support geometry
    base_points = np.random.randn(num_samples, 2)
    base_points[:, 0] *= 1.8 
    
    # Define test rotation spectrum
    test_angles = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 180.0]
    
    # Track metrics across the angular transformations
    history_g2 = {r: [] for r in radii}
    history_g4 = {r: [] for r in radii}
    
    print(f"  [➔] Evaluating {len(test_angles)} rigid coordinate rotations...")
    for phi in test_angles:
        rotated_x = rotate_points(base_points, phi)
        metrics = compute_harmonics_vectorized(rotated_x, radii)
        
        for idx, r in enumerate(radii):
            history_g2[r].append(metrics[idx][0])
            history_g4[r].append(metrics[idx][1])
            
    print("\n[📊] INVARIANCE TESTING RESULTS:")
    print("-" * 78)
    print(f"{'Scale (r)':>10} | {'Angle (deg)':>12} | {'g2 Magnitude':>14} | {'g4 Magnitude':>14}")
    print("-" * 78)
    
    for r in radii:
        g2_vals = history_g2[r]
        g4_vals = history_g4[r]
        
        # Calculate maximum variance across all transformations
        max_dev_g2 = np.max(g2_vals) - np.min(g2_vals)
        max_dev_g4 = np.max(g4_vals) - np.min(g4_vals)
        
        for i, phi in enumerate(test_angles):
            print(f"{r:10.4f} | {phi:12.1f}° | {g2_vals[i]:14.8f} | {g4_vals[i]:14.8f}")
            
        print(f"  ➔ Max Delta Scale {r:.2f}: Δg2 = {max_dev_g2:.2e} | Δg4 = {max_dev_g4:.2e}")
        
        # Invariance Assertion Checks
        if max_dev_g2 < 1e-12 and max_dev_g4 < 1e-12:
            print(r"  [✓] MATHEMATICAL INVARIANCE PROVEN FOR SCALE r=" + f"{r}")
        else:
            print(r"  [💥] FAILURE: Leakage detected at scale r=" + f"{r}")
        print("-" * 78)

if __name__ == '__main__':
    main()
