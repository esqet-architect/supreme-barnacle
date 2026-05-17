#!/usr/bin/env python3
"""
rauzy_dual_pipeline.py
======================
Implementation of the two valid paths forward for Rauzy Fractal analysis:
Option A: 1D Physical-Line Projection (for Chhabra-Jensen Multifractal analysis)
Option B: 2D Topological Data Analysis (via Ripser Persistent Homology)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Try importing ripser for Option B; handle gracefully if not present
try:
    from ripser import ripser
    from persim import plot_diagrams
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# IMPLEMENTATION MODULES
# ═══════════════════════════════════════════════════════════════════════════

def run_option_a_projection(pts):
    """
    Option A: Projects 2D Rauzy data onto a 1D physical line.
    Uses the classic Tribonacci scaling direction vector to map the points.
    """
    print("\n[OPTION A] Executing 1D Physical-Line Projection...")
    
    # Standard Tribonacci dominant eigenvector projection approximation
    # alpha satisfies t^3 - t^2 - t - 1 = 0 (approx 1.8392)
    alpha = 1.839286755214161
    proj_vector = np.array([1.0 / alpha, 1.0 / (alpha**2)])
    
    # Normalize the direction vector
    proj_vector /= np.linalg.norm(proj_vector)
    
    # Project 2D points to 1D via dot product
    pts_1d = np.dot(pts, proj_vector).reshape(-1, 1)
    
    print(f"  Successfully transformed 2D array shape {pts.shape} -> 1D shape {pts_1d.shape}")
    print(f"  1D Range: [{pts_1d.min():.4f}, {pts_1d.max():.4f}]")
    print("  -> Ready for your validated Chhabra-Jensen 1D engine.")
    return pts_1d


def run_option_b_ripser(pts):
    """
    Option B: Computes 2D Persistent Homology using Ripser.
    Tracks topological features (H0 components and H1 holes) across distance scales.
    """
    print("\n[OPTION B] Executing 2D Persistent Homology (Ripser)...")
    if not RIPSER_AVAILABLE:
        print("  [!] Error: ripser or persim libraries are not installed in this environment.")
        print("      Run: pip install ripser persim")
        return None

    # Subsample points if cloud is massive to keep compute times low
    max_pts = 1200
    if len(pts) > max_pts:
        print(f"  Subsampling point cloud from {len(pts)} to {max_pts} for performance...")
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), max_pts, replace=False)
        pts_sample = pts[idx]
    else:
        pts_sample = pts

    # Compute persistence diagrams up to dimension 1 (H0 and H1)
    diagrams = ripser(pts_sample, maxdim=1)['dgms']
    
    print("  Persistence calculation complete.")
    print(f"  Detected H0 features: {len(diagrams[0])}")
    print(f"  Detected H1 features: {len(diagrams[1])}")
    
    # Output the longest-lived structural voids (H1)
    if len(diagrams[1]) > 0:
        lifespans = diagrams[1][:, 1] - diagrams[1][:, 0]
        max_idx = np.argsort(lifespans)[::-1]
        print("\n  Top 3 Longest-Surviving H1 Cavities (Birth, Death, Lifespan):")
        for i in max_idx[:3]:
            if i < len(diagrams[1]):
                b, d = diagrams[1][i]
                print(f"    Feature {i}: Birth={b:.4f}, Death={d:.4f}, Lifespan={d-b:.4f}")
                
    return diagrams

# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    path = 'data/rauzy_points.npy'
    if not os.path.exists(path):
        print(f"Error: Target data file not found at {path}")
        return

    pts = np.load(path)
    print("=" * 65)
    print("RAUZY ANALYSIS: POST-VALIDATION ANALYSIS PIPELINE")
    print("=" * 65)
    print(f"Loaded Point Cloud Size: {pts.shape[0]} points in {pts.shape[1]}D")

    # Run Option A
    pts_1d = run_option_a_projection(pts)
    
    # Run Option B
    diagrams = run_option_b_ripser(pts)
    
    print("\n" + "=" * 65)
    print("Pipeline executed successfully.")
    print("=" * 65)

if __name__ == '__main__':
    main()
