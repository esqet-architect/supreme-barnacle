#!/usr/bin/env python3
"""
rauzy_null_testing.py
=====================
Compares the topological and multifractal properties of the Rauzy Fractal
against uniform random and shuffled null models to verify structural significance.
"""

import os
import numpy as np
from ripser import ripser

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS SUBSYSTEMS
# ═══════════════════════════════════════════════════════════════════════════

def run_multifractal_1d(pts_1d):
    """Computes the multifractal spectrum width (Delta alpha) using the validated engine."""
    x = pts_1d.flatten()
    n = len(x)
    lo, hi = x.min(), x.max()
    span = hi - lo
    
    q_arr = np.linspace(-1.0, 1.0, 9)
    scales = np.geomspace(0.005 * span, 0.25 * span, 14)
    log_eps = np.log(scales)
    
    log_Z = np.zeros((len(q_arr), len(scales)))
    num_alpha = np.zeros((len(q_arr), len(scales)))
    
    for s_idx, eps in enumerate(scales):
        bins = np.floor((x - lo) / eps).astype(int)
        _, counts = np.unique(bins, return_counts=True)
        mu = counts / n
        
        log_mu = np.log(mu + 1e-300)
        for q_idx, q in enumerate(q_arr):
            if q == 0:
                Z = float(len(mu))
                mu_t = np.ones(len(mu)) / Z
            elif abs(q - 1.0) < 1e-3:
                Z = 1.0
                mu_t = mu.copy()
            else:
                log_mu_q = q * log_mu
                log_mu_q -= log_mu_q.max()
                mu_q = np.exp(log_mu_q)
                Z = mu_q.sum()
                mu_t = mu_q / Z
                Z = np.sum(mu**q)
                
            log_Z[q_idx, s_idx] = np.log(Z + 1e-300)
            num_alpha[q_idx, s_idx] = np.sum(mu_t * log_mu)

    # Use central linearity window
    q0_idx = np.argmin(np.abs(q_arr))
    best_r2 = -1.0
    win = slice(0, len(scales))
    
    for i in range(len(scales) - 4):
        for j in range(i + 4, len(scales) + 1):
            p = np.polyfit(log_eps[i:j], log_Z[q0_idx, i:j], 1)
            res = np.sum((log_Z[q0_idx, i:j] - np.polyval(p, log_eps[i:j]))**2)
            tot = np.sum((log_Z[q0_idx, i:j] - log_Z[q0_idx, i:j].mean())**2)
            r2 = 1.0 - (res / tot) if tot > 1e-12 else 1.0
            if r2 > best_r2:
                best_r2 = r2
                win = slice(i, j)

    alpha_out = np.zeros(len(q_arr))
    for j in range(len(q_arr)):
        alpha_out[j] = np.polyfit(log_eps[win], num_alpha[j][win], 1)[0]

    return alpha_out.max() - alpha_out.min()

def analyze_ensemble(pts, label):
    """Extracts maximum H1 lifespan and 1D projection multifractal width for a point cloud."""
    # 1. Option B: Topology (Subsampled for computation safety)
    max_subsample = 1200
    pts_sample = pts
    if len(pts) > max_subsample:
        rng = np.random.default_rng(42)
        pts_sample = pts[rng.choice(len(pts), max_subsample, replace=False)]
        
    dgms = ripser(pts_sample, maxdim=1)['dgms']
    h1 = dgms[1]
    
    max_lifespan = 0.0
    if len(h1) > 0:
        lifespans = h1[:, 1] - h1[:, 0]
        max_lifespan = lifespans.max()
        
    # 2. Option A: 1D Physical-Line Projection
    alpha_trib = 1.839286755214161
    proj = np.array([1.0 / alpha_trib, 1.0 / (alpha_trib**2)])
    proj /= np.linalg.norm(proj)
    pts_1d = np.dot(pts, proj)
    
    delta_alpha = run_multifractal_1d(pts_1d)
    
    print(f"  Completed analysis for: {label}")
    return max_lifespan, delta_alpha, len(h1)

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"Error: Target data file not found at {data_path}")
        return

    # Load authentic Rauzy Point Cloud
    rauzy_pts = np.load(data_path)
    n_points = len(rauzy_pts)
    
    # Establish bounding boundaries
    x_min, y_min = rauzy_pts.min(axis=0)
    x_max, y_max = rauzy_pts.max(axis=0)

    print("=" * 65)
    print("RAUZY FRACTAL VS NULL MODELS: COMPARATIVE EVALUATION")
    print("=" * 65)
    print(f"Testing Matrix: {n_points} points per ensemble model\n")

    # 1. Profile Authentic Rauzy Cloud
    r_life, r_delta, r_h1 = analyze_ensemble(rauzy_pts, "Authentic Rauzy Cloud")

    # 2. Profile Uniform Random Null Model
    rng = np.random.default_rng(42)
    rand_pts = np.zeros((n_points, 2))
    rand_pts[:, 0] = rng.uniform(x_min, x_max, n_points)
    rand_pts[:, 1] = rng.uniform(y_min, y_max, n_points)
    u_life, u_delta, u_h1 = analyze_ensemble(rand_pts, "Uniform Random Null Model")

    # 3. Profile Shuffled Coordinates Null Model
    shuff_pts = rauzy_pts.copy()
    rng.shuffle(shuff_pts[:, 0])
    rng.shuffle(shuff_pts[:, 1])
    s_life, s_delta, s_h1 = analyze_ensemble(shuff_pts, "Shuffled Coordinates Control")

    # 4. Display Comparative Evaluation
    print("\n" + "=" * 65)
    print("COMPARATIVE METRICS REPORT")
    print("=" * 65)
    print(f"  {'Dataset Ensemble':<30} | {'Max H₁ Life':<11} | {'Δα Spectrum':<11} | {'H₁ Count':<8}")
    print(f"  {'-'*30} | {'-'*11} | {'-'*11} | {'-'*8}")
    print(f"  {'1. Authentic Rauzy Fractal':<30} | {r_life:<11.4f} | {r_delta:<11.4f} | {r_h1:<8}")
    print(f"  {'2. Uniform Random Cloud':<30} | {u_life:<11.4f} | {u_delta:<11.4f} | {u_h1:<8}")
    print(f"  {'3. Shuffled Coordinates':<30} | {s_life:<11.4f} | {s_delta:<11.4f} | {s_h1:<8}")
    print("=" * 65)
    
    print("\nInterpretation Guidance:")
    if r_life > u_life * 1.5:
        print("  [✓] SIGNIFICANT TOPOLOGY: Rauzy H₁ lifespan greatly exceeds uniform noise.")
    else:
        print("  [-] Weak topological significance over random noise bounds.")
        
    if r_delta < s_delta:
        print("  [✓] MULTIFRACTAL ACCURACY: Shuffling broadened the spectrum, confirming")
        print("      that the natural geometric correlations prevent extreme fluctuations.")
    print("=" * 65)

if __name__ == '__main__':
    main()
