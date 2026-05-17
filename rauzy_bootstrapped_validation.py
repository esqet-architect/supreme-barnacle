#!/usr/bin/env python3
"""
rauzy_bootstrapped_validation.py
================================
Runs a bootstrapped angular sweep using the stabilized multifractal engine
to calculate statistical confidence intervals for the anisotropy profile.
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# STABILIZED MULTIFRACTAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_stabilized_multifractal_1d(pts_1d):
    """1D Chhabra-Jensen multifractal engine with a globally locked scaling window."""
    x = pts_1d.flatten()
    n = len(x)
    lo, hi = x.min(), x.max()
    span = hi - lo
    
    q_arr = np.linspace(-1.0, 1.0, 9)
    scales = np.geomspace(0.015 * span, 0.15 * span, 12)
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

    alpha_out = np.zeros(len(q_arr))
    for j in range(len(q_arr)):
        alpha_out[j] = np.polyfit(log_eps, num_alpha[j], 1)[0]

    return alpha_out.max() - alpha_out.min()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN BOOTSTRAP EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    raw_pts = np.load(data_path)
    n_points = len(raw_pts)
    
    # Execution parameters
    n_bootstraps = 10
    angles_deg = np.linspace(0.0, 180.0, 12, endpoint=False)
    
    print("=" * 70)
    print("BOOTSTRAPPED MULTIFRACTAL ANISOTROPY VALIDATION")
    print("=" * 70)
    print(f"  Matrix: {n_points} points | {n_bootstraps} Bootstrap Iterations")
    print("  Status: Tracking statistical variance limits over angular field...\n")
    
    # Storage matrix for results: shape = (angles, bootstraps)
    sweep_matrix = np.zeros((len(angles_deg), n_bootstraps))
    rng = np.random.default_rng(42)
    
    for b in range(n_bootstraps):
        # Generate bootstrap sample with replacement
        boot_idx = rng.choice(n_points, size=n_points, replace=True)
        pts_sample = raw_pts[boot_idx]
        
        for a_idx, deg in enumerate(angles_deg):
            rad = np.radians(deg)
            proj_vec = np.array([np.cos(rad), np.sin(rad)])
            pts_1d = np.dot(pts_sample, proj_vec)
            sweep_matrix[a_idx, b] = run_stabilized_multifractal_1d(pts_1d)
            
        print(f"  [+] Completed bootstrap iteration {b+1}/{n_bootstraps}")

    print("\n" + "=" * 70)
    print("STABLE DIRECTIONAL SPECTRUM CONFIDENCE REPORT")
    print("=" * 70)
    print(f"  Angle (Deg)  | Mean Spec Width (Δα) | Standard Deviation (σ)")
    print(f"  ------------ | -------------------- | ----------------------")
    
    for i, deg in enumerate(angles_deg):
        mean_da = np.mean(sweep_matrix[i, :])
        std_da = np.std(sweep_matrix[i, :])
        print(f"  {deg:<12.2f} | {mean_da:<20.4f} | {std_da:.6f}")
        
    print("=" * 70)
    
    # Save statistics for data integrity
    np.save('data/bootstrapped_anisotropy.npy', sweep_matrix)
    print("  [✓] Statistical matrix preserved at data/bootstrapped_anisotropy.npy")
    print("=" * 70)

if __name__ == '__main__':
    main()
