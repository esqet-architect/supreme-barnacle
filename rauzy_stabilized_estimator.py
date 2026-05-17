#!/usr/bin/env python3
"""
rauzy_stabilized_estimator.py
=============================
Stabilizes the Chhabra-Jensen multifractal engine by enforcing a globally locked 
scaling window, eliminating discontinuous numerical phase-switching artifacts.
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# STABILIZED MULTIFRACTAL ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_stabilized_multifractal_1d(pts_1d):
    """
    1D Chhabra-Jensen multifractal engine with a globally locked scaling window.
    Eliminates adaptive window switching to ensure continuous angular response.
    """
    x = pts_1d.flatten()
    n = len(x)
    lo, hi = x.min(), x.max()
    span = hi - lo
    
    q_arr = np.linspace(-1.0, 1.0, 9)
    
    # Lock the scale limits globally to prevent window-switching discontinuities
    # Using 12 fixed scale increments between 1.5% and 15% of the macro span
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

    # REMOVED: Dynamic optimization window selection (if r2 > best_r2)
    # FORCE: Global linear fit across the entire locked scale range
    alpha_out = np.zeros(len(q_arr))
    for j in range(len(q_arr)):
        alpha_out[j] = np.polyfit(log_eps, num_alpha[j], 1)[0]

    return alpha_out.max() - alpha_out.min()

# ═══════════════════════════════════════════════════════════════════════════
# MAIN METHODOLOGICAL STEP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    pts = np.load(data_path)
    
    # Targeted micro-sweep test verification angles
    test_angles_deg = [27.70, 28.11, 28.53, 45.00, 118.53]
    
    print("=" * 70)
    print("STABILIZED GLOBAL WINDOW MULTIFRACTAL EVALUATION")
    print("=" * 70)
    print("  Status: Scaling range locked strictly at [0.015, 0.15] * Span")
    print("  Goal: Verify removal of numerical switching plateaus\n")
    
    print(f"  Angle (Deg)  | Stabilized Δα Spec Width")
    print(f"  ------------ | ------------------------")
    
    for deg in test_angles_deg:
        rad = np.radians(deg)
        proj_vec = np.array([np.cos(rad), np.sin(rad)])
        pts_1d = np.dot(pts, proj_vec)
        da_stabilized = run_stabilized_multifractal_1d(pts_1d)
        
        tag = ""
        if abs(deg - 28.11) < 0.01:
            tag = " <- Prior Artifact Spike Point"
        print(f"  {deg:<12.2f} | {da_stabilized:<24.4f}{tag}")
        
    print("=" * 70)

if __name__ == '__main__':
    main()
