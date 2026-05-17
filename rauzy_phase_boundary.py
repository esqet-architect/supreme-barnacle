#!/usr/bin/env python3
"""
rauzy_phase_boundary.py
=======================
Executes a high-resolution micro-sweep around the Tribonacci axis to isolate
the exact geometry of the high-gradient phase boundary layer.
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def run_multifractal_1d(pts_1d):
    """Validated 1D fixed-partition Chhabra-Jensen multifractal spectrum engine."""
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

# ═══════════════════════════════════════════════════════════════════════════
# BOUNDARY SCAN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    pts = np.load(data_path)
    
    # Core target angle
    center_deg = 28.53
    print("=" * 70)
    print("HIGH-RESOLUTION MICRO-BOUNDARY PHASE SCANNER")
    print("=" * 70)
    print(f"  Target Focus: {center_deg}° (±5.00° Boundary Window)\n")

    # Generate micro-window degrees
    deg_start = center_deg - 5.0
    deg_end = center_deg + 5.0
    sweep_degrees = np.linspace(deg_start, deg_end, 25)
    
    print(f"  [+] Scanning 25 micro-steps from {deg_start:.2f}° to {deg_end:.2f}°...")
    
    results = []
    for deg in sweep_degrees:
        rad = np.radians(deg)
        proj_vec = np.array([np.cos(rad), np.sin(rad)])
        pts_1d = np.dot(pts, proj_vec)
        da = run_multifractal_1d(pts_1d)
        results.append((deg, da))
        
    print("\n" + "=" * 70)
    print("MICRO-BOUNDARY GRADIENT MAPPING REPORT")
    print("=" * 70)
    print(f"  Angle (Deg)  | Spectrum Width (Δα) | Relative Realignment Status")
    print(f"  ------------ | ------------------- | ---------------------------")
    
    for deg, da in results:
        marker = " "
        if abs(deg - center_deg) < 0.1:
            marker = "[TARGET]* "
        elif da < 0.20:
            marker = "(Valley)  "
        elif da > 0.35:
            marker = "(Steep)   "
            
        print(f"  {deg:<12.2f} | {da:<19.4f} | {marker}")
        
    print("=" * 70)
    
    # Save trace arrays for preservation
    np.save('data/micro_sweep_results.npy', np.array(results))
    print("  [✓] Micro-sweep array preserved cleanly at data/micro_sweep_results.npy")
    print("=" * 70)

if __name__ == '__main__':
    main()
