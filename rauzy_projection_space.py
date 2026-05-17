#!/usr/bin/env python3
"""
rauzy_projection_space.py
=========================
Sweeps the 1D projection angle across 180 degrees to evaluate the directional
dependence of the multifractal spectrum width (Delta alpha) for the Rauzy cloud.
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS ENGINE
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
# EXECUTION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    pts = np.load(data_path)
    
    # Define reference axes
    alpha_trib = 1.839286755214161
    v_trib = np.array([1.0 / alpha_trib, 1.0 / (alpha_trib**2)])
    v_trib /= np.linalg.norm(v_trib)
    theta_trib = np.arctan2(v_trib[1], v_trib[0])
    if theta_trib < 0:
        theta_trib += np.pi

    # Perpendicular orthogonal complement axis
    theta_ortho = (theta_trib + np.pi / 2.0) % np.pi

    print("=" * 70)
    print("RAUZY PROJECTION-SPACE DIRECTIONAL DEPENDENCE SWEEP")
    print("=" * 70)
    
    # Perform uniform angular sweep across 180 degrees
    angles = np.linspace(0.0, np.pi, 36, endpoint=False)
    widths = []
    
    for theta in angles:
        proj_vec = np.array([np.cos(theta), np.sin(theta)])
        pts_1d = np.dot(pts, proj_vec)
        da = run_multifractal_1d(pts_1d)
        widths.append(da)
        
    widths = np.array(widths)
    
    # Evaluate explicit reference baselines
    da_trib = run_multifractal_1d(np.dot(pts, v_trib))
    v_ortho = np.array([np.cos(theta_ortho), np.sin(theta_ortho)])
    da_ortho = run_multifractal_1d(np.dot(pts, v_ortho))
    
    min_idx = np.argmin(widths)
    max_idx = np.argmax(widths)

    print("\n" + "=" * 70)
    print("ANGULAR SWEEP ANALYSIS REPORT")
    print("=" * 70)
    print(f"  Canonical Tribonacci Axis Angle : {np.degrees(theta_trib):.2f}° | Δα = {da_trib:.4f}")
    print(f"  Orthogonal Complement Axis Angle: {np.degrees(theta_ortho):.2f}° | Δα = {da_ortho:.4f}")
    print(f"  ------------------------------------------------------------------")
    print(f"  Global Minimum Spectrum Width   : {np.degrees(angles[min_idx]):.2f}° | Δα = {widths[min_idx]:.4f}")
    print(f"  Global Maximum Spectrum Width   : {np.degrees(angles[max_idx]):.2f}° | Δα = {widths[max_idx]:.4f}")
    print("=" * 70)

    print("\nScientific Validation Inference:")
    if abs(da_trib - widths[min_idx]) < 0.05 or theta_trib < angles[1]:
        print("  [✓] HIGH AXIAL UNIQUE PROFILE: The Tribonacci axis sits near or at the")
        print("      global minimum of measure variability, locking in its structural significance.")
    else:
        print("  [*] Anisotropic variance observed. The geometric support layout shifts scale profiles dynamically.")
    print("=" * 70)

if __name__ == '__main__':
    main()
