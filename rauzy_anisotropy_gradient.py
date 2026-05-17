#!/usr/bin/env python3
"""
rauzy_anisotropy_gradient.py
============================
Calculates the directional derivative (gradient) of the multifractal spectrum 
width across the angular field to isolate regions of structural stability.
"""

import os
import numpy as np

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

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    pts = np.load(data_path)
    
    # Target Target Angle: 28.53 degrees (Tribonacci Projection Coordinate)
    target_theta = np.radians(28.53)
    
    # Delta step size for numerical derivative approximation (1 degree in radians)
    h = np.radians(1.0)
    
    # Sample spectrum width at theta, theta + h, and theta - h
    da_center = run_multifractal_1d(np.dot(pts, np.array([np.cos(target_theta), np.sin(target_theta)])))
    da_plus   = run_multifractal_1d(np.dot(pts, np.array([np.cos(target_theta + h), np.sin(target_theta + h)])))
    da_minus  = run_multifractal_1d(np.dot(pts, np.array([np.cos(target_theta - h), np.sin(target_theta - h)])))
    
    # Central difference formula for derivative calculation
    directional_derivative = (da_plus - da_minus) / (2.0 * h)
    
    print("=" * 70)
    print("RAUZY GEOMETRIC GRADIENT & ANISOTROPY VOLATILITY ANALYSIS")
    print("=" * 70)
    print(f"  Target Verification Point     : 28.53° (Tribonacci Physical Axis)")
    print(f"  Local Multifractal Width (Δα) : {da_center:.4f}")
    print(f"  Directional Gradient (dΔα/dθ) : {directional_derivative:.4f} per radian")
    print("=" * 70)
    
    print("\nStructural Architecture Summary:")
    if abs(directional_derivative) < 0.15:
        print("  [✓] STRUCTURAL VALLEY STABILITY: The low directional derivative confirms")
        print("      the Tribonacci neighborhood resides in a stable multifractal valley.")
    else:
        print("  [*] HIGH PHASE SENSITIVITY: Rapid structural scaling adjustments occur near this axis.")
    print("=" * 70)

if __name__ == '__main__':
    main()
