#!/usr/bin/env python3
"""
rauzy_final_eval.py
===================
Executes both validated options for the 2D Rauzy point cloud:
1. Option A: 1D physical projection multifractal evaluation.
2. Option B: 2D persistent homology diagram generation via Ripser.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams

# ═══════════════════════════════════════════════════════════════════════════
# ENGINE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def run_multifractal_1d(pts_1d, q_arr):
    """Computes Chhabra-Jensen spectrum specifically tailored for 1D projections."""
    x = pts_1d.flatten()
    n = len(x)
    lo, hi = x.min(), x.max()
    span = hi - lo
    
    # Define 14 linear-to-geometric scales based on 1D span
    scales = np.geomspace(0.005 * span, 0.25 * span, 14)
    log_eps = np.log(scales)
    
    log_Z = np.zeros((len(q_arr), len(scales)))
    num_alpha = np.zeros((len(q_arr), len(scales)))
    num_f = np.zeros((len(q_arr), len(scales)))
    
    for s_idx, eps in enumerate(scales):
        # 1D Box grouping
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
            num_f[q_idx, s_idx] = np.sum(mu_t * np.log(mu_t + 1e-300))

    # Fit scaling window using central q=0 linearity
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
    f_out = np.zeros(len(q_arr))
    Dq_out = np.zeros(len(q_arr))
    
    for j in range(len(q_arr)):
        alpha_out[j] = np.polyfit(log_eps[win], num_alpha[j][win], 1)[0]
        f_out[j] = np.polyfit(log_eps[win], num_f[j][win], 1)[0]
        tau = np.polyfit(log_eps[win], log_Z[j][win], 1)[0]
        Dq_out[j] = alpha_out[j] if abs(q_arr[j] - 1.0) < 1e-3 else tau / (q_arr[j] - 1.0)

    return q_arr, alpha_out, np.maximum(f_out, 0.0), Dq_out, best_r2

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ROUTINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: {data_path} missing.")
        return

    pts = np.load(data_path)
    print("=" * 65)
    print("RAUZY FRACTAL UNIFIED EVALUATION SYSTEM")
    print("=" * 65)

    # ──── EXECUTE OPTION A ─────────────────────────────────────────────────
    alpha_trib = 1.839286755214161
    proj = np.array([1.0 / alpha_trib, 1.0 / (alpha_trib**2)])
    proj /= np.linalg.norm(proj)
    pts_1d = np.dot(pts, proj)
    
    q_range = np.linspace(-1.0, 1.0, 9)
    _, a, f, Dq, fit_r2 = run_multifractal_1d(pts_1d, q_range)
    
    D0 = Dq[np.abs(q_range).argmin()]
    print("\n[OPTION A RESULTS: 1D PHYSICAL PROJECTION]")
    print(f"  Linear Scaling Window Fit R² = {fit_r2:.5f}")
    print(f"  Capacity Dimension (D₀)      = {D0:.4f}")
    print(f"  Spectral Range Width (Δα)    = {a.max() - a.min():.4f}")
    
    # ──── EXECUTE OPTION B ─────────────────────────────────────────────────
    print("\n[OPTION B RESULTS: 2D PERSISTENT HOMOLOGY]")
    max_subsample = 1200
    pts_sample = pts
    if len(pts) > max_subsample:
        rng = np.random.default_rng(42)
        pts_sample = pts[rng.choice(len(pts), max_subsample, replace=False)]
        
    dgms = ripser(pts_sample, maxdim=1)['dgms']
    print(f"  Analyzed {len(pts_sample)} points.")
    print(f"  H₀ (Connected components) count: {len(dgms[0])}")
    print(f"  H₁ (Topological holes) count:    {len(dgms[1])}")
    
    # Save persistence plot file
    output_img = 'rauzy_persistence_diagram.png'
    plt.figure(figsize=(7, 7))
    plot_diagrams(dgms, show=False)
    plt.title("Rauzy Topological Persistence Diagram ($H_0$ / $H_1$)")
    plt.savefig(output_img, dpi=150)
    plt.close()
    print(f"  -> Saved topological persistence diagram to: {output_img}")
    print("=" * 65)

if __name__ == '__main__':
    main()
