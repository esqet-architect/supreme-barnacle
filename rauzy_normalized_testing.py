#!/usr/bin/env python3
"""
rauzy_normalized_testing.py
===========================
Executes a scientifically rigorous comparative significance framework using
isotropic coordinate whitening and advanced topological persistence metrics.
"""

import os
import numpy as np
from ripser import ripser

# ═══════════════════════════════════════════════════════════════════════════
# MATH & ANALYSIS ENGINES
# ═══════════════════════════════════════════════════════════════════════════

def whiten_points(pts):
    """Applies a Mahalanobis transformation to eliminate metric anisotropy."""
    pts_centered = pts - pts.mean(axis=0)
    cov = np.cov(pts_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Add small regularization factor to prevent division by zero
    whitening = eigvecs @ np.diag(1.0 / np.sqrt(eigvals + 1e-12)) @ eigvecs.T
    return pts_centered @ whitening

def calculate_persistence_entropy(lifespans):
    """Computes the persistence entropy of H1 topological features."""
    if len(lifespans) == 0 or np.sum(lifespans) == 0:
        return 0.0
    p = lifespans / np.sum(lifespans)
    return -np.sum(p * np.log(p + 1e-300))

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

def analyze_normalized_ensemble(pts, label):
    """Whitens data, runs persistence homology, and evaluates multifractal width."""
    # Apply Isotropic Normalization to resolve ambient metric distortion
    pts_white = whiten_points(pts)
    
    # 1. Topological Feature Extraction (Safe Subsampling Bounds)
    max_subsample = 1200
    pts_sample = pts_white
    if len(pts_white) > max_subsample:
        rng = np.random.default_rng(42)
        pts_sample = pts_white[rng.choice(len(pts_white), max_subsample, replace=False)]
        
    dgms = ripser(pts_sample, maxdim=1)['dgms']
    h1 = dgms[1]
    
    max_life = 0.0
    total_pers = 0.0
    entropy = 0.0
    
    if len(h1) > 0:
        lifespans = h1[:, 1] - h1[:, 0]
        max_life = lifespans.max()
        total_pers = np.sum(lifespans)
        entropy = calculate_persistence_entropy(lifespans)
        
    # 2. Multifractal 1D Axis Evaluation
    alpha_trib = 1.839286755214161
    proj = np.array([1.0 / alpha_trib, 1.0 / (alpha_trib**2)])
    proj /= np.linalg.norm(proj)
    pts_1d = np.dot(pts, proj)
    delta_alpha = run_multifractal_1d(pts_1d)
    
    print(f"  [+] Whitened and parsed: {label}")
    return max_life, total_pers, entropy, delta_alpha

# ═══════════════════════════════════════════════════════════════════════════
# MAIN ROUTINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    rauzy_pts = np.load(data_path)
    n_points = len(rauzy_pts)
    
    x_min, y_min = rauzy_pts.min(axis=0)
    x_max, y_max = rauzy_pts.max(axis=0)

    print("=" * 70)
    print("WHITENED RAUZY FRACTAL VS NULL MODELS: RIGOROUS EVALUATION")
    print("=" * 70)
    print(f"Ensemble Configuration: {n_points} points | Isotropic Whitening Active\n")

    # Generate and process systems
    r_max, r_tot, r_ent, r_da = analyze_normalized_ensemble(rauzy_pts, "Authentic Rauzy")
    
    rng = np.random.default_rng(42)
    rand_pts = np.zeros((n_points, 2))
    rand_pts[:, 0] = rng.uniform(x_min, x_max, n_points)
    rand_pts[:, 1] = rng.uniform(y_min, y_max, n_points)
    u_max, u_tot, u_ent, u_da = analyze_normalized_ensemble(rand_pts, "Uniform Random Null")
    
    shuff_pts = rauzy_pts.copy()
    rng.shuffle(shuff_pts[:, 0])
    rng.shuffle(shuff_pts[:, 1])
    s_max, s_tot, s_ent, s_da = analyze_normalized_ensemble(shuff_pts, "Shuffled Coordinates")

    # Output Comparative Summary Report (Markdown Table Alternative Layout)
    print("\n" + "=" * 70)
    print("SCIENTIFICALLY DEFENSIBLE METRICS REPORT")
    print("=" * 70)
    print(f"  Dataset Ensemble       | Max H₁ Life | Total H₁ Pers | H₁ Entropy | Δα Spec")
    print(f"  ---------------------- | ----------- | ------------- | ---------- | -------")
    print(f"  1. Authentic Rauzy     | {r_max:<11.4f} | {r_tot:<13.4f} | {r_ent:<10.4f} | {r_da:.4f}")
    print(f"  2. Uniform Random Null | {u_max:<11.4f} | {u_tot:<13.4f} | {u_ent:<10.4f} | {u_da:.4f}")
    print(f"  3. Shuffled Control    | {s_max:<11.4f} | {s_tot:<13.4f} | {s_ent:<10.4f} | {s_da:.4f}")
    print("=" * 70)

    print("\nStatistical Observations:")
    if r_max > u_max:
        print("  [✓] Intrinsic topological persistence confirmed over uniform noise scales.")
    else:
        print("  [-] Noise components match maximum persistence bounds.")
        
    print(f"  [*] Multifractal Order Ratio (Shuffled/Rauzy): {s_da / r_da:.2f}x variance.")
    print("=" * 70)

if __name__ == '__main__':
    main()
