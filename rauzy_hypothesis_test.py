#!/usr/bin/env python3
"""
rauzy_hypothesis_test.py
========================
Executes a formal hypothesis test by evaluating the Anisotropy Amplitude (A) 
of the Rauzy cloud against a covariance-matched Gaussian null ensemble.
"""

import os
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# LOCKED STABILIZED MULTIFRACTAL CORE
# ═══════════════════════════════════════════════════════════════════════════

def run_stabilized_multifractal_1d(pts_1d):
    """1D Chhabra-Jensen engine with a globally locked scaling window."""
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

def calculate_anisotropy_amplitude(pts, angles_rad):
    """Computes the global amplitude statistic A = max(da) - min(da) over a sweep."""
    da_list = []
    for rad in angles_rad:
        proj_vec = np.array([np.cos(rad), np.sin(rad)])
        pts_1d = np.dot(pts, proj_vec)
        da_list.append(run_stabilized_multifractal_1d(pts_1d))
    return max(da_list) - min(da_list)

# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL HYPOTHESIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    rauzy_pts = np.load(data_path)
    n_points = len(rauzy_pts)
    
    # Statistical parameters
    n_null_models = 10
    sweep_angles_rad = np.radians(np.linspace(0.0, 180.0, 8, endpoint=False))
    
    print("=" * 70)
    print("FORMAL HYPOTHESIS TESTING SUITE: ANISOTROPY AMPLITUDE")
    print("=" * 70)
    print(f"  Sample Density : {n_points} points")
    print(f"  Null Ensemble  : {n_null_models} Covariance-Matched Gaussian Clouds\n")
    
    # 1. Compute Empirical Properties of the Authentic Data
    mean_vector = rauzy_pts.mean(axis=0)
    covariance_matrix = np.cov(rauzy_pts.T)
    
    print("  [+] Mapping authentic anisotropy profile...")
    A_authentic = calculate_anisotropy_amplitude(rauzy_pts, sweep_angles_rad)
    print(f"  [✓] Authentic Rauzy Amplitude (A_rauzy): {A_authentic:.6f}")
    
    # 2. Generate and Evaluate the Null Models
    print("\n  [+] Synthesizing and scanning covariance-matched null models...")
    rng = np.random.default_rng(42)
    null_amplitudes = []
    
    for i in range(n_null_models):
        # Generate random cloud matching exactly the macroscopic layout of the fractal
        null_pts = rng.multivariate_normal(mean_vector, covariance_matrix, size=n_points)
        A_null = calculate_anisotropy_amplitude(null_pts, sweep_angles_rad)
        null_amplitudes.append(A_null)
        print(f"      - Null Model {i+1:02d} Amplitude (A_null): {A_null:.6f}")
        
    null_amplitudes = np.array(null_amplitudes)
    mean_A_null = np.mean(null_amplitudes)
    
    # 3. Compute Statistical Significance (p-value)
    extreme_count = np.sum(null_amplitudes >= A_authentic)
    p_value = extreme_count / n_null_models
    
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE MATRIX")
    print("=" * 70)
    print(f"  Authentic Anisotropy Amplitude (A)   : {A_authentic:.6f}")
    print(f"  Null Ensemble Mean Amplitude (E[A])  : {mean_A_null:.6f}")
    print(f"  Empirical Alpha Signatures (p-value) : {p_value:.4f}")
    print("=" * 70)
    
    print("\nFinal Defensible Scientific Inference:")
    if p_value < 0.05:
        print("  [✓] SIGNIFICANT GEOMETRIC ANISOTROPY: The Rauzy fractal exhibits spatial")
        print("      directional variations that cannot be explained by structural stretch alone.")
    else:
        print("  [-] INSIGNIFICANT DIRECTIONAL VARIANCE: The mild angular variability is")
        print("      statistically indistinguishable from a standard random spatial distribution.")
    print("=" * 70)

if __name__ == '__main__':
    main()
