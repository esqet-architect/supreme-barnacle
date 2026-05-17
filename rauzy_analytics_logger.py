#!/usr/bin/env python3
"""
rauzy_analytics_logger.py
=========================
Executes a monitored analysis sweep, tracking execution speed and performance metrics,
then exports the compiled data to an archival JSON log file.
"""

import os
import time
import json
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS CORE
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

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION AND PERFORMANCE TRACKING
# ═══════════════════════════════════════════════════════════════════════════

def main():
    data_path = 'data/rauzy_points.npy'
    if not os.path.exists(data_path):
        print(f"[-] Error: Data missing at {data_path}")
        return

    pts = np.load(data_path)
    n_points = len(pts)
    
    # Track runtime metrics
    start_time = time.time()
    
    test_angles = [0.0, 45.0, 90.0, 135.0]
    widths = []
    
    print("=" * 70)
    print("RUN-TIME PERFORMANCE LOGGER AND ANALYTICS PIPELINE")
    print("=" * 70)
    print(f"  Target Matrix : {n_points} Points")
    print(f"  Logging Mode  : Automated Execution Speed Benchmarking...\n")
    
    for deg in test_angles:
        rad = np.radians(deg)
        proj_vec = np.array([np.cos(rad), np.sin(rad)])
        
        # Benchmarking an individual transformation loop
        step_start = time.time()
        da = run_stabilized_multifractal_1d(np.dot(pts, proj_vec))
        step_duration = time.time() - step_start
        
        widths.append(da)
        print(f"  [+] Angle: {deg:>6.2f}° | Δα: {da:.4f} | Process Time: {step_duration:.4f}s")
        
    total_duration = time.time() - start_time
    print(f"\n  [✓] Complete Sweep Executed in: {total_duration:.4f} seconds")
    
    # Assemble structured analytics dictionary
    log_data = {
        "pipeline_metadata": {
            "dataset_source": data_path,
            "sample_size": n_points,
            "estimator_type": "Locked-Window Chhabra-Jensen"
        },
        "performance_metrics": {
            "total_execution_seconds": total_duration,
            "average_step_seconds": total_duration / len(test_angles)
        },
        "calculated_outputs": {
            "evaluated_angles_deg": test_angles,
            "spectrum_widths": widths,
            "observed_amplitude": max(widths) - min(widths)
        }
    }
    
    # Save metrics block to file system
    json_out_path = 'data/pipeline_metrics.json'
    with open(json_out_path, 'w') as f:
        json.dump(log_data, f, indent=4)
        
    print("\n" + "=" * 70)
    print(f"  Metrics file cleanly generated at: {json_out_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
