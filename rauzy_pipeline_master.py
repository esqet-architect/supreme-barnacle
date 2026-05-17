#!/usr/bin/env python3
"""
rauzy_pipeline_master.py
========================
Master pipeline class that encapsulates the stabilized multifractal calculation,
manages local execution states, and automates structured performance reporting.
"""

import os
import time
import json
import numpy as np

class RauzyMultifractalPipeline:
    """Orchestrates multifractal calculations and records performance telemetry."""
    
    def __init__(self, data_path='data/rauzy_points.npy', log_path='data/pipeline_metrics.json'):
        self.data_path = data_path
        self.log_path = log_path
        self.points = None
        
    def load_dataset(self):
        """Validates existence of and safely loads target point cloud."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Target data file not found at: {self.data_path}")
        self.points = np.load(self.data_path)
        return len(self.points)

    def calculate_1d_spectrum(self, pts_1d):
        """Executes a 1D Chhabra-Jensen evaluation with a fixed scaling window."""
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

    def execute_sweep(self, target_angles_deg):
        """Runs a complete angular evaluation, logs speeds, and saves JSON outputs."""
        n_points = self.load_dataset()
        start_time = time.time()
        
        print("=" * 70)
        print("INITIALIZING MASTER ORCHESTRATION PIPELINE")
        print("=" * 70)
        print(f"  Source Array : {self.data_path} ({n_points} points)")
        print("  Engine Mode  : Encapsulated Stable Angular Verification\n")
        
        widths = []
        for deg in target_angles_deg:
            rad = np.radians(deg)
            proj_vec = np.array([np.cos(rad), np.sin(rad)])
            
            step_start = time.time()
            pts_1d = np.dot(self.points, proj_vec)
            da = self.calculate_1d_spectrum(pts_1d)
            step_duration = time.time() - step_start
            
            widths.append(da)
            print(f"  [➔] Component angle: {deg:>6.2f}° | Width: {da:.4f} | Time: {step_duration:.4f}s")
            
        total_duration = time.time() - start_time
        
        # Save structural log packet
        log_packet = {
            "metadata": {"sample_size": n_points, "file": self.data_path},
            "performance": {"execution_seconds": total_duration},
            "results": {"angles": target_angles_deg, "widths": widths}
        }
        
        with open(self.log_path, 'w') as f:
            json.dump(log_packet, f, indent=4)
            
        print("\n" + "=" * 70)
        print(f"  Pipeline Master Run Completed. Log: {self.log_path}")
        print("=" * 70)

if __name__ == '__main__':
    # Define production parameters
    pipeline = RauzyMultifractalPipeline()
    pipeline.execute_sweep(target_angles_deg=[0.0, 45.0, 90.0, 135.0])
