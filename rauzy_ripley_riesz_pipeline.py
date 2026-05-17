#!/usr/bin/env python3
"""
rauzy_ripley_riesz_pipeline.py
==============================
Implements a rotation-invariant second-order structure field using a smooth
Riesz-kernel and a Ripley-style Fourier angular anisotropy spectrum.
"""

import os
import time
import json
import numpy as np

class RipleyRieszPipeline:
    """Computes scale-dependent Riesz structure tensors and Fourier angular spectra."""
    
    def __init__(self, data_path='data/rauzy_points.npy'):
        self.data_path = data_path
        self.points = None

    def load_data(self, max_samples=2000):
        """Loads and applies stable sub-sampling to optimize the pairwise loop."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Missing dataset at {self.data_path}")
        
        full_pts = np.load(self.data_path)
        if len(full_pts) > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(full_pts), size=max_samples, replace=False)
            self.points = full_pts[idx]
        else:
            self.points = full_pts
        return len(self.points)

    def compute_metrics_at_scale(self, r, alpha=2, eps=1e-6):
        """
        Computes the trace-normalized structure tensor, bounded anisotropy index,
        and m=2, m=4 Fourier angular invariants at scale r.
        """
        pts = self.points
        n = len(pts)
        
        T = np.zeros((2, 2))
        W = 0.0
        
        f2_sum = 0.0 + 0.0j
        f4_sum = 0.0 + 0.0j

        for i in range(n):
            diffs = pts[i+1:] - pts[i]
            if len(diffs) == 0:
                continue
                
            d2 = np.sum(diffs**2, axis=1)
            
            # Smooth Riesz-style weight kernel
            w = 1.0 / ((d2 + eps)**(alpha / 2.0)) * np.exp(-d2 / (r**2))
            
            valid = w > 0
            if not np.any(valid):
                continue
                
            diffs = diffs[valid]
            w = w[valid]
            
            # Accumulate structure tensor components
            T[0, 0] += np.sum(w * diffs[:, 0]**2)
            T[0, 1] += np.sum(w * diffs[:, 0] * diffs[:, 1])
            T[1, 1] += np.sum(w * diffs[:, 1]**2)
            W += np.sum(w)
            
            # Compute native 2D vector angles for the Fourier spectrum
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            
            # Fourier anisotropy components: w * e^(i * m * theta)
            f2_sum += np.sum(w * np.exp(2j * angles))
            f4_sum += np.sum(w * np.exp(4j * angles))

        T[1, 0] = T[0, 1]

        # Invariant Extraction Block
        if W > 0:
            T /= W
            # Extract magnitudes of the normalized angular invariants
            f2_mag = np.abs(f2_sum) / W
            f4_mag = np.abs(f4_sum) / W
        else:
            f2_mag, f4_mag = 0.0, 0.0

        # Calculate bounded anisotropy index via trace normalization
        eigs = np.linalg.eigvalsh(T)
        eigs = np.clip(eigs, 1e-12, None)
        trace = np.sum(eigs)
        
        if trace > 0:
            eigs_norm = eigs / trace
            anisotropy_idx = (eigs_norm[1] - eigs_norm[0]) / (eigs_norm[1] + eigs_norm[0])
        else:
            anisotropy_idx = 0.0

        return anisotropy_idx, f2_mag, f4_mag

    def run_multi_scale_pipeline(self):
        """Runs the complete spectral sweep across a range of spatial radii."""
        n_points = self.load_data()
        
        # Determine spatial range scales using the system standard deviation
        global_std = np.mean(np.std(self.points, axis=0))
        scales = np.geomspace(0.05 * global_std, 1.0 * global_std, 6)
        
        print("=" * 75)
        print("RIPLEY-RIESZ INVARIANT SPATIAL ANISOTROPY PIPELINE")
        print("=" * 75)
        print(f"  Matrix Target : {n_points} Points (Sub-sampled)")
        print(f"  Kernel Profile: Continuous Riesz Weighting (alpha=2)")
        print("-" * 75)
        print(f"  {'Scale (r)':>10} | {'Anisotropy (A)':>14} | {'Fourier F2':>12} | {'Fourier F4':>12}")
        print("-" * 75)
        
        results = []
        start_time = time.time()
        
        for r in scales:
            A, f2, f4 = self.compute_metrics_at_scale(r)
            print(f"  {r:10.4f} | {A:14.4f} | {f2:12.4f} | {f4:12.4f}")
            
            results.append({
                "radius": r,
                "anisotropy_index": A,
                "f2_ellipticity": f2,
                "f4_quadrupole": f4
            })
            
        total_time = time.time() - start_time
        print("-" * 75)
        print(f"  [✓] Multi-scale Analysis Completed in {total_time:.4f} seconds")
        print("=" * 75)
        
        # Save structured measurements log
        with open('data/ripley_riesz_metrics.json', 'w') as f:
            json.dump({"scales_evaluated": results, "execution_time": total_time}, f, indent=4)

if __name__ == '__main__':
    pipeline = RipleyRieszPipeline()
    pipeline.run_multi_scale_pipeline()
