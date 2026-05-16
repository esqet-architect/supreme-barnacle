#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pisot_spectral_analysis import run_full_analysis, tribonacci_spectrum

def trace_local_dynamics(delta, window_size=24, step=4):
    spec = tribonacci_spectrum()
    omega_target = spec['omega']
    
    j_axis = np.arange(len(delta), dtype=float)
    centers = []
    local_amps = []
    local_r2 = []
    
    # Slide across the scale domain
    for start in range(0, len(delta) - window_size, step):
        end = start + window_size
        j_seg = j_axis[start:end]
        d_seg = delta[start:end]
        
        # Linear regression profile for local power projection
        A_matrix = np.column_stack([np.cos(omega_target * j_seg), np.sin(omega_target * j_seg)])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(A_matrix, d_seg, rcond=None)
            amplitude = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
            
            ss_res = residuals[0] if len(residuals) > 0 else np.sum((d_seg - A_matrix @ coeffs)**2)
            ss_tot = np.sum((d_seg - d_seg.mean())**2)
            r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            centers.append(j_seg.mean())
            local_amps.append(amplitude)
            local_r2.append(max(0.0, min(r2, 1.0)))
        except np.linalg.LinAlgError:
            continue

    return np.array(centers), np.array(local_amps), np.array(local_r2)

if __name__ == '__main__':
    res = run_full_analysis(n_orbit=2000, verbose=False)
    delta = res['delta']
    
    centers, amps, r2_scores = trace_local_dynamics(delta)
    
    print("\n" + "═"*60)
    print("SPATIAL LOCALIZATION OF SPECTRAL COHERENCE")
    print("═"*60)
    print(f"{'Scale Index (j)':>16} | {'Local Amplitude':>16} | {'Local R²':>12}")
    print("─"*60)
    
    # Display step updates showing the intermittency
    for idx in range(0, len(centers), max(1, len(centers)//10)):
        print(f"{centers[idx]:16.2f} | {amps[idx]:16.5f} | {r2_scores[idx]:12.4f}")
    print("═"*60)
