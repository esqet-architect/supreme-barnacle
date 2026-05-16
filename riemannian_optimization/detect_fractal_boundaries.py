#!/usr/bin/env python3
import numpy as np
from track_intermittent_phases import trace_local_dynamics
from pisot_spectral_analysis import run_full_analysis

def identify_phase_transitions(centers, r2_scores, threshold=0.15):
    # Calculate the localized derivative of coherence shifts
    r2_diff = np.diff(r2_scores)
    transition_points = []
    
    print("\n" + "═"*60)
    print("DETECTED GEOMETRIC BOUNDARY TRANSITIONS")
    print("═"*60)
    print(f"{'Boundary Center':>15} | {'Coherence Shift (ΔR²)':>22} | {'Direction':>12}")
    print("─"*60)
    
    for i in range(len(r2_diff)):
        if np.abs(r2_diff[i]) >= threshold:
            direction = "COLLAPSE" if r2_diff[i] < 0 else "RECOVERY"
            pos = (centers[i] + centers[i+1]) / 2
            transition_points.append((pos, r2_diff[i], direction))
            print(f"{pos:15.2f} | {r2_diff[i]:+22.4f} | {direction:>12}")
            
    print("═"*60)
    return transition_points

if __name__ == '__main__':
    res = run_full_analysis(n_orbit=2000, verbose=False)
    centers, amps, r2_scores = trace_local_dynamics(res['delta'])
    
    boundaries = identify_phase_transitions(centers, r2_scores)
    if not boundaries:
        print("No sharp boundary step-changes detected under current threshold limits.")
        print("The structural decay behavior is operating continuously across these scales.")
        print("═"*60)
