#!/usr/bin/env python3
"""
map_substitution_depth.py
==========================
Maps the empirical DFA window scale collapse point to discrete 
substitution generation depths using the dominant Pisot eigenvalue.
"""
import numpy as np
from pisot_spectral_analysis import tribonacci_spectrum

def run_scale_mapping():
    spec = tribonacci_spectrum()
    beta = spec['beta']
    
    # Reconstruct the exact DFA window array used in the analysis
    N = 2000
    window_sizes = np.unique(np.logspace(np.log10(8), np.log10(N // 4), 60).astype(int))
    window_sizes = window_sizes[window_sizes >= 4]
    
    # Extract scales for the localized collapse indices
    s_21 = window_sizes[21]
    s_22 = window_sizes[22]
    s_mid = (s_21 + s_22) / 2.0
    
    print("\n" + "═"*60)
    print("STRUCTURAL SCALE TO SUBSTITUTION DEPTH MAPPING")
    print("═"*60)
    print(f"Empirical Collapse Index j : 21.5")
    print(f"  ↳ Window Scale s[21]      : {s_21} elements")
    print(f"  ↳ Window Scale s[22]      : {s_22} elements")
    print(f"  ↳ Effective Boundary Vector: ~{s_mid:.1f} elements")
    print("─"*60)
    
    print(f"{'Generation (n)':>16} | {'Characteristic Length (β^n)':>30} | {'Delta to Boundary':>18}")
    print("─"*60)
    
    # Compare window size against inflation steps
    best_n = 0
    min_diff = float('inf')
    
    for n in range(1, 12):
        char_length = beta**n
        diff = np.abs(char_length - s_mid)
        
        # Highlight the closest discrete generation
        marker = " <=" if diff < min_diff else ""
        if diff < min_diff:
            min_diff = diff
            best_n = n
            
        print(f"{n:16d} | {char_length:30.4f} | {diff:18.4f}{marker}")
        
    print("═"*60)
    print(f"CONCLUSION: Coherence collapses at Substitution Depth n = {best_n}")
    print(f"Finite-memory structural configurations exhaust themselves at generation {best_n}.")
    print("═"*60)

if __name__ == '__main__':
    run_scale_mapping()
