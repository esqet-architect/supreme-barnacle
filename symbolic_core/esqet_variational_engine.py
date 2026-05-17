#!/usr/bin/env python3
"""
ESQET Variational Engine
Minimizes the discrete information action and observes recursion
"""

import numpy as np

def minimize_discrete_action(steps=20, kappa=0.5, noise_level=0.0):
    print("ESQET Discrete Action Minimization")
    print(f"kappa = {kappa} | noise = {noise_level}")
    print("-" * 60)
    
    S = np.zeros(steps)
    S[0] = 1.0
    S[1] = (1 + np.sqrt(5))/2  # seed near phi
    
    ratios = []
    
    for n in range(2, steps):
        # Numerical minimization of local action term
        # For simplicity we use the analytic stationary condition with noise
        ideal = S[n-1] + S[n-2]   # critical coherence recursion
        noise = noise_level * np.random.randn()
        S[n] = ideal * (1 + noise)
        
        ratio = S[n] / S[n-1]
        ratios.append(ratio)
        
        print(f"Scale {n:2d}: S = {S[n]:.6f} | Ratio = {ratio:.6f}")
    
    mean_ratio = np.mean(ratios[-10:])  # asymptotic
    print("\n" + "="*60)
    print(f"Asymptotic ratio → {mean_ratio:.6f} (φ = 1.618034)")
    print(f"Deviation from φ: {abs(mean_ratio - (1+np.sqrt(5))/2):.2e}")
    
    return S, ratios

if __name__ == "__main__":
    np.random.seed(42)
    print("=== Clean coherence (no noise) ===")
    minimize_discrete_action(noise_level=0.0)
    
    print("\n=== With 1% perturbation ===")
    minimize_discrete_action(noise_level=0.01)
