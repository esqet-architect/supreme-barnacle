#!/usr/bin/env python3
"""
ESQET Variational Core - Stable Implementation
Directed memory action with proper numerical constraints
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def esqet_action(S, gamma=1.0):
    """
    S_n evolution governed by stationarity of:
    A = ½ Σ (ΔS)² - γ Σ S_{n} S_{n-1}
    """
    # Guard against overflow
    S_clipped = np.clip(S, -1e6, 1e6)
    transport = 0.5 * np.sum(np.diff(S_clipped)**2)
    memory = -gamma * np.sum(S_clipped[1:-1] * S_clipped[:-2])
    return transport + memory

def run_variational_analysis(steps=30, gamma=1.0, verbose=True):
    if verbose:
        print("="*70)
        print(f"ESQET Variational Analysis (γ = {gamma})")
        print("="*70)
    
    # Initial condition: small random perturbations
    np.random.seed(42)
    S0 = np.random.randn(steps) * 0.1
    S0[0] = 1.0  # Fix scale
    
    # Bounds to prevent blow-up
    bounds = [(-100, 100) for _ in range(steps)]
    
    result = minimize(
        esqet_action, S0, args=(gamma,),
        method='L-BFGS-B', bounds=bounds,
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    S_opt = result.x
    ratios = S_opt[2:] / (S_opt[1:-1] + 1e-8)
    mean_ratio = np.mean(ratios[-10:])
    std_ratio = np.std(ratios[-10:])
    phi = (1 + np.sqrt(5)) / 2
    
    if verbose:
        print(f"Optimal action: {result.fun:.6f}")
        print(f"Asymptotic ratio: {mean_ratio:.6f} ± {std_ratio:.6f}")
        print(f"Golden ratio φ: {phi:.6f}")
        print(f"Deviation: {abs(mean_ratio - phi):.2e}")
        if abs(mean_ratio - phi) < 0.05:
            print("✓ Golden ratio emerged from action minimization!")
        else:
            print("→ Different asymptotic ratio — γ-dependent")
    
    return S_opt, mean_ratio, result.fun

def scan_gamma(gamma_values=[0.5, 0.8, 1.0, 1.2, 1.5]):
    print("\n" + "="*70)
    print("GAMMA SCAN: Finding Critical Point")
    print("="*70)
    print(f"{'γ':<8} {'Ratio':<12} {'Deviation':<12} {'Action':<12}")
    print("-"*50)
    
    results = {}
    for g in gamma_values:
        _, ratio, action = run_variational_analysis(steps=25, gamma=g, verbose=False)
        results[g] = {'ratio': ratio, 'action': action}
        print(f"{g:<8.2f} {ratio:<12.6f} {abs(ratio - (1+np.sqrt(5))/2):<12.2e} {action:<12.2e}")
    
    return results

if __name__ == "__main__":
    # Single run at critical γ = 1
    S_opt, ratio, action = run_variational_analysis(steps=40, gamma=1.0)
    
    # Scan over γ values
    scan_gamma()
    
    # Plot the optimized sequence
    plt.figure(figsize=(10, 6))
    plt.plot(S_opt, 'bo-', linewidth=1, markersize=3)
    plt.xlabel('Step n')
    plt.ylabel('S_n')
    plt.title(f'ESQET Optimized Sequence at γ = 1.0\nAsymptotic ratio = {ratio:.4f}')
    plt.grid(True, alpha=0.3)
    plt.savefig('esqet_variational_sequence.png', dpi=150)
    plt.close()
    print("\n✅ Plot saved: esqet_variational_sequence.png")
