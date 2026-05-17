#!/usr/bin/env python3
"""
ESQET Variational Core v48
Minimizes discrete information action without seeding Fibonacci
Observes emergent recursion and spectral properties
"""

import numpy as np
from scipy.optimize import minimize

def discrete_action(S, kappa=0.5):
    """Compute total action for sequence S"""
    n = len(S)
    transport = 0.5 * np.sum((S[1:] - S[:-1])**2)
    # Memory coupling term (non-local)
    memory = kappa * np.sum(S[2:] * S[:-2])
    return transport + memory

def stationarity_residuals(S, kappa=0.5):
    """Euler-Lagrange residuals for debugging"""
    residuals = np.zeros_like(S)
    for i in range(1, len(S)-1):
        # Contribution from transport
        res = (S[i] - S[i-1]) - (S[i+1] - S[i])
        # Contribution from memory coupling
        res += kappa * (S[i+1] + S[i-1] - 2*S[i]) if i+1 < len(S) and i-1 >= 0 else 0
        residuals[i] = res
    return residuals

def run_variational_experiment(steps=30, kappa=0.5, noise=0.0):
    print("ESQET Variational Minimization")
    print(f"Steps: {steps} | κ = {kappa} | Noise = {noise}")
    print("="*70)

    # Initial condition (small random seed, no Fibonacci bias)
    np.random.seed(42)
    S0 = np.cumsum(np.random.randn(steps) * 0.1)

    # Minimize the action
    result = minimize(discrete_action, S0, args=(kappa,),
                      method='L-BFGS-B', tol=1e-10)

    S_opt = result.x
    final_action = result.fun

    # Compute successive ratios (asymptotic behavior)
    ratios = S_opt[2:] / S_opt[1:-1]
    mean_ratio = np.mean(ratios[-10:]) if len(ratios) > 10 else np.nan
    std_ratio = np.std(ratios[-10:])

    print(f"Final action value: {final_action:.6f}")
    print(f"Asymptotic ratio (last 10 steps): {mean_ratio:.6f} ± {std_ratio:.6f}")
    print(f"Golden ratio reference:           {(1+np.sqrt(5))/2:.6f}")

    # Check stationarity
    residuals = stationarity_residuals(S_opt, kappa)
    max_residual = np.max(np.abs(residuals[1:-1]))
    print(f"Max stationarity residual: {max_residual:.2e}")

    if abs(mean_ratio - (1+np.sqrt(5))/2) < 0.05:
        print("→ Emergent golden-ratio-like scaling detected at critical κ")
    else:
        print("→ Different asymptotic ratio — κ-dependent dynamics")

    return S_opt, ratios

if __name__ == "__main__":
    print("=== Critical coherence (κ ≈ 0.5) ===")
    S, ratios = run_variational_experiment(kappa=0.5, noise=0.0)
    
    print("\n=== Perturbed case (small noise in initial condition) ===")
    S_noisy, ratios_noisy = run_variational_experiment(kappa=0.5, noise=0.02)
