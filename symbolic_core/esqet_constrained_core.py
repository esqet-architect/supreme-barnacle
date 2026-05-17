#!/usr/bin/env python3
"""
ESQET Constrained Phase Space Engine
Implements L2 Spherical Normalization to map true structural emergence.
"""

import numpy as np
from scipy.optimize import minimize

def esqet_action_core(S, gamma):
    """
    Core action: Transport cost balanced against non-local memory tracking.
    A = 0.5 * sum((S_n - S_{n-1})^2) - gamma * sum(S_n * S_{n-2})
    """
    transport = 0.5 * np.sum(np.diff(S)**2)
    memory = -gamma * np.sum(S[2:] * S[:-2])
    return transport + memory

def energy_constraint(S):
    """Enforces total structural variance conservation (Unit Sphere)"""
    return np.sum(S**2) - 1.0

def evaluate_gamma(gamma, steps=25):
    # Uniform seed initialization
    np.random.seed(1618)
    S0 = np.ones(steps) / np.sqrt(steps)
    
    constraints = {'type': 'eq', 'fun': energy_constraint}
    
    result = minimize(
        esqet_action_core, S0, args=(gamma,),
        method='SLSQP', constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 1000}
    )
    
    S_opt = result.x
    # Track late-stage recursive ratios
    ratios = S_opt[1:] / (S_opt[:-1] + 1e-9)
    asymptotic_ratio = np.mean(ratios[-5:])
    
    return result.fun, asymptotic_ratio, S_opt

if __name__ == "__main__":
    print("=" * 65)
    print("CONSTRAINED VARIATIONAL SEARCH: Conserved Energy Domain")
    print("=" * 65)
    print(f"{'γ (Coupling)':<15} {'Minimized Action':<20} {'Asymptotic Ratio':<20}")
    print("-" * 65)
    
    gammas = [0.0, 0.5, 1.0, 1.5, 2.0]
    phi_target = (1 + np.sqrt(5)) / 2
    
    for g in gammas:
        action, ratio, _ = evaluate_gamma(g)
        print(f"{g:<15.2f} {action:<20.6f} {ratio:<20.6f}")
        
    print("-" * 65)
    print(f"Target Geometric Invariant (φ): {phi_target:.6f}")
    print("=" * 65)
