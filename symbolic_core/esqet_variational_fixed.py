#!/usr/bin/env python3
import numpy as np
from scipy.optimize import minimize

def esqet_action_fixed(S_inner, S_0, S_N, gamma=1.0):
    # Reconstruct the full sequence from optimized inner parts and fixed boundaries
    S = np.concatenate(([S_0], S_inner, [S_N]))
    
    transport = 0.5 * np.sum(np.diff(S)**2)
    memory = -gamma * np.sum(S[1:-1] * S[:-2])
    return transport + memory

def run_fixed_analysis(steps=30, gamma=1.0):
    S_0 = 1.0
    S_N = 2.0  # Anchor the endpoint to force a transport gradient
    
    # Optimize only the internal states
    np.random.seed(42)
    S0_inner = np.random.randn(steps - 2) * 0.1
    
    result = minimize(
        esqet_action_fixed, S0_inner, args=(S_0, S_N, gamma),
        method='BFGS', options={'gtol': 1e-8}
    )
    
    S_opt = np.concatenate(([S_0], result.x, [S_N]))
    ratios = S_opt[2:] / (S_opt[1:-1] + 1e-8)
    mean_ratio = np.mean(ratios[-5:])
    
    print(f"γ: {gamma:.2f} | Asymptotic Ratio: {mean_ratio:.6f} | Action: {result.fun:.4f}")

if __name__ == "__main__":
    print("Evaluating Variational Anchored System:")
    for g in [0.5, 1.0, 1.5]:
        run_fixed_analysis(steps=20, gamma=g)
