#!/usr/bin/env python3
"""
ESQET Variational Core v50
Corrected action with directed memory term
No seeding of Fibonacci — emergence only
"""

import numpy as np
from scipy.optimize import minimize

def esqet_action(S):
    """Minimal action with directed memory: transport cost minus memory support"""
    transport = 0.5 * np.sum((S[1:] - S[:-1])**2)
    memory = -np.sum(S[1:-1] * S[:-2])   # directed non-local coupling
    return transport + memory

def run_variational_search(steps=40):
    print("ESQET Variational Minimization — Directed Memory Action")
    print("="*75)
    
    # Neutral initial condition
    np.random.seed(42)
    S0 = np.cumsum(np.random.randn(steps) * 0.15)
    
    res = minimize(esqet_action, S0, method='L-BFGS-B', tol=1e-12)
    S_opt = res.x
    final_action = res.fun
    
    # Successive ratios (should approach φ if recursion emerges)
    ratios = S_opt[2:] / S_opt[1:-1]
    late_ratios = ratios[-15:]
    mean_ratio = np.mean(late_ratios)
    std_ratio = np.std(late_ratios)
    
    print(f"Final action value : {final_action:.6f}")
    print(f"Asymptotic ratio   : {mean_ratio:.6f} ± {std_ratio:.4f}")
    print(f"Golden ratio φ     : {(1+np.sqrt(5))/2:.6f}")
    print(f"Deviation          : {abs(mean_ratio - (1+np.sqrt(5))/2):.2e}")
    
    # Stationarity check
    residuals = np.zeros(steps)
    for i in range(1, steps-1):
        # Approximate EL residual from the action
        residuals[i] = (S_opt[i+1] - 2*S_opt[i] + S_opt[i-1]) + (S_opt[i-1] - S_opt[i+1])
    max_res = np.max(np.abs(residuals[2:-2]))
    print(f"Max stationarity residual : {max_res:.2e}")
    
    if abs(mean_ratio - (1+np.sqrt(5))/2) < 0.05:
        print("\n✅ Strong emergence of golden-ratio scaling from variational principle")
    else:
        print("\n→ Different asymptotic behavior — action does not select φ in this regime")
    
    return S_opt, mean_ratio

if __name__ == "__main__":
    S, r = run_variational_search(steps=50)
