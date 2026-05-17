#!/usr/bin/env python3
"""
ESQET Phase Boundary Mapping Engine
Executes fine-grained sweeps, Hessian tracking, and finite-size scaling analysis.
"""

import numpy as np
from scipy.optimize import minimize
import json

def esqet_action(S, gamma):
    transport = 0.5 * np.sum(np.diff(S)**2)
    memory = -gamma * np.sum(S[2:] * S[:-2])
    return transport + memory

def energy_constraint(S):
    return np.sum(S**2) - 1.0

def analyze_system(N, gamma):
    # Stabilized initialization
    np.random.seed(1618)
    S0 = np.ones(N) / np.sqrt(N)
    
    constraints = {'type': 'eq', 'fun': energy_constraint}
    
    res = minimize(
        esqet_action, S0, args=(gamma,),
        method='SLSQP', constraints=constraints,
        options={'ftol': 1e-12, 'maxiter': 2000}
    )
    
    if not res.success:
        return None
        
    S_opt = res.x
    
    # Calculate late-stage asymptotic ratio
    ratios = S_opt[1:] / (S_opt[:-1] + 1e-9)
    late_ratio = np.mean(ratios[-5:])
    
    # Numerical Hessian Evaluation at the minimum
    eps = 1e-5
    H = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Finite differences for second derivatives
            S_pp = S_opt.copy(); S_pp[i] += eps; S_pp[j] += eps
            S_pm = S_opt.copy(); S_pm[i] += eps; S_pm[j] -= eps
            S_mp = S_opt.copy(); S_mp[i] -= eps; S_mp[j] += eps
            S_mm = S_opt.copy(); S_mm[i] -= eps; S_mm[j] -= eps
            
            H[i, j] = (esqet_action(S_pp, gamma) - esqet_action(S_pm, gamma) - 
                       esqet_action(S_mp, gamma) + esqet_action(S_mm, gamma)) / (4 * eps**2)
                       
    # Calculate eigenvalues of the Hessian to verify stability matrix curvature
    eigvals = np.linalg.eigvalsh(H)
    min_eig = np.min(eigvals)
    
    return {
        "action": float(res.fun),
        "ratio": float(late_ratio),
        "min_eigenvalue": float(min_eig)
    }

if __name__ == "__main__":
    print("=" * 75)
    print("ESQET PHASE BOUNDARY ANALYSIS RUNNING")
    print("=" * 75)
    
    gamma_sweep = np.linspace(0.0, 2.0, 9)
    system_sizes = [20, 50, 100]
    
    phase_map = {}
    
    for N in system_sizes:
        print(f"\nScanning Lattice Space N = {N}:")
        print(f"{'γ':<10} {'Action':<15} {'Ratio':<15} {'Min Eig (Hessian)':<20} {'Status':<12}")
        print("-" * 75)
        
        phase_map[N] = []
        for g in gamma_sweep:
            metrics = analyze_system(N, g)
            if metrics is None:
                continue
                
            # A negative eigenvalue implies the optimized state is no longer a true local minimum
            status = "STABLE" if metrics["min_eigenvalue"] > -1e-4 else "COLLAPSED"
            
            print(f"{g:<10.2f} {metrics['action']:<15.6f} {metrics['ratio']:<15.6f} {metrics['min_eigenvalue']:<20.4e} {status:<12}")
            
            metrics["gamma"] = float(g)
            metrics["status"] = status
            phase_map[N].append(metrics)

    with open('Minerva-dea-mathematica/experimental_logs/phase_map_results.json', 'w') as f:
        json.dump(phase_map, f, indent=4)
        
    print("\n✅ Phase map data successfully logged to experimental_logs/phase_map_results.json")
