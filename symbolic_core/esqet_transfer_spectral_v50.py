#!/usr/bin/env python3
"""
ESQET Transfer Matrix + Spectral Analysis v50
Searches for critical recursion from matrix dynamics
No Fibonacci seeding
"""

import numpy as np

def transfer_matrix(gamma):
    """T such that [S_{n+1}, S_n]^T = T [S_n, S_{n-1}]^T"""
    return np.array([[2 - gamma, -1],
                     [1,          0]])

def analyze_spectrum(gamma):
    T = transfer_matrix(gamma)
    eigvals = np.linalg.eigvals(T)
    spectral_radius = np.max(np.abs(eigvals))
    trace = np.trace(T)
    det = np.linalg.det(T)
    
    return {
        'eigenvalues': eigvals,
        'spectral_radius': spectral_radius,
        'trace': trace,
        'det': det,
        'dominant_ratio': max(eigvals, key=abs).real if len(eigvals) > 0 else 0
    }

def scan_critical_regime():
    print("ESQET Transfer Matrix Spectral Scan")
    print("Looking for marginally stable hyperbolic regimes")
    print("="*75)
    
    gammas = np.linspace(0.0, 4.0, 41)
    results = []
    
    print(f"{'γ':<6} {'Trace':<8} {'|λ_max|':<10} {'Dominant':<10} {'Regime'}")
    print("-"*60)
    
    for g in gammas:
        spec = analyze_spectrum(g)
        ev = spec['eigenvalues']
        r = spec['spectral_radius']
        dom = spec['dominant_ratio']
        
        if abs(spec['det'] - 1.0) > 1e-8:
            regime = "Non-volume-preserving"
        elif r < 1.0:
            regime = "Elliptic / oscillatory"
        elif abs(r - 1.0) < 1e-6:
            regime = "Parabolic / marginal"
        else:
            regime = "Hyperbolic"
            
        if abs(dom - (1+np.sqrt(5))/2) < 0.05:
            marker = " ← φ-like"
        else:
            marker = ""
            
        print(f"{g:<6.2f} {spec['trace']:<8.3f} {r:<10.4f} {dom:<10.4f} {regime}{marker}")
        results.append((g, r, dom))
    
    # Find closest to golden
    closest = min(results, key=lambda x: abs(x[2] - (1+np.sqrt(5))/2))
    print("\n" + "="*75)
    print(f"Closest to φ at γ ≈ {closest[0]:.3f} with dominant eigenvalue {closest[2]:.6f}")

if __name__ == "__main__":
    np.random.seed(42)
    scan_critical_regime()
