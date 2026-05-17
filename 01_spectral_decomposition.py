#!/usr/bin/env python3
"""
Part 1: Tribonacci Spectral Decomposition
Computes eigenvalues and eigenvectors of the substitution matrix.
This runs instantly, no heavy computation.
"""

import numpy as np

M = np.array([[1, 1, 1],
              [1, 0, 0],
              [0, 1, 0]], dtype=float)

evals, evecs = np.linalg.eig(M)

# Sort by magnitude
idx = np.argsort(-np.abs(evals))
evals = evals[idx]
evecs = evecs[:, idx]

beta = float(evals[0].real)
v_beta = evecs[:, 0].real
v_beta /= v_beta.sum()

lam_p = evals[1]
r = float(np.abs(lam_p))
theta = float(np.angle(lam_p))

print("="*50)
print("TRIBONACCI SPECTRAL DECOMPOSITION")
print("="*50)
print(f"β (Pisot dominant)    = {beta:.10f}")
print(f"|λ±| = r              = {r:.10f}")
print(f"θ (arg λ+)            = {theta:.10f} rad")
print(f"ω = θ/log(β)          = {theta / np.log(beta):.10f}")
print(f"γ_algebraic           = {-np.log(r) / np.log(beta):.10f}")
print(f"Perron vector v_β     = {v_beta}")
print("="*50)

# Save for later parts
import json
with open('data/spectral.json', 'w') as f:
    json.dump({
        'beta': beta,
        'r': r,
        'theta': theta,
        'omega': theta / np.log(beta),
        'gamma_algebraic': -np.log(r) / np.log(beta),
        'v_beta': v_beta.tolist(),
        'evals': evals.real.tolist()
    }, f)
print("✅ Data saved to data/spectral.json")
