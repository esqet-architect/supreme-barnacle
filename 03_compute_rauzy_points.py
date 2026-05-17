#!/usr/bin/env python3
"""
Part 3: Compute Rauzy fractal points (Corrected Projective Mapping)
Projects explicitly onto the stable contractive plane of the complex conjugates.
"""

import numpy as np
import json
from tqdm import tqdm

# Load spectral data
M = np.array([[1, 1, 1],
              [1, 0, 0],
              [0, 1, 0]], dtype=float)

evals, evecs = np.linalg.eig(M)

# Sort by magnitude (Dominant first, then complex conjugates)
idx = np.argsort(-np.abs(evals))
evals = evals[idx]
evecs = evecs[:, idx]

# The stable contractive plane is spanned by the real and imaginary parts 
# of the eigenvector corresponding to the complex conjugate eigenvalue.
w_complex = evecs[:, 1]

# Construct the true orthogonal projection matrix to the internal stable space
pi_c = np.zeros((2, 3))
pi_c[0, :] = w_complex.real
pi_c[1, :] = w_complex.imag

# Normalize rows to keep geometry stable
pi_c[0, :] /= np.linalg.norm(pi_c[0, :])
pi_c[1, :] /= np.linalg.norm(pi_c[1, :])

# Load word
with open('data/tribonacci_word.txt', 'r') as f:
    word = f.read().strip()

basis = {'1': np.array([1, 0, 0]),
         '2': np.array([0, 1, 0]),
         '3': np.array([0, 0, 1])}

print("="*50)
print("COMPUTING CORRECTED RAUZY FRACTAL POINTS")
print("="*50)
print(f"Word length: {len(word)}")
print("Projecting onto true stable internal space...")

points = []
pos = np.zeros(3)
for letter in tqdm(word[:6000]):
    proj = pi_c @ pos
    points.append(proj)
    pos = pos + basis[letter]

points = np.array(points)

# Save corrected data
np.save('data/rauzy_points.npy', points)
print(f"✅ Saved {len(points)} points to data/rauzy_points.npy")
print(f"   Bounding box: x∈[{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
print(f"                 y∈[{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
