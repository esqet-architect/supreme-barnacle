#!/usr/bin/env python3
"""
MINERVA V6.3: Intrinsic Eigenvector Cocycle Engine
- Automatically derives step increments from the matrix Perron-Frobenius vector
- Tightened Pisot validation parameters to isolate true Rauzy structures
- Dynamic scale window adjustments to prevent NaN box-counting errors
"""

import numpy as np
from numpy.linalg import eig, eigh, norm
import csv
import warnings

warnings.filterwarnings('ignore')

class IntrinsicSubstitutionSystem:
    def __init__(self, name, matrix, substitution_rules):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.substitution_rules = substitution_rules
        self._derive_intrinsic_properties()

    def _derive_intrinsic_properties(self):
        """Extract exact Perron-Frobenius frequencies and test for strict Pisot structure."""
        evals, evecs = eig(self.matrix)
        
        # Identify the dominant (Perron-Frobenius) eigenvalue
        idx_dom = np.argmax(np.real(evals))
        self.dominant_eigenvalue = np.real(evals[idx_dom])
        
        # Calculate symbol frequencies from the dominant right eigenvector
        pf_vector = np.abs(np.real(evecs[:, idx_dom]))
        self.frequencies = pf_vector / np.sum(pf_vector)
        
        # Strict Pisot validation: dominant > 1, all other algebraic conjugates < 1
        other_moduli = [np.abs(ev) for i, ev in enumerate(evals) if i != idx_dom]
        self.is_pisot = self.dominant_eigenvalue > 1.0 and all(m < 0.99 for m in other_moduli)

        # Generate the intrinsic, automatically centered cocycle step increments
        # v_i = e_i - frequency_vector
        self.alphabet = list(self.substitution_rules.keys())
        self.increment_map = {}
        dim = len(self.alphabet)
        
        for i, symbol in enumerate(self.alphabet):
            e_i = np.zeros(dim)
            e_i[i] = 1.0
            # Pad to 3D for universal projection alignment if system is binary
            increment = e_i - self.frequencies
            if dim == 2:
                increment = np.append(increment, 0.0)
            self.increment_map[symbol] = increment

    def generate_word(self, length):
        current = [self.alphabet[0]]
        while len(current) < length:
            new_current = []
            for ch in current:
                new_current.extend(self.substitution_rules.get(ch, [ch]))
            current = new_current
            if len(current) > length * 2:
                break
        return current[:length]

    def generate_walk(self, length=1200):
        word = self.generate_word(length)
        increments = np.array([self.increment_map[ch] for ch in word])
        positions = np.cumsum(increments, axis=0)
        return positions

def get_projection_basis(system, points_3d):
    """Isolates the stable contracting plane using spectral methods or optimized PCA."""
    if system.is_pisot and len(system.alphabet) == 3:
        evals, evecs = eig(system.matrix)
        for i, ev in enumerate(evals):
            if abs(np.imag(ev)) > 1e-10:
                v = evecs[:, i]
                return np.vstack([np.real(v), np.imag(v)])
                
    # Optimized PCA Fallback for binary or non-Pisot configurations
    subsample = points_3d[::max(1, len(points_3d)//500)]
    centered = subsample - np.mean(subsample, axis=0)
    cov = centered.T @ centered / len(subsample)
    _, evecs = eigh(cov)
    return evecs[:, -2:].T

def calculate_box_counting_dimension(points_2d, min_scales=10):
    """Dynamic scale windowing to prevent underpopulation errors."""
    x, y = points_2d[:, 0], points_2d[:, 1]
    bbox_min, bbox_max = np.min(points_2d, axis=0), np.max(points_2d, axis=0)
    diag = norm(bbox_max - bbox_min) + 1e-15

    # Safe logspace span adjusted for tight boundaries
    scales = np.logspace(-2.0, -0.4, min_scales) * diag
    counts = []
    valid_scales = []

    for s in scales:
        xi = np.floor((x - bbox_min[0]) / s)
        yi = np.floor((y - bbox_min[1]) / s)
        n_boxes = len(set(zip(xi, yi)))
        if n_boxes > 2:  # Minimum box cutoff to secure regression stability
            counts.append(n_boxes)
            valid_scales.append(s)

    if len(valid_scales) >= 4:
        slope, _ = np.polyfit(np.log(1.0 / np.array(valid_scales)), np.log(counts), 1)
        return float(slope)
    return 0.0

if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA V6.3: Intrinsic Eigenvector Cocycles & Fixed Scaling")
    print("=======================================================================")

    systems = {
        'Tribonacci': IntrinsicSubstitutionSystem('Tribonacci', 
            [[1, 1, 1], [1, 0, 0], [0, 1, 0]], 
            {'a': ['a', 'b'], 'b': ['a', 'c'], 'c': ['a']}),
        'Fibonacci': IntrinsicSubstitutionSystem('Fibonacci', 
            [[1, 1], [1, 0]], 
            {'a': ['a', 'b'], 'b': ['a']}),
        'ThueMorse': IntrinsicSubstitutionSystem('ThueMorse', 
            [[1, 1], [1, 1]], 
            {'a': ['a', 'b'], 'b': ['b', 'a']}),
        'Sturmian': IntrinsicSubstitutionSystem('Sturmian', 
            [[1, 1], [1, 0]], 
            {'a': ['a', 'b'], 'b': ['a']})
    }

    results = []

    for name, sys in systems.items():
        print(f"\n🔬 System: {name}")
        print(f"   Strict Pisot: {sys.is_pisot} | Dominant Eig: {sys.dominant_eigenvalue:.4f}")
        print(f"   Derived Frequencies: " + ", ".join([f"{k}:{v:.3f}" for k, v in zip(sys.alphabet, sys.frequencies)]))
        
        walk = sys.generate_walk(length=1500)
        basis = get_projection_basis(sys, walk)
        pts_2d = walk @ basis.T if basis.shape[0] == 2 else walk[:, :2]
        
        d0 = calculate_box_counting_dimension(pts_2d)
        print(f"   Intrinsic Box Dimension D0: {d0:.4f}")
        
        results.append({
            'system': name,
            'is_pisot': sys.is_pisot,
            'dominant_eigenvalue': sys.dominant_eigenvalue,
            'd0': d0
        })

    # Save CSV
    csv_path = "minerva_v6.3_intrinsic_metrics.csv"
    fieldnames = ['system', 'is_pisot', 'dominant_eigenvalue', 'd0']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n=======================================================================")
    print(f"✅ Executed clean. Local metrics saved to: {csv_path}")
    print("=======================================================================")
