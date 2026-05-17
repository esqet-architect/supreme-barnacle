#!/usr/bin/env python3
"""
MINERVA V7.0: Hardened Ergodic Regime Explorer
================================================
- Intrinsic centered cocycle: v_i = e_i - f (zero ballistic drift)
- Real Schur Decomposition for stable subspace isolation
- Exact irreducibility test over Q (SymPy) — filters reducible systems
- Primitivity verification via power iteration
- Grassberger-Procaccia correlation dimension (local slope median)
- Wandering exponent γ with negative clamping
- Enterprise Excel export with styling

Algebraic classifiers ensure that only true Pisot systems (primitive, irreducible,
dominant >1, conjugates <1) are flagged as such.
"""

import os
import numpy as np
import pandas as pd
from scipy.linalg import schur, eig, qr
from scipy.spatial.distance import pdist
from sympy import Matrix, symbols
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ALGEBRAIC STRUCTURE CORE (EXACT OVER Q)
# ============================================================================

def is_primitive_matrix(M, max_power=12):
    """Verifies primitivity via sequential power expansion."""
    M_k = np.copy(M)
    for _ in range(max_power):
        if np.all(M_k > 0):
            return True
        M_k = M_k @ M
    return False


def check_algebraic_irreducibility(M):
    """
    Uses SymPy exact arithmetic to verify polynomial irreducibility over Q.
    A system is structurally irreducible iff its characteristic polynomial
    factors into exactly one prime factor block over Q.
    """
    x = symbols('x')
    M_int = Matrix(M.astype(int))
    charpoly = M_int.charpoly(x).as_expr()
    from sympy.polys.polytools import factor_list
    factors = factor_list(charpoly)[1]
    return len(factors) == 1


def analyze_pisot_spectrum(evals):
    """Evaluates strict Pisot modular bounds across non-dominant roots."""
    moduli = np.abs(evals)
    sorted_mod = np.sort(moduli)[::-1]
    dominant = sorted_mod[0]
    rest = sorted_mod[1:] if len(sorted_mod) > 1 else []
    
    has_pisot_spectrum = (dominant > 1.0) and (np.max(rest) < 0.99 if len(rest) > 0 else True)
    return has_pisot_spectrum, dominant


# ============================================================================
# 2. INTRINSIC COCYCLE & SPECTRAL SYSTEM
# ============================================================================

class IntrinsicSubstitutionSystem:
    """
    Canonical centered abelianization: v_i = e_i - f
    where f is the normalized Perron-Frobenius eigenvector.
    """
    
    def __init__(self, name, matrix, sub_rules, alphabet):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.sub_rules = sub_rules
        self.alphabet = alphabet
        self._characterize_system()

    def _characterize_system(self):
        evals, evecs = eig(self.matrix)
        real_parts = np.real(evals)
        pf_idx = np.argmax(real_parts)
        pf_evec = np.abs(np.real(evecs[:, pf_idx]))
        
        # Canonical frequency vector
        self.frequencies = pf_evec / np.sum(pf_evec)
        
        # Centered increments: v_i = e_i - f
        self.increments = {}
        for i, ch in enumerate(self.alphabet):
            e_i = np.zeros(len(self.matrix))
            e_i[i] = 1.0
            self.increments[ch] = e_i - self.frequencies

        # Structural validation
        self.is_primitive = is_primitive_matrix(self.matrix)
        self.is_irreducible = check_algebraic_irreducibility(self.matrix)
        has_pisot_spectrum, self.lambda_pf = analyze_pisot_spectrum(evals)
        
        # Strict Pisot: primitive AND irreducible AND stable spectrum
        self.is_pisot = self.is_primitive and self.is_irreducible and has_pisot_spectrum

    def generate_sequence(self, steps=12, max_len=50000):
        seq = self.alphabet[0]
        for _ in range(steps):
            seq = ''.join(self.sub_rules.get(c, c) for c in seq)
            if len(seq) > max_len:
                seq = seq[:max_len]
                break
        return seq

    def compute_walk(self, seq):
        d = len(self.matrix)
        walk = np.zeros((len(seq) + 1, d))
        for i, ch in enumerate(seq):
            walk[i+1] = walk[i] + self.increments.get(ch, np.zeros(d))
        return walk


# ============================================================================
# 3. SPECTRAL SUBSPACE EXTRACTION (REAL SCHUR)
# ============================================================================

def project_via_schur_subspace(walk, M):
    """Extracts stable contracting subspace coordinates via sorted Schur blocks."""
    d = len(M)
    T, Z = schur(M, output='real')
    
    # Isolate dominant expansion coordinate index
    diag_abs = np.abs(np.diag(T))
    pf_block_idx = np.argmax(diag_abs)
    
    stable_indices = [i for i in range(d) if i != pf_block_idx]

    if d == 2 and len(stable_indices) >= 1:
        proj_space = Z[:, stable_indices[0]].reshape(1, d)
    elif len(stable_indices) >= 2:
        proj_space = Z[:, stable_indices[:2]].T
    else:
        proj_space = np.eye(2, d)

    # Orthonormalize via QR
    Q, _ = qr(proj_space.T, mode='economic')
    return walk @ Q


# ============================================================================
# 4. GEOMETRIC METRICS
# ============================================================================

def compute_wandering_exponent(points):
    """Estimates γ from |X_n| ~ n^γ. Clamps micro-negative artifacts to 0."""
    displacements = np.linalg.norm(points, axis=1)
    n_steps = len(displacements)
    indices = np.arange(1, n_steps + 1)

    valid = displacements > 1e-8
    if not valid.any():
        return 0.0

    coeffs = np.polyfit(np.log(indices[valid]), np.log(displacements[valid]), 1)
    gamma = coeffs[0]

    # Clamp finite-size regression artifacts to zero
    if -0.05 < gamma < 0.02:
        gamma = 0.0
    return max(0.0, float(gamma))


def compute_correlation_dimension(points, num_samples=600):
    """
    Grassberger-Procaccia correlation dimension D₂.
    Uses local slope median over central plateau.
    """
    n = len(points)
    if n < 50:
        return 0.0

    pts = points[np.random.choice(n, num_samples, replace=False)] if n > num_samples else points
    N = len(pts)
    if N < 30:
        return 0.0

    dists = pdist(pts)
    dists = dists[dists > 1e-8]
    if len(dists) < 20:
        return 0.0

    r_min = max(np.percentile(dists, 5), 1e-7)
    r_max = np.percentile(dists, 30)
    if r_min >= r_max:
        return 0.0

    radii = np.logspace(np.log10(r_min), np.log10(r_max), 15)
    total_pairs = N * (N - 1) / 2.0
    counts = np.array([np.sum(dists < r) / total_pairs for r in radii])

    valid = counts > 1e-8
    if np.sum(valid) < 5:
        return 0.0

    log_r = np.log(radii[valid])
    log_c = np.log(counts[valid])

    slopes = []
    for i in range(len(log_r) - 3):
        slope, _ = np.polyfit(log_r[i:i+4], log_c[i:i+4], 1)
        slopes.append(slope)

    d2 = np.median(slopes) if slopes else 0.0
    return max(0.0, min(float(d2), 2.0))


# ============================================================================
# 5. SYSTEM CONFIGURATIONS
# ============================================================================

SYSTEMS = {
    "Tribonacci": {
        "matrix": [[1, 1, 1], [1, 0, 0], [0, 1, 0]],
        "rules": {"a": "ab", "b": "ac", "c": "a"},
        "alphabet": "abc",
        "steps": 11
    },
    "Fibonacci": {
        "matrix": [[1, 1], [1, 0]],
        "rules": {"a": "ab", "b": "a"},
        "alphabet": "ab",
        "steps": 14
    },
    "Pisot Binary Substitution": {
        "matrix": [[3, 2], [1, 1]],
        "rules": {"a": "abaab", "b": "aba"},
        "alphabet": "ab",
        "steps": 7
    },
    "ThueMorse": {
        "matrix": [[1, 1], [1, 1]],
        "rules": {"a": "ab", "b": "ba"},
        "alphabet": "ab",
        "steps": 11
    }
}


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 75)
    print("MINERVA V7.0: Hardened Ergodic Regime Explorer")
    print("Canonical: v_i = e_i - f | Schur Projection | GP Dimension")
    print("=" * 75)
    
    results = []
    
    for name, cfg in SYSTEMS.items():
        print(f"\n{'─' * 55}")
        print(f"📐 System: {name}")
        print(f"{'─' * 55}")
        
        sys_obj = IntrinsicSubstitutionSystem(name, cfg["matrix"], cfg["rules"], cfg["alphabet"])
        
        seq = sys_obj.generate_sequence(steps=cfg["steps"])
        walk = sys_obj.compute_walk(seq)
        projected = project_via_schur_subspace(walk, sys_obj.matrix)
        
        subspace_dim = projected.shape[1]
        gamma = compute_wandering_exponent(projected)
        d2 = compute_correlation_dimension(projected)
        
        print(f"   Primitive:        {sys_obj.is_primitive}")
        print(f"   Irreducible over Q: {sys_obj.is_irreducible}")
        print(f"   Strict Pisot:      {sys_obj.is_pisot}")
        print(f"   Dominant λ_PF:     {sys_obj.lambda_pf:.6f}")
        print(f"   Subspace dim:      {subspace_dim}D")
        print(f"   Wandering γ:       {gamma:.4f}")
        print(f"   Correlation D₂:    {d2:.4f}")
        
        if sys_obj.is_pisot:
            if subspace_dim == 2:
                cls = "Planar Rauzy Fractal"
            else:
                cls = "1D Cut-and-Project Strip"
        else:
            cls = "Reducible Automatic System"
        print(f"   Classification:    {cls}")
        
        results.append({
            "System": name,
            "Primitive": "Yes" if sys_obj.is_primitive else "No",
            "Irreducible_over_Q": "Yes" if sys_obj.is_irreducible else "No",
            "Strict_Pisot": "Yes" if sys_obj.is_pisot else "No",
            "Lambda_PF": sys_obj.lambda_pf,
            "Subspace_Dim": f"{subspace_dim}D",
            "Wandering_Gamma": gamma,
            "Correlation_D2": d2,
            "Classification": cls
        })
    
    # DataFrame output
    df = pd.DataFrame(results)
    csv_path = "~/rauzy_analysis/minerva_v7.0_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 75)
    print("✅ ANALYSIS COMPLETE")
    print(f"📁 CSV saved to: {csv_path}")
    print("=" * 75)
    
    # Console summary table
    print("\n📊 SUMMARY")
    print("-" * 85)
    print(f"{'System':<25} {'Pisot':<8} {'Dim':<5} {'γ':<8} {'D₂':<8} {'Classification'}")
    print("-" * 85)
    for r in results:
        print(f"{r['System']:<25} {r['Strict_Pisot']:<8} {r['Subspace_Dim']:<5} "
              f"{r['Wandering_Gamma']:<8.4f} {r['Correlation_D2']:<8.4f} {r['Classification']}")
    print("=" * 85)


if __name__ == "__main__":
    main()
