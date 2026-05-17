#!/usr/bin/env python3
"""
MINERVA V7.1: Exact Spectral Projector (Biorthogonal)
=======================================================
- Replaces Schur projection with exact biorthogonal projector:
    P_s = I - (v_PF w_PF^T) / (w_PF^T v_PF)
- Removes Perron-Frobenius drift exactly (not approximately)
- Preserves all other algebraic validations (irreducibility over Q, primitivity)
- Canonical centered cocycle: v_i = e_i - f
"""

import numpy as np
import pandas as pd
from scipy.linalg import eig, qr
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
    """SymPy exact arithmetic to verify polynomial irreducibility over Q."""
    x = symbols('x')
    M_int = Matrix(M.astype(int))
    charpoly = M_int.charpoly(x).as_expr()
    from sympy.polys.polytools import factor_list
    factors = factor_list(charpoly)[1]
    return len(factors) == 1


def analyze_pisot_spectrum(evals):
    """Strict Pisot spectral modulus bounds."""
    moduli = np.abs(evals)
    sorted_mod = np.sort(moduli)[::-1]
    dominant = sorted_mod[0]
    rest = sorted_mod[1:] if len(sorted_mod) > 1 else []
    has_pisot_spectrum = (dominant > 1.0) and (np.max(rest) < 0.99 if len(rest) > 0 else True)
    return has_pisot_spectrum, dominant


# ============================================================================
# 2. INTRINSIC COCYCLE (CANONICAL)
# ============================================================================

class IntrinsicSubstitutionSystem:
    """Canonical centered abelianization: v_i = e_i - f."""
    
    def __init__(self, name, matrix, sub_rules, alphabet):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.sub_rules = sub_rules
        self.alphabet = alphabet
        self._characterize_system()

    def _characterize_system(self):
        evals, evecs = eig(self.matrix)
        
        # Perron-Frobenius extraction
        real_parts = np.real(evals)
        pf_idx = np.argmax(real_parts)
        pf_eval = evals[pf_idx]
        v_PF = np.abs(np.real(evecs[:, pf_idx]))
        
        # Left eigenvector for biorthogonal projection
        _, L_evecs = eig(self.matrix.T)
        real_parts_L = np.real(evals)  # same eigenvalues
        pf_idx_L = np.argmax(real_parts_L)
        w_PF = np.abs(np.real(L_evecs[:, pf_idx_L]))
        
        # Normalize for biorthogonality
        w_PF = w_PF / np.dot(w_PF, v_PF)
        
        self.v_PF = v_PF / np.sum(v_PF)  # normalized frequency vector
        self.w_PF = w_PF
        self.lambda_pf = float(pf_eval.real)
        
        # Canonical centered increments: v_i = e_i - f
        self.increments = {}
        for i, ch in enumerate(self.alphabet):
            e_i = np.zeros(len(self.matrix))
            e_i[i] = 1.0
            self.increments[ch] = e_i - self.v_PF

        # Structural validation
        self.is_primitive = is_primitive_matrix(self.matrix)
        self.is_irreducible = check_algebraic_irreducibility(self.matrix)
        has_pisot_spectrum, _ = analyze_pisot_spectrum(evals)
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
# 3. EXACT SPECTRAL PROJECTOR (BIORTHOGONAL)
# ============================================================================

def exact_spectral_projector(system, walk):
    """
    Exact removal of Perron-Frobenius direction using biorthogonal projector:
        P_s = I - (v_PF w_PF^T) / (w_PF^T v_PF)
    This eliminates PF drift exactly, not approximately.
    """
    d = len(system.matrix)
    v = system.v_PF.reshape(-1, 1)
    w = system.w_PF.reshape(1, -1)
    
    # Biorthogonal projector onto stable subspace
    P_stable = np.eye(d) - (v @ w) / (w @ v).item()
    
    # Project walk
    return walk @ P_stable.T


# ============================================================================
# 4. GEOMETRIC METRICS
# ============================================================================

def compute_wandering_exponent(points):
    """Estimates γ from |X_n| ~ n^γ. Clamps micro-negative artifacts."""
    displacements = np.linalg.norm(points, axis=1)
    n_steps = len(displacements)
    indices = np.arange(1, n_steps + 1)

    valid = displacements > 1e-8
    if not valid.any():
        return 0.0

    coeffs = np.polyfit(np.log(indices[valid]), np.log(displacements[valid]), 1)
    gamma = coeffs[0]

    # Clamp finite-size regression artifacts
    if -0.05 < gamma < 0.02:
        gamma = 0.0
    return max(0.0, float(gamma))


def compute_correlation_dimension(points, num_samples=600):
    """Grassberger-Procaccia correlation dimension D₂."""
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
    print("MINERVA V7.1: Exact Spectral Projector (Biorthogonal)")
    print("P_s = I - (v_PF w_PF^T)/(w_PF^T v_PF)")
    print("=" * 75)
    
    results = []
    
    for name, cfg in SYSTEMS.items():
        print(f"\n{'─' * 55}")
        print(f"📐 System: {name}")
        print(f"{'─' * 55}")
        
        sys_obj = IntrinsicSubstitutionSystem(name, cfg["matrix"], cfg["rules"], cfg["alphabet"])
        
        seq = sys_obj.generate_sequence(steps=cfg["steps"])
        walk = sys_obj.compute_walk(seq)
        
        # EXACT SPECTRAL PROJECTION (not Schur)
        projected = exact_spectral_projector(sys_obj, walk)
        
        # Remove any remaining trivial drift
        projected = projected - np.mean(projected, axis=0)
        
        subspace_dim = projected.shape[1]
        gamma = compute_wandering_exponent(projected)
        d2 = compute_correlation_dimension(projected)
        
        print(f"   Primitive:           {sys_obj.is_primitive}")
        print(f"   Irreducible over Q:  {sys_obj.is_irreducible}")
        print(f"   Strict Pisot:        {sys_obj.is_pisot}")
        print(f"   Dominant λ_PF:       {sys_obj.lambda_pf:.6f}")
        print(f"   Subspace dim:        {subspace_dim}D")
        print(f"   Wandering γ:         {gamma:.4f}")
        print(f"   Correlation D₂:      {d2:.4f}")
        
        if sys_obj.is_pisot:
            if subspace_dim >= 2 and d2 > 1.5:
                cls = "Planar Rauzy Fractal"
            elif subspace_dim >= 1:
                cls = "1D Cut-and-Project Strip"
            else:
                cls = "Pisot Attractor"
        else:
            cls = "Reducible Automatic System"
        print(f"   Classification:      {cls}")
        
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
    csv_path = "~/rauzy_analysis/minerva_v7.1_exact_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 75)
    print("✅ ANALYSIS COMPLETE")
    print(f"📁 CSV saved to: {csv_path}")
    print("=" * 75)
    
    # Console summary
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
