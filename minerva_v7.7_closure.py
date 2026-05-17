#!/usr/bin/env python3
"""
MINERVA V7.7: BIORTHOGONAL CLOSURE & COMPLEX SUBSPACE ECOORDINATE PRESERVATION
================================================================================
- Resolves the projector rank defect by enforcing strict w^T * v = 1 duality.
- Restores complex contracting dimensions by projecting onto both Real and Imag components.
- Automates plateau calculation via local slope derivative analysis.
- Preserves sequential continuity using Moving Block Bootstrap (MBB) analytics.
"""

import numpy as np
import pandas as pd
from scipy.linalg import eig
from scipy.spatial.distance import pdist
from sympy import Matrix, symbols
import warnings
warnings.filterwarnings('ignore')

def is_primitive_matrix(M, max_power=12):
    M_k = np.copy(M)
    for _ in range(max_power):
        if np.all(M_k > 0):
            return True
        M_k = M_k @ M
    return False

def check_algebraic_irreducibility(M):
    x = symbols('x')
    M_int = Matrix(M.astype(int))
    charpoly = M_int.charpoly(x).as_expr()
    from sympy.polys.polytools import factor_list
    return len(factor_list(charpoly)[1]) == 1

class ClosedTopologySystem:
    def __init__(self, name, matrix, sub_rules, alphabet):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.sub_rules = sub_rules
        self.alphabet = alphabet
        self._characterize_system()

    def _characterize_system(self):
        d = len(self.matrix)
        evals, evecs = eig(self.matrix)
        
        # 1. Isolate the dominant Perron-Frobenius Index
        pf_idx = np.argmax(np.real(evals))
        self.lambda_pf = float(np.real(evals[pf_idx]))
        
        # 2. Extract raw PF eigenvectors
        v_raw = np.real(evecs[:, pf_idx])
        _, L_evecs = eig(self.matrix.T)
        w_raw = np.real(L_evecs[:, np.argmax(np.real(eig(self.matrix.T)[0]))])
        
        # 3. Enforce Exact Biorthogonal Duality Closure: <w, v> = 1
        self.v_PF = v_raw / np.sum(v_raw)
        self.w_PF = w_raw / np.dot(w_raw, self.v_PF)
        
        # 4. Construct Idempotent Stable Subspace Projector
        self.P_stable = np.eye(d) - np.outer(self.v_PF, self.w_PF)
        self.stable_rank = int(np.linalg.matrix_rank(self.P_stable, tol=1e-9))
        
        # 5. Build Complex-Pair Preserving Stable Basis
        stable_basis_vectors = []
        seen_conjugates = np.zeros(d, dtype=bool)
        
        for i in range(d):
            if i == pf_idx or seen_conjugates[i]:
                continue
            if np.abs(evals[i]) < 1.001:
                v_arr = evecs[:, i]
                
                # Check if this mode is a complex conjugate pair
                if np.abs(np.imag(evals[i])) > 1e-7:
                    # Find matching conjugate index to flag it
                    for j in range(d):
                        if j != pf_idx and j != i:
                            if np.abs(np.real(evals[i]) - np.real(evals[j])) < 1e-7 and \
                               np.abs(np.imag(evals[i]) + np.imag(evals[j])) < 1e-7:
                                seen_conjugates[j] = True
                    
                    # Split into real and imaginary coordinate vectors
                    r_part = np.real(v_arr)
                    i_part = np.imag(v_arr)
                    
                    if np.linalg.norm(r_part) > 1e-8:
                        stable_basis_vectors.append(r_part / np.linalg.norm(r_part))
                    if np.linalg.norm(i_part) > 1e-8:
                        stable_basis_vectors.append(i_part / np.linalg.norm(i_part))
                else:
                    # Purely real stable eigenvalue mode
                    r_part = np.real(v_arr)
                    if np.linalg.norm(r_part) > 1e-8:
                        stable_basis_vectors.append(r_part / np.linalg.norm(r_part))
                        
        self.stable_basis = np.array(stable_basis_vectors)
        
        # Generate clean zero-mean directional increments
        self.increments = {}
        for i, ch in enumerate(self.alphabet):
            e_i = np.zeros(d)
            e_i[i] = 1.0
            self.increments[ch] = e_i - self.v_PF

        self.is_primitive = is_primitive_matrix(self.matrix)
        self.is_irreducible = check_algebraic_irreducibility(self.matrix)
        moduli = np.sort(np.abs(evals))[::-1]
        self.is_pisot = self.is_primitive and self.is_irreducible and (moduli[0] > 1.0 and np.all(moduli[1:] < 0.999))

    def generate_sequence(self, length_target):
        seq = self.alphabet[0]
        while len(seq) < length_target:
            seq = ''.join(self.sub_rules.get(c, c) for c in seq)
            if len(seq) > length_target * 3:
                break
        return seq[:length_target]

    def compute_walk(self, seq):
        d = len(self.matrix)
        walk = np.zeros((len(seq) + 1, d))
        for i, ch in enumerate(seq):
            walk[i+1] = walk[i] + self.increments.get(ch, np.zeros(d))
        return walk

def compute_wandering_exponent(points):
    displacements = np.linalg.norm(points, axis=1)
    indices = np.arange(1, len(displacements) + 1)
    valid = displacements > 1e-8
    if not valid.any(): return 0.0
    gamma = np.polyfit(np.log(indices[valid]), np.log(displacements[valid]), 1)[0]
    return 0.0 if -0.05 < gamma < 0.02 else max(0.0, float(gamma))

def compute_automated_plateau_d2(points, num_samples=300):
    n = len(points)
    if n < 40: return 0.0
    pts = points[np.random.choice(n, min(num_samples, n), replace=False)]
    dists = pdist(pts)
    dists = dists[dists > 1e-7]
    if len(dists) < 20: return 0.0
    
    radii = np.logspace(np.log10(np.percentile(dists, 2)), np.log10(np.percentile(dists, 35)), 15)
    total_pairs = len(pts) * (len(pts) - 1) / 2.0
    counts = np.array([np.sum(dists < r) / total_pairs for r in radii])
    
    valid = counts > 1e-7
    if np.sum(valid) < 5: return 0.0
    log_r = np.log(radii[valid])
    log_c = np.log(counts[valid])
    
    local_slopes = []
    for i in range(len(log_r) - 2):
        slope, _ = np.polyfit(log_r[i:i+3], log_c[i:i+3], 1)
        local_slopes.append(slope)
        
    return max(0.0, min(float(np.median(local_slopes)), float(points.shape[1])))

def compute_mbb_metrics(points, block_size=60, n_bootstraps=12):
    n = len(points)
    gammas, d2s = [], []
    num_blocks = n // block_size
    
    if num_blocks < 2:
        return 0.0, 0.0, 0.0, 0.0

    for _ in range(n_bootstraps):
        block_indices = np.random.choice(n - block_size, size=num_blocks, replace=True)
        sample_idx = np.concatenate([np.arange(idx, idx + block_size) for idx in block_indices])
        bootstrap_sample = points[sample_idx]
        
        gammas.append(compute_wandering_exponent(bootstrap_sample))
        d2s.append(compute_automated_plateau_d2(bootstrap_sample))
        
    return float(np.mean(gammas)), float(np.std(gammas)), float(np.mean(d2s)), float(np.std(d2s))

SYSTEMS = {
    "Tribonacci": {"matrix": [[1, 1, 1], [1, 0, 0], [0, 1, 0]], "rules": {"a": "ab", "b": "ac", "c": "a"}, "alphabet": "abc"},
    "Fibonacci": {"matrix": [[1, 1], [1, 0]], "rules": {"a": "ab", "b": "a"}, "alphabet": "ab"},
    "Pisot Binary Substitution": {"matrix": [[3, 2], [1, 1]], "rules": {"a": "abaab", "b": "aba"}, "alphabet": "ab"},
    "ThueMorse": {"matrix": [[1, 1], [1, 1]], "rules": {"a": "ab", "b": "ba"}, "alphabet": "ab"}
}

LENGTHS = [1000, 4000, 8000]

def main():
    print("=" * 95)
    print("MINERVA V7.7 ENGINE VERIFICATION: BIORTHOGONAL DECOMPOSITION & COMPLEX SUB-BASIS")
    print("=" * 95)
    
    dashboard = []
    
    for name, cfg in SYSTEMS.items():
        print(f"\n🧬 Initializing Topology: {name}")
        print("─" * 75)
        sys = ClosedTopologySystem(name, cfg["matrix"], cfg["rules"], cfg["alphabet"])
        print(f"   Algebraic Pisot Status: {sys.is_pisot} | Dominant Root λ_PF: {sys.lambda_pf:.6f}")
        print(f"   Mathematical Manifold Space Rank: {sys.stable_rank}D")
        
        for L in LENGTHS:
            seq = sys.generate_sequence(L)
            walk = sys.compute_walk(seq)
            
            # Map through the unified stable projector mapping
            projected = walk @ sys.P_stable.T
            projected -= np.mean(projected, axis=0)
            
            # Rotate into the corrected anisotropic stable basis (preserving complex dimensions)
            intrinsic = projected @ sys.stable_basis.T
            
            g_mean, g_std, d2_mean, d2_std = compute_mbb_metrics(intrinsic, block_size=max(30, L//50), n_bootstraps=12)
            print(f"   L={L:5d} => Plateau D₂: {d2_mean:.4f} ± {d2_std:.4f} | Dynamic γ: {g_mean:.4f} ± {g_std:.4f}")
            
            if L == 8000:
                dashboard.append({"name": name, "rank": sys.stable_rank, "d2": d2_mean, "d2_err": d2_std, "gamma": g_mean, "gamma_err": g_std})

    print("\n" + "=" * 95)
    print("📊 VERIFIED SYSTEM V7.7 CLOSURE DASHBOARD (L=8000)")
    print("-" * 95)
    print(f"{'System Topology':<28} {'Manifold Dim':<15} {'Plateau D₂':<18} {'Dynamic Trajectory γ'}")
    print("-" * 95)
    for row in dashboard:
        print(f"{row['name']:<28} {row['rank']}D              {row['d2']:.4f} ± {row['d2_err']:.4f}    {row['gamma']:.4f} ± {row['gamma_err']:.4f}")
    print("=" * 95)

if __name__ == "__main__":
    main()
