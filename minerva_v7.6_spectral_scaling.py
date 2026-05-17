#!/usr/bin/env python3
"""
MINERVA V7.6: LYAPUNOV SPECTRAL SCALING & LOCAL PLATEAU DETECTION
==================================================================
- Moving Block Bootstrap (MBB) to preserve sequential walk dynamics.
- Local Slope Derivative Plateau Detection for Correlation Dimension D₂.
- Projections along specific stable algebraic directional axes.
- Robust Stable Rank detection using structural matrix rank rules.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from scipy.spatial.distance import pdist
from sympy import Matrix, symbols
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ALGEBRAIC CORE & RIGOROUS RANK EXTRACTION
# ============================================================================

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

class IntrinsicSubstitutionSystemV76:
    def __init__(self, name, matrix, sub_rules, alphabet):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.sub_rules = sub_rules
        self.alphabet = alphabet
        self._characterize_system()

    def _characterize_system(self):
        d = len(self.matrix)
        evals, evecs = eig(self.matrix)
        real_parts = np.real(evals)
        pf_idx = np.argmax(real_parts)
        
        self.evals = evals
        self.evecs = evecs
        self.lambda_pf = float(real_parts[pf_idx])
        
        # Right & Left PF Eigenvectors
        self.v_PF = np.abs(np.real(evecs[:, pf_idx]))
        _, L_evecs = eig(self.matrix.T)
        w_PF = np.abs(np.real(L_evecs[:, np.argmax(np.real(eig(self.matrix.T)[0]))]))
        self.w_PF = w_PF / np.dot(w_PF, self.v_PF)
        self.v_PF = self.v_PF / np.sum(self.v_PF)

        # Canonical Step Offsets
        self.increments = {}
        for i, ch in enumerate(self.alphabet):
            e_i = np.zeros(d)
            e_i[i] = 1.0
            self.increments[ch] = e_i - self.v_PF

        self.is_primitive = is_primitive_matrix(self.matrix)
        self.is_irreducible = check_algebraic_irreducibility(self.matrix)
        
        # Proper Strict Pisot Validation
        moduli = np.sort(np.abs(evals))[::-1]
        self.is_pisot = self.is_primitive and self.is_irreducible and (moduli[0] > 1.0 and np.all(moduli[1:] < 0.999))
        
        # Rigorous Structural Rank definition
        v = self.v_PF.reshape(-1, 1)
        w = self.w_PF.reshape(1, -1)
        self.P_stable = np.eye(d) - (v @ w)
        self.stable_rank = int(np.linalg.matrix_rank(self.P_stable, tol=1e-9))

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

# ============================================================================
# 2. LOCAL SLOPE PLATEAU DETECTION & METRICS
# ============================================================================

def compute_wandering_exponent(points):
    displacements = np.linalg.norm(points, axis=1)
    indices = np.arange(1, len(displacements) + 1)
    valid = displacements > 1e-8
    if not valid.any(): return 0.0
    gamma = np.polyfit(np.log(indices[valid]), np.log(displacements[valid]), 1)[0]
    return 0.0 if -0.05 < gamma < 0.02 else max(0.0, float(gamma))

def compute_local_plateau_d2(points, num_samples=300):
    n = len(points)
    if n < 40: return 0.0
    pts = points[np.random.choice(n, min(num_samples, n), replace=False)]
    dists = pdist(pts)
    dists = dists[dists > 1e-7]
    if len(dists) < 20: return 0.0
    
    radii = np.logspace(np.log10(np.percentile(dists, 2)), np.log10(np.percentile(dists, 40)), 18)
    total_pairs = len(pts) * (len(pts) - 1) / 2.0
    counts = np.array([np.sum(dists < r) / total_pairs for r in radii])
    
    valid = counts > 1e-7
    if np.sum(valid) < 5: return 0.0
    log_r = np.log(radii[valid])
    log_c = np.log(counts[valid])
    
    # Take localized derivatives (local slopes)
    slopes = []
    for i in range(len(log_r) - 2):
        slope, _ = np.polyfit(log_r[i:i+3], log_c[i:i+3], 1)
        slopes.append(slope)
        
    # The Plateau Region is isolated through the median of central derivatives
    return max(0.0, min(float(np.median(slopes)), float(points.shape[1])))

# ============================================================================
# 3. MOVING BLOCK BOOTSTRAP (MBB) FOR TRAJECTORIES
# ============================================================================

def moving_block_bootstrap(points, block_size=50, n_bootstraps=10):
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
        d2s.append(compute_local_plateau_d2(bootstrap_sample))
        
    return float(np.mean(gammas)), float(np.std(gammas)), float(np.mean(d2s)), float(np.std(d2s))

# ============================================================================
# 4. SPECTRAL ANISOTROPIC SCALING
# ============================================================================

def analyze_anisotropic_scaling(system, walk):
    # Base Projected Stable Frame
    projected = walk @ system.P_stable.T
    projected -= np.mean(projected, axis=0)
    
    # Compress via SVD to intrinsic space dimensions
    U, S, Vt = svd(system.P_stable, full_matrices=False)
    if system.stable_rank >= 1:
        intrinsic = projected @ Vt[:system.stable_rank].T
    else:
        intrinsic = projected
        
    return intrinsic

# ============================================================================
# 5. EXECUTION ENGINE
# ============================================================================

SYSTEMS = {
    "Tribonacci": {"matrix": [[1, 1, 1], [1, 0, 0], [0, 1, 0]], "rules": {"a": "ab", "b": "ac", "c": "a"}, "alphabet": "abc"},
    "Fibonacci": {"matrix": [[1, 1], [1, 0]], "rules": {"a": "ab", "b": "a"}, "alphabet": "ab"},
    "Pisot Binary Substitution": {"matrix": [[3, 2], [1, 1]], "rules": {"a": "abaab", "b": "aba"}, "alphabet": "ab"},
    "ThueMorse": {"matrix": [[1, 1], [1, 1]], "rules": {"a": "ab", "b": "ba"}, "alphabet": "ab"}
}

LENGTHS = [500, 1000, 2000, 4000, 8000]

def main():
    print("=" * 90)
    print("MINERVA V7.6: SPECTRAL SCALING & MOVING BLOCK LOCAL PLATEAU DERIVATIVES")
    print("=" * 90)
    
    summary_data = []
    
    for name, cfg in SYSTEMS.items():
        print(f"\n📐 System: {name}")
        print("─" * 70)
        sys_obj = IntrinsicSubstitutionSystemV76(name, cfg["matrix"], cfg["rules"], cfg["alphabet"])
        print(f"   Structural Stable Rank: {sys_obj.stable_rank}D | Is Strict Pisot: {sys_obj.is_pisot}")
        
        for L in LENGTHS:
            seq = sys_obj.generate_sequence(L)
            walk = sys_obj.compute_walk(seq)
            intrinsic = analyze_anisotropic_scaling(sys_obj, walk)
            
            g_m, g_s, d2_m, d2_s = moving_block_bootstrap(intrinsic, block_size=max(20, L//40), n_bootstraps=10)
            print(f"   L={L:5d} ── Local Plateau D₂ = {d2_m:.4f} ± {d2_s:.4f} | Trajectory γ = {g_m:.4f} ± {g_s:.4f}")
            
            if L == 8000:
                summary_data.append({"sys": name, "d2": d2_m, "d2_err": d2_s, "gamma": g_m, "gamma_err": g_s, "rank": sys_obj.stable_rank})

    print("\n" + "=" * 90)
    print("📊 FINAL V7.6 ANISOTROPIC ANALYSIS DASHBOARD")
    print("-" * 90)
    print(f"{'System':<25} {'Rank':<6} {'Plateau D₂':<18} {'Trajectory γ':<18}")
    print("-" * 90)
    for s in summary_data:
        print(f"{s['sys']:<25} {s['rank']}D     {s['d2']:.4f} ± {s['d2_err']:.4f}    {s['gamma']:.4f} ± {s['gamma_err']:.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()
