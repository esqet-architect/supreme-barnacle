#!/usr/bin/env python3
"""
MINERVA V7.5: CONVERGENCE ANALYSIS & ERROR ESTIMATION
=======================================================
- Length sweeps (N = 500, 1000, 2000, 4000, 8000)
- Bootstrap confidence intervals for D₂ and γ
- Scaling stability plots (D₂ vs N, γ vs N)
- Preserves: exact projector, centering, SVD compression to E^s
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig, svd
from scipy.spatial.distance import pdist
from sympy import Matrix, symbols
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ALGEBRAIC STRUCTURE CORE (EXACT OVER Q)
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
    factors = factor_list(charpoly)[1]
    return len(factors) == 1


def analyze_pisot_spectrum(evals):
    moduli = np.abs(evals)
    sorted_mod = np.sort(moduli)[::-1]
    dominant = sorted_mod[0]
    rest = sorted_mod[1:] if len(sorted_mod) > 1 else []
    has_pisot_spectrum = (dominant > 1.0) and (np.max(rest) < 0.99 if len(rest) > 0 else True)
    return has_pisot_spectrum, dominant


class IntrinsicSubstitutionSystem:
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
        pf_eval = evals[pf_idx]
        v_PF = np.abs(np.real(evecs[:, pf_idx]))
        
        _, L_evecs = eig(self.matrix.T)
        real_parts_L = np.real(evals)
        pf_idx_L = np.argmax(real_parts_L)
        w_PF = np.abs(np.real(L_evecs[:, pf_idx_L]))
        w_PF = w_PF / np.dot(w_PF, v_PF)
        
        self.v_PF = v_PF / np.sum(v_PF)
        self.w_PF = w_PF
        self.lambda_pf = float(pf_eval.real)
        
        self.increments = {}
        for i, ch in enumerate(self.alphabet):
            e_i = np.zeros(len(self.matrix))
            e_i[i] = 1.0
            self.increments[ch] = e_i - self.v_PF

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
# 2. INTRINSIC STABLE SUBSPACE COMPRESSION
# ============================================================================

def extract_intrinsic_coordinates(system, walk):
    d = len(system.matrix)
    v = system.v_PF.reshape(-1, 1)
    w = system.w_PF.reshape(1, -1)
    P_stable = np.eye(d) - (v @ w) / (w @ v).item()
    
    projected = walk @ P_stable.T
    projected = projected - np.mean(projected, axis=0)
    
    U, S, Vt = svd(P_stable, full_matrices=False)
    stable_rank = np.sum(S / S[0] > 1e-10)
    
    if stable_rank >= 1:
        basis = Vt[:stable_rank].T
        intrinsic = projected @ basis
    else:
        intrinsic = projected
        stable_rank = projected.shape[1]
    
    return intrinsic, stable_rank


# ============================================================================
# 3. METRICS WITH BOOTSTRAP
# ============================================================================

def compute_wandering_exponent(points):
    displacements = np.linalg.norm(points, axis=1)
    indices = np.arange(1, len(displacements) + 1)
    valid = displacements > 1e-8
    if not valid.any():
        return 0.0
    coeffs = np.polyfit(np.log(indices[valid]), np.log(displacements[valid]), 1)
    gamma = coeffs[0]
    if -0.05 < gamma < 0.02:
        gamma = 0.0
    return max(0.0, float(gamma))


def compute_correlation_dimension(points, num_samples=400):
    n = len(points)
    if n < 50:
        return 0.0
    pts = points[np.random.choice(n, min(num_samples, n), replace=False)]
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
    radii = np.logspace(np.log10(r_min), np.log10(r_max), 12)
    total_pairs = N * (N - 1) / 2.0
    counts = np.array([np.sum(dists < r) / total_pairs for r in radii])
    valid = counts > 1e-8
    if np.sum(valid) < 4:
        return 0.0
    log_r = np.log(radii[valid])
    log_c = np.log(counts[valid])
    slopes = []
    for i in range(len(log_r) - 3):
        slope, _ = np.polyfit(log_r[i:i+4], log_c[i:i+4], 1)
        slopes.append(slope)
    d2 = np.median(slopes) if slopes else 0.0
    return max(0.0, min(float(d2), 2.0))


def bootstrap_metrics(points, n_bootstrap=20, sample_frac=0.8):
    """Bootstrap confidence intervals for γ and D₂"""
    n = len(points)
    gammas = []
    d2s = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, int(n * sample_frac), replace=True)
        sample = points[idx]
        
        gamma = compute_wandering_exponent(sample)
        d2 = compute_correlation_dimension(sample)
        
        if gamma > 1e-6:
            gammas.append(gamma)
        if d2 > 1e-6:
            d2s.append(d2)
    
    gamma_mean = np.mean(gammas) if gammas else 0.0
    gamma_std = np.std(gammas) if gammas else 0.0
    d2_mean = np.mean(d2s) if d2s else 0.0
    d2_std = np.std(d2s) if d2s else 0.0
    
    return gamma_mean, gamma_std, d2_mean, d2_std


# ============================================================================
# 4. LENGTH CONVERGENCE SWEEP
# ============================================================================

def convergence_sweep(system_obj, lengths=[500, 1000, 2000, 4000, 8000], n_bootstrap=15):
    """Run metrics across different sequence lengths"""
    results = []
    
    for L in lengths:
        # Generate sequence of appropriate length
        seq = system_obj.generate_sequence(steps=max(7, int(np.log2(L)) + 5))
        if len(seq) < L:
            # Extend if needed
            steps = 1
            while len(seq) < L:
                seq = ''.join(system_obj.sub_rules.get(c, c) for c in seq)
                steps += 1
                if len(seq) > L * 2:
                    break
        seq = seq[:L]
        
        walk = system_obj.compute_walk(seq)
        intrinsic, stable_rank = extract_intrinsic_coordinates(system_obj, walk)
        
        if len(intrinsic) < 100:
            continue
        
        gamma_mean, gamma_std, d2_mean, d2_std = bootstrap_metrics(intrinsic, n_bootstrap=n_bootstrap)
        
        results.append({
            'length': L,
            'gamma': gamma_mean,
            'gamma_std': gamma_std,
            'd2': d2_mean,
            'd2_std': d2_std,
            'stable_rank': stable_rank
        })
    
    return results


# ============================================================================
# 5. SYSTEM CONFIGURATIONS
# ============================================================================

SYSTEMS = {
    "Tribonacci": {
        "matrix": [[1, 1, 1], [1, 0, 0], [0, 1, 0]],
        "rules": {"a": "ab", "b": "ac", "c": "a"},
        "alphabet": "abc",
    },
    "Fibonacci": {
        "matrix": [[1, 1], [1, 0]],
        "rules": {"a": "ab", "b": "a"},
        "alphabet": "ab",
    },
    "Pisot Binary Substitution": {
        "matrix": [[3, 2], [1, 1]],
        "rules": {"a": "abaab", "b": "aba"},
        "alphabet": "ab",
    },
    "ThueMorse": {
        "matrix": [[1, 1], [1, 1]],
        "rules": {"a": "ab", "b": "ba"},
        "alphabet": "ab",
    }
}

LENGTHS = [500, 1000, 2000, 4000, 8000]


# ============================================================================
# 6. MAIN EXECUTION & PLOTTING
# ============================================================================

def main():
    print("=" * 85)
    print("MINERVA V7.5: CONVERGENCE ANALYSIS & ERROR ESTIMATION")
    print("Length sweep + bootstrap confidence intervals")
    print("=" * 85)
    
    all_results = {}
    convergence_data = []
    
    for name, cfg in SYSTEMS.items():
        print(f"\n📐 System: {name}")
        print("─" * 65)
        
        sys_obj = IntrinsicSubstitutionSystem(name, cfg["matrix"], cfg["rules"], cfg["alphabet"])
        
        print(f"   Strict Pisot: {sys_obj.is_pisot}")
        print(f"   λ_PF: {sys_obj.lambda_pf:.6f}")
        
        # Convergence sweep
        sweep_results = convergence_sweep(sys_obj, lengths=LENGTHS, n_bootstrap=12)
        all_results[name] = sweep_results
        
        # Display convergence
        for r in sweep_results:
            print(f"   L={r['length']:5d}: γ={r['gamma']:.4f}±{r['gamma_std']:.4f}, D₂={r['d2']:.4f}±{r['d2_std']:.4f}")
            convergence_data.append({
                "System": name,
                "Length": r['length'],
                "Gamma": r['gamma'],
                "Gamma_Std": r['gamma_std'],
                "D2": r['d2'],
                "D2_Std": r['d2_std'],
                "Pisot": sys_obj.is_pisot
            })
    
    # Save CSV
    df = pd.DataFrame(convergence_data)
    df.to_csv("minerva_v7.5_convergence.csv", index=False)
    
    # Plot convergence
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = {'Tribonacci': 'blue', 'Fibonacci': 'green', 
              'Pisot Binary Substitution': 'orange', 'ThueMorse': 'red'}
    
    for name in SYSTEMS.keys():
        data = [r for r in convergence_data if r["System"] == name]
        if not data:
            continue
        lengths = [r["Length"] for r in data]
        gammas = [r["Gamma"] for r in data]
        gamma_err = [r["Gamma_Std"] for r in data]
        d2s = [r["D2"] for r in data]
        d2_err = [r["D2_Std"] for r in data]
        
        axes[0].errorbar(lengths, gammas, yerr=gamma_err, label=name, 
                        color=colors.get(name, 'gray'), marker='o', capsize=3)
        axes[1].errorbar(lengths, d2s, yerr=d2_err, label=name,
                        color=colors.get(name, 'gray'), marker='s', capsize=3)
    
    axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Wandering Exponent γ')
    axes[0].set_title('Convergence of γ with Sequence Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Correlation Dimension D₂')
    axes[1].set_title('Convergence of D₂ with Sequence Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('minerva_v7.5_convergence.png', dpi=150)
    print("\n📊 Convergence plot saved: minerva_v7.5_convergence.png")
    
    print("\n" + "=" * 85)
    print("📊 CONVERGENCE SUMMARY (L=8000)")
    print("-" * 85)
    for name in SYSTEMS.keys():
        data = [r for r in convergence_data if r["System"] == name and r["Length"] == 8000]
        if data:
            r = data[0]
            print(f"{name:<25}: γ={r['Gamma']:.4f}±{r['Gamma_Std']:.4f}, D₂={r['D2']:.4f}±{r['D2_Std']:.4f}")
    print("=" * 85)


if __name__ == "__main__":
    main()
