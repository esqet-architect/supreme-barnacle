"""
MINERVA V6.9 (Fixed): STABLE COCYCLE & SPECTRAL ANALYSIS
=======================================================
- Fixed: Resolved NumPy boolean ambiguity error in analyze_pisot_spectrum
- Fixed: Added explicit length verification for the remainder moduli array
- Preserved: Real Schur projection, canonical centering, and exponent clamping
"""

import numpy as np
import pandas as pd
from scipy.linalg import schur, eig, qr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. ALGEBRAIC STRUCTURE (Primitivity + Pisot)
# ============================================================================

def is_primitive_matrix(M, max_power=12):
    M_k = np.copy(M)
    for _ in range(max_power):
        if np.all(M_k > 0):
            return True
        M_k = M_k @ M
    return False

def analyze_pisot_spectrum(evals):
    """Check Pisot condition: dominant > 1, all others < 1 in modulus."""
    moduli = np.abs(evals)
    sorted_mod = np.sort(moduli)[::-1]
    dominant = sorted_mod[0]
    rest = sorted_mod[1:] if len(sorted_mod) > 1 else []
    
    # Safe array evaluation bypassing Python's implicit truth check
    has_pisot_spectrum = (dominant > 1.0) and (np.max(rest) < 0.99 if len(rest) > 0 else True)
    return has_pisot_spectrum, dominant, rest

# ============================================================================
# 2. SUBSTITUTION SYSTEM
# ============================================================================

def get_substitution_matrix(sub_dict, alphabet):
    d = len(alphabet)
    M = np.zeros((d, d), dtype=float)
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    for j, ch in enumerate(alphabet):
        target = sub_dict.get(ch, "")
        for bit in target:
            if bit in char_to_idx:
                M[char_to_idx[bit], j] += 1
    return M

def get_intrinsic_cocycles(M, alphabet):
    evals, evecs = eig(M)
    real_parts = np.real(evals)
    pf_idx = np.argmax(real_parts)
    pf_evec = np.abs(np.real(evecs[:, pf_idx]))
    f = pf_evec / np.sum(pf_evec)

    increments = {}
    for i, ch in enumerate(alphabet):
        e_i = np.zeros(len(M))
        e_i[i] = 1.0
        increments[ch] = e_i - f

    is_pisot, dom, _ = analyze_pisot_spectrum(evals)
    return increments, f, is_pisot, evals, dom

def generate_sequence(sub_dict, seed, steps=12, max_len=50000):
    seq = seed
    for _ in range(steps):
        seq = ''.join(sub_dict.get(c, c) for c in seq)
        if len(seq) > max_len:
            seq = seq[:max_len]
            break
    return seq

def compute_walk(seq, increments):
    d = len(next(iter(increments.values())))
    walk = np.zeros((len(seq) + 1, d))
    for i, ch in enumerate(seq):
        walk[i+1] = walk[i] + increments.get(ch, np.zeros(d))
    return walk

# ============================================================================
# 3. SPECTRAL PROJECTION (SCHUR DECOMPOSITION)
# ============================================================================

def project_via_schur(walk, M):
    d = len(M)
    T, Z = schur(M, output='real')

    diag_abs = np.abs(np.diag(T))
    pf_block_idx = np.argmax(diag_abs)

    stable_indices = [i for i in range(d) if i != pf_block_idx]

    if d == 2 and len(stable_indices) >= 1:
        proj_space = Z[:, stable_indices[0]].reshape(1, d)
    elif len(stable_indices) >= 2:
        proj_space = Z[:, stable_indices[:2]].T
    else:
        proj_space = np.eye(2, d)

    Q, _ = qr(proj_space.T, mode='economic')
    return walk @ Q

# ============================================================================
# 4. GEOMETRIC METRICS
# ============================================================================

def compute_wandering_exponent(points):
    displacements = np.linalg.norm(points, axis=1)
    indices = np.arange(1, len(displacements) + 1)

    valid = displacements > 1e-8
    if not valid.any():
        return 0.0

    coeffs = np.polyfit(np.log(indices[valid]), np.log(displacements[valid]), 1)
    gamma = coeffs[0]

    if -0.05 < gamma < 0.0:
        gamma = 0.0
    return max(0.0, gamma)

def compute_correlation_dimension(points, num_samples=600):
    n = len(points)
    if n < 50:
        return 0.0

    if n > num_samples:
        idx = np.random.choice(n, num_samples, replace=False)
        pts = points[idx]
    else:
        pts = points

    N = len(pts)
    if N < 30:
        return 0.0

    from scipy.spatial.distance import pdist
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
    return max(0.0, min(d2, 2.0))

# ============================================================================
# 5. SYSTEM CONFIGURATIONS
# ============================================================================

SYSTEMS = {
    "Tribonacci": {"sub": {"a": "ab", "b": "ac", "c": "a"}, "seed": "a", "alphabet": "abc", "steps": 11},
    "Fibonacci": {"sub": {"a": "ab", "b": "a"}, "seed": "a", "alphabet": "ab", "steps": 14},
    "Sturmian": {"sub": {"a": "abaab", "b": "aba"}, "seed": "a", "alphabet": "ab", "steps": 7},
    "ThueMorse": {"sub": {"a": "ab", "b": "ba"}, "seed": "a", "alphabet": "ab", "steps": 11}
}

def main():
    print("=" * 75)
    print("MINERVA V6.9 (FIXED): STABLE COCYCLE & SPECTRAL ANALYSIS")
    print("Canonical: v_i = e_i - f | Schur Projection | GP Dimension")
    print("=" * 75)

    results = []

    for name, config in SYSTEMS.items():
        print(f"\n{'─' * 55}")
        print(f"📐 System: {name}")
        print(f"{'─' * 55}")

        M = get_substitution_matrix(config["sub"], config["alphabet"])
        increments, f, is_pisot, evals, dom = get_intrinsic_cocycles(M, config["alphabet"])
        is_primitive = is_primitive_matrix(M)

        print(f"   Primitive:        {is_primitive}")
        print(f"   Pisot (strict):   {is_pisot}")
        print(f"   λ_PF:             {dom:.6f}")

        seq = generate_sequence(config["sub"], config["seed"], config["steps"])
        walk = compute_walk(seq, increments)
        print(f"   Sequence length:  {len(seq)}")

        projected = project_via_schur(walk, M)
        dim = projected.shape[1]
        print(f"   Subspace dim:     {dim}D")

        gamma = compute_wandering_exponent(projected)
        d2 = compute_correlation_dimension(projected)

        print(f"   Wandering γ:      {gamma:.4f}")
        print(f"   Correlation D₂:   {d2:.4f}")

        if is_pisot:
            if dim == 1:
                print("   → 1D Interval Exchange / Codimension-1 Cut-and-Project")
            elif dim == 2:
                if d2 > 1.5:
                    print("   → Planar Rauzy Fractal Attractor (compact, 2D)")
                else:
                    print("   → Lower-dimensional planar structure")
        else:
            print("   → Non-Pisot System")

        results.append({
            "System": name,
            "Primitive": is_primitive,
            "Pisot": is_pisot,
            "Dominant_Eigenvalue": dom,
            "Subspace_Dim": dim,
            "Wandering_Gamma": gamma,
            "Correlation_D2": d2,
            "Sequence_Length": len(seq)
        })

    print("\n📊 SUMMARY")
    print("-" * 75)
    print(f"{'System':<12} {'Pisot':<8} {'Dim':<5} {'γ':<8} {'D₂':<8} {'Classification'}")
    print("-" * 75)
    for r in results:
        if r["Pisot"]:
            cls = "Rauzy fractal" if (r["Subspace_Dim"] == 2 and r["Correlation_D2"] > 1.5) else "1D interval exchange"
        else:
            cls = "Non-Pisot"
        print(f"{r['System']:<12} {str(r['Pisot']):<8} {r['Subspace_Dim']:<5} {r['Wandering_Gamma']:<8.4f} {r['Correlation_D2']:<8.4f} {cls}")
    print("=" * 75)

if __name__ == "__main__":
    main()
