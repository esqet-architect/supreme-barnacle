"""
MINERVA V6.4 FINAL: Intrinsic Cocycle & Spectral Analysis
==========================================================
- v_i = e_i - f (canonical centered abelianization cocycle)
- Spectral projection onto contracting subspace (stable plane)
- Wandering exponent γ: |X_n| ~ n^γ
- Correlation dimension D₂: C(r) ~ r^{D₂}
- Strict Pisot validation (|λ_PF|>1, |λ_i|<1 for i≠PF)

Author: ESQET / Minerva Framework
"""

import numpy as np
import pandas as pd
from scipy.linalg import eig, qr
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. SUBSTITUTION MATRIX & PERRON-FROBENIUS EXTRACTION
# ============================================================================

def get_substitution_matrix(sub_dict, alphabet):
    """Construct incidence matrix from substitution rules."""
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
    """
    Compute intrinsic increments: v_i = e_i - f
    where f is the normalized Perron-Frobenius eigenvector.
    """
    d = len(M)
    evals, evecs = eig(M)
    
    # Perron-Frobenius extraction (real part, positive eigenvector)
    real_parts = np.real(evals)
    pf_idx = np.argmax(real_parts)
    pf_eval = real_parts[pf_idx]
    pf_evec = np.abs(np.real(evecs[:, pf_idx]))
    
    # Normalize to frequency vector f
    f = pf_evec / np.sum(pf_evec)
    
    # Intrinsic increments: v_i = e_i - f
    increments = {}
    for i, ch in enumerate(alphabet):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        increments[ch] = e_i - f
    
    # Strict Pisot condition: dominant > 1, all other |λ| < 1
    all_moduli = np.abs(evals)
    sorted_mod = np.sort(all_moduli)[::-1]
    dominant = sorted_mod[0]
    rest = sorted_mod[1:]
    is_pisot = (dominant > 1.0) and (np.max(rest) < 0.99 if len(rest) > 0 else True)
    
    return increments, f, is_pisot, evals, evecs


# ============================================================================
# 2. WORD GENERATION
# ============================================================================

def generate_sequence(sub_dict, seed, steps=12):
    """Generate substitution sequence up to reasonable length."""
    seq = seed
    for _ in range(steps):
        seq = ''.join(sub_dict.get(c, c) for c in seq)
        if len(seq) > 50000:  # Safety cap
            seq = seq[:50000]
            break
    return seq


def compute_walk(seq, increments):
    """Compute walk from increments."""
    d = len(next(iter(increments.values())))
    walk = np.zeros((len(seq) + 1, d))
    for i, ch in enumerate(seq):
        walk[i+1] = walk[i] + increments.get(ch, np.zeros(d))
    return walk


# ============================================================================
# 3. SPECTRAL PROJECTION (CONTRACTING SUBSPACE)
# ============================================================================

def project_spectrally(walk, evals, evecs):
    """
    Project onto the contracting subspace (stable plane) by removing
    the Perron-Frobenius direction and taking complex conjugate pairs.
    """
    d = len(evals)
    real_parts = np.real(evals)
    pf_idx = np.argmax(real_parts)
    
    # Find indices of non-dominant eigenvalues
    remaining = [i for i in range(d) if i != pf_idx]
    
    if len(remaining) >= 2:
        # Look for complex conjugate pair
        v1 = evecs[:, remaining[0]]
        if np.iscomplex(v1).any():
            # Use complex pair to form 2D contracting plane
            proj_space = np.vstack([np.real(v1), np.imag(v1)])
        else:
            # Use two smallest real eigenvectors
            v2 = evecs[:, remaining[1]]
            proj_space = np.vstack([np.real(v1), np.real(v2)])
    else:
        # Fallback: use first two coordinates (should not happen for 3D systems)
        proj_space = np.eye(2, d)
    
    # Orthonormalize projection basis
    Q, _ = qr(proj_space.T, mode='economic')
    return walk @ Q


# ============================================================================
# 4. GEOMETRIC METRICS
# ============================================================================

def compute_wandering_exponent(points):
    """
    Estimate wandering exponent γ from |X_n| ~ n^γ.
    γ = 0.5: diffusive, γ < 0.5: subdiffusive, γ ≈ 0: bounded.
    """
    displacements = np.linalg.norm(points, axis=1)
    n_steps = len(displacements)
    indices = np.arange(1, n_steps + 1)
    
    # Remove zeros for log fit
    valid = displacements > 1e-8
    if not valid.any():
        return 0.0
    
    log_n = np.log(indices[valid])
    log_r = np.log(displacements[valid])
    
    # Linear fit for scaling exponent
    coeffs = np.polyfit(log_n, log_r, 1)
    return coeffs[0]


def compute_correlation_dimension(points, num_samples=800):
    """
    Estimate correlation dimension D₂ from C(r) ~ r^{D₂}.
    Uses random subsampling for computational efficiency.
    """
    n = len(points)
    if n > num_samples:
        idx = np.random.choice(n, num_samples, replace=False)
        pts = points[idx]
    else:
        pts = points
    
    # Pairwise distances (correct axis: -1 for coordinate dimension)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-1).flatten()
    dists = dists[dists > 1e-8]  # Remove self-distances
    
    if len(dists) < 10:
        return 0.0
    
    # Scale window based on percentiles
    r_min = max(np.percentile(dists, 2), 1e-8)
    r_max = np.percentile(dists, 25)
    
    if r_min >= r_max:
        return 0.0
    
    radii = np.logspace(np.log10(r_min), np.log10(r_max), 15)
    counts = np.array([np.sum(dists < r) for r in radii])
    valid = counts > 0
    
    if np.sum(valid) < 4:
        return 0.0
    
    coeffs = np.polyfit(np.log(radii[valid]), np.log(counts[valid]), 1)
    return coeffs[0]


# ============================================================================
# 5. SYSTEM CONFIGURATIONS
# ============================================================================

SYSTEMS = {
    "Tribonacci": {
        "sub": {"a": "ab", "b": "ac", "c": "a"},
        "seed": "a",
        "alphabet": "abc",
        "steps": 11
    },
    "Fibonacci": {
        "sub": {"a": "ab", "b": "a"},
        "seed": "a",
        "alphabet": "ab",
        "steps": 14
    },
    "Sturmian": {
        "sub": {"a": "abaab", "b": "aba"},
        "seed": "a",
        "alphabet": "ab",
        "steps": 7
    },
    "ThueMorse": {
        "sub": {"a": "ab", "b": "ba"},
        "seed": "a",
        "alphabet": "ab",
        "steps": 11
    }
}


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("MINERVA V6.4 FINAL: Intrinsic Cocycle & Spectral Analysis")
    print("Canonical construction: v_i = e_i - f")
    print("=" * 70)
    
    results = []
    
    for name, config in SYSTEMS.items():
        print(f"\n{'─' * 50}")
        print(f"📐 System: {name}")
        print(f"{'─' * 50}")
        
        # Build matrix and cocycle
        M = get_substitution_matrix(config["sub"], config["alphabet"])
        increments, f, is_pisot, evals, evecs = get_intrinsic_cocycles(M, config["alphabet"])
        
        print(f"   Strict Pisot: {is_pisot}")
        print(f"   Dominant eigenvalue: {np.max(np.abs(evals)):.6f}")
        print(f"   PF frequencies: { {k: round(v, 4) for k, v in zip(config['alphabet'], f)} }")
        
        # Generate walk
        seq = generate_sequence(config["sub"], config["seed"], config["steps"])
        walk = compute_walk(seq, increments)
        print(f"   Sequence length: {len(seq)}")
        
        # Project to contracting plane
        projected = project_spectrally(walk, evals, evecs)
        
        # Compute metrics
        gamma = compute_wandering_exponent(projected)
        d2 = compute_correlation_dimension(projected)
        
        print(f"   Wandering exponent γ: {gamma:.4f}")
        print(f"   Correlation dimension D₂: {d2:.4f}")
        
        # Interpretation
        if is_pisot:
            if d2 < 1.5:
                print(f"   → Low-dimensional attractor (expected for Rauzy/Pisot)")
            else:
                print(f"   → Higher-dimensional structure")
        else:
            print(f"   → Non-Pisot: diffuse or self-similar without Rauzy fractal")
        
        results.append({
            "System": name,
            "Strict_Pisot": is_pisot,
            "Dominant_Eigenvalue": np.max(np.abs(evals)),
            "Wandering_Gamma": gamma,
            "Correlation_D2": d2,
            "Sequence_Length": len(seq)
        })
    
    # Export CSV
    df = pd.DataFrame(results)
    csv_path = "~/rauzy_analysis/minerva_v6.4_final_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print(f"📁 Results saved to: {csv_path}")
    print("=" * 70)
    print("\nInterpretation guide:")
    print("  γ ≈ 0.5 → diffusive")
    print("  γ < 0.5 → subdiffusive")
    print("  γ ≈ 0   → bounded (Rauzy fractal)")
    print("  D₂ ≈ 2  → space-filling")
    print("  D₂ < 1.5 → low-dimensional attractor")
    print("=" * 70)


if __name__ == "__main__":
    main()
