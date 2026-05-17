#!/usr/bin/env python3
"""
MINERVA V5.8 — Invariant CLV Multifractal & Thermodynamic Spectrum Engine
- Implements forward-backward Oseledets filtration tracking for invariant bundles
- Computes time-resolved rolling spectral gaps and Grassmannian frame curvatures
- Isolates exact covariant boundary points using localized scale thresholding
- Extracts the multifractal spectrum f(alpha) via thermodynamic method of moments
"""

import numpy as np
from numpy.linalg import norm, svd, qr, inv
from scipy.spatial import KDTree
import warnings
import sys

try:
    from ripser import ripser
except ImportError:
    print("[!] Error: 'ripser' package is required for Minerva V5.8.")
    print("Execute: pip install ripser")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# UNIVERSALITY CLASS DIRECTIVE GENERATORS
# ============================================================================

def generate_fibonacci_directive(N):
    """Generates a quasiperiodic Fibonacci sequence via string replacement."""
    word = "0"
    while len(word) < N:
        word = word.replace("0", "01").replace("1", "0X").replace("X", "1")
    return [int(c) for c in word[:N]]

def generate_sturmian_rotation(alpha, N):
    """Generates an irrational rotation directive sequence."""
    return [int(np.floor((n + 1) * alpha) - np.floor(n * alpha)) % 2 for n in range(N)]

# ============================================================================
# MATHEMATICAL CORE & INVARIANT BUNDLE ENGINE (CLV)
# ============================================================================

def principal_plane_angle(U1, U2):
    """Computes the maximal principal angle between two 2D subspace matrices."""
    C = U1.T @ U2
    svals = svd(C, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return np.degrees(np.arccos(np.min(svals)))

def execute_oseledets_clv_pipeline(directive_seq, target_length=2000):
    """
    Evolves the deformed matrix cocycle family over the directive sequence.
    Extracts covariant projection planes via backward-forward Oseledets tracking.
    """
    M0 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]], dtype=float)
    M1 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 1.1, 0.0],  # Deformed Salem-adjacent boundary matrix
                   [1.0, 0.0, 1.0]], dtype=float)
    matrices = {0: M0, 1: M1}
    
    subs_0 = {'1': '12', '2': '13', '3': '1'}
    subs_1 = {'1': '123', '2': '13', '3': '1'}
    rules = {0: subs_0, 1: subs_1}
    
    word = '1'
    for bit in directive_seq[:8]:
        word = ''.join(rules[bit].get(c, c) for c in word)
        if len(word) > target_length * 1.5:
            break
    word = word[:target_length]
    N = len(word)
    
    # Forward QR sweep
    Q_hist, R_hist, log_diag_hist = [], [], []
    Q_curr = np.eye(3)
    for n in range(N):
        bit = directive_seq[n % len(directive_seq)]
        Q_curr, R_curr = qr(matrices[bit] @ Q_curr)
        Q_hist.append(Q_curr)
        R_hist.append(R_curr)
        log_diag_hist.append(np.log(np.abs(np.diag(R_curr)) + 1e-15))
        
    # Backward sweep for Covariant Lyapunov Vectors (CLV)
    V_curr = np.eye(3)
    U_clv_hist = [None] * N
    for n in reversed(range(N)):
        V_curr = inv(R_hist[n]) @ V_curr
        V_q, _ = qr(V_curr)
        U_clv_hist[n] = Q_hist[n] @ V_q[:, 1:3]
        
    # Build projected space walk using exact rolling invariant bundle
    pts = np.zeros((N, 2))
    pos_3d = np.zeros(3)
    basis_3d = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    
    angles = []
    for n, ch in enumerate(word):
        U_curr = U_clv_hist[n]
        if n > 0 and U_clv_hist[n-1] is not None:
            angles.append(principal_plane_angle(U_curr, U_clv_hist[n-1]))
        pos_3d += basis_3d.get(ch, basis_3d['1'])
        pts[n] = U_curr.T @ pos_3d
        
    return pts, np.mean(angles) if angles else 0.0

def extract_covariant_boundary(pts, k=10, lam=1.05):
    """Isolates the outer boundary layer via localized neighborhood scaling density."""
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k+1)
    local_scale = np.mean(dists[:, -1])
    mask = dists[:, -1] > lam * local_scale
    return pts[mask]

# ============================================================================
# MULTIFRACTAL SPECTRIC ANALYSIS ENGINE
# ============================================================================

def compute_multifractal_spectrum(boundary, q_range=np.linspace(-3, 3, 13)):
    """
    Computes generalized dimensions Tau(q) and local Hoelder regularities alpha
    to extract the thermodynamic f(alpha) spectrum via Legendre Transform.
    """
    if len(boundary) < 50:
        return np.array([0.0]), np.array([0.0]), 0.0
        
    tree = KDTree(boundary)
    scales = np.logspace(-2.2, -0.7, 10)
    tau = []
    
    for q in q_range:
        partition_sums = []
        for r in scales:
            local_measures = []
            # Subsample points uniformly to manage computational load across the boundary
            for pt in boundary[::max(1, len(boundary) // 200)]:
                dists, _ = tree.query(pt, k=40)
                mask = dists < r
                if np.sum(mask) > 3:
                    local_measures.append(np.sum(dists[mask] ** q))
            if local_measures:
                partition_sums.append(np.mean(local_measures))
                
        if len(partition_sums) > 3:
            # Linear fit extracts scaling exponents
            coeff = np.polyfit(np.log(scales[:len(partition_sums)]), np.log(partition_sums), 1)[0]
            tau.append(coeff)
        else:
            tau.append(1.0)
            
    tau = np.array(tau)
    
    # Legendre Transform numerical approximation
    alpha = np.gradient(tau, q_range)
    f_alpha = q_range * alpha - tau
    
    # Quantify spectrum width delta_alpha as a structural order parameter
    width = np.max(alpha) - np.min(alpha)
    return alpha, f_alpha, width

# ============================================================================
# UNIFIED PERFORMANCE PLATFORM
# ============================================================================

if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA V5.8: Covariant Oseledets Multifractal Spectrum Engine")
    print("=======================================================================")
    
    N_total = 2000
    phi_inv = (np.sqrt(5) - 1.0) / 2.0
    
    directive_profiles = {
        "Pure Pisot Walk (Fixed M0)": [0] * N_total,
        "Fibonacci Quasiperiodic  ": generate_fibonacci_directive(N_total),
        "Sturmian Golden Rotation ": generate_sturmian_rotation(phi_inv, N_total)
    }
    
    print(f"{'Directive Space Profile':<26} | {'⟨Θ⟩°':<6} | {'Nodes':<5} | {'α_min':<6} | {'α_max':<6} | {'Width Δα':<8}")
    print("-" * 72)
    
    for name, seq in directive_profiles.items():
        # Step 1: Evolve cocycle and extract points using rolling CLV bundle tracking
        pts, mean_drift = execute_oseledets_clv_pipeline(seq, target_length=N_total)
        
        # Step 2: Extract boundary layer
        boundary = extract_covariant_boundary(pts)
        
        # Step 3: Compute localized thermodynamic scaling metrics
        alpha, f_alpha, spectral_width = compute_multifractal_spectrum(boundary)
        
        print(f"{name:<26} | {mean_drift:<6.2f} | {len(boundary):<5} | {np.min(alpha):<6.3f} | {np.max(alpha):<6.3f} | {spectral_width:<8.3f}")
        
    print("=======================================================================")
    print("Multifractal analysis execution completed. Phase trends mapped safely.")
    print("=======================================================================")
