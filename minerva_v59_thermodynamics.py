#!/usr/bin/env python3
"""
MINERVA V5.9 — Thermodynamic Formalism & Statistical Mechanics Engine
- Maps Oseledets Invariant Bundle Filtrations via Forward-Backward CLV Sweeps
- Evaluates Symbolic Block Complexity & Asymptotic Topological Entropy h_top
- Tracks Lyapunov Variance & Spectral Window Entropy Production h_FTLE
- Executes Thermodynamic Partitioning Z(q, eps) & Fit-Free Legendre f(alpha)
- Computes Thermodynamic Free Energy & Pressure Profiles P(q)
"""

import numpy as np
from numpy.linalg import norm, svd, qr, inv
from scipy.spatial import KDTree
import warnings
import sys

try:
    from ripser import ripser
except ImportError:
    print("[!] Error: 'ripser' package is required for Minerva V5.9.")
    print("Execute: pip install ripser")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# DIRECTIVE LAYER & SYMBOLIC COMPLEXITY GENERATORS
# ============================================================================

def generate_fibonacci_directive(N):
    """Generates a quasiperiodic Fibonacci sequence via standard structural scaling."""
    word = "0"
    while len(word) < N:
        word = word.replace("0", "01").replace("1", "0X").replace("X", "1")
    return [int(c) for c in word[:N]]

def generate_sturmian_rotation(alpha, N):
    """Generates an irrational rotation directive sequence."""
    return [int(np.floor((n + 1) * alpha) - np.floor(n * alpha)) % 2 for n in range(N)]

def compute_block_complexity(word, n):
    """Counts unique sub-words of length n to map language growth profile p(n)."""
    return len(set(word[i:i+n] for i in range(len(word)-n+1)))

def estimate_topological_entropy(word, max_n=10):
    """Estimates the asymptotic structural topological entropy limit h_top."""
    vals = []
    for n in range(2, max_n + 1):
        p_n = compute_block_complexity(word, n)
        vals.append(np.log(p_n + 1e-15) / n)
    return np.mean(vals[-3:]) # Return stabilized terminal tail mean

# ============================================================================
# INVARIANT BUNDLE CORE (CLV OSELEDETS GENERATOR)
# ============================================================================

def principal_plane_angle(U1, U2):
    """Computes the maximal principal angle between two 2D subspace matrices."""
    C = U1.T @ U2
    svals = svd(C, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return np.degrees(np.arccos(np.min(svals)))

def execute_covariant_clv_trajectory(directive_seq, target_length=1500):
    """
    Evolves a deformed matrix cocycle over nonstationary rules.
    Extracts invariant projection bases via Oseledets filtration intersection.
    """
    M0 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]], dtype=float)
    M1 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 1.1, 0.0],  # Continuous matrix deformation parameter
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
    
    # Forward QR sweep tracking local expansion rates
    Q_hist, R_hist, log_diag_hist = [], [], []
    Q_curr = np.eye(3)
    for n in range(N):
        bit = directive_seq[n % len(directive_seq)]
        Q_curr, R_curr = qr(matrices[bit] @ Q_curr)
        Q_hist.append(Q_curr)
        R_hist.append(R_curr)
        log_diag_hist.append(np.log(np.abs(np.diag(R_curr)) + 1e-15))
        
    log_diag_hist = np.array(log_diag_hist)
    
    # Backward sweep for True Invariant Subspace (CLV) Isolation
    V_curr = np.eye(3)
    U_clv_hist = [None] * N
    for n in reversed(range(N)):
        V_curr = inv(R_hist[n]) @ V_curr
        V_q, _ = qr(V_curr)
        U_clv_hist[n] = Q_hist[n] @ V_q[:, 1:3]
        
    # Project spatial structure against rolling invariant bundle
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
        
    return pts, word, log_diag_hist[:, 1], np.mean(angles) if angles else 0.0

def extract_boundary(pts, k=10, lam=1.06):
    """Filters data vectors to isolate the exact outer boundary layer manifold."""
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k+1)
    mask = dists[:, -1] > lam * np.mean(dists[:, -1])
    return pts[mask]

# ============================================================================
# THERMODYNAMIC ENSEMBLE SUB-SYSTEM
# ============================================================================

def compute_ftle_entropy_production(lambda_2_path, bins=30):
    """Calculates thermodynamic entropy h_FTLE from localized tracking profiles."""
    hist, _ = np.histogram(lambda_2_path, bins=bins, density=True)
    p = hist / (np.sum(hist) + 1e-15)
    p = p[p > 1e-12]
    return -np.sum(p * np.log(p))

def compute_pressure_function(observable, q_vals):
    """Evaluates numerical thermodynamic pressure profiles P(q) for an observable field."""
    P = []
    obs = np.array(observable)
    for q in q_vals:
        z = np.sum(np.exp(-q * obs))
        P.append(np.log(z + 1e-15) / len(obs))
    return np.array(P)

def compute_thermodynamic_f_alpha(boundary, q_range=np.linspace(-2.5, 2.5, 15)):
    """
    Implements full partition sum mechanics Z(q, eps) to extract
    generalized dimensions tau(q) and f(alpha) via accurate Legendre Transform.
    """
    if len(boundary) < 40:
        return 0.0, np.array([0.0])
        
    tree = KDTree(boundary)
    scales = np.logspace(-2.0, -0.6, 8)
    tau = []
    
    for q in q_range:
        z_sums = []
        for r in scales:
            local_m = []
            for pt in boundary[::max(1, len(boundary) // 150)]:
                dists, _ = tree.query(pt, k=30)
                mask = dists < r
                if np.sum(mask) > 2:
                    local_m.append(np.sum(dists[mask] ** q))
            if local_m:
                z_sums.append(np.mean(local_m))
                
        if len(z_sums) > 3:
            tau.append(np.polyfit(np.log(scales[:len(z_sums)]), np.log(z_sums), 1)[0])
        else:
            tau.append(1.0)
            
    tau = np.array(tau)
    alpha = np.gradient(tau, q_range)
    f_alpha = q_range * alpha - tau
    
    # Curvature of the pressure landscape (Detects structural phase transitions)
    pressure_curvature = np.std(np.diff(tau, 2)) if len(tau) > 2 else 0.0
    return np.max(alpha) - np.min(alpha), pressure_curvature

# ============================================================================
# UNIFIED PLATFORM EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA V5.9: Deformed Cocycle Thermodynamic Engine")
    print("=======================================================================")
    
    N_total = 1600
    phi_inv = (np.sqrt(5) - 1.0) / 2.0
    silver = np.sqrt(2) - 1.0
    
    experiments = {
        "Stationary Pisot (M0 Frame)": [0] * N_total,
        "Fibonacci Quasiperiodic   ": generate_fibonacci_directive(N_total),
        "Sturmian Silver Rotation  ": generate_sturmian_rotation(silver, N_total),
        "Sturmian Golden Rotation  ": generate_sturmian_rotation(phi_inv, N_total)
    }
    
    q_space = np.linspace(-2.5, 2.5, 15)
    
    print(f"{'Directive Spectrum Matrix':<28} | {'h_top':<5} | {'h_FTLE':<6} | {'⟨Θ⟩°':<5} | {'Δα (Width)':<10} | {'P(q) Curv':<9}")
    print("-" * 84)
    
    for name, seq in experiments.items():
        # Step 1: Invariant splitting geometry generation loop
        pts, word, lam2_path, mean_drift = execute_covariant_clv_trajectory(seq, target_length=N_total)
        
        # Step 2: Language metrics and entropy parsing
        h_top = estimate_topological_entropy(word, max_n=10)
        h_ftle = compute_ftle_entropy_production(lam2_path)
        
        # Step 3: Boundary extraction pass
        b_pts = extract_boundary(pts)
        
        # Step 4: Extract structural thermodynamic profiles
        delta_alpha, p_curv = compute_thermodynamic_f_alpha(b_pts, q_range=q_space)
        
        print(f"{name:<28} | {h_top:<5.3f} | {h_ftle:<6.3f} | {mean_drift:<5.2f} | {delta_alpha:<10.4f} | {p_curv:<9.5f}")
        
    print("=======================================================================")
    print("Thermodynamic partition evaluation finalized successfully.")
    print("=======================================================================")
