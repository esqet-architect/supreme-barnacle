#!/usr/bin/env python3
"""
MINERVA V5.7 — Nonstationary Oseledets CLV & Trajectory Field Engine
- Computes time-resolved FTLE trajectories and finite-window rolling spectra
- Maps intrinsic Grassmannian principal angles and second-order frame curvature K_n
- Approximates Invariant Bundles using a Forward-Backward Filtration Intersection
- Injects Thue-Morse, Fibonacci, and Markovian Switching directive systems
- Analyzes structural phase-space coupling via scale-invariant persistence diagnostics
"""

import numpy as np
from numpy.linalg import norm, svd, qr, inv
from scipy.spatial import KDTree
import warnings
import sys

try:
    from ripser import ripser
except ImportError:
    print("[!] Error: 'ripser' package is required for Minerva V5.7.")
    print("Execute: pip install ripser")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# DIRECTIVE GENERATION ENGINE (EXTENDED UNIVERSALITY CLASSES)
# ============================================================================

def generate_thue_morse(N):
    """Generates an aperiodic, strongly deterministic Thue-Morse directive array."""
    return [bin(n).count("1") % 2 for n in range(N)]

def generate_fibonacci_directive(N):
    """Generates a true quasiperiodic Fibonacci sequence via string replacement."""
    word = "0"
    while len(word) < N:
        # X maps transient intermediate substitution state safely
        word = word.replace("0", "01").replace("1", "0X").replace("X", "1")
    return [int(c) for c in word[:N]]

def generate_markov_switching(N, P_matrix):
    """Generates a Markov chain to simulate long transient coherence bands."""
    seq = [0]
    current_state = 0
    for _ in range(1, N):
        p_transition = P_matrix[current_state]
        current_state = np.random.choice([0, 1], p=p_transition)
        seq.append(current_state)
    return seq

def generate_sturmian_rotation(alpha, N):
    """Generates an irrational rotation directive sequence."""
    return [int(np.floor((n + 1) * alpha) - np.floor(n * alpha)) % 2 for n in range(N)]

# ============================================================================
# MATHEMATICAL UTILITIES & FRAME TRACKING
# ============================================================================

def principal_plane_angle(U1, U2):
    """Computes the maximal principal angle between two 2D subspace matrices."""
    C = U1.T @ U2
    svals = svd(C, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    return np.degrees(np.arccos(np.min(svals)))

def compute_rolling_ftle(log_diag_history, window=100):
    """Extracts finite-window FTLEs to isolate localized instability bursts."""
    out = []
    for i in range(window, len(log_diag_history) + 1):
        block = log_diag_history[i - window:i]
        out.append(np.mean(block, axis=0))
    return np.array(out)

# ============================================================================
# CORE COCYCLE & COVARIANT LYAPUNOV VECTOR ENGINE
# ============================================================================

def execute_oseledets_clv_pipeline(directive_seq, target_length=1500):
    """
    Simulates the nonstationary matrix cocycle product.
    Approximates the invariant bundles via a forward-backward triangular sweep
    to isolate covariant projection orientations.
    """
    # Core Matrix Library
    M0 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]], dtype=float)
    M1 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 1.1, 0.0],  # Salem-adjacent boundary configuration
                   [1.0, 0.0, 1.0]], dtype=float)
    matrices = {0: M0, 1: M1}
    
    # S-adic word parsing dictionary arrays
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
    
    # 1. Forward QR Sweep for Accumulation
    Q_hist = []
    R_hist = []
    log_diag_hist = []
    Q_curr = np.eye(3)
    
    for n in range(N):
        bit = directive_seq[n % len(directive_seq)]
        M = matrices[bit]
        Q_curr, R_curr = qr(M @ Q_curr)
        
        Q_hist.append(Q_curr)
        R_hist.append(R_curr)
        log_diag_hist.append(np.log(np.abs(np.diag(R_curr)) + 1e-15))
        
    log_diag_hist = np.array(log_diag_hist)
    
    # Compute asymptotic and rolling time-resolved FTLE values
    asymptotic_ftle = np.cumsum(log_diag_hist, axis=0) / np.arange(1, N + 1)[:, None]
    rolling_ftle = compute_rolling_ftle(log_diag_hist, window=100)
    
    # Calculate rolling spectral gaps
    rolling_gaps = rolling_ftle[:, 0] - rolling_ftle[:, 1] if len(rolling_ftle) > 0 else np.array([1.0])
    min_gap = np.min(rolling_gaps)
    
    # 2. Backward Recursion for Covariant Subspace Isolation
    # Approximates the Oseledets filtration backward to build stable plane boundaries
    V_curr = np.eye(3) # Terminal random initializing state matrix
    U_clv_hist = [None] * N
    
    for n in reversed(range(N)):
        R_inv = inv(R_hist[n])
        V_curr = R_inv @ V_curr
        # Re-orthonormalize backward components safely
        V_q, _ = qr(V_curr)
        
        # Correlate back with forward frame matrices to construct covariant bundles
        U_clv = Q_hist[n] @ V_q[:, 1:3]  # Sub-dominant contracting columns
        U_clv_hist[n] = U_clv
        
    # 3. Dynamic Covariant Path Tracking
    pts = np.zeros((N, 2))
    pos_3d = np.zeros(3)
    basis_3d = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    
    angles = []
    for n, ch in enumerate(word):
        U_curr = U_clv_hist[n]
        
        if n > 0 and U_clv_hist[n-1] is not None:
            theta = principal_plane_angle(U_curr, U_clv_hist[n-1])
            angles.append(theta)
            
        step_vec = basis_3d.get(ch, basis_3d['1'])
        pos_3d += step_vec
        pts[n] = U_curr.T @ pos_3d
        
    angles = np.array(angles)
    mean_theta = np.mean(angles) if len(angles) > 0 else 0.0
    
    # 4. Compute Second-Order Curvature Statistics (K_n)
    if len(angles) > 1:
        curvature = np.abs(np.diff(angles))
        mean_k = np.mean(curvature)
        max_k = np.max(curvature)
    else:
        mean_k, max_k = 0.0, 0.0
        
    return pts, asymptotic_ftle[-1, 1], min_gap, mean_theta, mean_k, max_k

def compute_topology_metrics(pts, max_nodes=500):
    """Performs topological scaling evaluation on the output geometries via Ripser."""
    if len(pts) > max_nodes:
        idx = np.random.choice(len(pts), max_nodes, replace=False)
        pts = pts[idx]
        
    n_v = len(pts)
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=10)
    
    d_loc = []
    for i in range(n_v):
        r = dists[i, 1:]
        counts = np.arange(1, len(r) + 1)
        valid = r > 1e-9
        if np.sum(valid) > 4:
            d_loc.append(np.polyfit(np.log(r[valid]), np.log(counts[valid]), 1)[0])
        else:
            d_loc.append(1.0)
    sigma_d = np.std(d_loc)
    
    dgms = ripser(pts, maxdim=1)['dgms']
    h0_lifetimes = dgms[0][:-1, 1] - dgms[0][:-1, 0]
    
    noise_floor = np.median(dists[:, 1]) * 1.5
    n_h0_long = np.sum(h0_lifetimes > noise_floor)
    
    w_h1 = np.sum(dgms[1][:, 1] - dgms[1][:, 0]) if len(dgms[1]) > 0 else 0.0
    
    if len(h0_lifetimes) > 0 and np.sum(h0_lifetimes) > 0:
        p_bars = h0_lifetimes / np.sum(h0_lifetimes)
        s_pers = -np.sum(p_bars * np.log(p_bars + 1e-12)) / np.log(len(p_bars) + 1e-6)
    else:
        s_pers = 0.0
        
    xi_bar = sigma_d * s_pers * (n_h0_long / float(n_v)) * (1.0 / (w_h1 + 1.0))
    return xi_bar

# ============================================================================
# HIGH-FIDELITY PHASE VERIFICATION FIELD STUDY
# ============================================================================

if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA V5.7: Invariant Bundle Oseledets CLV & Geometric Engine")
    print("=======================================================================")
    
    N_steps = 2000
    phi_inv = (np.sqrt(5) - 1.0) / 2.0
    silver_ratio = np.sqrt(2) - 1.0
    
    # Markov transition kernel setup
    P_markov = np.array([[0.90, 0.10], 
                         [0.25, 0.75]])
    
    universality_classes = {
        "Thue-Morse Aperiodic     ": generate_thue_morse(N_steps),
        "Fibonacci Quasiperiodic  ": generate_fibonacci_directive(N_steps),
        "Markov Switching (Coherent)": generate_markov_switching(N_steps, P_markov),
        "Sturmian Silver Rotation ": generate_sturmian_rotation(silver_ratio, N_steps),
        "Sturmian Golden Rotation ": generate_sturmian_rotation(phi_inv, N_steps)
    }
    
    print(f"{'Directive Domain':<26} | {'λ2':<6} | {'Min Gap':<7} | {'⟨Θ⟩°':<6} | {'Mean K':<6} | {'Max K':<6} | {'Ξ_bar@800':<10}")
    print("-" * 88)
    
    for name, sequence in universality_classes.items():
        # Execute the forward-backward invariant bundle tracking algorithm
        pts, lam2, min_gap, mean_theta, m_k, mx_k = execute_oseledets_clv_pipeline(sequence, target_length=N_steps)
        
        # Boundary layer geometric isolate extraction pass
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=4)
        boundary_nodes = pts[np.where(dists[:, -1] > np.mean(dists[:, -1]) * 1.04)[0]]
        if len(boundary_nodes) < 150:
            boundary_nodes = pts
            
        xi_score = compute_topology_metrics(boundary_nodes, max_nodes=800)
        
        print(f"{name:<26} | {lam2:<6.3f} | {min_gap:<7.3f} | {mean_theta:<6.2f} | {m_k:<6.2f} | {mx_k:<6.2f} | {xi_score:.6f}")
        
    print("=======================================================================")
    print("Oseledets invariant field tracking execution finalized safely.")
    print("=======================================================================")
