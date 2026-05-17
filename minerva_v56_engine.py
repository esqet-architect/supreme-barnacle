#!/usr/bin/env python3
"""
MINERVA V5.6 — Nonstationary Cocycle Geometry & Grassmannian Persistence Engine
- Implements strict dynamically covariant S-adic projection pipelines
- Drives matrix cocycles via Periodic, Sturmian, and Bernoulli sequences
- Computes intrinsic Grassmannian principal angles between sliding frames
- Evaluates finite-size scaling stability of the normalized disorder invariant
"""

import numpy as np
from numpy.linalg import norm, svd, qr
from scipy.spatial import KDTree
import warnings
import sys

try:
    from ripser import ripser
except ImportError:
    print("[!] Error: 'ripser' package is required for Minerva V5.6.")
    print("Execute: pip install ripser")
    sys.exit(1)

warnings.filterwarnings('ignore')

def get_sturmian_sequence(alpha, N):
    """Generates a quasiperiodic Sturmian directive sequence."""
    seq = []
    for n in range(N):
        bit = int(np.floor((n + 1) * alpha) - np.floor(n * alpha))
        seq.append(bit % 2)
    return seq

def principal_plane_angle(U1, U2):
    """Computes the maximal principal angle between two 2D planes in R^3."""
    C = U1.T @ U2
    svals = svd(C, compute_uv=False)
    svals = np.clip(svals, -1.0, 1.0)
    theta_max = np.degrees(np.arccos(np.min(svals)))
    return theta_max

def simulate_covariant_sadic_engine(directive_seq, target_length=2000):
    """
    Evolves the nonstationary matrix cocycle with sequential QR tracking
    and simultaneous dynamically covariant projection.
    """
    # Core rule matrices
    M0 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0]], dtype=float)
    M1 = np.array([[1.0, 1.0, 1.0],
                   [1.0, 1.1, 0.0], # Slightly perturbed Salem-adjacent boundary configuration
                   [1.0, 0.0, 1.0]], dtype=float)
    matrices = {0: M0, 1: M1}
    
    # Structural word building logic based on localized directives
    subs_0 = {'1': '12', '2': '13', '3': '1'}
    subs_1 = {'1': '123', '2': '13', '3': '1'}
    rules = {0: subs_0, 1: subs_1}
    
    word = '1'
    for bit in directive_seq[:8]:
        word = ''.join(rules[bit].get(c, c) for c in word)
        if len(word) > target_length * 1.5:
            break
    word = word[:target_length]
    
    # Cocycle allocation tracking
    N_steps = len(word)
    pts = np.zeros((N_steps, 2))
    pos_3d = np.zeros(3)
    
    A_cumulative = np.eye(3)
    Q_lyap = np.eye(3)
    lyap_sums = np.zeros(3)
    
    basis_3d = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    
    U_prev = None
    angles = []
    
    for n, ch in enumerate(word):
        bit = directive_seq[n % len(directive_seq)]
        M_curr = matrices[bit]
        
        # Evolve primary historical cocycle operator
        A_cumulative = M_curr @ A_cumulative
        
        # Stable Lyapunov tracking via QR factorization
        Q_lyap, R_lyap = qr(M_curr @ Q_lyap)
        lyap_sums += np.log(np.abs(np.diag(R_lyap)) + 1e-15)
        
        # Extract instantaneous contracting bundle via SVD
        _, _, Vt = svd(A_cumulative)
        U_curr = Vt[1:3, :].T  # Isolate 2D sub-dominant stable plane (3, 2)
        
        # Track intrinsic Grassmannian geometric drift velocity
        if U_prev is not None:
            theta_drift = principal_plane_angle(U_curr, U_prev)
            angles.append(theta_drift)
            
        U_prev = U_curr
        
        # Advance position and apply dynamic covariance projection
        step_vector = basis_3d.get(ch, basis_3d['1'])
        pos_3d += step_vector
        pts[n] = U_curr.T @ pos_3d
        
    ftle_spectrum = lyap_sums / N_steps
    mean_theta = np.mean(angles) if angles else 0.0
    
    return pts, ftle_spectrum[1], mean_theta

def compute_normalized_disorder(pts, scale_limit):
    """Performs topological analysis via Ripser at a fixed scale limit."""
    if len(pts) > scale_limit:
        idx = np.random.choice(len(pts), scale_limit, replace=False)
        pts = pts[idx]
        
    n_v = len(pts)
    tree = KDTree(pts)
    dists_knn, _ = tree.query(pts, k=10)
    
    # Compute local fractal dimension fields
    d_loc = []
    for i in range(n_v):
        r = dists_knn[i, 1:]
        counts = np.arange(1, len(r) + 1)
        valid = r > 1e-9
        if np.sum(valid) > 4:
            d_loc.append(np.polyfit(np.log(r[valid]), np.log(counts[valid]), 1)[0])
        else:
            d_loc.append(1.0)
    sigma_d_loc = np.std(d_loc)
    
    # Process persistent homological invariants
    dgms = ripser(pts, maxdim=1)['dgms']
    h0_lifetimes = dgms[0][:-1, 1] - dgms[0][:-1, 0]
    
    noise_floor = np.median(dists_knn[:, 1]) * 1.5
    n_h0_long = np.sum(h0_lifetimes > noise_floor)
    
    w_h1 = np.sum(dgms[1][:, 1] - dgms[1][:, 0]) if len(dgms[1]) > 0 else 0.0
    
    if len(h0_lifetimes) > 0 and np.sum(h0_lifetimes) > 0:
        p_bars = h0_lifetimes / np.sum(h0_lifetimes)
        s_pers_norm = -np.sum(p_bars * np.log(p_bars + 1e-12)) / np.log(len(p_bars) + 1e-6)
    else:
        s_pers_norm = 0.0
        
    xi_bar = sigma_d_loc * s_pers_norm * (n_h0_long / float(n_v)) * (1.0 / (w_h1 + 1.0))
    return xi_bar

# ============================================================================
# EXPERIMENTAL PHASE EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA V5.6: Nonstationary S-adic Engine (Covariant Execution)")
    print("=======================================================================")
    
    N_total = 3000
    phi_inv = (np.sqrt(5) - 1.0) / 2.0  # Invariant golden ratio parameter
    
    # Construct distinct families of directive arrays
    directives = {
        "Periodic Resonant (0,0,1)": [0, 0, 1] * (N_total // 3),
        "Sturmian Quasiperiodic   ": get_sturmian_sequence(phi_inv, N_total),
        "Random Bernoulli (p=0.4)": np.random.choice([0, 1], size=N_total, p=[0.6, 0.4]).tolist()
    }
    
    scale_benchmarks = [400, 800, 1600]
    
    print(f"{'Directive Profile':<26} | {'FTLE λ2':<8} | {'Drift ⟨Θ⟩':<9} | {'Ξ_bar@400':<9} | {'Ξ_bar@800':<9} | {'Ξ_bar@1600':<9}")
    print("-" * 87)
    
    for name, seq in directives.items():
        # Execute the fully nonstationary covariant tracking loop
        points, lambda_2, mean_drift = simulate_covariant_sadic_engine(seq, target_length=N_total)
        
        # Filter for boundary point extraction
        tree = KDTree(points)
        dists, _ = tree.query(points, k=4)
        b_pts = points[np.where(dists[:, -1] > np.mean(dists[:, -1]) * 1.04)[0]]
        if len(b_pts) < 200:
            b_pts = points
            
        xi_scaling = []
        for size in scale_benchmarks:
            val = compute_normalized_disorder(b_pts, min(size, len(b_pts)))
            xi_scaling.append(val)
            
        print(f"{name:<26} | {lambda_2:<8.3f} | {mean_drift:<9.3f} | {xi_scaling[0]:<9.5f} | {xi_scaling[1]:<9.5f} | {xi_scaling[2]:<9.5f}")
        
    print("=======================================================================")
    print("Covariant execution loop terminated. Data profiles locked.")
    print("=======================================================================")
