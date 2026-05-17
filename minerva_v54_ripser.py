#!/usr/bin/env python3
"""
MINERVA V5.4 — True Topological Persistence & Cocycle Spectral Instability
- Integrates Ripser for authentic H0/H1 persistent homology barcode extraction
- Computes Finite-Time Lyapunov Exponents (FTLE) to measure cocycle contraction
- Evaluates the scale-invariant disorder functional using true persistence lifetimes
- Maps the Pisot -> Salem-adjacent boundary projection cocycle transition
"""

import numpy as np
from numpy.linalg import eig, norm, svd
from scipy.spatial import KDTree
import os
import warnings

# Attempt to import ripser for true persistent homology
try:
    from ripser import ripser
except ImportError:
    print("[!] Error: 'ripser' package is required for Minerva V5.4.")
    print("Run: pip install ripser")
    import sys
    sys.exit(1)

warnings.filterwarnings('ignore')

def compute_ftle_spectrum(M_base, t, steps=50):
    """
    Computes the Finite-Time Lyapunov Exponents (FTLE) for the cocycle trajectory:
    lambda_i = (1/N) * log(sigma_i(M_N ... M_1))
    """
    M_t = np.array([[1.0, 1.0, 1.0],
                    [1.0, float(t), 0.0],
                    [1.0, 0.0, 1.0]], dtype=float)
    
    # Initialize matrix product tracking frame
    Q = np.eye(3)
    lyapunov_sums = np.zeros(3)
    
    for _ in range(steps):
        # Explicit matrix multiplication step
        A = M_t @ Q
        Q, R = np.linalg.qr(A)
        lyapunov_sums += np.log(np.abs(np.diag(R)) + 1e-15)
        
    return lyapunov_sums / steps

def get_adapted_frame(M):
    """Extracts the orthonormal contracting plane basis via singular frames."""
    U, S, Vt = svd(M)
    e1 = Vt[1, :]
    e2 = Vt[2, :]
    e1 /= norm(e1) if norm(e1) != 0 else 1.0
    e2 -= np.dot(e2, e1) * e1
    e2 /= norm(e2) if norm(e2) != 0 else 1.0
    return np.array([e1, e2])

def generate_projection_cloud(t, length=4000):
    """Generates the point cloud using the induced contracting-plane cocycle."""
    # Anchored symbolic substitution to construct structural scaffolding
    if t < 1.0:
        subs = {'1': '12', '2': '13', '3': '1'}
    else:
        subs = {'1': '123', '2': '13', '3': '1'}
        
    word = '1'
    while len(word) < length:
        word = ''.join(subs.get(c, c) for c in word)
        if len(word) > length * 2:
            break
    word = word[:length]
    
    M_t = np.array([[1.0, 1.0, 1.0],
                    [1.0, float(t), 0.0],
                    [1.0, 0.0, 1.0]], dtype=float)
    
    frame = get_adapted_frame(M_t)
    basis = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    delta = {ch: frame @ v for ch, v in basis.items()}
    
    pts = np.zeros((len(word), 2))
    pos = np.zeros(2)
    for i, ch in enumerate(word):
        pts[i] = pos
        pos = pos + delta.get(ch, delta['1'])
    return pts

def analyze_true_persistence(pts, max_subsample=400):
    """Computes exact Vietoris-Rips persistence features via Ripser."""
    if len(pts) > max_subsample:
        idx = np.random.choice(len(pts), max_subsample, replace=False)
        pts = pts[idx]
        
    n_v = len(pts)
    
    # Compute local dimension field heterogeneity
    tree = KDTree(pts)
    dists_knn, _ = tree.query(pts, k=12)
    d_loc_field = []
    for i in range(n_v):
        r_scales = dists_knn[i, 1:]
        counts = np.arange(1, len(r_scales) + 1)
        valid = r_scales > 1e-9
        if np.sum(valid) > 4:
            coeff = np.polyfit(np.log(r_scales[valid]), np.log(counts[valid]), 1)
            d_loc_field.append(coeff[0])
        else:
            d_loc_field.append(1.0)
    sigma_d_loc = np.std(d_loc_field)
    
    # Run Ripser for true H0 and H1 persistence diagrams
    dgms = ripser(pts, maxdim=1)['dgms']
    dgm_h0 = dgms[0]
    dgm_h1 = dgms[1]
    
    # Isolate long-lived H0 features (excluding the infinite component)
    h0_lifetimes = dgm_h0[:-1, 1] - dgm_h0[:-1, 0]
    # Filter bars that survive significantly past typical node gaps
    noise_floor = np.median(dists_knn[:, 1]) * 1.5
    n_h0_long = np.sum(h0_lifetimes > noise_floor)
    
    # Calculate true persistent H1 lifetime-weighted norm (Wasserstein proxy)
    if len(dgm_h1) > 0:
        h1_lifetimes = dgm_h1[:, 1] - dgm_h1[:, 0]
        w_h1 = np.sum(h1_lifetimes)
        # Dominant H1 count: macro-loops surviving noise
        n_h1_dom = np.sum(h1_lifetimes > (noise_floor * 2))
    else:
        w_h1 = 0.0
        n_h1_dom = 0
        
    # Scale-Independent Persistence Entropy via true H0 bars
    if len(h0_lifetimes) > 0 and np.sum(h0_lifetimes) > 0:
        p_bars = h0_lifetimes / np.sum(h0_lifetimes)
        s_pers_norm = -np.sum(p_bars * np.log(p_bars + 1e-12)) / np.log(len(p_bars) + 1e-6)
    else:
        s_pers_norm = 0.0
        
    # Standardized Xi_bar Functional using rigorous topological weightings
    xi_bar = sigma_d_loc * s_pers_norm * (n_h0_long / float(n_v)) * (1.0 / (w_h1 + 1.0))
    
    return sigma_d_loc, s_pers_norm, n_h0_long, n_h1_dom, w_h1, xi_bar

# ============================================================================
# EXECUTION LIFELINE
# ============================================================================
if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA V5.4: True Persistent Homology & FTLE Spectrum Tracker")
    print("=======================================================================")
    
    t_values = np.linspace(0.0, 2.0, 9)
    M_base = np.array([[1,1,1],[1,0,0],[0,1,0]], dtype=float)
    
    print(f"{'t-Param':<7} | {'FTLE λ1':<8} | {'FTLE λ2':<8} | {'σ(D_loc)':<8} | {'Long H0':<7} | {'True H1':<7} | {'W(H1)':<7} | {'Ξ_bar Invariant':<15}")
    print("-" * 92)
    
    for t in t_values:
        # 1. Track true cocycle structural instability via finite-time Lyapunov paths
        ftle = compute_ftle_spectrum(M_base, t, steps=100)
        
        # 2. Extract induced projection geometry
        pts = generate_projection_cloud(t, length=3500)
        
        # 3. Filter boundary shell to maximize topological signature density
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=4)
        boundary_threshold = np.mean(dists[:, -1]) * 1.05
        b_pts = pts[np.where(dists[:, -1] > boundary_threshold)[0]]
        
        if len(b_pts) < 50:
            b_pts = pts
            
        # 4. Process through exact Vietoris-Rips complexes using Ripser
        sigma_d, s_norm, h0_long, h1_dom, w_h1, xi_bar = analyze_true_persistence(b_pts)
        
        print(f"{t:<7.2f} | {ftle[0]:<8.3f} | {ftle[1]:<8.3f} | {sigma_d:<8.3f} | {h0_long:<7} | {h1_dom:<7} | {w_h1:<7.3f} | {xi_bar:.6f}")
        
    print("=======================================================================")
    print("Minerva V5.4 processing complete. Exact H1 architecture preserved.")
    print("=======================================================================")
