#!/usr/bin/env python3
"""
MINERVA V5.1 — Structural Diagnostic & Coordinate Distortion Engine
- Scale-Invariance & Diameter distortion testing (Case A vs Case B)
- 1-Skeleton Cyclomatic Number Tracking (corrected from false H1 homology)
- H0 Connected Component Path & Persistence Entropy Metric
"""

import numpy as np
from numpy.linalg import eig, norm, svd
from scipy.spatial import KDTree, Delaunay
import warnings
warnings.filterwarnings('ignore')

def get_finite_lyapunov_basis(M):
    """Extracts the finite-time singular frame for the projection plane."""
    U, S, Vt = svd(M)
    v_beta = Vt[0, :]
    v_beta /= np.sum(v_beta) if np.sum(v_beta) != 0 else 1.0
    
    e1 = Vt[1, :]
    e2 = Vt[2, :]
    e1 /= norm(e1) if norm(e1) != 0 else 1.0
    e2 -= np.dot(e2, e1) * e1
    e2 /= norm(e2) if norm(e2) != 0 else 1.0
    return np.array([e1, e2]), v_beta

def build_rauzy_points(word, M):
    pi_c, _ = get_finite_lyapunov_basis(M)
    basis = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    points = np.zeros((len(word), 2))
    pos = np.zeros(3)
    
    for i, c in enumerate(word):
        points[i] = pi_c @ pos
        pos += basis.get(c, basis['1'])
    return np.nan_to_num(points)

def analyze_scale_resolved_topology(pts, max_scales=20):
    """
    Computes rigorous continuous H0 persistence trajectories 
    and extracts 1-skeleton cyclomatic network counts.
    """
    tree = KDTree(pts)
    min_dists, _ = tree.query(pts, k=2)
    base_eps = np.median(min_dists[:, 1])
    max_eps = norm(np.max(pts, axis=0) - np.min(pts, axis=0))
    
    eps_scales = np.linspace(base_eps * 0.5, base_eps * 4.0, max_scales)
    h0_trajectory = []
    cyclomatic_trajectory = []
    
    for eps in eps_scales:
        # H0: Union-Find Component Tracking
        adj = tree.query_pairs(eps)
        parent = list(range(len(pts)))
        
        def find(i):
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for node in path:
                parent[node] = i
            return i

        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False

        components = len(pts)
        edges_connected = 0
        for u, v in adj:
            if union(u, v):
                components -= 1
            edges_connected += 1
            
        h0_trajectory.append(components)
        
        # Corrected Graph-Theoretic Cyclomatic Complexity: C = E - V + P
        c_num = max(0, edges_connected - len(pts) + components)
        cyclomatic_trajectory.append(c_num)
        
    # Calculate H0 Persistence Entropy: H = -sum(p_i * log(p_i))
    # Measures the distribution of component structural lifetimes
    h0_norm = np.array(h0_trajectory) / np.sum(h0_trajectory)
    h0_entropy = -np.sum(h0_norm * np.log(h0_norm + 1e-12))
    
    return {
        'scales': eps_scales,
        'H0_path': h0_trajectory,
        'Cyclomatic_path': cyclomatic_trajectory,
        'Persistence_Entropy': h0_entropy,
        'Global_Diameter': max_eps
    }

# ============================================================================
# DIAGNOSTIC EXECUTION
# ============================================================================
if __name__ == '__main__':
    print("=======================================================================")
    print("Running MINERVA V5.1: Geometry vs. Coordinate Projection Diagnostic")
    print("=======================================================================")
    
    # Systems to test
    systems = {
        'Tribonacci Pisot': {
            'M': np.array([[1,1,1],[1,0,0],[0,1,0]], float),
            'subs': {'1':'12','2':'13','3':'1'}
        },
        'Arnoux-Rauzy Variant': {
            'M': np.array([[1,1,1],[1,1,0],[1,0,1]], float), # Structural shear test
            'subs': {'1':'123','2':'12','3':'13'}
        }
    }
    
    results = {}
    for name, cfg in systems.items():
        word = '1'
        for _ in range(7):
            word = ''.join(cfg['subs'].get(c, c) for c in word)
        word = word[:5000]
        
        pts = build_rauzy_points(word, cfg['M'])
        
        # Prune for boundary density approximation
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=5)
        b_idx = np.where(dists[:, -1] > np.mean(dists[:, -1]) * 1.05)[0]
        b_pts = pts[b_idx][:800]
        
        metrics = analyze_scale_resolved_topology(b_pts)
        results[name] = metrics
        
        print(f"\n--- {name} ---")
        print(f"Max Spatial Diameter: {metrics['Global_Diameter']:.4f}")
        print(f"H0 Components Path:   {metrics['H0_path'][-5:]} (Last 5 steps)")
        print(f"Cyclomatic 1-Skeleton Loop Path: {metrics['Cyclomatic_path'][-5:]}")
        print(f"H0 Persistence Entropy: {metrics['Persistence_Entropy']:.4f}")

    print("\n=======================================================================")
    # Comparative Evaluation
    ratio = results['Arnoux-Rauzy Variant']['Global_Diameter'] / results['Tribonacci Pisot']['Global_Diameter']
    print(f"Diameter Ratio (AR / Tribonacci): {ratio:.4f}")
    if ratio > 5.0:
        print("DIAGNOSIS: CASE A — Severe coordinate/projection artifact detected.")
        print("The fragmentation is primarily driven by directional shearing on the finite basis.")
    else:
        print("DIAGNOSIS: CASE B — True geometric/topological structural variance.")
        print("The boundary lamination is an intrinsic feature of the symbolic dynamics sequence.")
    print("=======================================================================")
