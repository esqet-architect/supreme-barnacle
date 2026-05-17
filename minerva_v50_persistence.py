#!/usr/bin/env python3
"""
MINERVA V5.0 — Topological Persistence & Local Dimension Field Engine
- Dynamic cocycle-ready framework
- Localized neighborhood dimension profiling (D_loc Field)
- Intrinsic Vietoris-Rips H0/H1 topological persistence estimator
"""

import numpy as np
from numpy.linalg import eig, norm, svd
from scipy.spatial import KDTree, Delaunay
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DYNAMIC COCYCLE & PROJECTION ARCHITECTURE
# ============================================================================
def get_oseledets_basis(M_list):
    """
    Computes the invariant contracting plane for an S-adic cocycle sequence
    via SVD decomposition of the cumulative matrix product.
    """
    # Compute cumulative cocycle product: M_n * M_{n-1} * ... * M_1
    A = np.eye(3, dtype=float)
    for M in M_list:
        A = M @ A
        
    U, S, Vt = svd(A)
    # The stable/contracting directions correspond to the smallest singular values
    # Vt rows match the eigenspaces. We extract the dominant and contracting planes.
    v_beta = Vt[0, :]
    v_beta /= np.sum(v_beta) if np.sum(v_beta) != 0 else 1.0
    
    e1 = Vt[1, :]
    e2 = Vt[2, :]
    e1 /= norm(e1) if norm(e1) != 0 else 1.0
    e2 -= np.dot(e2, e1) * e1
    e2 /= norm(e2) if norm(e2) != 0 else 1.0
    
    return np.array([e1, e2]), v_beta

def build_cocycle_rauzy(word, M_list, subs_map):
    """Builds projected point cloud using the dynamically adapted basis."""
    pi_c, _ = get_oseledets_basis(M_list)
    basis = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    
    points = np.zeros((len(word), 2))
    pos = np.zeros(3)
    
    for i, c in enumerate(word):
        points[i] = pi_c @ pos
        pos += basis.get(c, basis['1'])
        
    return np.nan_to_num(points)

# ============================================================================
# 2. LOCAL GEOMETRIC FIELD ENGINE (D_loc)
# ============================================================================
def compute_local_dimension_field(pts, k_neighbors=40):
    """Calculates the localized box/neighborhood scaling factor for each node."""
    if len(pts) < k_neighbors:
        return np.zeros(len(pts))
    
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k_neighbors)
    
    d_loc_field = []
    # Evaluate localized scaling log(count) vs log(r) across neighbors
    for i in range(len(pts)):
        local_dists = dists[i, 1:] # Drop self-distance
        r_scales = np.sort(local_dists)
        counts = np.arange(1, len(r_scales) + 1)
        
        # Linear regression on log-log mass accumulation
        idx = r_scales > 1e-11
        if np.sum(idx) > 5:
            coeff = np.polyfit(np.log(r_scales[idx]), np.log(counts[idx]), 1)
            d_loc_field.append(coeff[0])
        else:
            d_loc_field.append(0.0)
            
    return np.array(d_loc_field)

# ============================================================================
# 3. INTRINSIC PERSISTENT HOMOLOGY ENGINE
# ============================================================================
def extract_topological_persistence(pts, max_scales=15):
    """
    Computes an intrinsic filtration profile mimicking Vietoris-Rips H0 and H1.
    Tracks connected component reduction (H0) and cycles (H1) across scales.
    """
    if len(pts) < 15:
        return {'H0_bars': 0, 'H1_loops': 0, 'stable_topology': False}
    
    # Calculate global scale bounding parameters
    tree = KDTree(pts)
    min_dists, _ = tree.query(pts, k=2)
    base_eps = np.median(min_dists[:, 1])
    max_eps = norm(np.max(pts, axis=0) - np.min(pts, axis=0))
    
    eps_scales = np.linspace(base_eps * 0.5, base_eps * 5.0, max_scales)
    h0_counts = []
    h1_counts = []
    
    # Simple alpha-complex/Rips graph connectivity walk across epsilon windows
    for eps in eps_scales:
        # H0: Extract connected components via adjacency graph construction
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
        for u, v in adj:
            if union(u, v):
                components -= 1
        h0_counts.append(components)
        
        # H1: Find persistent cycles via filtering localized Delaunay configurations
        try:
            tri = Delaunay(pts + np.random.normal(0, 1e-10, pts.shape))
            cycles = 0
            for simplex in tri.simplices:
                p = pts[simplex]
                d1 = norm(p[0]-p[1])
                d2 = norm(p[1]-p[2])
                d3 = norm(p[2]-p[0])
                if max(d1, d2, d3) <= eps:
                    cycles += 1
            h1_counts.append(max(0, cycles - (len(pts) - components)))
        except:
            h1_counts.append(0)

    # Analyze stability signatures
    h0_variance = np.std(h0_counts)
    max_h1_loops = np.max(h1_counts)
    
    return {
        'eps_scales': eps_scales,
        'H0_profile': h0_counts,
        'H1_profile': h1_counts,
        'H0_stable_val': h0_counts[-1],
        'Max_H1_Loops': max_h1_loops
    }

# ============================================================================
# EXECUTABLE VERIFICATION
# ============================================================================
if __name__ == '__main__':
    print("=======================================================================")
    print("Executing MINERVA V5.0: Invariant Topological Field Diagnostics")
    print("=======================================================================")
    
    # Setup test matrix definitions
    M_tribonacci = np.array([[1,1,1],[1,0,0],[0,1,0]], float)
    subs_tribonacci = {'1':'12','2':'13','3':'1'}
    
    # Generate substitution word
    word = '1'
    for _ in range(8):
        word = ''.join(subs_tribonacci.get(c, c) for c in word)
    word = word[:8000]
    
    print(f"Generated Sequence Length: {len(word)}")
    
    # 1. Coordinate Generation via dynamic-ready Oseledets setup
    pts = build_cocycle_rauzy(word, [M_tribonacci], subs_tribonacci)
    
    # Simulate a fast edge/boundary prune for tracking localized metrics
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=6)
    boundary_idx = np.where(dists[:, -1] > np.mean(dists[:, -1]) * 1.1)[0]
    boundary_pts = pts[boundary_idx][:1200]  # Subset for dense performance evaluation
    
    print(f"Isolated Boundary Feature Cloud Nodes: {len(boundary_pts)}")
    
    # 2. Extract Geometric Field Metrics
    d_loc = compute_local_dimension_field(boundary_pts)
    print(f"Local Field Properties -> Mean D_loc: {np.mean(d_loc):.4f} | Spatial Variance (σ): {np.std(d_loc):.4f}")
    
    # 3. Calculate Topological Persistence Signatures
    topo = extract_topological_persistence(boundary_pts)
    print("\n--- Topological Persistence Matrix Profile ---")
    print(f"Scale Step Epsilon Window: [{topo['eps_scales'][0]:.5f} to {topo['eps_scales'][-1]:.5f}]")
    print(f"H0 Components Path: {topo['H0_profile']}")
    print(f"H1 Micro-Cycles Path: {topo['H1_profile']}")
    print(f"Ultimate Connected State (H0 Val): {topo['H0_stable_val']}")
    print(f"Peak Cycle Persistence (Max H1 Loops): {topo['Max_H1_Loops']}")
    print("=======================================================================")
