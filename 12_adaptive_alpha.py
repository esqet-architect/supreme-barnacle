#!/usr/bin/env python3
"""
MINERVA ENGINE V4.7.5 — LOCAL DENSITY ADAPTIVE ALPHA COMPLEX
Features:
1. Local k-NN Distance Profiling per Vertex (alpha_i ~ d_k(x_i))
2. Simplex-Average Adaptive Thresholding (circum_r < lambda * mean(local_alpha))
3. Multi-Resolution Parametric Stability Grid (N, k, lambda)
"""

import numpy as np
from numpy.linalg import eig, norm
from scipy.spatial import Delaunay, KDTree
import warnings
warnings.filterwarnings('ignore')

print("===========================================================================")
print("MINERVA ENGINE V4.7.5: LOCAL DENSITY ADAPTIVE ALPHA COMPLEX")
print("===========================================================================")

# ============================================================================
# 1. CORE MATRICES & PROJECTION ENGINE
# ============================================================================

def get_stable_basis(M):
    evals, evecs = eig(M)
    idx = np.argsort(-np.abs(evals))
    evals, evecs = evals[idx], evecs[:, idx]
    v_beta = evecs[:, 0].real
    v_beta /= np.sum(v_beta) if np.sum(v_beta) != 0 else 1e-12

    w = evecs[:, 1]
    e1 = w.real.copy()
    e2 = w.imag.copy()
    e1 /= norm(e1) if norm(e1) != 0 else 1e-12
    e2 -= np.dot(e2, e1) * e1
    e2 /= norm(e2) if norm(e2) != 0 else 1e-12
    return np.array([e1, e2]), v_beta

def build_rauzy_points(word, M):
    pi_c, v_beta = get_stable_basis(M)
    basis = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    points = np.zeros((len(word), 2))
    pos = np.zeros(3)
    for i, c in enumerate(word):
        points[i] = pi_c @ pos
        pos += basis.get(c, basis['1'])
    return np.nan_to_num(points)

# ============================================================================
# 2. LOCAL ADAPTIVE ALPHA COMPLEX EXTRACTION
# ============================================================================

def extract_adaptive_boundary(points, k_neighbors, lambda_factor):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < k_neighbors + 2:
        return np.array([])

    # Compute local scales using the k-th nearest neighbor distance
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k_neighbors + 1)
    local_scales = dists[:, -1]  # d_k(x_i)

    # Jitter to avoid co-circular degeneracy in Delaunay
    rng = np.random.default_rng(42)
    pts_jittered = pts + rng.normal(0, 1e-9, pts.shape)

    try:
        tri = Delaunay(pts_jittered)
    except Exception:
        return np.array([])

    edge_count = {}
    for simplex in tri.simplices:
        pa, pb, pc = pts[simplex]
        a, b, c = norm(pb - pc), norm(pa - pc), norm(pa - pb)
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-14:
            continue
        circum_r = (a * b * c) / (4.0 * np.sqrt(area_sq))

        # Phase 2 Implementation: Local vertex-average thresholding
        local_alpha_threshold = lambda_factor * np.mean(local_scales[simplex])

        if circum_r < local_alpha_threshold:
            edges = [
                tuple(sorted((simplex[0], simplex[1]))),
                tuple(sorted((simplex[1], simplex[2]))),
                tuple(sorted((simplex[2], simplex[0])))
            ]
            for edge in edges:
                edge_count[edge] = edge_count.get(edge, 0) + 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    if len(boundary_edges) == 0:
        return np.array([])

    unique_indices = np.unique(boundary_edges)
    return pts[unique_indices]

# ============================================================================
# 3. SPATIAL BOX-COUNTING DIMENSION ESTIMATOR
# ============================================================================

def compute_box_dimension(boundary_pts):
    if len(boundary_pts) < 15:
        return 0.0
    min_xy = np.min(boundary_pts, axis=0)
    max_xy = np.max(boundary_pts, axis=0)
    max_side = np.max(max_xy - min_xy)
    if max_side < 1e-12:
        return 0.0
    
    scales = max_side / np.logspace(0.5, 2.3, 12)
    counts = []
    for eps in scales:
        grid = np.floor((boundary_pts - min_xy) / eps).astype(int)
        # Use a string or tuple view to leverage unique row finding efficiently
        unique_bins = np.unique(grid, axis=0)
        counts.append(len(unique_bins))
        
    coeffs = np.polyfit(np.log(1.0 / scales), np.log(counts), 1)
    return coeffs[0]

# ============================================================================
# 4. EXPERIMENTAL MATRIX TESTING PARAMETERS
# ============================================================================

systems_config = {
    'Tribonacci Pisot': {
        'subs': {'1': '12', '2': '13', '3': '1'},
        'M': np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    },
    'Plastic Number': {
        'subs': {'1': '2', '2': '3', '3': '12'},
        'M': np.array([[0, 0, 1], [1, 0, 1], [0, 1, 0]], dtype=float)
    },
    'Arnoux-Rauzy': {
        'subs': {'1': '123', '2': '13', '3': '1'},
        'M': np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=float)
    }
}

def generate_system_word(config, length):
    word = '1'
    subs = config['subs']
    while len(word) < length:
        word = ''.join(subs.get(c, c) for c in word)
    return word[:length]

# Parametric sweep array for Phase 1 Calibration
resolutions = [10000, 20000]
k_sweeps = [8, 16]
lambda_sweeps = [1.5, 2.0]

for name, config in systems_config.items():
    print(f"\n--- Sweeping Parameter Space For: {name} ---")
    for N in resolutions:
        word = generate_system_word(config, N)
        raw_pts = build_rauzy_points(word, config['M'])
        
        for k in k_sweeps:
            for lam in lambda_sweeps:
                boundary_cloud = extract_adaptive_boundary(raw_pts, k_neighbors=k, lambda_factor=lam)
                node_len = len(boundary_cloud)
                
                if node_len < 15:
                    print(f"  N={N:<5} | k={k:<2} | λ={lam:.1f} -> NUMERICALLY COLLAPSED")
                else:
                    d_0 = compute_box_dimension(boundary_cloud)
                    print(f"  N={N:<5} | k={k:<2} | λ={lam:.1f} -> D_0 = {d_0:.4f} | Nodes = {node_len}")
