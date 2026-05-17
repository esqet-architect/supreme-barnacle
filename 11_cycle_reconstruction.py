#!/usr/bin/env python3
"""
MINERVA ENGINE V4.7.1 — TOPOLOGICAL EDGE-CLOUD BOX DIMENSION
Features:
1. Bypasses Order-Dependent Traversals Entirely (No DFS, No Polar, No Cycles)
2. Extracts Raw Boundary Edge Cloud via Adaptive Alpha-Complex
3. Computes Pure Spatial Box-Counting Dimension (D_0) of the Fractal Boundary
"""

import numpy as np
from numpy.linalg import eig, norm
from scipy.spatial import Delaunay, KDTree
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=" * 75)
print("MINERVA ENGINE V4.7.1: UNORDERED EDGE-CLOUD BOX DIMENSION")
print("=" * 75)

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
# 2. RAW BOUNDARY EDGE EXTRACTION (NO TRAVERSAL)
# ============================================================================

def extract_boundary_cloud(points):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < 15:
        return np.array([])

    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=2)
    spacing = np.median(dists[:, 1])
    adaptive_alpha = 4.0 * spacing

    rng = np.random.default_rng(42)
    pts_jittered = pts + rng.normal(0, 1e-9, pts.shape)

    try:
        tri = Delaunay(pts_jittered)
    except Exception:
        return np.array([])

    edge_count = defaultdict(int)
    for simplex in tri.simplices:
        pa, pb, pc = pts[simplex]
        a, b, c = norm(pb - pc), norm(pa - pc), norm(pa - pb)
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-14:
            continue
        circum_r = (a * b * c) / (4.0 * np.sqrt(area_sq))

        if circum_r < adaptive_alpha:
            edges = [
                tuple(sorted((simplex[0], simplex[1]))),
                tuple(sorted((simplex[1], simplex[2]))),
                tuple(sorted((simplex[2], simplex[0])))
            ]
            for edge in edges:
                edge_count[edge] += 1

    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    if len(boundary_edges) == 0:
        return np.array([])

    # Extract all unique vertex coordinates contributing to the boundary
    unique_indices = np.unique(boundary_edges)
    return pts[unique_indices]

# ============================================================================
# 3. SPATIAL BOX-COUNTING DIMENSION ESTIMATOR
# ============================================================================

def compute_box_dimension(boundary_pts):
    if len(boundary_pts) < 20:
        return 0.0, 0

    # Define scale sizes relative to the boundary bounding box
    min_xyz = np.min(boundary_pts, axis=0)
    max_xyz = np.max(boundary_pts, axis=0)
    max_side = np.max(max_xyz - min_xyz)
    
    scales = max_side / np.logspace(0.5, 2.5, 15)
    counts = []
    
    for eps in scales:
        # Assign points to unique spatial grid bins of size epsilon
        grid = np.floor((boundary_pts - min_xyz) / eps).astype(int)
        unique_bins = np.unique(grid, axis=0)
        counts.append(len(unique_bins))
        
    # Linearly regress log(N) vs log(1/eps)
    coeffs = np.polyfit(np.log(1.0 / scales), np.log(counts), 1)
    return coeffs[0], len(boundary_pts)

# ============================================================================
# 4. EXECUTION LOOP
# ============================================================================

systems_config = {
    'Tribonacci Pisot': {
        'subs': {'1': '12', '2': '13', '3': '1'},
        'M': np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    },
    'Arnoux-Rauzy': {
        'subs': {'1': '123', '2': '13', '3': '1'},
        'M': np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=float)
    }
}

def generate_system_word(config, length=20000):
    word = '1'
    subs = config['subs']
    while len(word) < length:
        word = ''.join(subs.get(c, c) for c in word)
    return word[:length]

for name, config in systems_config.items():
    print(f"⏳ Extracting Raw Boundary Cloud: {name}...")
    word = generate_system_word(config, 20000)
    pts = build_rauzy_points(word, config['M'])
    boundary_cloud = extract_boundary_cloud(pts)
    
    d_0, total_nodes = compute_box_dimension(boundary_cloud)
    print(f"✅ {name:<20} : Boundary Box Dimension D_0 = {d_0:.4f} | Total Cloud Nodes = {total_nodes}")
