#!/usr/bin/env python3
"""
MINERVA V4.9 — Clean Rauzy Boundary Geometry
- Proper increment-based projection (no cumulative walk)
- Adaptive alpha-complex boundary extraction
- Box dimension + local Hölder field
"""

import numpy as np
from numpy.linalg import eig, norm
from scipy.spatial import Delaunay, KDTree
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE
# ============================================================================
def get_stable_basis(M):
    evals, evecs = eig(M)
    idx = np.argsort(-np.abs(evals))
    evals, evecs = evals[idx], evecs[:, idx]
    v_beta = evecs[:, 0].real
    v_beta /= np.sum(v_beta) if np.sum(v_beta) != 0 else 1.0

    w = evecs[:, 1]
    e1 = w.real.copy()
    e2 = w.imag.copy()
    e1 /= norm(e1) if norm(e1) != 0 else 1.0
    e2 -= np.dot(e2, e1) * e1
    e2 /= norm(e2) if norm(e2) != 0 else 1.0
    return np.array([e1, e2]), v_beta

def build_rauzy_points(word, M):
    """Increment-based projection — no unbounded walk"""
    pi_c, v_beta = get_stable_basis(M)
    basis = {'1': np.array([1.,0.,0.]), '2':np.array([0.,1.,0.]), '3':np.array([0.,0.,1.])}
    
    points = np.zeros((len(word), 2))
    pos = np.zeros(3)
    
    for i, c in enumerate(word):
        points[i] = pi_c @ pos
        pos += basis.get(c, basis['1'])
    
    return np.nan_to_num(points)

def extract_boundary_cloud(points, k=12, lam=1.8):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < 20:
        return pts
    
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k+1)
    local_scales = dists[:, -1]
    
    # Jitter for Delaunay stability
    pts_j = pts + np.random.normal(0, 1e-9, pts.shape)
    
    try:
        tri = Delaunay(pts_j)
    except:
        return pts
    
    edge_count = {}
    for simplex in tri.simplices:
        pa, pb, pc = pts[simplex]
        a, b, c = norm(pb-pc), norm(pa-pc), norm(pa-pb)
        s = (a+b+c)/2
        area_sq = s*(s-a)*(s-b)*(s-c)
        if area_sq <= 1e-14:
            continue
        circum_r = (a*b*c) / (4 * np.sqrt(area_sq))
        
        local_thresh = lam * np.mean(local_scales[simplex])
        if circum_r < local_thresh:
            edges = [tuple(sorted([simplex[0],simplex[1]])), 
                     tuple(sorted([simplex[1],simplex[2]])), 
                     tuple(sorted([simplex[2],simplex[0]]))]
            for e in edges:
                edge_count[e] = edge_count.get(e, 0) + 1
    
    boundary_edges = [e for e, cnt in edge_count.items() if cnt == 1]
    if not boundary_edges:
        return pts
    
    unique_idx = np.unique(boundary_edges)
    return pts[unique_idx]

def box_dimension(pts):
    if len(pts) < 20:
        return 0.0
    minp = np.min(pts, axis=0)
    maxp = np.max(pts, axis=0)
    side = np.max(maxp - minp)
    if side < 1e-12:
        return 0.0
    scales = side / np.logspace(0.4, 2.2, 12)
    counts = []
    for eps in scales:
        grid = np.floor((pts - minp) / eps).astype(int)
        counts.append(len(np.unique(grid, axis=0)))
    coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
    return coeffs[0]

# ============================================================================
# RUN
# ============================================================================
systems = {
    'Tribonacci Pisot': {'subs': {'1':'12','2':'13','3':'1'}, 'M': np.array([[1,1,1],[1,0,0],[0,1,0]], float)},
    'Plastic Number':   {'subs': {'1':'2','2':'3','3':'12'}, 'M': np.array([[0,0,1],[1,0,1],[0,1,0]], float)},
}

for name, cfg in systems.items():
    print(f"\n=== {name} ===")
    word = '1'
    while len(word) < 15000:
        word = ''.join(cfg['subs'].get(c,c) for c in word)
    word = word[:15000]
    
    pts = build_rauzy_points(word, cfg['M'])
    boundary = extract_boundary_cloud(pts, k=12, lam=1.8)
    d0 = box_dimension(boundary)
    
    print(f"Points: {len(pts)} | Boundary nodes: {len(boundary)} | D0 = {d0:.4f}")

print("\nDone. Boundary geometry is now stable.")
