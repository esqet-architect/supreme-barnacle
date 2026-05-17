#!/usr/bin/env python3
"""
MINERVA ENGINE V4.6.6 — RESOLUTION STABILITY & GEOMETRIC CONTINUATION
Features:
1. Center-of-Mass Angular Sweep Ordering (Eliminates DFS Branch Teleportation)
2. Multi-Resolution Invariant Convergence Testing (N = 5k, 10k, 20k)
3. Preserves Adaptive Alpha Horizons & Boundary-Relative Radii Scaling
"""

import numpy as np
from numpy.linalg import eig, norm
from scipy.spatial import Delaunay, KDTree
import networkx as nx
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=" * 75)
print("MINERVA ENGINE V4.6.6: RESOLUTION STABILITY & ANGULAR CONTINUATION")
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
# 2. TOPOLOGY & ANGULAR CONTINUATION
# ============================================================================

def order_boundary_points(points):
    """Replaces DFS tree traversal with a strict geometric polar sweep."""
    if len(points) == 0:
        return np.array([])
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    idx = np.argsort(angles)
    return points[idx]

def alpha_shape(points):
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
        a = norm(pb - pc)
        b = norm(pa - pc)
        c = norm(pa - pb)
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 1e-14:
            continue
        area = np.sqrt(area_sq)
        circum_r = (a * b * c) / (4.0 * area)

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

    G = nx.Graph()
    G.add_edges_from(boundary_edges)
    if len(G) == 0:
        return np.array([])
        
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    
    # Extract structural points from largest component and apply angular ordering
    component_pts = pts[np.array(list(largest))]
    return order_boundary_points(component_pts)

# ============================================================================
# 3. TURNING-ANGLE REGRESSION
# ============================================================================

def local_turning_oscillation(contour, i, radius):
    pt = contour[i]
    dists = norm(contour - pt, axis=1)
    mask = dists < radius
    if np.sum(mask) < 6:
        return 1e-6
    segment = contour[mask]
    diffs = np.diff(segment, axis=0)
    norms = norm(diffs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1e-12
    tangents = diffs / norms
    angles = np.arccos(np.clip(np.sum(tangents[:-1] * tangents[1:], axis=1), -1.0, 1.0))
    return np.sum(np.abs(angles)) + 1e-6

def compute_holder(contour):
    if len(contour) < 20:
        return np.array([0.0])
    
    edge_lengths = norm(np.diff(contour, axis=0), axis=1)
    h = np.median(edge_lengths) if len(edge_lengths) > 0 else 1e-3
    radii = h * np.logspace(0.3, 1.5, 12)

    alphas = []
    for i in range(2, len(contour) - 2, 1):
        local = [local_turning_oscillation(contour, i, r) for r in radii]
        local = np.array(local)
        valid = local > 1e-5
        if np.sum(valid) >= 5:
            coeffs = np.polyfit(np.log(radii[valid]), np.log(local[valid]), 1)
            alphas.append(coeffs[0])
            
    return np.array(alphas) if alphas else np.array([0.0])

# ============================================================================
# 4. CONFIGURATION & MULTI-RES EXPERIMENTAL LOOP
# ============================================================================

systems_config = {
    'Tribonacci Pisot': {
        'subs': {'1': '12', '2': '13', '3': '1'},
        'M': np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    },
    'Plastic Number': {
        'subs': {'1': '12', '2': '31', '3': '1'},
        'M': np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
    },
    'Arnoux-Rauzy': {
        'subs': {'1': '123', '2': '13', '3': '1'},
        'M': np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=float)
    },
    'Non-Pisot Control': {
        'subs': {'1': '12', '2': '21', '3': '3'},
        'M': np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    }
}

def generate_system_word(config, length):
    word = '1'
    subs = config['subs']
    while len(word) < length:
        word = ''.join(subs.get(c, c) for c in word)
    return word[:length]

resolutions = [5000, 10000, 20000]
stability_data = {name: {} for name in systems_config.keys()}

for N in resolutions:
    print(f"\n--- Running Resolution Experiment: N = {N} ---")
    for name, config in systems_config.items():
        word = generate_system_word(config, N)
        pts = build_rauzy_points(word, config['M'])
        contour = alpha_shape(pts)
        alphas = compute_holder(contour)
        
        if len(alphas) <= 1 and alphas[0] == 0.0:
            stability_data[name][N] = "COLLAPSED"
            print(f"❌ {name:<20} : Collapsed")
        else:
            mu, sigma = np.mean(alphas), np.std(alphas)
            stability_data[name][N] = (mu, sigma, len(alphas))
            print(f"✅ {name:<20} : μ = {mu:.4f}, σ = {sigma:.4f} | Nodes = {len(alphas)}")

print("\n" + "=" * 75)
print("📊 FINAL MULTI-RESOLUTION STABILITY MATRIX")
print("=" * 75)
for name in systems_config.keys():
    print(f"\nSYSTEM: {name}")
    for N in resolutions:
        val = stability_data[name][N]
        if val == "COLLAPSED":
            print(f"  N = {N:<5} : COLLAPSED")
        else:
            print(f"  N = {N:<5} : μ = {val[0]:.4f}, σ = {val[1]:.4f} | Nodes = {val[2]}")
print("=" * 75)
