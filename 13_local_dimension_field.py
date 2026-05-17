#!/usr/bin/env python3
"""
MINERVA ENGINE V4.8.0 — LOCAL DIMENSION FIELD DIAGNOSTIC
Features:
1. Local Density Adaptive Alpha Complex Extraction
2. Sliding Spatial Window Neighborhood Isolation
3. Point-by-Point Local Box Dimension Estimation (D_local(x))
4. Sector Variance and Spatial Anisotropy Profiling
"""

import numpy as np
from numpy.linalg import eig, norm
from scipy.spatial import Delaunay, KDTree
import warnings
warnings.filterwarnings('ignore')

print("===========================================================================")
print("MINERVA ENGINE V4.8.0: LOCAL DIMENSION FIELD DIAGNOSTIC")
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

def extract_adaptive_boundary(points, k_neighbors=12, lambda_factor=1.8):
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) < k_neighbors + 2:
        return np.array([])

    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=k_neighbors + 1)
    local_scales = dists[:, -1]

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
# 3. LOCAL FIELD DIMENSION ESTIMATOR (D_local(x))
# ============================================================================

def compute_local_dimension_field(boundary_pts, neighborhood_size=40):
    if len(boundary_pts) < neighborhood_size:
        return np.array([0.0])

    tree = KDTree(boundary_pts)
    local_dims = []

    # Sample a representative subset to map the field across the boundary structure
    sample_indices = np.linspace(0, len(boundary_pts) - 1, min(200, len(boundary_pts)), dtype=int)

    for idx in sample_indices:
        pt = boundary_pts[idx]
        # Isolate local spatial sliding window via k-NN query
        dists, neighbors = tree.query(pt, k=neighborhood_size)
        local_cluster = boundary_pts[neighbors]
        
        # Local box-counting on isolated neighborhood window
        min_xy = np.min(local_cluster, axis=0)
        max_xy = np.max(local_cluster, axis=0)
        max_side = np.max(max_xy - min_xy)
        
        if max_side < 1e-10:
            continue
            
        scales = max_side / np.logspace(0.2, 1.2, 6)
        counts = []
        for eps in scales:
            grid = np.floor((local_cluster - min_xy) / eps).astype(int)
            unique_bins = np.unique(grid, axis=0)
            counts.append(len(unique_bins))
            
        coeffs = np.polyfit(np.log(1.0 / scales), np.log(counts), 1)
        local_dims.append(coeffs[0])

    return np.array(local_dims)

# ============================================================================
# 4. EXECUTION AND DIAGNOSTIC EXTRACTION
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

def generate_system_word(config, length=20000):
    word = '1'
    subs = config['subs']
    while len(word) < length:
        word = ''.join(subs.get(c, c) for c in word)
    return word[:length]

for name, config in systems_config.items():
    print(f"\nEvaluating Local Field Architecture for: {name}...")
    word = generate_system_word(config, 20000)
    raw_pts = build_rauzy_points(word, config['M'])
    
    # Isolate boundary using stable baseline adaptive parameters
    boundary_cloud = extract_adaptive_boundary(raw_pts, k_neighbors=12, lambda_factor=1.8)
    
    if len(boundary_cloud) < 50:
        print(f"❌ {name:<20} : Insufficient edge nodes for local field extraction.")
        continue
        
    local_field = compute_local_dimension_field(boundary_cloud, neighborhood_size=40)
    
    mean_d = np.mean(local_field)
    std_d = np.std(local_field)
    max_d = np.max(local_field)
    min_d = np.min(local_field)
    
    print(f"✅ {name:<20} Field Results:")
    print(f"   Mean D_loc  = {mean_d:.4f}")
    print(f"   Std Dev     = {std_d:.4f}  <-- Anisotropy Indicator")
    print(f"   Field Range = [{min_d:.4f} , {max_d:.4f}]")
