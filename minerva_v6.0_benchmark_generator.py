#!/usr/bin/env python3
"""
MINERVA V6.0: Intrinsic Spectral Projection & Symbolic Walk Characterization Suite
- Generates precise balanced, codimension-1 symbolic walks embedded in R^3.
- Builds individual family-adapted substitution matrices to isolate intrinsic contracting planes.
- Extracts box-counting dimensions with explicit R^2 regression verification over 12 scales.
- Computes true 2D reciprocal grid diffraction landscapes to map structure factor peaks.
"""

import numpy as np
from numpy.linalg import eig, norm, eigh
from scipy.spatial import KDTree
import csv
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DETERMINISTIC SYSTEM GENERATORS & MULTI-ALPHABET MAPPINGS (H_TOP = 0)
# ============================================================================
def generate_thue_morse(N):
    return [bin(n).count("1") % 2 for n in range(N)]

def generate_fibonacci(N):
    word = "0"
    while len(word) < N:
        word = word.replace("0", "01").replace("1", "0X").replace("X", "1")
    return [int(c) for c in word[:N]]

def generate_sturmian_golden(N):
    phi_inv = (np.sqrt(5) - 1.0) / 2.0
    return [int(np.floor((n + 1) * phi_inv) - np.floor(n * phi_inv)) % 2 for n in range(N)]

def generate_tribonacci_word(N):
    word = [0]
    while len(word) < N:
        next_word = []
        for symbol in word:
            if symbol == 0:
                next_word.extend([0, 1])
            elif symbol == 1:
                next_word.extend([0, 2])
            elif symbol == 2:
                next_word.append(0)
        word = next_word
    return word[:N]

# ============================================================================
# INTRINSIC FAMILY-ADAPTED SPECTRAL PROJECTION PIPELINE
# ============================================================================
def get_family_adapted_matrix(name):
    """
    Constructs the intrinsic substitution / incidence matrix unique to each 
    symbolic family to define an appropriate canonical observation frame.
    """
    if name == "Tribonacci":
        # Standard ternary substitution matrix
        return np.array([[1.0, 1.0, 1.0], 
                         [1.0, 0.0, 0.0], 
                         [0.0, 1.0, 0.0]], dtype=float)
    elif name in ["Fibonacci", "Sturmian_Golden"]:
        # Standard binary substitution matrix expanded to R^3 to match ambient walk dimensions
        return np.array([[1.0, 1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 0.5]], dtype=float)
    else: # Thue-Morse
        # Equal-length block substitution approximation
        return np.array([[1.0, 1.0, 0.0],
                         [1.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]], dtype=float)

def generate_projected_walk(name, directive_seq, target_length=1500):
    M = get_family_adapted_matrix(name)
    evals, evecs = eig(M)
    
    # Locate the complex conjugate stable contracting subspace
    complex_ids = np.where(np.abs(np.imag(evals)) > 1e-10)[0]
    
    if len(complex_ids) > 0:
        v_stable = evecs[:, complex_ids[0]]
        proj_basis = np.vstack([np.real(v_stable), np.imag(v_stable)])
    else:
        # Fallback to the two lowest-magnitude real components if no complex pair exists natively
        sort_ids = np.argsort(np.abs(evals))
        proj_basis = np.real(evecs[:, sort_ids[:2]].T)
        
    pts = np.zeros((target_length, 2))
    pos_3d = np.zeros(3)
    
    # Balanced, codimension-1 symbolic walk increments summing to zero
    basis_3d = {
        0: np.array([1.0, -1.0, 0.0]), 
        1: np.array([0.0, 1.0, -1.0]),
        2: np.array([-1.0, 0.0, 1.0])
    }
    
    for n in range(target_length):
        symbol = directive_seq[n % len(directive_seq)]
        pos_3d += basis_3d.get(symbol, np.array([-1.0, 0.0, 1.0]))
        pts[n] = proj_basis @ pos_3d
        
    return pts

# ============================================================================
# COMPREHENSIVE MORPHOLOGICAL EXPERIMENTAL TESTING
# ============================================================================
def calculate_box_counting_dimension(pts, grid_steps=12):
    x, y = pts[:, 0], pts[:, 1]
    scales = np.logspace(-2.0, -0.2, grid_steps)
    box_counts = []
    
    for s in scales:
        x_indices = np.floor((x - np.min(x)) / (s + 1e-15))
        y_indices = np.floor((y - np.min(y)) / (s + 1e-15))
        unique_boxes = set(zip(x_indices, y_indices))
        box_counts.append(len(unique_boxes))
        
    log_inv_scales = np.log(1.0 / scales)
    log_counts = np.log(box_counts)
    
    slope, intercept = np.polyfit(log_inv_scales, log_counts, 1)
    
    predicted = slope * log_inv_scales + intercept
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    ss_res = np.sum((log_counts - predicted) ** 2)
    r_squared = 1.0 - (ss_res / (ss_tot + 1e-15))
    
    return float(slope), float(r_squared)

def calculate_spatial_anisotropy_ratio(pts):
    centered_pts = pts - np.mean(pts, axis=0)
    cov_tensor = (centered_pts.T @ centered_pts) / len(pts)
    eigenvalues, _ = eigh(cov_tensor)
    return float(eigenvalues[0] / (eigenvalues[1] + 1e-15))

def calculate_normalized_radial_pair_accumulation_variance(pts, r_samples=20):
    n_pts = len(pts)
    if n_pts < 2:
        return 0.0
    tree = KDTree(pts)
    
    min_dists, _ = tree.query(pts, k=2)
    r_max = np.median(min_dists[:, 1]) * 5.0
    radii = np.linspace(0.1, r_max, r_samples)
    
    accum_profiles = []
    for r in radii:
        pairs = tree.query_pairs(r)
        density = n_pts / (np.pi * (r_max ** 2))
        expected = 0.5 * n_pts * density * (np.pi * (r ** 2))
        accum_profiles.append(len(pairs) / (expected + 1e-15))
        
    return float(np.var(accum_profiles))

def extract_wandering_exponent_gamma(pts, window_splits=6):
    radii = norm(pts, axis=1)
    horizons = np.linspace(len(pts)//4, len(pts), window_splits, dtype=int)
    variances = []
    
    for h in horizons:
        variances.append(np.var(radii[:h]) + 1e-15)
        
    gamma, _ = np.polyfit(np.log(horizons), np.log(variances), 1)
    return float(gamma)

def evaluate_2d_diffraction_grid(pts, grid_size=15):
    centered = pts - np.mean(pts, axis=0)
    k_vals = np.linspace(-3.0, 3.0, grid_size)
    max_sk = 0.0
    all_sk = []
    
    for kx in k_vals:
        for ky in k_vals:
            if kx == 0.0 and ky == 0.0:
                continue
            vec_k = np.array([kx, ky])
            phases = np.exp(-1j * (centered @ vec_k))
            s_k = (np.abs(np.sum(phases)) ** 2) / len(pts)
            all_sk.append(s_k)
            if s_k > max_sk:
                max_sk = s_k
                
    return float(max_sk), float(np.mean(all_sk))

def compute_connected_component_fraction(pts):
    n_pts = len(pts)
    tree = KDTree(pts)
    min_dists, _ = tree.query(pts, k=2)
    base_r = np.median(min_dists[:, 1])
    
    parents = list(range(n_pts))
    def find(i):
        if parents[i] == i: return i
        parents[i] = find(parents[i])
        return parents[i]
        
    pairs = tree.query_pairs(base_r * 1.5)
    comp_count = n_pts
    for u, v in pairs:
        root_u, root_v = find(u), find(v)
        if root_u != root_v:
            parents[root_u] = root_v
            comp_count -= 1
            
    return float(comp_count) / float(n_pts)

# ============================================================================
# MASTER SWEEP PIPELINE
# ============================================================================
if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA V6.0: Intrinsic Spectral Projections & Centered Walks")
    print("=======================================================================\n")
    
    N_steps = 3500
    symbolic_families = {
        "Thue-Morse": generate_thue_morse(N_steps),
        "Fibonacci": generate_fibonacci(N_steps),
        "Sturmian_Golden": generate_sturmian_golden(N_steps),
        "Tribonacci": generate_tribonacci_word(N_steps)
    }
    
    csv_filename = "minerva_universality_metrics.csv"
    header = [
        "Symbolic_Family", 
        "Box_Counting_D0",
        "D0_R_Squared",
        "Spatial_Anisotropy_Ratio", 
        "Normalized_Radial_Pair_Accum_Var",
        "Centered_Wandering_Gamma",
        "Max_2D_Diffraction_Peak",
        "Mean_2D_Diffraction_BG",
        "Connected_Component_Fraction"
    ]
    
    print(f"Profiling intrinsic configurations across {len(symbolic_families)} linguistic lineages...")
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        
        for name, sequence in symbolic_families.items():
            print(f"  -> Building Intrinsic Space Planar Map for: {name}...")
            points = generate_projected_walk(name, sequence, target_length=2000)
            
            d0_dim, d0_r2 = calculate_box_counting_dimension(points)
            anisotropy = calculate_spatial_anisotropy_ratio(points)
            accum_var = calculate_normalized_radial_pair_accumulation_variance(points)
            gamma_exp = extract_wandering_exponent_gamma(points)
            max_peak, mean_bg = evaluate_2d_diffraction_grid(points)
            comp_fraction = compute_connected_component_fraction(points[::5])
            
            writer.writerow([
                name, 
                f"{d0_dim:.6f}",
                f"{d0_r2:.6f}",
                f"{anisotropy:.6f}", 
                f"{accum_var:.6f}", 
                f"{gamma_exp:.6f}",
                f"{max_peak:.6f}",
                f"{mean_bg:.6f}",
                f"{comp_fraction:.6f}"
            ])
            
    print(f"\n[Success] Comparative morphological profiling complete.")
    print(f"Clean summary data compiled to: ./{csv_filename}")
    print("=======================================================================")
