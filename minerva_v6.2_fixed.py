#!/usr/bin/env python3
"""
MINERVA V6.2: FIXED SUBSTITUTION COCYCLE
- Separates substitution rules from increment vectors
- Reduced computational load for Termux
- Bootstrap with reasonable defaults
"""

import numpy as np
from numpy.linalg import eig, eigh
from scipy.spatial import KDTree
import csv
import warnings
import time

warnings.filterwarnings('ignore')

# ============================================================================
# 1. SUBSTITUTION SYSTEMS WITH SEPARATED RULES AND INCREMENTS
# ============================================================================

class SubstitutionCocycle:
    """
    Substitution system with:
    - substitution_rules: symbolic rewriting (strings → list of strings)
    - increment_map: symbols → 3D vectors (sum to zero for centering)
    """
    
    def __init__(self, name, matrix, substitution_rules, increment_map):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.substitution_rules = substitution_rules
        self.increment_map = increment_map
        self._validate_pisot()
    
    def _validate_pisot(self):
        evals = eig(self.matrix)[0]
        moduli = np.abs(evals)
        sorted_mod = np.sort(moduli)[::-1]
        self.is_pisot = sorted_mod[0] > 1.0 and (sorted_mod[1] < 0.99 if len(sorted_mod) > 1 else True)
        self.dominant_eigenvalue = sorted_mod[0] if len(sorted_mod) > 0 else 1.0
    
    def generate_word(self, length):
        """Generate symbolic word using substitution rules only."""
        # Start with initial symbol
        symbols = list(self.substitution_rules.keys())[0]
        current = [symbols] if isinstance(symbols, str) else symbols
        
        while len(current) < length:
            new_current = []
            for ch in current:
                new_current.extend(self.substitution_rules.get(ch, [ch]))
            current = new_current
            # Prevent runaway growth
            if len(current) > length * 2:
                break
        return current[:length]
    
    def generate_cocycle_walk(self, length=1500):
        """Generate centered walk using increment vectors."""
        # Generate symbolic word
        word = self.generate_word(length)
        
        # Map to increments
        increments = np.array([self.increment_map.get(ch, [0, 0, 0]) for ch in word[:length]])
        
        # Ensure centered (should already sum to zero by design)
        increments = increments - np.mean(increments, axis=0)
        
        # Cumulative positions
        positions = np.cumsum(increments, axis=0)
        
        return positions, increments


# Define all systems with SEPARATED rules and increments
SYSTEMS = {
    'Tribonacci': SubstitutionCocycle(
        name='Tribonacci',
        matrix=[[1, 1, 1], [1, 0, 0], [0, 1, 0]],
        substitution_rules={
            'a': ['a', 'b'],
            'b': ['a', 'c'],
            'c': ['a']
        },
        increment_map={
            'a': [1, -1, 0],
            'b': [0, 1, -1],
            'c': [-1, 0, 1]
        }
    ),
    'Fibonacci': SubstitutionCocycle(
        name='Fibonacci',
        matrix=[[1, 1], [1, 0]],
        substitution_rules={
            'a': ['a', 'b'],
            'b': ['a']
        },
        increment_map={
            'a': [1, -1, 0],
            'b': [-1, 1, 0]
        }
    ),
    'ThueMorse': SubstitutionCocycle(
        name='ThueMorse',
        matrix=[[1, 1], [1, 1]],
        substitution_rules={
            'a': ['a', 'b'],
            'b': ['b', 'a']
        },
        increment_map={
            'a': [1, -1, 0],
            'b': [-1, 1, 0]
        }
    ),
    'Sturmian': SubstitutionCocycle(
        name='Sturmian',
        matrix=[[1, 1], [1, 0]],
        substitution_rules={
            'a': ['a', 'b'],
            'b': ['a']
        },
        increment_map={
            'a': [1, -1, 0],
            'b': [-1, 1, 0]
        }
    )
}


# ============================================================================
# 2. PROJECTION (LIGHTWEIGHT)
# ============================================================================

def get_projection_basis(system, points_3d):
    """
    Get projection basis:
    - Use spectral projection if Pisot and has complex pair
    - Otherwise use PCA of the points (lighter than full eigendecomposition)
    """
    if system.is_pisot:
        evals, evecs = eig(system.matrix)
        # Find complex conjugate pair
        for i, ev in enumerate(evals):
            if abs(np.imag(ev)) > 1e-10:
                v = evecs[:, i]
                basis = np.vstack([np.real(v), np.imag(v)])
                return basis
    # Fallback: PCA (only on subsampled points for speed)
    if len(points_3d) > 500:
        subsample = points_3d[::len(points_3d)//500]
    else:
        subsample = points_3d
    centered = subsample - np.mean(subsample, axis=0)
    cov = centered.T @ centered / len(subsample)
    evals, evecs = eigh(cov)
    return evecs[:, -2:].T


def project_to_plane(points_3d, basis):
    """Project 3D points to 2D using given basis."""
    return points_3d @ basis.T if basis.shape[0] == 2 else points_3d[:, :2]


# ============================================================================
# 3. LIGHTWEIGHT METRICS (REDUCED BOOTSTRAP FOR TERMUX)
# ============================================================================

def box_counting_dimension(points_2d, bootstrap=5, min_scales=8):
    """Lightweight box-counting with small bootstrap for Termux."""
    x, y = points_2d[:, 0], points_2d[:, 1]
    bbox_min = np.min(points_2d, axis=0)
    bbox_max = np.max(points_2d, axis=0)
    diag = np.linalg.norm(bbox_max - bbox_min)
    
    scales = np.logspace(-2.5, -0.5, min_scales)
    scales = scales[scales > diag / 500]
    scales = scales[:min_scales]
    
    dims = []
    for _ in range(bootstrap):
        # Subsample
        n = len(points_2d)
        idx = np.random.choice(n, min(n, n // 2), replace=False)
        x_sub = x[idx]
        y_sub = y[idx]
        
        counts = []
        valid_scales = []
        for s in scales:
            xi = np.floor((x_sub - bbox_min[0]) / (s + 1e-12))
            yi = np.floor((y_sub - bbox_min[1]) / (s + 1e-12))
            n_boxes = len(set(zip(xi, yi)))
            if n_boxes > 5:
                counts.append(n_boxes)
                valid_scales.append(s)
        
        if len(valid_scales) >= 4:
            log_s = np.log(1.0 / np.array(valid_scales))
            log_c = np.log(counts)
            slope, _ = np.polyfit(log_s, log_c, 1)
            dims.append(slope)
    
    return np.mean(dims) if dims else np.nan, np.std(dims) if dims else np.nan


def compute_metrics(points_2d):
    """Compute compact set of geometric metrics."""
    metrics = {}
    
    # 1. Box-counting dimension (reduced bootstrap)
    d0, d0_err = box_counting_dimension(points_2d, bootstrap=3)
    metrics['d0'] = d0
    metrics['d0_error'] = d0_err
    
    # 2. Anisotropy
    centered = points_2d - np.mean(points_2d, axis=0)
    cov = centered.T @ centered / len(points_2d)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    metrics['anisotropy'] = evals[0] / (evals[1] + 1e-12)
    
    # 3. Radial wandering (simplified)
    radii = np.linalg.norm(centered, axis=1)
    cum_r2 = np.cumsum(radii**2)
    start = len(radii) // 5
    log_t = np.log(np.arange(1, len(radii[start:]) + 1))
    log_r2 = np.log(cum_r2[start:] / np.arange(1, len(radii[start:]) + 1))
    if len(log_t) > 5:
        coeffs = np.polyfit(log_t, log_r2, 1)
        metrics['gamma'] = coeffs[0] / 2
    else:
        metrics['gamma'] = np.nan
    
    # 4. Connectivity (simplified)
    tree = KDTree(points_2d)
    min_dists, _ = tree.query(points_2d, k=2)
    base_r = np.median(min_dists[:, 1]) * 2.0
    pairs = tree.query_pairs(base_r)
    
    parent = list(range(len(points_2d)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    for u, v in pairs:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    
    components = len(set(find(i) for i in range(len(points_2d))))
    metrics['component_fraction'] = components / len(points_2d)
    
    # 5. Reciprocal space concentration (reduced grid)
    centered_norm = centered / (np.std(centered, axis=0) + 1e-12)
    k_grid = np.linspace(-2.0, 2.0, 8)
    structure_factors = []
    for kx in k_grid:
        for ky in k_grid:
            if abs(kx) < 1e-6 and abs(ky) < 1e-6:
                continue
            phases = np.exp(-1j * (centered_norm[:, 0] * kx + centered_norm[:, 1] * ky))
            sk = np.abs(np.sum(phases))**2 / len(centered_norm)
            structure_factors.append(sk)
    
    metrics['peak_to_bg'] = np.max(structure_factors) / (np.mean(structure_factors) + 1e-12)
    
    return metrics


# ============================================================================
# 4. MAIN BENCHMARK
# ============================================================================

def main():
    print("=" * 70)
    print("MINERVA V6.2: Fixed Substitution Cocycle (Termux-Optimized)")
    print("=" * 70)
    
    results = []
    
    for name, system in SYSTEMS.items():
        print(f"\n🔬 Processing: {name}")
        print(f"   Pisot: {system.is_pisot}, λ₁ = {system.dominant_eigenvalue:.4f}")
        
        # Generate walk
        t0 = time.time()
        positions, _ = system.generate_cocycle_walk(length=1200)
        print(f"   Walk generated: {len(positions)} points in {time.time()-t0:.2f}s")
        
        # Get projection
        basis = get_projection_basis(system, positions)
        pts_2d = project_to_plane(positions, basis)
        
        # Compute metrics
        t0 = time.time()
        metrics = compute_metrics(pts_2d)
        metrics['system'] = name
        results.append(metrics)
        print(f"   Metrics computed in {time.time()-t0:.2f}s")
        print(f"   d₀ = {metrics['d0']:.4f} ± {metrics['d0_error']:.4f}")
        print(f"   γ = {metrics['gamma']:.4f}")
        print(f"   Peak/BG = {metrics['peak_to_bg']:.2f}")
    
    # Save CSV
    csv_path = "minerva_v6.2_fixed_metrics.csv"
    fieldnames = ['system', 'd0', 'd0_error', 'gamma', 'anisotropy', 'component_fraction', 'peak_to_bg']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, np.nan) for k in fieldnames})
    
    print("\n" + "=" * 70)
    print(f"✅ Benchmark complete. Results: {csv_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
