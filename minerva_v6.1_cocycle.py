#!/usr/bin/env python3
"""
MINERVA V6.1: Substitution-Balanced Cocycle Increments
- Eliminates fixed symbolic walks; uses centered Abelianization statistics
- Family-adapted projection matrices with Pisot validation
- Bootstrap confidence intervals for box-counting dimensions
- Reciprocal space concentration statistics (not full diffraction)

Framing: Comparative geometric analysis of deterministic symbolic walks
under family-adapted spectral projections.
"""

import numpy as np
from numpy.linalg import eig, norm, eigh
from scipy.spatial import KDTree
import csv
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# 1. SUBSTITUTION SYSTEMS WITH COCYCLE INCREMENTS
# ============================================================================

class SubstitutionCocycle:
    """
    Represents a substitution system with centered Abelianization increments.
    The increments sum to zero, eliminating ballistic drift.
    """
    
    def __init__(self, name, matrix, alphabet_map):
        self.name = name
        self.matrix = np.array(matrix, dtype=float)
        self.alphabet_map = alphabet_map  # symbols -> increment vectors
        self._validate_pisot()
    
    def _validate_pisot(self):
        """Check if matrix has Pisot property (dominant eigenvalue >1, others <1)"""
        evals = eig(self.matrix)[0]
        moduli = np.abs(evals)
        sorted_mod = np.sort(moduli)[::-1]
        self.is_pisot = sorted_mod[0] > 1.0 and sorted_mod[1] < 0.99
        self.dominant_eigenvalue = sorted_mod[0]
    
    def generate_word(self, length):
        """Generate substitution word up to length"""
        word = list(self.alphabet_map.keys())[0]
        while len(word) < length:
            # Apply substitution rules
            new_word = []
            for letter in word:
                new_word.extend(self.alphabet_map.get(letter, [letter]))
            word = new_word
        return word[:length]
    
    def generate_cocycle_walk(self, length=2000):
        """
        Generate centered walk using Abelianization increments.
        Increments sum to zero by construction.
        """
        word = self.generate_word(length)
        
        # Build increment vectors
        increments = np.array([self.alphabet_map.get(ch, [0,0,0]) for ch in word])
        
        # Ensure centered (should already sum to zero by design)
        increments = increments - np.mean(increments, axis=0)
        
        # Cumulative sum for positions
        positions = np.cumsum(increments, axis=0)
        
        return positions, increments


# Define substitution systems with centered increments
SYSTEMS = {
    'Tribonacci': SubstitutionCocycle(
        name='Tribonacci',
        matrix=[[1, 1, 1], [1, 0, 0], [0, 1, 0]],
        alphabet_map={
            'a': [1, -1, 0],
            'b': [0, 1, -1],
            'c': [-1, 0, 1]
        }
    ),
    'Fibonacci': SubstitutionCocycle(
        name='Fibonacci',
        matrix=[[1, 1], [1, 0]],
        alphabet_map={
            'a': [1, -1, 0],
            'b': [-1, 1, 0]
        }
    ),
    'ThueMorse': SubstitutionCocycle(
        name='ThueMorse',
        matrix=[[1, 1], [1, 1]],
        alphabet_map={
            'a': [1, -1, 0],
            'b': [-1, 1, 0]
        }
    ),
    'Sturmian': SubstitutionCocycle(
        name='Sturmian',
        matrix=[[1, 1], [1, 0]],
        alphabet_map={
            'a': [1, -1, 0],
            'b': [-1, 1, 0]
        }
    )
}


# ============================================================================
# 2. FAMILY-ADAPTED SPECTRAL PROJECTION
# ============================================================================

def get_projection_basis(system):
    """
    Extract contracting plane projection basis adapted to the substitution matrix.
    For non-Pisot systems, falls back to PCA of the walk itself.
    """
    if system.is_pisot:
        evals, evecs = eig(system.matrix)
        # Find complex conjugate pair (contracting plane)
        complex_idx = [i for i, ev in enumerate(evals) if abs(np.imag(ev)) > 1e-10]
        if len(complex_idx) >= 2:
            v1 = evecs[:, complex_idx[0]]
            basis = np.vstack([np.real(v1), np.imag(v1)])
            return basis
        else:
            # Fallback to two smallest real eigenvalues
            real_evals = [(i, ev.real) for i, ev in enumerate(evals) if abs(np.imag(ev)) < 1e-10]
            real_evals.sort(key=lambda x: abs(x[1]))
            idx = [real_evals[0][0], real_evals[1][0]] if len(real_evals) >= 2 else [0, 1]
            return np.real(evecs[:, idx].T)
    else:
        return None  # Will use PCA fallback


def project_to_plane(points, basis=None):
    """Project 3D points to 2D using given basis or PCA"""
    if basis is not None and basis.shape == (2, 3):
        return points @ basis.T
    else:
        # PCA fallback
        centered = points - np.mean(points, axis=0)
        cov = centered.T @ centered / len(points)
        evals, evecs = eigh(cov)
        return centered @ evecs[:, -2:]


# ============================================================================
# 3. ROBUST MORPHOLOGICAL METRICS
# ============================================================================

def box_counting_dimension(points, min_scales=8, bootstrap=20):
    """
    Compute box-counting dimension with bootstrap confidence intervals.
    Filters underpopulated scales automatically.
    """
    x, y = points[:, 0], points[:, 1]
    bbox = np.array([[np.min(x), np.max(x)], [np.min(y), np.max(y)]])
    diag = np.sqrt(np.sum((bbox[:, 1] - bbox[:, 0])**2))
    
    scales = np.logspace(-2.5, -0.5, min_scales * 2)
    scales = scales[scales > diag / 1000]
    scales = scales[:min_scales]
    
    # Bootstrap for confidence
    dims = []
    for _ in range(bootstrap):
        # Subsample with replacement
        idx = np.random.choice(len(points), len(points), replace=True)
        pts_sub = points[idx]
        
        counts = []
        valid_scales = []
        for s in scales:
            x_idx = np.floor((x[idx] - bbox[0, 0]) / (s + 1e-12))
            y_idx = np.floor((y[idx] - bbox[1, 0]) / (s + 1e-12))
            unique_boxes = len(set(zip(x_idx, y_idx)))
            if unique_boxes > 10:
                counts.append(unique_boxes)
                valid_scales.append(s)
        
        if len(valid_scales) >= 4:
            log_s = np.log(1.0 / np.array(valid_scales))
            log_c = np.log(counts)
            slope, _ = np.polyfit(log_s, log_c, 1)
            dims.append(slope)
    
    if len(dims) == 0:
        return np.nan, np.nan, 0
    
    mean_dim = np.mean(dims)
    std_dim = np.std(dims)
    return mean_dim, std_dim, len(dims)


def compute_geometric_metrics(pts_2d):
    """Compute comprehensive geometric statistics."""
    metrics = {}
    
    # 1. Box-counting dimension
    d0, d0_err, n_boots = box_counting_dimension(pts_2d)
    metrics['d0'] = d0
    metrics['d0_error'] = d0_err
    metrics['d0_bootstrap_n'] = n_boots
    
    # 2. Spatial anisotropy (covariance eigenvalue ratio)
    centered = pts_2d - np.mean(pts_2d, axis=0)
    cov = centered.T @ centered / len(pts_2d)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(evals)[::-1]
    metrics['anisotropy'] = evals[0] / (evals[1] + 1e-12)
    
    # 3. Radial wandering exponent γ (〈r²〉 ~ t^{2γ})
    radii = np.linalg.norm(centered, axis=1)
    times = np.arange(1, len(radii) + 1)
    # Fit power law after removing initial transient
    start = len(radii) // 10
    log_t = np.log(times[start:])
    log_r2 = np.log(np.cumsum(radii[start:]**2) / np.arange(1, len(radii[start:]) + 1))
    coeffs = np.polyfit(log_t, log_r2, 1)
    metrics['gamma'] = coeffs[0] / 2
    
    # 4. Connected component fraction (cluster compactness)
    tree = KDTree(pts_2d)
    min_dists, _ = tree.query(pts_2d, k=2)
    base_r = np.median(min_dists[:, 1]) * 1.5
    pairs = tree.query_pairs(base_r)
    
    # Union-find for connectivity
    parent = list(range(len(pts_2d)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
            return True
        return False
    
    for u, v in pairs:
        union(u, v)
    components = len(set(find(i) for i in range(len(pts_2d))))
    metrics['component_fraction'] = components / len(pts_2d)
    
    # 5. Reciprocal space concentration (empirical, not full diffraction)
    centered_pts = centered / (np.std(centered, axis=0) + 1e-12)
    k_grid = np.linspace(-2.5, 2.5, 12)
    structure_factors = []
    for kx in k_grid:
        for ky in k_grid:
            if abs(kx) < 1e-6 and abs(ky) < 1e-6:
                continue
            phases = np.exp(-1j * (centered_pts[:, 0] * kx + centered_pts[:, 1] * ky))
            sk = np.abs(np.sum(phases))**2 / len(centered_pts)
            structure_factors.append(sk)
    
    metrics['max_peak'] = np.max(structure_factors)
    metrics['mean_bg'] = np.mean(structure_factors)
    metrics['peak_to_bg'] = metrics['max_peak'] / (metrics['mean_bg'] + 1e-12)
    
    return metrics


# ============================================================================
# 4. MAIN BENCHMARK
# ============================================================================

def main():
    print("=" * 70)
    print("MINERVA V6.1: Substitution-Balanced Cocycle Geometry")
    print("Framing: Comparative geometric analysis under family-adapted projections")
    print("=" * 70)
    
    results = []
    
    for name, system in SYSTEMS.items():
        print(f"\n🔬 Processing: {name}")
        print(f"   Pisot property: {system.is_pisot}, λ₁ = {system.dominant_eigenvalue:.4f}")
        
        # Generate cocycle walk
        positions, increments = system.generate_cocycle_walk(length=2500)
        
        # Get projection basis
        basis = get_projection_basis(system)
        if basis is not None:
            print(f"   Using spectral projection (complex pair)")
        else:
            print(f"   Using PCA fallback projection")
        
        # Project to 2D
        pts_2d = project_to_plane(positions, basis)
        
        # Compute metrics
        metrics = compute_geometric_metrics(pts_2d)
        metrics['system'] = name
        results.append(metrics)
        
        print(f"   d₀ = {metrics['d0']:.4f} ± {metrics['d0_error']:.4f}")
        print(f"   γ = {metrics['gamma']:.4f}")
        print(f"   Anisotropy = {metrics['anisotropy']:.4f}")
        print(f"   Peak/BG ratio = {metrics['peak_to_bg']:.2f}")
    
    # Save CSV
    csv_path = "minerva_v6.1_cocycle_metrics.csv"
    fieldnames = ['system', 'd0', 'd0_error', 'gamma', 'anisotropy', 
                  'component_fraction', 'max_peak', 'mean_bg', 'peak_to_bg']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, np.nan) for k in fieldnames})
    
    print("\n" + "=" * 70)
    print("✅ Benchmark complete")
    print(f"   Results saved to: {csv_path}")
    print("\nInterpretive note: Metrics are comparative geometric statistics,")
    print("not physical diffraction or transport coefficients.")
    print("=" * 70)


if __name__ == "__main__":
    main()
