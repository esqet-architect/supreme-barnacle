#!/usr/bin/env python3
"""
08_sadic_boundary_visualizer.py
===============================
Generates and renders a side-by-side comparison of Stationary vs Non-Stationary 
S-adic Rauzy boundary dynamics. Maps localized Hölder roughness spatially 
to reveal singular geometry variations.
"""

import numpy as np
from numpy.linalg import eig, norm
from scipy.spatial import KDTree
import os

class SAdicVisualizer:
    def __init__(self):
        self.morphisms = {
            'A': {'1': '12', '2': '13', '3': '1'},
            'B': {'1': '13', '2': '1',  '3': '21'}
        }
        self.matrices = {
            'A': np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
            'B': np.array([[1, 1, 1], [0, 0, 1], [1, 0, 0]], dtype=float)
        }

    def generate_sadic_word(self, seq_pattern, target_len=10000):
        word = '1'
        idx = 0
        while len(word) < target_len:
            rule = self.morphisms[seq_pattern[idx % len(seq_pattern)]]
            word = ''.join(rule[ch] for ch in word)
            idx += 1
        return word[:target_len]

    def compute_projection_plane(self, seq_pattern):
        M_composite = np.eye(3)
        for label in reversed(seq_pattern):
            M_composite = M_composite @ self.matrices[label]
            
        evals, evecs = eig(M_composite)
        sort_idx = np.argsort(-np.abs(evals))
        
        w = evecs[:, sort_idx[1]]
        e1, e2 = w.real.copy(), w.imag.copy()
        e1 /= norm(e1)
        e2 -= np.dot(e2, e1) * e1
        e2 /= norm(e2)
        return np.array([e1, e2])

    def generate_points(self, word, pi_c):
        basis_R3 = {
            '1': np.array([1., 0., 0.]),
            '2': np.array([0., 1., 0.]),
            '3': np.array([0., 0., 1.])
        }
        delta = {ch: pi_c @ v for ch, v in basis_R3.items()}
        points = np.zeros((len(word), 2))
        pos = np.zeros(2)
        for i, ch in enumerate(word):
            points[i] = pos
            pos += delta[ch]
        return points

    def extract_boundary_with_local_scores(self, points, k_neighbors=15, percentile_threshold=85):
        tree = KDTree(points)
        boundary_scores = []
        
        for idx, pt in enumerate(points):
            distances, indices = tree.query(pt, k=k_neighbors)
            neighbors = points[indices[1:]]
            local_vectors = neighbors - pt
            mean_vector = np.mean(local_vectors, axis=0)
            score = norm(mean_vector) / (np.mean(distances[1:]) + 1e-6)
            boundary_scores.append((score, pt))
            
        scores = np.array([s[0] for s in boundary_scores])
        thresh = np.percentile(scores, percentile_threshold)
        
        boundary_indices = [i for i, s in enumerate(boundary_scores) if s[0] >= thresh]
        boundary_pts = np.array([boundary_scores[i][1] for i in boundary_indices])
        raw_scores = scores[boundary_indices]
        
        return boundary_pts, raw_scores

if __name__ == "__main__":
    print("=" * 65)
    print("INITIALIZING S-ADIC RENDER ENGINE")
    print("=" * 65)
    
    viz = SAdicVisualizer()
    scenarios = {
        "Stationary (AAAA)": "AAAA",
        "S-adic Mix (ABAB)": "ABAB"
    }
    
    plot_data = {}
    for name, seq in scenarios.items():
        print(f"Generating data for {name}...")
        word = viz.generate_sadic_word(seq, target_len=8000)
        pi_c = viz.compute_projection_plane(seq)
        points = viz.generate_points(word, pi_c)
        boundary_pts, scores = viz.extract_boundary_with_local_scores(points, percentile_threshold=86)
        plot_data[name] = (points, boundary_pts, scores)

    try:
        import matplotlib.pyplot as plt
        
        BG, WHT, DIM = '#0d0d0f', '#e8e2d6', '#555050'
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), facecolor=BG)
        
        for idx, (name, (points, boundary, scores)) in enumerate(plot_data.items()):
            # Row 0: Full Attractor
            ax_full = axes[0, idx]
            ax_full.set_facecolor('#131316')
            ax_full.scatter(points[:, 0], points[:, 1], c='#1a9b9b', s=0.2, alpha=0.3, rasterized=True)
            ax_full.set_title(f"{name} - Full Attractor Cloud", color=WHT, fontsize=10)
            ax_full.set_aspect('equal')
            ax_full.tick_params(colors=DIM, labelsize=8)
            ax_full.grid(True, color='#1e1e22', linewidth=0.5)
            
            # Row 1: Boundary Geometry Singularities
            ax_bound = axes[1, idx]
            ax_bound.set_facecolor('#131316')
            sc = ax_bound.scatter(boundary[:, 0], boundary[:, 1], c=scores, cmap='inferno', s=0.8, alpha=0.8, rasterized=True)
            ax_bound.set_title(f"{name} - Isolated Geometric Boundary", color=WHT, fontsize=10)
            ax_bound.set_aspect('equal')
            ax_bound.tick_params(colors=DIM, labelsize=8)
            ax_bound.grid(True, color='#1e1e22', linewidth=0.5)
            
            cbar = fig.colorbar(sc, ax=ax_bound, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7, colors=DIM)
            cbar.set_label('Asymmetry Multiplier (Edge Roughness)', color=DIM, fontsize=8)

        fig.suptitle('Renormalization Boundaries: Stationary vs Non-Stationary S-adic Space', color=WHT, fontsize=12, y=0.96)
        os.makedirs('output', exist_ok=True)
        plt.savefig('output/sadic_boundary_comparison.png', dpi=200, facecolor=BG, bbox_inches='tight')
        print("\n✅ Render Complete: output/sadic_boundary_comparison.png")
        
    except ImportError:
        print("\nMatplotlib missing. Exporting direct raw data arrays to disk.")
        os.makedirs('data', exist_ok=True)
        for name, (points, boundary, scores) in plot_data.items():
            clean_name = name.replace(" ", "_").replace("(", "").replace(")", "")
            np.save(f'data/{clean_name}_points.npy', points)
            np.save(f'data/{clean_name}_boundary.npy', boundary)
            np.save(f'data/{clean_name}_scores.npy', scores)
        print("✅ Data vectors preserved in data/ directory.")
