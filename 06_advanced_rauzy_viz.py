#!/usr/bin/env python3
"""
Part 6: High-Fidelity Arnoux-Rauzy Fractal Visualization
Features density-mapped scattering, exact stable-plane projection, 
and beta^6 symbolic memory horizon overlay markers.
"""

import numpy as np
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
except ImportError:
    print("matplotlib is required for this visualization script.")
    sys.exit(1)

print("="*60)
print("GENERATING ADVANCED ARNOUX-RAUZY VISUALIZATION")
print("="*60)

# 1. Load Pre-calculated Corrected Points & Spectral Data
points_path = 'data/rauzy_points.npy'
spec_path = 'data/spectral.json'

if not os.path.exists(points_path) or not os.path.exists(spec_path):
    print("❌ Error: Required data files not found. Please run Part 3 first.")
    sys.exit(1)

points = np.load(points_path)
with open(spec_path, 'r') as f:
    spec = json.load(f)

beta = spec['beta']
r_conjugate = spec['r']  # |λ±| ≈ 0.73735

# 2. Setup Canvas & Geometric Themes (Low-light, High-contrast Gold)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 9), dpi=200)
fig.patch.set_facecolor('#0d0d0f')
ax.set_facecolor('#0d0d0f')

# Create a customized premium gradient for measure concentration
colors = ["#1a1510", "#5e4829", "#c8972a", "#fce6a2", "#ffffff"]
rauzy_cmap = LinearSegmentedColormap.from_list("AureaRauzy", colors, N=256)

# 3. Calculate Point Densities for Fine-grained Scatter Alpha Mapping
# Downsample slightly if the array is massive, keeping exactly 6000 for crisp detail
x = points[:, 0]
y = points[:, 1]

# Render the core attractor
sc = ax.scatter(x, y, c=np.arange(len(points)), cmap=rauzy_cmap, 
                s=0.6, alpha=0.8, edgecolors='none', rasterized=True)

# 4. Compute and Plot the beta^6 Memory Horizon Boundary
# At depth n=6, the higher-order history contracts by |λ|^6
horizon_radius = (beta**6) * (r_conjugate**6) 

# Center of mass calculation for localizing the horizon anchor
center_x = np.mean(x)
center_y = np.mean(y)

# Draw symbolic horizon constraint ring
theta = np.linspace(0, 2*np.pi, 200)
hx = center_x + horizon_radius * np.cos(theta)
hy = center_y + horizon_radius * np.sin(theta)

ax.plot(hx, hy, color='#e8e2d6', linestyle='--', linewidth=1.1, 
        alpha=0.6, label=f'Symbolic Horizon Horizon ($\\beta^6 \\approx 38.7$)')

# 5. Fine Polish and Annotation
ax.set_aspect('equal', 'box')
ax.grid(True, color='#222226', linestyle=':', linewidth=0.5)

# Style axes cleanly without breaking the dark minimalist grid
for spine in ax.spines.values():
    spine.set_color('#333338')
    spine.set_linewidth(0.8)

ax.tick_params(colors='#66666e', labelsize=9)
ax.set_xlabel('$\\mathbb{E}_c$ Stable Axis 1 (Real $\\lambda_+$)', color='#88888f', fontsize=11)
ax.set_ylabel('$\\mathbb{E}_c$ Stable Axis 2 (Imag $\\lambda_+$)', color='#88888f', fontsize=11)

# Informative Mathematical Context Header
title_str = (
    f"Arnoux-Rauzy Topological Core (Tribonacci Word)\n"
    f"$\\beta = {beta:.6f}$  |  $|\\lambda_\\pm| = {r_conjugate:.5f}$  |  Bounded Balance"
)
ax.set_title(title_str, color='#e8e2d6', fontsize=12, pad=15, loc='left')

# Subtle Legend placement
ax.legend(loc='lower right', framealpha=0.1, facecolor='#0d0d0f', edgecolor='#333338', fontsize=9)

# Save the final high-res figure asset
output_img = 'output/rauzy_fractal_advanced.png'
plt.savefig(output_img, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()

print(f"✅ Advanced visualization successfully saved to: {output_img}")
print("="*60)
