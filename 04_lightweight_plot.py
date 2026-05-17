#!/usr/bin/env python3
"""
Part 4: Lightweight Rauzy fractal plot
Only runs if matplotlib is available, otherwise skips.
"""

import numpy as np
import json
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available — skipping plot")
    sys.exit(0)

print("="*50)
print("GENERATING RAUZY FRACTAL PLOT")
print("="*50)

# Load points
points = np.load('data/rauzy_points.npy')
with open('data/spectral.json', 'r') as f:
    spec = json.load(f)

# Downsample for performance
step = max(1, len(points) // 10000)
points_small = points[::step]

fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(points_small[:, 0], points_small[:, 1], 
           c='#c8972a', s=0.5, alpha=0.6, rasterized=True)
ax.set_facecolor('#0d0d0f')
ax.set_xlabel('E_c axis 1', color='#555050')
ax.set_ylabel('E_c axis 2', color='#555050')
ax.set_title(f'Rauzy Fractal\nβ = {spec["beta"]:.6f}', color='#e8e2d6')
ax.set_aspect('equal')
plt.savefig('output/rauzy_fractal.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved to output/rauzy_fractal.png")
