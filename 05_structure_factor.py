#!/usr/bin/env python3
"""
Part 5: Compute structure factor from 1D projection
Lightweight FFT computation.
"""

import numpy as np
import json
from scipy.fft import fft, fftfreq

# Load spectral data
with open('data/spectral.json', 'r') as f:
    spec = json.load(f)

# Load word and convert to numbers
with open('data/tribonacci_word.txt', 'r') as f:
    word = f.read().strip()

# Convert to numeric sequence
mapping = {'1': 1.0, '2': spec['gamma_algebraic'], '3': spec['gamma_algebraic']**2}
seq = np.array([mapping[ch] for ch in word[:4000]])

# Compute structure factor
seq_centered = seq - np.mean(seq)
fft_vals = fft(seq_centered)
power = np.abs(fft_vals)**2
freqs = fftfreq(len(seq))

# Positive frequencies only
mask = freqs > 0
freqs_pos = freqs[mask]
power_pos = power[mask]

# Find peaks
from scipy.signal import find_peaks
peaks, _ = find_peaks(power_pos, height=power_pos.max() * 0.1)

print("="*50)
print("STRUCTURE FACTOR ANALYSIS")
print("="*50)
print(f"Top 5 peaks:")
top5 = np.argsort(power_pos[peaks])[-5:][::-1]
for i in top5[:5]:
    print(f"  k = {freqs_pos[peaks[i]]:.4f}  I = {power_pos[peaks[i]]:.2f}")
print(f"\nExpected fundamental frequency: 1/β = {1/spec['beta']:.4f}")

# Save
np.savez('output/structure_factor.npz', freqs=freqs_pos, power=power_pos)
print("✅ Saved to output/structure_factor.npz")
