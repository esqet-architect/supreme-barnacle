#!/usr/bin/env python3
"""
RG Stability Analyzer for ESQET
Identifies the weakest links in the ESQET core.

The most vulnerable point: the spectral index n_T ≈ 4.90
This is a massive extrapolation from the RG flow.
"""

import numpy as np
from constraints import RGStabilityCriteria, AdversarialFalsifier

print("="*70)
print("ESQET RG STABILITY ANALYSIS")
print("Identifying the weakest link")
print("="*70)

# The problematic prediction
n_T_esqet = 4.90
n_T_cmb_bound = 0.1  # Typical CMB bound on tensor tilt

print(f"\n⚠️ WEAKEST LINK IDENTIFIED: Spectral index n_T ≈ {n_T_esqet}")
print(f"   This is {n_T_esqet / n_T_cmb_bound:.0f}× larger than typical CMB bounds")
print(f"   Must be reinterpreted as 'transient effective exponent'")

# Compute stability criteria for this prediction
criteria = RGStabilityCriteria()

print("\n📊 RG Stability Check:")
print(f"   Max allowed coupling: {criteria.thresholds['max_coupling']}")
print(f"   Is n_T={n_T_esqet} physically plausible?")
print(f"   → Only as a transient feature, not asymptotic tilt")

# Adversarial test
falsifier = AdversarialFalsifier()
print("\n⚔️ Adversarial Test:")
print("   Random constant substitution test:")
print("   If π, e, √2 work as well as φ, the 'Golden-Ratio Attractor' is an artifact.")

# Create a synthetic test
phi = (1 + np.sqrt(5)) / 2
constants = {'φ': phi, 'π': np.pi, 'e': np.e, '√2': np.sqrt(2), 'γ': 0.57721566}

# Mock data (would be real RG flow data in practice)
mock_data = np.linspace(0, 10, 100)

# Mock prediction functions
def predict_with_constant(c, x):
    return c * x * np.exp(-x / (c + 1))

for name, const in constants.items():
    pred = predict_with_constant(const, mock_data)
    # Compare with actual data (mock for now)
    # In real implementation, compare with RG flow results

print("\n" + "="*70)
print("RECOMMENDATION")
print("="*70)
print("""
The weakest link in ESQET is the spectral index n_T ≈ 4.90.
This value is likely a truncation artifact, not a physical prediction.

To pass adversarial falsification:
1. Reinterpret n_T as 'transient effective exponent'
2. Add explicit caveat that persistent blue tilt would be excluded
3. Test whether other constants (π, e, √2) produce equally good fits

The RG stability analysis should be the primary loss function,
not symbolic similarity to φ.
""")
