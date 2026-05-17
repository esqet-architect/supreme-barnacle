#!/usr/bin/env python3
"""
ESQET Spectral Bridge Engine
Derives the substitution matrix eigenvalues and extracts the algebraic log-periodic frequency.
"""

import numpy as np
import cmath

class SpectralBridgeEngine:
    def __init__(self):
        # Tribonacci substitution matrix M for: 0->01, 1->02, 2->0
        self.M = np.array([
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 0]
        ])

    def compute_eigenstructures(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.M)
        
        # Sort by absolute magnitude to cleanly isolate dominant vs subdominant modes
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        return eigenvalues[idx], eigenvectors[:, idx]

def main():
    engine = SpectralBridgeEngine()
    vals, vecs = engine.compute_eigenstructures()
    
    print("====================================================================")
    print("🔱 ALGEBRAIC EIGENSTRUCTURE & DISCRETE RENORMALIZATION FREQUENCY")
    print("====================================================================")
    print("Substitution Matrix M:\n", engine.M)
    print("-" * 68)
    
    beta = vals[0].real
    lam_1 = vals[1]
    lam_2 = vals[2]
    
    print(f"Dominant Pisot Root (β):        {beta:.15f}")
    print(f"Subdominant Conjugate 1 (λ_1):  {lam_1:.15f}")
    print(f"Subdominant Conjugate 2 (λ_2):  {lam_2:.15f}")
    print("-" * 68)
    
    # Extract the complex argument (phase angle) of the subdominant Galois modes
    theta_1 = cmath.phase(lam_1)
    theta_2 = cmath.phase(lam_2)
    mag_sub = np.abs(lam_1)
    
    print(f"Subdominant Modulus |λ_k|:     {mag_sub:.15f}")
    print(f"Complex Phase θ_1 (radians):    {theta_1:.15f} ({np.degrees(theta_1):.4f}°)")
    print(f"Complex Phase θ_2 (radians):    {theta_2:.15f} ({np.degrees(theta_2):.4f}°)")
    print("-" * 68)
    
    # Theoretical DSI scale step frequency calculation
    # In dyadic wavelet scale coordinates j ~ log2(n), the frequency scales against the base
    omega_theoretical = theta_1 / np.log(2)
    expected_period_j = (2 * np.pi) / omega_theoretical if omega_theoretical != 0 else 0
    
    print(f"Predicted Log-Periodic Frequency (ω): {omega_theoretical:.15f}")
    print(f"Predicted Wavelet Scale Period (Δj):  {expected_period_j:.4f} dyadic scales")
    print("====================================================================")

if __name__ == "__main__":
    main()
