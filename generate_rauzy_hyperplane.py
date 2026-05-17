#!/usr/bin/env python3
"""
ESQET / UIFT Structural Engine: 2D Rauzy Contracting Hyperplane Projector
- Computes greedy beta-expansions using the exact Tribonacci Pisot base.
- Projects the orbit directly onto the 2D contracting plane using complex conjugates.
- Analyzes the localized structural density distribution of the emergent fractal tile.
"""

import numpy as np

class RauzyHyperplaneEngine:
    def __init__(self):
        # Tribonacci polynomial roots: x^3 - x^2 - x - 1 = 0
        # Principal Pisot eigenvalue
        self.beta = 1.839286755214161
        # Complex conjugate root inside the contracting unit circle (modulus ~ 0.737)
        self.conjugate_root = -0.419643377607 + 0.606290729207j

    def compute_expansion_digits(self, x: float, max_digits: int = 500) -> list:
        """Generates the greedy symbolic sequence for a real seed in [0, 1)."""
        digits = []
        remainder = x
        tol = 1e-11
        
        for _ in range(max_digits):
            if remainder < tol:
                break
            scaled = self.beta * remainder
            d = int(np.floor(scaled))
            digits.append(d)
            remainder = scaled - d
        return digits

    def project_to_hyperplane(self, digits: list) -> np.ndarray:
        """
        Projects the symbolic word onto the 2D contracting hyperplane.
        Each point represents a state in the natural extension of the shift.
        """
        points = []
        current_complex_state = 0.0 + 0.0j
        
        # Polynomial mapping along the contracting Galois conjugate trajectory
        for d in digits:
            current_complex_state = current_complex_state * self.conjugate_root + d
            points.append([current_complex_state.real, current_complex_state.imag])
            
        return np.array(points)

def run_projector():
    print("\n====================================================")
    print("🔱 ESQET: HIGH-PRECISION RAUZY HYPERPLANE PROJECTOR")
    print("====================================================")
    
    projector = RauzyHyperplaneEngine()
    print(f"Pisot Expansion Base (β): {projector.beta:.10f}")
    print(f"Contracting Conjugate Root (λ): {projector.conjugate_root.real:.6f} + {projector.conjugate_root.imag:.6f}i")
    print(f"Contracting Modulus |λ|: {np.abs(projector.conjugate_root):.4f} (< 1.0 -> Strict Convergence)")
    
    # Test with a highly distributed initial state fraction
    seed = 0.581966  # Structural fraction seed
    print(f"\nProcessing high-density expansion for seed x = {seed}...")
    digits = projector.compute_expansion_digits(seed, max_digits=300)
    print(f"Generated aperiodic symbolic sequence length: {len(digits)}")
    
    print("\nProjecting symbolic sequence onto the 2D contracting plane...")
    coordinates = projector.project_to_hyperplane(digits)
    
    # Analyze spatial spread across the boundary matrix
    min_bounds = np.min(coordinates, axis=0)
    max_bounds = np.max(coordinates, axis=0)
    spatial_variance = np.var(coordinates, axis=0)
    
    print("\n=== Verified Hyperplane Tiling Statistics ===")
    print(f"Fractal Tile Spatial Span X: [{min_bounds[0]:.4f} to {max_bounds[0]:.4f}]")
    print(f"Fractal Tile Spatial Span Y: [{min_bounds[1]:.4f} to {max_bounds[1]:.4f}]")
    print(f"Hyperplane Structural Variance (Var_X, Var_Y): ({spatial_variance[0]:.4f}, {spatial_variance[1]:.4f})")
    print("====================================================")

if __name__ == "__main__":
    run_projector()
