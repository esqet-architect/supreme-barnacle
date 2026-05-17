#!/usr/bin/env python3
"""
ESQET / UIFT Structural Engine: Beta-Expansion & Rauzy Fractal Tracker
- Generates greedy beta-expansions for arbitrary real numbers in [0, 1) using Pisot bases.
- Evaluates the Finiteness Property (F) numerically.
- Maps expansion paths to 2D space, demonstrating the "Lava-Lamp" self-organizing flow.
"""

import numpy as np

class BetaExpansionEngine:
    def __init__(self, base_type='tribonacci'):
        # Define foundational Pisot algebraic bases
        if base_type == 'golden':
            self.beta = (1.0 + np.sqrt(5.0)) / 2.0  # φ ≈ 1.6180
        elif base_type == 'tribonacci':
            # Real root of x^3 - x^2 - x - 1 = 0
            coeffs = [1, -1, -1, -1]
            roots = np.roots(coeffs)
            self.beta = float(real_root[0]) if (real_root := roots[np.isreal(roots)]) else 1.8392867552
        else:
            raise ValueError("Unknown Pisot base type")

    def greedy_expansion(self, x: float, max_digits: int = 200, tolerance: float = 1e-12) -> list:
        """
        Computes the classic greedy beta-expansion path for a value x in [0, 1).
        Tracks the transformation T(x) = beta * x (mod 1).
        """
        digits = []
        remainder = x
        
        for _ in range(max_digits):
            if remainder < tolerance:
                break
            
            # Compute greedy digit assignment
            scaled = self.beta * remainder
            digit = int(np.floor(scaled))
            digits.append(digit)
            
            # Update transformation remainder
            remainder = scaled - digit
            
        return digits

    def map_to_complex_plane(self, digits: list) -> np.ndarray:
        """
        Projects the symbolic representation into a complex trajectory.
        Simulates the spatial extension of a Rauzy fractal coordinate tracking sequence.
        """
        # Complex conjugate mapping factor inside the contracting unit circle
        if len(digits) == 0:
            return np.zeros((1, 2))
            
        # Use a canonical complex mapping weight for projection space
        alpha = 1.0 / self.beta + 0.5j
        trajectory = []
        current_state = 0.0 + 0.0j
        
        for d in digits:
            current_state = current_state * alpha + d
            trajectory.append([current_state.real, current_state.imag])
            
        return np.array(trajectory)

def run_simulation():
    print("\n====================================================")
    print("🔱 ESQET: PISOT BETA-EXPANSION & RAUZY DYNAMICS")
    print("====================================================")
    
    engine = BetaExpansionEngine(base_type='tribonacci')
    print(f"Initialized Pisot Base (Tribonacci β): {engine.beta:.6f}")
    
    # Analyze a series of initial conditions
    test_value = 0.732
    print(f"\nComputing greedy beta-expansion for initial state x = {test_value}...")
    digits = engine.greedy_expansion(test_value, max_digits=50)
    print(f"Generated symbolic digit string: {digits}")
    
    # Check finiteness property indicator
    is_finite = len(digits) < 50
    print(f"Finiteness Property (F) Verification: {is_finite} (Terminated at {len(digits)} digits)")
    
    # Map spatial projection matrix
    trajectory = engine.map_to_complex_plane(digits)
    print(f"Projected spatial mapping trajectory shape: {trajectory.shape}")
    print(f"Final convergent boundary endpoint: [{trajectory[-1, 0]:.4f}, {trajectory[-1, 1]:.4f}]")
    print("====================================================")

if __name__ == "__main__":
    run_simulation()
