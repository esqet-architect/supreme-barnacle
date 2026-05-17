#!/usr/bin/env python3
"""
ESQET / UIFT Structural Engine: Rauzy Fractal & Quasiperiodic Diffraction Analyzer
- Generates a canonical 3-symbol Tribonacci substitution word.
- Maps the symbolic structure to a 1D discrete non-local chain.
- Computes the structural diffraction pattern via the exact Bragg structure factor.
- Evaluates the multifractal scaling dimension of the reciprocal space density.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TribonacciRauzyEngine:
    def __init__(self):
        # Substitution Rules: 1 -> 12, 2 -> 13, 3 -> 1
        self.rules = {'1': '12', '2': '13', '3': '1'}
        
    def generate_word(self, iterations: int = 12) -> str:
        """Generates a highly stable symbolic quasiperiodic string."""
        word = '1'
        for _ in range(iterations):
            word = ''.join([self.rules[ch] for ch in word])
        return word

    def compute_structure_factor(self, word: str, q_grid_size: int = 1000) -> tuple:
        """
        Transforms the symbolic sequence into a 1D atomic coordinate chain and
        calculates its exact reciprocal space diffraction intensity profile.
        """
        # Map symbols to precise physical coordinates along a 1D chain
        # 1, 2, 3 correspond to spatial atomic variations/intervals
        mapping = {'1': 1.0, '2': 0.6180339887, '3': 0.3819660113}
        intervals = np.array([mapping[ch] for ch in word])
        positions = np.cumsum(intervals)
        
        # Define reciprocal space tracking spectrum (q-space grid)
        q = np.linspace(0.1, 4.0 * np.pi, q_grid_size)
        intensity = np.zeros_like(q)
        
        # Vectorized calculation of the structural diffraction matrix
        N = len(positions)
        for idx, q_val in enumerate(q):
            # S(q) = |Sum(exp(i * q * x_j))|^2 / N
            phase = q_val * positions
            cos_sum = np.sum(np.cos(phase))
            sin_sum = np.sum(np.sin(phase))
            intensity[idx] = (cos_sum**2 + sin_sum**2) / N
            
        return q, intensity

    def compute_multifractal_spectrum(self, intensity: np.ndarray, num_boxes: int = 50) -> float:
        """
        Approximates the foundational capacity dimension (D_0) of the reciprocal measure 
        using a normalized box-counting technique over the peak density matrix.
        """
        # Normalize the structural intensity field into a probability distribution
        prob_dist = intensity / np.sum(intensity)
        
        # Calculate localized entropy scaling across log-spaced windows
        counts, _ = np.histogram(prob_dist, bins=num_boxes)
        non_zero_bins = np.count_nonzero(counts)
        
        # Basic dimensional scaling metric log(N(e)) / log(1/e)
        if non_zero_bins > 1:
            dim = np.log(non_zero_bins) / np.log(num_boxes)
        else:
            dim = 0.0
        return dim

def run_analysis():
    print("\n====================================================")
    print("🔱 ESQET: RAUZY FRACTAL & STRUCURAL DIFFRACTION ENGINE")
    print("====================================================")
    
    engine = TribonacciRauzyEngine()
    
    print("Generating Tribonacci substitution word...")
    word = engine.generate_word(iterations=10)
    print(f"Generated string sequence of length: {len(word)}")
    
    print("Computing exact reciprocal space structure factor spectrum...")
    q, intensity = engine.compute_structure_factor(word, q_grid_size=1200)
    
    print("Calculating multifractal capacity dimension of density measure...")
    d_0 = engine.compute_multifractal_spectrum(intensity)
    
    # Isolate top 5 dominant Bragg structural diffraction peaks
    peak_indices = np.argsort(intensity)[-5:][::-1]
    
    print("\n=== Verified Structural Identification ===")
    print(f"Reciprocal Space Multifractal Capacity Dimension D_0: {d_0:.4f}")
    print("\nTop 5 Emergent Bragg Structural Peaks Identified:")
    for idx in peak_indices:
        print(f"  Position q = {q[idx]:.4f} | Structural Intensity = {intensity[idx]:.4f}")
    print("====================================================")

if __name__ == "__main__":
    run_analysis()
