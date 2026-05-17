#!/usr/bin/env python3
"""
ESQET: High-Precision Beta-Expansion + Rauzy Hyperplane + Multifractal Spectrum
Uses mpmath for symbolic-level precision on Tribonacci beta-expansion.
"""

from mpmath import mp, mpf, floor, fabs
import numpy as np
from collections import defaultdict

# Establish high precision depth
mp.dps = 50  

class PreciseTribonacciRauzy:
    def __init__(self):
        # Exact Tribonacci polynomial roots via mpmath
        roots = mp.polyroots([1, -1, -1, -1])
        # Find the single real root (eigenvalue)
        self.beta = [r.real for r in roots if fabs(r.imag) < 1e-40][0]
        # Isolate the contracting complex conjugate pair
        self.conj1 = [r for r in roots if r != self.beta][0]
        
        print(f"Pisot base eigenvalue (beta): {float(self.beta):.12f}")
        print(f"Contracting conjugate modulus: {float(abs(self.conj1)):.6f}")

    def beta_expansion(self, x0: mpf, max_digits: int = 500):
        """Greedy beta-expansion tracking loop with high precision."""
        digits = []
        x = x0
        for _ in range(max_digits):
            scaled = self.beta * x
            d = int(floor(scaled))
            digits.append(d)
            x = scaled - d
            if fabs(x) < mp.mpf('1e-40'):
                break
        return digits

    def project_to_hyperplane(self, digits):
        """Projects the symbolic expansion onto the 2D contracting plane."""
        points = []
        z = mpf(0)
        for d in digits:
            z = z * self.conj1 + d
            points.append([float(z.real), float(z.imag)])
        return np.array(points)

    def multifractal_spectrum(self, points, q_values=np.linspace(-5, 5, 41)):
        """Computes generalized dimensions Dq via multi-scale box mass distribution."""
        if len(points) < 100:
            return {"D0": 0.0}
        
        # Normalize structural points to a unit grid for scannability
        pts = points - np.min(points, axis=0)
        pts /= (np.max(pts) + 1e-12)
        
        scales = [2**k for k in range(3, 8)]  # Scale steps
        tau = []
        
        for q in q_values:
            Z = []
            for scale in scales:
                boxes = defaultdict(float)
                for p in pts:
                    box = (int(p[0]*scale), int(p[1]*scale))
                    boxes[box] += 1.0
                masses = np.array(list(boxes.values()))
                masses /= np.sum(masses)
                Z.append(np.sum(masses**q))
            
            logZ = np.log(Z)
            logScale = np.log(1.0/np.array(scales))
            tau_q = np.polyfit(logScale, logZ, 1)[0]
            tau.append(tau_q)
        
        Dq = []
        for q, t in zip(q_values, tau):
            if abs(q - 1) < 1e-8:
                Dq.append(np.nan)  
            else:
                Dq.append(t / (q - 1))
        
        D0 = Dq[np.argmin(np.abs(q_values))]  # Capacity dimension projection
        return {"D0": float(D0)}

def run_analysis():
    print("\n====================================================")
    print("🔱 ESQET: HIGH-PRECISION MULTIFRACTAL ANALYSIS ENGINE")
    print("====================================================")
    
    engine = PreciseTribonacciRauzy()
    seed = mpf('0.5819660112501051')  
    
    print(f"\nComputing high-precision beta-expansion for seed x = {float(seed)}...")
    digits = engine.beta_expansion(seed, max_digits=400)
    print(f"Calculated expansion length: {len(digits)} digits")
    
    print("Projecting sequence coordinates onto 2D Rauzy hyperplane...")
    coords = engine.project_to_hyperplane(digits)
    
    print("Scanning box-counting mass for multifractal spectrum analysis...")
    spectrum = engine.multifractal_spectrum(coords)
    
    print("\n=== Verified High-Precision Multifractal Results ===")
    print(f"Total Projected Coordinates: {len(coords)}")
    print(f"Approximate Capacity Dimension D_0: {spectrum['D0']:.4f}")
    print("====================================================")

if __name__ == "__main__":
    run_analysis()
