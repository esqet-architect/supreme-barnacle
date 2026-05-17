#!/usr/bin/env python3
"""
ESQET Advanced Multifractal Engine: Chhabra-Jensen + Wavelet Leaders
High-precision beta-expansion + Rauzy projection + two multifractal methods.
"""

from mpmath import mp, mpf, floor, fabs
import numpy as np
from collections import defaultdict

mp.dps = 60

class AdvancedRauzyMultifractal:
    def __init__(self):
        # Tribonacci polynomial roots: x^3 - x^2 - x - 1 = 0
        roots = mp.polyroots([1, -1, -1, -1])
        self.beta = max(roots, key=lambda r: float(r.real)).real
        self.conj = [r for r in roots if r != self.beta][0]
        print(f"Pisot eigenvalue (beta): {float(self.beta):.12f}")
        print(f"Contracting conjugate modulus |lambda|: {float(abs(self.conj)):.6f}")

    def beta_expansion(self, x0: mpf, max_digits=600):
        """High-precision greedy beta-expansion."""
        digits = []
        x = x0
        for _ in range(max_digits):
            scaled = self.beta * x
            d = int(floor(scaled))
            digits.append(d)
            x = scaled - d
            if fabs(x) < mp.mpf('1e-50'):
                break
        return digits

    def project_rauzy(self, digits):
        """Project symbolic sequence to 2D contracting hyperplane."""
        points = []
        z = mpf(0)
        for d in digits:
            z = z * self.conj + d
            points.append([float(z.real), float(z.imag)])
        return np.array(points)

    def chhabra_jensen(self, points, q_range=np.linspace(-10, 10, 81), box_scales=None):
        """Chhabra-Jensen direct configuration method for f(alpha) spectrum."""
        if box_scales is None:
            box_scales = [2**k for k in range(3, 9)]
        
        pts = points - np.min(points, axis=0)
        pts /= (np.max(pts, axis=0) + 1e-12)
        
        alpha = []
        f_alpha = []
        
        for scale in box_scales:
            boxes = defaultdict(float)
            for p in pts:
                box = (int(p[0] * scale), int(p[1] * scale))
                boxes[box] += 1.0
            p_i = np.array(list(boxes.values()))
            p_i /= np.sum(p_i)
            
            for q in q_range:
                if abs(q) < 1e-8:
                    mu_q = p_i * np.log(p_i + 1e-12)
                    tau = np.sum(mu_q) / np.log(1/scale)
                else:
                    mu_q = p_i ** q
                    mu_q /= np.sum(mu_q)
                    tau = np.sum(mu_q * np.log(p_i + 1e-12)) / np.log(1/scale)
                
                alpha_q = tau / (q - 1) if abs(q - 1) > 1e-8 else np.nan
                f_q = q * alpha_q - tau if abs(q - 1) > 1e-8 else np.nan
                
                if np.isfinite(alpha_q) and np.isfinite(f_q):
                    alpha.append(alpha_q)
                    f_alpha.append(f_q)
        
        return {"alpha": np.array(alpha), "f_alpha": np.array(f_alpha)}

    def wavelet_leaders_spectrum(self, signal, max_scale=6):
        """Basic wavelet leaders scaling estimation via 1D mapping."""
        sig = signal[:, 0]
        sig = (sig - np.mean(sig)) / np.std(sig)
        
        leaders = []
        for j in range(1, max_scale + 1):
            step = 2**j
            diffs = np.abs(np.diff(sig, n=step))
            if len(diffs) > 0:
                leaders.append(np.max(diffs))
        
        scales = np.arange(1, len(leaders)+1)
        log_lead = np.log(np.array(leaders) + 1e-12)
        log_scale = np.log(scales)
        
        coeffs = np.polyfit(log_scale, log_lead, 1)
        return {"holder_est": float(coeffs[0])}

def run_full_analysis():
    print("\n====================================================")
    print("🔱 ESQET: ADVANCED CHHABRA-JENSEN & WAVELET ENGINE")
    print("====================================================")
    
    engine = AdvancedRauzyMultifractal()
    seed = mpf('0.5819660112501051518')
    
    digits = engine.beta_expansion(seed, 400)
    coords = engine.project_rauzy(digits)
    
    print(f"\nGenerated point trace coordinates: {len(coords)}")
    
    print("\nExecuting Chhabra-Jensen direct singularity check...")
    cj = engine.chhabra_jensen(coords)
    print(f"  Singularity alpha range: [{np.min(cj['alpha']):.4f} to {np.max(cj['alpha']):.4f}]")
    print(f"  Maximum f(alpha) density scaling: {np.max(cj['f_alpha']):.4f}")
    
    print("\nExecuting Wavelet Leaders scale calculation...")
    wl = engine.wavelet_leaders_spectrum(coords)
    print(f"  Estimated localized Holder exponent h: {wl['holder_est']:.4f}")
    print("====================================================")

if __name__ == "__main__":
    run_full_analysis()
