#!/usr/bin/env python3
"""
esqet_multifractal_suite.py
===========================
ESQET: Full Multifractal Suite — MFDFA Implementation
High-precision Tribonacci β-expansion + Rauzy projection.
"""

from mpmath import mp, mpf, floor, fabs
import numpy as np

mp.dps = 60

class ESQET_Multifractal_Suite:
    def __init__(self):
        # Algebraic equation for Tribonacci: x^3 - x^2 - x - 1 = 0
        roots = mp.polyroots([1, -1, -1, -1])
        self.beta = [r.real for r in roots if fabs(r.imag) < 1e-40][0]
        self.conj = [r for r in roots if r != self.beta][0]
        print(f"β = {float(self.beta):.12f} | Contracting |λ| = {float(abs(self.conj)):.6f}")

    def beta_expansion(self, x0: mpf, max_digits=1500):
        digits = []
        x = x0
        for _ in range(max_digits):
            scaled = self.beta * x
            d = int(floor(scaled))
            digits.append(d)
            x = scaled - d
            if fabs(x) < mp.mpf('1e-55'):
                break
        return digits

    def project_rauzy(self, digits):
        points = []
        z = mpf(0)
        for d in digits:
            z = z * self.conj + d
            points.append([float(z.real), float(z.imag)])
        return np.array(points)

    def mfdfa(self, signal, q_values=np.linspace(-5, 5, 41), min_scale=16, max_scale_factor=0.25):
        N = len(signal)
        scales = np.unique(np.logspace(np.log10(min_scale), np.log10(int(N * max_scale_factor)), 40).astype(int))
        
        hq = np.zeros(len(q_values))
        
        # Track fluctuation functions over moments q
        for iq, q in enumerate(q_values):
            Fq = []
            for s in scales:
                if s < 5 or s >= N//2:
                    continue
                n_segments = N // s
                rms = []
                for v in range(n_segments):
                    start = v * s
                    end = start + s
                    seg = signal[start:end]
                    
                    # Local linear detrending of individual segments
                    x = np.arange(s)
                    poly = np.polyfit(x, seg, 1)
                    trend = np.polyval(poly, x)
                    detrended = seg - trend
                    rms.append(np.sqrt(np.mean(detrended**2)))
                rms = np.array(rms)
                
                if len(rms) > 0:
                    # Guard against zeros during non-linear moment scaling
                    rms_filtered = rms[rms > 0]
                    if q == 0:
                        # Logarithmic standard for zeroth moment
                        Fq.append(np.exp(np.mean(np.log(rms_filtered))))
                    else:
                        Fq.append(np.mean(rms_filtered ** q) ** (1.0 / q))
                        
            if len(Fq) > 1:
                logF = np.log(Fq)
                logS = np.log(scales[:len(Fq)])
                slope, _ = np.polyfit(logS, logF, 1)
                hq[iq] = slope

        # Classical Legendre Transform mappings to pull singularity coordinates
        tau = hq * q_values - 1.0
        alpha = np.gradient(tau) / np.gradient(q_values)
        f_alpha = q_values * alpha - tau

        return {
            "q": q_values,
            "hq": hq,
            "alpha": alpha,
            "f_alpha": f_alpha,
            "D0_est": float(np.max(f_alpha[np.isfinite(f_alpha)])) if len(f_alpha) > 0 else 0.0
        }

def main():
    engine = ESQET_Multifractal_Suite()
    # Transcendental initialization seed
    seed = mpf('0.5819660112501051518')
    
    digits = engine.beta_expansion(seed, 1500)
    coords = engine.project_rauzy(digits)
    sig = coords[:, 0]  # First dimensional spatial projection analysis

    print(f"Projected Points: {len(coords)}")
    print("\n=== EXECUTING MFDFA ANALYSIS ENGINE ===")
    
    mf = engine.mfdfa(sig)
    
    # Locate index closest to standard Hurst h(2)
    idx_h2 = np.argmin(np.abs(mf['q'] - 2))
    
    print(f"Generalized Hurst h(2)  ≈ {mf['hq'][idx_h2]:.4f}")
    print(f"Singularity Spectrum Edge Bounds:")
    print(f"  ↳ α_min               ≈ {np.min(mf['alpha']):.4f}")
    print(f"  ↳ α_max               ≈ {np.max(mf['alpha']):.4f}")
    print(f"Estimated Hausdorff Dimension D0: {mf['D0_est']:.4f}")
    print("═"*60)

if __name__ == "__main__":
    main()
