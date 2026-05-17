#!/usr/bin/env python3
"""
ESQET Mesoscopic Coherence Boundary Engine
Quantifies the Structural Coherence Length (Lc) where R^2 drops below 0.80.
"""

from mpmath import mp, mpf, floor, fabs
import numpy as np

mp.dps = 60

class CoherenceBoundsEngine:
    def __init__(self):
        roots = mp.polyroots([1, -1, -1, -1])
        self.beta = [r.real for r in roots if fabs(r.imag) < 1e-40][0]
        self.conj = [r for r in roots if r != self.beta][0]
        
        # Pure db2 filters
        s3 = np.sqrt(3)
        s8 = 4.0 * np.sqrt(2)
        self.db2_h = np.array([(1.0 + s3)/s8, (3.0 + s3)/s8, (3.0 - s3)/s8, (1.0 - s3)/s8])
        self.db2_g = np.array([self.db2_h[3], -self.db2_h[2], self.db2_h[1], -self.db2_h[0]])

    def beta_expansion(self, x0: mpf, max_digits=50000):
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

    def extract_wavelet_leaders(self, signal, max_depth=12):
        n_padded = 2**int(np.ceil(np.log2(len(signal))))
        x = np.interp(np.linspace(0, len(signal)-1, n_padded), np.arange(len(signal)), signal)
        
        coeffs = {}
        a = x.copy()
        for j in range(1, max_depth + 1):
            n_curr = len(a)
            if n_curr < 4:
                max_depth = j - 1
                break
            a_ext = np.pad(a, (0, 3), mode='wrap')
            lowpass = np.zeros(n_curr // 2)
            highpass = np.zeros(n_curr // 2)
            for k in range(0, n_curr, 2):
                lowpass[k//2] = np.sum(a_ext[k:k+4] * self.db2_h)
                highpass[k//2] = np.sum(a_ext[k:k+4] * self.db2_g)
            coeffs[j] = highpass
            a = lowpass

        leaders = {}
        for j in range(1, max_depth + 1):
            nj = len(coeffs[j])
            leaders_at_j = np.zeros(nj)
            for k in range(nj):
                k_min, k_max = max(0, k - 1), min(nj - 1, k + 1)
                local_sup = 0.0
                for j_prime in range(j, max_depth + 1):
                    scale_factor = 2**(j_prime - j)
                    kp_min = k_min * scale_factor
                    kp_max = min(len(coeffs[j_prime]) - 1, (k_max + 1) * scale_factor - 1)
                    if kp_min <= kp_max:
                        max_val = np.max(np.abs(coeffs[j_prime][kp_min:kp_max + 1]))
                        if max_val > local_sup:
                            local_sup = max_val
                leaders_at_j[k] = local_sup
            leaders[j] = leaders_at_j[leaders_at_j > 1e-12]
        return leaders

    def compute_r2(self, leaders, fit_j):
        log_scales = np.array([np.log(2**(-j)) for j in fit_j])
        log_S = [np.log(np.mean(leaders[j]) + 1e-15) for j in fit_j]
        slope, intercept = np.polyfit(log_scales, log_S, 1)
        y_pred = slope * log_scales + intercept
        ss_tot = np.sum((log_S - np.mean(log_S))**2)
        ss_res = np.sum((log_S - y_pred)**2)
        return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def main():
    engine = CoherenceBoundsEngine()
    seed = mpf('0.5819660112501051518')
    
    print("====================================================================")
    print("🔱 MAPPING MESOSCOPIC COHERENCE BOUNDS (Lc)")
    print("====================================================================")
    
    digits = engine.beta_expansion(seed, max_digits=50000)
    coords = engine.project_rauzy(digits)
    signal = coords[:, 0]
    leaders = engine.extract_wavelet_leaders(signal, max_depth=10)
    
    print(f"{'Sliding Window':<20} | {'Mean R²':<10} | {'Coherence Status':<18}")
    print("-" * 56)
    
    # Sweep sliding windows of depth length 3
    for start_j in range(1, 8):
        window = [start_j, start_j+1, start_j+2]
        r2 = engine.compute_r2(leaders, window)
        status = "🟢 COHERENT (R² > 0.8)" if r2 >= 0.80 else "🔴 DECAYED"
        print(f"j = {window[0]}-{window[2]:<13} | {r2:<10.4f} | {status}")
        
    print("====================================================================")

if __name__ == "__main__":
    main()
