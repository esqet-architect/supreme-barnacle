#!/usr/bin/env python3
"""
ESQET DSI Verification Engine
Extracts log-periodic residuals and tests sensitivity against boundary artifacts.
"""

from mpmath import mp, mpf, floor, fabs
import numpy as np

mp.dps = 60

class DSIVerificationEngine:
    def __init__(self):
        roots = mp.polyroots([1, -1, -1, -1])
        self.beta = [r.real for r in roots if fabs(r.imag) < 1e-40][0]
        self.conj = [r for r in roots if r != self.beta][0]
        
        s3 = np.sqrt(3)
        s8 = 4.0 * np.sqrt(2)
        self.db2_h = np.array([(1.0 + s3)/s8, (3.0 + s3)/s8, (3.0 - s3)/s8, (1.0 - s3)/s8])
        self.db2_g = np.array([self.db2_h[3], -self.db2_h[2], self.db2_h[1], -self.db2_h[0]])

    def beta_expansion(self, x0: mpf, max_digits=30000):
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

    def extract_wavelet_leaders(self, signal, mode='wrap', max_depth=10):
        n_padded = 2**int(np.ceil(np.log2(len(signal))))
        x = np.interp(np.linspace(0, len(signal)-1, n_padded), np.arange(len(signal)), signal)
        
        coeffs = {}
        a = x.copy()
        
        for j in range(1, max_depth + 1):
            n_curr = len(a)
            if n_curr < 4:
                max_depth = j - 1
                break
            a_ext = np.pad(a, (0, 3), mode=mode)
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

    def analyze_residuals(self, leaders, fit_j):
        log_scales = np.array([np.log(2**(-j)) for j in fit_j])
        log_S = np.array([np.log(np.mean(leaders[j]) + 1e-15) for j in fit_j])
        
        slope, intercept = np.polyfit(log_scales, log_S, 1)
        y_pred = slope * log_scales + intercept
        residuals = log_S - y_pred
        
        return slope, residuals

def main():
    engine = DSIVerificationEngine()
    seed = mpf('0.5819660112501051518')
    digits = engine.beta_expansion(seed, max_digits=30000)
    coords = engine.project_rauzy(digits)
    signal = coords[:, 0]
    
    print("====================================================================")
    print("🔱 DSI RESIDUAL DETRENDING & BOUNDARY SENSITIVITY")
    print("====================================================================")
    
    fit_j = [2, 3, 4, 5, 6, 7, 8]
    
    for padding_mode in ['wrap', 'reflect']:
        print(f"\n[Boundary Condition: {padding_mode.upper()}]")
        print("-" * 50)
        leaders = engine.extract_wavelet_leaders(signal, mode=padding_mode, max_depth=9)
        slope, residuals = engine.analyze_residuals(leaders, fit_j)
        
        print(f"Base Scaling Exponent (Mean h): {slope:.4f}")
        print("Scale (j) | Log-Periodic Residual (Δ_j)")
        for idx, j in enumerate(fit_j):
            print(f"  j = {j}   | Residual: {residuals[idx]:.5f}")
            
        amplitude = np.max(residuals) - np.min(residuals)
        print(f"Peak-to-Peak Residual Variance: {amplitude:.5f}")
        
    print("====================================================================")

if __name__ == "__main__":
    main()
