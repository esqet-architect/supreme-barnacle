#!/usr/bin/env python3
"""
ESQET Large-Scale Scale-Window Convergence Audit
Evaluates R^2 stability, plateau convergence, and confidence intervals up to N=50,000.
"""

from mpmath import mp, mpf, floor, fabs
import numpy as np

mp.dps = 60

class ESQETAuditEngine:
    def __init__(self):
        roots = mp.polyroots([1, -1, -1, -1])
        self.beta = [r.real for r in roots if fabs(r.imag) < 1e-40][0]
        self.conj = [r for r in roots if r != self.beta][0]
        
        # db2 filters
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

    def evaluate_window(self, leaders, fit_j, q=1.0):
        """Computes scaling exponent and R^2 for a specific choice of scale window."""
        log_scales = np.array([np.log(2**(-j)) for j in fit_j])
        log_S = []
        for j in fit_j:
            S_q_j = np.mean(leaders[j]**q)
            log_S.append(np.log(S_q_j + 1e-15))
            
        log_S = np.array(log_S)
        slope, intercept = np.polyfit(log_scales, log_S, 1)
        
        y_pred = slope * log_scales + intercept
        ss_tot = np.sum((log_S - np.mean(log_S))**2)
        ss_res = np.sum((log_S - y_pred)**2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return slope, r2

def main():
    engine = ESQETAuditEngine()
    seeds = [mpf('0.5819660112501051518'), mpf('0.6180339887498948482'), mpf('0.7320508075688772935')]
    
    print("====================================================================")
    print("🔱 ESQET MASSIVE-N SCALE-WINDOW CONVERGENCE PROFILE")
    print("====================================================================")
    
    windows = {
        "Intermediate Window (j: 3-6)": [3, 4, 5, 6],
        "Deep Asymptotic Window (j: 4-8)": [4, 5, 6, 7, 8]
    }
    
    for size in [10000, 50000]:
        print(f"\n[Orbit Size N = {size}]")
        print("-" * 68)
        
        for win_name, fit_j in windows.items():
            h_estimates = []
            r2_estimates = []
            
            for seed in seeds:
                digits = engine.beta_expansion(seed, max_digits=size)
                coords = engine.project_rauzy(digits)
                signal_x = coords[:, 0]
                
                leaders = engine.extract_wavelet_leaders(signal_x)
                # Verify depth availability before calculating
                available_j = [j for j in fit_j if j in leaders and len(leaders[j]) > 0]
                
                if len(available_j) >= 2:
                    h_val, r2_val = engine.evaluate_window(leaders, available_j, q=1.0)
                    h_estimates.append(h_val)
                    r2_estimates.append(r2_val)
            
            if h_estimates:
                mean_h = np.mean(h_estimates)
                std_h = np.std(h_estimates)
                mean_r2 = np.mean(r2_estimates)
                print(f"{win_name:<32} | Mean h: {mean_h:.4f} ± {std_h:.4f} | Mean R²: {mean_r2:.4f}")
            else:
                print(f"{win_name:<32} | Insufficient scale depth hierarchy.")
                
    print("====================================================================")

if __name__ == "__main__":
    main()
