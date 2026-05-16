#!/usr/bin/env python3
"""
pisot_spectral_analysis.py
==========================
Spectral analysis of Rauzy-projected Pisot substitution dynamics.
"""
import numpy as np
from numpy.linalg import eigvals, eig
from scipy.optimize import curve_fit, minimize
from scipy.signal import welch, find_peaks
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

M_TRIBONACCI = np.array([[1,1,1],[1,0,0],[0,1,0]], dtype=float)

def tribonacci_spectrum():
    evals = eigvals(M_TRIBONACCI)
    real_evals = evals[np.abs(evals.imag) < 1e-10].real
    cplx_evals = evals[np.abs(evals.imag) >= 1e-10]
    beta = float(np.max(np.abs(real_evals)))
    lam = cplx_evals[0]
    r = float(np.abs(lam))
    theta = float(np.angle(lam))
    omega = theta / np.log(beta)
    return {'beta': beta, 'r': r, 'theta': theta, 'omega': omega,
            'eigenvalues': evals, 'log_contraction': np.log(r)/np.log(beta)}

def rauzy_projection_matrix():
    evals, evecs = eig(M_TRIBONACCI)
    idx = np.argsort(-np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]
    v_expand = evecs[:, 0].real
    v_contract_re = evecs[:, 1].real
    v_contract_im = evecs[:, 1].imag
    return {'dominant': v_expand, 'contract_re': v_contract_re,
            'contract_im': v_contract_im, 'eigenvalues': evals}

def generate_tribonacci_orbit(n_steps=4000):
    spec = tribonacci_spectrum()
    beta = spec['beta']
    _, evecs = eig(M_TRIBONACCI)
    idx = np.argsort(-np.abs(eigvals(M_TRIBONACCI)))
    v0 = np.abs(evecs[:, idx[0]].real)
    v0 /= v0.sum()
    orbit = np.zeros(n_steps)
    v = v0.copy()
    for j in range(n_steps):
        orbit[j] = v[0] / v.sum()
        v = M_TRIBONACCI @ v
        v = v / v.sum()
    scales = np.arange(n_steps, dtype=float)
    return orbit, scales

def extract_scaling_residuals(orbit, window_sizes=None):
    N = len(orbit)
    if window_sizes is None:
        window_sizes = np.unique(np.logspace(np.log10(8), np.log10(N // 4), 60).astype(int))
        window_sizes = window_sizes[window_sizes >= 4]
    profile = np.cumsum(orbit - orbit.mean())
    log_s, log_F = [], []
    for s in window_sizes:
        n_seg = N // s
        if n_seg < 2:
            continue
        rms_vals = []
        for v in range(n_seg):
            seg = profile[v*s:(v+1)*s]
            x = np.arange(s)
            p = np.polyfit(x, seg, 1)
            res = seg - np.polyval(p, x)
            rms_vals.append(np.sqrt(np.mean(res**2)))
        F = np.mean(rms_vals)
        if F > 0:
            log_s.append(np.log(s))
            log_F.append(np.log(F))
    log_s = np.array(log_s)
    log_F = np.array(log_F)
    slope, intercept, r, _, se = linregress(log_s, log_F)
    fit_line = slope * log_s + intercept
    delta = log_F - fit_line
    fit = {'slope': slope, 'intercept': intercept, 'r2': r**2, 'se': se}
    return log_s, log_F, delta, fit

def single_mode_model(j, A, omega, phi, gamma):
    return A * np.cos(omega * j + phi) * np.exp(-gamma * j)

def multi_mode_model(j, params):
    signal = np.zeros_like(j, dtype=float)
    K = len(params) // 4
    for k in range(K):
        A, omega, phi, gamma = params[4*k:4*k+4]
        signal += A * np.cos(omega * j + phi) * np.exp(-np.abs(gamma) * j)
    return signal

def fit_single_mode(delta, j=None, omega_init=None):
    if j is None:
        j = np.arange(len(delta), dtype=float)
    if omega_init is None:
        omega_init = tribonacci_spectrum()['omega']
    try:
        popt, pcov = curve_fit(single_mode_model, j, delta,
                               p0=[np.std(delta), omega_init, 0.0, 0.01],
                               bounds=([-np.inf, 0, -np.pi, 0],
                                       [np.inf, np.inf, np.pi, np.inf]),
                               maxfev=10000)
        residuals = delta - single_mode_model(j, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((delta - delta.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        perr = np.sqrt(np.diag(pcov))
        phase_wrapped = (popt[2] % (2*np.pi)) > np.pi
        return {'success': True, 'params': popt, 'param_errors': perr,
                'r2': r2, 'residuals': residuals, 'phase_wrapped': phase_wrapped,
                'sufficient': r2 > 0.7 and not phase_wrapped,
                'verdict': 'SUFFICIENT' if (r2 > 0.7 and not phase_wrapped)
                           else 'INSUFFICIENT'}
    except Exception as e:
        return {'success': False, 'error': str(e), 'sufficient': False, 'verdict': 'FIT FAILED'}

def run_full_analysis(n_orbit=2000, verbose=True):
    if verbose:
        print("\n" + "="*60)
        print("PISOT SUBSTITUTION SPECTRAL ANALYSIS")
        print("="*60)
    spec = tribonacci_spectrum()
    if verbose:
        print(f"β = {spec['beta']:.6f}, ω = {spec['omega']:.6f}")
    orbit, scales = generate_tribonacci_orbit(n_steps=n_orbit)
    log_s, log_F, delta, fit = extract_scaling_residuals(orbit)
    j = np.arange(len(delta), dtype=float)
    sm = fit_single_mode(delta, j, omega_init=spec['omega'])
    if verbose:
        print(f"Single-mode R² = {sm.get('r2', 0):.4f}")
        print(f"Verdict: {sm.get('verdict', 'UNKNOWN')}")
    return {'spectrum': spec, 'orbit': orbit, 'delta': delta,
            'scaling_fit': fit, 'single_mode': sm}

if __name__ == '__main__':
    results = run_full_analysis(n_orbit=2000, verbose=True)
    print("\n✅ Analysis complete")

def fit_multi_mode(delta, j=None, K=3, omega_base=None):
    if j is None:
        j = np.arange(len(delta), dtype=float)
    if omega_base is None:
        omega_base = tribonacci_spectrum()['omega']

    fs = 1.0
    nperseg = min(len(delta) // 2, 256)
    freqs, psd = welch(delta, fs=fs, nperseg=nperseg)
    omega_freqs = 2 * np.pi * freqs

    peaks, _ = find_peaks(psd, height=np.percentile(psd, 60))
    peak_omegas = omega_freqs[peaks] if len(peaks) > 0 else [omega_base]
    peak_amps = np.sqrt(psd[peaks]) if len(peaks) > 0 else [np.std(delta)]

    p0 = []
    for k in range(K):
        if k < len(peak_omegas):
            om = peak_omegas[k]
        else:
            om = omega_base * (k + 1)
        A = peak_amps[k] if k < len(peak_amps) else np.std(delta) * 0.3
        p0.extend([A, om, 0.0, 0.02])
    p0 = np.array(p0)

    def objective(params):
        pred = multi_mode_model(j, params)
        return np.sum((delta - pred)**2)

    result = minimize(objective, p0, method='Nelder-Mead',
                      options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-10})

    params_fit = result.x
    pred = multi_mode_model(j, params_fit)
    residuals = delta - pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((delta - delta.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    modes = []
    for k in range(K):
        A, om, phi, gamma = params_fit[4*k:4*k+4]
        modes.append({
            'k': k, 'A': A, 'omega': om, 'phi': phi, 'gamma': abs(gamma),
            'period_logscale': 2*np.pi / om if om != 0 else np.inf
        })

    return {
        'success': result.success or r2 > 0.4, 'params': params_fit,
        'r2': r2, 'residuals': residuals, 'modes': modes, 'K': K,
        'prediction': pred
    }

def coherence_windows(delta, j=None, window=32, step=8, omega_ref=None):
    if j is None:
        j = np.arange(len(delta), dtype=float)
    if omega_ref is None:
        omega_ref = tribonacci_spectrum()['omega']

    n = len(delta)
    centers, coherences = [], []

    for start in range(0, n - window, step):
        end = start + window
        seg = delta[start:end]
        j_seg = j[start:end]
        center = j[start + window // 2]

        basis_re = np.cos(omega_ref * j_seg)
        basis_im = np.sin(omega_ref * j_seg)
        power = (np.dot(seg, basis_re)**2 + np.dot(seg, basis_im)**2) / window

        centers.append(center)
        coherences.append(power)

    centers = np.array(centers)
    coherences = np.array(coherences)
    norm_coh = coherences / (coherences.max() + 1e-12)
    mean_coh = float(norm_coh.mean())
    cv_coh = float(norm_coh.std() / (norm_coh.mean() + 1e-12))

    return {
        'j_centers': centers, 'coherence': norm_coh,
        'mean_coherence': mean_coh, 'cv_coherence': cv_coh,
        'interpretation': (
            'SUSTAINED DSI' if mean_coh > 0.6 and cv_coh < 0.5 else
            'INTERMITTENT DSI' if mean_coh > 0.3 and cv_coh >= 0.5 else
            'WEAK / ABSENT DSI'
        )
    }

def run_advanced_diagnostics(delta, j):
    print("\n" + "─"*60)
    print("RUNNING MULTI-MODE DECOMPOSITION & OPTIMIZATION")
    print("─"*60)
    
    print(f"{'K_modes':>8} {'R² Score':>12} {'AIC Value':>14}")
    print("─"*40)
    
    n = len(delta)
    best_r2 = 0.0
    best_fit = None
    
    for K in [1, 2, 3, 4]:
        fit = fit_multi_mode(delta, j, K=K)
        ss_res = np.sum(fit['residuals']**2)
        aic = n * np.log(ss_res / n + 1e-15) + 2 * (4 * K)
        print(f"{K:>8} {fit['r2']:>12.4f} {aic:>14.2f}")
        
        if fit['r2'] > best_r2:
            best_r2 = fit['r2']
            best_fit = fit

    print("─"*60)
    print(f"Optimized Multi-Mode Recovery (K={best_fit['K']}) R² = {best_fit['r2']:.4f}")
    for m in best_fit['modes']:
        print(f"  ↳ Mode {m['k']+1}: Amp={m['A']:+.4f} | ω={m['omega']:.4f} | Decay(γ)={m['gamma']:.4f}")

    coh = coherence_windows(delta, j)
    print("\n" + "─"*60)
    print("LOCAL COHERENCE SCAN DETAILS")
    print("─"*60)
    print(f"  Mean Structural Coherence : {coh['mean_coherence']:.4f}")
    print(f"  Coefficient of Variation   : {coh['cv_coherence']:.4f}")
    print(f"  Classification             : {coh['interpretation']}")
    print("─"*60)

# Patching execution into entry point dynamically
import sys
if __name__ == '__main__':
    # Re-run full orbit analysis to generate variables locally
    res = run_full_analysis(n_orbit=2000, verbose=False)
    j_coords = np.arange(len(res['delta']), dtype=float)
    run_advanced_diagnostics(res['delta'], j_coords)
