#!/usr/bin/env python3
"""
ESQET Residual Phase Matching Engine
Optimizes the phenomenological amplitude and phase offset against the algebraic frequency.
"""

import numpy as np
from scipy.optimize import curve_fit

def main():
    print("====================================================================")
    print("🔱 EMPIRICAL RESIDUAL SPECTRUM VS. ALGEBRAIC PREDICTION")
    print("====================================================================")
    
    # 1. Empirical dyadic scales and detrended residuals from your boundary runs
    fit_j = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
    # Averaged residuals across the invariant WRAP and REFLECT test runs
    residuals_empirical = np.array([0.2846, -0.1311, -0.1549, -0.1265, -0.0404, 0.0275, 0.1408])
    
    # 2. Hardcoded algebraic frequency derived from the Tribonacci spectrum
    omega_predicted = -3.139644  # theta / log(2)
    
    # 3. Define target DSI model function with fixed frequency
    def dsi_model(j, A, phi):
        return A * np.cos(omega_predicted * j + phi)
    
    # 4. Perform optimization to fit Amplitude (A) and Phase (phi)
    popt, pcov = curve_fit(dsi_model, fit_j, residuals_empirical, p0=[0.2, 0.0])
    A_fit, phi_fit = popt
    
    # 5. Calculate fitness diagnostics (R^2 and Residual Variance)
    fitted_values = dsi_model(fit_j, A_fit, phi_fit)
    ss_res = np.sum((residuals_empirical - fitted_values) ** 2)
    ss_tot = np.sum((residuals_empirical - np.mean(residuals_empirical)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    print(f"Fixed Algebraic Frequency (ω_pred): {omega_predicted:.5f}")
    print(f"Calculated Dyadic Scale Period (Δj):  {2 * np.pi / abs(omega_predicted):.4f}")
    print("-" * 68)
    print(f"Optimized Coherence Amplitude (A):    {A_fit:.5f}")
    print(f"Optimized Phase Offset (φ):          {phi_fit:.5f} rad")
    print(f"Phase Matching Alignment (R²):       {r_squared:.5f}")
    print("-" * 68)
    
    print("Scale (j) | Empirical Residual | Model Fit | Delta")
    for idx, j in enumerate(fit_j):
        print(f"  j = {int(j)}   |       {residuals_empirical[idx]:.5f}      |  {fitted_values[idx]:.5f}  | {residuals_empirical[idx]-fitted_values[idx]:+.5f}")
    print("====================================================================")

if __name__ == "__main__":
    main()
