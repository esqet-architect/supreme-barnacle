#!/usr/bin/env python3
"""
rauzy_spatial_plotter.py
========================
Reads the updated json logs containing explicit variance properties and plots
confidence bounds using raw string literals to prevent escape anomalies.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless execution configuration
import matplotlib.pyplot as plt

def main():
    json_path = 'data/ensemble_statistical_metrics.json'
    if not os.path.exists(json_path):
        print(f"[-] Error: Missing data source file at {json_path}. Please run the validator first.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    records = data["scale_results"]
    radii = [r["radius"] for r in records]

    # Directly pull explicit structural sigmas to avoid z-score division bugs
    auth_g2 = [r["g2_stat"]["authentic"] for r in records]
    null_mu_g2 = [r["g2_stat"]["null_mean"] for r in records]
    null_sigma_g2 = [r["g2_stat"]["null_sigma"] for r in records]

    auth_g4 = [r["g4_stat"]["authentic"] for r in records]
    null_mu_g4 = [r["g4_stat"]["null_mean"] for r in records]
    null_sigma_g4 = [r["g4_stat"]["null_sigma"] for r in records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    fig.suptitle('Anisotropic Harmonic Scale-Space Profile vs. Occupancy Mask CSR Ensembles', fontsize=14, fontweight='bold')

    # --- Panel 1: Second-Order Angular Coherence Mode (m=2) ---
    ax1.plot(radii, auth_g2, color='#d32f2f', marker='o', linewidth=2.5, label='Authentic Rauzy Set')
    # Using raw string r'...' fixes internal \m warning completely
    ax1.plot(radii, null_mu_g2, color='#1976d2', linestyle='--', linewidth=2, label=r'Null CSR Mean ($\mu$)')
    
    # Using raw string r'...' fixes internal \p warning completely
    ax1.fill_between(radii, np.array(null_mu_g2) - 2*np.array(null_sigma_g2), np.array(null_mu_g2) + 2*np.array(null_sigma_g2), color='#1976d2', alpha=0.15, label=r'$\pm2\sigma$ Null Envelope')
    ax1.fill_between(radii, np.array(null_mu_g2) - np.array(null_sigma_g2), np.array(null_mu_g2) + np.array(null_sigma_g2), color='#1976d2', alpha=0.3, label=r'$\pm1\sigma$ Null Envelope')

    ax1.set_title('Second-Order Angular Coherence Mode ($m=2$)', fontsize=12, fontweight='semibold')
    ax1.set_xlabel('Spatial Evaluation Scale (r)', fontsize=11)
    ax1.set_ylabel('Harmonic Magnitude $|g_2(r)|$', fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best')

    # --- Panel 2: Fourth-Order Axial Coherence Mode (m=4) ---
    ax2.plot(radii, auth_g4, color='#d32f2f', marker='s', linewidth=2.5, label='Authentic Rauzy Set')
    ax2.plot(radii, null_mu_g4, color='#188038', linestyle='--', linewidth=2, label=r'Null CSR Mean ($\mu$)')

    ax2.fill_between(radii, np.array(null_mu_g4) - 2*np.array(null_sigma_g4), np.array(null_mu_g4) + 2*np.array(null_sigma_g4), color='#188038', alpha=0.15, label=r'$\pm2\sigma$ Null Envelope')
    ax2.fill_between(radii, np.array(null_mu_g4) - np.array(null_sigma_g4), np.array(null_mu_g4) + np.array(null_sigma_g4), color='#188038', alpha=0.3, label=r'$\pm1\sigma$ Null Envelope')

    ax2.set_title('Fourth-Order Axial Coherence Mode ($m=4$)', fontsize=12, fontweight='semibold')
    ax2.set_xlabel('Spatial Evaluation Scale (r)', fontsize=11)
    ax2.set_ylabel('Harmonic Magnitude $|g_4(r)|$', fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='best')

    plt.tight_layout()
    output_img = 'data/harmonic_ensemble_bounds.png'
    plt.savefig(output_img, dpi=300)
    print(f"[✓] Publication-grade chart safely saved without warnings to: {output_img}")

if __name__ == '__main__':
    main()
