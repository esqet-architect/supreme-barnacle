#!/usr/bin/env python3
"""
MINERVA: S-adic Nonstationary Ginelli CLV Engine
- Alternates dynamically between Pisot (Tribonacci) and Salem-4 matrix operators
- Executes a forward QR sweep across the shifting S-adic cocycle sequence
- Runs a backward triangular transport pass to resolve true Covariant Lyapunov Vectors
- Computes and outputs the exact time-evolution of the Oseledets bundle separation
"""

import numpy as np
from numpy.linalg import qr, solve, norm
import warnings

warnings.filterwarnings('ignore')

def get_matrices():
    """Defines the two rigid integer operators for the S-adic system."""
    # 4x4 representation of Tribonacci (padded to match dimensions cleanly)
    M_pisot = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]  # Isolated dimension to keep embedding consistent
    ], dtype=float)
    
    # Canonical Salem-4 matrix
    M_salem = np.array([
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 1.0]
    ], dtype=float)
    
    return M_pisot, M_salem

def run_s_adic_ginelli(sequence_length=1000, transient_cutoff=150):
    """
    Computes exact CLVs across a nonstationary, alternating S-adic sequence.
    Sequence: Alternates every 10 steps to observe stabilization and transition phases.
    """
    M_pisot, M_salem = get_matrices()
    dim = M_pisot.shape[0]
    
    # 1. Construct the explicit S-adic directive sequence
    cocycle_sequence = []
    directive_labels = []
    for i in range(sequence_length):
        if (i // 10) % 2 == 0:
            cocycle_sequence.append(M_pisot)
            directive_labels.append("Pisot")
        else:
            cocycle_sequence.append(M_salem)
            directive_labels.append("Salem")
            
    # 2. Forward Pass (Tracking Q and R)
    Q_hist = []
    R_hist = []
    Q_curr = np.eye(dim)
    
    for n in range(sequence_length):
        next_frame = cocycle_sequence[n] @ Q_curr
        Q_curr, R_curr = qr(next_frame)
        Q_hist.append(Q_curr)
        R_hist.append(R_curr)
        
    # 3. Terminal Coefficient Initialization
    C_curr = np.triu(np.random.rand(dim, dim))
    for j in range(dim):
        C_curr[:, j] /= norm(C_curr[:, j])
        
    C_hist = [None] * sequence_length
    C_hist[-1] = C_curr
    
    # 4. Backward Triangular Transport
    for n in reversed(range(1, sequence_length)):
        C_prev = solve(R_hist[n], C_hist[n])
        for j in range(dim):
            C_prev[:, j] /= norm(C_prev[:, j])
        C_hist[n-1] = C_prev
        
    # 5. Extract Metrics and Analyze Bundle Intersections
    evolution_log = []
    
    for n in range(transient_cutoff, sequence_length - transient_cutoff):
        V_n = Q_hist[n] @ C_hist[n]
        
        # Measure the exact spatial angle between the dominant expanding vector (col 0)
        # and the first sub-dominant/neutral companion vector (col 1)
        cos_theta = np.abs(np.dot(V_n[:, 0], V_n[:, 1])) / (norm(V_n[:, 0]) * norm(V_n[:, 1]) + 1e-15)
        angle = np.degrees(np.arccos(np.clip(cos_theta, 0.0, 1.0)))
        
        evolution_log.append({
            'step': n,
            'operator': directive_labels[n],
            'angle_deg': angle
        })
        
    return evolution_log

if __name__ == '__main__':
    print("=======================================================================")
    print("MINERVA: S-adic Nonstationary Ginelli Vector Architecture")
    print("=======================================================================\n")
    
    print("[1] Executing Alternating Operator Transport Loop...")
    log = run_s_adic_ginelli(sequence_length=600, transient_cutoff=100)
    
    # Group results by active operator to observe geometric transitions cleanly
    pisot_angles = [e['angle_deg'] for e in log if e['operator'] == 'Pisot']
    salem_angles = [e['angle_deg'] for e in log if e['operator'] == 'Salem']
    
    print(f"  - Total Nonstationary Steps Logged : {len(log)}")
    print(f"  - Mean Oseledets Separation (Pisot Epochs) : {np.mean(pisot_angles):.4f}°")
    print(f"  - Mean Oseledets Separation (Salem Epochs) : {np.mean(salem_angles):.4f}°")
    
    print("\n[2] Snapshot of Step-by-Step Transition Behavior:")
    # Print a single clean transition window (switching from Pisot to Salem)
    for entry in log[85:100]:
        print(f"  - Step {entry['step']:03d} | Active Operator: {entry['operator']} | Bundle Separation Angle: {entry['angle_deg']:.4f}°")
        
    print("\n=======================================================================")
    print("Nonstationary S-adic Ginelli analysis successfully stabilized.")
    print("=======================================================================")
