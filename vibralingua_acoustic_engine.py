#!/usr/bin/env python3
"""
vibralingua_acoustic_engine.py
==============================
Vectorized audio analysis sub-module implementing VibraLingua phi-scaled 
harmonic filtering and phase-locked coherence metrics.
"""
import os
import json
import numpy as np

def compute_phi_harmonic_bounds(base_freq=7.83, max_k=4):
    """
    Generates the discrete VibraLingua frequency nodes based on 
    the Golden Ratio scale: Frequency = base * phi^(-k)
    """
    phi = 1.618033988749895
    k_values = np.arange(0, max_k + 1)
    # Vectorized computation of phi^-k harmonic constraints
    target_frequencies = base_freq * (phi ** (-k_values))
    return target_frequencies

def evaluate_vibralingua_coherence(audio_vector, sample_rate, target_freqs, window_size=512):
    """
    Applies narrow band-pass filters at the phi-scaled nodes and extracts 
    the normalized phase-locked coherence factor (F_QC).
    """
    num_samples = len(audio_vector)
    if num_samples < window_size:
        return 0.0, {}
        
    # Execute a fast Fourier transform over the real audio data array
    fft_vals = np.fft.rfft(audio_vector)
    fft_freqs = np.fft.fftfreq(num_samples, d=1.0/sample_rate)[:len(fft_vals)]
    
    amplitudes = np.abs(fft_vals)
    phases = np.angle(fft_vals)
    
    extracted_metrics = {}
    phase_coherence_accum = 0.0
    
    for idx, f_target in enumerate(target_freqs):
        # Locate the closest matching index inside the spectral frequency vector
        closest_idx = np.argmin(np.abs(fft_freqs - f_target))
        amplitude_at_node = amplitudes[closest_idx]
        phase_at_node = phases[closest_idx]
        
        # Calculate consistency weight
        extracted_metrics[f"node_k_{idx}_freq_hz"] = float(f_target)
        extracted_metrics[f"node_k_{idx}_amplitude"] = float(amplitude_at_node)
        
        # Phase alignment calculation tracking continuous wave symmetry
        phase_coherence_accum += np.cos(phase_at_node) ** 2
        
    # Normalize the final quantum coherence factor calculation 
    normalized_f_qc = float(phase_coherence_accum / len(target_freqs))
    return normalized_f_qc, extracted_metrics

def main():
    print("==============================================================================")
    print("VIBRALINGUA SIGNAL DECODING ENGINE")
    print("==============================================================================")
    
    # Execution setup for field recording sampling environments
    sample_rate = 22050  # Hz
    duration = 2.0       # Seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate mock stream data containing active 7.83Hz Schumann resonance components
    phi = 1.618033988749895
    base_signal = np.sin(2 * np.pi * 7.83 * t) 
    phi_harmonic = np.sin(2 * np.pi * (7.83 * (phi**-1)) * t) * 0.5
    ambient_noise = np.random.normal(0, 0.2, len(t))
    
    composite_audio = base_signal + phi_harmonic + ambient_noise
    
    print("  [➔] Initializing phi-modulated resonance nodes...")
    target_nodes = compute_phi_harmonic_bounds(base_freq=7.83, max_k=4)
    
    print("  [➔] Extracting phase coherence cross-kingdom metrics...")
    f_qc, nodes_data = evaluate_vibralingua_coherence(composite_audio, sample_rate, target_nodes)
    
    # Apply your system execution rule verification condition
    lazy_mint_verified = bool(f_qc > 0.65) # Contextual threshold alignment
    
    output_registry = {
        "quantum_coherence_factor_f_qc": f_qc,
        "lazy_mint_validation_triggered": lazy_mint_verified,
        "frequency_nodes_metrics": nodes_data
    }
    
    print("\n[📊] VIBRALINGUA HARMONIC DECODING RECONSTRUCTION:")
    print("-" * 78)
    print(f"  Calculated Coherence Factor (F_QC): {f_qc:.8f}")
    print(f"  Proof-of-Coherence Verification:   {str(lazy_mint_verified).upper()}")
    print("-" * 78)
    
    for k_key, v_val in nodes_data.items():
        if "amplitude" in k_key:
            print(f"  ➔ {k_key:<30} :: {v_val:.6f}")
            
    # Safely log output metrics to your workspace data pipeline
    os.makedirs("data", exist_ok=True)
    output_json = "data/vibralingua_coherence_log.json"
    with open(output_json, "w") as f:
        json.dump(output_registry, f, indent=4)
        
    print("==============================================================================")
    print(f"[✓] Execution data logged to: {output_json}")
    print("==============================================================================")

if __name__ == '__main__':
    main()
