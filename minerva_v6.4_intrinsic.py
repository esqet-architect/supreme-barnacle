import os
import numpy as np
import pandas as pd

def get_substitution_matrix(sub_dict, alphabet):
    d = len(alphabet)
    M = np.zeros((d, d))
    char_to_idx = {char: i for i, char in enumerate(alphabet)}
    for j, char in enumerate(alphabet):
        target = sub_dict.get(char, "")
        for bit in target:
            if bit in char_to_idx:
                M[char_to_idx[bit], j] += 1
    return M

def get_intrinsic_cocycles(M, alphabet):
    d = len(M)
    # Perron-Frobenius extraction
    evals, evecs = np.linalg.eig(M)
    pf_idx = np.argmax(np.abs(evals))
    pf_vec = np.abs(evecs[:, pf_idx])
    
    # Normalize to frequency vector f
    f = pf_vec / np.sum(pf_vec)
    
    # Intrinsic increments: v_i = e_i - f
    increments = {}
    for i, char in enumerate(alphabet):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        increments[char] = e_i - f
        
    # Check strict Pisot condition
    all_evals = np.sort(np.abs(evals))[::-1]
    dominant = all_evals[0]
    rest = all_evals[1:]
    is_pisot = float(dominant) > 1.0 and all(float(r) < 1.0 for r in rest if np.abs(r) > 1e-9)
    
    return increments, f, is_pisot, evals, evecs

def generate_sequence(sub_dict, seed, steps=12):
    seq = seed
    for _ in range(steps):
        seq = "".join(sub_dict.get(c, c) for c in seq)
        if len(seq) > 50000:
            seq = seq[:50000]
            break
    return seq

def compute_walk(seq, increments):
    d = len(next(iter(increments.values())))
    walk = np.zeros((len(seq) + 1, d))
    for i, char in enumerate(seq):
        walk[i+1] = walk[i] + increments.get(char, np.zeros(d))
    return walk

def project_spectrally(walk, evals, evecs):
    # Find non-dominant components for the stable plane
    pf_idx = np.argmax(np.abs(evals))
    d = len(evals)
    
    # Filter out the dominant PF dimension
    remaining_indices = [i for i in range(d) if i != pf_idx]
    
    if len(remaining_indices) >= 2:
        # Check for complex conjugate pairs in non-dominant space
        v1 = evecs[:, remaining_indices[0]]
        if np.iscomplex(v1).any():
            proj_space = np.vstack([np.real(v1), np.imag(v1)])
        else:
            v2 = evecs[:, remaining_indices[1]]
            proj_space = np.vstack([np.real(v1), np.real(v2)])
    else:
        # Fallback to standard projection if dimension < 3
        proj_space = np.eye(2, d)
        
    return dot_project(walk, proj_space)

def dot_project(walk, proj_space):
    # Orthogonalize projection space for clean spatial metrics
    q, _ = np.linalg.qr(proj_space.T)
    return walk @ q

def compute_wandering_exponent(points):
    n_steps = len(points)
    indices = np.arange(1, n_steps)
    # Distance from origin in the projected contracting plane
    displacements = np.linalg.norm(points[1:], axis=1)
    
    # Filter out zeros to prevent log issues
    valid = displacements > 1e-8
    if not valid.any():
        return 0.0
        
    log_n = np.log(indices[valid])
    log_r = np.log(displacements[valid])
    
    # Linear regression to find gamma: <|X_n|> ~ n^gamma
    gamma, _ = np.polyfit(log_n, log_r, 1)
    return gamma

def compute_correlation_dimension(points, num_samples=500):
    n = len(points)
    if n > num_samples:
        idx = np.random.choice(n, num_samples, replace=False)
        sampled_points = points[idx]
    else:
        sampled_points = points
        
    # Vectorized pairwise Euclidean distances
    diffs = sampled_points[:, np.newaxis, :] - sampled_points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=-2).flatten()
    dists = dists[dists > 1e-6]  # Exclude self-distances
    
    if len(dists) == 0:
        return 0.0
        
    # Define scale window based on data distribution
    r_min = np.percentile(dists, 5)
    r_max = np.percentile(dists, 30)
    
    if r_min == r_max:
        return 0.0
        
    radii = np.logspace(np.log10(r_min), np.log10(r_max), 15)
    c_r = []
    
    for r in radii:
        count = np.sum(dists < r)
        c_r.append(count)
        
    c_r = np.array(c_r)
    valid = c_r > 0
    
    if np.sum(valid) < 2:
        return 0.0
        
    d2, _ = np.polyfit(np.log(radii[valid]), np.log(c_r[valid]), 1)
    return d2

# System Configurations
systems = {
    "Tribonacci": {"sub": {"a": "ab", "b": "ac", "c": "a"}, "seed": "a", "alpha": "abc", "steps": 11},
    "Fibonacci": {"sub": {"a": "ab", "b": "a"}, "seed": "a", "alpha": "ab", "steps": 14},
    "Sturmian": {"sub": {"a": "abaab", "b": "aba"}, "seed": "a", "alpha": "ab", "steps": 7},
    "Thue-Morse": {"sub": {"a": "ab", "b": "ba"}, "seed": "a", "alpha": "ab", "steps": 11}
}

results = []

print("=== MINERVA V6.4: INTRINSIC COCYCLE & SPECTRAL ANALYSIS ===")

for name, config in systems.items():
    M = get_substitution_matrix(config["sub"], config["alpha"])
    increments, f, is_pisot, evals, evecs = get_intrinsic_cocycles(M, config["alpha"])
    
    seq = generate_sequence(config["sub"], config["seed"], config["steps"])
    walk = compute_walk(seq, increments)
    projected = project_spectrally(walk, evals, evecs)
    
    gamma = compute_wandering_exponent(projected)
    d2 = compute_correlation_dimension(projected)
    
    results.append({
        "System": name,
        "Strict Pisot": is_pisot,
        "Wandering Gamma": round(gamma, 4),
        "Correlation D2": round(d2, 4),
        "Sequence Length": len(seq)
    })
    
    print(f"\nSystem: {name}")
    print(f"  Strict Pisot: {is_pisot}")
    print(f"  PF Frequencies: { {k: round(v, 4) for k, v in zip(config['alpha'], f)} }")
    print(f"  Wandering Exponent (Gamma): {gamma:.4f}")
    print(f"  Correlation Dimension (D2): {d2:.4f}")

# Export directly to CSV
df = pd.DataFrame(results)
df.to_csv("~/rauzy_analysis/minerva_v6.4_metrics.csv", index=False)
print("\nResults exported successfully to ~/rauzy_analysis/minerva_v6.4_metrics.csv")
