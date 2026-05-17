import numpy as np

def analyze_constrained_stability(S, gamma):
    N = len(S)
    # Ensure S is explicitly normalized
    S = S / np.linalg.norm(S)
    
    # 1. Compute Unconstrained Hessian H for the action:
    # A = 0.5 * sum(S_n - S_{n-1})^2 - gamma * sum(S_n * S_{n-2})
    H = np.zeros((N, N))
    for i in range(N):
        H[i, i] = 2.0
        if i > 0:
            H[i, i-1] = -1.0
            H[i-1, i] = -1.0
        # Coupling terms (symmetric)
        if i > 1:
            H[i, i-2] -= gamma
            H[i-2, i] -= gamma
            
    # 2. Construct Tangent Space Projection Operator P = I - S * S^T
    I = np.eye(N)
    P = I - np.outer(S, S)
    
    # 3. Project Hessian onto Tangent Bundle
    H_proj = P @ H @ P
    
    # 4. Compute eigenvalues (excluding the trivial zero eigenvalue from projection)
    eigvals = np.linalg.eigvalsh(H_proj)
    
    # Sort eigenvalues to safely locate internal modes
    eigvals_sorted = np.sort(eigvals)
    
    # The projection guarantees at least one eigenvalue is effectively 0 (radial direction)
    # We look at the remaining N-1 eigenvalues
    non_zero_threshold = 1e-10
    internal_eigvals = eigvals_sorted[np.abs(eigvals_sorted) > non_zero_threshold]
    
    print(f"--- Stability Analysis (gamma = {gamma}) ---")
    print(f"Unconstrained Lambda Min: {np.linalg.eigvalsh(H)[0]:.6f}")
    print(f"Projected Tangent Bundle Lambda Min: {np.min(internal_eigvals):.6f}")
    print(f"Total Negative Internal Modes: {np.sum(internal_eigvals < -non_zero_threshold)}")
    
    return internal_eigvals

# Quick execution verification with placeholder state vector
N = 10
S_test = np.array([0.79**i for i in range(N)])
analyze_constrained_stability(S_test, gamma=0.1)
