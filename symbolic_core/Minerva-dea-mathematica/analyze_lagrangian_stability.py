import numpy as np

def compute_action_and_gradient(S, gamma):
    N = len(S)
    # A = 0.5 * sum(S_n - S_{n-1})^2 - gamma * sum(S_n * S_{n-2})
    # Assumes periodic boundary conditions for a clean closed system
    val = 0.0
    grad = np.zeros(N)
    
    for i in range(N):
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        ip1 = (i + 1) % N
        ip2 = (i + 2) % N
        
        val += 0.5 * (S[i] - S[im1])**2 - gamma * S[i] * S[im2]
        
        # Gradient contributions
        grad[i] += (S[i] - S[im1]) + (S[i] - S[ip1]) - gamma * (S[im2] + S[ip2])
        
    return val, grad

def find_stationary_state(N, gamma, max_iter=2000, lr=0.01):
    # Initialize with a random normalized state
    np.random.seed(42)
    S = np.random.randn(N)
    S /= np.linalg.norm(S)
    
    for iteration in range(max_iter):
        val, grad = compute_action_and_gradient(S, gamma)
        
        # Project gradient onto tangent space to stay on the sphere
        grad_tangent = grad - np.dot(grad, S) * S
        
        # Step against the tangent gradient
        S -= lr * grad_tangent
        S /= np.linalg.norm(S) # Correct numerical drift
        
        if np.linalg.norm(grad_tangent) < 1e-7:
            break
            
    return S

def analyze_lagrangian_hessian(S, gamma):
    N = len(S)
    _, grad = compute_action_and_gradient(S, gamma)
    
    # Compute the precise Lagrange multiplier: mu = 0.5 * S^T * grad(A)
    mu = 0.5 * np.dot(S, grad)
    
    # Construct base Action Hessian H_A
    H_A = np.zeros((N, N))
    for i in range(N):
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        ip1 = (i + 1) % N
        ip2 = (i + 2) % N
        
        H_A[i, i] = 2.0
        H_A[i, im1] -= 1.0
        H_A[i, ip1] -= 1.0
        H_A[i, im2] -= gamma
        H_A[i, ip2] -= gamma

    # Construct Projection Operator P
    I = np.eye(N)
    P = I - np.outer(S, S)
    
    # True Constrained Lagrangian Hessian: P * H_A * P - 2 * mu * P
    H_rigorous = P @ H_A @ P - 2.0 * mu * P
    
    # Compute spectrum
    eigvals = np.linalg.eigvalsh(H_rigorous)
    
    # Filter out the trivial zero eigenvalue belonging to the radial direction
    internal_eigvals = eigvals[np.abs(eigvals) > 1e-9]
    
    print(f"--- Rigorous Lagrangian Stability (N={N}, gamma={gamma}) ---")
    print(f"Lagrange Multiplier (mu): {mu:.6f}")
    print(f"True Tangent Bundle Lambda Min: {np.min(internal_eigvals):.6f}")
    print(f"Remaining Negative Internal Modes: {np.sum(internal_eigvals < -1e-9)}")
    
    return S, internal_eigvals

# Execute verification on a valid stationary configuration
S_stat = find_stationary_state(N=50, gamma=0.1)
analyze_lagrangian_hessian(S_stat, gamma=0.1)
