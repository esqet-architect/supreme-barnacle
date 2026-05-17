import numpy as np

def compute_action_and_gradient(S, gamma):
    N = len(S)
    val = 0.0
    grad = np.zeros(N)
    for i in range(N):
        im1 = (i - 1) % N
        im2 = (i - 2) % N
        ip1 = (i + 1) % N
        ip2 = (i + 2) % N
        val += 0.5 * (S[i] - S[im1])**2 - gamma * S[i] * S[im2]
        grad[i] += (S[i] - S[im1]) + (S[i] - S[ip1]) - gamma * (S[im2] + S[ip2])
    return val, grad

def find_stationary_state(N, gamma, max_iter=3000, lr=0.01):
    np.random.seed(42)
    S = np.random.randn(N)
    S /= np.linalg.norm(S)
    for iteration in range(max_iter):
        val, grad = compute_action_and_gradient(S, gamma)
        grad_tangent = grad - np.dot(grad, S) * S
        S -= lr * grad_tangent
        S /= np.linalg.norm(S)
        if np.linalg.norm(grad_tangent) < 1e-7:
            break
    return S

def check_stability(S, gamma):
    N = len(S)
    _, grad = compute_action_and_gradient(S, gamma)
    mu = 0.5 * np.dot(S, grad)
    
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

    I = np.eye(N)
    P = I - np.outer(S, S)
    H_rigorous = P @ H_A @ P - 2.0 * mu * P
    eigvals = np.linalg.eigvalsh(H_rigorous)
    internal_eigvals = eigvals[np.abs(eigvals) > 1e-9]
    return mu, np.min(internal_eigvals) if len(internal_eigvals) > 0 else 0.0

# Scan gamma values to find the exact phase transition
N = 50
gamma_steps = np.linspace(0.0, 0.5, 21)
print(f"Gamma Sweep for N={N}:")
print(f"{'Gamma':<10}{'Lagrange (mu)':<15}{'Lambda Min':<15}{'Status'}")
print("-" * 55)

for g in gamma_steps:
    S_stat = find_stationary_state(N, g)
    mu, lam_min = check_stability(S_stat, g)
    status = "STABLE" if lam_min > 0 else "UNSTABLE/SADDLE"
    print(f"{g:<10.3f}{mu:<15.6f}{lam_min:<15.6f}{status}")
