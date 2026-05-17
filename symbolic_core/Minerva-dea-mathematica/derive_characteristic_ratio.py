import numpy as np

def find_characteristic_roots(gamma):
    """
    For a linear recurrence variant of the discrete Euler-Lagrange equations:
    (S_n - S_{n-1}) - (S_{n+1} - S_n) - gamma * (S_{n-2} + S_{n+2}) = 0
    Rearranging to a symmetric characteristic polynomial around r^0:
    -gamma*r^{-2} - 1*r^{-1} + 2*r^0 - 1*r^1 - gamma*r^2 = 0
    Multiplying through by r^2 gives:
    -gamma*r^4 - r^3 + 2*r^2 - r - gamma = 0
    """
    # Polynomial coefficients from highest degree (r^4) to lowest (r^0)
    coeffs = [-gamma, -1.0, 2.0, -1.0, -gamma]
    roots = np.roots(coeffs)
    
    # We are looking for real, sub-unitary roots (0 < r < 1) 
    # representing stable spatial decay profiles
    valid_roots = [np.real(r) for r in roots if np.isreal(r) and 0 < np.real(r) < 1]
    
    return valid_roots

print(f"{'Gamma':<10}{'Real Sub-Unitary Roots (r)':<30}")
print("-" * 40)
for g in [0.05, 0.10, 0.13, 0.20, 0.25]:
    roots = find_characteristic_roots(g)
    roots_str = ", ".join([f"{r:.6f}" for r in roots])
    print(f"{g:<10.2f}{roots_str:<30}")
