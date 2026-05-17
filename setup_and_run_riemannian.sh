#!/usr/bin/env bash
set -e

echo "================================================================="
echo "📱 ENVIRONMENT BOOTSTRAP: Samsung Galaxy A35 5G (AnLinux-Parrot OS)"
echo "================================================================="

# 1. Update package lists and install essential system dependencies
echo "Updating package registries and verifying python3 / numpy stack..."
apt-get update -y
apt-get install -y python3 python3-pip python3-numpy git curl python3-setuptools

# 2. Establish a clear, structured repository architecture for the geometric path
TARGET_WORKSPACE="symbolic_core/Minerva-dea-mathematica/riemannian_optimization"
mkdir -p "${TARGET_WORKSPACE}"

# 3. Deploy Grok's extended Riemannian manifold scan script into the workspace
echo "Writing extended manifold optimizer engine to workspace..."
cat << 'PY_EOF' > "${TARGET_WORKSPACE}/riemannian_manifold_scan.py"
import numpy as np
import sys

def compute_action_and_grad(S, gamma):
    """
    Computes the discrete non-local action and its standard Euclidean gradient.
    A[S] = 0.5 * sum(S_n - S_{n-1})^2 - gamma * sum(S_n * S_{n-2})
    """
    N = len(S)
    # Kinetic term (discrete spatial differences with periodic boundary constraints)
    diff = S - np.roll(S, 1)
    kinetic = 0.5 * np.sum(diff**2)
    
    # Nonlocal memory interaction tracking
    interaction = -gamma * np.sum(S * np.roll(S, 2))
    action = kinetic + interaction
    
    # Euclidean gradient calculation (dA/dS)
    grad = 2*S - np.roll(S, 1) - np.roll(S, -1) - gamma * (np.roll(S, 2) + np.roll(S, -2))
    return action, grad

def run_riemannian_scan(N=60, gammas=[0.05, 0.10, 0.15, 0.20], steps=8000, alpha=0.005):
    """
    Executes Projected Riemannian Gradient Descent across multiple gamma parameters.
    Forces the optimization path along the tangent bundle of the S^{N-1} manifold.
    """
    print(f"=== Riemannian Manifold Optimization Profile ===")
    print(f"Lattice Size N: {N} | Optimization Steps: {steps} | Learning Rate: {alpha}")
    print("-------------------------------------------------")
    
    results = {}
    for gamma in gammas:
        # Enforce exact seed reproducibility across scans
        np.random.seed(42)
        S = np.random.randn(N)
        S /= np.linalg.norm(S)
        
        ratios = []
        for step in range(steps):
            _, grad_e = compute_action_and_grad(S, gamma)
            
            # Riemannian Tangent Bundle Projection: P = I - SS^T
            # grad_riem = grad_e - <grad_e, S> * S
            grad_riem = grad_e - np.dot(grad_e, S) * S   
            
            # Update position along the tangent space
            S = S - alpha * grad_riem
            
            # Retract back onto the hypersphere manifold surface
            S /= np.linalg.norm(S)
            
            # Extract sequence ratios from late-stage steady state convergence
            if step > (steps // 2):
                # Mask out potential near-zero division edge cases safely
                with np.errstate(divide='ignore', invalid='ignore'):
                    step_ratios = np.abs(S[1:] / S[:-1])
                    # Filter out any numerical infs or nans from the ratio matrix
                    step_ratios = step_ratios[np.isfinite(step_ratios)]
                    if len(step_ratios) > 0:
                        ratios.extend(step_ratios)
        
        # Calculate trailing mean across late-stage updates
        if len(ratios) > 0:
            mean_ratio = np.nanmean(ratios[-200:])
        else:
            mean_ratio = np.nan
            
        results[gamma] = {
            "mean_ratio": mean_ratio,
            "final_norm": np.linalg.norm(S)
        }
        print(f"γ = {gamma:.3f} → Late-stage ratio ≈ {mean_ratio:.4f} (Manifold Norm: {results[gamma]['final_norm']:.2f})")
    
    print("-------------------------------------------------")
    print("✅ Manifold optimization sweep finalized successfully.")
    return results

if __name__ == "__main__":
    run_riemannian_scan()
PY_EOF

# 4. Enforce strict permissions and execute the script inside the OS layer
chmod +x "${TARGET_WORKSPACE}/riemannian_manifold_scan.py"

echo "Executing Python tracking engine..."
echo "================================================================="
python3 "${TARGET_WORKSPACE}/riemannian_manifold_scan.py"
echo "================================================================="

# 5. Commit the new workspace artifacts to the verified Git architecture
if [ -d ".git" ]; then
    echo "Staging scripts to Git repository..."
    git add "${TARGET_WORKSPACE}/riemannian_manifold_scan.py"
    git commit -m "Infrastructure: Integrate Grok's Projected Riemannian optimization manifold engine" || true
fi

echo "🚀 Run absolute. Files written to: ${TARGET_WORKSPACE}/"
