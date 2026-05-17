# Scoped Analysis of Constrained Nonlocal Discrete Transport Functionals

This repository contains the frozen research trace, numerical exploration engines, and mathematical documentation investigating discrete transport functionals modified by non-local memory coupling parameters ($\gamma$). 

## Core Focus & Boundaries
This project is **not** an experimentally validated physical theory. It is a rigorous numerical and mathematical investigation into the stability and scaling anomalies of discrete variational systems under global $L^2$ hyperspherical constraints ($\mathbb{S}^{N-1}$).

### 1. Verified Geometric Facts
* **Manifold Isolation:** We demonstrate that while unconstrained Euclidean optimization indicates structural collapse, isolating the system via a rigorous tangent-bundle projection operator $P = I - SS^\top$ reveals a robust, positive-definite internal spectrum ($\lambda_{\min} > 0$).

### 2. Documented Structural Gaps
* **The 0.79 Attractor Anomaly:** Numerical optimization routines persistently converge toward a late-stage decay sequence ratio of $S_n / S_{n-1} \approx 0.79$. However, linearized characteristic root analysis across the symmetric polynomial $-\gamma r^4 - r^3 + 2r^2 - r - \gamma = 0$ yields **exclusively complex conjugate roots**, proving that a real exponential decay sequence is analytically impossible in the linearized limit. This emergent phenomenon remains an open, unresolved gap (likely driven by non-linear constraint-curvature or optimizer artifacting).

## Repository Architecture
* `symbolic_core/`: Python-based variational engines, gradient descent scanners, and projected Hessian spectral evaluation utilities.
* `symbolic_core/Minerva-dea-mathematica/manuscript/main.tex`: Core LaTeX preprint explicitly documenting the tangent-bundle proofs, the linearized root failure contradiction, and known failure modes.
* `symbolic_core/Minerva-dea-mathematica/cognitive_pipeline_data/three_bin_audit.json`: The formal claims registry enforcing epistemic boundaries between mathematical facts and speculative interpretations.

## Release Tagging
This workspace is frozen under strict version control to preserve the integrity of the adversarial development process.
