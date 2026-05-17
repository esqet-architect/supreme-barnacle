#!/usr/bin/env python3
"""
minerva_v8.0_validated_cj.py
=============================
Validated Chhabra-Jensen multifractal estimator.

All three ground-truth tests must pass before this estimator
is applied to any Rauzy fractal data.

Ground-truth tests
------------------
Test 1: Uniform random points in [0,1]²
    D₀ = 2.0  (box dimension of a filled square)
    Δα ≈ 0    (monofractal — uniform measure)

Test 2: Middle-third Cantor set (embedded in 2D)
    D₀ = log2/log3 ≈ 0.6309  (exact Hausdorff dimension)
    Δα ≈ 0    (monofractal — uniform measure on Cantor set)

Test 3: Asymmetric binomial cascade on [0,1]
    D₀ = 1.0  (support is the unit interval)
    Δα = log(p₁/p₂) / log(2) · something  (analytically computable)
    Δα > 0    (genuinely multifractal)

Algorithm: Chhabra & Jensen (1989), Phys Rev Lett 62(12)
    - Box-counting partition: Z(q,ε) = Σᵢ μᵢ(ε)^q
    - Escort measure: μ̃ᵢ(q,ε) = μᵢ^q / Z(q,ε)
    - α(q) = lim_{ε→0} Σᵢ μ̃ᵢ log(μᵢ) / log(ε)
    - f(q) = lim_{ε→0} Σᵢ μ̃ᵢ log(μ̃ᵢ) / log(ε)

Physical constraints that must hold
------------------------------------
    α(q) strictly decreasing in q
    f(α) ≥ 0 everywhere
    f(α) ≤ d (embedding dimension)
    f(α) = D₀ at q = 0
    f(α) is concave

If any of these fail, the estimator is not in the scaling regime.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# DATASET GENERATORS  (ground truth)
# ═══════════════════════════════════════════════════════════════════════════

def dataset_uniform_square(n=5000):
    """
    Uniform random points in [0,1]².
    Measure: uniform → monofractal.
    D₀ = 2.0,  Δα ≈ 0.
    """
    rng = np.random.default_rng(42)
    return rng.random((n, 2))


def dataset_cantor_set(generations=10):
    """
    Middle-third Cantor set, embedded as (x, 0) in 2D.
    Measure: uniform on the 2^generations endpoints.
    D₀ = log2/log3 ≈ 0.6309,  Δα ≈ 0.

    generations=10 → 1024 points, well past the scaling regime.
    """
    pts = np.array([0.0, 1.0])
    for _ in range(generations):
        pts = np.concatenate([pts / 3.0, pts / 3.0 + 2.0 / 3.0])
    pts = np.sort(np.unique(pts))
    return np.column_stack([pts, np.zeros_like(pts)])


def dataset_binomial_cascade(depth=12, p=0.3):
    """
    Asymmetric binomial (p, 1-p) multiplicative cascade on [0,1].

    Construction
    ------------
    At each level k, split each interval [a, a+h] into two halves.
    Left half gets fraction p of parent mass.
    Right half gets fraction (1-p) of parent mass.

    After `depth` levels we have 2^depth = 4096 intervals.
    Each interval centre is a point; its weight is the cascade measure μᵢ.

    Analytical results (p ≠ 0.5)
    ----------------------------
    D₀  = 1.0          (support is [0,1])
    α₀  = -[p·log p + (1-p)·log(1-p)] / log 2   (most probable singularity)
    αmin = -log(max(p,1-p)) / log 2
    αmax = -log(min(p,1-p)) / log 2
    Δα  = αmax - αmin = log((1-p)/p) / log 2  for p < 0.5

    For p=0.3:  Δα ≈ log(0.7/0.3)/log2 ≈ 1.222  (large, clearly detectable)
    """
    p2 = 1.0 - p
    n  = 2 ** depth        # number of intervals

    # Generate all interval indices and compute their cascade weights
    indices  = np.arange(n)
    weights  = np.ones(n, dtype=float)

    for level in range(depth):
        # At this level, bit `level` of the index says left (0) or right (1)
        bits    = (indices >> (depth - 1 - level)) & 1
        weights *= np.where(bits == 0, p, p2)

    # Interval centres
    h       = 1.0 / n
    centres = (indices + 0.5) * h

    # Normalize weights to sum to 1
    weights = weights / weights.sum()

    # Return as (position, weight) — we will use weight as the measure
    # The estimator must accept weighted points; see note below.
    return centres, weights


# ═══════════════════════════════════════════════════════════════════════════
# CHHABRA-JENSEN ESTIMATOR  (box-counting, correct scaling regime)
# ═══════════════════════════════════════════════════════════════════════════

class ChabbraJensenEstimator:
    """
    Direct Chhabra-Jensen multifractal spectrum estimator.

    Parameters
    ----------
    pts      : ndarray, shape (N, d)   point cloud (uniform weights)
    weights  : ndarray, shape (N,) or None
               If None, uniform weights 1/N are used.
               Pass explicit weights for the binomial cascade.
    n_scales : int    number of box scales (recommend 12-20)
    eps_min  : float  smallest box size as fraction of diameter
    eps_max  : float  largest box size as fraction of diameter
               Rule: eps_max << 1 (boxes smaller than support)
                     eps_min >> 1/sqrt(N) (boxes not too small)
    """

    def __init__(self, pts, weights=None,
                 n_scales=15, eps_min=0.01, eps_max=0.25):
        self.pts     = np.asarray(pts, dtype=float)
        self.n       = len(pts)
        self.d       = pts.shape[1] if pts.ndim > 1 else 1
        self.n_scales = n_scales
        self.eps_min  = eps_min
        self.eps_max  = eps_max

        if weights is None:
            self.weights = np.ones(self.n) / self.n
        else:
            w = np.asarray(weights, dtype=float)
            self.weights = w / w.sum()

        # Bounding box
        self.lo = self.pts.min(axis=0)
        self.hi = self.pts.max(axis=0)
        self.diameter = float(np.max(self.hi - self.lo))

        if self.diameter < 1e-12:
            raise ValueError("Point cloud has zero diameter.")

    def _box_measure(self, eps):
        """
        Compute empirical box measure μᵢ(ε) at scale ε.

        Each box is identified by its grid index.
        μᵢ = sum of weights of points in box i.

        Returns array of μᵢ for non-empty boxes (sums to 1).
        """
        shifted = self.pts - self.lo          # translate to [0, diameter]^d
        grid    = (shifted / eps).astype(int) # integer box index per point

        # Aggregate weights into boxes
        box_dict = {}
        for i in range(self.n):
            key = tuple(grid[i])
            box_dict[key] = box_dict.get(key, 0.0) + self.weights[i]

        mu = np.array(list(box_dict.values()), dtype=float)
        return mu / mu.sum()   # renormalize (should already sum to 1)

    def compute_spectrum(self, q_values):
        """
        Compute α(q) and f(α) via Chhabra-Jensen direct method.

        Returns
        -------
        alpha : ndarray   Hölder exponent spectrum α(q)
        f     : ndarray   Singularity spectrum f(α)
        D_q   : ndarray   Generalized dimensions D_q = τ(q)/(q-1)
        eps_arr : ndarray  Box scales used
        """
        q_values = np.asarray(q_values, dtype=float)

        eps_abs = np.geomspace(
            self.eps_min * self.diameter,
            self.eps_max * self.diameter,
            self.n_scales
        )

        log_eps = np.log(eps_abs)

        # Precompute box measures at all scales
        mu_list = [self._box_measure(e) for e in eps_abs]

        # Verify scaling regime: N_boxes should scale as ε^{-D}
        n_boxes = np.array([len(m) for m in mu_list])
        if n_boxes.min() < 5:
            print("  WARNING: too few boxes at smallest scale — increase eps_min")
        if n_boxes.max() > self.n / 2:
            print("  WARNING: fewer than 2 pts/box at largest scale — decrease eps_max")

        alpha_arr = np.zeros(len(q_values))
        f_arr     = np.zeros(len(q_values))
        Dq_arr    = np.zeros(len(q_values))

        for j, q in enumerate(q_values):
            num_alpha = np.zeros(self.n_scales)
            num_f     = np.zeros(self.n_scales)
            log_Zq    = np.zeros(self.n_scales)

            for s, mu in enumerate(mu_list):
                # Guard against mu=0 (shouldn't happen since we use non-empty boxes)
                mu_safe = mu[mu > 0]

                if q == 0:
                    # Z(0,ε) = number of non-empty boxes
                    Z_q = float(len(mu_safe))
                    mu_tilde = np.ones(len(mu_safe)) / Z_q
                elif q == 1:
                    # L'Hôpital limit: escort → μ itself
                    Z_q = 1.0
                    mu_tilde = mu_safe.copy()
                else:
                    Z_q = np.sum(mu_safe ** q)
                    if Z_q < 1e-300:
                        Z_q = 1e-300
                    mu_tilde = mu_safe ** q / Z_q

                log_Zq[s]    = np.log(Z_q)
                num_alpha[s] = np.sum(mu_tilde * np.log(mu_safe + 1e-300))
                num_f[s]     = np.sum(mu_tilde * np.log(mu_tilde + 1e-300))

            # Linear regression against log(ε)
            # α(q) = d[Σ μ̃ log μ] / d[log ε]
            # f(q) = d[Σ μ̃ log μ̃] / d[log ε]
            # τ(q) = d[log Z(q,ε)] / d[log ε]
            alpha_arr[j] = np.polyfit(log_eps, num_alpha, 1)[0]
            f_arr[j]     = np.polyfit(log_eps, num_f,     1)[0]
            tau_q        = np.polyfit(log_eps, log_Zq,    1)[0]
            Dq_arr[j]    = tau_q / (q - 1) if abs(q - 1) > 0.05 else alpha_arr[j]

        return alpha_arr, f_arr, Dq_arr, eps_abs

    def print_spectrum(self, q_values, alpha, f, Dq):
        """Print spectrum with physical constraint checks."""
        print(f"\n  {'q':>6}  {'α(q)':>8}  {'f(α)':>8}  {'D_q':>8}  Constraints")
        print(f"  {'-'*56}")

        alpha_decreasing = all(alpha[i] >= alpha[i+1] - 0.01
                               for i in range(len(alpha)-1))
        for j, q in enumerate(q_values):
            f_ok  = 0.0 <= f[j] <= self.d + 0.1
            flags = ""
            if not f_ok:
                flags += " ✗f<0" if f[j] < 0 else " ✗f>d"
            print(f"  {q:>6.2f}  {alpha[j]:>8.4f}  {f[j]:>8.4f}  "
                  f"{Dq[j]:>8.4f}  {'✓' if f_ok else flags}")

        print(f"\n  α(q) decreasing: {'✓' if alpha_decreasing else '✗'}")
        print(f"  Δα = {alpha.max() - alpha.min():.4f}")
        print(f"  D₀ (q→0) = {f[np.abs(q_values).argmin()]:.4f}")
        print(f"  D₁ (q=1) = {alpha[np.abs(q_values - 1).argmin()]:.4f}  "
              f"(= information dimension)")


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION SUITE
# ═══════════════════════════════════════════════════════════════════════════

def run_validation():
    q_range = np.linspace(-2.0, 2.0, 17)

    print("=" * 65)
    print("CHHABRA-JENSEN VALIDATION SUITE")
    print("Three ground-truth tests — all must pass.")
    print("=" * 65)

    results = {}

    # ── Test 1: Uniform square ───────────────────────────────────────────
    print("\nTEST 1: Uniform random square")
    print("  Target: D₀ = 2.000,  Δα ≈ 0  (monofractal)")
    pts1 = dataset_uniform_square(n=5000)
    est1 = ChabbraJensenEstimator(pts1, n_scales=15,
                                   eps_min=0.03, eps_max=0.30)
    a1, f1, Dq1, _ = est1.compute_spectrum(q_range)
    D0_1 = f1[np.abs(q_range).argmin()]
    da1  = a1.max() - a1.min()
    est1.print_spectrum(q_range, a1, f1, Dq1)
    ok1  = abs(D0_1 - 2.0) < 0.15 and da1 < 0.30
    print(f"\n  → {'✅ PASSED' if ok1 else '❌ FAILED'}  "
          f"(D₀={D0_1:.3f}, Δα={da1:.3f})")
    results['uniform'] = ok1

    # ── Test 2: Cantor set ───────────────────────────────────────────────
    print("\nTEST 2: Middle-third Cantor set")
    target_cantor = np.log(2) / np.log(3)
    print(f"  Target: D₀ = {target_cantor:.4f},  Δα ≈ 0  (monofractal)")
    pts2 = dataset_cantor_set(generations=10)
    # Cantor set is 1D embedded in 2D — use 1D projection
    pts2_1d = pts2[:, :1]   # just x-coordinate
    est2 = ChabbraJensenEstimator(pts2_1d, n_scales=15,
                                   eps_min=0.002, eps_max=0.15)
    a2, f2, Dq2, _ = est2.compute_spectrum(q_range)
    D0_2 = f2[np.abs(q_range).argmin()]
    da2  = a2.max() - a2.min()
    est2.print_spectrum(q_range, a2, f2, Dq2)
    ok2  = abs(D0_2 - target_cantor) < 0.10 and da2 < 0.20
    print(f"\n  → {'✅ PASSED' if ok2 else '❌ FAILED'}  "
          f"(D₀={D0_2:.3f}, target={target_cantor:.3f}, Δα={da2:.3f})")
    results['cantor'] = ok2

    # ── Test 3: Binomial cascade ─────────────────────────────────────────
    p = 0.3
    print(f"\nTEST 3: Binomial cascade  (p={p}, 1-p={1-p:.1f})")
    da_theory = np.log((1-p)/p) / np.log(2)
    print(f"  Target: D₀ = 1.000,  Δα = {da_theory:.3f}  (multifractal)")
    centres3, weights3 = dataset_binomial_cascade(depth=12, p=p)
    pts3 = centres3.reshape(-1, 1)
    est3 = ChabbraJensenEstimator(pts3, weights=weights3, n_scales=15,
                                   eps_min=0.005, eps_max=0.20)
    a3, f3, Dq3, _ = est3.compute_spectrum(q_range)
    D0_3 = f3[np.abs(q_range).argmin()]
    da3  = a3.max() - a3.min()
    est3.print_spectrum(q_range, a3, f3, Dq3)
    ok3  = abs(D0_3 - 1.0) < 0.10 and da3 > 0.5
    print(f"\n  → {'✅ PASSED' if ok3 else '❌ FAILED'}  "
          f"(D₀={D0_3:.3f}, Δα={da3:.3f}, theory Δα={da_theory:.3f})")
    results['cascade'] = ok3

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    all_pass = all(results.values())
    for name, ok in results.items():
        print(f"  {name:<12}  {'✅ PASS' if ok else '❌ FAIL'}")
    print()
    if all_pass:
        print("  ALL TESTS PASSED — estimator is ready for Rauzy fractal data.")
    else:
        print("  SOME TESTS FAILED — do not apply to Rauzy data yet.")
        print("  Tune eps_min / eps_max and rerun until all pass.")
    print("=" * 65)

    return all_pass, (a1, f1), (a2, f2), (a3, f3)


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    all_pass, r1, r2, r3 = run_validation()

    if all_pass:
        print("\nNEXT STEP: apply to Rauzy fractal data")
        print("  from minerva_v8_0_validated_cj import ChabbraJensenEstimator")
        print("  import numpy as np")
        print("  pts = np.load('data/rauzy_points.npy')")
        print("  est = ChabbraJensenEstimator(pts, n_scales=15,")
        print("                               eps_min=0.01, eps_max=0.25)")
        print("  alpha, f, Dq, _ = est.compute_spectrum(np.linspace(-2,2,17))")
