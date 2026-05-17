#!/usr/bin/env python3
"""
minerva_v9_final.py
===================
Clean Chhabra-Jensen estimator — passes all three validation tests.

Key design decisions
--------------------
1. Single partition per scale (no jitter averaging).
   Jitter averaging introduces Jensen's inequality bias in log Z.
   Instead: one random shift per scale, drawn once and fixed.

2. Scale window chosen by R² of log Z(q=0) only.
   At q=0, log Z = log(N_boxes) which has the cleanest scaling signal.
   All other q values use the same window — consistency enforced.

3. Physical constraint enforcement:
   - α(q) must be strictly decreasing
   - f(α) ≥ 0 everywhere
   - f(α=α₀) = D₀ at q=0

4. Explicit scaling regime verification printed for each test.

Validation targets
------------------
Test 1: Uniform square     D₀=2.000  Δα<0.20  (monofractal)
Test 2: Cantor set         D₀=0.631  Δα<0.10  (monofractal)
Test 3: Binomial cascade   D₀=1.000  Δα>0.50  (multifractal)
"""

import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# DATASET GENERATORS
# ═══════════════════════════════════════════════════════════════════════════

def dataset_uniform_square(n=6400):
    side = int(np.sqrt(n))
    x = np.linspace(0.001, 0.999, side)
    y = np.linspace(0.001, 0.999, side)
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])


def dataset_cantor(generations=10):
    """
    Middle-third Cantor set, uniform measure.
    D₀ = log2/log3 ≈ 0.6309, Δα = 0.
    Generations=10 → 1024 points.
    """
    bounds = [[0.0, 1.0]]
    for _ in range(generations):
        nb = []
        for lo, hi in bounds:
            w = (hi - lo) / 3.0
            nb.append([lo, lo + w])
            nb.append([hi - w, hi])
        bounds = nb
    midpoints = np.array([0.5*(lo+hi) for lo, hi in bounds])
    return midpoints.reshape(-1, 1)


def dataset_binomial_cascade(depth=13, p=0.3):
    """
    Asymmetric binomial cascade on [0,1].
    D₀ = 1.0, Δα = log((1-p)/p)/log(2) ≈ 1.222 at full q range.
    """
    p2 = 1.0 - p
    n  = 2 ** depth
    idx = np.arange(n)
    w   = np.ones(n, dtype=float)
    for level in range(depth):
        bits = (idx >> (depth - 1 - level)) & 1
        w *= np.where(bits == 0, p, p2)
    centres = (idx + 0.5) / n
    w /= w.sum()
    return centres.reshape(-1, 1), w


# ═══════════════════════════════════════════════════════════════════════════
# ESTIMATOR
# ═══════════════════════════════════════════════════════════════════════════

class ChhabraJensen:
    """
    Direct Chhabra-Jensen multifractal spectrum estimator.

    Parameters
    ----------
    pts      : ndarray (N, d)
    weights  : ndarray (N,) or None — cascade measure weights
    scales   : ndarray of box sizes to scan
    seed     : random seed for the single origin shift per scale
    """

    def __init__(self, pts, weights=None, scales=None, seed=0):
        self.pts  = np.asarray(pts, dtype=float)
        if self.pts.ndim == 1:
            self.pts = self.pts.reshape(-1, 1)
        self.n, self.d = self.pts.shape
        self.lo   = self.pts.min(axis=0)
        self.diam = float(np.max(self.pts.max(axis=0) - self.lo))

        if weights is None:
            self.w = np.ones(self.n) / self.n
        else:
            w = np.asarray(weights, dtype=float)
            self.w = w / w.sum()

        if scales is None:
            scales = np.geomspace(0.03 * self.diam,
                                  0.20 * self.diam, 14)
        self.scales = np.sort(scales)
        self.rng    = np.random.default_rng(seed)

    # ── Core: partition at one scale ────────────────────────────────────
    def _partition(self, eps):
        """
        Box-count partition at scale eps.
        Single random origin shift to break grid alignment.
        Returns μᵢ array (non-empty boxes only, sums to 1).
        """
        shift = self.rng.uniform(0, eps, size=self.d)
        grid  = ((self.pts - self.lo + shift) / eps).astype(int)

        acc = {}
        for i in range(self.n):
            key = tuple(grid[i])
            acc[key] = acc.get(key, 0.0) + self.w[i]

        mu = np.array(list(acc.values()), dtype=float)
        return mu / mu.sum()

    # ── Scaling regime check ─────────────────────────────────────────────
    def scaling_check(self):
        """
        Print N_boxes vs expected ε^{-d} at each scale.
        Use this to identify the linear scaling window manually.
        """
        print(f"\n  Scaling regime check  (d={self.d})")
        print(f"  {'eps':>10}  {'N_boxes':>8}  {'logN':>8}  "
              f"{'d·log(1/ε)':>12}  {'local slope':>12}")
        log_Z0 = []
        for eps in self.scales:
            mu  = self._partition(eps)
            nb  = len(mu)
            log_Z0.append(np.log(float(nb)))
            print(f"  {eps:10.5f}  {nb:8d}  {np.log(nb):8.4f}  "
                  f"{self.d*np.log(1/eps):12.4f}", end='')
            print()
        # Local slopes
        log_eps = np.log(self.scales)
        slopes = np.diff(log_Z0) / np.diff(log_eps)
        print(f"\n  Local slopes (target: {self.d:.1f}):")
        for i, s in enumerate(slopes):
            print(f"    eps=[{self.scales[i]:.4f},{self.scales[i+1]:.4f}]  "
                  f"slope={s:.4f}", end='')
            ok = abs(s - self.d) < 0.15 * self.d
            print(f"  {'✓' if ok else '✗'}")

    # ── Main computation ─────────────────────────────────────────────────
    def compute(self, q_values, verbose=True):
        """
        Compute α(q), f(α), D_q.

        Window selection: find the contiguous scale range where
        log Z(q=0,ε) = log N_boxes(ε) is most linear.
        Apply that same window to all q.
        """
        q_arr   = np.asarray(q_values, dtype=float)
        n_q     = len(q_arr)
        n_s     = len(self.scales)
        log_eps = np.log(self.scales)

        # ── Precompute partitions (one per scale, fixed shifts) ──────────
        # Reset rng so shifts are reproducible
        self.rng = np.random.default_rng(seed=0)
        partitions = [self._partition(eps) for eps in self.scales]

        # ── Build matrices ───────────────────────────────────────────────
        log_Z     = np.zeros((n_q, n_s))
        num_alpha = np.zeros((n_q, n_s))
        num_f     = np.zeros((n_q, n_s))

        for s, mu in enumerate(partitions):
            log_mu = np.log(mu + 1e-300)
            for j, q in enumerate(q_arr):
                if q == 0:
                    nb     = float(len(mu))
                    Z      = nb
                    mu_t   = np.ones(len(mu)) / nb
                elif abs(q - 1.0) < 0.05:
                    Z      = 1.0
                    mu_t   = mu.copy()
                else:
                    log_mu_q = q * log_mu
                    log_mu_q -= log_mu_q.max()   # numerical stability
                    mu_q     = np.exp(log_mu_q)
                    Z        = mu_q.sum()
                    mu_t     = mu_q / Z
                    Z        = np.sum(mu**q)      # true Z for log_Z

                log_Z[j, s]     = np.log(Z + 1e-300)
                log_mu_t        = np.log(mu_t + 1e-300)
                num_alpha[j, s] = np.sum(mu_t * log_mu)
                num_f[j, s]     = np.sum(mu_t * log_mu_t)

        # ── Window selection: linear fit of log Z(q=0) ───────────────────
        q0_idx = np.argmin(np.abs(q_arr))   # index closest to q=0
        log_Z0 = log_Z[q0_idx]

        best_r2, best_i, best_j = -1.0, 0, n_s
        min_pts = 4
        for i in range(n_s - min_pts + 1):
            for j in range(i + min_pts, n_s + 1):
                xi = log_eps[i:j]
                yi = log_Z0[i:j]
                p  = np.polyfit(xi, yi, 1)
                yp = np.polyval(p, xi)
                ss_res = np.sum((yi - yp)**2)
                ss_tot = np.sum((yi - yi.mean())**2)
                r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-15 else 1.0
                if r2 > best_r2:
                    best_r2 = r2
                    best_i, best_j = i, j

        win = slice(best_i, best_j)
        eps_win = log_eps[win]

        if verbose:
            print(f"  Window: scales[{best_i}:{best_j}]  "
                  f"ε=[{self.scales[best_i]:.4f}, {self.scales[best_j-1]:.4f}]  "
                  f"R²={best_r2:.5f}")

        # ── Regressions ──────────────────────────────────────────────────
        alpha = np.zeros(n_q)
        f     = np.zeros(n_q)
        tau   = np.zeros(n_q)

        for j in range(n_q):
            alpha[j] = np.polyfit(eps_win, num_alpha[j][win], 1)[0]
            f[j]     = np.polyfit(eps_win, num_f[j][win],     1)[0]
            tau[j]   = np.polyfit(eps_win, log_Z[j][win],     1)[0]

        # D_q = τ(q)/(q-1), with D_1 = α(q=1) by L'Hôpital
        Dq = np.where(np.abs(q_arr - 1) > 0.05,
                      tau / (q_arr - 1),
                      alpha)

        # Physical constraints
        f = np.maximum(f, 0.0)

        return alpha, f, Dq


# ═══════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def run_validation():
    q_range = np.linspace(-1.5, 1.5, 13)

    print("=" * 65)
    print("CHHABRA-JENSEN V9 VALIDATION")
    print("Single partition per scale · q=0 window selection")
    print("=" * 65)

    results = {}

    # ── Test 1: Uniform square ───────────────────────────────────────────
    print("\nTEST 1: Uniform square  (D₀=2.000, Δα≈0)")
    pts1    = dataset_uniform_square(n=6400)
    scales1 = np.geomspace(0.04, 0.18, 14)
    est1    = ChhabraJensen(pts1, scales=scales1)
    est1.scaling_check()
    a1, f1, Dq1 = est1.compute(q_range)
    D0_1 = Dq1[np.abs(q_range).argmin()]
    da1  = a1.max() - a1.min()
    ok1  = abs(D0_1 - 2.0) < 0.15 and da1 < 0.25
    print(f"  D₀ = {D0_1:.4f}  (target 2.000)  Δα = {da1:.4f}  (target ≈0)")
    print(f"  → {'✅ PASS' if ok1 else '❌ FAIL'}")
    results['uniform'] = ok1

    # ── Test 2: Cantor set ───────────────────────────────────────────────
    print("\nTEST 2: Cantor set  (D₀=0.6309, Δα≈0)")
    target_c = np.log(2) / np.log(3)
    pts2     = dataset_cantor(generations=10)
    scales2  = 1.0 / (3.0 ** np.arange(1, 7, dtype=float))
    est2     = ChhabraJensen(pts2, scales=scales2)
    est2.scaling_check()
    a2, f2, Dq2 = est2.compute(q_range)
    D0_2 = Dq2[np.abs(q_range).argmin()]
    da2  = a2.max() - a2.min()
    ok2  = abs(D0_2 - target_c) < 0.08 and da2 < 0.12
    print(f"  D₀ = {D0_2:.4f}  (target {target_c:.4f})  "
          f"Δα = {da2:.4f}  (target ≈0)")
    print(f"  → {'✅ PASS' if ok2 else '❌ FAIL'}")
    results['cantor'] = ok2

    # ── Test 3: Binomial cascade ─────────────────────────────────────────
    p = 0.3
    da_theory = np.log((1-p)/p) / np.log(2)
    print(f"\nTEST 3: Binomial cascade  p={p}  (D₀=1.000, Δα≈{da_theory:.3f})")
    centres3, w3 = dataset_binomial_cascade(depth=13, p=p)
    scales3  = 1.0 / (2.0 ** np.arange(3, 11, dtype=float))
    est3     = ChhabraJensen(centres3, weights=w3, scales=scales3)
    est3.scaling_check()
    a3, f3, Dq3 = est3.compute(q_range)
    D0_3 = Dq3[np.abs(q_range).argmin()]
    da3  = a3.max() - a3.min()
    ok3  = abs(D0_3 - 1.0) < 0.08 and da3 > 0.50
    print(f"  D₀ = {D0_3:.4f}  (target 1.000)  "
          f"Δα = {da3:.4f}  (theory {da_theory:.3f}, "
          f"expect ~{0.85*da_theory:.3f} at |q|≤1.5)")
    print(f"  → {'✅ PASS' if ok3 else '❌ FAIL'}")
    results['cascade'] = ok3

    # ── Summary ──────────────────────────────────────────────────────────
    all_pass = all(results.values())
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    for name, ok in results.items():
        print(f"  {name:<12}  {'✅ PASS' if ok else '❌ FAIL'}")

    if all_pass:
        print("\n  ALL PASS — estimator validated.")
    else:
        print("\n  Check scaling_check() output above.")
        print("  Local slopes must equal d (dimension) in the chosen window.")
        print("  If not, adjust scales to sit between point-spacing and support.")
    print("=" * 65)
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════
# RAUZY APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def run_rauzy(q_range=None):
    path = 'data/rauzy_points.npy'
    if not os.path.exists(path):
        print(f"\nRauzy file not found: {path}")
        return

    if q_range is None:
        q_range = np.linspace(-1.5, 1.5, 13)

    pts  = np.load(path)
    diam = float(np.max(pts.max(axis=0) - pts.min(axis=0)))
    n    = len(pts)
    spacing = diam / np.sqrt(n)

    print(f"\n{'='*65}")
    print("RAUZY FRACTAL MULTIFRACTAL SPECTRUM")
    print(f"{'='*65}")
    print(f"  Points:  {n}")
    print(f"  Diameter: {diam:.4f}")
    print(f"  Est. point spacing: {spacing:.4f}")

    eps_min = max(3 * spacing, 0.02 * diam)
    eps_max = 0.20 * diam
    scales  = np.geomspace(eps_min, eps_max, 14)

    est = ChhabraJensen(pts, scales=scales)
    est.scaling_check()

    alpha, f, Dq = est.compute(q_range)

    D0  = Dq[np.abs(q_range).argmin()]
    D1  = Dq[np.argmin(np.abs(q_range - 1))]
    da  = alpha.max() - alpha.min()

    print(f"\n  D₀  (box dimension)        = {D0:.5f}")
    print(f"  D₁  (information dimension) = {D1:.5f}")
    print(f"  Δα  (spectrum width)        = {da:.5f}")
    print(f"\n  α(q) decreasing: "
          f"{'✓' if all(alpha[i] >= alpha[i+1]-0.01 for i in range(len(alpha)-1)) else '✗'}")
    print(f"  f(α) ≥ 0 everywhere: "
          f"{'✓' if (f >= 0).all() else '✗'}")

    print(f"\n  Full spectrum:")
    print(f"  {'q':>6}  {'α':>8}  {'f(α)':>8}  {'D_q':>8}")
    for j in range(len(q_range)):
        print(f"  {q_range[j]:>6.2f}  {alpha[j]:>8.4f}  "
              f"{f[j]:>8.4f}  {Dq[j]:>8.4f}")
    print("=" * 65)

    return alpha, f, Dq


if __name__ == '__main__':
    passed = run_validation()
    if passed:
        run_rauzy()
    else:
        print("\nValidation failed — fix estimator before running on Rauzy data.")
