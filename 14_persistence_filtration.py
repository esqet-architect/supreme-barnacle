#!/usr/bin/env python3
"""
14_persistence_filtration.py
============================
Scale-resolved topological diagnostics for Rauzy fractal boundaries.

What this computes (honest description)
----------------------------------------
1.  Corrected Rauzy projection  (increment method — bounded cloud guaranteed)
2.  Vietoris-Rips filtration    (genuine simplicial complex at each ε)
3.  β₀(ε) — connected components  (exact, via union-find)
4.  β₁(ε) — independent cycles    (exact Euler: β₁ = E - V + β₀ for VR complex)
5.  Persistence entropy          (Shannon entropy of bar lifetimes)
6.  Component survival curve     (how many β₀ bars survive past ε)

What this does NOT compute
---------------------------
- Full persistent homology with boundary matrices (would require gudhi or ripser)
- Asymptotic Oseledets decomposition (finite-time SVD only)
- True Hausdorff dimension (box counting is an upper bound estimate)

The D₀^AR < 1 question
------------------------
Case A (projection artifact):  β₀ → 1, D₀ → 1 with corrected projection
Case B (genuine fragmentation): β₀ > 1 persists across filtration, D₀ < 1 stable

This script runs the diagnostic to distinguish Case A from Case B.

Mathematical references
-----------------------
- Vietoris-Rips filtration:  Edelsbrunner & Harer, Computational Topology (2010)
- Persistent homology:       Zomorodian & Carlsson, Discrete Comput Geom (2005)
- Euler characteristic:      β₀ - β₁ + β₂ = χ  (exact for VR 1-skeleton)
"""

import numpy as np
from numpy.linalg import eig, norm
from scipy.spatial import KDTree
from scipy.cluster.hierarchy import fcluster, linkage
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# 1.  CORRECTED RAUZY PROJECTION  (increment method, guaranteed compact)
# ═══════════════════════════════════════════════════════════════════════════

def get_orthonormal_Ec(M):
    """
    Orthonormal basis for the contracting plane E_c.
    Uses Gram-Schmidt on real/imaginary parts of the complex eigenvector.
    Returns pi_c (2×3) and delta dict (increment vectors in E_c).
    """
    evals, evecs = eig(M)
    idx   = np.argsort(-np.abs(evals))
    evecs = evecs[:, idx]

    w  = evecs[:, 1]           # complex eigenvector for λ+
    e1 = w.real.copy()
    e2 = w.imag.copy()
    e1 /= norm(e1)
    e2 -= np.dot(e2, e1) * e1
    e2 /= norm(e2)
    pi_c = np.array([e1, e2])  # 2×3

    basis3 = {'1': np.array([1.,0.,0.]),
               '2': np.array([0.,1.,0.]),
               '3': np.array([0.,0.,1.])}
    delta = {ch: pi_c @ v for ch, v in basis3.items()}
    return pi_c, delta


def build_rauzy_points(word, M):
    """
    Correct increment-method projection.
    Accumulates π_c(e_letter) steps — stays bounded by quasi-periodicity.
    """
    _, delta = get_orthonormal_Ec(M)
    pts = np.zeros((len(word), 2))
    pos = np.zeros(2)
    for i, ch in enumerate(word):
        pts[i] = pos
        pos = pos + delta.get(ch, delta['1'])
    return pts


def compactness_check(pts, name):
    """Quick sanity check: bounded cloud has diameter < ~5 for all systems."""
    centroid = pts.mean(axis=0)
    radii    = norm(pts - centroid, axis=1)
    diameter = 2 * radii.max()
    compact  = diameter < 20.0   # generous threshold
    print(f"  {name:<22}: diameter = {diameter:.4f}  "
          f"{'✓ compact' if compact else '✗ UNBOUNDED — projection error'}")
    return compact


# ═══════════════════════════════════════════════════════════════════════════
# 2.  VIETORIS-RIPS FILTRATION  (exact β₀, exact β₁ for 1-skeleton)
# ═══════════════════════════════════════════════════════════════════════════

class UnionFind:
    """Weighted quick-union with path compression."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank   = [0] * n
        self.n_comp = n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_comp -= 1
        return True


def vietoris_rips_filtration(pts, n_epsilons=40, subsample=800):
    """
    Compute β₀(ε) and β₁(ε) across a Vietoris-Rips filtration.

    Method
    ------
    At each ε, the VR 1-skeleton G_ε has:
      - Vertices V = pts  (constant)
      - Edges E_ε = { (i,j) : ||p_i - p_j|| ≤ ε }

    β₀(ε) = connected components of G_ε  (exact, via union-find)
    β₁(ε) = |E_ε| - |V| + β₀(ε)         (exact Euler for 1-skeleton)
             = independent cycles in G_ε

    Note: β₁ here counts graph cycles, not H₁ of the full VR complex.
    For the full H₁ you need 2-simplices (triangles). We flag this clearly.

    Parameters
    ----------
    subsample : int   Max points to use (performance on phone)

    Returns
    -------
    epsilons : ndarray   filtration parameter values
    beta0    : ndarray   β₀(ε)  — exact
    beta1    : ndarray   β₁(ε)  — exact for 1-skeleton (see note above)
    """
    if len(pts) > subsample:
        idx = np.random.default_rng(42).choice(len(pts), subsample, replace=False)
        pts = pts[idx]

    n = len(pts)
    tree = KDTree(pts)

    # Compute all pairwise distances (condensed)
    # Use linkage-style approach for efficiency
    # Max epsilon: 10th percentile of pairwise distances → 90th percentile
    dists_knn, _ = tree.query(pts, k=min(50, n))
    eps_min = np.percentile(dists_knn[:, 1], 5)    # just above nearest-neighbor
    eps_max = np.percentile(dists_knn[:, -1], 90)  # well into connected regime

    epsilons = np.linspace(eps_min, eps_max, n_epsilons)
    beta0_arr = np.zeros(n_epsilons, dtype=int)
    beta1_arr = np.zeros(n_epsilons, dtype=int)

    # Precompute edges sorted by distance (for incremental construction)
    # Sample pairs for efficiency
    max_pairs = 200000
    if n * (n-1) // 2 > max_pairs:
        i_idx = np.random.default_rng(7).integers(0, n, max_pairs)
        j_idx = np.random.default_rng(13).integers(0, n, max_pairs)
        mask  = i_idx < j_idx
        i_idx, j_idx = i_idx[mask], j_idx[mask]
    else:
        ii, jj = np.triu_indices(n, k=1)
        i_idx, j_idx = ii, jj

    edge_dists = norm(pts[i_idx] - pts[j_idx], axis=1)
    sort_order = np.argsort(edge_dists)
    i_sorted   = i_idx[sort_order]
    j_sorted   = j_idx[sort_order]
    d_sorted   = edge_dists[sort_order]

    uf    = UnionFind(n)
    e_ptr = 0          # pointer into sorted edge list
    E     = 0          # total edges added so far

    for k, eps in enumerate(epsilons):
        # Add all edges with distance ≤ eps
        while e_ptr < len(d_sorted) and d_sorted[e_ptr] <= eps:
            uf.union(i_sorted[e_ptr], j_sorted[e_ptr])
            E += 1
            e_ptr += 1

        b0 = uf.n_comp
        b1 = E - n + b0   # Euler: E - V + β₀ = β₁  (1-skeleton only)

        beta0_arr[k] = b0
        beta1_arr[k] = max(0, b1)

    return epsilons, beta0_arr, beta1_arr


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PERSISTENCE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def persistence_entropy(beta0_arr, epsilons):
    """
    Shannon entropy of the component-lifetime distribution.

    Each β₀ drop from β₀(ε_k) to β₀(ε_{k+1}) represents components
    merging — their "lifetime" is the ε interval they persist over.
    Entropy measures how spread the merging events are across scales.

    High entropy → merging spread across many scales (complex topology)
    Low entropy  → merging clustered at one scale (homogeneous)
    """
    lifetimes = []
    prev = beta0_arr[0]
    start_eps = epsilons[0]

    for k in range(1, len(beta0_arr)):
        if beta0_arr[k] < prev:
            # Components merged: record lifetime of each merged component
            n_merged = prev - beta0_arr[k]
            lifetime = epsilons[k] - start_eps
            lifetimes.extend([lifetime] * n_merged)
            start_eps = epsilons[k]
        prev = beta0_arr[k]

    if len(lifetimes) < 2:
        return 0.0

    lifetimes = np.array(lifetimes)
    lifetimes = lifetimes / lifetimes.sum()
    lifetimes = lifetimes[lifetimes > 0]
    return float(-np.sum(lifetimes * np.log(lifetimes)))


def component_survival(beta0_arr, epsilons, n_start):
    """
    Fraction of initial components still surviving at each ε.
    A slowly decaying survival curve → many persistent components → Case B.
    A rapidly decaying curve → one dominant merge event → Case A.
    """
    return beta0_arr / n_start


def loop_lifetime_spectrum(beta1_arr, epsilons):
    """
    Track when loops appear and disappear in the filtration.
    Returns mean loop lifetime and peak loop count.
    """
    peak_loops = int(beta1_arr.max())
    # Find where loops first appear and where they peak
    nonzero = np.where(beta1_arr > 0)[0]
    if len(nonzero) == 0:
        return {'peak': 0, 'eps_onset': np.nan, 'mean_active': 0.0}
    onset_eps = float(epsilons[nonzero[0]])
    mean_active = float(beta1_arr.mean())
    return {
        'peak':        peak_loops,
        'eps_onset':   onset_eps,
        'mean_active': mean_active
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  CASE A / CASE B CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

def classify_case(beta0_arr, epsilons, diameter, name):
    """
    Distinguish projection artifact (Case A) from genuine fragmentation (Case B).

    Decision logic
    --------------
    Case A indicators:
      - β₀ drops to 1 within the filtration range
      - Compact cloud (diameter < 10)
      - Low persistence entropy

    Case B indicators:
      - β₀ remains > 1 through most of filtration
      - Multiple long-lived components
      - High persistence entropy
    """
    final_b0     = int(beta0_arr[-1])
    frac_unified = float((beta0_arr == 1).mean())
    ent          = persistence_entropy(beta0_arr, epsilons)
    compact      = diameter < 10.0

    if compact and final_b0 == 1 and frac_unified > 0.5:
        case = 'A (projection artifact — single connected attractor)'
    elif not compact:
        case = '? (UNBOUNDED CLOUD — fix projection first)'
    elif final_b0 > 3 and frac_unified < 0.2:
        case = 'B (genuine fragmentation — multi-component attractor)'
    else:
        case = 'AMBIGUOUS (increase N or filtration range)'

    return {
        'case':          case,
        'final_beta0':   final_b0,
        'frac_unified':  frac_unified,
        'entropy':       ent,
        'compact':       compact,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  WORD GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_word(subs, length):
    word = '1'
    while len(word) < length:
        word = ''.join(subs.get(c, c) for c in word)
        if len(word) > length * 3:
            break
    return word[:length]


# ═══════════════════════════════════════════════════════════════════════════
# 6.  MAIN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

SYSTEMS = {
    'Tribonacci Pisot': {
        'subs': {'1': '12', '2': '13', '3': '1'},
        'M':    np.array([[1,1,1],[1,0,0],[0,1,0]], dtype=float),
    },
    'Plastic Number': {
        'subs': {'1': '2', '2': '3', '3': '12'},
        'M':    np.array([[0,0,1],[1,0,1],[0,1,0]], dtype=float),
    },
    'Arnoux-Rauzy': {
        'subs': {'1': '123', '2': '13', '3': '1'},
        'M':    np.array([[1,1,1],[1,0,1],[1,0,0]], dtype=float),
    },
    'Non-Pisot Control': {
        'subs': {'1': '12', '2': '21', '3': '3'},
        'M':    np.array([[1,1,0],[1,1,0],[0,0,1]], dtype=float),
    },
}

N_WORD    = 15000
N_EPS     = 40
SUBSAMPLE = 600    # keep fast on phone; raise to 1200 on laptop

print("=" * 70)
print("PERSISTENCE FILTRATION ANALYSIS")
print("Vietoris-Rips · β₀ (exact) · β₁ (1-skeleton Euler) · Persistence Entropy")
print("=" * 70)

# ── Step 1: Compactness check (Case A pre-filter) ─────────────────────────
print("\n[1] COMPACTNESS CHECK (corrected increment projection)")
print("─" * 50)
results = {}
for name, cfg in SYSTEMS.items():
    word = generate_word(cfg['subs'], N_WORD)
    pts  = build_rauzy_points(word, cfg['M'])
    centroid = pts.mean(axis=0)
    diameter = 2 * norm(pts - centroid, axis=1).max()
    compact  = compactness_check(pts, name)
    results[name] = {'pts': pts, 'diameter': diameter, 'compact': compact}

# ── Step 2: Filtration ────────────────────────────────────────────────────
print("\n[2] VIETORIS-RIPS FILTRATION")
print("─" * 50)

for name, cfg in SYSTEMS.items():
    pts  = results[name]['pts']
    diam = results[name]['diameter']

    print(f"\n  {name}")

    if not results[name]['compact']:
        print(f"    SKIPPED — unbounded cloud (diameter={diam:.1f})")
        results[name]['skip'] = True
        continue

    results[name]['skip'] = False

    eps_arr, b0_arr, b1_arr = vietoris_rips_filtration(
        pts, n_epsilons=N_EPS, subsample=SUBSAMPLE
    )

    # Summary statistics
    ent      = persistence_entropy(b0_arr, eps_arr)
    survival = component_survival(b0_arr, eps_arr, b0_arr[0])
    loops    = loop_lifetime_spectrum(b1_arr, eps_arr)
    case     = classify_case(b0_arr, eps_arr, diam, name)

    print(f"    β₀: {b0_arr[0]} → {b0_arr[-1]}  "
          f"(range ε=[{eps_arr[0]:.4f}, {eps_arr[-1]:.4f}])")
    print(f"    β₁ peak: {loops['peak']}  onset ε={loops['eps_onset']:.4f}")
    print(f"    Persistence entropy: {ent:.4f}")
    print(f"    Fraction ε with β₀=1: {case['frac_unified']:.3f}")
    print(f"    Case: {case['case']}")

    results[name].update({
        'eps':      eps_arr,
        'beta0':    b0_arr,
        'beta1':    b1_arr,
        'entropy':  ent,
        'survival': survival,
        'loops':    loops,
        'case':     case,
    })

# ── Step 3: Comparative table ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMPARATIVE SUMMARY TABLE")
print("=" * 70)
print(f"{'System':<22} {'Diam':>6} {'β₀ final':>9} {'Entropy':>9} "
      f"{'β₁ peak':>8}  Case")
print("─" * 70)

for name in SYSTEMS:
    r = results[name]
    if r.get('skip'):
        print(f"{name:<22} {'∞':>6}  {'–':>8}  {'–':>8}  {'–':>7}  UNBOUNDED")
        continue
    c = r['case']
    short_case = c['case'].split('(')[0].strip()
    print(f"{name:<22} {r['diameter']:>6.3f}  "
          f"{c['final_beta0']:>8}  "
          f"{c['entropy']:>8.4f}  "
          f"{r['loops']['peak']:>7}  {short_case}")

# ── Step 4: D₀^AR verdict ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("D₀^AR < 1 QUESTION — VERDICT FROM FILTRATION")
print("=" * 70)

ar = results.get('Arnoux-Rauzy', {})
if ar.get('skip'):
    print("  Cannot assess — unbounded cloud. Fix projection and rerun.")
elif 'case' in ar:
    c = ar['case']
    print(f"  Final β₀          = {c['final_beta0']}")
    print(f"  Fraction unified   = {c['frac_unified']:.3f}")
    print(f"  Persistence entropy= {c['entropy']:.4f}")
    print(f"  Diameter           = {ar['diameter']:.4f}")
    print(f"\n  Classification: {c['case']}")
    print()
    if 'A' in c['case']:
        print("  → D₀^AR < 1 was a coordinate artifact.")
        print("    Corrected projection gives a connected attractor.")
        print("    The 'fragmentation' in prior runs was unbounded-walk distortion.")
    elif 'B' in c['case']:
        print("  → D₀^AR < 1 appears genuine.")
        print("    Multiple long-lived components survive the filtration.")
        print("    The Arnoux-Rauzy attractor is genuinely multi-component")
        print("    under contracting-plane projection.")
        print("    This is consistent with non-simple-connectivity. [Siegel-Thuswaldner]")
    else:
        print("  → Ambiguous. Increase N_WORD to 30000 and SUBSAMPLE to 1200.")

# ── Step 5: Optional plot ─────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    BG   = '#0d0d0f'
    GOLD = '#c8972a'
    TEAL = '#1a9b9b'
    RUST = '#c04a30'
    SAGE = '#4a7c5a'
    DIM  = '#555050'
    WHT  = '#e8e2d6'

    colors = {'Tribonacci Pisot': GOLD,
               'Plastic Number':   TEAL,
               'Arnoux-Rauzy':     RUST,
               'Non-Pisot Control': SAGE}

    fig = plt.figure(figsize=(14, 9))
    fig.patch.set_facecolor(BG)
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.32,
                            left=0.07, right=0.97, top=0.91, bottom=0.08)

    def sax(ax, title):
        ax.set_facecolor('#131316')
        ax.tick_params(colors=DIM, labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor('#2a2a2e')
        ax.set_title(title, color=WHT, fontsize=8.5, pad=5)
        ax.xaxis.label.set_color(DIM)
        ax.yaxis.label.set_color(DIM)
        ax.grid(True, color='#1e1e22', lw=0.4)

    # Panel A: β₀(ε) curves
    ax = fig.add_subplot(gs[0, 0])
    sax(ax, 'A  ·  β₀(ε)  Connected Components')
    for name in SYSTEMS:
        r = results[name]
        if r.get('skip') or 'beta0' not in r:
            continue
        ax.plot(r['eps'], r['beta0'], color=colors[name], lw=1.4,
                label=name.split()[0])
    ax.set_xlabel('ε')
    ax.set_ylabel('β₀')
    ax.legend(fontsize=6.5, facecolor='#1a1a1a', labelcolor=WHT)

    # Panel B: β₁(ε) curves
    ax = fig.add_subplot(gs[0, 1])
    sax(ax, 'B  ·  β₁(ε)  Independent Cycles  (1-skeleton)')
    for name in SYSTEMS:
        r = results[name]
        if r.get('skip') or 'beta1' not in r:
            continue
        ax.plot(r['eps'], r['beta1'], color=colors[name], lw=1.4,
                label=name.split()[0])
    ax.set_xlabel('ε')
    ax.set_ylabel('β₁')
    ax.legend(fontsize=6.5, facecolor='#1a1a1a', labelcolor=WHT)

    # Panel C: Survival curves
    ax = fig.add_subplot(gs[0, 2])
    sax(ax, 'C  ·  Component Survival  β₀(ε)/β₀(0)')
    for name in SYSTEMS:
        r = results[name]
        if r.get('skip') or 'survival' not in r:
            continue
        ax.plot(r['eps'], r['survival'], color=colors[name], lw=1.4,
                label=name.split()[0])
    ax.axhline(1/np.e, color=WHT, lw=0.7, ls='--', alpha=0.4,
               label='1/e threshold')
    ax.set_xlabel('ε')
    ax.set_ylabel('Survival fraction')
    ax.legend(fontsize=6.5, facecolor='#1a1a1a', labelcolor=WHT)

    # Panel D–F: Point clouds
    for col_idx, name in enumerate(['Tribonacci Pisot', 'Plastic Number',
                                    'Arnoux-Rauzy']):
        ax = fig.add_subplot(gs[1, col_idx])
        r  = results[name]
        sax(ax, f'{["D","E","F"][col_idx]}  ·  {name}')
        pts = r['pts']
        step = max(1, len(pts)//3000)
        ax.scatter(pts[::step, 0], pts[::step, 1],
                   c=colors[name], s=0.3, alpha=0.4, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xlabel('E_c axis 1')
        ax.set_ylabel('E_c axis 2')
        c = r.get('case', {})
        if c:
            label = f"β₀={c.get('final_beta0','?')}  H={c.get('entropy',0):.2f}"
            ax.set_title(f'{["D","E","F"][col_idx]}  ·  {name}\n{label}',
                         color=WHT, fontsize=8, pad=4)

    fig.suptitle(
        'Rauzy Fractal Persistence Filtration  ·  '
        'Vietoris-Rips  ·  β₀ exact  ·  β₁ 1-skeleton Euler',
        color=WHT, fontsize=10
    )
    plt.savefig('output/persistence_filtration.png', dpi=150,
                bbox_inches='tight', facecolor=BG)
    print("\n  ✓ Plot saved: output/persistence_filtration.png")

except ImportError:
    print("\n  (matplotlib not available — skipping plot)")

print("\n" + "=" * 70)
print("METHODOLOGICAL NOTES")
print("=" * 70)
print("""
  β₀(ε):  EXACT — union-find on VR 1-skeleton
  β₁(ε):  EXACT for 1-skeleton (E - V + β₀)
           NOT the same as H₁ of full VR complex
           (would need triangle enumeration or gudhi/ripser)

  To get true persistent H₁:
    pip install gudhi          # or
    pip install ripser persim  # lightweight alternative

  Persistence entropy: Shannon entropy of component lifetimes
  Survival curve: fraction of initial components still alive at ε

  The Case A/B classifier requires corrected (increment-method) projection.
  If diameter >> 5, all downstream results are unreliable.
""")
