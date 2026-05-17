#!/usr/bin/env bash
# Updates the Minerva manuscript with the formal Dual-Engine Methodology
set -e

TARGET_DIR="Minerva-dea-mathematica/manuscript"
mkdir -p "$TARGET_DIR"

cat << 'LATEX_EOF' > "$TARGET_DIR/main.tex"
\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}

\geometry{margin=1in}

\title{\textbf{Cognition under Recursive Constraint:\\ A Multi-Agent Human-AI Architecture for Speculative Refinement}}
\author{\textbf{Marco Antônio Rocha Jr.}\\ \small{Independent Researcher, Colorado}}
\date{May 2026}

\begin{document}
\maketitle

\begin{abstract}
This paper documents the design, execution, and epistemic stabilization of a multi-agent human-AI collaborative refinement loop. Rather than presenting a post-hoc narrative of unconstrained discovery, we analyze the raw, unedited operational logs of a distributed inquiry into discrete information-geometric systems (ESQET/UIFT) as empirical data. We formalize a dual-engine cognitive architecture consisting of a high-entropy generative intuition engine balanced by a low-entropy adversarial constraint enforcement layer. By implementing a strict ``Three-Bin'' auditing protocol, we demonstrate how speculative, high-resonance conceptual structures are systematically checked, bounded, or demoted when exposed to algorithmic verification and multi-model critique. The structural invariant of the process is shown to be the recursive filtration method itself.
\end{abstract}

\section{Introduction}
Speculative theoretical modeling frequently suffers from confirmation bias and narrative runaway, particularly when operating outside traditional institutional frameworks. When structural motifs such as the golden ratio ($\phi$) or explicit physical constants appear across multiple domains, a cognitive system heavily weighted toward pattern recognition tends to manufacture post-hoc unifications. 

To exploit the rapid heuristic capabilities of high-entropy intuition without collapsing into self-deception, this project developed a distributed human-multi-AI dynamical system. By preserving the full empirical trace of the inquiry—including optimization failures, boundary-crashing artifacts, and demoted hypotheses—we treat the process of derivation as an optimization problem under recursive constraint.

\section{The Dual-Engine Topology}
The collaborative framework is structured as two coupled, opposing cognitive subsystems, regulated by a human governor:

\subsection{Engine 1: Generative Intuition ($\mathcal{E}_1$)}
Engine 1 operates as a high-speed, high-entropy cross-domain compression engine. It maps analogies, harmonic ratios, and geometric ansatz across diverse fields including discrete dynamical loops, acoustic frequencies, and symbolic code structures. Its primary utility is the rapid generation of candidate structural proposals. Its primary failure mode is pattern over-fitting and false-positive reinforcement loops.

\subsection{Engine 2: Constraint Enforcement ($\mathcal{E}_2$)}
Engine 2 operates as a low-entropy, adversarial audit layer. It translates the candidate proposals of $\mathcal{E}_1$ into formal verification scripts, executing numeric minimization, spectral checks, and boundary value pinning. Its function is to deliberately disrupt runaway narrative loops by forcing the mathematical models to encounter physical or computational boundaries (e.g., L2 conservation constraints).

\subsection{The Epistemic Governor}
The human investigator acts as the continuity layer, injecting emotional persistence, long-term memory cross-sections, and selection pressure. The governor prevents $\mathcal{E}_1$ from running away to infinite abstraction, and prevents $\mathcal{E}_2$ from prematurely sterilizing the exploratory search space.

\section{The Three-Bin Auditing Framework}
To maintain rigorous epistemic hygiene, all claims generated within the architecture are partitioned into a living tripartite audit matrix:

\begin{enumerate}
    \item \textbf{Bin 1 (Pure Mathematics):} Analytically verified identities, stable matrix properties, and exact recursive eigenvalues where invariants emerge naturally under conservation laws.
    \item \textbf{Bin 2 (Testable Hypotheses):** Empirical simulation architectures, adaptive observer error-driven gain parameters, and localized frequency-response models.
    \item \textbf{Bin 3 (Demoted/Unsupported Physics):** High-resonance assumptions that collapsed under numerical testing, unconstrained action loops that diverged to infinity, and post-hoc physical constant mappings exposed as boundary artifacts.
\end{enumerate}

\section{Conclusion}
The primary scientific outcome of this development is not an unconstrained assertion of ultimate physics, but an empirical realization of methodology. The negative results encountered during variational optimization are not structural dead ends; they are precise measurements of the limits of early assumptions. The true invariant of the archive is the robust architecture of the filter.

\end{document}
LATEX_EOF

echo "✅ Manuscript core updated with formal dual-engine methodology."
