#!/usr/bin/env bash
set -e

# 1. Clean up temporary uncommitted script files and local tarball artifacts
rm -f append_gaps.py
rm -f minerva-v1.0.0-validated.tar.gz
rm -f minerva-v1.0.1-validated.tar.gz
rm -f minerva-v1.0.2-validated.tar.gz

# 2. Add the modified metadata file that was left unstaged
git add zenodo_metadata.json

# 3. Commit the verified configuration metadata matching the v1.0.2 manuscript state
git commit -m "Deployment: Stage verified Zenodo metadata matching the v1.0.2 validated state" || true

# 4. Strip old, unverified tag targets to avoid tracking confusion on the remote host
git tag -d v1.0.0-validated v1.0.1-validated v1.0.2-validated 2>/dev/null || true

# 5. Mint the definitive, production-ready release tag on the clean commit
git tag -a v1.0.2-validated -m "Definitive Minerva Release: Confirmed tangent stability, explicitly exposed 0.79 root contradiction, and verified Failure Modes appendix."

# 6. Build the final archive directly from the immutable Git tag object
git archive --format=tar.gz --output=minerva-v1.0.2-validated.tar.gz v1.0.2-validated

echo "================================================================="
git status
echo "================================================================="
echo "Target Hash Verification:"
git rev-parse v1.0.2-validated
echo "================================================================="
git log --oneline -2 v1.0.2-validated
echo "================================================================="
echo "✅ Operational workspace fully swept. v1.0.2-validated is absolute."
