#!/usr/bin/env bash
set -e

TARGET_DIR="symbolic_core/Minerva-dea-mathematica/riemannian_optimization"
mkdir -p "${TARGET_DIR}"

if [ -f "advanced_rauzy_multifractal.py" ]; then
    mv advanced_rauzy_multifractal.py "${TARGET_DIR}/"
    echo "Advanced engine safely relocated to repository directory structure."
fi

if [ -f "run_advanced_multifractal.sh" ]; then
    rm -f run_advanced_multifractal.sh
fi

git add symbolic_core/
git commit -m "Build: Integrate advanced Chhabra-Jensen and Wavelet Leaders analyzer" || true
git push origin main

echo "================================================================="
echo "⛵ Advanced multifractal analytics synced to supreme-barnacle!"
echo "================================================================="
