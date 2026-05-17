#!/usr/bin/env bash
set -e

TARGET_DIR="symbolic_core/Minerva-dea-mathematica/riemannian_optimization"
mkdir -p "${TARGET_DIR}"

if [ -f "integrated_rauzy_engine.py" ]; then
    mv integrated_rauzy_engine.py "${TARGET_DIR}/"
    echo "Integrated engine stowed successfully inside: ${TARGET_DIR}/"
fi

if [ -f "run_integrated_engine.sh" ]; then
    rm -f run_integrated_engine.sh
fi

git add symbolic_core/
git commit -m "Build: Deploy unified high-precision Chhabra-Jensen & Wavelet Leaders pipeline" || true
git push origin main

echo "================================================================="
echo "⛵ Unified framework updates safely anchored to supreme-barnacle!"
echo "================================================================="
