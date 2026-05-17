#!/bin/bash
echo "Opening Zenodo deposit page..."
echo ""

Try various browsers

if command -v termux-open &> /dev/null; then
termux-open "https://zenodo.org/deposit/new"
elif command -v xdg-open &> /dev/null; then
xdg-open "https://zenodo.org/deposit/new"
elif command -v python3 &> /dev/null; then
python3 -m webbrowser "https://zenodo.org/deposit/new"
else
echo "Please open this URL manually: https://zenodo.org/deposit/new"
fi

echo ""
echo "When the page loads, use these values:"
echo ""
echo "Title: ESQET: Emergent Spacetime Quantum-Entanglement Theory — Complete Whitepaper Series"
echo "Author: Rocha Jr., Marco Antônio"
echo "ORCID: 0009-0004-9757-2853"
echo ""
echo "Upload these files from: $(pwd)"
ls -la README_WHITEPAPER_SERIES.md publications/scope_limits.tex zenodo_instructions.html 2>/dev/null
