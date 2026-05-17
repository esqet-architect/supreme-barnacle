#!/bin/bash
FILES=(
    "README_WHITEPAPER_SERIES.md"
    "publications/scope_limits.tex"
    "zenodo_instructions.html"
)

echo "--- Checking Zenodo Package Manifest ---"
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "[OK] Found: $file"
    else
        echo "[MISSING] Error: $file not found!"
    fi
done

echo "----------------------------------------"
echo "Metadata Summary for Zenodo:"
grep -E "Title:|Author:|ORCID:" zenodo_instructions.html | sed 's/<[^>]*>//g'
