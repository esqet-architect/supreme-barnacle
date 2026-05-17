#!/usr/bin/env bash
# Verifies and prepares the final state of the repository components
set -e

REPO_NAME="Minerva-dea-mathematica"

echo "================================================================="
echo "MINERVA ARCHIVE: FINAL INTEGRITY CHECK"
echo "================================================================="

# Check files
files_to_verify=(
    "${REPO_NAME}/README.md"
    "${REPO_NAME}/manuscript/main.tex"
    "${REPO_NAME}/cognitive_pipeline_data/three_bin_audit.json"
)

for file in "${files_to_verify[@]}"; do
    if [ -f "$file" ]; then
        echo "──> [VERIFIED]: $file is present and written."
    else
        echo "──> [ERROR]: Missing expected target component: $file"
        exit 1
    fi
done

echo "-----------------------------------------------------------------"
echo "Current Living Audit Configuration Status:"
echo "-----------------------------------------------------------------"
python3 -c "
import json
with open('${REPO_NAME}/cognitive_pipeline_data/three_bin_audit.json') as f:
    d = json.load(f)
print('Bin 1 (Math):', len(d['bins']['bin_1_pure_math']))
print('Bin 2 (Hypotheses):', len(d['bins']['bin_2_testable_hypotheses']))
print('Bin 3 (Demoted):', len(d['bins']['bin_3_demoted_or_unsupported']))
"
echo "================================================================="
echo "✅ Verification complete. All files synchronized to the stable manifold."
