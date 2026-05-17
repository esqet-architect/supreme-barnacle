#!/usr/bin/env bash

# Navigate to workspace
cd ~/rauzy_analysis || exit 1

# Verify we have the remote origin linked correctly
git remote set-url origin https://github.com/esqet-architect/supreme-barnacle.git

# Fetch the remote changes to see what's causing the conflict
echo "Fetching remote repository history..."
git fetch origin main

# Rebase your local commit on top of the remote changes to align histories
echo "Rebasing local commits onto origin/main..."
git rebase origin/main

# If the remote was just an empty placeholder repo with a README, 
# and a standard rebase encounters an issue, we force-allow independent histories:
if [ $? -ne 0 ]; then
    echo "Standard rebase failed, forcing integration of unrelated histories..."
    git rebase --abort
    git pull origin main --allow-unrelated-histories --no-edit
fi

# Push the synchronized branch back to GitHub
echo "Pushing architecture code to GitHub..."
git push -u origin main
