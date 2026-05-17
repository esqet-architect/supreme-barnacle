#!/usr/bin/env bash

# Navigate to workspace
cd ~/rauzy_analysis || exit 1

# Commit any loose modifications to the README if not already finalized
git add README.md 2>/dev/null
git commit -m "docs: finalize Supreme Barnacle architectural engine specs" 2>/dev/null

# Fetch the current remote state safely
echo "Fetching remote repository status..."
git fetch origin main

# Rebase local milestones on top of the remote updates to reconcile history cleanly
echo "Rebasing local commits onto origin/main..."
git rebase origin/main

# Push the unified tracking branch back to GitHub
echo "Pushing updated architecture to GitHub..."
git push -u origin main
