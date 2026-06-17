#!/usr/bin/env bash
# Run this in Git Bash from the finalwrtstand folder:
#   bash git-setup.sh

set -e

echo "=== Setting up git and pushing to GitHub ==="

# Remove broken .git from any previous attempt
if [ -d ".git" ]; then
  echo "Removing existing .git folder..."
  rm -rf .git
fi

git init -b main
git config user.email "ayca.alpman@gmail.com"
git config user.name "Ayca Alpman"
git add .
git commit -m "Initial commit: ENG 101 writing standardization tool"
git remote add origin https://github.com/AycAlp/engwritingnew.git
git push -u origin main

echo ""
echo "Done! Code is live at: https://github.com/AycAlp/engwritingnew"
