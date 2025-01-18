#!/bin/bash

# Create symbolic link for pre-push hook
ln -sf "../../dev/git-hooks/pre-push" ".git/hooks/pre-push"
chmod +x "dev/git-hooks/pre-push"

echo "Git hooks installed successfully!" 