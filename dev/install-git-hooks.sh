#!/bin/bash

# Create symbolic link for pre-commit hook
ln -sf "../../dev/git-hooks/pre-commit" ".git/hooks/pre-commit"
chmod +x "dev/git-hooks/pre-commit"

echo "Git hooks installed successfully!" 