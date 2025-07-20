#!/bin/bash

# Stage all changes
git add .

# Prompt for commit message
read -p "Enter commit message: " commit_message

# Commit with the provided message
git commit -m "$commit_message"

# Push to the current branch
git push

