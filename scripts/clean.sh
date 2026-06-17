#!/bin/bash

# List of projects to clean
projects=(
    "."
    "apps/fl_client"
    "apps/fl_server"
    "apps/restapi"
    "apps/status_listener"
    "common"
)

# Function to clean environments and temporary files
clean() {
    for project in "${projects[@]}"; do
        echo "Cleaning environment for $project..."
        cd "$project" || exit
        # Remove local virtual environment if it exists
        rm -rf .venv
        # Remove __pycache__ directories
        find . -name '__pycache__' -exec rm -rf {} +
        # Remove .pyc files
        find . -name '*.pyc' -exec rm -f {} +
        cd - > /dev/null || exit
    done
}

# Get the directory where the current script is located
script_dir="$(dirname "$(realpath "$0")")"

# Change to the parent directory
cd "$(dirname "$script_dir")" || exit

clean
