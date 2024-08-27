#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed. Please install Poetry first."
    exit 1
fi

# Projects to generate environments for
projects=(
    "."
    "apps/fl_client"
    "apps/fl_server"
    "apps/restapi"
    "common"
)

# Function to generate environments
generate_envs() {
    for project in "${projects[@]}"; do
        echo "Setting up environment for $project..."
        cd "$project" || exit
        poetry install
        cd - > /dev/null || exit
    done
}

# Get the directory of the script
script_dir="$(dirname $0)"

# CD into the parent directory
cd "../$script_dir" || exit

generate_envs
