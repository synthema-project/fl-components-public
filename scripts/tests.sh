#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    exit 1
fi

# List of application and library directories
projects=(
    "apps/fl_client"
    "apps/fl_server"
    "apps/restapi"
    "apps/status_listener"
    "common"
)

# Function to run uv commands and handle errors
run_uv_command() {
    local command=$1
    local message=$2
    local project=$3

    echo "Running ${message} for ${project:-"root project"}..."
    uv run $command > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: ${message} failed ${project:+for $project}."
        exit 1
    fi
    echo "Success: ${message} completed."
}

# Function to run tests and quality tools
run_tests() {
    # Run global checks
    run_uv_command "ruff check --fix" "ruff check"
    run_uv_command "ruff format" "ruff format"
    echo "==================================================================="

    # Run tests and checks for each project
    for project in "${projects[@]}"; do
        echo "Running tests and checks for $project..."
        cd "$project" || exit
        run_uv_command "pytest" "pytest" "$project"
        run_uv_command "python -m mypy . --explicit-package-bases" "mypy" "$project"
        cd - > /dev/null || exit
        echo "==================================================================="
    done
}

# Get the directory where the current script is located
script_dir="$(dirname "$(realpath "$0")")"

# Change to the parent directory
cd "$(dirname "$script_dir")" || exit

# Check if .env file exists
if [ -f .env ]; then
    # Export environment variables from .env file
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

run_tests

echo "Running e2e tests..."
bash e2e/docker_full/run.sh > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "e2e test failed"
    exit 1
fi
echo "Success"
