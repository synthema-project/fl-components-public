#!/bin/bash

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed. Please install Poetry first."
    exit 1
fi


# Lista de directorios de aplicaciones y librerías
projects=(
    "apps/fl_client"
    "apps/fl_server"
    "apps/restapi"
    "common"
)

# Función para ejecutar pruebas y herramientas de calidad
run_tests() {
    echo "Running ruff check..."
    poetry run ruff check > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: ruff check failed."
        exit 1
    else
        echo "ruff check passed."
        echo "==================================================================="
    fi

    echo "Running ruff format..."
    poetry run ruff format > /dev/null
    if [ $? -ne 0 ]; then
        echo "Error: ruff format failed."
        exit 1
    else
        echo "ruff format passed."
        echo "==================================================================="
    fi

    for project in "${projects[@]}"; do
        echo "Running tests and checks for $project..."
        cd "$project" || exit
        poetry run pytest > /dev/null
        if [ $? -ne 0 ]; then
            echo "Error: pytest failed for $project."
            exit 1
        fi
        poetry run python -m mypy . --explicit-package-bases > /dev/null
        if [ $? -ne 0 ]; then
            echo "Error: mypy failed for $project."
            exit 1
        fi
        cd - > /dev/null || exit
        echo "Tests and checks passed for $project."
        echo "==================================================================="
    done
}


# Obtener el directorio donde se encuentra el script actual
script_dir="$(dirname $0)"

# Cambiar al directorio padre
cd "../$script_dir" || exit

# Check if .env file exists
if [ -f .env ]; then
    # Export environment variables from .env file
    export $(grep -v '^#' .env | xargs)
else
    echo "Error: .env file not found."
    exit 1
fi

run_tests
