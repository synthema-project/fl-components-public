#!/bin/bash

# Lista de directorios de aplicaciones y librerías
projects=(
    "."
    "apps/fl_client"
    "apps/fl_server"
    "apps/restapi"
    "common"
)

# Función para limpiar entornos virtuales y archivos temporales
clean() {
    for project in "${projects[@]}"; do
        echo "Cleaning environment for $project..."
        cd "$project" || exit
        # Eliminar el entorno virtual
        poetry env remove --all
        # Eliminar archivos __pycache__
        find . -name '__pycache__' -exec rm -rf {} +
        # Eliminar archivos .pyc
        find . -name '*.pyc' -exec rm -f {} +
        cd - > /dev/null || exit
    done
}

# Obtener el directorio donde se encuentra el script actual
script_dir="$(dirname $0)"

# Cambiar al directorio padre
cd "../$script_dir" || exit

clean
