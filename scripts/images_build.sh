#!/bin/bash
set -e

# Argumentos del script
MODE=$1
VERSION=$2
DOCKER_REGISTRY=$3

# Directorios y versionamiento
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPS_DIR="$ROOT_DIR/apps"

# Función para construir imágenes
build_images() {
    echo "Building images..."

    images_to_build=(
        "${DOCKER_REGISTRY}synthema-fl-common:$ROOT_DIR/common"
        "${DOCKER_REGISTRY}synthema-fl-client:$APPS_DIR/fl_client"
        "${DOCKER_REGISTRY}synthema-fl-server:$APPS_DIR/fl_server"
        "${DOCKER_REGISTRY}synthema-fl-restapi:$APPS_DIR/restapi"
    )

    for image_info in "${images_to_build[@]}"; do
        IFS=":" read -r name path <<< "$image_info"
        echo "Building $name..."
        docker build "$path" -t "$name:$VERSION" --build-arg CURRENT_VERSION="$VERSION" --build-arg BASE_IMAGE="${DOCKER_REGISTRY}synthema-fl-common" --rm
        echo "Done."
    done
}

# Función para limpiar imágenes
clean_images() {
    echo "Removing images..."

    images_to_remove=(
        "${DOCKER_REGISTRY}synthema-fl-common"
        "${DOCKER_REGISTRY}synthema-fl-client"
        "${DOCKER_REGISTRY}synthema-fl-server"
        "${DOCKER_REGISTRY}synthema-fl-restapi"
    )

    for name in "${images_to_remove[@]}"; do
        echo "Removing $name..."
        docker rmi "$name:$VERSION" || true
        echo "Done."
    done
}

# Lógica principal para ejecutar según el modo
if [ "$MODE" == "build" ]; then
    build_images
elif [ "$MODE" == "clean" ]; then
    clean_images
else
    echo "Invalid mode. Use 'build' or 'clean'."
    exit 1
fi
