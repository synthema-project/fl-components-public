set -e

VERSION=$1
DOCKER_REGISTRY=$2

images=(
    "${DOCKER_REGISTRY}synthema-fl-client"
    "${DOCKER_REGISTRY}synthema-fl-server"
    "${DOCKER_REGISTRY}synthema-fl-restapi"
)

for name in "${images[@]}"; do
    echo "Testing $name..."
    docker run --rm "$name:$VERSION" pytest tests
    echo "Done."
done