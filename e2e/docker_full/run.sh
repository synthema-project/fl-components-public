script_dir="$(dirname "$(realpath "$0")")"
cd $script_dir || exit

docker compose down --remove-orphans -v
docker compose up --build --exit-code-from synthema-fl-test

if [ $? -eq 0 ]; then
  echo "e2e test passed"
  docker compose down --remove-orphans -v
  exit 0
else
  echo "Test failed"
  docker compose down --remove-orphans -v
  exit 1
fi
