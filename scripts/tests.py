import os
import subprocess
import json
import shutil
from pathlib import Path
from typing import Any, Optional

ENV_VARS_TO_REMOVE = ["VIRTUAL_ENV", "CONDA_PREFIX"]


def handle_error(message, exception=None, exit_program=True):
    """Centralized error handling."""
    print(f"[ERROR] {message}")
    if exception:
        print(f"[DETAILS] {exception}")
    if exit_program:
        exit(1)


def run_command(command: str, description: str, project: Optional[str] = None):
    """Run a shell command and handle errors."""
    context = f" for {project}" if project else ""
    print(f"Running {description}{context}...")

    env = os.environ.copy()
    root = Path(__file__).resolve().parents[1]
    paths = [root / 'common', root] + [root / 'apps' / name / 'src'
            for name in ('fl_client', 'fl_server', 'restapi', 'status_listener')]
    env['PYTHONPATH'] = os.pathsep.join(map(str, paths)) + os.pathsep + env.get('PYTHONPATH', '')
    for var in ENV_VARS_TO_REMOVE:
        env.pop(var, None)  # Ensure a clean environment

    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        print(f"[SUCCESS] {description} completed{context}.")
    except subprocess.CalledProcessError as e:
        handle_error(
            f"{description} failed{context}.",
            exception=f"Command: {command}\nExit Code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}",
        )


def load_docker_config() -> Any:
    """Load the Docker configuration file."""
    config_file = Path(__file__).resolve().parent / "docker_config.json"
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        handle_error(f"Failed to load Docker configuration from {config_file}.", e)


def run_tests():
    """Run tests and quality tools for each project in the configuration."""
    docker_config = load_docker_config()
    script_dir = Path(__file__).resolve().parent.parent

    os.chdir(script_dir)
    # Global checks
    for tool in ["ruff check", "ruff format --check"]:
        run_command(f"uv run --locked {tool}", tool)

    print("=" * 67)

    for image in docker_config["images"]:
        project_path = script_dir / Path(image["path"])
        if not project_path.exists():
            handle_error(f"Project path {project_path} does not exist.")

        os.chdir(project_path)
        print(f"Running tests for {project_path}...")

        if image.get("test"):
            for test_cmd in ["pytest", "python -m mypy . --explicit-package-bases"]:
                run_command(
                    f"uv run --locked {test_cmd}", test_cmd, project=str(project_path)
                )

        os.chdir(script_dir)  # Restore working directory
        print("=" * 67)


def build_docker_images():
    """Build Docker images with tag 'test' before e2e tests."""
    script_dir = Path(__file__).resolve().parent
    images_script = script_dir / "images.py"

    if not images_script.exists():
        handle_error(f"Image management script not found: {images_script}")

    run_command(f"python {images_script} --mode build --tag test", "Docker image build")


def run_e2e_tests():
    """Run end-to-end tests."""
    script_dir = Path(__file__).resolve().parent
    e2e_script = script_dir.parent / "e2e/docker_full/run.sh"

    if not e2e_script.exists():
        handle_error(f"End-to-end test script not found at {e2e_script}.")

    run_command(f"bash {e2e_script}", "e2e tests")


def main():
    """Main function to execute test and e2e processes."""
    if not shutil.which("uv"):
        handle_error("uv is not installed. Please install uv first.")

    run_tests()
    build_docker_images()  # New step before running e2e tests
    run_e2e_tests()


if __name__ == "__main__":
    main()
