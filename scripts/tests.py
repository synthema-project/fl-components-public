import os
import subprocess
import json
import shutil
from pathlib import Path


def handle_error(message, exception=None, exit_program=True):
    """
    Centralized error handling function.
    Args:
        message (str): Error message to display.
        exception (Exception, optional): Exception object to log. Defaults to None.
        exit_program (bool): Whether to exit the program. Defaults to True.
    """
    print(f"[ERROR] {message}")
    if exception:
        print(f"[DETAILS] {exception}")
    if exit_program:
        exit(1)


def run_command(command, description, project=None):
    """
    Run a shell command and handle errors.
    Args:
        command (str): Command to execute.
        description (str): Description of the operation.
        project (str, optional): Project name for context. Defaults to None.
    """
    context = f" for {project}" if project else ""
    print(f"Running {description}{context}...")

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)  # Deactivate any active virtualenv
    env.pop("CONDA_PREFIX", None)  # Deactivate any active Conda environment

    try:
        _ = subprocess.run(
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


def load_docker_config():
    """
    Load the Docker configuration file relative to the script directory.
    Returns:
        dict: Parsed JSON configuration.
    """
    script_dir = Path(__file__).resolve().parent
    config_file = script_dir / "docker_config.json"

    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        handle_error(
            f"Failed to load Docker configuration from {config_file}.", exception=e
        )


def run_tests():
    """
    Run tests and quality tools for each project in the configuration.
    """
    # Load Docker configuration
    docker_config = load_docker_config()

    # Resolve root directory
    script_dir = Path(__file__).resolve().parent.parent

    # Run global checks
    run_command("poetry run ruff check --fix", "ruff check")
    run_command("poetry run ruff format", "ruff format")
    print("=" * 67)

    # Iterate over projects and run tests
    for image in docker_config["images"]:
        project_path = script_dir / Path(image["path"])
        print(f"Running tests and checks for {project_path}...")
        if not project_path.exists():
            handle_error(f"Project path {project_path} does not exist.")

        os.chdir(project_path)

        if image.get("test"):
            run_command("poetry run pytest", "pytest", project=project_path)
            run_command(
                "poetry run python -m mypy . --explicit-package-bases",
                "mypy",
                project=project_path,
            )

        os.chdir(script_dir)  # Return to the original script directory
        print("=" * 67)


def run_e2e_tests():
    """
    Run end-to-end tests.
    """
    script_dir = Path(__file__).resolve().parent
    e2e_script = script_dir.parent / "e2e/docker_full/run.sh"
    print(f"Running e2e tests at {e2e_script}...")

    if not e2e_script.exists():
        handle_error(f"End-to-end test script not found at {e2e_script}.")

    try:
        subprocess.run(
            f"bash {e2e_script}",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("[SUCCESS] e2e tests completed.")
    except subprocess.CalledProcessError as e:
        handle_error(
            "e2e tests failed.",
            exception=f"Exit Code: {e.returncode}\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}",
        )


def main():
    # Ensure Poetry is installed
    if not shutil.which("poetry"):
        handle_error("Poetry is not installed. Please install Poetry first.")

    # Run tests and tools
    run_tests()

    # Run e2e tests
    run_e2e_tests()


if __name__ == "__main__":
    main()
