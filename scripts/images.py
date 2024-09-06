import argparse
from pathlib import Path

import docker
from docker.errors import ImageNotFound

# Argument parser setup
parser = argparse.ArgumentParser(description="Start the FastAPI app")
parser.add_argument(
    "--mode", type=str, choices=["build", "clean"], help="Action to trigger in Docker"
)
parser.add_argument("--version", type=str, help="Version to tag the Docker images with")
args = parser.parse_args()
mode = args.mode
version = args.version

# Directories and versioning setup
ROOT_DIR = Path(__file__).resolve().parent.parent
APPS_DIR = ROOT_DIR.joinpath("apps")

# Docker client setup
client = docker.from_env()


# Functions for building and cleaning Docker images
def build_images():
    """Build Docker images for different components."""
    print("Building images...")

    images_to_build = [
        ("synthema-fl-common", ROOT_DIR.joinpath("common")),
        ("synthema-fl-client", APPS_DIR.joinpath("fl_client")),
        ("synthema-fl-server", APPS_DIR.joinpath("fl_server")),
        ("synthema-fl-restapi", APPS_DIR.joinpath("restapi")),
    ]

    for name, path in images_to_build:
        print(f"Building {name}...")
        client.images.build(
            path=str(path),
            tag=f"{name}:{version}",
            rm=True,
            buildargs={"CURRENT_VERSION": version}
            if "client" in name or "server" in name or "restapi" in name
            else {},
        )
        print("Done.")


def clean_images():
    """Remove Docker images for different components."""
    print("Removing images...")

    images_to_remove = [
        "synthema-fl-common",
        "synthema-fl-client",
        "synthema-fl-server",
        "synthema-fl-restapi",
    ]

    for name in images_to_remove:
        print(f"Removing {name}...")
        try:
            client.images.remove(f"{name}:{version}")
        except ImageNotFound:
            pass
        print("Done.")


# Main logic for mode handling
if mode == "build":
    build_images()
elif mode == "clean":
    clean_images()
