import argparse
import subprocess as sp
from pathlib import Path

import docker

parser = argparse.ArgumentParser(description="Start the FastAPI app")
parser.add_argument(
    "--mode", type=str, choices=["build", "clean"], help="Action to trigger in docker"
)
args = parser.parse_args()
mode = args.mode

ROOT_DIR = Path(__file__).resolve().parent.parent
APPS_DIR = ROOT_DIR.joinpath("apps")

CURRENT_VERSION = (
    sp.run(["git", "describe", "--tags"], cwd=ROOT_DIR, check=True, capture_output=True)
    .stdout.strip()
    .decode()
)

client = docker.from_env()
if mode == "build":
    print("Building images...")
    print("Building synthema-fl-common...")
    client.images.build(
        path=str(ROOT_DIR.joinpath("common")),
        tag=f"synthema-fl-common:{CURRENT_VERSION}",
        rm=True,
    )
    print("Done.")
    print("Building synthema-fl-client...")
    client.images.build(
        path=str(APPS_DIR.joinpath("fl_client")),
        tag=f"synthema-fl-client:{CURRENT_VERSION}",
        rm=True,
        buildargs={"CURRENT_VERSION": CURRENT_VERSION},
    )
    print("Done.")
    print("Building synthema-fl-server...")
    client.images.build(
        path=str(APPS_DIR.joinpath("fl_server")),
        tag=f"synthema-fl-server:{CURRENT_VERSION}",
        rm=True,
        buildargs={"CURRENT_VERSION": CURRENT_VERSION},
    )
    print("Done.")
    print("Building synthema-fl-restapi...")
    client.images.build(
        path=str(APPS_DIR.joinpath("restapi")),
        tag=f"synthema-fl-restapi:{CURRENT_VERSION}",
        rm=True,
        buildargs={"CURRENT_VERSION": CURRENT_VERSION},
    )
    print("Done.")

elif mode == "clean":
    print("Removing images...")
    print("Removing synthema-fl-common...")
    try:
        client.images.remove(f"synthema-fl-common:{CURRENT_VERSION}")
    except docker.errors.ImageNotFound:
        pass
    print("Removing synthema-fl-client...")
    try:
        client.images.remove(f"synthema-fl-client:{CURRENT_VERSION}")
    except docker.errors.ImageNotFound:
        pass
    print("Done.")
    print("Removing synthema-fl-server...")
    try:
        client.images.remove(f"synthema-fl-server:{CURRENT_VERSION}")
    except docker.errors.ImageNotFound:
        pass
    print("Done.")
    print("Removing synthema-fl-restapi...")
    try:
        client.images.remove(f"synthema-fl-restapi:{CURRENT_VERSION}")
    except docker.errors.ImageNotFound:
        pass
    print("Done.")
