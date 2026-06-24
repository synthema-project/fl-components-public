#!/usr/bin/env python3
"""
Docker Image Management Script (Complete Version)

Usage:
    python images.py --mode <build|remove|test|tag|untag|push> [options]

Modes:
    build
        Build Docker images defined in docker_config.json.
        The first image is treated as the base image, which will be built first.
        Options:
            --tag <tag>            Required. Tag for the images (e.g., 1.0.0).
    remove
        Remove all images listed in docker_config.json (including all their tags).
        No additional options required.
    test
        Run tests (e.g., pytest) in images that have "test": true in the config.
        Options:
            --tag <tag>            Required. Tag of the images to test.
    tag
        Assign a new tag to images that currently have an old tag.
        Options:
            --old-tag <old_tag>    Required. Existing tag to look for.
            --new-tag <new_tag>    Required. New tag to assign.
    untag
        Remove a specific tag from all images (but do not remove the image if it has other tags).
        Options:
            --tag <tag>            Required. Tag to remove.
    push
        Push images to a registry (e.g., Docker Hub or a private registry).
        Options:
            --tag <tag>            Required. Tag of the images to push.
            --registry <registry>  Required. Registry prefix (e.g., 'myregistry.com/').

Examples:
    python images.py --mode build --tag 1.0.0
    python images.py --mode remove
    python images.py --mode test --tag 1.0.0
    python images.py --mode tag --old-tag 1.0.0 --new-tag latest
    python images.py --mode untag --tag latest
    python images.py --mode push --tag 1.0.0 --registry myregistry.com/
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path
import docker
from docker.errors import ImageNotFound, APIError, BuildError
import os
os.environ["DOCKER_BUILDKIT"] = "1"

# Local checkout of the model-store-interface package (a sibling of the repo root).
# It cannot be reached from a Docker build context, so we build its wheel and vendor
# it into each context that needs it (see prepare_msi_wheel / Dockerfile UV_FIND_LINKS).
MSI_DIRNAME = "app-model_store-interface"


# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["build", "remove", "test", "tag", "untag", "push"],
        required=True,
        help="Action to perform on Docker images.",
    )
    parser.add_argument(
        "--tag", type=str, help="Tag for the images (or tag to remove)."
    )
    parser.add_argument(
        "--registry", type=str, help="Registry prefix (required for push)."
    )
    parser.add_argument(
        "--old-tag", type=str, help="Old tag (required for mode 'tag')."
    )
    parser.add_argument(
        "--new-tag", type=str, help="New tag (required for mode 'tag')."
    )
    return parser.parse_args()


# ------------------------------------------------------------------------------
# Load Configuration
# ------------------------------------------------------------------------------
def load_config(config_file):
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[ERROR] Configuration error: {e}")
        exit(1)


# ------------------------------------------------------------------------------
# Path Handling
# ------------------------------------------------------------------------------
def get_context_path(root_dir, image_path):
    """Resolve the absolute path for the Docker build context."""
    resolved_path = root_dir / image_path
    if not resolved_path.exists():
        print(f"[ERROR] Context path does not exist: {resolved_path}")
        exit(1)
    return resolved_path


# ------------------------------------------------------------------------------
# model-store-interface wheel vendoring
# ------------------------------------------------------------------------------
def prepare_msi_wheel(root_dir, context_path):
    """Build the model-store-interface wheel and vendor it into the build context.

    The Docker build context cannot reach the sibling ``app-model_store-interface``
    repo, so we build its wheel and copy it into ``<context>/wheels/`` where the
    Dockerfile installs it via ``UV_FIND_LINKS`` during ``uv sync``.
    """
    interface_dir = (root_dir.parent / MSI_DIRNAME).resolve()
    if not interface_dir.exists():
        print(f"[ERROR] model-store-interface repo not found at {interface_dir}")
        return False

    print("[INFO] Building model-store-interface wheel...")
    if subprocess.run(["uv", "build", "--wheel"], cwd=str(interface_dir)).returncode != 0:
        print("[ERROR] Failed to build model-store-interface wheel")
        return False

    wheels_dir = context_path / "wheels"
    wheels_dir.mkdir(exist_ok=True)
    for stale in wheels_dir.glob("model_store_interface-*.whl"):
        stale.unlink()
    copied = 0
    for whl in (interface_dir / "dist").glob("model_store_interface-*.whl"):
        shutil.copy2(whl, wheels_dir / whl.name)
        copied += 1
    if copied == 0:
        print("[ERROR] No model-store-interface wheel found after build")
        return False
    print(f"[OK] Vendored {copied} model-store-interface wheel(s) into {wheels_dir}")
    return True


# ------------------------------------------------------------------------------
# Docker Operations
# ------------------------------------------------------------------------------
def build_image(client, image_def, image_tag, root_dir, base_image=None):
    """Build a Docker image.

    Uses the Docker CLI with BuildKit enabled, since the docker-py SDK does not
    support BuildKit (required by Dockerfiles that use `RUN --mount=...`).
    """
    path = get_context_path(root_dir, image_def["path"])
    full_tag = f"{image_def['name']}:{image_tag}"

    # Images that depend on model-store-interface need its wheel vendored into the context.
    if image_def.get("msi"):
        if not prepare_msi_wheel(root_dir, path):
            return False

    build_args = {"CURRENT_VERSION": image_tag} if image_def.get("buildargs") else {}
    if base_image:
        build_args["BASE_IMAGE"] = base_image

    cmd = ["docker", "build", "--rm", "--platform", "linux/amd64", "-t", full_tag]
    for key, value in build_args.items():
        cmd += ["--build-arg", f"{key}={value}"]
    cmd.append(str(path))

    env = {**os.environ, "DOCKER_BUILDKIT": "1"}
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Failed to build {full_tag}")
        return False

    print(f"[OK] Built image: {full_tag}")
    return True


def tag_image(client, config, old_tag, new_tag):
    """Tag images with a new tag."""
    if not old_tag or not new_tag:
        print("[ERROR] --old-tag and --new-tag are required for tagging.")
        return
    for img in config["images"]:
        old = f"{img['name']}:{old_tag}"
        new = f"{img['name']}:{new_tag}"
        try:
            client.images.get(old).tag(new)
            print(f"[OK] Tagged {old} as {new}")
        except ImageNotFound:
            print(f"[ERROR] Image not found: {old}")
        except APIError as e:
            print(f"[ERROR] Failed to tag image {old}: {e}")


def untag_image(client, config, tag):
    """Remove a specific tag from images."""
    if not tag:
        print("[ERROR] --tag is required for untagging.")
        return
    for img in config["images"]:
        try:
            client.images.remove(f"{img['name']}:{tag}")
            print(f"[OK] Removed tag {tag} from {img['name']}")
        except APIError as e:
            print(f"[ERROR] Failed to remove tag {tag}: {e}")


def push_images(client: docker.DockerClient, config, tag, registry):
    """Push images to a registry."""
    if not tag or not registry:
        print("[ERROR] --tag and --registry are required for pushing.")
        return

    new_imgs = []

    for img in config["images"]:
        local_tag = f"{img['name']}:{tag}"
        full_tag = f"{registry.rstrip('/')}/{img['name']}:{tag}"
        try:
            # Ensure the image exists locally
            local_image = client.images.get(local_tag)

            # Tag the image for the registry
            local_image.tag(full_tag)
            new_imgs.append(client.images.get(full_tag))
            print(f"[INFO] Tagged {local_tag} as {full_tag}")

            # Push the image to the registry
            for line in client.images.push(
                repository=full_tag, stream=True, decode=True
            ):
                if "error" in line:
                    print(f"[ERROR] Push error: {line['error']}")
                    return

            print(f"[OK] Successfully pushed {full_tag}")
        except ImageNotFound:
            print(f"[ERROR] Local image not found: {local_tag}")
        except APIError as e:
            print(f"[ERROR] API error during push for {full_tag}: {e}")


def test_images(client: docker.DockerClient, config, tag):
    """Run tests for Docker images that have 'test': true in the configuration."""
    if not tag:
        print("[ERROR] --tag is required for testing images.")
        return

    for img in config["images"]:
        if img.get("test", False):  # Check if 'test' is enabled in the config
            full_tag = f"{img['name']}:{tag}"
            try:
                # Ensure the image exists locally
                client.images.get(full_tag)

                print(f"[INFO] Running tests for image: {full_tag}")
                client.containers.run(
                    image=full_tag,
                    command="pytest tests",  # Replace with your actual test command
                    remove=True,
                )

                print(f"[OK] Tests passed for image: {full_tag}")
            except ImageNotFound:
                print(f"[ERROR] Local image not found for testing: {full_tag}")
            except APIError as e:
                print(f"[ERROR] API error during testing for {full_tag}: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error during testing for {full_tag}: {e}")


# ------------------------------------------------------------------------------
# Handlers for Each Mode
# ------------------------------------------------------------------------------
def handle_build(client, config, tag, root_dir):
    """Build all images in the configuration."""
    if not tag:
        print("[ERROR] --tag is required for building images.")
        return
    base_image_def = config["images"][0]
    dependent_images = config["images"][1:]

    print("[INFO] Building base image...")
    if not build_image(client, base_image_def, tag, root_dir):
        print("[ERROR] Base image build failed. Aborting.")
        return

    print("[INFO] Building dependent images...")
    for img in dependent_images:
        build_image(client, img, tag, root_dir, base_image=base_image_def["name"])


def handle_remove(client, config):
    """Remove all images listed in the configuration."""
    print("[INFO] Removing images...")
    for image_def in config["images"][::-1]:
        try:
            images = client.images.list(name=image_def["name"])
            for img in images:
                client.images.remove(image=img.id, force=True)
                print(f"[OK] Removed: {img.id}")
        except APIError as e:
            print(f"[ERROR] Failed to remove images: {e}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    config_file = Path(__file__).resolve().parent / "docker_config.json"
    config = load_config(config_file)
    root_dir = config_file.parent.parent
    client = docker.from_env()

    if args.mode == "build":
        handle_build(client, config, args.tag, root_dir)
    elif args.mode == "remove":
        handle_remove(client, config)
    elif args.mode == "tag":
        tag_image(client, config, args.old_tag, args.new_tag)
    elif args.mode == "untag":
        untag_image(client, config, args.tag)
    elif args.mode == "push":
        push_images(client, config, args.tag, args.registry)
    elif args.mode == "test":
        test_images(client, config, args.tag)
    else:
        print("[ERROR] Unsupported mode.")


if __name__ == "__main__":
    main()
