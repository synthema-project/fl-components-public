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
from pathlib import Path
import docker
from docker.errors import ImageNotFound, APIError, BuildError

# ------------------------------------------------------------------------------
# Centralized Error Handling
# ------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["build", "remove", "test", "tag", "untag", "push"], required=True, help="Action to perform on Docker images.")
    parser.add_argument("--tag", type=str, help="Tag for the images (or tag to remove).")
    parser.add_argument("--registry", type=str, help="Registry prefix (required for push).")
    parser.add_argument("--old-tag", type=str, help="Old tag (required for mode 'tag').")
    parser.add_argument("--new-tag", type=str, help="New tag (required for mode 'tag').")
    return parser.parse_args()

# ------------------------------------------------------------------------------
# Load Configuration
# ------------------------------------------------------------------------------
def load_config(config_file):
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        handle_error("Configuration error", exception=e)

# ------------------------------------------------------------------------------
# Path Handling
# ------------------------------------------------------------------------------
def get_context_path(root_dir, image_path):
    """Resolve the absolute path for the Docker build context."""
    resolved_path = root_dir / image_path
    if not resolved_path.exists():
        handle_error(f"Context path does not exist: {resolved_path}")
    return resolved_path

# ------------------------------------------------------------------------------
# Docker Operations
# ------------------------------------------------------------------------------
def build_image(client, image_def, image_tag, root_dir, base_image=None):
    """Build a Docker image."""
    path = get_context_path(root_dir, image_def["path"])
    full_tag = f"{image_def['name']}:{image_tag}"
    build_args = {"CURRENT_VERSION": image_tag} if image_def.get("buildargs") else {}
    if base_image:
        build_args["BASE_IMAGE"] = base_image

    try:
        client.images.build(path=str(path), tag=full_tag, rm=True, buildargs=build_args)
        print(f"[OK] Built image: {full_tag}")
    except (BuildError, APIError) as e:
        handle_error(f"Failed to build {full_tag}", exception=e)

# Similar updates for tag_image, untag_image, push_images, test_images, and other functions.

# ------------------------------------------------------------------------------
# Main Handlers for Each Mode
# ------------------------------------------------------------------------------
def handle_build(client, config, tag, root_dir):
    """Build all images in the configuration."""
    if not tag:
        handle_error("--tag is required for building images.")
    base_image_def = config["images"][0]
    dependent_images = config["images"][1:]

    print(f"[INFO] Building base image...")
    build_image(client, base_image_def, tag, root_dir)

    print(f"[INFO] Building dependent images...")
    for img in dependent_images:
        build_image(client, img, tag, root_dir, base_image=base_image_def["name"])

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    config_file = Path(__file__).resolve().parent / "docker_config.json"
    config = load_config(config_file)
    root_dir = config_file.parent.parent
    client = docker.from_env()

    try:
        if args.mode == "build":
            handle_build(client, config, args.tag, root_dir)
        elif args.mode == "remove":
            print("[INFO] Removing images...")  # Add remove logic here
        elif args.mode == "tag":
            print("[INFO] Tagging images...")  # Add tag logic here
        elif args.mode == "untag":
            print("[INFO] Untagging images...")  # Add untag logic here
        elif args.mode == "push":
            print("[INFO] Pushing images...")  # Add push logic here
        elif args.mode == "test":
            print("[INFO] Testing images...")  # Add test logic here
        else:
            handle_error("Unsupported mode.")
    except Exception as e:
        handle_error("An unexpected error occurred", exception=e)

if __name__ == "__main__":
    main()