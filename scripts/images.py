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
    print(f"[ERROR] {message}")
    if exception:
        print(f"[DETAILS] {exception}")
    if exit_program:
        exit(1)


# ------------------------------------------------------------------------------
# Resolve and validate Docker build context path
# ------------------------------------------------------------------------------
def get_context_path(root_dir, image_path):
    """
    Resolve and validate the absolute path for the Docker build context.
    Args:
        root_dir (Path): Root directory where the config file is located.
        image_path (str): Relative path to the Docker build context as defined in the config.
    Returns:
        Path: Absolute path to the Docker build context.
    """
    resolved_path = root_dir / image_path
    if not resolved_path.exists():
        handle_error(f"Context path does not exist: {resolved_path}")
    return resolved_path


# ------------------------------------------------------------------------------
# Parse Arguments
# ------------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["build", "remove", "test", "tag", "untag", "push"], required=True)
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
        handle_error("Failed to load configuration file.", exception=e)


# ------------------------------------------------------------------------------
# Docker Operations
# ------------------------------------------------------------------------------
def build_image(client, image_def, image_tag, root_dir, base_image=None):
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


def tag_image(client, image_def, old_tag, new_tag):
    old_full_tag = f"{image_def['name']}:{old_tag}"
    new_full_tag = f"{image_def['name']}:{new_tag}"

    try:
        image = client.images.get(old_full_tag)
        image.tag(image_def["name"], tag=new_tag)
        print(f"[OK] Tagged image: {old_full_tag} -> {new_full_tag}")
    except ImageNotFound:
        handle_error(f"Image not found: {old_full_tag}")
    except APIError as e:
        handle_error(f"Failed to tag image: {old_full_tag}", exception=e)


def untag_image(client, image_def, tag):
    full_tag = f"{image_def['name']}:{tag}"

    try:
        client.images.remove(full_tag)
        print(f"[OK] Untagged image: {full_tag}")
    except ImageNotFound:
        print(f"[INFO] Image not found, skipping: {full_tag}")
    except APIError as e:
        handle_error(f"Failed to untag image: {full_tag}", exception=e)


def push_image(client, image_def, tag, registry):
    full_tag = f"{registry}{image_def['name']}:{tag}"

    try:
        client.images.push(full_tag)
        print(f"[OK] Pushed image: {full_tag}")
    except APIError as e:
        handle_error(f"Failed to push image: {full_tag}", exception=e)


# ------------------------------------------------------------------------------
# Main Handlers for Each Mode
# ------------------------------------------------------------------------------
def handle_build(client, config, tag, root_dir):
    if not tag:
        handle_error("--tag is required for building images.")
    base_image_def = config["images"][0]
    dependent_images = config["images"][1:]

    build_image(client, base_image_def, tag, root_dir)

    for img in dependent_images:
        build_image(client, img, tag, root_dir, base_image=base_image_def["name"])


def handle_remove(client, config):
    for img in config["images"]:
        try:
            client.images.remove(img["name"], force=True)
            print(f"[OK] Removed image: {img['name']}")
        except ImageNotFound:
            print(f"[INFO] Image not found, skipping: {img['name']}")
        except APIError as e:
            handle_error(f"Failed to remove image: {img['name']}", exception=e)


def handle_test(client, config, tag):
    if not tag:
        handle_error("--tag is required for testing.")
    for img in config["images"]:
        if img.get("test"):
            print(f"[INFO] Running tests for image: {img['name']}:{tag}")
            # Insert test logic here


def handle_tag(client, config, old_tag, new_tag):
    if not old_tag or not new_tag:
        handle_error("--old-tag and --new-tag are required for tagging.")
    for img in config["images"]:
        tag_image(client, img, old_tag, new_tag)


def handle_untag(client, config, tag):
    if not tag:
        handle_error("--tag is required for untagging.")
    for img in config["images"]:
        untag_image(client, img, tag)


def handle_push(client, config, tag, registry):
    if not tag or not registry:
        handle_error("--tag and --registry are required for pushing images.")
    for img in config["images"]:
        push_image(client, img, tag, registry)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    args = parse_args()
    config_file = Path(__file__).resolve().parent / "docker_config.json"
    config = load_config(config_file)
    root_dir = config_file.parent.parent
    client = docker.from_env()

    # Dispatcher with handlers
    handlers = {
        "build": lambda: handle_build(client, config, args.tag, root_dir),
        "remove": lambda: handle_remove(client, config),
        "test": lambda: handle_test(client, config, args.tag),
        "tag": lambda: handle_tag(client, config, args.old_tag, args.new_tag),
        "untag": lambda: handle_untag(client, config, args.tag),
        "push": lambda: handle_push(client, config, args.tag, args.registry),
    }

    try:
        handler = handlers.get(args.mode)
        if handler:
            handler()  # Call the handler with specific parameters
        else:
            handle_error("Unsupported mode.")
    except Exception as e:
        handle_error("An unexpected error occurred.", exception=e)


if __name__ == "__main__":
    main()