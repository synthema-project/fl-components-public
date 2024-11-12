#!/bin/bash

set -e

# Check if the remote repository, tag, username, and password arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <remote-repo> <tag> <username> <password>"
    exit 1
fi

# Define the remote repository, tag, username, and password
remote_repo=$1
tag=$2


# Define an array of images with their repositories
images=(
    "synthema-fl-client"
    "synthema-fl-server"
    "synthema-fl-restapi"
)

# Loop through the array and push each image with the specified tag
for image in "${images[@]}"; do
    docker push "${remote_repo}${image}:${tag}"
done