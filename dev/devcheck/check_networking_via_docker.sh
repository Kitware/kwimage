#!/bin/bash
__doc__="
Test what fails when networking is disabled
"

cd "$HOME"/code/kwimage
xdev freshpyenv

# ---- Inside Docker

# Install the package
pip install -e .[all,headless] -v

# ---- Outside Docker

# List networks
docker network ls

# Find the container ID
docker ps
export CONTAINER_ID=35abe4082ff6

# List networks used by container
docker inspect $CONTAINER_ID
docker inspect $CONTAINER_ID | jq '.[0].NetworkSettings.Networks | keys'

docker network disconnect bridge $CONTAINER_ID
docker network connect bridge $CONTAINER_ID

