#!/bin/bash
__doc__="
Test what fails when networking is disabled
"

cd "$HOME"/code/kwimage
xdev freshpyenv

# ---- Inside Docker

# Install the package
pip install -e .[all] -v

# ---- Outside Docker

# List networks
docker network ls

# Find the container ID
docker ps
export CONTAINER_ID=4d0bc4c5cbab

# List networks used by container
docker inspect $CONTAINER_ID
docker inspect $CONTAINER_ID | jq '.[0].NetworkSettings.Networks | keys'

docker network disconnect bridge 4d0bc4c5cbab
docker network connect bridge 4d0bc4c5cbab

