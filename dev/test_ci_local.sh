
# Commands to help developers debug pipelines on their local machine
# Grab the base docker image, (forwarding your ssh credentials), clone
# the watch repo, create the environment, and run the tests. 
#docker login gitlab.kitware.com:4567
#IMAGE_NAME=gitlab.kitware.com:4567/computer-vision/ci-docker/gl-python:3.8

# Use whatever image is defined for Python39 in this file and start a docker session
IMAGE_NAME=$(cat .gitlab-ci.yml | yq -r '.".image_python3_10"')
docker run -v "$PWD":/io:ro -v "$HOME"/.cache/pip:/pip_cache -it "$IMAGE_NAME" bash

# Will need to chmod things afterwords
export PIP_CACHE_DIR=/pip_cache
echo $PIP_CACHE_DIR
chmod -R o+rw $PIP_CACHE_DIR
chmod -R o+rw $PIP_CACHE_DIR
chmod -R g+rw $PIP_CACHE_DIR
USER=$(whoami)
chown -R "$USER" $PIP_CACHE_DIR
cd "$HOME"
git clone /io ./repo

cd "$HOME"/repo

# Make a virtualenv
export PYVER=$(python -c "import sys; print('{}{}'.format(*sys.version_info[0:2]))")
pip install virtualenv
virtualenv venv"$PYVER"
source venv"$PYVER"/bin/activate
#pip install pip -U
#pip install pip setuptools -U

# FULL STRICT VARIANT
pip install -e .[all-strict,headless-strict]
./run_tests.py

# FULL LOOSE VARIANT
pip install -e .[all,headless]
./run_tests.py

# MINIMAL STRICT VARIANT
pip install -e .[runtime-strict,tests-strict]
./run_tests.py

# MINIMAL LOOSE VARIANT
pip install -e .[tests]
./run_tests.py
