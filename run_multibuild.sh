#!/bin/bash

DOCKER_IMAGE=${DOCKER_IMAGE:="quay.io/pypa/manylinux2010_x86_64"}
#PARENT_USER=${PARENT_USER:="$USER"}

# Valid multibuild python versions are:
# cp27-cp27m  cp27-cp27mu  cp34-cp34m  cp35-cp35m  cp36-cp36m  cp37-cp37m
MB_PYTHON_VERSION=${MB_PYTHON_VERSION:="cp36-cp36m"}


if [[ "$_INSIDE_DOCKER" != "YES" ]]; then
    docker run --rm \
        -v $PWD:/io \
        -e _INSIDE_DOCKER="YES" \
        -e MB_PYTHON_VERSION="$MB_PYTHON_VERSION" \
        $DOCKER_IMAGE /io/run_multibuild.sh

    __interactive__='''
    docker run --rm \
        -v $PWD:/io \
        -e _INSIDE_DOCKER="YES" \
        -e MB_PYTHON_VERSION="$MB_PYTHON_VERSION" \
        -it $DOCKER_IMAGE bash

    set +e
    set +x
    '''

    exit 0;
fi


set -x
set -e

cd /io
ls

PYPREFIX=/opt/python/$MB_PYTHON_VERSION
PYEXE=${PYPREFIX}/bin/python
VENV_DIR=venv_$MB_PYTHON_VERSION

chmod -R 777 $VENV_DIR
echo "VENV_DIR = $VENV_DIR"

$PYEXE --version  # Print out python version for debugging
$PYEXE -m pip install virtualenv
$PYEXE -m virtualenv $VENV_DIR


#set +x
echo "activate virtualenv"
source $VENV_DIR/bin/activate
echo "activated virtualenv"
#set -x

PIP_CACHE_DIR="$VENV_DIR/.cache/pip"

pip install pip -U
pip install pip setuptools -U
pip install -r requirements.txt
chmod -R 777 $VENV_DIR

# virtualenv doesn't correctly set library_path and ld_library_path
# ACTUALLY: we shouldnt be linking against libpython anyway
#LD_LIBRARY_PATH=$VENV_DIR/lib:$LD_LIBRARY_PATH 
#LIBRARY_PATH=$VENV_DIR/lib:$LIBRARY_PATH
#CPATH=$VENV_DIR/include:$LD_LIBRARY_PATH 
#LD_LIBRARY_PATH=$PYPREFIX/lib:$LD_LIBRARY_PATH 
#LIBRARY_PATH=$PYPREFIX/lib:$LIBRARY_PATH
#CPATH=$PYPREFIX/include:$LD_LIBRARY_PATH 
python setup.py bdist_wheel
chmod -R 777 _skbuild
chmod -R 777 dist

auditwheel repair dist/kwimage-0.5.0-$MB_PYTHON_VERSION-linux_x86_64.whl
chmod -R 777 wheelhouse


#echo "DOCKER_IMAGE = $DOCKER_IMAGE"
#-e MB_PYTHON_VERSION="$MB_PYTHON_VERSION" \
#-e UNICODE_WIDTH="$UNICODE_WIDTH" \
#-e CONFIG_PATH="$CONFIG_PATH" \
#-e WHEEL_SDIR="$WHEEL_SDIR" \
#-e MANYLINUX_URL="$MANYLINUX_URL" \
#-e TEST_DEPENDS="$TEST_DEPENDS" \


