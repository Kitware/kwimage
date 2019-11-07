#!/bin/bash
__heredoc__="""


notes:

    # TODO: use dind as the base image,
    # Then run the multibuild in docker followed by a test in a different
    # docker container
    docker run --rm -it --entrypoint="" docker:dind sh

    docker run --rm -v $PWD:/io -it python:2.7 bash
     
        cd /io
        pip install -r requirements.txt
        pip install pygments
        pip install wheelhouse/kwimage-0.5.0-cp27-cp27mu-manylinux1_x86_64.whl

        cd /
        xdoctest kwimage
        pytest io/tests

        cd /io
        python run_tests.py


MB_PYTHON_TAG=cp37-cp37m ./run_multibuild.sh

MB_PYTHON_TAG=cp36-cp36m ./run_multibuild.sh

MB_PYTHON_TAG=cp35-cp35m ./run_multibuild.sh

MB_PYTHON_TAG=cp27-cp27m ./run_multibuild.sh

# MB_PYTHON_TAG=cp27-cp27mu ./run_nmultibuild.sh

"""


#DOCKER_IMAGE=${DOCKER_IMAGE:="soumith/manylinux-cuda100"}
DOCKER_IMAGE=${DOCKER_IMAGE:="quay.io/pypa/manylinux2010_x86_64"}
#PARENT_USER=${PARENT_USER:="$USER"}

# Valid multibuild python versions are:
# cp27-cp27m  cp27-cp27mu  cp34-cp34m  cp35-cp35m  cp36-cp36m  cp37-cp37m
MB_PYTHON_TAG=${MB_PYTHON_TAG:="cp36-cp36m"}


if [[ "$_INSIDE_DOCKER" != "YES" ]]; then
    docker run --rm \
        -v $PWD:/io \
        -e _INSIDE_DOCKER="YES" \
        -e MB_PYTHON_TAG="$MB_PYTHON_TAG" \
        $DOCKER_IMAGE /io/run_multibuild.sh

    __interactive__='''
    docker run --rm \
        -v $PWD:/io \
        -e _INSIDE_DOCKER="YES" \
        -e MB_PYTHON_TAG="$MB_PYTHON_TAG" \
        -it $DOCKER_IMAGE bash

    set +e
    set +x
    '''

    exit 0;
fi


set -x
set -e

cd /io
#ls

PYPREFIX=/opt/python/$MB_PYTHON_TAG
PYEXE=${PYPREFIX}/bin/python
VENV_DIR=venv_$MB_PYTHON_TAG

echo "VENV_DIR = $VENV_DIR"

$PYEXE --version  # Print out python version for debugging
$PYEXE -m pip install virtualenv
$PYEXE -m virtualenv $VENV_DIR

chmod -R o+rw $VENV_DIR
#setfacl -d -m g::rwx $VENV_DIR
#setfacl -d -m o::rwx $VENV_DIR

set +x
echo "activate virtualenv"
source $VENV_DIR/bin/activate
echo "activated virtualenv"
set -x

export PIP_CACHE_DIR="$VENV_DIR/cache_pip"

pip install pip -U
pip install pip setuptools -U

pip install numpy==1.15
pip install pandas==0.23.2  # hack for python2

pip install -r requirements.txt

chmod -R o+rw $VENV_DIR

# virtualenv doesn't correctly set library_path and ld_library_path
# ACTUALLY: we shouldnt be linking against libpython anyway
#LD_LIBRARY_PATH=$VENV_DIR/lib:$LD_LIBRARY_PATH 
#LIBRARY_PATH=$VENV_DIR/lib:$LIBRARY_PATH
#CPATH=$VENV_DIR/include:$LD_LIBRARY_PATH 
#LD_LIBRARY_PATH=$PYPREFIX/lib:$LD_LIBRARY_PATH 
#LIBRARY_PATH=$PYPREFIX/lib:$LIBRARY_PATH
#CPATH=$PYPREFIX/include:$LD_LIBRARY_PATH 

python setup.py bdist_wheel
chmod -R o+rw _skbuild
chmod -R o+rw dist

auditwheel repair dist/kwimage-*-$MB_PYTHON_TAG-*.whl
chmod -R o+rw wheelhouse

chmod -R o+rw kwimage.egg-info


_debug_torch_issue(){
    #pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

    # the torch python libraries require GLIB_214 or greater, which is not
    # manylinux complient. We can build without these, but we do need them to test.

    #yum install -y devtoolset-7-gcc devtoolset-7-gcc-c++ devtoolset-7-gcc-gfortran devtoolset-7-binutils
    yum install gcc -y
    yum install -y devtoolset-7

    # Remove /opt/rh/devtoolset-8/root/usr/bin
    export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib

    mkdir ~/glibc_install; cd ~/glibc_install 
    curl http://ftp.gnu.org/gnu/glibc/glibc-2.14.tar.gz > glibc-2.14.tar.gz
    tar zxvf glibc-2.14.tar.gz
    cd glibc-2.14
    mkdir build
    cd build
    CC=/usr/bin/gcc ../configure --prefix=/opt/glibc-2.14
    make -j4
    make install

    # This just causes a segfault
    export LD_LIBRARY_PATH="/opt/glibc-2.14/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python -c "import torch"


    #echo "DOCKER_IMAGE = $DOCKER_IMAGE"
    #-e MB_PYTHON_TAG="$MB_PYTHON_TAG" \
    #-e UNICODE_WIDTH="$UNICODE_WIDTH" \
    #-e CONFIG_PATH="$CONFIG_PATH" \
    #-e WHEEL_SDIR="$WHEEL_SDIR" \
    #-e MANYLINUX_URL="$MANYLINUX_URL" \
    #-e TEST_DEPENDS="$TEST_DEPENDS" \
}
