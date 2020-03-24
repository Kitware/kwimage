#!/bin/bash 

# Install dependency packages
pip install -r requirements.txt

# Install in developer mode

# For some reason there is a bug with using pip and skbuild 
# Calling setup.py directly seems to work though
python setup.py clean
#python setup.py develop
pip install -e .
