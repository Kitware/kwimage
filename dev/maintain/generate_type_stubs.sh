#!/bin/bash
__doc__="
Helper to generate type stubs
"

cd "$HOME"/code/kwimage

xdev doctypes ./kwimage

mypy ./kwimage
