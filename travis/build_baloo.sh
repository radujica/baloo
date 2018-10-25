#!/bin/bash

export WELD_HOME="$TRAVIS_BUILD_DIR/weld"
pip install "$TRAVIS_BUILD_DIR/weld/python/pyweld"
python setup.py install
