#!/bin/bash

export WELD_HOME="$TRAVIS_BUILD_DIR/weld"
pip install "$TRAVIS_BUILD_DIR/weld/python/pyweld"
make -C "$TRAVIS_BUILD_DIR/baloo/weld/convertors"
# TODO: pytest seems to look in the download folder, not installation folder, for the .so; same in doctest
# TODO shall be fixed when pypi/deployment is settled; not a problem when installed with -e IN download folder
python setup.py install

make sure we return here
cd $TRAVIS_BUILD_DIR
