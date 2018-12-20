import os

import tabulate

# While not explicit, the code here gets executed on baloo import because of pyweld and convertors importing from here
tabulate.PRESERVE_WHITESPACE = True

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LIBS_DIR = ROOT_DIR + '/weld/libs'
# TODO: If adding support for Windows/MAC, should check here file extension (check history of bindings.py)
WELD_PATH = os.path.join(LIBS_DIR, 'libweld.so')
ENCODERS_PATH = os.path.join(LIBS_DIR, 'numpy_weld_convertor.so')
