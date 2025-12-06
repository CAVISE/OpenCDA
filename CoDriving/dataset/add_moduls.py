import sys
import os

rel_path = '../'
lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), rel_path))

if lib_path not in sys.path:
  sys.path.insert(0, lib_path)
