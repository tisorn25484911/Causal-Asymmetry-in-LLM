import pathlib
import sys

# HMM_processes and process_generator import each other by bare name, so the
# package directory has to be on the path before the tests import either.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
