"""
Make the split source tree importable for pytest.

REORGANISATION_FIX_PLAN.md 5.2.  The modules in this repo are flat -- they import
each other as `from utils import ...`, not `from Transformer_model.utils import
...` -- but they live in two sibling directories.  Python only ever puts the
*entry script's own* directory on sys.path, and pytest imports test files rather
than running them as scripts, so neither directory ends up importable on its own.

Putting the bootstrap here rather than only in tests/test_theory.py means it
applies to the whole session, once, and to any test file added later.  pytest
finds conftest.py by rootdir, so this also makes `pytest` work from any cwd.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for _d in ("Transformer_model", "Experimental_setup"):
    _p = os.path.join(ROOT, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
