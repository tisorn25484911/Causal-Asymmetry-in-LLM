"""Execute section 7 alone (setup + its cells) into the scratchpad, on the runs found under ROOT with STEPS updates
(the tiny smoke runs while the real ones train).  Usage: python build_section7_test.py [ROOT STEPS]"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import new_processes_section as S                                          # noqa: E402

ROOT = sys.argv[1] if len(sys.argv) > 1 else None
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 50_000
OUT = HERE.parent / "new_processes" / "section7_test.ipynb"
TM = "/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models"

cells = [nbformat.v4.new_code_cell(f"""
import sys
from pathlib import Path
sys.path.insert(0, "{TM}")
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import process as P
import training as TR
import analysis as A
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
""".strip())]
for kind, text in S.CELLS:
    if kind == "code":
        text = text.replace("NEW_ROOT, NEW_STEPS = None, 50_000", f"NEW_ROOT, NEW_STEPS = {ROOT!r}, {STEPS}")
    cells.append(nbformat.v4.new_code_cell(text) if kind == "code" else nbformat.v4.new_markdown_cell(text))
for text in getattr(S, "FINDINGS", []):
    cells.append(nbformat.v4.new_markdown_cell(text))
nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "qdrug", "language": "python",
                                                                     "name": "python3"}})
start = time.time()
try:
    NotebookClient(nb, kernel_name="python3", timeout=7200, resources={"metadata": {"path": TM}}).execute()
finally:
    nbformat.write(nb, OUT)
print(f"executed {OUT} in {time.time() - start:.0f} s")
