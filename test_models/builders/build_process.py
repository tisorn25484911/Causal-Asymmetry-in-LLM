"""Build test_models/process.py: HMM_processes.py and process_generator.py copied verbatim into one file (the only
change: process_generator's `from HMM_processes import (...)` is dropped, since both now live in the same module),
followed by process_extras.py."""
import re
from pathlib import Path
SRC = Path("/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/model_modeularised")
OUT = Path("/Users/tisornnaphattalung/Desktop/Quantum/URECA/LLM_final_version/test_models/process.py")
EXTRAS = Path(__file__).with_name("process_extras.py")
hmm = (SRC / "HMM_processes.py").read_text()
gen = (SRC / "process_generator.py").read_text()
gen_new, n = re.subn(r"from HMM_processes import \([^)]*\)\n", "", gen)
assert n == 1, "expected exactly one import of HMM_processes in process_generator.py"
header = '''"""
process.py -- every process of interest, its generator, and its analytical results, in one file.

Copied on 2026-10-26 from model_modeularised/:
    part 1  HMM_processes.py      verbatim
    part 2  process_generator.py  verbatim, except its `from HMM_processes import (...)` line (both are in this file)
    part 3  added for test_models: `Process`, one process with its parameters, and `WindowSampler`, the fresh
            training windows of section 16 (checks/section16_experiments.Sampler), for any process.

The original docstring of HMM_processes.py follows.
"""
'''
body = hmm.split('"""', 2)
assert body[0] == "" , "HMM_processes.py should start with its docstring"
text = (header + '"""' + body[1] + '"""' + body[2].rstrip() + "\n\n\n"
        + "#" * 110 + "\n# PART 2: process_generator.py (verbatim; its import of HMM_processes removed)\n" + "#" * 110 + "\n"
        + gen_new.rstrip() + "\n\n\n"
        + "#" * 110 + "\n# PART 3: added for test_models\n" + "#" * 110 + "\n"
        + EXTRAS.read_text())
OUT.write_text(text)
print("wrote", OUT, len(text.splitlines()), "lines")
