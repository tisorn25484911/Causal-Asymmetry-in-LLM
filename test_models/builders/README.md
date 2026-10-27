# builders

The scripts that wrote parts of this folder (copied from the session scratchpad on 2026-10-26; paths inside them may
still point to that scratchpad):

* `build_walkthrough.py` builds and executes `../walkthrough.ipynb` from `walkthrough_findings.py` (section 6) and
  `new_processes_section.py` (section 7). `build_section7_test.py` executes section 7 alone.
* `build_models.py` writes the hybrid `model_<name>.py` files from shared blocks (it overwrites them).
* `build_process.py` writes `../process.py` from the repository's process files and `process_extras.py`.
* `progress.py` prints the progress of section 7's runs; `cache_new_runs.py` caches their reports as they finish.
* `findings_numbers.py` prints every number section 7's findings quote, from the cached reports.
