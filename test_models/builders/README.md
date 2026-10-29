# builders

The scripts that wrote parts of this folder (copied from the session scratchpad on 2026-10-26; paths inside them may
still point to that scratchpad):

* `build_walkthrough.py` builds and executes `../walkthrough.ipynb` from `walkthrough_findings.py` (section 6) and
  `new_processes_section.py` (sections 7–9). `build_section7_test.py` executes sections 7–9 alone.
* `build_models.py` writes the hybrid `model_<name>.py` files from shared blocks (it overwrites them).
* `build_process.py` writes `../process.py` from the repository's process files and `process_extras.py`.
* `progress.py` prints the progress of sections 7–8's runs; `cache_new_runs.py <folders file>` caches the reports of a
  plan's runs as they finish (`plans/new_processes_folders.json`, `plans/flower_20_30_folders.json`).
* `findings_numbers.py` prints every number section 7's findings quote, from the cached reports.
* `de_bruijn_section.py` holds section 10 (the de Bruijn hybrid); `de_bruijn_configs.py` its processes and settings;
  `build_de_bruijn_bic.py` writes `../model_de_bruijn_bic.py` from `../model_de_bruijn.py`.
