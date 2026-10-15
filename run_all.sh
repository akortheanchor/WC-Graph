#!/usr/bin/env bash
# Reproduce every table and figure (about 1.5-2 h on 2 CPU cores, no GPU).
set -euo pipefail
cd "$(dirname "$0")/experiments"
python run_complexity.py
python run_main.py --reps 100 --seed-reps 50
python run_sweeps.py --reps 50
python run_weight_stability.py
python run_synthetic.py --reps 100 --envs 3
python analyze.py
python make_figures.py
cd .. && python -m pytest -q
