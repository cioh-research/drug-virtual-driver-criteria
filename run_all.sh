#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
EXCLUDE_PROXY=1 PARAMS_NAME=params_noproxy.json python src/synth.py
PARAMS_NAME=params_noproxy.json RESULTS3_NAME=results3_noproxy.json python src/exp3_discrimination_criteria.py
python src/synth.py
python src/exp3_discrimination_criteria.py
python src/exp1_criteria_verification.py
python src/exp2_camera_requirements.py
python src/make_figures.py
python src/make_generation_figure.py
python src/make_main_figures.py
python src/make_block_diagrams.py
python src/make_overview_figure.py
