@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set EXCLUDE_PROXY=1
set PARAMS_NAME=params_noproxy.json
python src\synth.py
set RESULTS3_NAME=results3_noproxy.json
python src\exp3_discrimination_criteria.py
set EXCLUDE_PROXY=
set PARAMS_NAME=
set RESULTS3_NAME=
python src\synth.py || exit /b 1
python src\exp3_discrimination_criteria.py || exit /b 1
python src\exp1_criteria_verification.py || exit /b 1
python src\exp2_camera_requirements.py || exit /b 1
python src\make_figures.py || exit /b 1
python src\make_generation_figure.py || exit /b 1
python src\make_main_figures.py || exit /b 1
python src\make_block_diagrams.py || exit /b 1
python src\make_overview_figure.py || exit /b 1
