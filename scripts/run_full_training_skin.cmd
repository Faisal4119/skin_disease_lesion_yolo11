@echo off
rem Stage 6 - full training, both models, sequential, detached from any terminal.
rem Runs against the project-local `skin` uv venv (torch 2.14.0+cu132, ultralytics 8.4.156)
rem instead of whatever `python` happens to be on PATH.
rem
rem Recreate that venv from scratch with:
rem   uv venv skin --python 3.12
rem   uv pip install --python skin\Scripts\python.exe --torch-backend=auto -r requirements-skin.txt
rem
rem Logs:        logs\06_train_disease.log, logs\06_train_malignancy.log
rem Done marker: logs\06_train_ALL_DONE.flag
cd /d D:\personal_project\yolo_model\datasheet

set PY=skin\Scripts\python.exe
if not exist "%PY%" (
    echo ERROR: %PY% not found. Create the venv first - see the header of this file.
    exit /b 1
)

rem Refuse to start a second trainer: two runs on one 8 GB card thrash VRAM and both stall.
tasklist /fi "imagename eq python.exe" /v /fo csv | findstr /i "06_train.py" >nul
if not errorlevel 1 (
    echo ERROR: a 06_train.py process is already running. Let it finish or stop it first.
    exit /b 1
)

del /q logs\06_train_ALL_DONE.flag 2>nul
echo %date% %time% START disease > logs\06_train_progress.log
"%PY%" -u scripts\06_train.py --task disease > logs\06_train_disease.log 2>&1
echo %date% %time% END disease exit=%errorlevel% >> logs\06_train_progress.log
echo %date% %time% START malignancy >> logs\06_train_progress.log
"%PY%" -u scripts\06_train.py --task malignancy > logs\06_train_malignancy.log 2>&1
echo %date% %time% END malignancy exit=%errorlevel% >> logs\06_train_progress.log
echo DONE > logs\06_train_ALL_DONE.flag
