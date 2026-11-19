@echo off
rem Stage 6 — full training, both models, sequential. Detached from any terminal.
rem Logs: logs\06_train_disease.log, logs\06_train_malignancy.log
rem Done marker: logs\06_train_ALL_DONE.flag
cd /d D:\personal_project\yolo_model\datasheet
echo %date% %time% START disease > logs\06_train_progress.log
python -u scripts\06_train.py --task disease > logs\06_train_disease.log 2>&1
echo %date% %time% END disease exit=%errorlevel% >> logs\06_train_progress.log
echo %date% %time% START malignancy >> logs\06_train_progress.log
python -u scripts\06_train.py --task malignancy > logs\06_train_malignancy.log 2>&1
echo %date% %time% END malignancy exit=%errorlevel% >> logs\06_train_progress.log
echo DONE > logs\06_train_ALL_DONE.flag
