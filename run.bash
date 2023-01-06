#!/bin/sh
cd /home/ubuntu/ocr_discharge_paper/ppy_paddle
conda activate py39
uvicorn main:app --host 0.0.0.0 > console_log.txt & disown
# uvicorn main:app --host 0.0.0.0