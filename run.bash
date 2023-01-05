#!/bin/sh
conda activate py39
uvicorn main:app --host 0.0.0.0