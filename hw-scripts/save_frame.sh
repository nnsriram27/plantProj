#!/bin/bash

# Activate Conda environment and run Python script
source /home/ilimnuc/miniconda3/bin/activate plantEnv
python /home/ilimnuc/plantProj/hw-scripts/save_frame.py >> /home/ilimnuc/plantProj/logs/save_frame.log 2>&1