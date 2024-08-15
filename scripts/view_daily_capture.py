import numpy as np
import glob
import os
import sys

workspace_path = '/home/ilimnuc/plantProj/'
sys.path.append(workspace_path)
os.chdir(workspace_path)

daily_capture_data_path = './data/daily_capture/'


# Find all the npz files in the daily_capture_data_path directory
npz_files = glob.glob(daily_capture_data_path + '*.npz')

print(f'Found {len(npz_files)} npz files in {daily_capture_data_path} directory.')