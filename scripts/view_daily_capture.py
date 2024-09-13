import numpy as np
import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory of the script directory
parent_dir = os.path.dirname(script_dir)

# Set the workspace path
workspace_path = parent_dir

print(f'Workspace path: {workspace_path}')
sys.path.append(workspace_path)
os.chdir(workspace_path)

from lib.opencv_video_utils import videoPlayerVisThermal
import argparse

daily_capture_data_path = './data/daily_capture/'

# Get from and to date time input and only show the frames between these two date times. The files are stored in the format '%Y-%m-%d_%H-%M-%S'
parser = argparse.ArgumentParser(description='View daily capture data')
parser.add_argument('--from-date', type=str, help='From date in the format %Y-%m-%d %H-%M-%S')
parser.add_argument('--to-date', type=str, help='To date in the format %Y-%m-%d %H:%M:%S')
args = parser.parse_args()

from_date = np.datetime64(args.from_date) if args.from_date is not None else None
to_date = np.datetime64(args.to_date) if args.to_date is not None else None

print(f'From date: {from_date}, To date: {to_date}')

# Find all the npz files in the daily_capture_data_path directory
npz_files = glob.glob(daily_capture_data_path + '*.npz')
print(f'Found {len(npz_files)} npz files in {daily_capture_data_path} directory.')

def convert_filename_to_datetime(filename):
    filename = filename.split('/')[-1].split('.')[0].split('_')
    filename[1] = filename[1].replace('-', ':')
    return np.datetime64(' '.join(filename))

# Filter the npz files based on the from and to date. The file names are in the format '%Y-%m-%d_%H-%M-%S.npz'
if from_date is not None:
    npz_files = [file for file in npz_files if convert_filename_to_datetime(file) >= from_date]
if to_date is not None:
    npz_files = [file for file in npz_files if convert_filename_to_datetime(file) <= to_date]

print(f'Found {len(npz_files)} npz files between {from_date} and {to_date}.')

# Sort the npz files by date and time
npz_files.sort()
N_frames = len(npz_files)

vis_thermal_H_path = os.path.join(workspace_path, 'data', 'misc')
# List all files that start with 'vis_thermal_H' in the vis_thermal_H_path directory
vis_thermal_H_files = glob.glob(vis_thermal_H_path + '/vis_thermal_H*')
# Order them by date and time
vis_thermal_H_files.sort()
# Take the last file in the list
vis_thermal_H_file = vis_thermal_H_files[-1]
print(f'Using homography matrix from {vis_thermal_H_file}')
estimated_H_path = vis_thermal_H_file
H_thr2vis = np.load(estimated_H_path)['H_boson2bfly']



def get_vis_thermal_img_func(frame_number, return_datetime=False):
    data = np.load(npz_files[frame_number])
    vis_img = data['vis_img']
    thermal_img = data['thermal_img']
    if return_datetime:
        return vis_img, thermal_img, convert_filename_to_datetime(npz_files[frame_number])
    return vis_img, thermal_img

def get_vis_img_func(frame_number):
    data = np.load(npz_files[frame_number])
    return data['vis_img']

def get_thermal_img_func(frame_number):
    data = np.load(npz_files[frame_number])
    thermal_img = data['thermal_img']
    return thermal_img

@classmethod
def get_img_fname(cls, data_counter):
    return npz_files[data_counter].split('/')[-1]

videoPlayerVisThermal.get_img_fname = get_img_fname
    
player = videoPlayerVisThermal(N_frames, get_vis_thermal_img_func, thr2Vis_HMatrix=H_thr2vis, workspace_path=workspace_path)
# Print the description of loop_control function in player
print(player.loop_control.__doc__)
player.play_video(show_frame_number=True)
plt.close('all')