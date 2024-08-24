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
parser.add_argument('--from_date', type=str, help='From date in the format %Y-%m-%d %H-%M-%S')
parser.add_argument('--to_date', type=str, help='To date in the format %Y-%m-%d %H:%M:%S')
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

def read_pixel_values_for_allfiles(selected_pix_vis, selected_pix_thermal):
    pix_value_vis = []
    pix_value_thermal = []
    pix_datetime = []
    selected_pix_vis = np.array(selected_pix_vis)
    selected_pix_thermal = np.array(selected_pix_thermal)
    print(selected_pix_vis.shape, selected_pix_thermal.shape)
    for i in range(N_frames):
        try:
            vis_img, thermal_img, img_datetime = get_vis_thermal_img_func(i, return_datetime=True)
            pix_datetime.append(img_datetime)
            if vis_img is not None:
                pix_value_vis.append(vis_img[selected_pix_vis[:, 1],selected_pix_vis[:, 0]])
            if thermal_img is not None:
                pix_value_thermal.append(thermal_img[selected_pix_thermal[:, 1],selected_pix_thermal[:, 0]])
        except Exception as e:
            print(f'Error reading pixel values for frame {i}: {e}')
            # Make sure the length of pix_value_vis and pix_value_thermal are the same
            if len(pix_value_vis) != len(pix_value_thermal):
                pix_value_vis = pix_value_vis[:min(len(pix_value_vis), len(pix_value_thermal))]
                pix_value_thermal = pix_value_thermal[:min(len(pix_value_vis), len(pix_value_thermal))]
            print(thermal_img.shape, vis_img.shape)
    return np.stack(pix_value_vis), np.stack(pix_value_thermal), np.stack(pix_datetime)

# def plot_pixel_values_across_time(cls, selected_pix_vis, selected_pix_thermal):
#     pix_value_vis, pix_value_thermal, pix_datetime = read_pixel_values_for_allfiles(selected_pix_vis, selected_pix_thermal)
#     print(pix_value_vis.shape, pix_value_thermal.shape, pix_datetime.shape)
    
#     cls.pixel_plot_fig = plt.figure(figsize=(20, 10))
    
#     if pix_value_vis.size != 0:
#         ax1 = cls.pixel_plot_fig.add_subplot(211)
#         for i in range(pix_value_vis.shape[1]):
#             line, = ax1.plot(pix_datetime, pix_value_vis[:, i], label=f'Pixel {selected_pix_thermal[i]}')
#             # Adding hover functionality
#             mplcursors.cursor(line, hover=True)
#         ax1.set_title('Visible Camera Pixel Values')
#         ax1.set_xlabel('Date Time')
#         ax1.set_ylabel('Pixel Value')
#         ax1.legend()
    
#     if pix_value_thermal.size != 0:
#         ax2 = cls.pixel_plot_fig.add_subplot(212)
#         for i in range(pix_value_thermal.shape[1]):
#             line, = ax2.plot(pix_datetime, pix_value_thermal[:, i], label=f'Pixel {selected_pix_thermal[i]}')
#             # Adding hover functionality
#             mplcursors.cursor(line, hover=True)
#         ax2.set_title('Thermal Camera Pixel Values')
#         ax2.set_xlabel('Date Time')
#         ax2.set_ylabel('Pixel Value')
#         ax2.legend()
    
#     # Enable interactive mode
#     plt.ion()
#     plt.show()

#     # Add a pause to keep the plot window open (optional)
#     input("Press Enter to continue...")

#     # Turn off interactive mode
#     plt.ioff()


def plot_pixel_values_across_time(cls, selected_pix_vis, selected_pix_thermal):
    pix_value_vis, pix_value_thermal, pix_datetime = read_pixel_values_for_allfiles(selected_pix_vis, selected_pix_thermal)
    print(pix_value_vis.shape, pix_value_thermal.shape, pix_datetime.shape)

    if cls.pixel_plot_fig is None:
        # Create new subplots if figure doesn't exist
        cls.pixel_plot_fig = make_subplots(rows=2, cols=1, subplot_titles=('Visible Camera Pixel Values', 'Thermal Camera Pixel Values'))

    # Add traces for visible camera pixel values
    if pix_value_vis.size != 0:
        for i in range(cls.pixel_plot_counter, pix_value_vis.shape[1]):
            cls.pixel_plot_fig.add_trace(
                go.Scatter(x=pix_datetime, y=pix_value_vis[:, i], name=f'Pixel {selected_pix_vis[i]}',
                           hovertemplate='<b>Date Time</b>: %{x}<br>' +
                                         '<b>Pixel Value</b>: %{y}<br>' +
                                         '<b>Pixel</b>: ' + str(selected_pix_vis[i])),
                row=1, col=1
            )
    
    # Add traces for thermal camera pixel values
    if pix_value_thermal.size != 0:
        for i in range(cls.pixel_plot_counter, pix_value_thermal.shape[1]):
            cls.pixel_plot_fig.add_trace(
                go.Scatter(x=pix_datetime, y=pix_value_thermal[:, i], name=f'Pixel {selected_pix_thermal[i]}',
                           hovertemplate='<b>Date Time</b>: %{x}<br>' +
                                         '<b>Pixel Value</b>: %{y}<br>' +
                                         '<b>Pixel</b>: ' + str(selected_pix_thermal[i])),
                row=2, col=1
            )
    cls.pixel_plot_counter = max(pix_value_vis.shape[1], pix_value_thermal.shape[1])
    
    # Update layout
    cls.pixel_plot_fig.update_layout(
        height=800,
        width=1200,
        title_text="Pixel Values Across Time",
        hovermode="x unified"
    )
    
    # Update x and y axis labels
    cls.pixel_plot_fig.update_xaxes(title_text="Date Time", row=1, col=1)
    cls.pixel_plot_fig.update_xaxes(title_text="Date Time", row=2, col=1)
    cls.pixel_plot_fig.update_yaxes(title_text="Pixel Value", row=1, col=1)
    cls.pixel_plot_fig.update_yaxes(title_text="Pixel Value", row=2, col=1)
    
    # Show the plot
    cls.pixel_plot_fig.show()


videoPlayerVisThermal.plot_pixel_values_across_time = plot_pixel_values_across_time
    
player = videoPlayerVisThermal(N_frames, get_vis_thermal_img_func, thr2Vis_HMatrix=H_thr2vis, workspace_path=workspace_path)
player.play_video(show_frame_number=True)
plt.close('all')