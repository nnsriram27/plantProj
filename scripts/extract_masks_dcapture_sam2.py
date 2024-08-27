import numpy as np
import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm


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
checkpoints_path = os.path.join(workspace_path, 'checkpoints')

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


### CUDA setup ####

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
#####

# Load the SAM2 model
sam2_checkpoint = os.path.join(checkpoints_path, "sam2_hiera_large.pt")
model_cfg = "sam2_hiera_l.yaml"

# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# # mask_generator = SAM2AutomaticMaskGenerator(sam2)
# mask_generator = SAM2AutomaticMaskGenerator(
#     model=sam2,
#     points_per_side=64,
#     points_per_batch=128,
#     pred_iou_thresh=0.7,
#     stability_score_thresh=0.92,
#     stability_score_offset=0.7,
#     crop_n_layers=1,
#     box_nms_thresh=0.7,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=25.0,
#     use_m2m=True,
# )


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

print("SAM2 model loaded successfully.")

@classmethod
def run_auto_img_mask_generator(cls, img):
    # Run the automatic mask generator
    masks = mask_generator.generate(img)
    return masks

@classmethod
def get_img_fname(cls, data_counter):
    return npz_files[data_counter].split('/')[-1]

inference_state = None

@classmethod
def initialize_segmentation_predictor(cls):
    global inference_state, predictor
    inference_state = predictor.init_state(video_path=os.path.join(workspace_path, 'data', 'tmp'))

@classmethod
def add_anns_to_predictor(cls, ann_frame_idx, ann_obj_id, video_ann_pts=None, video_ann_pts_label=None, box_pts = None):
    global inference_state, predictor
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=np.array(video_ann_pts),
        labels=np.array(video_ann_pts_label),
        box=np.array(box_pts),
    )
    mask_logits_np = out_mask_logits.cpu().numpy()
    return out_obj_ids, mask_logits_np

@classmethod
def add_box_points_to_all_frames(cls, box_pts, ann_obj_id, skip_frames_every=5):
    global inference_state, predictor
    print(f'Adding box points to all frames for object id {ann_obj_id}')
    count = 0
    for i in tqdm(range(0, N_frames, skip_frames_every)):
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=count,
            obj_id=ann_obj_id,
            points=None,
            labels=None,
            box=np.array(box_pts).flatten(),
        )
        count += 1
    return None

@classmethod
def propagate_anns_across_frames(cls):
    global inference_state, predictor
    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = np.zeros_like((out_mask_logits[0] > 0.0).cpu().numpy()[0]).astype(int)
        for i, out_obj_id in enumerate(out_obj_ids):
            video_segments[out_frame_idx] += (out_mask_logits[i] > 0.0).cpu().numpy()[0] * out_obj_id
    return video_segments

@classmethod
def reset_inference_state(cls):
    global inference_state, predictor
    predictor.reset_state(inference_state)


videoPlayerVisThermal.run_auto_img_mask_generator = run_auto_img_mask_generator
videoPlayerVisThermal.get_img_fname = get_img_fname
videoPlayerVisThermal.initialize_segmentation_predictor = initialize_segmentation_predictor
videoPlayerVisThermal.add_anns_to_predictor = add_anns_to_predictor
videoPlayerVisThermal.propagate_anns_across_frames = propagate_anns_across_frames
videoPlayerVisThermal.reset_inference_state = reset_inference_state
videoPlayerVisThermal.add_box_points_to_all_frames = add_box_points_to_all_frames

player = videoPlayerVisThermal(N_frames, get_vis_thermal_img_func, thr2Vis_HMatrix=H_thr2vis, workspace_path=workspace_path)

# Print the description of loop_control function in player
print(player.loop_control.__doc__)
player.play_video(show_frame_number=True)
plt.close('all')