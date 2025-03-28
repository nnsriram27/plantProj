import sys
sys.path.append('../plantProj')

import threading
import argparse
import json
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from cameras.wrapper_boson import BosonWithTelemetry
from cameras.wrapper_pyspin import Blackfly
from cameras.wrapper_filterstage import FilterStage
from lights.light_intensity_controller import LightIntensityController

## TODO:
# 1. Collect some flat correction frames for the thermal camera. Done
# 2. Check that the light level is constant. Done. 
# 3. Verify the Camera Response Function of the visible camera.
# 4. Use homography to consistently pick points in both visible and thermal images.
# 5. Save the data in compressed format.


def compute_flat_correction_frames(cam_obj, num_frames_to_avg=100, debug=False, out_filename=None):
    input('Place a uniform object in front of the camera and press Enter to start computing the correction term...')

    cam_obj.camera.set_ffc_manual()

    frames = []
    for _ in tqdm(range(num_frames_to_avg)):
        frame, _, _, _ = cam_obj.get_next_image()
        frames.append(frame)
    frames = np.stack(frames, axis=0)
    frame_avg = np.mean(frames, axis=0)
    correction_term = np.median(frame_avg) - frame_avg

    cam_obj.camera.set_ffc_auto()

    if debug:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(frames[-1], cmap='turbo')
        ax[0].set_title('Last Frame')
        ax[1].imshow(frames[-1] + correction_term, cmap='turbo')
        ax[1].set_title('Corrected Frame')
        ax[2].imshow(correction_term, cmap='turbo')
        ax[2].set_title('Correction Term')
        plt.tight_layout()
        plt.show()

    if out_filename is not None:
            np.save(out_filename, correction_term)
        
    return correction_term


def parse_args():
    parser = argparse.ArgumentParser(description="Collect data for a pulsetrain experiment")
    parser.add_argument("--data-slug", type=str, required=True, help="Slug for the data")
    parser.add_argument("--pulsetrain-settings", type=str, required=True, help="Path to the pulsetrain settings file")
    parser.add_argument("--compute-flat-correction", action="store_true", help="Compute the flat correction term for the thermal camera")
    parser.add_argument("--check-exposure", action="store_true", help="Check the exposure settings for the cameras")
    parser.add_argument("--auto-exposure", action="store_true", help="Use auto exposure for the visible camera")
    return parser.parse_args()

args = parse_args()
data_slug = args.data_slug

try:
    pulsetrain_settings = json.load(open(args.pulsetrain_settings, "r"))
except:
    print("Failed to load pulsetrain settings file")
    sys.exit()

# Confirm the pulsetrain settings
print("Pulsetrain settings:")
print(json.dumps(pulsetrain_settings, indent=4))
confirm = input("Are these settings correct? (y/n): ") == "y"
if not confirm:
    sys.exit()

# Confirm the data slug
print(f"Data slug: {data_slug}")
confirm = input("Is this the correct data slug? (y/n): ") == "y"
if not confirm:
    sys.exit()

## Initialize the cameras and other devices
try:
    vis_cam_obj = Blackfly()
except:
    print("Failed to connect to visible camera")
    sys.exit()
vis_cam_obj.set_fps(60)
vis_cam_obj.set_gain(0)
vis_cam_obj.set_auto_exposure(True)

try:
    thr_cam_obj = BosonWithTelemetry()
except:
    print("Failed to connect to thermal camera")
    sys.exit()
thr_cam_obj.camera.do_ffc()

try:
    light_obj = LightIntensityController()
except:
    print("Failed to connect to light controller")
    sys.exit()
light_obj.set_voltage(0)

try:
    stage_obj = FilterStage()
except:
    print("Failed to connect to filter stage")
    sys.exit()
stage_obj.set_filter_position("clear", blocking=True)

def shutdown():
    vis_cam_obj.stop()
    thr_cam_obj.stop()
    light_obj.close()
    stage_obj.close()

logdir = Path(f"./data/{data_slug}")
logdir.mkdir(exist_ok=True, parents=True)

if args.compute_flat_correction:
    output_filename = logdir / f"flat_correction_term_{time.strftime('%Y%m%d-%H%M%S')}.npy"
    thr_correction_term = compute_flat_correction_frames(thr_cam_obj, num_frames_to_avg=100, debug=True, out_filename=output_filename)
else:
    default_thr_correction_term = np.load("./data/default_thr_flat_correction.npy")

num_pulses = len(pulsetrain_settings['light_control_voltages'])
# Capture sample images with the different exposure settings and confirm they are not over or under exposed
if args.check_exposure:
    print("Checking exposure settings for the cameras")
    if args.auto_exposure:
        vis_cam_obj.set_auto_exposure_limits(100, 1000000)
        vis_cam_obj.set_auto_exposure(True)

        def get_stable_exposure_value():
            stable = False
            last_exp_val = 0
            stable_count = 0
            while not stable:
                vis_img, vis_tstamp, _, vis_exp, _ = vis_cam_obj.get_next_image()
                # if np.abs(last_exp_val - vis_exp) / last_exp_val < 0.05:
                if np.abs(last_exp_val - vis_exp) < 100:
                    stable_count += 1
                else:
                    stable_count = 0
                if stable_count > 10:
                    stable = True
                last_exp_val = vis_exp
            return last_exp_val, vis_img
    else:
        vis_cam_obj.set_auto_exposure(False)

    ## Check for SP700 position
    print("Checking exposure settings for SP700 position")
    stage_obj.set_filter_position("sp700", blocking=True)
    vis_imgs_sp700 = []
    vis_exposures_sp700 = []

    for i, exposure in enumerate(pulsetrain_settings['vis_cam_exposure_sp700']):
        print(f"Checking for pulse {i+1} of {num_pulses}")
        light_obj.set_voltage(pulsetrain_settings['light_control_voltages'][i])
        time.sleep(0.1)
        if args.auto_exposure:
            exp_val, vis_img = get_stable_exposure_value()
            vis_imgs_sp700.append(vis_img)
            vis_exposures_sp700.append(exp_val)
        else:
            vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_sp700'][i])
            vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
            vis_imgs_sp700.append(vis_img)
            vis_exposures_sp700.append(pulsetrain_settings['vis_cam_exposure_sp700'][i])
    light_obj.set_voltage(0.0)

    ## Check for LP700 position
    print("Checking exposure settings for LP700 position")
    stage_obj.set_filter_position("lp700", blocking=True)
    vis_imgs_lp700 = []
    vis_exposures_lp700 = []
    for i, exposure in enumerate(pulsetrain_settings['vis_cam_exposure_lp700']):
        print(f"Checking for pulse {i+1} of {num_pulses}")
        light_obj.set_voltage(pulsetrain_settings['light_control_voltages'][i])
        time.sleep(0.1)
        if args.auto_exposure:
            exp_val, vis_img = get_stable_exposure_value()
            vis_imgs_lp700.append(vis_img)
            vis_exposures_lp700.append(exp_val)
        else:
            vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_lp700'][i])
            vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
            vis_imgs_lp700.append(vis_img)
            vis_exposures_lp700.append(pulsetrain_settings['vis_cam_exposure_lp700'][i])
    light_obj.set_voltage(0.0)

    for i in range(len(vis_imgs_sp700)):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"Exposure Test {i+1}, SP700 exposure: {vis_exposures_sp700[i]}, LP700 exposure: {vis_exposures_lp700[i]}")
        print(f"Exposure Test {i+1}, SP700 exposure: {vis_exposures_sp700[i]}, LP700 exposure: {vis_exposures_lp700[i]}")
        ax[0].imshow(vis_imgs_sp700[i])
        num_saturated = np.sum(vis_imgs_sp700[i] >= 0.95 * 2**16)
        ax[0].set_title(f"SP700, Mean: {np.mean(vis_imgs_sp700[i]):.2f}, Std: {np.std(vis_imgs_sp700[i]):.2f}, Saturated: {num_saturated}")
        print(f"SP700, Mean: {np.mean(vis_imgs_sp700[i]):.2f}, Std: {np.std(vis_imgs_sp700[i]):.2f}, Saturated: {num_saturated}, Min: {np.min(vis_imgs_sp700[i]):.2f}, Max: {np.max(vis_imgs_sp700[i]):.2f}")
        ax[0].axis("off")
        ax[1].imshow(vis_imgs_lp700[i])
        num_saturated = np.sum(vis_imgs_lp700[i] >= 0.95 * 2**16)
        ax[1].set_title(f"LP700, Mean: {np.mean(vis_imgs_lp700[i]):.2f}, Std: {np.std(vis_imgs_lp700[i]):.2f}, Saturated: {num_saturated}")
        print(f"LP700, Mean: {np.mean(vis_imgs_lp700[i]):.2f}, Std: {np.std(vis_imgs_lp700[i]):.2f}, Saturated: {num_saturated}, Min: {np.min(vis_imgs_lp700[i]):.2f}, Max: {np.max(vis_imgs_lp700[i]):.2f}")
        ax[1].axis("off")
        plt.show()
        confirm = input("Are these images correctly exposed? (y/n): ") == "y"
        if not confirm:
            shutdown()
            sys.exit()

    print(f"Images are correctly exposed. Now select some pixels of interest")
    continue_with_experiment = input("Continue with the experiment? (y/n): ") == "y"
    if not continue_with_experiment:
        shutdown()
        sys.exit()
else:
    print(f"Skipping exposure check.  Now select some pixels of interest")

vis_cam_obj.set_auto_exposure(True)
vis_filter_to_use = pulsetrain_settings['vis_filter']
stage_obj.set_filter_position(vis_filter_to_use, blocking=True)

# Select pixels of interest
vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
thr_img, thr_tstamp, _, _ = thr_cam_obj.get_next_image(hflip=True)

vis_selected_pixels = []
thr_selected_pixels = []

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Select pixels of interest")

def update_plot():
    ax[0].clear()
    ax[1].clear()
    ax[0].imshow(vis_img)
    ax[0].set_title("Visible Image")
    ax[0].axis("off")
    ax[1].imshow(thr_img)
    ax[1].set_title("Thermal Image")
    ax[1].axis("off")
    for i, pixel in enumerate(vis_selected_pixels):
        ax[0].scatter(pixel[0], pixel[1], color=f"C{i}")
    for i, pixel in enumerate(thr_selected_pixels):
        ax[1].scatter(pixel[0], pixel[1], color=f"C{i}")
    plt.draw()

def pyplot_on_click(event):
    global vis_selected_pixel
    if event.inaxes is ax[0]:
        vis_selected_pixels.append((int(event.xdata), int(event.ydata)))
    elif event.inaxes is ax[1]:
        thr_selected_pixels.append((int(event.xdata), int(event.ydata)))
    update_plot()

fig.canvas.mpl_connect('button_press_event', pyplot_on_click)
update_plot()
plt.show()

confirm = input("Are these pixels correct? (y/n): ") == "y"
if not confirm:
    shutdown()
    sys.exit()

vis_cam_obj.set_auto_exposure(False)

# Enable logging acclimation response
acc_vis_frames = []
acc_vis_tstamps = []
acc_thr_frames = []
acc_thr_tstamps = []
acc_sleep_time = 1
acc_vis_pixel_vals = [[] for _ in range(len(vis_selected_pixels))]
acc_thr_pixel_vals = [[] for _ in range(len(thr_selected_pixels))]
log_acclimation = True

def vis_update_thread():
    global acc_vis_frames, acc_vis_tstamps, acc_sleep_time, acc_vis_pixel_vals, vis_selected_pixels, log_acclimation
    while log_acclimation:
        vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
        acc_vis_frames.append(vis_img)
        acc_vis_tstamps.append(vis_tstamp)
        for i, pixel in enumerate(vis_selected_pixels):
            acc_vis_pixel_vals[i].append(vis_img[pixel[1], pixel[0]])
        time.sleep(acc_sleep_time)

def thr_update_thread():
    global acc_thr_frames, acc_thr_tstamps, acc_sleep_time, acc_thr_pixel_vals, thr_selected_pixels, log_acclimation
    while log_acclimation:
        thr_img, thr_tstamp, _, _ = thr_cam_obj.get_next_image(hflip=True)
        acc_thr_frames.append(thr_img)
        acc_thr_tstamps.append(thr_tstamp)
        for i, pixel in enumerate(thr_selected_pixels):
            acc_thr_pixel_vals[i].append(thr_img[pixel[1], pixel[0]])
        time.sleep(acc_sleep_time)

vis_thread = threading.Thread(target=vis_update_thread)
vis_thread.daemon = True

thr_thread = threading.Thread(target=thr_update_thread)
thr_thread.daemon = True

vis_thread.start()
thr_thread.start()
time.sleep(2)

# Set the light and camera to nominal settings
light_obj.set_voltage(pulsetrain_settings['nominal_light_voltage'])
vis_cam_obj.set_exposure(pulsetrain_settings['nominal_exposure'])

# Wait for the leaves to regain steady state
data_stable = False
while not data_stable:
    latest_vis_idx = min(len(acc_vis_frames)-1, len(acc_vis_tstamps)-1)
    latest_thr_idx = min(len(acc_thr_frames)-1, len(acc_thr_tstamps)-1)
    latest_vis_img = acc_vis_frames[latest_vis_idx]
    latest_thr_img = acc_thr_frames[latest_thr_idx]

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(latest_vis_img, cmap="gray")
    ax[0, 0].set_title("Visible Image")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(latest_thr_img, cmap="turbo")
    ax[0, 1].set_title("Thermal Image")
    ax[0, 1].axis("off")
    for i, pixel in enumerate(vis_selected_pixels):
        ax[0, 0].scatter(pixel[0], pixel[1], color=f"C{i}")
        ax[1, 0].plot(acc_vis_tstamps - acc_vis_tstamps[0], acc_vis_pixel_vals[i], '.', label=f"Pixel {i}", color=f"C{i}")
    ax[1, 0].set_title("Visible Pixel Values")
    ax[1, 0].set_xlabel("Timestamp")
    ax[1, 0].set_ylabel("Pixel Value")
    for i, pixel in enumerate(thr_selected_pixels):
        ax[0, 1].scatter(pixel[0], pixel[1], color=f"C{i}")
        ax[1, 1].plot(acc_thr_tstamps - acc_thr_tstamps[0], acc_thr_pixel_vals[i], '.', label=f"Pixel {i}", color=f"C{i}")
    ax[1, 1].set_title("Thermal Pixel Values")
    ax[1, 1].set_xlabel("Timestamp")
    ax[1, 1].set_ylabel("Pixel Value")
    ax[1, 0].grid()
    ax[1, 1].grid()
    plt.show()
    # save figure
    fig.savefig(logdir / f"data_stabilization_{time.strftime('%Y%m%d-%H%M%S')}.png")

    data_stable = input("Is the data stable? (y/n): ") == "y"

# Stop the threads
log_acclimation = False
vis_thread.join()
thr_thread.join()

# Save the data
acc_vis_frames = np.array(acc_vis_frames)
acc_vis_tstamps = np.array(acc_vis_tstamps)
acc_thr_frames = np.array(acc_thr_frames)
acc_thr_tstamps = np.array(acc_thr_tstamps)

output_filename = logdir / f"acclimation_data_{time.strftime('%Y%m%d-%H%M%S')}.npz"
np.savez(output_filename, 
                            acc_vis_imgs=acc_vis_frames,
                            acc_vis_tstamps=acc_vis_tstamps,
                            acc_thr_imgs=acc_thr_frames,
                            acc_thr_tstamps=acc_thr_tstamps,
                            acc_vis_pixel_vals=acc_vis_pixel_vals,
                            acc_thr_pixel_vals=acc_thr_pixel_vals,
                            **pulsetrain_settings)
print(f"Data saved to {output_filename}")

# Start experiment or abort
start_experiment = input("Do you want to start the experiment? (y/n): ") == "y"
if not start_experiment:
    shutdown()
    sys.exit()

running = True
## Set up capture thread for visible camera
ls_vis_frames = []
ls_vis_tstamps = []
ls_vis_frame_numbers = []
ls_vis_exposures = []
ls_vis_gains = []

def visible_capture_thread_v2():
    global ls_vis_frames, ls_vis_tstamps, ls_vis_frame_numbers, ls_vis_exposures, ls_vis_gains, running
    
    while running:
        vis_img, vis_tstamp, vis_frame_number, vis_exp, vis_gain = vis_cam_obj.get_next_image()
        ls_vis_frames.append(vis_img)
        ls_vis_tstamps.append(vis_tstamp)
        ls_vis_frame_numbers.append(vis_frame_number)
        ls_vis_exposures.append(vis_exp)
        ls_vis_gains.append(vis_gain)

vis_cap_thread = threading.Thread(target=visible_capture_thread_v2)
vis_cap_thread.daemon = True

# Set FFC to manual for the thermal camera
thr_cam_obj.camera.set_ffc_manual()
# Do FFC for the thermal camera
thr_cam_obj.camera.do_ffc()
time.sleep(1)

light_change_times = []

if vis_filter_to_use == "sp700":
    pulse_exposures = pulsetrain_settings['vis_cam_exposure_sp700']
else:
    pulse_exposures = pulsetrain_settings['vis_cam_exposure_lp700']

# Start the threads
light_change_times.append(time.time())
vis_cap_thread.start()
thr_cam_obj.start_logging()

# Light pulse routine
# Initial relaxation period
time.sleep(1)
for i in range(num_pulses):
    print(f"Pulse {i+1} of {num_pulses}")

    # Set the exposure for the visible camera
    vis_cam_obj.set_exposure(pulse_exposures[i], check=False)

    # Set the light intensity
    # print(f"Setting light intensity to {pulsetrain_settings['light_intensity_perc'][i]}%")
    light_change_times.append(time.time())
    light_obj.set_voltage(pulsetrain_settings['light_control_voltages'][i])
    # Record the light pulse
    time.sleep(pulsetrain_settings['light_pulse_duration'][i])

# Set FFC to Auto
thr_cam_obj.camera.set_ffc_auto()
thr_cam_obj.camera.do_ffc()
time.sleep(1)

print("Stopping the threads")
running = False
vis_cap_thread.join()
thr_cam_obj.stop_logging()

# Collect visible images for SP700 for all the pulses
vis_frames_sp700 = []
vis_tstamps_sp700 = []
vis_frame_numbers_sp700 = []
vis_exposures_sp700 = []
vis_gains_sp700 = []

vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_sp700'][0])
stage_obj.set_filter_position("sp700", blocking=True)

vis_frame, vis_tstamp, vis_frame_number, vis_exp, vis_gain = vis_cam_obj.get_next_image()
vis_frames_sp700.append(vis_frame)
vis_tstamps_sp700.append(vis_tstamp)
vis_frame_numbers_sp700.append(vis_frame_number)
vis_exposures_sp700.append(vis_exp)
vis_gains_sp700.append(vis_gain)

for i in range(num_pulses):
    light_obj.set_voltage(pulsetrain_settings['light_control_voltages'][i])
    time.sleep(0.1)

    vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_sp700'][i])
    
    vis_frame, vis_tstamp, vis_frame_number, vis_exp, vis_gain = vis_cam_obj.get_next_image()
    vis_frames_sp700.append(vis_frame)
    vis_tstamps_sp700.append(vis_tstamp)
    vis_frame_numbers_sp700.append(vis_frame_number)
    vis_exposures_sp700.append(vis_exp)
    vis_gains_sp700.append(vis_gain)

light_obj.set_voltage(0.0)
vis_cam_obj.set_auto_exposure(True)

stage_obj.set_filter_position(vis_filter_to_use, blocking=True)

print("Saving the data")
ls_vis_frames = np.array(ls_vis_frames)
ls_vis_tstamps = np.array(ls_vis_tstamps)
ls_vis_frame_numbers = np.array(ls_vis_frame_numbers)
ls_vis_exposures = np.array(ls_vis_exposures)
ls_vis_gains = np.array(ls_vis_gains)

raw_thr_frames = np.array(thr_cam_obj.logged_images)
raw_thr_tstamps = np.array(thr_cam_obj.logged_tstamps)
thr_cam_timestamp_offset = thr_cam_obj.timestamp_offset

vis_frames_sp700 = np.array(vis_frames_sp700)
vis_tstamps_sp700 = np.array(vis_tstamps_sp700)
vis_frame_numbers_sp700 = np.array(vis_frame_numbers_sp700)
vis_exposures_sp700 = np.array(vis_exposures_sp700)
vis_gains_sp700 = np.array(vis_gains_sp700)

output_filename = logdir / f"pulsetrain_data_{time.strftime('%Y%m%d-%H%M%S')}.npz"
np.savez(output_filename, 
                            ls_vis_frames=ls_vis_frames,
                            ls_vis_tstamps=ls_vis_tstamps,
                            ls_vis_frame_numbers=ls_vis_frame_numbers,
                            ls_vis_exposures=ls_vis_exposures,
                            ls_vis_gains=ls_vis_gains,
                            raw_thr_frames=raw_thr_frames,
                            raw_thr_tstamps=raw_thr_tstamps,
                            vis_frames_sp700=vis_frames_sp700,
                            vis_tstamps_sp700=vis_tstamps_sp700,
                            vis_frame_numbers_sp700=vis_frame_numbers_sp700,
                            vis_exposures_sp700=vis_exposures_sp700,
                            vis_gains_sp700=vis_gains_sp700,
                            thr_cam_timestamp_offset=thr_cam_timestamp_offset,
                            **pulsetrain_settings)
print(f"Data saved to {output_filename}")

# Shutdown the cameras
shutdown()

## Preview the data
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0, 0].imshow(ls_vis_frames[-1, :, :], cmap='gray')
ax[0, 0].set_title('Visible Image')
for i, pixel in enumerate(vis_selected_pixels):
    ax[1, 0].plot(ls_vis_tstamps - ls_vis_tstamps[0], ls_vis_frames[:, pixel[1], pixel[0]] / ls_vis_exposures, '.', label=f"Pixel {i}", color=f"C{i}")
    ax[0, 0].scatter(pixel[0], pixel[1], color=f"C{i}")
ax[1, 0].set_title('Visible Pixel Value')
ax[1, 0].set_xlabel('Time (s)')
ax[1, 0].set_ylabel('Pixel Value')
ax[0, 1].imshow(raw_thr_frames[-1, 2:, ::-1], cmap='turbo')
ax[0, 1].set_title('Thermal Image')
for i, pixel in enumerate(thr_selected_pixels):
    ax[1, 1].plot(raw_thr_tstamps - raw_thr_tstamps[0], raw_thr_frames[:, 2:, ::-1][:, pixel[1], pixel[0]], '.', label=f"Pixel {i}", color=f"C{i}")
    ax[0, 1].scatter(pixel[0], pixel[1], color=f"C{i}")
ax[1, 1].set_title('Thermal Pixel Value')
ax[1, 1].set_xlabel('Time (s)')
ax[1, 1].set_ylabel('Pixel Value')
png_filename = str(output_filename).replace(".npz", "_preview.png")
fig.savefig(png_filename)
print(f"Preview saved to {png_filename}")
# plt.show()

# input("Press enter to exit")

print("Data collection script complete")