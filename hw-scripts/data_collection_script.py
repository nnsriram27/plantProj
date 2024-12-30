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

from cameras.wrapper_boson import BosonWithTelemetry
from cameras.wrapper_pyspin import Blackfly
from cameras.wrapper_filterstage import FilterStage
from cameras.cam_threads import VisCamThread, ThrCamThread
from lights.light_intensity_controller import LightIntensityController

def parse_args():
    parser = argparse.ArgumentParser(description="Collect data for a pulsetrain experiment")
    parser.add_argument("--data-slug", type=str, required=True, help="Slug for the data")
    parser.add_argument("--pulsetrain-settings", type=str, required=True, help="Path to the pulsetrain settings file")
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

print("Pulsetrain settings:")
print(json.dumps(pulsetrain_settings, indent=4))
confirm = input("Are these settings correct? (y/n): ") == "y"
if not confirm:
    sys.exit()

## Initialize the cameras and other devices
try:
    vis_cam_obj = Blackfly()
except:
    print("Failed to connect to visible camera")
    sys.exit()
vis_cam_obj.set_fps(100)
vis_cam_obj.set_gain(0)
vis_cam_obj.set_exposure(1000)

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

# Capture sample images with the different exposure settings and confirm they are not over or under exposed
num_pulses = len(pulsetrain_settings['light_control_voltages'])
if args.check_exposure:
    if args.auto_exposure:
        vis_cam_obj.set_auto_exposure(True)

        def get_stable_exposure_value():
            stable = False
            last_exp_val = 0
            while not stable:
                vis_img, vis_tstamp, _, vis_exp, _ = vis_cam_obj.get_next_image()
                if np.abs(last_exp_val - vis_exp) / last_exp_val < 0.05:
                    stable = True
                last_exp_val = vis_exp
            return last_exp_val, vis_img

    ## Check for SP700 position
    stage_obj.set_filter_position("sp700", blocking=True)
    vis_imgs_sp700 = []
    vis_exposures_sp700 = []
    if args.auto_exposure:
        exp_val, vis_img = get_stable_exposure_value()
        vis_imgs_sp700.append(vis_img)
        vis_exposures_sp700.append(exp_val)
    else:
        vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_sp700'][0])
        vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
        vis_imgs_sp700.append(vis_img)
        vis_exposures_sp700.append(pulsetrain_settings['vis_cam_exposure_sp700'][0])

    for i, exposure in enumerate(pulsetrain_settings['vis_cam_exposure_sp700']):
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
    stage_obj.set_filter_position("lp700", blocking=True)
    vis_imgs_lp700 = []
    vis_exposures_lp700 = []
    if args.auto_exposure:
        exp_val, vis_img = get_stable_exposure_value()
        vis_imgs_lp700.append(vis_img)
        vis_exposures_lp700.append(exp_val)
    else:
        vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_lp700'][0])
        vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
        vis_imgs_lp700.append(vis_img)
        vis_exposures_lp700.append(pulsetrain_settings['vis_cam_exposure_lp700'][0])

    for i, exposure in enumerate(pulsetrain_settings['vis_cam_exposure_lp700']):
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

    vis_cam_obj.set_auto_exposure(False)

    for i in range(len(vis_imgs_sp700)):
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        fig.suptitle(f"Exposure Test {i+1}, SP700 exposure: {vis_exposures_sp700[i]}, LP700 exposure: {vis_exposures_lp700[i]}")
        print(f"Exposure Test {i+1}, SP700 exposure: {vis_exposures_sp700[i]}, LP700 exposure: {vis_exposures_lp700[i]}")
        ax[0].imshow(vis_imgs_sp700[i])
        num_saturated = np.sum(vis_imgs_sp700[i] >= 0.95 * 2**16)
        ax[0].set_title(f"SP700, Mean: {np.mean(vis_imgs_sp700[i]):.2f}, Std: {np.std(vis_imgs_sp700[i]):.2f}, Saturated: {num_saturated}")
        print(f"SP700, Mean: {np.mean(vis_imgs_sp700[i]):.2f}, Std: {np.std(vis_imgs_sp700[i]):.2f}, Saturated: {num_saturated}")
        ax[0].axis("off")
        ax[1].imshow(vis_imgs_lp700[i])
        num_saturated = np.sum(vis_imgs_lp700[i] >= 0.95 * 2**16)
        ax[1].set_title(f"LP700, Mean: {np.mean(vis_imgs_lp700[i]):.2f}, Std: {np.std(vis_imgs_lp700[i]):.2f}, Saturated: {num_saturated}")
        print(f"LP700, Mean: {np.mean(vis_imgs_lp700[i]):.2f}, Std: {np.std(vis_imgs_lp700[i]):.2f}, Saturated: {num_saturated}")
        ax[1].axis("off")
        plt.show()
        confirm = input("Are these images correctly exposed? (y/n): ") == "y"
        if not confirm:
            shutdown()
            sys.exit()

    print(f"Images are correctly exposed. Now select some pixels of interest")
else:
    print(f"Skipping exposure check.  Now select some pixels of interest")

# Display first image and select pixels
stage_obj.set_filter_position("lp700", blocking=True)
vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_lp700'][0])

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

# Wait for the leaves to regain steady state

data_stable = False
vis_pixel_vals = []
acc_vis_imgs = []
acc_vis_tstamps = []
thr_pixel_vals = []
acc_thr_imgs = []
acc_thr_tstamps = []

while not data_stable:

    ## Collect data for 10 seconds and plot data
    for frame_idx in tqdm(range(15)):
        vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
        thr_img, thr_tstamp, _, _ = thr_cam_obj.get_next_image(hflip=True)
        for i, pixel in enumerate(vis_selected_pixels):
            if frame_idx == 0:
                vis_pixel_vals.append([])
            vis_pixel_vals[i].append(vis_img[pixel[1], pixel[0]])
        for i, pixel in enumerate(thr_selected_pixels):
            if frame_idx == 0:
                thr_pixel_vals.append([])
            thr_pixel_vals[i].append(thr_img[pixel[1], pixel[0]])
        acc_vis_imgs.append(vis_img)
        acc_vis_tstamps.append(vis_tstamp)
        acc_thr_imgs.append(thr_img)
        acc_thr_tstamps.append(thr_tstamp)
        time.sleep(1)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(vis_img, cmap="turbo")
    ax[0, 0].set_title("Visible Image")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(thr_img, cmap="turbo")
    ax[0, 1].set_title("Thermal Image")
    ax[0, 1].axis("off")
    for i, pixel in enumerate(vis_selected_pixels):
        ax[0, 0].scatter(pixel[0], pixel[1], color=f"C{i}")
        ax[1, 0].plot(acc_vis_tstamps, vis_pixel_vals[i], '.', label=f"Pixel {i}", color=f"C{i}")
    ax[1, 0].set_title("Visible Pixel Values")
    ax[1, 0].set_xlabel("Timestamp")
    ax[1, 0].set_ylabel("Pixel Value")
    for i, pixel in enumerate(thr_selected_pixels):
        ax[0, 1].scatter(pixel[0], pixel[1], color=f"C{i}")
        ax[1, 1].plot(acc_thr_tstamps, thr_pixel_vals[i], '.', label=f"Pixel {i}", color=f"C{i}")
    ax[1, 1].set_title("Thermal Pixel Values")
    ax[1, 1].set_xlabel("Timestamp")
    ax[1, 1].set_ylabel("Pixel Value")
    plt.show()
    # save figure
    fig.savefig(f"./data/{data_slug}_data_stabilization_{time.strftime('%Y%m%d-%H%M%S')}.png")

    data_stable = input("Is the data stable? (y/n): ") == "y"

# Confirm the data slug
print(f"Data slug: {data_slug}")
confirm = input("Is this the correct data slug? (y/n): ") == "y"
if not confirm:
    shutdown()
    sys.exit()

running = True
pause_recording = False

save_fps = pulsetrain_settings['fps_between_pulses']
light_obj.set_voltage(0.0)
stage_obj.set_filter_position("lp700", blocking=True)
vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_lp700'][0])


## Set up capture thread for visible camera
ls_vis_frames = []
ls_vis_tstamps = []
ls_vis_frame_numbers = []
ls_vis_exposures = []
ls_vis_gains = []

def visible_capture_thread():
    global ls_vis_frames
    global ls_vis_tstamps
    global ls_vis_frame_numbers
    global ls_vis_exposures
    global ls_vis_gains
    global running
    global pause_recording
    global save_fps
    last_frame_time = time.time()

    while running:
        if pause_recording:
            continue
        vis_img, vis_tstamp, vis_frame_number, vis_exp, vis_gain = vis_cam_obj.get_next_image()
        if vis_tstamp - last_frame_time > 1 / save_fps:
            ls_vis_frames.append(vis_img)
            ls_vis_tstamps.append(vis_tstamp)
            ls_vis_frame_numbers.append(vis_frame_number)
            ls_vis_exposures.append(vis_exp)
            ls_vis_gains.append(vis_gain)
            last_frame_time = vis_tstamp

vis_cap_thread = threading.Thread(target=visible_capture_thread)

# Set up capture thread for thermal camera
ls_thr_frames = []
ls_thr_tstamps = []
ls_thr_frame_numbers = []
ls_thr_telemetry = []

def thermal_capture_thread():
    global ls_thr_frames
    global ls_thr_tstamps
    global ls_thr_frame_numbers
    global ls_thr_telemetry
    global running
    global pause_recording
    global save_fps
    _, last_frame_time, _, _ = thr_cam_obj.get_next_image(hflip=True)

    while running:
        if pause_recording:
            continue
        thr_img, thr_tstamp, thr_frame_num, thr_telemetry = thr_cam_obj.get_next_image(hflip=True)
        if thr_tstamp - last_frame_time > 1 / save_fps:
            ls_thr_frames.append(thr_img)
            ls_thr_tstamps.append(thr_tstamp)
            ls_thr_frame_numbers.append(thr_frame_num)
            ls_thr_telemetry.append(thr_telemetry)
            last_frame_time = thr_tstamp

thr_cap_thread = threading.Thread(target=thermal_capture_thread)

# Set FFC to manual for the thermal camera
thr_cam_obj.camera.set_ffc_manual()
# Do FFC for the thermal camera
thr_cam_obj.camera.do_ffc()
time.sleep(1)

# Start the threads
vis_cap_thread.start()
thr_cap_thread.start()

# Light pulse routine
# Initial relaxation period
time.sleep(10)
for i in range(num_pulses):
    print(f"Pulse {i+1} of {num_pulses}")

    # Set the exposure for the visible camera
    vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_lp700'][i], check=False)

    # Change the save fps
    save_fps = pulsetrain_settings['fps_during_pulse']
    time.sleep(1)

    # Set the light intensity
    print(f"Setting light intensity to {pulsetrain_settings['light_intensity_perc'][i]}%")
    light_obj.set_voltage(pulsetrain_settings['light_control_voltages'][i])
    # Record the light pulse
    time.sleep(pulsetrain_settings['light_pulse_duration'][i])

    # Turn off the light
    print("Turning off the light")
    light_obj.set_voltage(0.0)
    time.sleep(1)

    # Set the exposure for the visible camera
    vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_lp700'][0], check=False)

    # Change the save fps
    save_fps = pulsetrain_settings['fps_between_pulses']

    # Record the relaxation period
    time.sleep(pulsetrain_settings['relaxation_duration'][i])

# Set FFC to Auto
thr_cam_obj.camera.set_ffc_auto()
thr_cam_obj.camera.do_ffc()

print("Stopping the threads")
running = False
vis_cap_thread.join()
thr_cap_thread.join()

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
    time.sleep(1)

    vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_sp700'][i])
    
    vis_frame, vis_tstamp, vis_frame_number, vis_exp, vis_gain = vis_cam_obj.get_next_image()
    vis_frames_sp700.append(vis_frame)
    vis_tstamps_sp700.append(vis_tstamp)
    vis_frame_numbers_sp700.append(vis_frame_number)
    vis_exposures_sp700.append(vis_exp)
    vis_gains_sp700.append(vis_gain)

light_obj.set_voltage(0.0)
vis_cam_obj.set_exposure(pulsetrain_settings['vis_cam_exposure_lp700'][0])

stage_obj.set_filter_position("lp700", blocking=True)

print("Saving the data")
acc_vis_imgs = np.array(acc_vis_imgs)
acc_vis_tstamps = np.array(acc_vis_tstamps)
acc_thr_imgs = np.array(acc_thr_imgs)
acc_thr_tstamps = np.array(acc_thr_tstamps)

ls_vis_frames = np.array(ls_vis_frames)
ls_vis_tstamps = np.array(ls_vis_tstamps)
ls_vis_frame_numbers = np.array(ls_vis_frame_numbers)
ls_vis_exposures = np.array(ls_vis_exposures)
ls_vis_gains = np.array(ls_vis_gains)

ls_thr_frames = np.array(ls_thr_frames)
ls_thr_tstamps = np.array(ls_thr_tstamps)
ls_thr_frame_numbers = np.array(ls_thr_frame_numbers)
ls_thr_telemetry = np.array(ls_thr_telemetry)

vis_frames_sp700 = np.array(vis_frames_sp700)
vis_tstamps_sp700 = np.array(vis_tstamps_sp700)
vis_frame_numbers_sp700 = np.array(vis_frame_numbers_sp700)
vis_exposures_sp700 = np.array(vis_exposures_sp700)
vis_gains_sp700 = np.array(vis_gains_sp700)

output_filename = f"./data/{data_slug}_{time.strftime('%Y%m%d-%H%M%S')}.npz"
np.savez(output_filename, 
                            acc_vis_imgs=acc_vis_imgs,
                            acc_vis_tstamps=acc_vis_tstamps,
                            acc_thr_imgs=acc_thr_imgs,
                            acc_thr_tstamps=acc_thr_tstamps,
                            ls_vis_frames=ls_vis_frames,
                            ls_vis_tstamps=ls_vis_tstamps,
                            ls_vis_frame_numbers=ls_vis_frame_numbers,
                            ls_vis_exposures=ls_vis_exposures,
                            ls_vis_gains=ls_vis_gains,
                            ls_thr_frames=ls_thr_frames,
                            ls_thr_tstamps=ls_thr_tstamps,
                            ls_thr_frame_numbers=ls_thr_frame_numbers,
                            ls_thr_telemetry=ls_thr_telemetry,
                            vis_frames_sp700=vis_frames_sp700,
                            vis_tstamps_sp700=vis_tstamps_sp700,
                            vis_frame_numbers_sp700=vis_frame_numbers_sp700,
                            vis_exposures_sp700=vis_exposures_sp700,
                            vis_gains_sp700=vis_gains_sp700,
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
ax[0, 1].imshow(ls_thr_frames[-1, :, :], cmap='turbo')
ax[0, 1].set_title('Thermal Image')
for i, pixel in enumerate(thr_selected_pixels):
    ax[1, 1].plot(ls_thr_tstamps - ls_thr_tstamps[0], ls_thr_frames[:, pixel[1], pixel[0]], '.', label=f"Pixel {i}", color=f"C{i}")
    ax[0, 1].scatter(pixel[0], pixel[1], color=f"C{i}")
ax[1, 1].set_title('Thermal Pixel Value')
ax[1, 1].set_xlabel('Time (s)')
ax[1, 1].set_ylabel('Pixel Value')
png_filename = output_filename.replace(".npz", "_preview.png")
fig.savefig(png_filename)
print(f"Preview saved to {png_filename}")
# plt.show()

# input("Press enter to exit")

print("Data collection script complete")
