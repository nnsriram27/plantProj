import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import subprocess
import signal
from datetime import datetime
import time
import sys
import glob
workspace_path = '/home/ilimnuc/plantProj/'
sys.path.append(workspace_path)
os.chdir(workspace_path)


# ------ VISIBLE CAMERA ------
from cameras.ids_ueye import uEye

from lib.thermal_utils import *

print('Import completed.')

########### Macros ###############
CAMERA = 'Boson'
RADIOMETRY_ENABLE = True
TSTABLE = True
TLINEAR = False
IMG_H, IMG_W = 512,640  
CAMERA_FPS   = 60


############ Camera Operation  ############
vis_cam_obj = uEye()
vis_cam_obj.set_exposure(10)
print("Visible camera connected.")

thermalCamera_connected = False
password = "ilima401!"

# Loop through available /dev/ttyACM* ports
for port in glob.glob('/dev/ttyACM*'):
    try:
        # Change permissions for the port
        os.system(f"echo {password} | sudo -S chmod 777 {port}")
        
        # Try to connect to the thermal camera
        thermal_cam = BosonCamera(port=port)
        myCam = CamAPI.pyClient(manualport=port)
        
        thermalCamera_connected = True
        print(f"Connected to thermal camera on {port}")
        break  # Exit loop once connected

    except Exception as e:
        print(e)
        print(f"Failed to connect on {port}. Trying the next port...")
        time.sleep(1)  # Optional: add a small delay before trying the next port

if CAMERA == 'Boson' and RADIOMETRY_ENABLE:
    myCam.radiometrySetTempStableEnable(FLR_ENABLE_E.FLR_ENABLE)
    if TLINEAR:
        myCam.TLinearSetControl(FLR_ENABLE_E.FLR_ENABLE)
    else:
        myCam.TLinearSetControl(FLR_ENABLE_E.FLR_DISABLE)

try:
    assert thermalCamera_connected, "Thermal camera not connected."
except:
    del vis_cam_obj
    print("Visible camera disconnected.")
    sys.exit(1)


thread = FrameThreadGeneral(thermal_cam)
thread.start()

# Read 3 frames before capturing the vis-thermal frame
for _ in range(3):
    timg, _ = thread.read()
    vis_img = vis_cam_obj.getNextImage()
    # print(timg.shape, vis_img.shape)
    time.sleep(0.1)

vis_img, vis_tstamp = vis_cam_obj.getNextImage(), time.time()
(thermal_img, _), thermal_tstamp = thread.read(), time.time()
print("Vis Img shape:", vis_img.shape, "Thermal img shape:", thermal_img.shape)
print("Vis-Thermal frames captured.")

# Save the vis-thermal frame
save_data = {
    'vis_img': vis_img,
    'thermal_img': thermal_img,
    'vis_tstamp': vis_tstamp,
    'thermal_tstamp': thermal_tstamp
}

save_path = os.path.join(workspace_path, f"data/daily_capture/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.npz")
os.makedirs(os.path.dirname(save_path), exist_ok=True)
np.savez(save_path, **save_data)
print(f"Data saved at {save_path}")

thread.stop()
thread.join()
del vis_cam_obj
del thermal_cam
print("Cameras disconnected.")
sys.exit(1)