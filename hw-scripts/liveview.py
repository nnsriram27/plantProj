import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import subprocess
from datetime import datetime
import glob
import time
import sys
sys.path.append('/home/ilimnuc/plantProj/')
os.chdir("/home/ilimnuc/plantProj/")

# ------ GLOBALS ------
import matplotlib as mpl 
# %matplotlib widget

# ------ VISIBLE CAMERA ------
from cameras.ids_ueye import uEye

from lib.image_processing import *
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
    # thermal_cam.start_grabber()
except:
    del vis_cam_obj
    print("Visible camera disconnected.")
    sys.exit(1)


def preview_vis_thermal_cam(vis_cam,thermal_cam,resize_factor = 1.5, is_color=True): 
    try:
        cv2.namedWindow('Visible')
        cv2.namedWindow('Thermal')
        use_minmax = None
        thread = FrameThreadGeneral(thermal_cam)
        thread.start()

        while(True):

            ## Visible Camera ##
            vis_img, vis_tstamp = vis_cam.getNextImage(), time.time()
            vis_img_color = np.floor(cv2.cvtColor(vis_img, cv2.COLOR_BayerRGGB2BGR) / 16).astype(np.uint8)
            # Apply gamma correction to the visible image
            gamma_val = 0.6
            vis_img_gamma = np.power(vis_img_color / 255.0, gamma_val)
            vis_img_gamma = (vis_img_gamma * 255).astype(np.uint8)
            cv2.imshow('Visible', vis_img_gamma)
            # cv2.imshow('Visible', vis_img_color)
                
            # thermal_frame, _ = thermal_cam.get_latest_frame()
            thermal_frame, _ = thread.read()
            curr_frame_min, curr_frame_max = thermal_frame.min(), thermal_frame.max()
            if is_color:
                thermal_frame = cv2.cvtColor(thermal_frame, cv2.COLOR_BAYER_BG2BGR)
                thermal_frame, minmax = thermal_cam_operation(thermal_frame,use_minmax)
            if resize_factor!=1:
                thermal_frame = cv2.resize(thermal_frame,None,fx = resize_factor, fy = resize_factor)
            
            # horizontally flip thermal image
            thermal_frame = cv2.flip(thermal_frame, 1)
                
            # Display curr_frame_min and curr_frame_max
            cv2.putText(thermal_frame,'Min: {}'.format(curr_frame_min),(30,50), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(thermal_frame,'Max: {}'.format(curr_frame_max),(30,70), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)    

            cv2.imshow("Thermal",thermal_frame)

            key = cv2.waitKey(1) 
            if key==13:
                break
            elif key==32: #space
                use_minmax=minmax if use_minmax is None else None
            elif key==40:
                gamma_val-= 0.1
            elif key==41:
                gamma_val+= 0.1
                
    except Exception:
        traceback.print_exc()
        print('Closing camera thread.') 

    # cleanup

    cv2.destroyAllWindows()
    thread.stop()
    thread.join()
    del vis_cam
    del thermal_cam
    print("Cameras disconnected.")
    return
preview_vis_thermal_cam(vis_cam_obj, thermal_cam,resize_factor=1)
sys.exit(1)