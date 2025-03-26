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
from cameras.cam_threads import VisCamThread, ThrCamThread


def parse_args():
    parser = argparse.ArgumentParser(description='Alignment check')
    return parser.parse_args()

args = parse_args()

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


def shutdown():
    vis_cam_obj.stop()
    thr_cam_obj.stop()

old_vis_tstamp = time.time()
old_thr_tstamp = time.time()
while True:
    vis_img, vis_tstamp, _, _, _ = vis_cam_obj.get_next_image()
    thr_img, thr_tstamp, _, _ = thr_cam_obj.get_next_image(hflip=True)

    # Center crop vis image to have same size of thr image
    vis_img_cropped = vis_img[2:-2, 25:-25]
    vis_img_resized = cv2.resize(vis_img_cropped, (thr_img.shape[1], thr_img.shape[0]))
    
    thr_img_normed = cv2.normalize(thr_img, None, 0, 255, cv2.NORM_MINMAX)
    thr_img_viz = cv2.applyColorMap(thr_img_normed.astype(np.uint8), cv2.COLORMAP_HOT)
    vis_img_normed = cv2.normalize(vis_img_resized, None, 0, 255, cv2.NORM_MINMAX)
    vis_img_viz = cv2.applyColorMap(vis_img_normed.astype(np.uint8), cv2.COLORMAP_COOL)

    # Blend the two images
    alpha = 0.5
    blended = cv2.addWeighted(vis_img_viz, alpha, thr_img_viz, 1-alpha, 0)

    cv2.imshow("Blended", blended)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    print(f"Vis: {vis_tstamp - old_vis_tstamp:.04f}, Thr: {thr_tstamp - old_thr_tstamp:.04f} \r", end='')
    old_vis_tstamp = vis_tstamp
    old_thr_tstamp = thr_tstamp

cv2.destroyAllWindows()

shutdown()
print("Done")