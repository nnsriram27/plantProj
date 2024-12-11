import sys
sys.path.append('../plantProj')

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

from cameras.wrapper_boson import BosonWithTelemetry
from cameras.wrapper_pyspin import Blackfly
from cameras.wrapper_filterstage import FilterStage
from cameras.cam_threads import VisCamThread, ThrCamThread
from lights.light_intensity_controller import LightIntensityController


display_running = True
# log_pixels = True
# log_images = False
# log_metadata = False

# light_obj = LightIntensityController()
# light_obj.set_voltage(0)

# stage_obj = FilterStage()
# stage_obj.set_filter_position("clear", blocking=True)

vis_cam_obj = Blackfly()
if vis_cam_obj == False:
    print("Failed to connect to visible camera")
    sys.exit()

vis_cam_obj.set_fps(60)
vis_cam_obj.set_gain(0)
vis_cam_obj.set_exposure(100)

thr_cam_obj = BosonWithTelemetry()
thr_cam_obj.camera.do_ffc()

vis_selected_pixels = [(360, 270)]
vis_pixel_vals = [[]]
vis_tstamps = []

thr_selected_pixels = [(320, 256)]
thr_pixel_vals = [[]]
thr_tstamps = []

# vis_thread = VisCamThread(vis_cam_obj)
# thr_thread = ThrCamThread(thr_cam_obj)

def mouse_callback_vis(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked pixel at ({x}, {y}) in visible image")
        vis_selected_pixels.append((x, y))
        vis_pixel_vals = [[] for _ in vis_selected_pixels]
        vis_tstamps = []

def mouse_callback_thr(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked pixel at ({x}, {y}) in thermal image")
        thr_selected_pixels.append((x, y))
        thr_pixel_vals = [[] for _ in thr_selected_pixels]
        thr_tstamps = []

cv2.namedWindow("Thermal", cv2.WINDOW_NORMAL)
cv2.namedWindow("Visible", cv2.WINDOW_NORMAL)

cv2.setMouseCallback("Visible", mouse_callback_vis)
cv2.setMouseCallback("Thermal", mouse_callback_thr)

# fig, ax_vis = plt.subplots(1, 1)
# ax_thr = ax_vis.twinx()

# plt.ion()

# vis_thread.start()
# thr_thread.start()
# vis_thread.update_logging(log_images, log_pixels, log_metadata)
# thr_thread.update_logging(log_images, log_pixels, log_metadata)

time.sleep(1)

while True:
    thr_img, thr_timestamp, thr_frame_number, thr_telemetry = thr_cam_obj.get_next_image()
    vis_img, vis_timestamp, vis_frame_number, vis_exposure, vis_gain = vis_cam_obj.get_next_image()

    thr_img_viz = cv2.normalize(thr_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr_img_viz = cv2.applyColorMap(thr_img_viz, cv2.COLORMAP_TURBO)

    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    vis_img = vis_img.astype(np.float32) / vis_exposure

    # ax_thr.clear()
    thr_tstamps.append(thr_timestamp)
    for i, pixel in enumerate(thr_selected_pixels):
        cv2.drawMarker(thr_img_viz, (pixel[0], pixel[1]), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
        thr_pixel_vals[i].append(thr_img[pixel[1], pixel[0]])
        # ax_thr.plot(thr_tstamps, thr_pixel_vals[i], label="Thermal Pixel")
    cv2.imshow("Thermal", thr_img_viz)

    # ax_vis.clear()
    vis_tstamps.append(vis_timestamp)
    for i, pixel in enumerate(vis_selected_pixels):
        cv2.drawMarker(vis_img, (pixel[0], pixel[1]), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
        vis_pixel_vals[i].append(vis_img[pixel[1], pixel[0]])
        # ax_vis.plot(vis_tstamps, vis_pixel_vals[i], label="Visible Pixel")
    cv2.imshow("Visible", vis_img)

    # plt.pause(0.01)

    # if log_pixels:
    #     for i, pixel in enumerate(vis_thread.selected_pixels):
    #         if len(vis_thread.pixel_vals[i]) > 0:
    #             ax_vis.clear()
    #             ax_vis.plot(vis_thread.pixel_timestamps, vis_thread.pixel_vals[i], label=f"Pixel {pixel}")
    #             ax_vis.set_xlabel("Timestamp")
    #             ax_vis.set_ylabel("Visible Pixel Value")
    #     for i, pixel in enumerate(thr_thread.selected_pixels):
    #         if len(thr_thread.pixel_vals[i]) > 0:
    #             ax_thr.clear()
    #             ax_thr.plot(thr_thread.pixel_timestamps, thr_thread.pixel_vals[i], label=f"Pixel {pixel}")
    #             ax_thr.set_xlabel("Timestamp")
    #             ax_thr.set_ylabel("Thermal Pixel Value")
    #     plt.draw()
    #     plt.pause(0.001)

    key = cv2.waitKey(1)
    if key == ord('q'):
        display_running = False
        break

# vis_thread.stop()
# thr_thread.stop()

# del vis_cam_obj
# del thr_cam_obj
thr_cam_obj.stop()
vis_cam_obj.stop()

# light_obj.close()
# stage_obj.close()
# plt.close()
cv2.destroyAllWindows()

# import sys, traceback, threading
# thread_names = {t.ident: t.name for t in threading.enumerate()}
# for thread_id, frame in sys._current_frames().iteritems():
#     print("Thread %s:" % thread_names.get(thread_id, thread_id))
#     traceback.print_stack(frame)
#     print()