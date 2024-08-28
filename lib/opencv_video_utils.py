import numpy as np
import cv2

import traceback
import pickle
import skimage
import os

from lib.image_processing import vis_cam_operation, thermal_cam_operation
import datetime
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT = cv2.FONT_HERSHEY_SIMPLEX

class videoPlayerVisThermal():
    def __init__(self, N_frames, get_vis_thermal_img_func: lambda x: [None, None], process_vis_img_func = vis_cam_operation, process_thermal_img_func = thermal_cam_operation,\
        thr2Vis_HMatrix = None, workspace_path=None):
        self.N_frames      = N_frames
        self.data_counter  = 0
        self.get_vis_thermal_img_func = get_vis_thermal_img_func
        # self.get_vis_img_func = get_vis_img_func
        # self.get_thermal_img_func = thermal_img_func
        self.process_vis_img_func = process_vis_img_func
        self.process_thermal_img_func = process_thermal_img_func
        self.thermal_minmax = None
        self.selected_pix_vis = []
        self.selected_pix_thermal = []
        self.selected_thr_mask_id = []
        self.thr2visH = thr2Vis_HMatrix
        self.speed_level = 1
        self.overlay_vis_thr = False
        self.alpha = 0.5
        self.pixel_plot_fig = None
        self.pixel_plot_counter = 0
        self.skip_frames_for_seg = 5

        self.warp_vis_img = False
        self.homography_mode = False
        self.workspace_path = workspace_path
        self.show_segmentation_mask = False
        self.show_visible = True
        self.show_thermal = True
        self.saved_thermal_frames_tmp = False
        self.shift_toggle = False
        self.mouse_location = None
        self.video_segmentation_mode = False
        self.add_points_to_predictor = False
        self.propagate_anns_bool = False
        self.video_masks = None
        self.all_thr_mask_names = None

        self.common_box_pts = None
        self.video_ann_pts = []
        self.video_ann_pts_label = []

    def plot_pixel_values_across_time(self, selected_pix_vis, selected_pix_thermal):
        pass
    
    # -------- Loop control -------- #
    def loop_control(self,key):
        """
            Loop control function.
            Press 'shift and then h' to print docstring.
            Press `Enter` to exit the loop.
            Press `+` or `-` to increase or decrease frame number.
            Press `Space` to toggle thermal minmax.
            Press `w` to toggle warp visible image.
            Press `c` to clear selected pixels.
            Press `d` or `a` to increase or decrease speed level.
            Press `o` to overlay visible and thermal images.
            Press `i` or `u` to increase or decrease alpha value.
            Press `h` to toggle homography mode.
            Press `s` to save homography matrix.
            Press `p` to plot pixel values across time.
            Press `m` to toggle segmentation mask.
            Press `v` to toggle visible image.
            Press `t` to toggle thermal image.
            Press `shift` to toggle shift key.
            Press `shift and then s` to go to samv2 segmentation mode.

            In samv2 segmentation mode:
            Press `4` or `6` to increase or decrease object id number.
            Press `f` or `b` to add a foreground or background annotation point.
            Press `g` to add a common box points.
            Press `g` and then `shift` to remove common box points.
            Press `c` to clear annotation points.
            Press `r` to add annotation points to predictor.
            Press `r` and then `shift` to propagate annotations across frames.
            Press `e` to add box points to all frames.
            Press `w` and then `shift` to reset inference state.
            Press `t` and then `shift` to save video segmentation masks.

        """
        if key==13:
            return 1
        elif key==43:
            self.data_counter=(self.data_counter+self.speed_level) % self.N_frames 
            if self.video_segmentation_mode:
                self.video_ann_pts = []
                self.video_ann_pts_label = []
        elif key==45:
            self.data_counter=(self.data_counter-self.speed_level) % self.N_frames 
            if self.video_segmentation_mode:
                self.video_ann_pts = []
                self.video_ann_pts_label = []

        elif key==32:
            if self.thermal_minmax is not None:
                self.thermal_minmax = None
            else:
                vis_img_raw, thermal_img = self.get_vis_thermal_img_func(self.data_counter)
                self.thermal_minmax = (thermal_img.min(),thermal_img.max())
        elif key==ord("w"):
            if self.thr2visH is not None:
                self.warp_vis_img = not self.warp_vis_img
                # self.update_plot()

        elif key==ord("c") and not self.video_segmentation_mode:
            self.selected_pix_vis = []
            self.selected_pix_thermal = []
            if self.pixel_plot_fig is not None:
                self.pixel_plot_fig.data = []
                self.pixel_plot_counter = 0
                
            # self.update_plot()

        elif key==ord("d"):
            self.speed_level += 10
            self.speed_level = max(self.speed_level, 1)
            print('Speed level: {}'.format(self.speed_level))
        elif key==ord("a"):
            self.speed_level -= 10 
            self.speed_level = max(self.speed_level, 1)
            print('Speed level: {}'.format(self.speed_level))

        elif key == ord("o"):
            if self.thr2visH is not None:
                self.overlay_vis_thr = not self.overlay_vis_thr
                if not self.overlay_vis_thr:
                    cv2.destroyWindow("Overlay Images")
                    self.total_width_occ -= 640
                else:
                    cv2.namedWindow("Overlay Images")
                    cv2.setMouseCallback("Overlay Images", self.mouse_callback_overlay)
                    cv2.moveWindow("Overlay Images", self.total_width_occ, 50)
                    self.total_width_occ += 640
                # self.update_plot()
        
        elif key == ord("i"):
            self.alpha += 0.05
            self.alpha = min(max(self.alpha, 0.0), 1.0)
        elif key == ord("u"):
            self.alpha -= 0.05
            self.alpha = min(max(self.alpha, 0.0), 1.0)
        elif key == ord("h") and not self.shift_toggle:
            self.homography_mode = not self.homography_mode
            if self.homography_mode:
                self.selected_pix_vis = []
                self.selected_pix_thermal = []
            # self.update_plot()
        
        elif key == ord("s") and not self.shift_toggle:
            if self.homography_mode:
                print('Saving homography matrix to vis_thermal_H.npz')
                save_path = os.path.join(self.workspace_path, 'data', 'misc', f'vis_thermal_H_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.npz')
                np.savez(save_path, H_boson2bfly=self.thr2visH)
        
        elif key == ord("p"):
            if not self.show_segmentation_mask:
                self.plot_pixel_values_across_time(self.selected_pix_vis, self.selected_pix_thermal)
            else:
                self.plot_obj_pixel_values_across_time(self.selected_thr_mask_id)

        elif key == ord("m"):
            self.show_segmentation_mask = not self.show_segmentation_mask
            # self.update_plot()
        elif key == ord("v"):
            self.show_visible = not self.show_visible
            if not self.show_visible:
                cv2.destroyWindow("Visible Image")
            else:
                cv2.namedWindow("Visible Image")
                cv2.setMouseCallback("Visible Image", self.mouse_callback_vis)
            # self.update_plot()
        elif key == ord("t") and not self.video_segmentation_mode and not self.shift_toggle:
            self.show_thermal = not self.show_thermal
            if not self.show_thermal:
                cv2.destroyWindow("Thermal Image")
            else:
                cv2.namedWindow("Thermal Image")
                cv2.setMouseCallback("Thermal Image", self.mouse_callback_thermal)
            # self.update_plot()

        # If key is shift + s then save thermal frames to tmp
        elif key == ord("s") and self.shift_toggle:
            self.video_segmentation_mode = not self.video_segmentation_mode
            if self.video_segmentation_mode:
                print("Going to samv2 segmentation mode")
                self.save_thermal_frames_to_tmp()
                self.obj_id_num = 1
                self.show_segmentation_mask = True
                self.initialize_segmentation_predictor()
                print("First use 'g' to add a common box points. Then press 'e' to add box points to all frames.")
                print("Then refine predictions by adding point annotations to the predictor.\n")
                print("Press '4' or '6' to increase or decrease object id number.")
                print("Press 'f' or 'b' to add a foreground or background annotation point.")
            else:
                print("Exiting samv2 segmentation mode.")
                self.show_segmentation_mask = False
                self.obj_id_num = None

        if key == ord("h") and self.shift_toggle:
            print(self.loop_control.__doc__)

        if self.video_segmentation_mode:
            # print("Entering video loop")
            if key == ord("4"):
                self.obj_id_num += 1
                self.add_points_to_predictor = True
                # self.update_plot()
                print(f'Object id number: {self.obj_id_num}')
            if key == ord("6"):
                self.obj_id_num -= 1
                self.obj_id_num = max(self.obj_id_num, 1)
                self.add_points_to_predictor = True
                # self.update_plot()
                print(f'Object id number: {self.obj_id_num}')
            if key == ord("f"):
                self.video_ann_pts.append(self.mouse_location)
                self.video_ann_pts_label.append(1)
            if key == ord("b"):
                self.video_ann_pts.append(self.mouse_location)
                self.video_ann_pts_label.append(0)
            if key == ord("g") and not self.shift_toggle:
                if self.common_box_pts is None:
                    self.common_box_pts = {}
                    self.common_box_pts[self.obj_id_num] = []
                if len(self.common_box_pts[self.obj_id_num]) == 2:
                    self.common_box_pts[self.obj_id_num].pop(0)
                self.common_box_pts[self.obj_id_num].append(self.mouse_location)
            if key == ord("g") and self.shift_toggle:
                self.common_box_pts[self.obj_id_num] = []
            if key == ord("c"):
                self.video_ann_pts = []
                self.video_ann_pts_label = []
            if key == ord("r") and not self.shift_toggle:
                self.add_points_to_predictor = True
            if key == ord("r") and self.shift_toggle:
                self.propagate_anns_bool = True
                print("Propagating annotations across frames...")
                self.video_masks = self.propagate_anns_across_frames()
                self.show_segmentation_mask = True
            if key == ord("e"):
                self.add_box_points_to_all_frames(self.common_box_pts[self.obj_id_num], self.obj_id_num, self.skip_frames_for_seg)
            if key == ord("w") and self.shift_toggle:
                self.reset_inference_state()
                print("Inference state reset.")
            if key == ord("t") and self.shift_toggle:
                self.save_video_segmentation_masks()
                

        # Toggle shift key. This should be the last key to check
        if key == 225:
            self.shift_toggle = True
            print('Shift key pressed.')
        else:
            self.shift_toggle = False

        return 0
    
    def save_video_segmentation_masks(self):
        print('Saving video segmentation masks to data/daily_capture_thr_masks directory.')
        
        os.makedirs(os.path.join(self.workspace_path, 'data', 'daily_capture_thr_masks'), exist_ok=True)
        for key in tqdm(self.video_masks.keys()):
            i = int(key) * self.skip_frames_for_seg
            frame_fname = self.get_img_fname(i).split('/')[-1]
            thr_mask = self.video_masks[key]
            max_id = np.max(thr_mask)
            if os.path.exists(os.path.join(self.workspace_path, 'data', 'daily_capture_thr_masks', frame_fname)):
                presaved_mask = np.load(os.path.join(self.workspace_path, 'data', 'daily_capture_thr_masks', frame_fname))['mask']
                # Replace the locations where the new mask is not zero
                ids_getting_replaced = np.unique(presaved_mask[thr_mask != 0]).tolist()
                replaced_mask = presaved_mask.copy()
                for id in ids_getting_replaced:
                    replaced_mask[replaced_mask == id] = 0
                thr_mask[replaced_mask != 0] = replaced_mask[replaced_mask != 0] + max_id
            np.savez(os.path.join(self.workspace_path, 'data', 'daily_capture_thr_masks', frame_fname), mask=thr_mask)
        
    def read_thermal_segmentation_mask(self, frame_idx):
        fname = self.get_img_fname(frame_idx)
        fname_datetime = self.convert_filename_to_datetime(fname)
        # Find the mask file with the closest datetime name
        if self.all_thr_mask_names is None:
            self.all_thr_mask_names = os.listdir(os.path.join(self.workspace_path, 'data', 'daily_capture_thr_masks'))
        all_mask_names = [self.convert_filename_to_datetime(mask_name) for mask_name in self.all_thr_mask_names]
        closest_mask_name = all_mask_names[np.argmin(np.abs(all_mask_names - fname_datetime))]
        datetime_diff = np.abs(closest_mask_name - fname_datetime)
        # Check if the time difference is less than 1 day
        if datetime_diff > np.timedelta64(1, 'D'):
            return None
        mask = np.load(os.path.join(self.workspace_path, 'data', 'daily_capture_thr_masks', self.convert_datetime_to_filename(closest_mask_name)))
        return mask['mask']

    def compute_homography(self, vis_img, thermal_img):
        # Use selected visible and thermal pixels to compute homography
        if len(self.selected_pix_vis) == len(self.selected_pix_thermal) and len(self.selected_pix_vis) > 3:
            vis_pts = np.array(self.selected_pix_vis).astype(np.float32)
            thermal_pts = np.array(self.selected_pix_thermal).astype(np.float32)
            self.thr2visH = cv2.findHomography(thermal_pts, vis_pts, cv2.RANSAC)[0]
            print('Homography matrix:')
            print(self.thr2visH)
        elif len(self.selected_pix_vis) < 3:
            print('Select at least 4 points in each image.')
        else:
            print(f'Visible image has {len(self.selected_pix_vis)} points and thermal image has {len(self.selected_pix_thermal)} points. Select the same number of points in both images.')

    
    def additional_thermal_processing(self, thermal_img, minmax):
        color = (255,255,255) if thermal_img.dtype=='uint8' else (1,1,1)
        cv2.putText(thermal_img,'frame {}'.format(self.data_counter),
                (10,20), FONT, 0.75,color,2,cv2.LINE_AA)
        
        # Display curr_frame_min and curr_frame_max
        cv2.putText(thermal_img,'Min: {}'.format(minmax[0]),(30,50), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(thermal_img,'Max: {}'.format(minmax[1]),(30,70), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)

        return thermal_img
    
    def update_plot(self):
        # vis_img_raw = self.get_vis_img_func(self.data_counter)
        # thermal_img_raw = self.get_thermal_img_func(self.data_counter)
        vis_img_raw, thermal_img_raw = self.get_vis_thermal_img_func(self.data_counter)
        if vis_img_raw is not None:
            vis_img = (self.process_vis_img_func(vis_img_raw.copy())[:, ::-1]).astype(np.uint8)
            if len(self.selected_pix_vis)>0:
                for i in range(len(self.selected_pix_vis)):
                    cv2.drawMarker(vis_img,self.selected_pix_vis[i],(255,255,255),markerType=cv2.MARKER_STAR,markerSize=10,thickness=2)
            if self.warp_vis_img:
                vis_img = skimage.transform.warp(vis_img.astype(np.uint8), self.thr2visH, output_shape=(512, 640))
                vis_img = (vis_img*255).astype(np.uint8)
            # if self.show_segmentation_mask:
            #     masks = self.get_visible_segmentation_mask(vis_img)
            #     vis_img = self.draw_segmentation_mask(vis_img, masks)
            if self.show_visible:
                cv2.imshow("Visible Image",vis_img)

        if thermal_img_raw is not None:
            minmax = (thermal_img_raw.min(),thermal_img_raw.max())
            thermal_img, _ = self.process_thermal_img_func(thermal_img_raw, minmax=self.thermal_minmax)
            thermal_img = self.additional_thermal_processing(thermal_img, minmax)
            if len(self.selected_pix_thermal)>0:
                for i in range(len(self.selected_pix_thermal)):
                    cv2.drawMarker(thermal_img,self.selected_pix_thermal[i],(255,255,255),markerType=cv2.MARKER_STAR,markerSize=10,thickness=2)
            if self.video_segmentation_mode:
                if self.add_points_to_predictor:
                    if len(self.video_ann_pts) > 0:
                        print(f'Adding annotations to predictor...')
                        obj_ids, obj_masks = self.add_anns_to_predictor(self.data_counter//self.skip_frames_for_seg, self.obj_id_num, self.video_ann_pts, self.video_ann_pts_label, \
                                                                        box_pts=np.array(self.common_box_pts[self.obj_id_num]).flatten())
                        thermal_img = self.overlay_object_masks(thermal_img, obj_masks, obj_ids)
                    self.add_points_to_predictor = False
                    self.video_ann_pts = []
                    self.video_ann_pts_label = []
                else:
                    ann_color = [(255, 0, 0), (0, 255, 0)]
                    for i in range(len(self.video_ann_pts)):
                        cv2.drawMarker(thermal_img, tuple(self.video_ann_pts[i]), ann_color[self.video_ann_pts_label[i]], markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)
                    if self.common_box_pts is not None:
                        for key in self.common_box_pts.keys():
                            if len(self.common_box_pts[key]) == 2:
                                cv2.rectangle(thermal_img, tuple(self.common_box_pts[key][0]), tuple(self.common_box_pts[key][1]), (255, 255, 255), 2)
                            for i in range(len(self.common_box_pts[key])):
                                cv2.drawMarker(thermal_img, tuple(self.common_box_pts[key][i]), (255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)

            if self.show_segmentation_mask:
                thermal_img = self.draw_segmentation_mask_thermal(thermal_img)

            if self.show_thermal:
                cv2.imshow("Thermal Image",thermal_img)
        
        if self.overlay_vis_thr:
            thermal_img_jet = thermal_img_raw.copy()
            thermal_img_jet = (thermal_img_jet - np.min(thermal_img_jet)) / (np.max(thermal_img_jet) - np.min(thermal_img_jet))
            thermal_img_jet = (thermal_img_jet * 255).astype(np.uint8)
            thermal_img_jet = cv2.applyColorMap(thermal_img_jet, cv2.COLORMAP_JET)
            vis_img = (self.process_vis_img_func(vis_img_raw.copy())[:, ::-1]).astype(np.uint8)
            vis_img = skimage.transform.warp(vis_img.astype(np.uint8), self.thr2visH, output_shape=(512, 640))
            vis_img = (vis_img*255).astype(np.uint8)
            overlay = cv2.addWeighted(vis_img.astype(np.uint8), self.alpha, thermal_img_jet.astype(np.uint8), 1 - self.alpha, 0).astype(np.uint8)
            cv2.putText(overlay,'alpha: {:.2f}'.format(self.alpha),(10,20), FONT, 0.75,(255,255,255),2,cv2.LINE_AA)
            if self.homography_mode:
                cv2.putText(overlay, 'Press "s" to save homography matrix.', (10, 40), FONT, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            if len(self.selected_pix_thermal)>0:
                for i in range(len(self.selected_pix_thermal)):
                    cv2.drawMarker(overlay,self.selected_pix_thermal[i],(255,255,255),markerType=cv2.MARKER_STAR,markerSize=10,thickness=2)
            cv2.imshow("Overlay Images", overlay)
        
        if self.homography_mode and vis_img_raw is not None and thermal_img_raw is not None:
            self.compute_homography(vis_img, thermal_img)
    
    def mouse_callback_vis(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_pix_vis.append((x,y))
            # Convert pix coordinate in visible to pix coordinate in thermal given thr2vis homography
            if self.thr2visH is not None and not self.homography_mode:
                vis2thrH = np.linalg.inv(self.thr2visH)
                visible_point_homogeneous = np.array([x, y, 1])
                thermal_point_homogeneous = np.dot(vis2thrH, visible_point_homogeneous)            
                thermal_coord = (thermal_point_homogeneous[:2] / thermal_point_homogeneous[2]).astype(int)
                self.selected_pix_thermal.append((thermal_coord[0], thermal_coord[1]))

            # Update vis image drawing the selected pixels. Add a star marker
            self.update_plot()

    
    def mouse_callback_thermal(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.show_segmentation_mask:
                self.selected_pix_thermal.append((x,y))
                # Convert pix coordinate in thermal to pix coordinate in visible given thr2vis homography
                if self.thr2visH is not None and not self.homography_mode:
                    thermal_point_homogeneous = np.array([x,y,1])
                    visible_point_homogeneous = np.dot(self.thr2visH, thermal_point_homogeneous)
                    visible_coord = (visible_point_homogeneous[:2] / visible_point_homogeneous[2]).astype(int)
                    self.selected_pix_vis.append((visible_coord[0], visible_coord[1]))
            else:
                segmask = self.read_thermal_segmentation_mask(self.data_counter)
                if segmask is not None:
                    self.selected_thr_mask_id.append(segmask[y, x])
                
            # Update thermal image drawing the selected pixels. Add a star marker
            self.update_plot()
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_location = [x, y]

    def mouse_callback_overlay(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_pix_thermal.append((x,y))
            thermal_point_homogeneous = np.array([x,y,1])
            visible_point_homogeneous = np.dot(self.thr2visH, thermal_point_homogeneous)
            visible_coord = (visible_point_homogeneous[:2] / visible_point_homogeneous[2]).astype(int)
            self.selected_pix_vis.append((visible_coord[0], visible_coord[1]))
            self.update_plot()

    def get_visible_segmentation_mask(self, vis_img):
        # Check if the mask exists
        if os.path.exists(os.path.join(self.workspace_path, 'data', 'misc', 'daily_capture_vis_masks', self.get_img_fname(self.data_counter))):
            masks = np.load(os.path.join(self.workspace_path, 'data', 'misc', 'daily_capture_vis_masks', self.get_img_fname(self.data_counter)))
        else:
            masks = self.run_auto_img_mask_generator(vis_img)
            print(len(masks))
        return masks
    
    def get_vis_mask_from_thr_mask(self, thr_mask, vis_img_shape):
        # Use the homography matrix to convert the thermal mask to visible mask
        vis_mask = np.zeros(vis_img_shape)
        non_zero_locs = np.where(thr_mask != 0)
        thr_locs_homogeneous = np.vstack((non_zero_locs[1], non_zero_locs[0], np.ones_like(non_zero_locs[0])))
        vis_locs_homogeneous = np.dot(self.thr2visH, thr_locs_homogeneous)
        vis_locs = (vis_locs_homogeneous[:2] / vis_locs_homogeneous[2]).astype(int)
        vis_mask[vis_locs[1], vis_locs[0]] = thr_mask[non_zero_locs]
        return vis_mask

    
    # def draw_segmentation_mask(self, vis_img, masks):
        
    #     if len(masks) == 0:
    #         return vis_img
    #     sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)

    #     for ann in sorted_anns:
    #         mask = ann['segmentation']
    #         color_mask = np.zeros_like(vis_img)
    #         random_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
    #         color_mask[mask] = random_color
    #         vis_img = cv2.addWeighted(vis_img, 1, color_mask, 0.5, 0)
    #     return vis_img

    def draw_segmentation_mask_thermal(self, thermal_img):
        masks = None
        if self.video_masks is not None:
            if self.data_counter//self.skip_frames_for_seg in self.video_masks:
                masks = self.video_masks[self.data_counter//self.skip_frames_for_seg]
        else:
            masks = self.read_thermal_segmentation_mask(self.data_counter)
        if masks is not None:
            unique_obj_ids = np.unique(masks)
            plt_colors = plt.cm.get_cmap('tab20')(np.arange(len(unique_obj_ids)))
            color_mask = np.zeros_like(thermal_img)
            for i in range(len(unique_obj_ids)):
                if unique_obj_ids[i] == 0:
                    continue
                mask = (masks == unique_obj_ids[i])
                color_mask[mask] = plt_colors[i][:3] * 255 if unique_obj_ids[i] not in self.selected_thr_mask_id else (255, 255, 255)
            # only blend where color_mask is not black
            non_black_locs = np.any(color_mask != 0, axis=-1)
            thermal_img[non_black_locs] = thermal_img[non_black_locs] * 0.5 + color_mask[non_black_locs] * 0.5
            
            # thermal_img = cv2.addWeighted(thermal_img, 0.5, color_mask, 0.5, 0)
        return thermal_img
    

    def overlay_object_masks(self, thermal_img, obj_masks, obj_ids):
        for i in range(len(obj_masks)):
            mask = (obj_masks[i] > 0.0)[0]
            color_mask = np.zeros_like(thermal_img)
            # Get a color from a colormap given the object id
            cmap_color = (np.array(plt.cm.get_cmap('tab20')(obj_ids[i]))[:3] * 255).astype(np.uint8)[::-1]
            color_mask[mask] = cmap_color
            thermal_img = cv2.addWeighted(thermal_img, 0.5, color_mask, 0.5, 0)
        return thermal_img
    
    def save_thermal_frames_to_tmp(self):
        if not self.saved_thermal_frames_tmp:
            print('Saving thermal frames (1 for every self.skip_frames_for_seg frames) to tmp directory.')
            # Delete tmp directory
            if os.path.exists(os.path.join(self.workspace_path, 'data', 'tmp')):
                os.system(f'rm -r {os.path.join(self.workspace_path, "data", "tmp")}')
            os.makedirs(os.path.join(self.workspace_path, 'data', 'tmp'), exist_ok=True)
            # Interleave 5 frames as default and save them to tmp directory
            count = 0
            for i in tqdm(range(0, self.N_frames, self.skip_frames_for_seg)):
                vis_img, thermal_img = self.get_vis_thermal_img_func(i)
                minmax = (thermal_img.min(),thermal_img.max())
                thermal_img, _ = self.process_thermal_img_func(thermal_img, minmax=minmax)
                cv2.imwrite(os.path.join(self.workspace_path, 'data', 'tmp', f'{count:05d}.jpeg'), thermal_img)
                count += 1
            self.saved_thermal_frames_tmp = True
        else:
            print('Thermal frames already saved to tmp directory.')
        
    def convert_filename_to_datetime(self, filename):
        filename = filename.split('/')[-1].split('.')[0].split('_')
        filename[1] = filename[1].replace('-', ':')
        return np.datetime64(' '.join(filename))
    
    def convert_datetime_to_filename(self, datetime):
        # Convert numpy datetime to string
        datetime_str = str(datetime).replace('T', '_').replace(':', '-').replace(' ', '_')
        return datetime_str + '.npz'

    def read_pixel_values_for_allfiles(self, selected_pix_vis, selected_pix_thermal):
        pix_value_vis = []
        pix_value_thermal = []
        pix_datetime = []
        selected_pix_vis = np.array(selected_pix_vis)
        selected_pix_thermal = np.array(selected_pix_thermal)
        print(selected_pix_vis.shape, selected_pix_thermal.shape)
        for i in range(self.N_frames):
            try:
                vis_img, thermal_img, img_datetime = self.get_vis_thermal_img_func(i, return_datetime=True)
                pix_datetime.append(img_datetime)
                if vis_img is not None:
                    vis_img = self.process_vis_img_func(vis_img)
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


    def plot_pixel_values_across_time(self, selected_pix_vis, selected_pix_thermal):
        pix_value_vis, pix_value_thermal, pix_datetime = self.read_pixel_values_for_allfiles(selected_pix_vis, selected_pix_thermal)
        print(pix_value_vis.shape, pix_value_thermal.shape, pix_datetime.shape)

        if self.pixel_plot_fig is None:
            # Create new subplots if figure doesn't exist
            self.pixel_plot_fig = make_subplots(rows=2, cols=2, 
                                            subplot_titles=('Blue Channel', 'Green Channel', 'Red Channel', 'Thermal Camera Pixel Values'))
            self.pixel_plot_counter = 0

        colors = ['blue', 'green', 'red']
        
        # Add traces for visible camera BGR pixel values
        if pix_value_vis.size != 0:
            for i in range(self.pixel_plot_counter, pix_value_vis.shape[1]):
                for j, color in enumerate(colors):
                    self.pixel_plot_fig.add_trace(
                        go.Scatter(x=pix_datetime, y=pix_value_vis[:, i, j], 
                                name=f'Pixel {selected_pix_vis[i]}',
                                hovertemplate='<b>Date Time</b>: %{x}<br>' +
                                                f'<b>{color.capitalize()} Value</b>: %{{y}}<br>' +
                                                '<b>Pixel</b>: ' + str(selected_pix_vis[i])),
                        row=int(j/2)+1, col=(j%2)+1
                    )

        # Add traces for thermal camera pixel values
        if pix_value_thermal.size != 0:
            for i in range(self.pixel_plot_counter, pix_value_thermal.shape[1]):
                self.pixel_plot_fig.add_trace(
                    go.Scatter(x=pix_datetime, y=pix_value_thermal[:, i], 
                            name=f'Pixel {selected_pix_thermal[i]}',
                            hovertemplate='<b>Date Time</b>: %{x}<br>' +
                                            '<b>Pixel Value</b>: %{y}<br>' +
                                            '<b>Pixel</b>: ' + str(selected_pix_thermal[i])),
                    row=2, col=2
                )

        self.pixel_plot_counter = max(pix_value_vis.shape[1], pix_value_thermal.shape[1])
        
        # Update layout
        self.pixel_plot_fig.update_layout(
            title_text="Pixel Values Across Time",
            hovermode="x unified"
        )
        
        # Update x and y axis labels
        for i in range(4):
            self.pixel_plot_fig.update_xaxes(title_text="Date Time", row=i+1, col=1)
            if i < 3:
                self.pixel_plot_fig.update_yaxes(title_text=f"{colors[i].capitalize()} Value", row=i+1, col=1)
            else:
                self.pixel_plot_fig.update_yaxes(title_text="Thermal Value", row=i+1, col=1)

        # Show the plot
        self.pixel_plot_fig.show()
    
    def read_mean_obj_id_values_allfiles(self, selected_obj_ids):
        print("Reading pixel values for plotting....")
        vis_values_obj_id = {obj_id : [] for obj_id in selected_obj_ids}
        thermal_values_obj_id = {obj_id : [] for obj_id in selected_obj_ids}
        obj_id_datetime = {obj_id : [] for obj_id in selected_obj_ids}
        for i in tqdm(range(self.N_frames)):
            try:
                vis_img, thermal_img, img_datetime = self.get_vis_thermal_img_func(i, return_datetime=True)
                if vis_img is not None:
                    vis_img = self.process_vis_img_func(vis_img)
                thr_mask = self.read_thermal_segmentation_mask(i)
                if self.thr2visH is not None:
                    vis_mask = self.get_vis_mask_from_thr_mask(thr_mask, vis_img.shape[:2])
                if thr_mask is not None:
                    unique_obj_ids = np.unique(thr_mask).tolist()
                    for obj_id in selected_obj_ids:
                        if obj_id in unique_obj_ids:
                            if self.thr2visH is not None:
                                vis_values_obj_id[obj_id].append(np.mean(vis_img[vis_mask == obj_id], axis=0))
                            thermal_values_obj_id[obj_id].append(np.mean(thermal_img[thr_mask == obj_id]))
                            obj_id_datetime[obj_id].append(img_datetime)
            except Exception as e:
                print(f'Error reading pixel values for frame {i}: {e}')
        for obj_id in selected_obj_ids:
            if self.thr2visH is not None:
                vis_values_obj_id[obj_id] = np.stack(vis_values_obj_id[obj_id])
            thermal_values_obj_id[obj_id] = np.array(thermal_values_obj_id[obj_id])
            obj_id_datetime[obj_id] = np.array(obj_id_datetime[obj_id])
        return vis_values_obj_id, thermal_values_obj_id, obj_id_datetime

    def plot_obj_pixel_values_across_time(self, selected_obj_ids):
        vis_values_obj_id, thermal_values_obj_id, obj_id_datetime = self.read_mean_obj_id_values_allfiles(selected_obj_ids)
        # print(vis_values_obj_id.keys(), thermal_values_obj_id.keys(), obj_id_datetime.keys())
        obj_plot_fig = make_subplots(rows=2, cols=2, subplot_titles=('Blue Channel', 'Green Channel', 'Red Channel', 'Thermal Camera Object Values'))

        colors = ['blue', 'green', 'red']
        for obj_id in selected_obj_ids:
            if self.thr2visH is not None:
                print("Vis values shape:", vis_values_obj_id[obj_id].shape)
                for j, color in enumerate(colors):
                    obj_plot_fig.add_trace(
                        go.Scatter(x=obj_id_datetime[obj_id], y=vis_values_obj_id[obj_id][:, j], 
                                name=f'Object {obj_id}',
                                hovertemplate='<b>Date Time</b>: %{x}<br>' +
                                                f'<b>{color.capitalize()} Value</b>: %{{y}}<br>' +
                                                '<b>Object</b>: ' + str(obj_id)),
                        row=int(j/2)+1, col=(j%2)+1
                    )
            print("Thermal values shape:", thermal_values_obj_id[obj_id].shape)
            obj_plot_fig.add_trace(
                go.Scatter(x=obj_id_datetime[obj_id], y=thermal_values_obj_id[obj_id], 
                        name=f'Object {obj_id}',
                        hovertemplate='<b>Date Time</b>: %{x}<br>' +
                                        '<b>Object Value</b>: %{y}<br>' +
                                        '<b>Object</b>: ' + str(obj_id)),
                row=2, col=2
            )

        # Update layout
        obj_plot_fig.update_layout(
            title_text="Object Values Across Time",
            hovermode="x unified"
        )

        # Update x and y axis labels
        for i in range(4):
            obj_plot_fig.update_xaxes(title_text="Date Time", row=i+1, col=1)
            if i < 3:
                obj_plot_fig.update_yaxes(title_text=f"{colors[i].capitalize()} Value", row=i+1, col=1)
            else:
                obj_plot_fig.update_yaxes(title_text="Thermal Value", row=i+1, col=1)

        # Show the plot
        obj_plot_fig.show()


    def play_video(self,show_frame_number=True):
        try:
            self.total_width_occ = 0
            vis_img, thermal_img = self.get_vis_thermal_img_func(self.data_counter)
            if vis_img is not None:
                cv2.namedWindow("Visible Image")
                vis_img = self.process_vis_img_func(vis_img)
                self.total_width_occ += int(vis_img.shape[1]*1.2)
                cv2.setMouseCallback("Visible Image",self.mouse_callback_vis)

            if thermal_img is not None:
                cv2.namedWindow("Thermal Image")
                cv2.moveWindow("Thermal Image",self.total_width_occ+10,50)
                # thermal_img = self.get_thermal_img_func(self.data_counter)
                self.total_width_occ += int(thermal_img.shape[1]*1.2)
                cv2.setMouseCallback("Thermal Image",self.mouse_callback_thermal) 
            
            while(True):
                self.update_plot()
                key = cv2.waitKey()
                if self.loop_control(key): 
                    break

        except Exception:
            traceback.print_exc()
            print('Closing camera thread.')    
        cv2.destroyAllWindows()

class videoPlayer():
    def __init__(self,frame_func,N_frames,resize_factor=None):
        self.frame_func    = frame_func
        self.data_counter  = 0
        self.resize_factor = resize_factor
        self.gamma         = 1
        self.N_frames      = N_frames
    #     self.frame_recordings = None
    #     self.timestamps = None

    # def set_frame_recordings(self,frame_recordings):
    #     self.frame_recordings = frame_recordings

    # def set_timestamps(self,timestamps):
    #     self.timestamps = timestamps

        # -------- Loop control
    def loop_control(self,key):
        if key==13:
            return 1
        if key==43:
            self.data_counter=(self.data_counter+1) % self.N_frames 
        if key==45:
            self.data_counter=(self.data_counter-1) % self.N_frames 
        if key==40:
            self.gamma-= 0.1
        if key==41:
            self.gamma+= 0.1
        return 0

    def additional_loop_control(self,key):
        return 0
    
    def correct_gamma_show(self, img):
        img = img.astype('float32')**(1/self.gamma)
        if self.gamma!=1:
            cv2.putText(img,'g={:.1f}'.format(self.gamma),
                        (img.shape[1]-150,20), FONT, 0.75,(1,1,1),2,cv2.LINE_AA)
        return img
    
    def get_frame_show(self):
        return self.frame_func(self.data_counter)

    def play_video(self, move_window=1,show_frame_number=True):
        try:
            cv2.namedWindow("frame")  
            if move_window:
                cv2.moveWindow("frame",1920*2+5,0)
            
            while(True):
                frame_show = self.get_frame_show()

                if self.resize_factor is not None:
                    frame_show = cv2.resize(frame_show,None,fx = self.resize_factor, fy = self.resize_factor)


                frame_show = self.correct_gamma_show(frame_show)
                color = (255,255,255) if frame_show.dtype=='uint8' else (1,1,1)
                if show_frame_number:
                    cv2.putText(frame_show,'frame {}'.format(self.data_counter),
                            (10,20), FONT, 0.75,color,2,cv2.LINE_AA)
                
                cv2.imshow("frame",frame_show)
                key = cv2.waitKey()
                #if key!=-1:
                #    print(key)
                self.additional_loop_control(key)
                if self.loop_control(key): 
                    break

        except Exception:
            traceback.print_exc()
            print('Closing camera thread.')    
            # cleanup
        cv2.destroyAllWindows()
        
    def export_video(self,filename,is_flip_RGB = 0, gamma = 1.0, FPS=30, show_frame_number=0):
        IMG_H, IMG_W  = self.get_frame_show().shape[:2]
        fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        video = cv2.VideoWriter(filename,
        fourcc,    #cv2.VideoWriter_fourcc('X','2','6','4'),
        FPS,
        (IMG_W, IMG_H))
        self.gamma = gamma
            
        for i in range(self.N_frames):
            self.data_counter = i
            frame_show        = self.get_frame_show()
            if is_flip_RGB:
                frame_show = frame_show[:,:,::-1]
            
            if gamma!=1:
                frame_show = self.correct_gamma_show(frame_show)
            # frame is [0,1]
            if frame_show.dtype!='uint8':
                frame_show        = (frame_show.clip(0,1)*255).astype('uint8')
            if show_frame_number:
                color = (255,255,255)
                cv2.putText(frame_show,'frame {}'.format(self.data_counter), (10,20), FONT, 0.75,color,2,cv2.LINE_AA)

            if frame_show.ndim==2:
                frame_show = np.stack(((frame_show, ) * 3),axis=-1)
            video.write(frame_show)
        video.release()
        print('done exporting video to {}'.format(filename))
    
    def export_raw_as_pkl(self,filename, frame_recording, timestamps, **kwargs):
        recordings_dict = {'frame_recording':frame_recording,
                            'timestamps':timestamps}
        recordings_dict.update(kwargs)
        
        with open(filename,'wb') as f:
            pickle.dump(recordings_dict,f)
        print('done exporting raw data to {}'.format(filename))

    def export_raw_as_npz(self,filename, frame_recording, timestamps):
        np.savez_compressed(filename, frame_recording=frame_recording, timestamps=timestamps)
        print('done exporting raw data to {}'.format(filename))