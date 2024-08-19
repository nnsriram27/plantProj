import numpy as np
import cv2

import traceback
import pickle
import skimage
import os

from lib.thermal_utils import thermal_cam_operation
from lib.image_processing import vis_cam_operation
import datetime

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
        self.thr2visH = thr2Vis_HMatrix
        self.speed_level = 1
        self.overlay_vis_thr = False
        self.alpha = 0.5
        self.pixel_plot_fig = None
        self.pixel_plot_counter = 0

        self.warp_vis_img = False
        self.homography_mode = False
        self.workspace_path = workspace_path

    def plot_pixel_values_across_time(self, selected_pix_vis, selected_pix_thermal):
        pass

    # -------- Loop control -------- #
    def loop_control(self,key):
        if key==13:
            return 1
        if key==43:
            self.data_counter=(self.data_counter+self.speed_level) % self.N_frames 
        if key==45:
            self.data_counter=(self.data_counter-self.speed_level) % self.N_frames 

        if key==32:
            if self.thermal_minmax is not None:
                self.thermal_minmax = None
            else:
                vis_img_raw, thermal_img = self.get_vis_thermal_img_func(self.data_counter)
                self.thermal_minmax = (thermal_img.min(),thermal_img.max())
        if key==ord("w"):
            if self.thr2visH is not None:
                self.warp_vis_img = not self.warp_vis_img
                self.update_plot()

        if key==ord("c"):
            self.selected_pix_vis = []
            self.selected_pix_thermal = []
            if self.pixel_plot_fig is not None:
                self.pixel_plot_fig.data = []
                self.pixel_plot_counter = 0
                
            self.update_plot()

        if key==ord("d"):
            self.speed_level += 10
            self.speed_level = max(self.speed_level, 1)
            print('Speed level: {}'.format(self.speed_level))
        if key==ord("a"):
            self.speed_level -= 10 
            self.speed_level = max(self.speed_level, 1)
            print('Speed level: {}'.format(self.speed_level))

        if key == ord("o"):
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
                self.update_plot()
        
        if key == ord("i"):
            self.alpha += 0.05
            self.alpha = min(max(self.alpha, 0.0), 1.0)
        if key == ord("u"):
            self.alpha -= 0.05
            self.alpha = min(max(self.alpha, 0.0), 1.0)
        if key == ord("h"):
            self.homography_mode = not self.homography_mode
            if self.homography_mode:
                self.selected_pix_vis = []
                self.selected_pix_thermal = []
            self.update_plot()
        
        if key == ord("s"):
            if self.homography_mode:
                print('Saving homography matrix to vis_thermal_H.npz')
                save_path = os.path.join(self.workspace_path, 'data', 'misc', f'vis_thermal_H_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.npz')
                np.savez(save_path, H_boson2bfly=self.thr2visH)
        
        if key == ord("p"):
            self.plot_pixel_values_across_time(self.selected_pix_vis, self.selected_pix_thermal)

        return 0

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
            cv2.imshow("Visible Image",vis_img)

        if thermal_img_raw is not None:
            minmax = (thermal_img_raw.min(),thermal_img_raw.max())
            thermal_img, _ = self.process_thermal_img_func(thermal_img_raw, minmax=self.thermal_minmax)
            thermal_img = self.additional_thermal_processing(thermal_img, minmax)
            if len(self.selected_pix_thermal)>0:
                for i in range(len(self.selected_pix_thermal)):
                    cv2.drawMarker(thermal_img,self.selected_pix_thermal[i],(255,255,255),markerType=cv2.MARKER_STAR,markerSize=10,thickness=2)
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
            self.selected_pix_thermal.append((x,y))
            # Convert pix coordinate in thermal to pix coordinate in visible given thr2vis homography
            if self.thr2visH is not None and not self.homography_mode:
                thermal_point_homogeneous = np.array([x,y,1])
                visible_point_homogeneous = np.dot(self.thr2visH, thermal_point_homogeneous)
                visible_coord = (visible_point_homogeneous[:2] / visible_point_homogeneous[2]).astype(int)
                self.selected_pix_vis.append((visible_coord[0], visible_coord[1]))
                
            # Update thermal image drawing the selected pixels. Add a star marker
            self.update_plot()

    def mouse_callback_overlay(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_pix_thermal.append((x,y))
            thermal_point_homogeneous = np.array([x,y,1])
            visible_point_homogeneous = np.dot(self.thr2visH, thermal_point_homogeneous)
            visible_coord = (visible_point_homogeneous[:2] / visible_point_homogeneous[2]).astype(int)
            self.selected_pix_vis.append((visible_coord[0], visible_coord[1]))
            self.update_plot()

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


