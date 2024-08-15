import numpy as np
import cv2

import traceback
import pickle

from lib.thermal_utils import thermal_cam_operation
from lib.image_processing import vis_cam_operation

FONT = cv2.FONT_HERSHEY_SIMPLEX

class videoPlayerVisThermal():
    def __init__(self, N_frames, get_vis_img_func=None, thermal_img_func= None, process_vis_img_func = vis_cam_operation, process_thermal_img_func = thermal_cam_operation):
        self.N_frames      = N_frames
        self.data_counter  = 0
        self.get_vis_img_func = get_vis_img_func
        self.get_thermal_img_func = thermal_img_func
        self.process_vis_img_func = process_vis_img_func
        self.process_thermal_img_func = process_thermal_img_func
        self.thermal_minmax = None

    # -------- Loop control -------- #
    def loop_control(self,key):
        if key==13:
            return 1
        if key==43:
            self.data_counter=(self.data_counter+1) % self.N_frames 
        if key==45:
            self.data_counter=(self.data_counter-1) % self.N_frames 
        return 0

    def additional_loop_control(self,key, vis_img, thermal_img):
        if key==32:
            if self.thermal_minmax is not None:
                self.thermal_minmax = None
            else:
                self.thermal_minmax = (thermal_img.min(),thermal_img.max())

    def play_video(self,show_frame_number=True):
        try:
            if self.get_vis_img_func is not None:
                cv2.namedWindow("Visible Image")
            if self.get_thermal_img_func is not None:
                cv2.namedWindow("Thermal Image")  
                cv2.moveWindow("Thermal Image",1920*2+5,0)
            
            while(True):
                if self.get_vis_img_func is not None:
                    vis_img = self.get_vis_img_func(self.data_counter)
                    vis_img = self.process_vis_img_func(vis_img)
                    cv2.imshow("Visible Image",vis_img)
                if self.get_thermal_img_func is not None:
                    thermal_img = self.get_thermal_img_func(self.data_counter)
                    thermal_img = self.process_thermal_img_func(thermal_img, minmax=self.thermal_minmax)
                    
                    color = (255,255,255) if thermal_img.dtype=='uint8' else (1,1,1)
                    if show_frame_number:
                        cv2.putText(thermal_img,'frame {}'.format(self.data_counter),
                                (10,20), FONT, 0.75,color,2,cv2.LINE_AA)
                    
                    # Display curr_frame_min and curr_frame_max
                    cv2.putText(thermal_img,'Min: {}'.format(thermal_img),(30,50), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)
                    cv2.putText(thermal_img,'Max: {}'.format(thermal_img),(30,70), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)

                    cv2.imshow("Thermal Image",thermal_img)
                
                key = cv2.waitKey()
                self.additional_loop_control(key, vis_img, thermal_img)
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


