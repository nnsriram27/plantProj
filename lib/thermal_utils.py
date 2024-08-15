import cv2
import numpy as np
from threading import Thread
from flirpy.camera.boson import Boson
from BosonSDK import *

FONT = cv2.FONT_HERSHEY_SIMPLEX
class FrameThreadGeneral(Thread):
    def __init__(self, cam, copy=True):
        super(FrameThreadGeneral, self).__init__()
        self.cam = cam
        self.running = True
        self.frame = np.zeros((100,100,1),dtype='uint8')
        self.frame_counter = 0

    def run(self):
        while self.running:
            image,ret = self.cam.get_frame()
            if ret == True:
                self.frame = image
                self.frame_counter =  (self.frame_counter + 1) % 256

    def read(self):
        return self.frame, self.frame_counter
    def stop(self):
        self.cam.stop_video()
        self.running = False
        
class CaptureThermal():
    # MAX_FRAMES = 1200
    def __init__(self, cam, cam_name, N_frames = None, VIS_H = 512, VIS_W = 640, vis_capture = False, is_color = False):
        # super(CaptureThread, self).__init__()
        self.cam         = cam
        self.cam_name    = cam_name
        self.vis_capture = vis_capture    
        self.N_frames    = N_frames #if N_frames is not None else self.MAX_FRAMES
        # In case we want to capture indefinitely (only possible if you're visualizing)
        # if N_frames is None and self.vis_capture:
        #     self.N_frames = None
        self.VIS_H       = VIS_H
        self.VIS_W       = VIS_W
        self.is_color = is_color
        self.data        = [] #np.zeros((N_frames,IMG_H,IMG_W),dtype=dtype)
        self.timestamps  = []# np.zeros((N_frames,4),dtype=int)
        self.succ_frames = [] # np.zeros((N_frames,),dtype=bool)
        self.use_minmax = False
        self.minmax = None

        self.stop_video = False
        self.record_video = False
        
    def run(self):
        self.frame_counter = 0
        self.cam.start_video()
        # self.succ_frames[:] = True
        if self.vis_capture:
            print('CaptureThread: starting visualization')
            cv2.namedWindow(self.cam_name)
            # self.fig = plt.figure()
            # self.ax = self.fig.add_subplot(111)
        if self.N_frames is not None:
            print('CaptureThread: recording {} frames'.format(self.N_frames))
            self.record_video = True
        while True:
            try:
                img_data,timestamp = self.cam.get_frame()
                if not self.vis_capture:
                    self.data.append(img_data)
                    self.timestamps.append(timestamp)
                    self.succ_frames.append(True)
                else:
                    self.show_capture(img_data.copy())
                    key = cv2.waitKey(1)
                    if key == 27:
                        self.stop_video = True
                    elif key == 32:
                        self.use_minmax = not self.use_minmax
                        if self.use_minmax:
                            print('CaptureThread: using minmax')
                        else:
                            print('CaptureThread: not using minmax')
                    elif key == ord('r'):
                        self.record_video = not self.record_video
                        if self.record_video:
                            print('CaptureThread: recording video')
                        else:
                            print('CaptureThread: stopped recording video')
                            self.stop_video = True
                    if self.record_video:
                        self.data.append(img_data)
                        self.timestamps.append(timestamp)
                        self.succ_frames.append(True)

            except:
                self.data.append(np.zeros((self.cam.IMG_H,self.cam.IMG_W),dtype='uint16'))
                self.timestamps.append(np.zeros((4,),dtype=int))
                self.succ_frames.append(False)
                print('CaptureThread: frame {} failed'.format(self.frame_counter))
            
            self.frame_counter += 1
            if self.N_frames is not None and self.frame_counter >= self.N_frames:
                self.stop_video = True
                self.record_video = False
            
            if self.stop_video:
                self.cam.stop_video()
                if self.vis_capture:
                    cv2.destroyWindow(self.cam_name)
                    # plt.close(self.fig)
                break

    
    def show_capture(self, frame):
        curr_frame_min, curr_frame_max = frame.min(), frame.max()
        if self.is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        else:
            frame, minmax = thermal_cam_operation(frame,self.minmax)
            if self.use_minmax and not(self.minmax is not None):
                self.minmax = minmax
                print('Minmax values set:', self.minmax)
            elif not self.use_minmax:
                self.minmax = None
        if frame.shape[0] != self.VIS_H or frame.shape[1] != self.VIS_W:
            frame = cv2.resize(frame,(self.VIS_W,self.VIS_H))
        cv2.putText(frame,self.cam_name,(30,30), FONT, 0.40,(255,255,255),2,cv2.LINE_AA)
        # Display curr_frame_min and curr_frame_max
        cv2.putText(frame,'Min: {}'.format(curr_frame_min),(30,50), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(frame,'Max: {}'.format(curr_frame_max),(30,70), FONT, 0.60,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow(self.cam_name,frame)    

    def read_data(self):
        return np.array(self.data).astype('uint16')

    def read_succ(self):
        return np.array(self.succ_frames)

    def read_times(self):
        return np.array(self.timestamps)

def capture(thermalCam_handle, N_frames_to_record = None, vis_capture = False):
    imager = CaptureThermal(thermalCam_handle,'cam_t', N_frames_to_record, vis_capture=vis_capture, is_color=False)
    imager.run()
    frame_recording = [[]]
    frame_recording[0] = imager.read_data()
    time_stamps = [[]]
    time_stamps[0] = imager.read_times()
    return frame_recording, time_stamps

def thermal_cam_operation(frame,minmax=None):
    frame = frame.astype(np.float32)

    # Rescale to 8 bit
    if minmax is None:
        frame_min = frame.min()
        frame_max = frame.max()
    else:
        frame_min,frame_max = minmax
                
    # print(frame_min,frame_max)
    frame = 255*(frame - frame_min)/(frame_max-frame_min)    
    # Apply colourmap - try COLORMAP_JET if INFERNO doesn't work.
    # You can also try PLASMA or MAGMA
    frame = cv2.applyColorMap(frame.astype(np.uint8), cv2.COLORMAP_INFERNO)
    return frame, (frame_min,frame_max)


class BosonCamera():
    def __init__(self,port):
        super(BosonCamera, self).__init__()
        self.cam = Boson(port=port)
        
    #def reconnect(self):
    #    self.cam = Boson()

    def get_frame(self):
        try:
            image = self.cam.grab()
            succ  = 1
        except:
            image=None
            succ = 0
        return image,succ
    
    def close(self):
        self.cam.close()
        
    def stop_video(self):
        pass
    
    def start_video(self):
        pass