import numpy as np
import cv2
import traceback

FONT = cv2.FONT_HERSHEY_SIMPLEX

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

def debayer_image(img,mode='BGR'):
    if mode is 'BGR':
        mode = cv2.COLOR_BayerRGGB2BGR
    else:
        mode = cv2.COLOR_BayerRGGB2RGB
    return cv2.cvtColor(img,mode)

def vis_cam_operation(frame):
    frame = debayer_image(frame)
    return np.floor(frame/16).astype('uint8')

def adjustGamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def convert_frame_to_uint8(frame):
    # frame is debayered
    if frame.dtype=='uint8':
        return frame
    else:
        return np.floor_divide(frame,256).astype('uint8')

def debayer_sequence(seq,mode='BGR'):
    if mode is 'BGR':
        mode = cv2.COLOR_BAYER_BG2BGR
    else:
        mode = cv2.COLOR_BAYER_BG2RGB
        
    seq_color = np.zeros((seq.shape[0],seq.shape[1],seq.shape[2],3),dtype=seq.dtype)
    for i in range(seq.shape[0]):
        seq_color[i] = cv2.cvtColor(seq[i],mode)
    return seq_color

def display_16bit_BG(frame,is_out_RGB=1,gamma=1.4):
    frame = convert_frame_to_uint8(frame)
    if len(frame.shape)==2:
        frame = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2BGR)
    if is_out_RGB:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = adjustGamma(frame,gamma)
    return frame

def bgr2rgb(img):
    return img[...,::-1].copy()


# -------- CAMERA -------- 

def combine_multiple_frames(frames):
    
    frame_show = frames[0]
    H,W        = frames[0].shape[0:2]

    try:
        for i in range(1,len(frames)):
            if frames[i].shape[0]==frame_show.shape[0]:
                frame_show   = np.concatenate((frame_show,frames[i]),axis=1)
            else:
                Hn,Wn        = frames[i].shape[0:2]

                frame_to_add = cv2.resize(frames[i],None,fx = H/Hn,fy = H/Hn)

                frame_show   = np.concatenate((frame_show,frame_to_add),axis=1)                        

    except Exception:
        traceback.print_exc()
        pass
    
    if len(frames)>1:
        cv2.line(frame_show,(frames[0].shape[1],0),
                            (frames[0].shape[1],frames[0].shape[0]), 
                            (0,0,255),1)
    return frame_show


def normalize_img(img):
    img_n = img-img.min()
    img_n/= img_n.max()
    img_n = (img_n*255).astype('uint8')
    return img_n


