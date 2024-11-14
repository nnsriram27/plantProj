import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import PySpin


class Blackfly(object):
    def __init__(self, cam_id=0, pixel_format="Mono16"):

        # Initialize the camera object
        self.system = PySpin.System.GetInstance()
        cam_list = self.system.GetCameras()
        num_cameras = cam_list.GetSize()
        assert num_cameras > 0 and num_cameras > cam_id, f'Found only {num_cameras}. Requested camera is {cam_id}'
        self.cam = cam_list[cam_id]
        cam_list.Clear()

        self.cam.Init()
        self.nodemap = self.cam.GetNodeMap()
        self.set_buffer_handling_mode()

        # Load UserSet2
        node_user_set_selector = PySpin.CEnumerationPtr(self.nodemap.GetNode('UserSetSelector'))
        if not PySpin.IsAvailable(node_user_set_selector) or not PySpin.IsWritable(node_user_set_selector):
            print('Unable to load UserSet0... Aborting...')
            return False
        node_user_set_selector.SetIntValue(PySpin.UserSetSelector_UserSet1)
        node_user_set_load = PySpin.CCommandPtr(self.nodemap.GetNode('UserSetLoad'))
        if not PySpin.IsAvailable(node_user_set_load) or not PySpin.IsWritable(node_user_set_load):
            print('Unable to load UserSet0... Aborting...')
            return False
        node_user_set_load.Execute()

        # Set Pixel Format
        self.image_processor = None
        if pixel_format == "Mono16":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
            self.bpp = 16
        elif pixel_format == "Mono8":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
            self.bpp = 8
        elif pixel_format == "BayerRG8":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG8)
            self.bpp = 8
            self.image_processor = PySpin.ImageProcessor()
            self.image_processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        elif pixel_format == "BayerRG16":
            self.cam.PixelFormat.SetValue(PySpin.PixelFormat_BayerRG16)
            self.bpp = 16
            self.image_processor = PySpin.ImageProcessor()
            self.image_processor.SetColorProcessing(PySpin.SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR)
        else:
            raise NotImplementedError()

        node_exposure = PySpin.CFloatPtr(self.nodemap.GetNode('ExposureTime'))
        self.min_exp_val = node_exposure.GetMin()
        self.max_exp_val = node_exposure.GetMax()

        node_gain = PySpin.CFloatPtr(self.nodemap.GetNode('Gain'))
        self.min_gain_val = node_gain.GetMin()
        self.max_gain_val = node_gain.GetMax()

        # Set Acquisition Mode
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

    def __del__(self):
        self.cam.EndAcquisition()
        self.cam.DeInit()
        del self.cam
        self.system.ReleaseInstance()

    def set_buffer_handling_mode(self, mode='NewestOnly'):
        sNodemap = self.cam.GetTLStreamNodeMap()
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False
        node_newestonly = node_bufferhandling_mode.GetEntryByName(mode)
        if not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False
        node_newestonly_mode = node_newestonly.GetValue()
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    def set_exposure(self, exposure_time):
        if exposure_time < self.min_exp_val:
            print(f'Given Exposure Time of {exposure_time} is less than min exp value, using {self.min_exp_val}...')
            exposure_time = self.min_exp_val
        elif exposure_time > self.max_exp_val:
            print(f'Given Exposure Time of {exposure_time} is greater than max exp value, using {self.max_exp_val}...')
            exposure_time = self.max_exp_val
        self.cam.ExposureTime.SetValue(exposure_time)
        print('Exposure set to %f us...' % self.get_exposure())
        
    def get_exposure(self):
        return self.cam.ExposureTime.GetValue()
    
    def set_gain(self, gain):
        if gain < self.min_gain_val:
            print(f'Given Gain of {gain} is less than min gain value, using {self.min_gain_val}...')
            gain = self.min_gain_val
        elif gain > self.max_gain_val:
            print(f'Given Gain of {gain} is greater than max gain value, using {self.max_gain_val}...')
            gain = self.max_gain_val
        
        self.cam.Gain.SetValue(gain)
        print('Gain set to %f dB...' % self.get_gain())
        
    def get_gain(self):
        return self.cam.Gain.GetValue()
    
    def set_fps(self, fps):
        # self.cam.AcquisitionFrameRateEnable.SetValue(True)
        self.cam.AcquisitionFrameRate.SetValue(fps)
        # Get the current frame rate
        print('Current FPS: %f...' % self.get_fps())
    
    def get_fps(self):
        return self.cam.AcquisitionFrameRate.GetValue()

    def get_next_image(self, metadata=False):
        try:
            image_result = self.cam.GetNextImage()

            #  Ensure image completion
            if image_result.IsIncomplete():
                print('Image incomplete with image status ', PySpin.Image_GetImageStatusDescription(image_result.GetImageStatus()))

                image_data, exposure_time, gain, timestamp = None, None, None, None
            else:                    
                # meta_data = image_result.GetChunkData()
                # exposure_time = meta_data.GetExposureTime()
                # gain = meta_data.GetGain()
                # timestamp = meta_data.GetTimestamp()
                exposure_time = 0.0
                gain = 0.0
                timestamp = 0.0

                if self.image_processor is not None:
                    # Getting the image data as a numpy array
                    if self.bpp == 8:
                        image_converted = self.image_processor.Convert(image_result, PySpin.PixelFormat_BGR8)
                    elif self.bpp == 16:
                        image_converted = self.image_processor.Convert(image_result, PySpin.PixelFormat_BGR16)
                    else:
                        raise ValueError()
                    image_data = image_converted.GetNDArray()
                else:
                    image_data = image_result.GetNDArray()

            image_result.Release()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False
        
        if metadata:
            return image_data, exposure_time, gain, timestamp
        else:
            return image_data


def main():

    bfly_obj = Blackfly()

    while True:
        image_data = bfly_obj.getNextImage()
        
        cv2.imshow("image", image_data)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    del bfly_obj   
    

if __name__ == '__main__':
    main()