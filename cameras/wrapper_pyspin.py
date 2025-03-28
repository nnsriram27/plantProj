import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import time
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
        self.load_userset("UserSet0")
        self.timestamp_offset = self.get_timestamp_offset()

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

        node_fps = PySpin.CFloatPtr(self.nodemap.GetNode('AcquisitionFrameRate'))
        self.min_fps_val = node_fps.GetMin()
        self.max_fps_val = node_fps.GetMax()

        # Set Acquisition Mode
        self.cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
        self.cam.BeginAcquisition()

        # Keep current exposure, gain and fps values
        self.exposure = self.get_exposure()
        self.gain = self.get_gain()
        self.fps = self.get_fps()

    def stop(self):
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

    def load_userset(self, userset_name):
        node_user_set_selector = PySpin.CEnumerationPtr(self.nodemap.GetNode('UserSetSelector'))
        if not PySpin.IsReadable(node_user_set_selector) or not PySpin.IsWritable(node_user_set_selector):
            print('Unable to load UserSet0... Aborting...')
            return False
        node_user_set_value = node_user_set_selector.GetEntryByName(userset_name)
        if not PySpin.IsReadable(node_user_set_value):
            print('Unable to load UserSet0... Aborting...')
            return False
        node_user_set_selector.SetIntValue(node_user_set_value.GetValue())

        node_user_set_load = PySpin.CCommandPtr(self.nodemap.GetNode('UserSetLoad'))
        if not PySpin.IsAvailable(node_user_set_load) or not PySpin.IsWritable(node_user_set_load):
            print('Unable to load UserSet0... Aborting...')
            return False
        node_user_set_load.Execute()

    def get_timestamp_offset(self):
        # Execute Timestamp Latch and return the value
        node_timestamp_latch = PySpin.CCommandPtr(self.nodemap.GetNode('TimestampLatch'))
        if not PySpin.IsAvailable(node_timestamp_latch) or not PySpin.IsWritable(node_timestamp_latch):
            print('Unable to execute TimestampLatch... Aborting...')
            return False
        system_time = time.time()
        node_timestamp_latch.Execute()
        node_timestamp_latch_value = PySpin.CIntegerPtr(self.nodemap.GetNode('TimestampLatchValue'))
        if not PySpin.IsAvailable(node_timestamp_latch_value) or not PySpin.IsReadable(node_timestamp_latch_value):
            print('Unable to read TimestampLatchValue... Aborting...')
            return False
        latched_time = node_timestamp_latch_value.GetValue()
        print(f'System Time: {system_time} s...')
        print(f'Timestamp Latch Value: {latched_time} ns...')
        timestamp_offset = system_time - latched_time / 1e9
        return timestamp_offset
    
    def set_auto_exposure(self, auto_exp=True):
        if auto_exp:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
        else:
            self.cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        return self.get_auto_exposure()

    def get_auto_exposure(self):
        return self.cam.ExposureAuto.GetValue()
    
    def set_auto_exposure_limits(self, min_exp, max_exp):
        ae_lb = PySpin.CFloatPtr(self.nodemap.GetNode('AutoExposureExposureTimeLowerLimit'))
        ae_ub = PySpin.CFloatPtr(self.nodemap.GetNode('AutoExposureExposureTimeUpperLimit'))
        ae_lb.SetValue(min_exp)
        ae_ub.SetValue(max_exp)
        return self.get_auto_exposure_limits()
    
    def get_auto_exposure_limits(self):
        ae_lb = PySpin.CFloatPtr(self.nodemap.GetNode('AutoExposureExposureTimeLowerLimit'))
        ae_ub = PySpin.CFloatPtr(self.nodemap.GetNode('AutoExposureExposureTimeUpperLimit'))
        return ae_lb.GetValue(), ae_ub.GetValue()
    
    def set_exposure(self, exposure_time, check=True):
        if exposure_time < self.min_exp_val:
            print(f'Given Exposure Time of {exposure_time} is less than min exp value, using {self.min_exp_val}...')
            exposure_time = self.min_exp_val
        elif exposure_time > self.max_exp_val:
            print(f'Given Exposure Time of {exposure_time} is greater than max exp value, using {self.max_exp_val}...')
            exposure_time = self.max_exp_val
        self.cam.ExposureTime.SetValue(exposure_time)
        self.exposure = self.get_exposure()
        print('Exposure set to %f us...' % self.exposure)
        if check:
            while True:
                _, _, _, curr_exp, _ = self.get_next_image(return_metadata=True)
                if abs(curr_exp - self.exposure) < 5:
                    break
        return self.exposure
        
    def get_exposure(self):
        return self.cam.ExposureTime.GetValue()
    
    def scan_exposures(self, max_exp_val=None):
        curr_exp = self.min_exp_val
        self.exposure_vals = [self.min_exp_val]
        if max_exp_val is None:
            max_exp_val = self.max_exp_val
        while curr_exp < max_exp_val:
            self.cam.ExposureTime.SetValue(curr_exp)
            new_exp = self.cam.ExposureTime.GetValue()
            if new_exp > self.exposure_vals[-1]:
                print('Exposure set to %f us...' % new_exp)
                self.exposure_vals.append(new_exp)
            curr_exp += 1
        self.exposure_vals.append(max_exp_val)
        return self.exposure_vals
    
    def set_gain(self, gain):
        if gain < self.min_gain_val:
            print(f'Given Gain of {gain} is less than min gain value, using {self.min_gain_val}...')
            gain = self.min_gain_val
        elif gain > self.max_gain_val:
            print(f'Given Gain of {gain} is greater than max gain value, using {self.max_gain_val}...')
            gain = self.max_gain_val
        self.cam.Gain.SetValue(gain)
        self.gain = self.get_gain()
        print('Gain set to %f dB...' % self.gain)
        while True:
            _, _, _, _, curr_gain = self.get_next_image(return_metadata=True)
            if abs(curr_gain - self.gain) < 0.1:
                break
        return self.gain
        
    def get_gain(self):
        return self.cam.Gain.GetValue()
    
    def set_fps(self, fps):
        curr_exp = self.get_exposure()
        if fps < self.min_fps_val:
            print(f'Given FPS of {fps} is less than min fps value, using {self.min_fps_val}...')
            fps = self.min_fps_val
        elif fps > self.max_fps_val:
            print(f'Given FPS of {fps} is greater than max fps value, using {self.max_fps_val}...')
            fps = self.max_fps_val
        self.cam.AcquisitionFrameRate.SetValue(fps)
        self.fps = self.get_fps()
        print('Current FPS: %f...' % self.fps)

        # Update min and max exposure values
        self.min_exp_val = self.cam.ExposureTime.GetMin()
        self.max_exp_val = self.cam.ExposureTime.GetMax()

        # Set exposure to the previous value
        self.set_exposure(curr_exp)
        return self.fps
    
    def get_fps(self):
        return self.cam.AcquisitionFrameRate.GetValue()

    def get_next_image(self, return_metadata=True):
        try:
            image_result, software_tstamp = self.cam.GetNextImage(), time.time()

            #  Ensure image completion
            if image_result.IsIncomplete():
                print('Image incomplete with image status ', PySpin.Image_GetImageStatusDescription(image_result.GetImageStatus()))

                image_data, timestamp, frame_number, exposure, gain = None, None, None, None, None
            else:                    
                metadata = image_result.GetChunkData()
                timestamp = metadata.GetTimestamp()
                timestamp = timestamp / 1e9 + self.timestamp_offset
                frame_number = metadata.GetFrameID()
                exposure = metadata.GetExposureTime()
                gain = metadata.GetGain()

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
        
        if return_metadata:
            return image_data, timestamp, frame_number, exposure, gain
        else:
            return image_data, software_tstamp


def main():

    bfly_obj = Blackfly()
    fps = bfly_obj.get_fps()
    print(f'Initial FPS: {fps}')

    while True:
        image_data, timestamp, frame_number, exposure, gain = bfly_obj.get_next_image(return_metadata=True)
        
        print(f"Timestamp: {timestamp} s, Frame number: {frame_number}, Exposure Time: {exposure} us, Gain: {gain} dB \r", end='')
        
        cv2.imshow("image", image_data)
        key = cv2.waitKey(1)
        if key == ord('e'):
            exposure = bfly_obj.set_exposure(exposure + 100)
        elif key == ord('d'):
            exposure = bfly_obj.set_exposure(exposure - 100)
        elif key == ord('g'):
            gain = bfly_obj.set_gain(gain + 1)
        elif key == ord('f'):
            gain = bfly_obj.set_gain(gain - 1)
        elif key == ord('r'):
            fps = bfly_obj.set_fps(fps + 1)
        elif key == ord('t'):
            fps = bfly_obj.set_fps(fps - 1)
        elif key == ord('q'):
            break

    print("\nExiting...")
    bfly_obj.stop()
    cv2.destroyAllWindows()
    
    

if __name__ == '__main__':
    main()