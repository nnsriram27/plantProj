import cv2
import numpy as np

from pyueye import ueye


class uEye(object):
    def __init__(self, cam_id=0) -> None:
        self.hcam = ueye.HIDS(cam_id)
        ret = ueye.is_InitCamera(self.hcam, None)
        assert ret == ueye.IS_SUCCESS, f'Failed to initialize camera. Error code: {ret}'

        nNumber = ueye.c_uint()
        ret = ueye.is_ParameterSet(self.hcam, ueye.IS_PARAMETERSET_CMD_GET_NUMBER_SUPPORTED, nNumber, ueye.sizeof(nNumber))
        assert ret == ueye.IS_SUCCESS, f'Failed to get number of parameter sets. Error code: {ret}'
        print(f'Number of parameter sets: {nNumber.value}')

        # load parameter set
        ret = ueye.is_ParameterSet(self.hcam, ueye.IS_PARAMETERSET_CMD_LOAD_EEPROM, ueye.c_uint(0), ueye.sizeof(ueye.c_uint(0)))
        assert ret == ueye.IS_SUCCESS, f'Failed to load parameter set. Error code: {ret}'
        print(f'Loaded parameter set 0')

        # set color mode
        ret = ueye.is_SetColorMode(self.hcam, ueye.IS_CM_SENSOR_RAW12)
        assert ret == ueye.IS_SUCCESS, f'Failed to set color mode. Error code: {ret}'

        # get sensor info
        self.sensor_info = ueye.SENSORINFO()
        ret = ueye.is_GetSensorInfo(self.hcam, self.sensor_info)
        assert ret == ueye.IS_SUCCESS, f'Failed to get sensor info. Error code: {ret}'

        # allocate memory
        self.mem_ptr = ueye.c_mem_p()
        self.mem_id = ueye.int()
        self.width = self.sensor_info.nMaxWidth.value
        self.height = self.sensor_info.nMaxHeight.value
        self.bpp = 16
        ret = ueye.is_AllocImageMem(self.hcam, self.width, self.height, self.bpp, self.mem_ptr, self.mem_id)
        assert ret == ueye.IS_SUCCESS, f'Failed to allocate memory. Error code: {ret}'

        # set memory active
        ret = ueye.is_SetImageMem(self.hcam, self.mem_ptr, self.mem_id)
        assert ret == ueye.IS_SUCCESS, f'Failed to set memory active. Error code: {ret}'

        self.lineinc = ueye.c_int()
        ret = ueye.is_GetImageMemPitch(self.hcam, self.lineinc)
        assert ret == ueye.IS_SUCCESS, f'Failed to get image memory pitch. Error code: {ret}'

        # # continuous capture to memory
        # ret = ueye.is_CaptureVideo(self.hcam, ueye.IS_DONT_WAIT)
        # assert ret == ueye.IS_SUCCESS, f'Failed to start video capture. Error code: {ret}'

        # Get exposure min max values
        self.exposure_min = ueye.double()
        self.exposure_max = ueye.double()
        self.exposure_increment = ueye.double()

        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, self.exposure_min, ueye.sizeof(self.exposure_min))
        assert ret == ueye.IS_SUCCESS, f'Failed to get exposure min. Error code: {ret}'

        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, self.exposure_max, ueye.sizeof(self.exposure_max))
        assert ret == ueye.IS_SUCCESS, f'Failed to get exposure max. Error code: {ret}'

        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, self.exposure_increment, ueye.sizeof(self.exposure_increment))
        assert ret == ueye.IS_SUCCESS, f'Failed to get exposure increment. Error code: {ret}'

        print(f"Camera initialized and has exposure range: [{self.exposure_min.value}, {self.exposure_max.value}] ms")

    def set_fps(self, fps):
        # set frame rate
        self.target_fps = ueye.double(fps)
        self.fps = ueye.double()
        ret = ueye.is_SetFrameRate(self.hcam, self.target_fps, self.fps)
        assert ret == ueye.IS_SUCCESS, f'Failed to set frame rate. Error code: {ret}'
        print(f'Target frame rate: {self.target_fps.value}, actual frame rate: {self.fps.value}')

        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MIN, self.exposure_min, ueye.sizeof(self.exposure_min))
        assert ret == ueye.IS_SUCCESS, f'Failed to get exposure min. Error code: {ret}'

        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_MAX, self.exposure_max, ueye.sizeof(self.exposure_max))
        assert ret == ueye.IS_SUCCESS, f'Failed to get exposure max. Error code: {ret}'

        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE_INC, self.exposure_increment, ueye.sizeof(self.exposure_increment))
        assert ret == ueye.IS_SUCCESS, f'Failed to get exposure increment. Error code: {ret}'

        print(f"Changed frame rate to {self.fps} and camera now has exposure range: [{self.exposure_min.value}, {self.exposure_max.value}] ms")


    def set_exposure(self, exposure_val, print_message=True):
        if exposure_val < self.exposure_min.value:
            if print_message:
                print(f"Exposure value {exposure_val} is less than minimum {self.exposure_min.value}")
            exposure_val = self.exposure_min.value
        elif exposure_val > self.exposure_max.value:
            exposure_val = self.exposure_max.value
            if print_message:
                print(f"Exposure value {exposure_val} is greater than maximum {self.exposure_max.value}")

        # exposure_val is in milliseconds
        time_exposure = ueye.double(exposure_val)
        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_SET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))
        assert ret == ueye.IS_SUCCESS, f'Failed to set exposure. Error code: {ret}'

        ret = ueye.is_Exposure(self.hcam, ueye.IS_EXPOSURE_CMD_GET_EXPOSURE, time_exposure, ueye.sizeof(time_exposure))
        assert ret == ueye.IS_SUCCESS, f'Failed to get exposure. Error code: {ret}'
        if print_message:
            print(f'Exposure set to {time_exposure.value} ms')

        # throwing the next image since it usually has old exposure value
        ret = ueye.is_FreezeVideo(self.hcam, ueye.IS_WAIT)
        assert ret == ueye.IS_SUCCESS, f'Failed to freeze video. Error code: {ret}'
        return time_exposure.value  

    # def set_expose_auto(self, value, value_to_return=0):
    #     """
    #     Set auto expose to on/off.

    #     Args
    #     =======
    #     value: int Number;
    #     value_to_return: int Number
    #     Returns
    #     =======
    #     list: report
    #     """

    #     print("set auto gain to {}, (additional sets to: {})".format(value, value_to_return))
    #     value = ueye.c_double(float(value))
    #     value_to_return = ueye.c_double(float(value_to_return))
    #     print("cTypes: arg1: '{}' arg2: '{}'".format(value.value, value_to_return.value))
    #     report = [ueye.is_SetAutoParameter(self.hcam,
    #                                        ueye.IS_AUTO_EXPOSURE_RUNNING,
    #                                        value,
    #                                        value_to_return
    #                                        ), value.value, value_to_return.value]
    #     print(report)
    #     return report
    
    def getNextImage(self):
        ret = ueye.is_FreezeVideo(self.hcam, ueye.IS_WAIT)
        assert ret == ueye.IS_SUCCESS, f'Failed to freeze video. Error code: {ret}'

        # get image data
        data = ueye.get_data(self.mem_ptr, self.width, self.height, self.bpp, self.lineinc, copy=True).tobytes()
        data_uint16 = np.frombuffer(data, dtype=np.uint16)
        img = np.reshape(data_uint16, (self.height, self.width))
        return img
    
    def get_frame(self):
        ret = ueye.is_FreezeVideo(self.hcam, ueye.IS_WAIT)
        assert ret == ueye.IS_SUCCESS, f'Failed to freeze video. Error code: {ret}'

        # get image data
        data = ueye.get_data(self.mem_ptr, self.width, self.height, self.bpp, self.lineinc, copy=True).tobytes()
        data_uint16 = np.frombuffer(data, dtype=np.uint16)
        img = np.reshape(data_uint16, (self.height, self.width))
        return img, True

    def __del__(self):
        # clean up
        ret = ueye.is_StopLiveVideo(self.hcam, ueye.IS_FORCE_VIDEO_STOP)
        assert ret == ueye.IS_SUCCESS, f'Failed to stop video capture. Error code: {ret}'
        ret = ueye.is_ExitCamera(self.hcam)
        assert ret == ueye.IS_SUCCESS, f'Failed to exit camera. Error code: {ret}'


def main():
    cam = uEye()
    while True:
        img = cam.getNextImage()
        import pdb; pdb.set_trace()
        img_8bit = np.uint8(img / 16)
        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    del cam


if __name__ == '__main__':
    main()