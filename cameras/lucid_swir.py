import numpy as np
from matplotlib import pyplot as plt

from arena_api.system import system
from arena_api.buffer import *


class LucidSWIR(object):
    def __init__(self) -> None:
        devices = system.create_device()
        if len(devices) == 0:
            raise Exception("No devices were found.")
        
        print(f"Found {len(devices)} devices from Lucid, connecting to the first device.")
        self.device = devices[0]

        # Initialize
        tl_stream_nodemap = self.device.tl_stream_nodemap
        tl_stream_nodemap['StreamBufferHandlingMode'].value = "NewestOnly"
        tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
        tl_stream_nodemap['StreamPacketResendEnable'].value = True

        self.nodemap = self.device.nodemap
        nodes = self.nodemap.get_node(['AcquisitionMode', 'PixelFormat', 'ExposureAuto', 'ExposureTime', 'GainAuto', 'Gain'])
        nodes['AcquisitionMode'].value = 'Continuous'
        nodes['PixelFormat'].value = 'Mono16'
        nodes['ExposureAuto'].value = 'Off'
        print(f"Disabled auto exposure.")
        print(f"Current exposure time is {nodes['ExposureTime'].value}, min exposure time is {nodes['ExposureTime'].min}, max exposure time is {nodes['ExposureTime'].max}")

        nodes['GainAuto'].value = 'Off'
        print(f"Disabled auto gain")
        print(f"Current gain value is {nodes['Gain'].value}, min gain value is {nodes['Gain'].min}, max gain value is {nodes['Gain'].max}")

        self.streaming = False

    def set_fps(self, fps):
        self.nodemap.get_node('AcquisitionFrameRateEnable').value = True
        self.nodemap.get_node('AcquisitionFrameRate').value = float(fps)
        print(f"Set acquisition frame rate to {self.nodemap.get_node('AcquisitionFrameRate').value}")

    def set_gain(self, gain_val):
        gain_node = self.nodemap.get_node('Gain')
        if gain_node.is_writable is False:
            raise Exception("Gain node not writeable")
        if gain_val > gain_node.max:
            gain_node.value = gain_node.max
            print(f"Clipping requested gain value to max value of {gain_node.max}")
        elif gain_val < gain_node.min:
            gain_node.value = gain_node.min
            print(f"Clipping requested gain value to min value of {gain_node.min}")
        else:
            gain_node.value = float(gain_val)
        print(f"Set gain value to {gain_node.value}")

    def set_exposure(self, exposure_time):
        exposure_node = self.nodemap.get_node('ExposureTime')
        if exposure_node.is_writable is False:
            raise Exception("Exposure Time node not writeable")
        if exposure_time > exposure_node.max:
            exposure_time = exposure_node.max
            print(f"Clipping requested exposure time to max value of {exposure_node.max}")
        elif exposure_time < exposure_node.min:
            exposure_time = exposure_node.min
            print(f"Clipping requested exposure time to min value of {exposure_node.min}")
        exposure_node.value = float(exposure_time)
        print(f"Set exposure time to {exposure_node.value}")

        # Resetting the streaming to apply the new exposure time
        if self.streaming is True:
            self.device.stop_stream()
            self.streaming = False
            self.device.start_stream()
            self.streaming = True

        return self._get_node_value('ExposureTime')
    
    def _get_node_value(self, nodename):
        return self.nodemap.get_node(nodename).value
    
    def getNextImage(self, bpp=16):
        if self.streaming is False:
            self.device.start_stream()
            self.streaming = True
        buffer = self.device.get_buffer()
        item = BufferFactory.copy(buffer)
        self.device.requeue_buffer(buffer)
        img_np = np.frombuffer(bytes(item.data), dtype=np.uint16).reshape(item.height, item.width)
        BufferFactory.destroy(item)
        return img_np
    
    def printTemperatureInfo(self):
        self.nodemap['DeviceTemperatureSelector'].value = 'Sensor'
        print(f"Sensor temperature is: {self.nodemap['DeviceTemperature'].value}C")
        self.nodemap['DeviceTemperatureSelector'].value = 'TEC'
        print(f"TEC temperature is: {self.nodemap['DeviceTemperature'].value}C")
        self.nodemap['DeviceTemperatureSelector'].value = 'Sensor' # resetting to default value
        feature_names = ['TECControlTemperatureSetPoint', 'TECCurrent', 'TECPower', 'TECStatus', 'TECVoltage']
        for f in feature_names:
            print(f'{f} = {self.nodemap.get_node(f).value}')

    def __del__(self):
        if self.streaming is True:
            self.device.stop_stream()
            self.streaming = False
        system.destroy_device()

# for node in nodemap.feature_names:
#     if 'gain' in node.casefold():
#         print(node)
#         if nodemap.get_node(node).is_readable is True:
#             print(f"\t {nodemap.get_node(node).value}")
    
if __name__ == '__main__':
    cam = LucidSWIR()
    cam.printTemperatureInfo()
    cam.set_fps(10)
    cam.set_gain(0)
    cam.set_exposure(50000)
    
    plt.figure()
    while True:
        img = cam.getNextImage()
        img = img[:, ::-1]
        min_val, max_val = np.percentile(img, (1, 99))
        img = np.clip((img - min_val) / (max_val - min_val), 0, 1)
    
        plt.imshow(img**0.6, cmap='gray')
        # plt.pause(0.1)
        if plt.waitforbuttonpress(0.1):
            break
        plt.cla()
    
    del cam