import time
import sys
import argparse
sys.path.append('/home/ilimnuc/plantProj/analog_discovery/')

from WF_SDK import device as wf_dev, wavegen as wf_wavegen


class AnalogDiscoveryDevice(object):
    def __init__(self):
        self.dev = wf_dev.open()

    def __del__(self):
        wf_wavegen.close(self.dev)
        wf_dev.close(self.dev)
    
    def set_dc_voltage(self, voltage, channel='ch1'):
        wf_wavegen.generate(self.dev, channel=1, function=wf_wavegen.function.dc, offset=voltage)
        # self.device.analog_output['ch1'].setup(function='dc', start=True, offset=voltage)
    
    def reset(self):
        # self.device.analog_output['ch1'].reset()
        wf_wavegen.close(self.dev)


def parse_args():
    parser = argparse.ArgumentParser(description='Control output of analog discovery')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = AnalogDiscoveryDevice()

    while True:
        key = input('Enter command (q -> quit, r -> reset wavegen, number -> dc offset voltage): ')
        if key == 'q':
            break
        elif key == 'r':
            device.reset()
        else:
            try:
                voltage = float(key)
                print('Setting voltage to', voltage)
                device.set_dc_voltage(voltage)
            except Exception as e:
                print('Invalid Voltage:', e)
                continue