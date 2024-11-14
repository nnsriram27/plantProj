import nidaqmx


class LightIntensityController:
    def __init__(self, device_name='Dev1', channel='ao1'):
        self.device_name = device_name
        self.channel = channel
        self.task = nidaqmx.Task()
        self.task.ao_channels.add_ao_voltage_chan(f'{self.device_name}/{self.channel}')
        self.min_voltage = 0.0
        self.max_voltage = 10.0

    def set_voltage(self, voltage):
        if voltage < self.min_voltage:
            voltage = self.min_voltage
            print(f'Given voltage of {voltage} is less than min voltage value, using {self.min_voltage}...')
        elif voltage > self.max_voltage:
            voltage = self.max_voltage
            print(f'Given voltage of {voltage} is greater than max voltage value, using {self.max_voltage}...')

        self.task.write(voltage)

    def close(self):
        self.task.close()