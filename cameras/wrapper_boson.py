import cv2
import time
import logging
from flirpy.camera.threadedboson import ThreadedBoson


class BosonWithTelemetry(ThreadedBoson):
    def __init__(self):
        super().__init__(device=None, port=None, baudrate=921600, loglevel=logging.WARNING)

        self.configure()
        self.start()
        self.camera.do_ffc()        

    def compute_timestamp_offset(self):
        latest_image, system_time = self.camera.grab(), time.time()
        _, cam_timestamp = self.parse_telemetry(latest_image[:2, :])
        self.timestamp_offset = system_time - cam_timestamp

    def configure(self):
        self.camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 514)
        self.camera.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
        self.camera.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.camera.cap.grab()

        self.compute_timestamp_offset()
    
    def get_next_image(self):
        latest_image = self.latest()

        telemetry = latest_image[:2, :, 0]
        image = latest_image[2:, :, 0]
        _, cam_timestamp = self.parse_telemetry(telemetry)
        timestamp = cam_timestamp + self.timestamp_offset

        return image, timestamp

    def parse_telemetry(self, telemetry):
        frame_counter = telemetry[0, 42] * 2**16 + telemetry[0, 43]
        timestamp_in_ms = telemetry[0, 140] * 2**16 + telemetry[0, 141]
        timestamp = timestamp_in_ms / 1000.0
        return frame_counter, timestamp
