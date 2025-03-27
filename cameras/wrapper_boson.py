import cv2
import time
import logging
import numpy as np

from flirpy.camera.threadedboson import ThreadedBoson


class BosonWithTelemetry(ThreadedBoson):
    def __init__(self):
        super().__init__(device=None, port=None, baudrate=921600, loglevel=logging.WARNING)

        self.configure()
        self.start()
        self.camera.do_ffc()
        self.logged_images = []
        self.logged_tstamps = []
        self.enable_logging = False

    def __del__(self):
        self.stop()
        self.camera.close()
    
    def stop_logging(self):
        self.enable_logging = False

    def start_logging(self):
        self.enable_logging = True
        self.add_post_callback(self.post_cap_hook)
    
    def post_cap_hook(self, image):
        if self.enable_logging:
            self.logged_images.append(image)
            self.logged_tstamps.append(time.time())

    def compute_timestamp_offset(self):
        (_, latest_image), system_time = self.camera.cap.read(), time.time()
        _, cam_timestamp = self.parse_telemetry(latest_image[:2, :])
        self.timestamp_offset = system_time - cam_timestamp

    def configure(self):
        self.camera.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 514)
        self.camera.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'Y16 '))
        self.camera.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.camera.cap.grab()

        self.compute_timestamp_offset()
    
    def get_next_image(self, hflip=False):
        latest_image = self.latest()

        telemetry = latest_image[:2, :, 0]
        image = latest_image[2:, :, 0]
        frame_number, cam_timestamp = self.parse_telemetry(telemetry)
        timestamp = cam_timestamp + self.timestamp_offset

        if hflip:
            image = cv2.flip(image, 1)
        return image, timestamp, frame_number, telemetry

    def parse_telemetry(self, telemetry):
        frame_counter = telemetry[0, 42] * 2**16 + telemetry[0, 43]
        timestamp_in_ms = telemetry[0, 140] * 2**16 + telemetry[0, 141]
        timestamp = timestamp_in_ms / 1000.0
        return frame_counter, timestamp

    

if __name__ == "__main__":
    
    boson = BosonWithTelemetry()
    
    cv2.namedWindow("Boson", cv2.WINDOW_NORMAL)
    
    while True:
        image, timestamp, frame_number, _ = boson.get_next_image()

        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = np.uint8(image)
        print(f"Timestamp: {timestamp}, Frame number: {frame_number}\r", end="")

        cv2.imshow("Boson", image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    
    boson.stop()
    cv2.destroyAllWindows()