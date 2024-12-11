import threading


class CamThread():
    def __init__(self, cam_obj):
        self._cam_obj = cam_obj
        self._running = False

        self.latest_image = None

        self.log_pixels = False        
        self.selected_pixels = []
        self.pixel_vals = []
        self.pixel_timestamps = []
        
        self.log_images = False
        self.images = []
        self.timestamps = []

        self.log_metadata = False
        self.metadata = []
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def update_logging(self, log_images=False, log_pixels=False, log_metadata=False):
        self.log_images = log_images
        self.log_pixels = log_pixels
        self.log_metadata = log_metadata
    
    def stop(self):
        self._running = False
        self._thread.join()

    def run(self):
        while self._running:
            pass

    def clicked_pixel(self, x, y):
        if self.log_pixels:
            self.selected_pixels.append((y, x))
            self.pixel_vals.append([])
        
    def clear_pixels(self):
        self.selected_pixels = []
        self.pixel_vals = []
        self.pixel_timestamps = []

class VisCamThread(CamThread):
    def __init__(self, cam_obj):
        super().__init__(cam_obj)

    def run(self):
        while self._running:
            img, tstamp, frame_number, exposure, gain = self._cam_obj.get_next_image()
            self.latest_image = img.copy()

            if self.log_images:
                self.images.append(img)
                self.timestamps.append(tstamp)
            if self.log_metadata:
                self.metadata.append((frame_number, exposure, gain))

            if self.log_pixels:
                self.pixel_timestamps.append(tstamp)
                for i, pixel in enumerate(self.selected_pixels):
                    self.pixel_vals[i].append(img[pixel[0], pixel[1]])


class ThrCamThread(CamThread):
    def __init__(self, cam_obj):
        super().__init__(cam_obj)

    def run(self):
        while self._running:
            img, tstamp, frame_number, telemetry = self._cam_obj.get_next_image()
            img = img[:, ::-1]
            self.latest_image = img.copy()

            if self.log_images:
                self.images.append(img)
            if self.log_pixels:
                for i, pixel in enumerate(self.selected_pixels):
                    self.pixel_vals[i].append(img[pixel[0], pixel[1]])
            if self.log_images or self.log_pixels:
                self.timestamps.append(tstamp)
            if self.log_metadata:
                self.metadata.append((frame_number, telemetry))
