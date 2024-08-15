import time
import ctypes
import numpy as np
from multiprocessing import Process, RawArray, Value
import multiprocessing
from flirpy.camera.boson import Boson


class BosonFrameGrabber(object):
    def __init__(self):
        self.camera = Boson()
        self.height = 512
        self.width = 640
        self._shared_array = RawArray(ctypes.c_uint16, self.height * self.width)
        self._frame = None
        self._tstamp = Value('d', 0.0)
        self._got_new_frame = Value('i', 0)
        self._running = False

    def start_grabber(self):
        self._running = True
        self._grabber_thread = Process(target=self.run_grabber)
        self._grabber_thread.start()
    
    def stop_grabber(self):
        self._running = False
        self.camera.release()
        self._grabber_thread.join()
    
    def run_grabber(self):
        while self._running:
            frame = np.frombuffer(self._shared_array, dtype=np.uint16, count=self.height * self.width)
            # Add a timeout to avoid blocking indefinitely
            grabbed_frame = self.camera.grab()
            
            if grabbed_frame is None:
                self._running = False
                print("Camera disconnected")
                break
            np.copyto(frame, grabbed_frame.reshape(-1))    # blocks until new frame is available
            self._tstamp.value = time.time()
            self._got_new_frame.value = 1
    
    def get_latest_frame(self):
        while self._running and not self._got_new_frame.value:
            time.sleep(0.001)
        if self._got_new_frame.value == 0:
            raise RuntimeError("Camera disconnected")
        self._frame = np.copy(np.frombuffer(self._shared_array, dtype=np.uint16)).reshape((self.height, self.width))
        self._got_new_frame.value = 0
        return self._frame, self._tstamp.value