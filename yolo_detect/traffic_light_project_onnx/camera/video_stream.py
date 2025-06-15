from picamera2 import Picamera2
import time
import cv2

class VideoStream:
    def __init__(self, resolution=(640, 480)):
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(main={"format": "RGB888", "size": resolution})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)  # Warm-up time

    def read(self):
        frame = self.picam2.capture_array()
        return frame

    def release(self):
        self.picam2.stop()
