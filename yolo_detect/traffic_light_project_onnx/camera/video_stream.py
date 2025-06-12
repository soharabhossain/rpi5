import cv2

class VideoStream:
    def __init__(self, use_usb=False):
        self.cap = cv2.VideoCapture(0 if use_usb else 0)
        if not self.cap.isOpened():
            raise IOError("Cannot open camera")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to grab frame")
        return frame

    def release(self):
        self.cap.release()
