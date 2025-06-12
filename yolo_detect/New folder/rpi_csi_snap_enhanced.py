#  rpi_csi_snap_enhanced.py (Enhanced Version)

"""
Dependencies:

sudo apt update
sudo apt install python3-picamera
pip3 install pillow opencv-python

"""

import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Camera imports
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
    CSI_AVAILABLE = True
except ImportError:
    CSI_AVAILABLE = False

import cv2

# Setup
OUTPUT_DIR = "captured_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RPi Camera Capture")
        self.running = False
        self.using_csi = CSI_AVAILABLE
        self.camera_type = tk.StringVar(value="CSI" if self.using_csi else "USB")

        # UI Elements
        self.prefix_label = tk.Label(root, text="Filename Prefix:")
        self.prefix_label.pack()
        self.prefix_entry = tk.Entry(root)
        self.prefix_entry.insert(0, "image")
        self.prefix_entry.pack()

        self.camera_select = ttk.Combobox(root, values=["CSI", "USB"], state="readonly", textvariable=self.camera_type)
        self.camera_select.pack()

        self.canvas = tk.Canvas(root, width=640, height=480, bg="gray")
        self.canvas.pack()

        self.capture_button = ttk.Button(root, text="Capture", command=self.capture_image)
        self.capture_button.pack(pady=5)

        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.cap = None
        self.csi_camera = None
        self.canvas_image = None

        self.init_camera()
        self.running = True
        self.video_thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def init_camera(self):
        if self.camera_type.get() == "CSI" and CSI_AVAILABLE:
            self.using_csi = True
            self.csi_camera = PiCamera()
            self.csi_camera.resolution = (640, 480)
            self.raw_capture = PiRGBArray(self.csi_camera, size=(640, 480))
            time.sleep(0.1)
        else:
            self.using_csi = False
            self.cap = cv2.VideoCapture(0)

    def switch_camera(self):
        if self.cap:
            self.cap.release()
        if self.csi_camera:
            self.csi_camera.close()
        self.init_camera()

    def video_loop(self):
        while self.running:
            frame = None
            if self.using_csi and self.csi_camera:
                self.csi_camera.capture(self.raw_capture, format="bgr", use_video_port=True)
                frame = self.raw_capture.array
                self.raw_capture.truncate(0)
            elif self.cap:
                ret, frame = self.cap.read()
                if not ret:
                    continue

            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480))
                self.tk_image = ImageTk.PhotoImage(img)

                if self.canvas_image is None:
                    self.canvas_image = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
                else:
                    self.canvas.itemconfig(self.canvas_image, image=self.tk_image)

            time.sleep(0.03)

    def capture_image(self):
        prefix = self.prefix_entry.get().strip() or "image"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        frame = None
        if self.using_csi and self.csi_camera:
            self.csi_camera.capture(filepath)
        elif self.cap:
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(filepath, frame)

        print(f"Saved: {filepath}")

    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        if self.csi_camera:
            self.csi_camera.close()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    app.camera_select.bind("<<ComboboxSelected>>", lambda e: app.switch_camera())
    root.mainloop()
