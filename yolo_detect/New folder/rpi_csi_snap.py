"""
Dependencies:

picamera (already available on RPi OS or can be installed via sudo apt install python3-picamera)

PIL (can be installed via pip install pillow)
"""

import os
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from picamera import PiCamera

# Create output folder
OUTPUT_DIR = "captured_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize camera
camera = PiCamera()
camera.resolution = (640, 480)

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RPi CSI Camera Capture")

        # Prefix entry
        self.prefix_label = tk.Label(root, text="Filename Prefix:")
        self.prefix_label.pack(pady=5)
        self.prefix_entry = tk.Entry(root)
        self.prefix_entry.insert(0, "image")
        self.prefix_entry.pack(pady=5)

        # Canvas to display image
        self.canvas = tk.Canvas(root, width=640, height=480, bg="gray")
        self.canvas.pack(pady=10)

        # Capture button
        self.capture_button = ttk.Button(root, text="Capture", command=self.capture_image)
        self.capture_button.pack(pady=10)

        self.image_on_canvas = None
        self.capture_count = 0

    def capture_image(self):
        prefix = self.prefix_entry.get().strip() or "image"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)

        # Capture and save image
        camera.capture(filepath)

        # Load and display image on canvas
        pil_image = Image.open(filepath)
        pil_image = pil_image.resize((640, 480), Image.ANTIALIAS)
        self.tk_image = ImageTk.PhotoImage(pil_image)

        if self.image_on_canvas is None:
            self.image_on_canvas = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        else:
            self.canvas.itemconfig(self.image_on_canvas, image=self.tk_image)

        print(f"Captured: {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
