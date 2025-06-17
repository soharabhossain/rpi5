import cv2
import os
import time
from picamera2 import Picamera2

# --- Settings ---
SAVE_DIR = "captured_images"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create the folder if it doesn't exist

# --- Initialize Camera ---
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

print("Press 'c' to capture image. Press 'q' to quit.")

image_count = 0

while True:
    # Capture current frame
    frame = picam2.capture_array()

    # Show the frame
    cv2.imshow("PiCamera Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        # Save the frame
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"image_{timestamp}_{image_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Captured: {filename}")
        image_count += 1

    elif key == ord('q'):
        print("Exiting...")
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
