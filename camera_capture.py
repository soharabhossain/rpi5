# camera_capture_picamera2.py

import time
import cv2
from picamera2 import Picamera2
import os

def main():
    # Create output folder
    os.makedirs("output", exist_ok=True)

    # Initialize Picamera2 and preview configuration
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # warm-up

    print("📷 Press 'i' to capture image, 'v' to start/stop video, 'q' to quit.")

    recording = False
    video_writer = None

    while True:
        # Get frame from camera
        frame = picam2.capture_array()
        cv2.imshow("Camera Preview", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('i'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_path = f"output/image_{timestamp}.jpg"
            cv2.imwrite(img_path, frame)
            print(f"✅ Image saved: {img_path}")

        elif key == ord('v'):
            recording = not recording
            if recording:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                video_path = f"output/video_{timestamp}.avi"
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = 20.0
                size = (frame.shape[1], frame.shape[0])
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, size)
                print(f"🎥 Recording started: {video_path}")
            else:
                video_writer.release()
                video_writer = None
                print("🛑 Recording stopped")

        elif key == ord('q'):
            break

        if recording and video_writer is not None:
            video_writer.write(frame)

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    picam2.stop()
    print("🔚 Camera closed.")

if __name__ == "__main__":
    main()
