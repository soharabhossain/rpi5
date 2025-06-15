import cv2
import time
import os

def main():
    # Create output folder if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Open the CSI camera (usually index 0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return

    print("📷 Camera opened. Press 'i' to capture image, 'v' to start video, 'q' to quit.")

    video_writer = None
    recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        # Show the live preview
        cv2.imshow("Camera Preview", frame)

        key = cv2.waitKey(1) & 0xFF

        # Press 'i' to capture an image
        if key == ord('i'):
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            image_path = os.path.join(output_dir, f"image_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"✅ Image saved: {image_path}")

        # Press 'v' to start/stop video recording
        elif key == ord('v'):
            recording = not recording
            if recording:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                video_path = os.path.join(output_dir, f"video_{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                fps = 20.0
                frame_size = (frame.shape[1], frame.shape[0])
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
                print(f"🎥 Recording started: {video_path}")
            else:
                video_writer.release()
                video_writer = None
                print("🛑 Recording stopped")

        # Press 'q' to quit
        elif key == ord('q'):
            break

        # Write frame to video if recording
        if recording and video_writer is not None:
            video_writer.write(frame)

    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print("🔚 Camera closed")

if __name__ == "__main__":
    main()
