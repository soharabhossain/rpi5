import cv2
import numpy as np
import threading
import time
from picamera2 import Picamera2
from ultralytics import YOLO


# --- Global Frame and Lock ---
frame_lock = threading.Lock()
current_frame = None
running = True

# --- Colors for Bounding Boxes ---
bbox_colors = [
    (164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133), (88, 159, 106),
    (96, 202, 231), (159, 124, 168), (169, 162, 241), (98, 118, 150), (172, 176, 184)
]


# --- Frame Capture Thread ---
def capture_frames(picamera):
    global current_frame, running
    while running:
        frame_bgra = picamera.capture_array()
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        with frame_lock:
            current_frame = frame_bgr.copy()


# --- YOLOv8n Inference Class ---
class YOLOv8nNCNN:
    def __init__(self, model_dir, input_size=640):
        self.model = YOLO(model_dir, task='detect')
        self.input_size = input_size
        self.labels = self.model.names

    def detect_objects(self, frame, min_thresh=0.5):
        """
        Perform object detection on a given frame using the YOLOv8 model.

        Args:
            frame (ndarray): The image frame on which detection is to be performed.
            min_thresh (float): Minimum confidence threshold for displaying detected objects.

        Returns:
            detections (list of dict): List of detection details.
            frame_out (ndarray): Frame with drawn bounding boxes.
            object_count (int): Number of detected objects above threshold.
        """
        results = self.model(frame, verbose=False)
        detections_raw = results[0].boxes
        detections = []
        object_count = 0

        for i in range(len(detections_raw)):
            xyxy = detections_raw[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            classidx = int(detections_raw[i].cls.item())
            classname = self.labels[classidx]
            conf = detections_raw[i].conf.item()

            if conf > min_thresh:
                color = bbox_colors[classidx % len(bbox_colors)]
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                label = f'{classname}: {int(conf * 100)}%'
                label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_ymin = max(ymin, label_size[1] + 10)
                cv2.rectangle(frame, (xmin, label_ymin - label_size[1] - 10),
                              (xmin + label_size[0], label_ymin + baseLine - 10), color, cv2.FILLED)
                cv2.putText(frame, label, (xmin, label_ymin - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                detections.append({
                    'class': classname,
                    'confidence': conf,
                    'bbox': [xmin, ymin, xmax, ymax]
                })
                object_count += 1

        return detections, frame, object_count


# --- Main Execution ---
def main():
    global running

    # Initialize YOLOv8n Detector
    detector = YOLOv8nNCNN(model_dir="yolov8n_ncnn_model", input_size=640)

    # Initialize PiCamera
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam.start()

    # Start Frame Capture Thread
    capture_thread = threading.Thread(target=capture_frames, args=(picam,))
    capture_thread.start()

    # FPS tracking
    frame_rate_buffer = []
    fps_avg_len = 100

    try:
        while True:
            t_start = time.perf_counter()

            with frame_lock:
                if current_frame is None:
                    time.sleep(0.01)
                    continue
                frame = current_frame.copy()

            # Perform Detection
            detections, frame, object_count = detector.detect_objects(frame)

            # FPS Calculation
            t_end = time.perf_counter()
            fps = 1.0 / (t_end - t_start)
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(fps)
            avg_fps = np.mean(frame_rate_buffer)

            # Show Metrics
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Objects: {object_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display Frame
            cv2.imshow("YOLOv8n - PiCamera", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    except Exception as e:
        print(f"Error occurred: {e}")

    finally:
        running = False
        capture_thread.join()
        picam.stop()
        cv2.destroyAllWindows()
        print(f"Average pipeline FPS: {np.mean(frame_rate_buffer):.2f}")


if __name__ == "__main__":
    main()
