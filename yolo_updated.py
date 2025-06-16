import cv2
import numpy as np
import threading
import time
from picamera2 import Picamera2
import ncnn  # Assumes you have a working Python wrapper for NCNN

# --- Global Frame and Lock ---
frame_lock = threading.Lock()
current_frame = None
running = True

# --- Colors for Bounding Boxes ---
bbox_colors = [
    (164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
    (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)
]

# --- Frame Capture Thread ---
def capture_frames(picamera):
    global current_frame, running
    while running:
        frame_bgra = picamera.capture_array()
        frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)

        with frame_lock:
            current_frame = frame_bgr.copy()

# --- YOLOv8n NCNN Inference Class ---
class YOLOv8nNCNN:
    def __init__(self, model_dir, input_size=640):
        self.net = ncnn.Net()
        self.net.load_param(f"{model_dir}/model.param")
        self.net.load_model(f"{model_dir}/model.bin")
        self.input_size = input_size

    def detect(self, image, conf_threshold=0.4):
        img = cv2.resize(image, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        mat_in = ncnn.Mat.from_pixels(img_normalized, ncnn.Mat.PixelType.PIXEL_RGB, self.input_size, self.input_size)
        extractor = self.net.create_extractor()
        extractor.input("images", mat_in)
        ret, mat_out = extractor.extract("output")  # Adjust based on your NCNN output name

        results = []
        for i in range(mat_out.h):
            row = mat_out.row(i)
            values = row.numpy()
            x1, y1, x2, y2, score, class_id = values
            if score > conf_threshold:
                # Map back to original image size
                x1 = int(x1 / self.input_size * image.shape[1])
                y1 = int(y1 / self.input_size * image.shape[0])
                x2 = int(x2 / self.input_size * image.shape[1])
                y2 = int(y2 / self.input_size * image.shape[0])
                results.append((x1, y1, x2, y2, score, int(class_id)))
        return results

# --- Drawing Helper (from Old Code Logic) ---
def draw_detections(frame, detections, labels, conf_thresh=0.4):
    object_count = 0
    for x1, y1, x2, y2, conf, class_id in detections:
        if conf < conf_thresh:
            continue
        label = f"{labels[class_id]}: {int(conf * 100)}%"
        color = bbox_colors[class_id % len(bbox_colors)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_size, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_ymin = max(y1, label_size[1] + 10)
        cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10),
                      (x1 + label_size[0], label_ymin + baseLine - 10), color, cv2.FILLED)
        cv2.putText(frame, label, (x1, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        object_count += 1
    return object_count

# --- Main Execution ---
def main():
    global running

    # Class names (should match model training classes)
    labels = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "dog", "cat", "bottle", "chair"]

    # Initialize YOLOv8n NCNN
    detector = YOLOv8nNCNN(model_dir="yolov8_ncnn_model", input_size=640)

    # Initialize PiCamera
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam.start()

    # Start frame capture thread
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
                    continue
                frame = current_frame.copy()

            detections = detector.detect(frame)
            object_count = draw_detections(frame, detections, labels)

            # FPS calc
            t_end = time.perf_counter()
            fps = 1.0 / (t_end - t_start)
            if len(frame_rate_buffer) >= fps_avg_len:
                frame_rate_buffer.pop(0)
            frame_rate_buffer.append(fps)
            avg_fps = np.mean(frame_rate_buffer)

            # Show metrics
            cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Objects: {object_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display
            cv2.imshow("YOLOv8n NCNN - PiCam", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        running = False
        capture_thread.join()
        picam.stop()
        cv2.destroyAllWindows()
        print(f"Average pipeline FPS: {np.mean(frame_rate_buffer):.2f}")

if __name__ == "__main__":
    main()
