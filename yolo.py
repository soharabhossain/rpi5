import cv2
import numpy as np
import threading
import time
from picamera2 import Picamera2
import ncnn  # Assuming you have a working Python wrapper for ncnn


# --- Global Frame and Lock ---
frame_lock = threading.Lock()
current_frame = None
running = True


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

    def detect(self, image, conf_threshold=0.4, iou_threshold=0.5):
        img = cv2.resize(image, (self.input_size, self.input_size))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0

        mat_in = ncnn.Mat.from_pixels(img_normalized, ncnn.Mat.PixelType.PIXEL_RGB, self.input_size, self.input_size)

        extractor = self.net.create_extractor()
        extractor.input("images", mat_in)
        ret, mat_out = extractor.extract("output")  # Modify output blob name if different

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


# --- Drawing Helper ---
def draw_detections(frame, detections, labels):
    for x1, y1, x2, y2, conf, cls_id in detections:
        label = f"{labels[cls_id]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


# --- Main Execution ---
def main():
    global running

    # Class names (modify according to your dataset)
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

    try:
        while True:
            t1 = time.time()
            with frame_lock:
                if current_frame is None:
                    continue
                frame = current_frame.copy()

            detections = detector.detect(frame)
            draw_detections(frame, detections, labels)

            fps = 1.0 / (time.time() - t1)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("YOLOv8n NCNN - PiCam", frame)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        running = False
        capture_thread.join()
        picam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

