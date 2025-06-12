import onnxruntime as ort
import numpy as np
import cv2

class VehicleDetector:
    def __init__(self, model_path='models/yolov5s.onnx'):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.img_size = 640

    def preprocess(self, image):
        img = cv2.resize(image, (self.img_size, self.img_size))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
        img /= 255.0
        return np.expand_dims(img, axis=0), img

    def detect(self, frame):
        input_tensor, _ = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})[0]

        lane_1, lane_2 = 0, 0
        h, w, _ = frame.shape
        for det in outputs[0]:
            conf = det[4]
            if conf < 0.25:
                continue
            x0, y0, x1, y1 = int(det[0] * w), int(det[1] * h), int(det[2] * w), int(det[3] * h)
            label = int(det[5])
            color = (0, 255, 0) if label in [2, 3, 5, 7] else (0, 0, 255)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            x_center = (x0 + x1) / 2
            if x_center < w // 2:
                lane_1 += 1
            else:
                lane_2 += 1
        return frame, {'lane_1': lane_1, 'lane_2': lane_2}
