from camera.video_stream import VideoStream
from detection.vehicle_detector import VehicleDetector
from gpio.traffic_light import TrafficLight
from utils.timer_logic import compute_new_timings

class TrafficController:
    def __init__(self):
        self.video = VideoStream()
        self.detector = VehicleDetector()
        self.light1 = TrafficLight(red_pin=17, green_pin=27)
        self.light2 = TrafficLight(red_pin=22, green_pin=23)
        self.base_time = 90
        self.current_state = ['RED', 'GREEN']

    def process_frame(self):
        frame = self.video.read()
        annotated, counts = self.detector.detect(frame)
        new_time_1, new_time_2 = compute_new_timings(self.base_time, counts['lane_1'], counts['lane_2'])

        if counts['lane_1'] > counts['lane_2']:
            self.light1.set_state('GREEN')
            self.light2.set_state('RED')
            self.current_state = ['GREEN', 'RED']
        else:
            self.light1.set_state('RED')
            self.light2.set_state('GREEN')
            self.current_state = ['RED', 'GREEN']

        self.last_frame = annotated
        self.last_times = (new_time_1, new_time_2)
        self.last_counts = counts

    def get_frame(self):
        return self.last_frame

    def get_states(self):
        return self.current_state, self.last_times
