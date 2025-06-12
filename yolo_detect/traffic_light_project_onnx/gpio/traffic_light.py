import RPi.GPIO as GPIO

class TrafficLight:
    def __init__(self, red_pin, green_pin):
        self.red_pin = red_pin
        self.green_pin = green_pin
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.red_pin, GPIO.OUT)
        GPIO.setup(self.green_pin, GPIO.OUT)

    def set_state(self, state):
        if state == 'RED':
            GPIO.output(self.red_pin, GPIO.HIGH)
            GPIO.output(self.green_pin, GPIO.LOW)
        elif state == 'GREEN':
            GPIO.output(self.red_pin, GPIO.LOW)
            GPIO.output(self.green_pin, GPIO.HIGH)

    def cleanup(self):
        GPIO.cleanup()
