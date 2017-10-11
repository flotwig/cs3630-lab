import numpy as np
import cozmo
import cv2
from PIL import Image, ImageDraw

DEBUG = True # change to false to disable debug printing

PINK_LOWER = np.array(np.array([168, 150, 141]).round(), np.uint8)
PINK_UPPER = np.array(np.array([180, 224, 255]).round(), np.uint8)

COLOR_PINK = cozmo.lights.Color(rgb=(255, 0, 255))
COLOR_LIME = cozmo.lights.Color(rgb=(0, 255, 0))

LIGHT_EXCITED = cozmo.lights.Light(on_color=COLOR_PINK, off_color=COLOR_LIME, on_period_ms=100, off_period_ms=100,
                                   transition_on_period_ms=50, transition_off_period_ms=50)
LIGHT_CALM = cozmo.lights.Light(on_color=cozmo.lights.blue, off_color=cozmo.lights.white, on_period_ms=300,
                                off_period_ms=300, transition_on_period_ms=1000, transition_off_period_ms=1000)
threshold_window_name = "Adjust Thresholds"
lower_threshold = PINK_LOWER
upper_threshold = PINK_UPPER

head_angle = -5

IMAGE_WIDTH = 320

# generates an image with state's name centered
def generate_face(state):
    dim = cozmo.oled_face.dimensions()
    text_image = Image.new('RGBA', dim, (0, 0, 0, 255))
    dc = ImageDraw.Draw(text_image)
    dc.text((dim[0]/2 - (len(state.name) * 3), dim[1]/2), state.name, fill=(255, 255, 255, 255))
    return text_image

def create_threshold_trackbars():
    def nothing(x):
        pass
    cv2.namedWindow(threshold_window_name)
    cv2.createTrackbar("Hue Lower", threshold_window_name, lower_threshold[0], 180, nothing)
    cv2.createTrackbar("Hue Upper", threshold_window_name, upper_threshold[0], 180, nothing)
    cv2.createTrackbar("Sat Lower", threshold_window_name, lower_threshold[1], 255, nothing)
    cv2.createTrackbar("Sat Upper", threshold_window_name, upper_threshold[1], 255, nothing)
    cv2.createTrackbar("Val Lower", threshold_window_name, lower_threshold[2], 255, nothing)
    cv2.createTrackbar("Val Upper", threshold_window_name, upper_threshold[2], 255, nothing)

def adjust_thresholds():
    cv2.waitKey(1)
    global lower_threshold
    lower_threshold = np.array([
        cv2.getTrackbarPos("Hue Lower", threshold_window_name),
        cv2.getTrackbarPos("Sat Lower", threshold_window_name),
        cv2.getTrackbarPos("Val Lower", threshold_window_name)
    ])
    global upper_threshold
    upper_threshold = np.array([
        cv2.getTrackbarPos("Hue Upper", threshold_window_name),
        cv2.getTrackbarPos("Sat Upper", threshold_window_name),
        cv2.getTrackbarPos("Val Upper", threshold_window_name)
    ])

# x displacement of cube blob from center of screen (as a percentage of IMAGE_WIDTH / 2)
def cube_delta(cube):
    return (cube[0] - (IMAGE_WIDTH / 2)) / (IMAGE_WIDTH / 2)

# for some reason stop_all_motors leaves cozmo wiggling, this is to circumvent that
def really_stop(robot: cozmo.robot.Robot):
    robot.stop_all_motors()
    robot.drive_wheel_motors(0, 0)

def debug(*args): # will print if DEBUG is true
    if DEBUG:
        print("[DEBUG]", args)

class RunningAverage:
    def __init__(self, size):
        self.list = list()
        self.size = size

    def record(self, item):
        self.list.append(item)
        if len(self.list) > self.size:
            self.list.pop[0]

    def average(self):
        if len(self.list) == 0:
            return None
        return sum(self.list) / len(self.list)
