#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import cozmo
import cv2
import numpy as np
from PIL import Image, ImageDraw
from time import sleep
import math

### Zach Bloomquist & Taylor Hearn
### CS 3630 Lab 3

def run(robot: cozmo.robot.Robot):
    # initial setup, variables
    create_threshold_trackbars()

    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True

    robot.set_robot_volume(.3)
    robot.set_head_angle(cozmo.util.degrees(head_angle), in_parallel=True)
    #robot.world.get_light_cube(TARGET_CUBE_ID).set_lights(LIGHT_CALM)

    # state machine
    last_state = None
    state = FindARCube
    while state:
        if last_state != state:  # state change
            face = cozmo.oled_face.convert_image_to_screen_data(
                generate_face(state))
            robot.display_oled_face_image(face, 30000, in_parallel=True)
            if last_state is not None:
                print("Leaving state: " + last_state.name)
            print("Entering state: " + state.name)
            robot.say_text(
                "Enter " + state.phonetic_name,
                use_cozmo_voice=False,
                in_parallel=True).wait_for_completed()
        last_state = state
        state = state.act(robot)

class FindARCube:
    phonetic_name = "Find Ay R Cube"
    name = "Find AR Cube"

    def act(robot: cozmo.robot.Robot):
        adjust_thresholds()
        rotation_speed = 8
        cube_distance = 90
        stop_distance = 5
        stop_angle = 15
        cube = None
        
        robot.drive_wheels(-1 * rotation_speed, rotation_speed)  # begin rotating
        cube = robot.world.wait_for_observed_light_cube()
        #cube.set_lights(LIGHT_EXCITED)
        really_stop(robot)
        angle = cube.pose.rotation.angle_z
        destination = cozmo.util.pose_z_angle(cube.pose.position.x - cube_distance * math.cos(angle.radians), cube.pose.position.y - cube_distance * math.sin(angle.radians), cube.pose.position.z, angle)
        go_to_pose = robot.go_to_pose(destination, in_parallel=True, relative_to_robot=False)
        got_to_cube = False
        
        go_to_pose.wait_for_completed()
        really_stop(robot)
        #cube.set_lights(LIGHT_CALM)
        return FindColorCubeLeft


class FindColorCubeLeft:
    phonetic_name = name = "Find Color Cube Left"
    
    def act(robot: cozmo.robot.Robot):
        return find_color_cube(robot, -1)
        

class FindColorCubeRight:
    phonetic_name = name = "Find Color Cube Right"
    
    def act(robot: cozmo.robot.Robot):
        return find_color_cube(robot, 1)


# 1 = clockwise/right, -1 = counterclockwise/left
def find_color_cube(robot: cozmo.robot.Robot, direction):
    robot.drive_wheels(direction * 15, -direction * 15)
    while True:
        adjust_thresholds()
        event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)  # get camera image
        if event.image is not None:
            image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
            cube = find_cube(image, lower_threshold, upper_threshold)
            if cube is not None:
                delta = cube_delta(cube)
                if abs(delta) < 0.5:
                    robot.stop_all_motors()
                    return MoveToColorCube


class MoveToColorCube:
    phonetic_name = name = "Move to Color Cube"
    
    def act(robot: cozmo.robot.Robot):
        delta = 0
        failures = 0
        while True:
            adjust_thresholds()
            event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)  # get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
                cube = find_cube(image, lower_threshold, upper_threshold)
                if cube is None:
                    robot.stop_all_motors()
                    failures += 1
                    if failures > 20:
                        if delta > 0:
                            return FindColorCubeRight
                        else:
                            return FindColorCubeLeft
                else:
                    failures = 0
                    cube_size = cube[2]
                    if cube_size > 100:
                        robot.stop_all_motors()
                        return Stop                        
                    else:
                        delta = cube_delta(cube)
                        base = 20
                        turn_strength = min(max(cube_size, 10), 40)
                        left = base + max(turn_strength * delta, 0)
                        right = base + max(turn_strength * -delta, 0)
                        robot.drive_wheels(left, right)
                        
                        
class Stop:
    phonetic_name = name = "Stop"
    
    def act(robot: cozmo.robot.Robot):
        delta = 0
        failures = 0
        while True:
            adjust_thresholds()
            event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)  # get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
                cube = find_cube(image, lower_threshold, upper_threshold)
                if cube is None:
                    failures += 1
                    if failures > 20:
                        if delta > 0:
                            return FindColorCubeRight
                        else:
                            return FindColorCubeLeft
                else:
                    cube_size = cube[2]
                    if cube_size < 95:
                        failures += 1
                        if failures > 20:
                            return MoveToColorCube
                    else:
                        failures = 0
                    delta = cube_delta(cube)
                   

# boxAnnotator.py
# Define a decorator as a subclass of Annotator; displays the keypoint
class BoxAnnotator(cozmo.annotate.Annotator):

    cube = None

    def apply(self, image, scale):
        d = ImageDraw.Draw(image)
        bounds = (0, 0, image.width, image.height)

        if BoxAnnotator.cube is not None:

            #double size of bounding box to match size of rendered image
            BoxAnnotator.cube = np.multiply(BoxAnnotator.cube, 2)

            #define and display bounding box with params:
            #msg.img_topLeft_x, msg.img_topLeft_y, msg.img_width, msg.img_height
            box = cozmo.util.ImageBox(
                BoxAnnotator.cube[0] - BoxAnnotator.cube[2] / 2,
                BoxAnnotator.cube[1] - BoxAnnotator.cube[2] / 2,
                BoxAnnotator.cube[2], BoxAnnotator.cube[2])
            cozmo.annotate.add_img_box_to_image(image, box, "green", text=None)

            BoxAnnotator.cube = None
            

# find_cube.py
def filter_image(img, hsv_lower, hsv_upper):
    img_filt = cv2.medianBlur(img, 11)
    hsv = cv2.cvtColor(img_filt, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    return mask


    ###############################################################################
    ### You might need to change the parameter values to get better results
    ###############################################################################
def detect_blob(mask):
    '''img = cv2.medianBlur(mask, 9)
   # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;
    #filter by color (on binary)
    params.filterByColor = True
    params.blobColor = 255  # this looks at binary image 0 for looking for dark areas
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 200
    params.maxArea = 20000
    # Filter by Circularity
    params.filterByCircularity = False
    # Filter by Convexity
    params.filterByConvexity = False
    # Filter by Inertia
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
    # Detect blobs.
    keypoints = detector.detect(img)
    return keypoints'''

    img = mask

    # Set up the SimpleBlobdetector with default parameters with specific values.
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 100000

    params.filterByColor = True
    params.blobColor = 255

    params.filterByInertia = False
    params.filterByConvexity = False
    params.filterByCircularity = False

    #ADD CODE HERE

    # builds a blob detector with the given parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # use the detector to detect blobs.
    keypoints = detector.detect(img)
    keypointsImage = cv2.drawKeypoints(
        img, keypoints,
        np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Blobs Detected", keypointsImage)
    return keypoints


def find_cube(img, hsv_lower, hsv_upper):
    """Find the cube in an image.
        Arguments:
        img -- the image
        hsv_lower -- the h, s, and v lower bounds
        hsv_upper -- the h, s, and v upper bounds
        Returns [x, y, radius] of the target blob, and [0,0,0] or None if no blob is found.
    """
    mask = filter_image(img, hsv_lower, hsv_upper)
    keypoints = detect_blob(mask)

    if keypoints == []:
        return None

    ###############################################################################
    # Todo: Sort the keypoints in a certain way if multiple key points get returned
    ###############################################################################

    keypoints = sorted(keypoints, key=lambda keypoint: keypoint.size, reverse=True)
    return [keypoints[0].pt[0], keypoints[0].pt[1], keypoints[0].size]


#helpers.py
DEBUG = True # change to false to disable debug printing

PINK_LOWER = np.array(np.array([168, 150, 141]).round(), np.uint8)
PINK_UPPER = np.array(np.array([180, 224, 255]).round(), np.uint8)

COLOR_PINK = cozmo.lights.Color(rgb=(255, 0, 255))
COLOR_LIME = cozmo.lights.Color(rgb=(0, 255, 0))

LIGHT_EXCITED = cozmo.lights.Light(on_color=COLOR_PINK, off_color=COLOR_LIME, on_period_ms=100, off_period_ms=100,
                                   transition_on_period_ms=50, transition_off_period_ms=50)
LIGHT_CALM = cozmo.lights.Light(on_color=cozmo.lights.blue, off_color=cozmo.lights.white, on_period_ms=300,
                                off_period_ms=300, transition_on_period_ms=1000, transition_off_period_ms=1000)
LIGHT_ERROR = cozmo.lights.red_light.flash()

TARGET_CUBE_ID = cozmo.objects.LightCube1Id

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

        
if __name__ == "__main__":
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
