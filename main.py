#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import asyncio
import sys
import cv2
import numpy as np
import cozmo
import time
import os
from PIL import ImageDraw, Image
from glob import glob
from boxAnnotator import BoxAnnotator
from find_cube import *
import pdb

### Zach Bloomquist & Taylor Hearn
### CS 3630 Lab 3

PINK_LOWER = np.array(np.array([168, 150, 141]).round(), np.uint8)
PINK_UPPER = np.array(np.array([180, 224, 255]).round(), np.uint8)

COLOR_PINK = cozmo.lights.Color(rgb=(255, 0, 255))
COLOR_LIME = cozmo.lights.Color(rgb=(0, 255, 0))

LIGHT_EXCITED = cozmo.lights.Light(on_color=COLOR_PINK, off_color=COLOR_LIME, on_period_ms=100, off_period_ms=100,
                                   transition_on_period_ms=50, transition_off_period_ms=50)
LIGHT_CALM = cozmo.lights.Light(on_color=cozmo.lights.blue, off_color=cozmo.lights.white, on_period_ms=300,
                                off_period_ms=300, transition_on_period_ms=1000, transition_off_period_ms=1000)
thresholdWindowName = "Adjust Thresholds"
lowerThreshold = PINK_LOWER
upperThreshold = PINK_UPPER

headAngle = -5

IMAGE_WIDTH = 320


def run(robot: cozmo.robot.Robot):
    # initial setup, variables
    createThresholdTrackbars()
    
    gain, exposure, mode = 390, 3, 1

    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True
    robot.set_robot_volume(.3)

    robot.set_head_angle(cozmo.util.degrees(headAngle), in_parallel=True)

    # state machine
    last_state = None
    state = FindARCube
    while state:
        event = robot.world.wait_for(
            cozmo.camera.EvtNewRawCameraImage, timeout=30)  # get camera image
        if event.image is not None:
            image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)

        if last_state != state:  # state change
            face = cozmo.oled_face.convert_image_to_screen_data(
                generate_face(state))
            display = robot.display_oled_face_image(face, 30000, in_parallel=True)

            if last_state != None:
                print("Leaving state: " + state.name)
            print("Entering state: " + state.name)
            robot.say_text(
                "Entering " + state.phonetic_name,
                use_cozmo_voice=False,
                in_parallel=True)

        last_state = state
        state = state.act(robot)

# generates an image with state's name centered
def generate_face(state):
    dim = cozmo.oled_face.dimensions()
    text_image = Image.new('RGBA', dim, (0, 0, 0, 255))
    dc = ImageDraw.Draw(text_image)
    dc.text((dim[0]/2 - (len(state.name) * 3), dim[1]/2), state.name, fill=(255, 255, 255, 255))
    return text_image

    
def createThresholdTrackbars():
    def nothing(x):
        pass
    cv2.namedWindow(thresholdWindowName)
    cv2.createTrackbar("Hue Lower", thresholdWindowName, lowerThreshold[0], 180, nothing)
    cv2.createTrackbar("Hue Upper", thresholdWindowName, upperThreshold[0], 180, nothing)
    cv2.createTrackbar("Sat Lower", thresholdWindowName, lowerThreshold[1], 255, nothing)
    cv2.createTrackbar("Sat Upper", thresholdWindowName, upperThreshold[1], 255, nothing)
    cv2.createTrackbar("Val Lower", thresholdWindowName, lowerThreshold[2], 255, nothing)
    cv2.createTrackbar("Val Upper", thresholdWindowName, upperThreshold[2], 255, nothing)

def adjustThresholds():
    cv2.waitKey(1)
    global lowerThreshold
    lowerThreshold = np.array([
        cv2.getTrackbarPos("Hue Lower", thresholdWindowName),
        cv2.getTrackbarPos("Sat Lower", thresholdWindowName),
        cv2.getTrackbarPos("Val Lower", thresholdWindowName)
    ])
    global upperThreshold
    upperThreshold = np.array([
        cv2.getTrackbarPos("Hue Upper", thresholdWindowName),
        cv2.getTrackbarPos("Sat Upper", thresholdWindowName),
        cv2.getTrackbarPos("Val Upper", thresholdWindowName)
    ])


class FindARCube:
    phonetic_name = "Find Ay R Cube"
    name = "Find AR Cube"

    def act(robot: cozmo.robot.Robot):
        adjustThresholds()
        rotation_speed = 8
        stop_distance = 70
        stop_angle = 5
        while True:
            robot.drive_wheels(-1 * rotation_speed, rotation_speed)  # begin rotating
            cube = robot.world.wait_for_observed_light_cube()
            cube.set_lights(LIGHT_EXCITED)
            really_stop(robot)
            go_to_pose = robot.go_to_pose(cube.pose, in_parallel=True, relative_to_robot=False)
            while not (go_to_pose.is_completed or go_to_pose.is_aborting):
                difference = go_to_pose.pose - robot.pose
                distance = np.sqrt(difference.position.x ** 2 + difference.position.y ** 2)
                angle_diff = abs(difference.rotation.angle_z.degrees)
                print(go_to_pose.pose.is_comparable(robot.pose), distance, angle_diff)
                if distance <= stop_distance and angle_diff <= stop_angle:
                    go_to_pose.abort()
                    really_stop(robot)
                time.sleep(.02)
            go_to_pose.wait_for_completed()
            print(go_to_pose.result, go_to_pose.has_succeeded, go_to_pose.has_failed, go_to_pose.failure_reason)
            if go_to_pose.result == cozmo.action.ActionResults.CANCELLED_WHILE_RUNNING:
                break
        cube.set_lights(LIGHT_CALM)
        return FindColorCubeLeft

class FindColorCubeLeft:
    phonetic_name = name = "Find Color Cube Left"
    
    def act(robot: cozmo.robot.Robot):
        return findColorCube(robot, -1)
        

class FindColorCubeRight:
    phonetic_name = name = "Find Color Cube Right"
    
    def act(robot: cozmo.robot.Robot):
        return findColorCube(robot, 1)
        
# 1 = clockwise/right, -1 = counterclockwise/left
def findColorCube(robot: cozmo.robot.Robot, direction):
    robot.drive_wheels(direction * 15, -direction * 15)
    while True:
        adjustThresholds()
        event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)  #get camera image
        if event.image is not None:
            image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
            cube = find_cube(image, lowerThreshold, upperThreshold)
            if cube != None:
                delta = cubeDelta(cube)
                if(abs(delta) < 0.5):
                    robot.stop_all_motors()
                    return MoveToColorCube


class MoveToColorCube:
    phonetic_name = name = "Move to Color Cube"
    
    def act(robot: cozmo.robot.Robot):
        delta = 0
        failures = 0
        while True:
            adjustThresholds()
            event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)  #get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
                cube = find_cube(image, lowerThreshold, upperThreshold)
                if cube == None:
                    robot.stop_all_motors()
                    failures += 1
                    if failures > 20:
                        if delta > 0:
                            return FindColorCubeRight
                        else:
                            return FindColorCubeLeft
                else:
                    failures = 0
                    cubeSize = cube[2]
                    if cubeSize > 100:
                        robot.stop_all_motors()
                        return Stop                        
                    else:
                        delta = cubeDelta(cube)
                        base = 20
                        turnStrength = min(max(cubeSize, 10), 40)
                        left = base + max(turnStrength * delta, 0)
                        right = base + max(turnStrength * -delta, 0)
                        robot.drive_wheels(left, right)
                        
                        
class Stop:
    phonetic_name = name = "Stop"
    
    def act(robot: cozmo.robot.Robot):
        delta = 0
        failures = 0
        while True:
            adjustThresholds()
            event = robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)  #get camera image
            if event.image is not None:
                image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
                cube = find_cube(image, lowerThreshold, upperThreshold)
                if cube == None:
                    failures += 1
                    if failures > 20:
                        if delta > 0:
                            return FindColorCubeRight
                        else:
                            return FindColorCubeLeft
                else:
                    cubeSize = cube[2]
                    if cubeSize < 95:
                        failures += 1
                        if failures > 20:
                            return MoveToColorCube
                    else:
                        failures = 0
                    delta = cubeDelta(cube)
               
# x displacement of cube blob from center of screen (as a percentage of IMAGE_WIDTH / 2)
def cubeDelta(cube):
    return (cube[0] - (IMAGE_WIDTH / 2)) / (IMAGE_WIDTH / 2)

# for some reason stop_all_motors leaves cozmo wiggling, this is to circumvent that
def really_stop(robot: cozmo.robot.Robot):
    robot.stop_all_motors()
    robot.drive_wheel_motors(0, 0)

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