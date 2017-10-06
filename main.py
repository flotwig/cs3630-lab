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
            cozmo.camera.EvtNewRawCameraImage, timeout=30)  #get camera image
        if event.image is not None:
            image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)

        if last_state != state:  #state change
            face = cozmo.oled_face.convert_image_to_screen_data(
                generate_face(state))
            robot.display_oled_face_image(face, 30000, in_parallel=True)
            if last_state != None:
                print("Leaving state: " + state.name)
            print("Entering state: " + state.name)
            robot.say_text(
                "Entering " + state.name,
                use_cozmo_voice=False,
                in_parallel=True).wait_for_completed()

        last_state = state
        state = state.act(robot)


def generate_face(state):
    # make a blank image for the text, initialized to opaque black
    text_image = Image.new('RGBA', cozmo.oled_face.dimensions(), (0, 0, 0,
                                                                  255))
    dc = ImageDraw.Draw(text_image)
    dc.text((0, 0), state.name, fill=(255, 255, 255, 255))
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
    lowerThreshold = np.array([
        cv2.getTrackbarPos("Hue Lower", thresholdWindowName),
        cv2.getTrackbarPos("Sat Lower", thresholdWindowName),
        cv2.getTrackbarPos("Val Lower", thresholdWindowName)
    ])
    upperThreshold = np.array([
        cv2.getTrackbarPos("Hue Upper", thresholdWindowName),
        cv2.getTrackbarPos("Sat Upper", thresholdWindowName),
        cv2.getTrackbarPos("Val Upper", thresholdWindowName)
    ])


class FindARCube:
    name = "Find A R Cube"

    def act(robot: cozmo.robot.Robot):
        adjustThresholds()
        robot.drive_wheels(-10.0, 10.0)
        cube = robot.world.wait_for_observed_light_cube()
        robot.stop_all_motors()
        robot.go_to_object(cube,
                           cozmo.util.distance_mm(100)).wait_for_completed()
        return LocateARFace


class LocateARFace:
    name = "Locate A R Face"

    def act(robot: cozmo.robot.Robot):
        adjustThresholds()
        try:
            cube = robot.world.wait_for_observed_light_cube(timeout=5)
        except:  #maybe it got moved, let's search more
            return FindARCube
        robot.dock_with_cube(
            cube,
            approach_angle=cozmo.util.degrees(180),
            alignment_type=cozmo.robot_alignment.RobotAlignmentTypes.Custom,
            distance_from_marker=cozmo.util.distance_mm(
                70)).wait_for_completed()
        print(cube)
        return LocateARFace

class FindColorCubeLeft:
    name = "Find Color Cube Left"
    
    def act(robot: cozmo.robot.Robot):
        return findColorCube(robot, -1)
        

class FindColorCubeRight:
    name = "Find Color Cube Right"
    
    def act(robot: cozmo.robot.Robot):
        return findColorCube(robot, 1)
        
# 1 = clockwise/right, -1 = counterclockwise/left
def findColorCube(robot: cozmo.robot.Robot, direction):
    robot.drive_wheels(direction * 10.0, -direction * 10.0)
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
    name = "Move" #"Move to Color Cube"
    
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
                    if failures > 10:
                        if delta > 0:
                            return FindColorCubeRight
                        else:
                            return FindColorCubeLeft
                else:
                    cubeSize = cube[2]
                    print(str(cubeSize))
                    if cubeSize > 100:
                        robot.stop_all_motors()
                        return Stop                        
                    else:
                        delta = cubeDelta(cube)
                        base = 15
                        turnStrength = min(max(cubeSize, 10), 35)
                        left = base + max(turnStrength * delta, 0)
                        right = base + max(turnStrength * -delta, 0)
                        robot.drive_wheels(left, right)
                        
                        
class Stop:
    name = "Stop"
    
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
                    if failures > 10:
                        if delta > 0:
                            return FindColorCubeRight
                        else:
                            return FindColorCubeLeft
                else:
                    cubeSize = cube[2]
                    if cubeSize < 5:
                        return MoveToColorCube
                    delta = cubeDelta(cube)
               
# x displacement of cube blob from center of screen (as a percentage of IMAGE_WIDTH / 2)
def cubeDelta(cube):
    return (cube[0] - (IMAGE_WIDTH / 2)) / (IMAGE_WIDTH / 2)
                        
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