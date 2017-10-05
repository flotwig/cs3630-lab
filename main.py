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

### Zach Bloomquist & Taylor Hearn
### CS 3630 Lab 3


def run(robot: cozmo.robot.Robot):
    # initial setup, variables
    gain, exposure, mode = 390, 3, 1

    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True
    robot.set_robot_volume(.3)

    robot.set_head_angle(cozmo.util.degrees(0), in_parallel=True)

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


class FindARCube:
    name = "Find A R Cube"

    def act(robot: cozmo.robot.Robot):
        robot.drive_wheels(-10.0, 10.0)
        cube = robot.world.wait_for_observed_light_cube()
        robot.stop_all_motors()
        robot.go_to_object(cube,
                           cozmo.util.distance_mm(100)).wait_for_completed()
        return LocateARFace


class LocateARFace:
    name = "Locate A R Face"

    def act(robot: cozmo.robot.Robot):
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


if __name__ == "__main__":
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)