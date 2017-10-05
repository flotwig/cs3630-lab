#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import asyncio
import sys
import cv2
import numpy as np
import cozmo
import time
import os
from PIL import ImageDraw
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

    # state machine
    last_state = None
    state = FindARCube
    while state:
        cv2.waitKey(1)

        event = robot.world.wait_for(
            cozmo.camera.EvtNewRawCameraImage, timeout=30)  #get camera image
        if event.image is not None:
            image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)

        if last_state != state:  #state change
            last_state = state
        state = state.act(robot)
    cv2

    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()


class FindARCube:
    def act(robot: cozmo.robot.Robot):
        robot.drive_wheels(-10.0, 10.0)
        cube = robot.world.wait_for_observed_light_cube()
        robot.stop_all_motors()
        robot.go_to_object(cube,
                           cozmo.util.distance_mm(100)).wait_for_completed()
        return LocateARFace


class LocateARFace:
    def act(robot: cozmo.robot.Robot):
        cube = robot.world.wait_for_observed_light_cube(timeout=5)
        if cube is None:  #maybe it got moved, let's search more
            return FindARCube
        return LocateARFace


if __name__ == "__main__":
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)