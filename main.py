#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import asyncio
import sys
import cv2
import numpy as np
import cozmo
import time
import os
from glob import glob
from boxAnnotator import BoxAnnotator

### Zach Bloomquist & Taylor Hearn
### CS 3630 Lab 3


def run(robot: cozmo.robot.Robot):
    last_state = None
    state = FindARCube
    while state:
        if last_state != state:
            last_state = state
        state = state.act()
    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True

    robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    gain, exposure, mode = 390, 3, 1

    lowerThreshold = PINK_LOWER
    upperThreshold = PINK_UPPER

    last_turn = 0  # direction of last turn, 1 to right, -1 to left
    oscillations = 0  # how many times we've bounced left to right
    near_mode = False

    try:

        while True:
            cv2.waitKey(1)

            event = robot.world.wait_for(
                cozmo.camera.EvtNewRawCameraImage,
                timeout=30)  #get camera image
            if event.image is not None:
                image = cv2.cvtColor(
                    np.asarray(event.image), cv2.COLOR_BGR2RGB)
                if mode == 1:
                    robot.camera.enable_auto_exposure = True
                else:
                    robot.camera.set_manual_exposure(exposure, fixed_gain)
    except KeyboardInterrupt:
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)


class FindARCube:
    def act(robot: cozmo.robot.Robot):
        ayncio.ensure_future(
            robot.drive_straight(
                cozmo.util.distance_inches(2), cozmo.util.speed_mmps(800)))
        return self


if __name__ == "__main__":
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)