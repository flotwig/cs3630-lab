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

    windowName = "Threshold Adjuster"
    cv2.namedWindow(windowName)
    cv2.createTrackbar("Hue Lower", windowName, lowerThreshold[0], 180,
                       nothing)
    cv2.createTrackbar("Hue Upper", windowName, upperThreshold[0], 180,
                       nothing)
    cv2.createTrackbar("Sat Lower", windowName, lowerThreshold[1], 255,
                       nothing)
    cv2.createTrackbar("Sat Upper", windowName, upperThreshold[1], 255,
                       nothing)
    cv2.createTrackbar("Val Lower", windowName, lowerThreshold[2], 255,
                       nothing)
    cv2.createTrackbar("Val Upper", windowName, upperThreshold[2], 255,
                       nothing)

    last_turn = 0  # direction of last turn, 1 to right, -1 to left
    oscillations = 0  # how many times we've bounced left to right
    near_mode = False

    try:

        while True:
            cv2.waitKey(1)

            lowerThreshold = np.array([
                cv2.getTrackbarPos("Hue Lower", windowName),
                cv2.getTrackbarPos("Sat Lower", windowName),
                cv2.getTrackbarPos("Val Lower", windowName)
            ])
            upperThreshold = np.array([
                cv2.getTrackbarPos("Hue Upper", windowName),
                cv2.getTrackbarPos("Sat Upper", windowName),
                cv2.getTrackbarPos("Val Upper", windowName)
            ])

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

                #find the cube
                cube = find_cube(image, lowerThreshold, upperThreshold)
                print(cube)
                BoxAnnotator.cube = cube

                if (near_mode):
                    # we're near the box, blob detection gets glitchy
                    # keep track of last 5 diameters, if the average falls below drop out of this mode
                    last_diameters = []
                    if cube == None:
                        last_diameters.append(0)
                    else:
                        last_diameters.append(cube[2])
                    if len(last_diameters) >= 41:
                        last_diameters = last_diameters[1:41]
                    if np.average(np.array(last_diameters)) > 100:
                        continue
                    else:
                        print("dropping out of near mode")
                        last_diameters = []
                        near_mode = False

                # if no keypoint, start turning til there is one
                if (cube is None):
                    robot.turn_in_place(
                        cozmo.util.degrees(37)).wait_for_completed()
                else:  # turn until it is in the center
                    delta = (IMAGE_WIDTH / 2) - cube[0]
                    oscillations += (last_turn == np.sign(delta) * -1)
                    if abs(delta) > 30:
                        last_turn = np.sign(delta)
                        robot.turn_in_place(
                            cozmo.util.degrees(
                                np.sign(delta) * np.max([
                                    delta / 7 - oscillations, 5
                                ]))).wait_for_completed()
                    else:
                        if cube[2] < 200:
                            robot.drive_straight(
                                cozmo.util.distance_inches(2),
                                cozmo.util.speed_mmps(
                                    800)).wait_for_completed()
                        else:
                            near_mode = True

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