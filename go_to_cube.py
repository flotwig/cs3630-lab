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

from find_cube import *

### Zach Bloomquist & Taylor Hearn
### CS 3630 Lab 2

try:
    from PIL import ImageDraw, ImageFont
except ImportError:
    sys.exit('run `pip3 install --user Pillow numpy` to run this example')


def nothing(x):
    pass


PINK_LOWER = np.array(np.array([168, 150, 141]).round(), np.uint8)
PINK_UPPER = np.array(np.array([180, 224, 255]).round(), np.uint8)

IMAGE_WIDTH = 320


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


async def run(robot: cozmo.robot.Robot):

    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True

    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

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

            event = await robot.world.wait_for(
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
                    await robot.turn_in_place(
                        cozmo.util.degrees(37)).wait_for_completed()
                else:  # turn until it is in the center
                    delta = (IMAGE_WIDTH / 2) - cube[0]
                    oscillations += (last_turn == np.sign(delta) * -1)
                    if abs(delta) > 30:
                        last_turn = np.sign(delta)
                        await robot.turn_in_place(
                            cozmo.util.degrees(
                                np.sign(delta) * np.max([
                                    delta / 7 - oscillations, 5
                                ]))).wait_for_completed()
                    else:
                        if cube[2] < 200:
                            await robot.drive_straight(
                                cozmo.util.distance_inches(2),
                                cozmo.util.speed_mmps(800)
                            ).wait_for_completed()
                        else:
                            near_mode = True

    except KeyboardInterrupt:
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)


if __name__ == '__main__':
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)