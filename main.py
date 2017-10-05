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

PINK_LOWER = np.array(np.array([168, 150, 141]).round(), np.uint8)
PINK_UPPER = np.array(np.array([180, 224, 255]).round(), np.uint8)

IMAGE_WIDTH = 320

def run(robot: cozmo.robot.Robot):
    robot.world.image_annotator.annotation_enabled = True
    robot.world.image_annotator.add_annotator('box', BoxAnnotator)

    robot.camera.image_stream_enabled = True
    robot.camera.color_image_enabled = True
    robot.camera.enable_auto_exposure = True

    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    gain, exposure, mode = 390, 3, 1
    robot.camera.set_manual_exposure(exposure, fixed_gain)

    lowerThreshold = PINK_LOWER
    upperThreshold = PINK_UPPER
    
    try:
        while True:
            cv2.waitKey(1)

            

                

    except KeyboardInterrupt:
        print("Exit requested by user")
    except cozmo.RobotBusy as e:
        print(e)
        

class FindARCube:
    def act(robot):
        event = await robot.world.wait_for(
            cozmo.camera.EvtNewRawCameraImage,
            timeout=30)  #get camera image
        if event.image is not None:
            image = cv2.cvtColor(np.asarray(event.image), cv2.COLOR_BGR2RGB)
            
            
        


if __name__ == "__main__":
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)