#!/usr/bin/env python3
#!c:/Python35/python3.exe -u
import math

from boxAnnotator import BoxAnnotator
from helpers import *

from lab2.find_cube import *


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

if __name__ == "__main__":
    cozmo.run_program(run, use_viewer=True, force_viewer_on_top=True)
