#!/usr/bin/env python3

import cv2
import cozmo
import numpy as np
from numpy.linalg import inv
import threading
import time
import asyncio

from ar_markers.hamming.detect import detect_markers

from grid import CozGrid
from gui import GUIWindow
from particle import Particle, Robot
from setting import *
from particle_filter import *
from utils import *

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

#marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))

# goal location for the robot to drive to, (x, y, theta)
goal = (6,10,0)

# pose of the robot in the grid coord frame
robot_grid_pose = None

# particle filter for the robot to localize
particle_filter = None

# map
Map_filename = "map_arena.json"


async def image_processing(robot):

    global camK, marker_size

    event = await robot.world.wait_for(cozmo.camera.EvtNewRawCameraImage, timeout=30)

    # convert camera image to opencv format
    opencv_image = np.asarray(event.image)
    
    # detect markers
    markers = detect_markers(opencv_image, marker_size, camK)
    
    # show markers
    for marker in markers:
        marker.highlite_marker(opencv_image, draw_frame=True, camK=camK)
        #print("ID =", marker.id);
        #print(marker.contours);
    cv2.imshow("Markers", opencv_image)

    return markers

#calculate marker pose
def cvt_2Dmarker_measurements(ar_markers):
    
    marker2d_list = []
    
    for m in ar_markers:
        R_1_2, J = cv2.Rodrigues(m.rvec)
        R_1_1p = np.matrix([[0,0,1], [0,-1,0], [1,0,0]])
        R_2_2p = np.matrix([[0,-1,0], [0,0,-1], [1,0,0]])
        R_2p_1p = np.matmul(np.matmul(inv(R_2_2p), inv(R_1_2)), R_1_1p)
        #print('\n', R_2p_1p)
        yaw = -np.arctan2(R_2p_1p[2,0], R_2p_1p[0,0])
        
        x, y = m.tvec[2][0] + 0.5, -m.tvec[0][0]
        # print('x =', x, 'y =', y,'theta =', yaw)
        
        # remove any duplate markers
        dup_thresh = 2.0
        find_dup = False
        for m2d in marker2d_list:
            if grid_distance(m2d[0], m2d[1], x, y) < dup_thresh:
                find_dup = True
                break
        if not find_dup:
            marker2d_list.append((x,y,math.degrees(yaw)))

    return marker2d_list


#compute robot odometry based on past and current pose
def compute_odometry(curr_pose, cvt_inch=True):
    global last_pose
    last_x, last_y, last_h = last_pose.position.x, last_pose.position.y, \
        last_pose.rotation.angle_z.degrees
    curr_x, curr_y, curr_h = curr_pose.position.x, curr_pose.position.y, \
        curr_pose.rotation.angle_z.degrees

    if cvt_inch:
        last_x, last_y = last_x / 25.6, last_y / 25.6
        curr_x, curr_y = curr_x / 25.6, curr_y / 25.6

    return [[last_x, last_y, last_h],[curr_x, curr_y, curr_h]]

#particle filter functionality
class ParticleFilter:

    def __init__(self, grid):
        self.particles = Particle.create_random(PARTICLE_COUNT, grid)
        self.grid = grid

    def update(self, odom, r_marker_list):

        # ---------- Motion model update ----------
        self.particles = motion_update(self.particles, odom)


        # ---------- Sensor (markers) model update ----------
        self.particles = measurement_update(self.particles, r_marker_list, self.grid)

        # ---------- Show current state ----------
        # Try to find current best estimate for display
        m_x, m_y, m_h, m_confident = compute_mean_pose(self.particles)
        return (m_x, m_y, m_h, m_confident)

        
async def run(robot: cozmo.robot.Robot):
    global last_pose
    global grid, gui
    global particle_filter

    # start streaming
    robot.camera.image_stream_enabled = True
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    ############################################################################
    ######################### YOUR CODE HERE####################################
    
    last_pose = robot.pose

    state = Localize
    while state is not None:
        state = await state(robot)
    
    ############################################################################
    

async def Localize(robot: cozmo.robot.Robot):
    global particle_filter, last_pose, robot_grid_pose
    kidnapped = False

    # start rotating
    rotation_speed = 8
    robot.drive_wheel_motors(-1 * rotation_speed, rotation_speed)

    # reset particle filter
    particle_filter = ParticleFilter(grid)
    gui.particles = particle_filter.particles

    # define event handler that returns Kidnapped when robot picked up
    async def handle_kidnapping(e: cozmo.robot.EvtRobotStateUpdated, robot: cozmo.robot.Robot, **kwargs):
        nonlocal kidnapped
        if robot.is_picked_up and not kidnapped:
            kidnapped = True
            really_stop(robot)
            await play_animation(robot, cozmo.anim.Triggers.CodeLabUnhappy).wait_for_completed()
    robot.world.add_event_handler(cozmo.robot.EvtRobotStateUpdated, handle_kidnapping)

    confident = False
    result = None
    while not confident and not kidnapped:
        odom = compute_odometry(robot.pose)
        last_pose = robot.pose
        markers = cvt_2Dmarker_measurements(await asyncio.ensure_future(image_processing(robot)))
        result = particle_filter.update(odom, markers)
        confident = result[3]
        gui.robot = Particle(result[0], result[1], result[2])
        gui.show_mean(result[0], result[1], result[2], result[3])
        gui.updated.set()

    robot_grid_pose = (result[0], result[1], result[2])

    if kidnapped:
        return Kidnapped
    else:
        return Navigate

        
async def Navigate(robot: cozmo.robot.Robot):
    global robot_grid_pose, goal
    kidnapped = False

    # define event handler that returns Kidnapped when robot picked up
    async def handle_kidnapping(e: cozmo.robot.EvtRobotStateUpdated, **kwargs):
        nonlocal kidnapped
        if robot.is_picked_up:
            kidnapped = True
            really_stop(robot) # TODO: will this interrupt go_to_pose? if not maybe it's better to manually move to goal_pose since no obstacles
    robot.world.add_event_handler(cozmo.robots.EvtRobotStateUpdated, handle_kidnapping)

    x = robot.pose.position.x
    y = robot.pose.position.y
    heading = robot.pose.rotation.angle_z.degrees

    goal_x = x + goal[0] - robot_grid_pose[0]
    goal_y = y + goal[1] - robot_grid_pose[1]
    goal_heading = proj_angle_deg(heading + goal[2] - robot_grid_pose[2])
    goal_pose = cozmo.util.Pose(goal_x, goal_y, 0,angle_z=cozmo.util.Angle(degrees=goal_heading))

    if kidnapped:
        return Kidnapped
    else:
        robot.go_to_pose(goal_pose)
        return Arrived
            
        
async def Kidnapped(robot: cozmo.robot.Robot):
    if robot.is_picked_up:
        return Kidnapped
    else:
        return Localize

        
async def Arrived(robot: cozmo.robot.Robot):
    play_animation(robot, cozmo.anim.Triggers.CodeLabSurprise).wait_for_completed()
    return Arrived
            
            
# for some reason stop_all_motors leaves cozmo wiggling, this is to circumvent that
def really_stop(robot: cozmo.robot.Robot):
    robot.stop_all_motors()
    robot.drive_wheel_motors(0, 0)
            

def play_animation(robot: cozmo.robot.Robot, trigger: cozmo.anim.AnimationTrigger):
    return robot.play_anim_trigger(trigger, use_lift_safe=False, ignore_body_track=True,
                                            ignore_head_track=True, ignore_lift_track=True)


class CozmoThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=False)

    def run(self):
        cozmo.run_program(run, use_viewer=False)


if __name__ == '__main__':

    # cozmo thread
    cozmo_thread = CozmoThread()
    cozmo_thread.start()

    # init
    grid = CozGrid(Map_filename)
    gui = GUIWindow(grid)
    gui.start()
