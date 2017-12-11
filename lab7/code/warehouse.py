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

# ## Zach Bloomquist & Taylor Hearn
# ## CS 3630 Lab 7

# camera params
camK = np.matrix([[295, 0, 160], [0, 295, 120], [0, 0, 1]], dtype='float32')

# marker size in inches
marker_size = 3.5

# tmp cache
last_pose = cozmo.util.Pose(0,0,0,angle_z=cozmo.util.Angle(degrees=0))

# location of the origin of cozmo's coord frame in the grid coord frame
cozmo_origin = None

# angle between the x-axis of cozmo's coord frame and x-axis of grid coord frame
cozmo_rotation = None

# particle filter for the robot to localize
particle_filter = None

# map
Map_filename = "map_arena.json"

# docking (left) and warehouse (right) regions divided by the line x = 13
region_vertical_divider = 13

# fragile zone is below y = 4
fragile_horizontal_divider = 4

# storage zone is to the right of x = 22
storage_vertical_divider = 22

# pickup zone is to the left of x = 8 and above y = 10
pickup_divider = (8, 10)

# relay zone is in the rectangle x = [10, 16], y = [6, 12]
relay_rectangle = ((10, 6), (16, 12))

# number of boxes successfully transported
boxes_moved = 0

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
    
    last_pose = robot.pose

    state = Localize
    while state is not None:
        state = await state(robot)
    
    ############################################################################
    
    
async def Localize(robot: cozmo.robot.Robot):
    global particle_filter, last_pose, cozmo_origin, cozmo_rotation
    print("Entering state Localize")
    
    await robot.set_head_angle(cozmo.util.degrees(0)).wait_for_completed()

    # start rotating
    rotation_speed = 8
    robot.drive_wheel_motors(-1 * rotation_speed, rotation_speed)

    # reset particle filter
    particle_filter = ParticleFilter(grid)

    confident = False
    result = None
    times_confident = 0
    while not confident:
        odom = compute_odometry(robot.pose)
        last_pose = robot.pose
        markers = cvt_2Dmarker_measurements(await asyncio.ensure_future(image_processing(robot)))
        result = particle_filter.update(odom, markers)
        times_confident = (times_confident + 1) * result[3]  # times result[3] has been true in a row
        confident = times_confident > 10
        gui.robot = Particle(result[0], result[1], result[2])
        gui.show_particles(particle_filter.particles)
        gui.show_mean(result[0], result[1], result[2], result[3])
        gui.updated.set()

    really_stop(robot)
    
    if result is None:
        return Localize
        
    robot_grid_pose = (result[0], result[1], result[2])
    
    cozmo_rotation = proj_angle_deg(robot_grid_pose[2] - robot.pose.rotation.angle_z.degrees)
    
    # angle and magnitude of the vector pointing from cozmo to his origin (both in the grid coord frame)
    angle = math.atan2(robot.pose.position.y, robot.pose.position.x) + math.radians(cozmo_rotation) + math.pi
    magnitude = math.sqrt(mm_to_inches(robot.pose.position.x)**2 + mm_to_inches(robot.pose.position.y)**2)
    
    cozmo_origin_x = robot_grid_pose[0] + magnitude * math.cos(angle)
    cozmo_origin_y = robot_grid_pose[1] + magnitude * math.sin(angle)
    cozmo_origin = (cozmo_origin_x, cozmo_origin_y)

    if robot_grid_pose[0] < region_vertical_divider:
        return Pickup
    else:
        return Storage


async def Pickup(robot: cozmo.robot.Robot):
    print("Entering state Pickup")
    
    # between the fragile and pickup zones facing the pickup zone
    wait_x = region_vertical_divider / 2
    wait_y = (fragile_horizontal_divider + pickup_divider[1]) / 2
    wait_pose = (wait_x, wait_y, 110)
    
    # in the middle of the relay zone
    dropoff_x = (relay_rectangle[0][0] + relay_rectangle[1][0]) / 2
    dropoff_y = (relay_rectangle[0][1] + relay_rectangle[1][1]) / 2
    dropoff_pose = (dropoff_x, dropoff_y, 0)
    
    def cube_pose_constraints(cube_grid_pose):
        return cube_grid_pose[0] <= pickup_divider[0] and cube_grid_pose[1] >= pickup_divider[1]
    
    await deliver_cube(robot, wait_pose, dropoff_pose, cube_pose_constraints)
    
    return Pickup
    
    
async def Storage(robot: cozmo.robot.Robot):
    global boxes_moved
    print("Entering state Storage")
    
    # between the relay and storage zones facing the relay zone
    wait_x = (relay_rectangle[1][0] + storage_vertical_divider) / 2
    wait_y = (relay_rectangle[0][1] + relay_rectangle[1][1]) / 2
    wait_pose = (wait_x, wait_y, 130)
    
    # in the middle of the relay zone
    dropoff_x = (storage_vertical_divider + grid.width) / 2
    dropoff_y = grid.height - 2.5 * (boxes_moved + 1)
    dropoff_pose = (dropoff_x, dropoff_y, 0)
    
    def cube_pose_constraints(cube_grid_pose):
        x_constraints = cube_grid_pose[0] >= relay_rectangle[0][0] and cube_grid_pose[0] <= relay_rectangle[1][0]
        y_constraints = cube_grid_pose[1] >= relay_rectangle[0][1] and cube_grid_pose[1] <= relay_rectangle[1][1]
        return x_constraints and y_constraints
        
    await deliver_cube(robot, wait_pose, dropoff_pose, cube_pose_constraints)

    return Storage
    
    
async def deliver_cube(robot: cozmo.robot.Robot, wait_pose, dropoff_pose, cube_pose_constraints):
    global boxes_moved
    
    print("Driving to pickup...")
    await robot.go_to_pose(cozmo_pose(robot, wait_pose)).wait_for_completed()
    print("Drove to pickup...")
    
    print("Waiting for cube...")
    cube = None
    while cube is None:
        cube = await robot.world.wait_for_observed_light_cube()
        if cube is not None and cube.pose is not None:
            cube_grid_pose = grid_pose(robot, cube.pose)
            print(cube_grid_pose)
            if cube_pose_constraints(cube_grid_pose):
                break
        cube = None
    print("Found cube")
    
    print("Picking up cube...")
    await robot.pickup_object(cube).wait_for_completed()
    print("Picked up cube")
    
    print("Driving to dropoff...")
    await robot.go_to_pose(cozmo_pose(robot, dropoff_pose)).wait_for_completed()
    print("Drove to dropoff")
    
    print("Dropping off cube...")
    await robot.place_object_on_ground_here(cube).wait_for_completed()
    print("Dropped off cube")
    
    boxes_moved += 1

    
# give a pose in cozmo coord frame as a pose in the grid coord frame
def grid_pose(robot: cozmo.robot.Robot, cozmo_pose):
    robot_x = mm_to_inches(cozmo_pose.position.x)
    robot_y = mm_to_inches(cozmo_pose.position.y)
    rotation = math.radians(cozmo_rotation)
    x = cozmo_origin[0] + (robot_x * math.cos(rotation) - robot_y * math.sin(rotation))
    y = cozmo_origin[1] + (robot_x * math.sin(rotation) + robot_y * math.cos(rotation))
    angle = proj_angle_deg(cozmo_rotation + cozmo_pose.rotation.angle_z.degrees)
    return (x, y, angle)

# give a pose in grid coord frame as a pose in cozmo coord frame
def cozmo_pose(robot: cozmo.robot.Robot, grid_pose):
    rotation = -math.radians(cozmo_rotation)
    offset_grid_x = grid_pose[0] - cozmo_origin[0]
    offset_grid_y = grid_pose[1] - cozmo_origin[1]
    x = inches_to_mm(offset_grid_x * math.cos(rotation) - offset_grid_y * math.sin(rotation))
    y = inches_to_mm(offset_grid_x * math.sin(rotation) + offset_grid_y * math.cos(rotation))
    angle = cozmo.util.Angle(degrees=proj_angle_deg(grid_pose[2] - cozmo_rotation))
    return cozmo.util.pose_z_angle(x, y, 0, angle)
          
          
def mm_to_inches(mm):
    return cozmo.util.distance_mm(mm).distance_inches
    
    
def inches_to_mm(inches):
    return cozmo.util.distance_inches(inches).distance_mm
          
          
# for some reason stop_all_motors leaves cozmo wiggling, this is to circumvent that
def really_stop(robot: cozmo.robot.Robot):
    robot.stop_all_motors()
    robot.drive_wheel_motors(0, 0)
            

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
