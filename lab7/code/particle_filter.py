from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np

# ## Zach Bloomquist & Taylor Hearn
# ## CS 3630 Lab 5

ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4 = [.02, .02, .01, .01]
MIN_PROBABILITY = 0.05  # particles with p < this will be removed. increase to find a fix faster, but lowers stability


# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion upda

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- noisy odometry measurement, a pair of robot pose, i.e. last time
                step pose and current time step pose, both in form [x, y, theta]

        Returns: the list of particle represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    # algorithm https://i.imgur.com/rJ6CUmu.png
    prev_odom, cur_odom = odom
    if np.array_equal(prev_odom, cur_odom):
        return particles
    # rot1: angle robot needs to turn before moving
    delta_rot1 = diff_heading_deg(np.degrees(np.arctan2(cur_odom[1] - prev_odom[1],
                                                        cur_odom[0] - prev_odom[0])), prev_odom[2])
    # trans: distance robot should traverse forward after rotating
    delta_trans = np.sqrt((prev_odom[0] - cur_odom[0])**2 + (prev_odom[1] - cur_odom[1])**2)
    # rot2: angle robot should turn when reaching destination to match heading
    delta_rot2 = proj_angle_deg(cur_odom[2] - prev_odom[2] - delta_rot1)
    global moved
    moved = delta_rot1 + delta_trans + delta_rot2 > 0.1
    for i, particle in enumerate(particles):
        # add gaussian noise to deltas
        pdelta_rot1 = delta_rot1 - random.gauss(0.0, ALPHA_1 * delta_rot1 + ALPHA_2 * delta_trans)
        pdelta_trans = delta_trans - random.gauss(0.0, ALPHA_3 * delta_trans + ALPHA_4 * (delta_rot1 + delta_rot2))
        pdelta_rot2 = delta_rot2 - random.gauss(0.0, ALPHA_1 * delta_rot2 + ALPHA_2 * delta_trans)
        particle.move(pdelta_rot1, pdelta_trans, pdelta_rot2)
        particles[i] = Particle(particle.x, particle.y, heading=proj_angle_deg(particle.h))  # needs a new object
    return particles


moved = True
last_measured_marker_list = None


# ------------------------------------------------------------------------
def new_info(measured_marker_list):
    global last_measured_marker_list, moved
    
    if len(measured_marker_list) == 0:
        return False
        
    if last_measured_marker_list is None or len(last_measured_marker_list) != len(measured_marker_list) or moved:
        return True
        
    for i, marker in enumerate(measured_marker_list):
        last_marker = last_measured_marker_list[i]
        if marker[0] != last_marker[0] or marker[1] != last_marker[1] or marker[2] != last_marker[2]:
            return True
        
    return False

    
def calculate_particle_probability(particle, measured_marker_list, grid):
    # Obtain the list of localization markers that a robot would see if it were really at this particle
    markers_visible_to_particle = particle.read_markers(grid)
    if not grid.is_free(particle.x, particle.y) or len(measured_marker_list) != len(markers_visible_to_particle):
        # particles within an obstacle or outside the map should have a weight of 0
        return 0
        
    # compare the list of markers obtained by simulating the particle’s field of view to the observed markers
    found_markers = []
    for measured_marker in measured_marker_list:
        # try to match the markers by distance - nearest marker is probably it
        min_distance = -1
        min_marker = None
        i = None
        for i, visible_marker in enumerate(markers_visible_to_particle):
            marker_distance = grid_distance(measured_marker[0], measured_marker[1],
                                            visible_marker[0], visible_marker[1])
            if marker_distance < min_distance or min_distance < 0:
                min_distance = marker_distance
                min_marker = visible_marker
        if i is not None:
            markers_visible_to_particle.pop(i)
            found_markers.append([measured_marker, min_marker, min_distance])
            
    # calculate weight of this particle based on how likely it is to see the markers the robot sees
    prob = 1.0
    if len(found_markers) == 0:
        prob = 0.0
    for marker in found_markers:
        measured_marker, min_marker, dist = marker
        angle = diff_heading_deg(measured_marker[2], min_marker[2])
        prob *= math.exp(0 - ((dist**2)/(2*(MARKER_TRANS_SIGMA**2)) + (angle**2)/(2*(MARKER_ROT_SIGMA**2))))
    return prob
    

def replace_unlikely_particles(particles, probabilities, measured_marker_list, grid):
    for (i, prob) in enumerate(probabilities):
        if prob < MIN_PROBABILITY:
            particles[i] = Particle.create_random(1, grid)[0]
            probabilities[i] = calculate_particle_probability(particles[i], measured_marker_list, grid)


def measurement_update(particles, measured_marker_list, grid):
    """ Particle filter measurement update

        Arguments: 
        particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
                before meansurement update
        measured_marker_list -- robot detected marker list, each marker has format:
                measured_marker_list[i] = (rx, ry, rh)
                rx -- marker's relative X coordinate in robot's frame
                ry -- marker's relative Y coordinate in robot's frame
                rh -- marker's relative heading in robot's frame, in degree
        grid -- grid world map, which contains the marker information, 
                see grid.h and CozGrid for definition

        Returns: the list of particle represents belief p(x_{t} | u_{t})
                after measurement update
    """
    global last_measured_marker_list, moved
    
    # no input, what we're doing is useless
    if not new_info(measured_marker_list):
        return particles
    last_measured_marker_list = list(measured_marker_list)
    
    probabilities = []
    for j, particle in enumerate(particles):
        probabilities.append(calculate_particle_probability(particle, measured_marker_list, grid))
        
    max_prob = max(probabilities)
    if max_prob < 1 / len(probabilities):
        return particles
        
    replace_unlikely_particles(particles, probabilities, measured_marker_list, grid)
            
    # normalize probabilities and choose particles
    probabilities = np.divide(probabilities, [np.sum(probabilities)])
    # sanity check on probs - if none of these particles are likely, return stale particle distribution
    max_prob = max(probabilities)
    if max_prob < 1 / len(probabilities):
        return particles
    measured_particles = np.random.choice(particles, p=probabilities, size=PARTICLE_COUNT)

    if len(measured_particles) == 0:
        return particles
    return measured_particles


