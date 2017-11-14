from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np

ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4 = [0.01, 0.01, 0.01, 0.01]
MIN_PROBABILITY = .05  # particles with p < this will be removed

# ------------------------------------------------------------------------
def motion_update(particles, odom):
    """ Particle filter motion update

        Arguments:
        particles -- input list of particle represents belief p(x_{t-1} | u_{t-1})
                before motion update
        odom -- noisy odometry measurement, a pair of robot pose, i.e. last time
                step pose and current time step pose, both in form [x, y, theta]

        Returns: the list of particle represents belief \tilde{p}(x_{t} | u_{t})
                after motion update
    """
    #algorithm https://i.imgur.com/rJ6CUmu.png
    #odom = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA) # necessary?
    prev_odom, cur_odom = odom
    if np.array_equal(prev_odom, cur_odom):
        return particles
    # rot1: angle robot needs to turn before moving
    delta_rot1 = np.degrees(np.arctan2(cur_odom[1] - prev_odom[1], cur_odom[0] - prev_odom[0])) - prev_odom[2]
    # trans: distance robot should traverse forward after rotating
    delta_trans = np.sqrt((cur_odom[1] - prev_odom[1])**2 + (cur_odom[0] - prev_odom[0])**2)
    # rot2: angle robot should turn when reaching destination to match heading
    delta_rot2 = cur_odom[2] - prev_odom[2] - delta_rot1
    for particle in particles:
        # add gaussian noise to deltas
        pdelta_rot1 = delta_rot1 - random.gauss(0.0, ALPHA_1 * delta_rot1 + ALPHA_2 * delta_trans)
        pdelta_trans = delta_trans - random.gauss(0.0, ALPHA_3 * delta_trans + ALPHA_4 * (delta_rot1 + delta_rot2))
        pdelta_rot2 = delta_rot2 - random.gauss(0.0, ALPHA_1 * delta_rot2 + ALPHA_2 * delta_trans)
        particle.move(pdelta_rot1, pdelta_trans, pdelta_rot2)
    return particles

# ------------------------------------------------------------------------
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
    measured_particles = []
    probabilities = []
    for j, particle in enumerate(particles):  # for each particle:
        # Obtain the list of localization markers that a robot would see if it were really at this particle
        markers_visible_to_particle = particle.read_markers(grid)
        if not grid.is_free(particle.x, particle.y):
            # particles within an obstacle or outside the map should have a weight of 0
            probabilities.append(0)
            continue
        # compare the list of markers obtained by simulating the particle’s field of view to the observed markers
        found_markers = []
        for measured_marker in measured_marker_list:
            # try to match the markers by distance - nearest marker is probably it
            min_distance = -1
            min_marker = None
            i = None
            for i, visible_marker in enumerate(markers_visible_to_particle):
                marker_distance = grid_distance(measured_marker[0], measured_marker[1], visible_marker[0], visible_marker[1])
                if marker_distance < min_distance or min_distance < 0:
                    min_distance = marker_distance
                    min_marker = visible_marker
            if i is not None:
                markers_visible_to_particle.pop(i)
                found_markers.append([measured_marker, min_marker, min_distance])
        # calculate weight of this particle based on how likely it is to see the markers the robot sees
        prob = 1.0
        for marker in found_markers:
            measured_marker, min_marker, dist = marker
            angle = diff_heading_deg(measured_marker[2], min_marker[2])
            prob *= math.exp(0 - ((dist**2)/(2*(MARKER_TRANS_SIGMA**2)) + (angle**2)/(2*(MARKER_ROT_SIGMA**2))))
        if prob < MIN_PROBABILITY:
            prob = 0  # eliminating any particles with very low probabilities and replacing them with random samples
        probabilities.append(prob)
    # normalize probabilities and choose particles
    probabilities = np.divide(probabilities, [np.sum(probabilities)])
    measured_particles = np.random.choice(particles, p=probabilities, size=4975)
    # fill in the remaining particles with random ones - better way to do this? uniqify measured particles?
    new_particles = Particle.create_random(5000 - len(measured_particles), grid)
    return np.concatenate([measured_particles, new_particles])