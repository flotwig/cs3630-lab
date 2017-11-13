from grid import *
from particle import Particle
from utils import *
from setting import *
import numpy as np

ALPHA_1, ALPHA_2, ALPHA_3, ALPHA_4 = [0.1, 0.1, 0.1, 0.1]

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
    #odom = add_odometry_noise(odom, ODOM_HEAD_SIGMA, ODOM_TRANS_SIGMA) # necessary?
    prev_odom, cur_odom = odom
    diff_odom = np.subtract(cur_odom, prev_odom)
    delta_rot1 = diff_heading_deg(np.arctan2(diff_odom[1], diff_odom[0]), prev_odom[2])
    delta_trans = np.sqrt(diff_odom[0]**2 + diff_odom[1]**2)
    delta_rot2 = diff_odom[2] - delta_rot1
    for particle in particles:
        pdelta_rot1 = delta_rot1 - np.random.normal(loc=0, scale=abs(ALPHA_1 * delta_rot1 + ALPHA_2 * delta_trans))
        pdelta_trans = delta_trans - np.random.normal(loc=0, scale=abs(ALPHA_3 * delta_trans + ALPHA_4 * (delta_rot1 + delta_rot2)))
        pdelta_rot2 = delta_rot2 - np.random.normal(loc=0, scale=abs(ALPHA_1 * delta_rot2 + ALPHA_2 * delta_trans))
        particle.x += pdelta_trans * np.cos(particle.h + pdelta_rot1)
        particle.y += pdelta_trans * np.sin(particle.h + pdelta_rot1)
        particle.h += pdelta_rot1 + pdelta_rot2
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
    for particle in particles:
        # identify pairings between markers seen by particle & particles cozmo knows of
        markers_visible_to_particle = particle.read_markers(grid)
        # if the particle is outside map or in an obstacle, remove it
        if not grid.is_free(particle.x, particle.y):
            probabilities.append(0)
            continue
        found_markers = []
        for measured_marker in measured_marker_list:
            # find closest marker out of ones visible to particle
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
                found_markers.append([measured_marker, min_marker])
        # calculate weight of this particle
        prob = 1.0
        for marker in found_markers:
            measured_marker, min_marker = marker
            dist = grid_distance(measured_marker[0], measured_marker[1], min_marker[0], min_marker[1])
            angle = diff_heading_deg(measured_marker[2], min_marker[2])
            prob *= math.e ** (0 - ((dist**2)/(2*(MARKER_TRANS_SIGMA**2)) + (angle**2)/(2*(MARKER_ROT_SIGMA**2))))
        probabilities.append(prob)
    # normalize probabilities and choose particles
    probabilities = np.divide(probabilities, np.sum(probabilities))
    measured_particles = np.random.choice(particles, p=probabilities, size=5000)
    return measured_particles
