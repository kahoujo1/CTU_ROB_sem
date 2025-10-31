import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.robot_wrapper import Robot
import lib.utils as utils

import os
MIN_X = 0.3 # [m]
MAX_X = 0.5 # [m]
DELTA_X = 0.08 # [m]

MIN_Z = 0.25 # [m]
MAX_Z = 0.35 # [m]
DELTA_Z = 0.04 # [m]

MIN_Y = -0.4 # [q in rad]
MAX_Y = 0.4 # [q in rad]
DELTA_Y = 0.1 # [q in rad]

Q5_MIN = 0
Q5_MAX = np.pi/8 + 0.01
Q5_DELTA = np.pi/8

Q6_MIN = 0#-np.pi/4
Q6_MAX = 0#np.pi/4+0.01
Q6_DELTA = np.pi/4


def generate_robot_positions(robot: Robot):
    """
    Generate a list of robot positions (in) 
    """
    robot_positions = []
    for x in np.arange(MIN_X, MAX_X, DELTA_X):
        for z in np.arange(MIN_Z, MAX_Z, DELTA_Z):
            # calculate the inverse kinematics for the robot
            tvec = np.array([x, 0, z])
            rvec = np.array([0, 160, 0])
            robot_pos = utils.vector_to_se3(tvec, rvec)
            ik = robot.return_best_ik(robot_pos.homogeneous())
            if ik is not None:
                for y in np.arange(MIN_Y, MAX_Y, DELTA_Y): # go through y values
                    for q5 in np.arange(Q5_MIN, Q5_MAX, Q5_DELTA): # second last joint
                            q6 = 0
                            pos = ik + [y, 0, 0, 0, q5, q6]
                            if robot.in_limits(pos):
                                robot_positions.append(pos)
    print("Number of robot positions: ", len(robot_positions))
    return robot_positions

def main():
    r = Robot()
    robot_positions = generate_robot_positions(r)
    #print(robot_positions)

if __name__ == "__main__":
    main()