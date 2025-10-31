
""" Move the robot to a specific pose or joint values """

import sys 
import os 
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.robot_wrapper import Robot
from lib.utils import get_R

r = Robot()

while True:
    pose = r.get_flange_pos() # T = [R P, 0 1]
    q = r.get_q()
    print(f"Current pose {pose}")
    print(f"Current joint values {q}")

    in1 = input("do you want to set another pose 1 for world, 2 for angles, n for stop?")
    if in1 == 'n':
        break
    # get new position
    elif in1 == '1':
        in2 = input("Enter new coordinates as x,y,z: ")
        if in2 != "s":
            x, y, z = map(float, in2.split(','))
            pose[:3, 3] = np.array([x, y, z]).T
        
        # get new rotation
        in3 = input("Enter new rotation as roll, pitch, yaw:")
        if in3 != "s":
            roll, pitch, yaw = map(float, in3.split(','))
            R = get_R(roll, pitch, yaw)
            pose[:3, :3] = R

        print(f"moving to new pose {pose}")
        sakis = r.move_to_coordinates(pose)
        print(f"robot moved: {sakis}")
    elif in1 == "2":
        in5 = input("Enter mode '+' for add else for absolut")
        if in5 =="+":
            in4 = input("enter the angles q1, ... , qn: ")
            if in4 != "s":
                q1, q2, q3, q4, q5, q6 = map(float, in4.split(','))
                q += np.array([q1, q2, q3, q4, q5, q6])

            r.move_to_angles(q)
        else:
            in6 = input("enter the angles q1, ... , qn: ")
            if in6 != "s":
                q1, q2, q3, q4, q5, q6 = map(float, in6.split(','))
                q = np.array([q1, q2, q3, q4, q5, q6])

            r.move_to_angles(q)
    
    else:
        continue



r.close()
