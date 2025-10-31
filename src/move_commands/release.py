""" Release the gripper and robot."""

import sys 
import os 

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from lib.robot_wrapper import Robot

r = Robot(False)
# r.gripper_pick()
r.gripper_drop()
r.release()
