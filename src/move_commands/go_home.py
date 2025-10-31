""" Sned the robot to the home position """

import sys 
import os 
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lib.robot_wrapper import Robot

r = Robot(True) 
r.soft_home()
r.close()