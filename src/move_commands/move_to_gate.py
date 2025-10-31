""" Moves the robot to the gate. """
import sys 
import os 
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from lib.robot_wrapper import Robot

r = Robot(False)
r.move_to_gate()
r.close()