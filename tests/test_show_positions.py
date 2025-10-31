import sys 
import os
import numpy as np

# appending a path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.display import Display


T1 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
T2 = np.array([[0, 1, 0, 2], [1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
T3 = np.array([[1, 0, 0, 3], [0, 1, 0, 2], [0, 0, 1, 1], [0, 0, 0, 1]])
T4 = np.array([[0, 0, 1, 4], [0, 1, 0, 3], [1, 0, 0, 2], [0, 0, 0, 1]])

a = [T1, T2, T3, T4]

display = Display()

display.setup_positions(a)  

display.display_positions()