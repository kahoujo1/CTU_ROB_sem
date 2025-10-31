import cv2
import copy
import sys
import os
import json
import numpy as np
import time
from typing import List

CALIBRATION_IMAGES_DIR = "handeye_cal_img"
ROBOT_POSITIONS = "robot_positions.json"  # expected format is list of [x,y,z] positions in robot "coordinates"
ROBOT_HAND_ARUCO_ID = 6  # id of the aruco marker on the robot hand
ROBOT_HAND_ARUCO_SIZE = 30  # size of the aruco marker in mm
CALIBRATION_DATA_DIR = "handeye_cal_data.json"  # filename for the hand eye calibration json file
CALIBRATION_RESULTS_DIR = "handeye_cal_results.json"  # filename for the hand eye calibration results json file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

#from lib.robot_wrapper import Robot
from lib.se3 import SE3
from lib.so3 import SO3
#from lib.basler_camera import BaslerCamera
from lib import utils

def solve_AX_YB(a, b):
    """Solve A^iX=YB^i, return X, Y
    Args:
        a: list of SE3 objects
        b: list of SE3 objects
    Returns:
        X, Y: SE3 objects
    """
    rvec_a = [T.rotation.log() for T in a]
    tvec_a = [T.translation for T in a]
    rvec_b = [T.rotation.log() for T in b]
    tvec_b = [T.translation for T in b]

    Rx, tx, Ry, ty = cv2.calibrateRobotWorldHandEye(rvec_a, tvec_a, rvec_b, tvec_b)
    # return T_gripper_target, T_robot_camera
    return SE3(tx[:, 0], SO3(Rx)), SE3(ty[:, 0], SO3(Ry))

def load_json_data(filename: str) -> List[List[SE3]]:
    """Load the json data from the file

    Args:
        filename (str): filename to load the data from

    Returns:
        List[List[SE3]]: list of robot positions and eye positions
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    # Access the robot positions
    valid_robot_positions = data.get("valid_robot_positions", [])
    robot_positions = []
    # Process each position
    for i, position in enumerate(valid_robot_positions):
        rvec = np.array(position["rotation_matrix"])
        tvec = np.array(position["translation_vector"])
        robot_positions.append(SE3(tvec, SO3().exp(rvec)))
    eye_positions = []
    valid_eye_positions = data.get("hand_positions", [])
    for i, position in enumerate(valid_eye_positions):
        rvec = np.array(position["rotation_matrix"])
        tvec = np.array(position["translation_vector"])
        eye_positions.append(SE3(tvec, SO3().exp(rvec)))
    return robot_positions, eye_positions


def main():
    # read calib data from json file
    hand_pos, eye_pos = load_json_data(CALIBRATION_DATA_DIR)
    for eye in eye_pos:
        eye.translation /= 1000
    # calculate hand-eye calibration
    T_gt, T_rc = solve_AX_YB(hand_pos, eye_pos)
    data = {"T_gt": T_gt.to_dict(), "T_rc": T_rc.to_dict()}
    with open(os.path.join(os.path.dirname(__file__),CALIBRATION_RESULTS_DIR), "w") as f:
        json.dump(data, f, indent=4)
    print("Calibration successful, results saved to: ", CALIBRATION_RESULTS_DIR)

if __name__ == "__main__":
    main()
