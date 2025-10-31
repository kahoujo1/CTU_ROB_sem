import cv2
import copy
import sys
import os
import numpy as np


# appending a path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Our Basler camera interface
from lib.basler_camera import BaslerCamera
from lib.base_pose_estimator import BasePoseEstimator
import json
from lib.robot_wrapper import Robot
from lib.se3 import SE3
from lib.so3 import SO3
from lib.basler_camera import BaslerCamera
from lib import utils
from lib.display import Display
ROBOT_HAND_ARUCO_SIZE = 29  # size of the aruco marker in mm
ROBOT_HAND_ARUCO_ID = 6  # id of the aruco marker on the robot hand

def find_aruco_id_vec(corners, ids, ARUCO_SIZE: int, wanted_id: int) -> SE3:
    """
    Find the vectors of the wanted aruco id in the image.
    """
    calibration_file_path = os.path.join(os.path.dirname(__file__), "test_charuco.json")
    with open(calibration_file_path, "r") as f:
        calibration_data = json.load(f)

    camera_matrix = np.array(calibration_data["camera_matrix"])
    dist_coeff = np.array(calibration_data["dist_coeff"])
    camera_matrix_new = np.array(calibration_data["new_camera_matrix"])

    base_points = np.array(
        [
            # bottom left aruco marker
            [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top left corner
            [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top right corner
            [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom right corner
            [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom left corner
        ]
    )
    world_points = corners[0]
    ret, rvec, tvec = cv2.solvePnP(
        base_points,
        world_points,
        camera_matrix,
        dist_coeff,
        False,
        cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if ret:
        rvec = np.reshape(rvec, (3,))
        tvec = np.reshape(tvec, (3,))
        return utils.vector_to_se3(tvec, rvec)
    else:
        print("Error in finding vectors for ARUCO ID: ", wanted_id) 


def main():
    camera: BaslerCamera = BaslerCamera()
    estimator = BasePoseEstimator()
    d = Display()
    display_points = []
    # Camera can be connected based on its' IP or name:
    # Camera for robot CRS 93
    #   camera.connect_by_ip("192.168.137.107")
    #   camera.connect_by_name("camera-crs93")
    # Camera for robot CRS 97
    #   camera.connect_by_ip("192.168.137.106")
    #   camera.connect_by_name("camera-crs97")
    camera.connect_by_name("camera-crs97")
 
    # Open the communication with the camera
    camera.open()
    # Set capturing parameters from the camera object.
    # The default parameters (set by constructor) are OK.
    # When starting the params should be send into the camera.
    camera.set_parameters()
    # Capture image
    r = Robot()
    img = camera.grab_image()
    T_RC = SE3(translation=[
            0.4278984573655581,
            -0.025652871856891732,
            1.1942397951486097
        ], rotation=SO3().exp([
            2.1878865977867985,
            2.226904463166639,
            0.06594805363458514
        ],))
    T_GT = SE3(translation=[
            -0.0573864908197208,
            0.0424589254817036,
            0.008623086301348781
        ], rotation=SO3().exp([
            0.03506602902221078,
            -3.1353644128153566,
            0.041207171854970734
        ]))
    #T_RC.translation /= 1000
    #T_GT.translation /= 1000
    T_RG = r.fk(r.get_q())
    T_RG = SE3(translation=T_RG[:3,3],rotation=SO3(T_RG[:3,:3]))
    print(T_RC.homogeneous())
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()

    corners, ids, rejected = aruco.detectMarkers(
            img, aruco_dict, parameters=parameters)
    T_CT = find_aruco_id_vec(corners, ids, ROBOT_HAND_ARUCO_SIZE, ROBOT_HAND_ARUCO_ID)
    T_CT.translation /= 1000
    T_RT = T_RC*T_CT
    T_RG_2 = T_RC * T_CT * T_GT.inverse()
    print("T_RC")
    print(T_RC.translation, T_RC.rotation.log())

    print("T_RG from forward kinematics: ")
    print(T_RG.translation, T_RG.rotation.log())
    print("T_RG from calibration: ")
    print(T_RG_2.translation, T_RG_2.rotation.log())
    display_points.append(SE3(translation = np.array([0,0,0])).homogeneous())# robot
    display_points.append(T_RG.homogeneous())
    display_points.append(T_RG_2.homogeneous())
    display_points.append(T_RC.homogeneous())
    display_points.append(T_RT.homogeneous())
    d.setup_positions(display_points)
    d.display_positions()
if __name__ == "__main__":
    main()