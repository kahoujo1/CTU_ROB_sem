"""Gets the camera image and detects the Aruco markers on it."""
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
from lib.base import Base, BASE_SIZE_X, BASE_SIZE_Y, CUBE_SIZE, ARUCO_SIZE
from lib.robot_wrapper import Robot

from lib.se3 import SE3 # SE2/3, SO2/3
from lib.so3 import SO3
import json
from lib.display import Display

OFFSET_Z_MATRIX = SE3([0, 0, -0.02])

def main():
    d = Display()
    display_points = []
    camera: BaslerCamera = BaslerCamera()
    estimator = BasePoseEstimator()

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
    # Starts capturing images by camera.
    # If it is not done, the grab_image method starts it itself.
    #camera.start()
    r = Robot()
    r.initialize()
    
    mybase = Base()
    csv_file_path = os.path.join(os.path.dirname(__file__), '../bases/positions_plate_01-02.csv')
    mybase.csv2base(csv_file_path)
    estimator.set_base(mybase)

    # Load camera calibration data
    calibration_file_path = os.path.join(os.path.dirname(__file__), 'test_charuco.json')
    with open(calibration_file_path, 'r') as f:
        calibration_data = json.load(f)

    camera_matrix = np.array(calibration_data['camera_matrix'])
    dist_coeff = np.array(calibration_data['dist_coeff'])
    camera_matrix_new = np.array(calibration_data['new_camera_matrix'])
    
    estimator.set_camera_matrix(camera_matrix)
    estimator.set_distortion_coefficients(dist_coeff)
    r.move_to_gate()
    # Take one image from the camera
    img = camera.grab_image()
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeff, None, camera_matrix_new)

    if not ((img is not None) and (img.size > 0)) or not estimator.calculate_pose(img):
        print("Pose estimation failed.")
        return
    mybase = estimator.get_updated_base()
    transf = mybase.get_target_transforms_from_camera()[0]
    
    print("Transformation:",transf.translation, "Rotation:",transf.rotation.log())
    # T_RC [ 2.21034901  2.09587107 -0.1103611 ] [246.53357674  59.30931939 890.59925509]
    """
    old trasformation
    T_RC = SE3(translation = [
            0.4278984573655581,
            -0.025652871856891732,
            1.1942397951486097
        ], rotation = SO3.exp([
            2.1878865977867985,
            2.226904463166639,
            0.06594805363458514
        ]))
    """
    T_RC = SE3(translation = [
            0.506487725677565,
            -0.06794535466141832,
            1.2124340501061202
        ], rotation = SO3.exp([
            -2.1923665660351577,
            -2.171859767723388,
            0.014541513508662035
        ]))
    T_RT = mybase.get_target_transforms_from_robot(T_RC)[0]
    display_points.append(T_RC.homogeneous())
    display_points.append(T_RT.homogeneous())
    display_points.append(SE3(translation = np.array([0,0,0])).homogeneous())
    display_points.append(r.get_gripper_pos())
    tvec = T_RT.translation
    rvec = T_RT.rotation.log()
    print("tvec = ", T_RT.translation, "rvec = ", T_RT.rotation.log())
    print(r.get_gripper_pos())
    display_points.append(r.get_desiret_pos_of_custom_gripper(T_RT).homogeneous())
    r.gripper_drop()
    _ = r.move_custom_gripper(T_RT*OFFSET_Z_MATRIX)

    display_points.append(r.get_gripper_pos())
    
    d.setup_positions(display_points)
    d.display_positions()

    camera.close()

if __name__ == '__main__':
    main()