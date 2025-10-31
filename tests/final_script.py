"""
First attempt at creating consequences of moves.
"""
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
import time


DEFAULT_POS = np.deg2rad([0, -30, -90, 0, -45, -10])
GET_OUT_OF_THE_FRAME_YOU_IDIOT_POSITION = np.deg2rad([-60, 0, -90, 0, -45, 0])
OFFSET_Z_MATRIX = SE3([0, 0, -0.05])
NUM_OF_IMAGES = 9


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

    r = Robot(True) # initialize
    # r = Robot(False)

    base1 = Base()
    base2 = Base()
    csv_path_1 = os.path.join(os.path.dirname(__file__), '../bases/positions_plate_09-10_test.csv')
    csv_path_2 = os.path.join(os.path.dirname(__file__), '../bases/positions_plate_11-12_test.csv')
    base1.csv2base(csv_path_1)
    base2.csv2base(csv_path_2)
    # load camera calibration data
    # Load camera calibration data
    calibration_file_path = os.path.join(os.path.dirname(__file__), '../lib/camera_matrix_4.json')
    with open(calibration_file_path, 'r') as f:
        calibration_data = json.load(f)

    camera_matrix = np.array(calibration_data['camera_matrix'])
    dist_coeff = np.array(calibration_data['dist_coeff'])
    camera_matrix_new = np.array(calibration_data['new_camera_matrix'])
    # estimate homogeneousboth bases
    estimator.set_camera_matrix(camera_matrix)
    estimator.set_distortion_coefficients(dist_coeff)
    for k in range(4):
        _ = r.move_to_angles(GET_OUT_OF_THE_FRAME_YOU_IDIOT_POSITION)
        time.sleep(2)
        # grab multiple images
        images = []
        for i in range(NUM_OF_IMAGES):
            time.sleep(0.02)
            img = camera.grab_image()

            if img is None or img.size == 0:
                print("No image captured")
                continue
            
            images.append(img)

        estimator.set_base(base1)
        if not estimator.calculate_pose_multiple_photos(images):
            print("Pose estimation failed.")
            return
        base1 = estimator.get_updated_base()
        estimator.set_base(base2)
        if not estimator.calculate_pose_multiple_photos(images):
            print("Pose estimation failed.")
            return
        base2 = estimator.get_updated_base()

        T_RC = SE3(translation = [
                0.4709404392331024,
                -0.001012245901015851,
                1.1588433705611427
            ], rotation = SO3.exp([
                2.2453229541584725,
                2.189286312669311,
                -0.04055677484668194
            ]))
        # take the first targets from the bases
        T_RT1 = base1.get_target_transforms_from_robot(T_RC)[k]
        T_RT2 = base2.get_target_transforms_from_robot(T_RC)[k]
        print("target transform 1")
        print(T_RT1.homogeneous())
        print("target transform 2")
        print(T_RT2.homogeneous())
        display_points.append(T_RC.homogeneous())
        display_points.append(T_RT1.homogeneous())
        display_points.append(T_RT2.homogeneous())
        display_points.append(SE3(translation = np.array([0,0,0])).homogeneous())
        
        # make the moves
        r.move_to_q(DEFAULT_POS)
        r.wait_for_motion_stop()
        r.gripper_drop()
        time.sleep(1)
        # BASE 1
        r.move_custom_gripper(T_RT1*OFFSET_Z_MATRIX)
       
        r.move_custom_gripper(T_RT1*SE3(translation=[0,0,-0.01]))
       
        
        r.move_custom_gripper(T_RT1*SE3(translation=[0,0,-0.002]))
        time.sleep(2)
        r.wiggle_wiggle_wiggle()   
        r.gripper_pick()
        #r.gripper_drop()
        time.sleep(2)

        r.move_custom_gripper(T_RT1*SE3(translation=[0,0,-0.01]))
        

        r.move_custom_gripper(T_RT1*OFFSET_Z_MATRIX)
       
        r.move_to_q(DEFAULT_POS)
        r.wait_for_motion_stop()
        time.sleep(1)

        # BASE 2
        r.move_custom_gripper(T_RT2*OFFSET_Z_MATRIX)
        r.move_custom_gripper(T_RT2*SE3(translation=[0,0,-0.01]))
        r.move_custom_gripper(T_RT2*SE3(translation=[0,0,-0.002]))
        #r.move_custom_gripper(T_RT2)
        time.sleep(2)
        #r.gripper_pick()
        r.wiggle_wiggle_wiggle()
        r.gripper_drop()
        time.sleep(2)
        r.move_custom_gripper(T_RT2*SE3(translation=[0,0,-0.002]))
        r.move_custom_gripper(T_RT2*SE3(translation=[0,0,-0.01]))
        r.move_custom_gripper(T_RT2*OFFSET_Z_MATRIX)
        r.move_to_q(DEFAULT_POS)
        r.wait_for_motion_stop()
        #d.setup_positions(display_points)
        #d.display_positions()

if __name__ == "__main__":
    main()