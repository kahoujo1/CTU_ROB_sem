"""
Module for hand-eye calibration using images of a calibration target taken from the camera.
"""

import cv2
import copy
import sys
import os
import json
import numpy as np
import time
import math
from typing import List, Tuple
from scipy.optimize import minimize, least_squares
# appending a path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import utils
from display import Display
from se3 import SE3
from so3 import SO3
ROBOT_HAND_ARUCO_ID = 6  # id of the aruco marker on the robot hand
ROBOT_HAND_ARUCO_SIZE = 0.0446 # size of the aruco marker in m
MY_RESULT = "calibration_data/reprojection_error_results.json"

class HandEyeCalib:
    def __init__(self) -> None:
        self.iter = 0


    def reprojection_error(self, params: np.array, K: np.array, dist_coef: np.array, points_3d: np.array, points_2d: np.array, T_RG: List[SE3]) -> float:
        """
        Calculates the reprojection error.

        Used as the optimazation criterion to calculate the robot to camera (and gripper to target) calibration.
        
        Args:aruco_ids:
            - params: np.array - Parameters to optimize (rvecs and tvecs of the transformations) [tvec_RC, rvec_RC, tvec_GT, rvec_GT] (tvecs in meters)
            - K: np.array - Camera matrix
            - dist_coef: np.array - Distortion coefficients
            - points_3d: np.array - 3D points of the calibration target (in the target coordinate system)
            - points_2d: np.array - 2D points of the calibration target (the image points)
            - T_RG: List[SE3] - List of gripper to robot transformation matrices (from forward kinematics) (in meters)

        Returns:
            - float - Reprojection error
        """
        assert len(params) == 12, "The parameters should have 12 elements."
        assert len(points_3d) == len(points_2d), "The number of 3D and 2D points should be the same."
        # create the SE3 objects from the parameters
        tvec_RC = params[:3]
        rvec_RC = params[3:6]
        tvec_GT = params[6:9]
        rvec_GT = params[9:]
        d = Display()
        display_points = []
        T_RC = SE3(translation=tvec_RC, rotation=SO3.exp(rvec_RC))
        T_GT = SE3(translation=tvec_GT, rotation=SO3.exp(rvec_GT))
        error = 0
        for i in range(len(T_RG)):
            T_CT = T_RC.inverse() * T_RG[i] * T_GT
            display_points.append(T_CT.homogeneous())
            _, rvec_a, tvec_a = cv2.solvePnP(points_3d[i], points_2d[i][0], K, dist_coef, False, cv2.SOLVEPNP_IPPE_SQUARE)
            T_CT_a = SE3(translation=tvec_a.flatten(), rotation=SO3.exp(rvec_a.flatten()))
            display_points.append(T_CT_a.homogeneous())
            for x in range(4): # for each marker corner
                T_T_Corner = SE3(translation=points_3d[i][x], rotation=SO3())
                T_C_Corner = T_CT * T_T_Corner
                display_points.append(T_C_Corner.homogeneous())
                tvec = T_C_Corner.translation
                rvec = T_C_Corner.rotation.log()
                points_2d_proj = cv2.projectPoints(np.array([tvec]), np.zeros(3), np.zeros(3), K, dist_coef)[0][0][0]
                cur_error = np.linalg.norm(points_2d_proj - points_2d[i][0][x])**2
                error += cur_error 
        self.iter += 1
        if self.iter == 1:
            print("initial error", error/((len(T_RG)*4)))
        if self.iter % 100 == 0:
            print("parameters", params)
            print("error normalized",  error/((len(T_RG)*4)))
        return error
    
    def reprojection_error_ls(self, params: np.array, K: np.array, dist_coef: np.array, points_3d: np.array, points_2d: np.array, T_RG: List[SE3]) -> np.array:
        """
        New and revised reprojection error function for least squares optimization.

        Used as the optimazation criterion to calculate the robot to camera (and gripper to target) calibration.

        Args:
            - params: np.array - Parameters to optimize (rvecs and tvecs of the transformations) [tvec_RC, rvec_RC, tvec_GT, rvec_GT] (tvecs in meters)
            - K: np.array - Camera matrix
            - dist_coef: np.array - Distortion coefficients
            - points_3d: np.array - 3D points of the calibration target (in the target coordinate system)
            - points_2d: np.array - 2D points of the calibration target (the image points)
            - T_RG: List[SE3] - List of gripper to robot transformation matrices (from forward kinematics) (in meters)

        Returns:
            - np.array - Reprojection error residuals
        """
        assert len(params) == 12, "The parameters should have 12 elements."
        assert len(points_3d) == len(points_2d), "The number of 3D and 2D points should be the same."
        # create the SE3 objects from the parameters
        tvec_RC = params[:3]
        rvec_RC = params[3:6]
        tvec_GT = params[6:9]
        rvec_GT = params[9:]
        T_RC = SE3(translation=tvec_RC, rotation=SO3.exp(rvec_RC))
        T_GT = SE3(translation=tvec_GT, rotation=SO3.exp(rvec_GT))
        error = []
        projected_points = []
        for i in range(len(T_RG)):
            T_CT = T_RC.inverse() * T_RG[i] * T_GT
            tmp_projected_points = cv2.projectPoints(points_3d[i], T_CT.rotation.log(), T_CT.translation, K, dist_coef)[0]
            projected_points.append(tmp_projected_points.squeeze())
        points_2d = np.array(points_2d)
        projected_points = np.array(projected_points)
        error = projected_points - points_2d
        return error.reshape(-1)
        

            
    def prepare_points(self, calib_imgs_path: str) -> Tuple[List[np.array], List[np.array], List[SE3]]:
        """
        Prepare the 3D and 2D points for the optimization.

        Args:
            - calib_imgs_path: str - Path to the calibration images

        Returns:
            - Tuple[List[np.array], List[np.array]] - 3D points in target coordinate system and 2D points in camera image
        """
        img_files = utils.get_images_from_dir(calib_imgs_path)
        points_3d = []
        points_2d = []
        print("len of img_files",len(img_files))
        for img_path in img_files:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # find the aruco markers
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
            parameters = cv2.aruco.DetectorParameters()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
            if ROBOT_HAND_ARUCO_ID not in ids:
                print(f"Robot hand aruco marker not found in {img_path}")
                continue
            # the 3D points in the target coordinate system (located in the middle of the aruco marker)
            tmp_points_3d = np.array([[-ROBOT_HAND_ARUCO_SIZE / 2, ROBOT_HAND_ARUCO_SIZE / 2, 0],
                                        [ROBOT_HAND_ARUCO_SIZE / 2, ROBOT_HAND_ARUCO_SIZE / 2, 0],
                                        [ROBOT_HAND_ARUCO_SIZE / 2, -ROBOT_HAND_ARUCO_SIZE / 2, 0],
                                        [-ROBOT_HAND_ARUCO_SIZE / 2, -ROBOT_HAND_ARUCO_SIZE / 2, 0]])
            # the 2D points in the image
            tmp_points_2d = np.array(corners[np.where(ids == ROBOT_HAND_ARUCO_ID)[0][0]])
            points_3d.append(tmp_points_3d)
            points_2d.append(tmp_points_2d[0])
        return points_3d, points_2d.squeeze()

    def load_fk_transformation(self, hand_eye_calib_data: str) -> List[SE3]:
        """Loads the forward kinematics data from a json file.

        Args:
            hand_eye_calib_data (str): Path to the hand-eye calibration data

        Returns:
            List[SE3]: List of SE3 objects representing the forward kinematics data
        """
        with open(hand_eye_calib_data, "r") as f:
            data = json.load(f)
        fk_data = data["valid_robot_positions"]
        print("LEN OF FK DATA",len(fk_data))
        list_T_RG = []
        for i, position in enumerate(fk_data):
            tvec = position["translation_vector"]
            rvec = position["rotation_matrix"]
            list_T_RG.append(SE3(translation=tvec, rotation=SO3().exp(rvec)))
        return list_T_RG
    
    def load_fk_transformation_and_points(self, hand_eye_calib_data: str) -> Tuple[List[SE3], List[np.array], List[np.array]]:
        """Loads the forward kinematics data from a json file, and the corners of aruco markers.
        
        Args:
            hand_eye_calib_data (str): Path to the hand-eye calibration data

        Returns:
            Tuple[List[SE3], List[np.array], List[np.array]]: List of SE3 objects representing the forward kinematics data, 3D points in target coordinate system and 2D points in camera image
        """
        with open(hand_eye_calib_data, "r") as f:
            data = json.load(f)
        fk_data = data["valid_robot_positions"]
        corners = data["hand_aruco_corners"]
        print("LEN OF FK DATA",len(fk_data))
        list_T_RG = []
        points_3d = []
        points_2d = []
        for i, position in enumerate(fk_data):
            tvec = position["translation_vector"]
            rvec = position["rotation_matrix"]
            list_T_RG.append(SE3(translation=tvec, rotation=SO3().exp(rvec)))
            # the 3D points in the target coordinate system (located in the middle of the aruco marker)
            tmp_points_3d = np.array([[-ROBOT_HAND_ARUCO_SIZE / 2, ROBOT_HAND_ARUCO_SIZE / 2, 0],
                                        [ROBOT_HAND_ARUCO_SIZE / 2, ROBOT_HAND_ARUCO_SIZE / 2, 0],
                                        [ROBOT_HAND_ARUCO_SIZE / 2, -ROBOT_HAND_ARUCO_SIZE / 2, 0],
                                        [-ROBOT_HAND_ARUCO_SIZE / 2, -ROBOT_HAND_ARUCO_SIZE / 2, 0]])
            # the 2D points in the image

            tmp_points_2d = np.array(corners[i])
            points_3d.append(tmp_points_3d)
            points_2d.append(tmp_points_2d.squeeze())
        return list_T_RG, points_3d, points_2d

    def load_initial_parameters(self, hand_eye_calib_results: str) -> np.array:
        """Loads the initial parameters from a json file.

        Args:
            hand_eye_calib_results (str): Path to the initial hand-eye calibration results

        Returns:
            np.array: Initial parameters [tvec_RC, rvec_RC, tvec_GT, rvec_GT]
        """
        with open(hand_eye_calib_results, "r") as f:
            data = json.load(f)
        T_GT = data["T_gt"]
        T_RC = data["T_rc"]
        tvec_RC = T_RC["translation_vector"]
        rvec_RC = T_RC["rotation_matrix"]
        tvec_GT = T_GT["translation_vector"]
        rvec_GT = T_GT["rotation_matrix"]
        return np.concatenate((tvec_RC, rvec_RC, tvec_GT, rvec_GT))
    
    def reprojection_optimization(self, camera_calib_filename: str, hand_eye_calib_data:str, hand_eye_calib_results: str) -> bool:
        """
        Optimizes the camera to robot and gripper to target calibration using reprojection error.

        Args:
            camera_calib_filename: (str) Camera calibration filename
            hand_eye_calib_data: (str) Path to the hand-eye calibration data
            hand_eye_calib_results: (str) Path to the initial hand-eye calibration results

        Returns:
            bool: True if the optimization was successful, False otherwise
        """
        # load the camera calibration data
        camera_matrix, dist_coef,_ = utils.load_camera_calibration_data(camera_calib_filename)
        print("Camera data loaded.")
        print("Camera matrix:\n", camera_matrix)
        print("Distortion coefficients:\n", dist_coef)
        # load the forward kinematics data and the points
        list_T_RG, points_3d, points_2d = self.load_fk_transformation_and_points(hand_eye_calib_data)
        # initial parameters
        init_params = self.load_initial_parameters(hand_eye_calib_results)
        self.reprojection_error_ls(init_params, camera_matrix, dist_coef, points_3d, points_2d, list_T_RG)
        # create bounds for the rvecs
        tvec_RC = np.array(init_params[:3])
        rvec_RC = np.array(init_params[3:6])
        tvec_GT = np.array(init_params[6:9])
        rvec_GT = np.array(init_params[9:])
        # for now, make the rvec vectors static
        tvec_offset = 0.5
        lower_bounds = [tvec_RC[0] - tvec_offset, tvec_RC[1] - tvec_offset, tvec_RC[2] - tvec_offset, rvec_RC[0], rvec_RC[1], rvec_RC[2],
                tvec_GT[0] - tvec_offset, tvec_GT[1] - tvec_offset, tvec_GT[2] - tvec_offset, rvec_GT[0], rvec_GT[1], rvec_GT[2]]
        upper_bounds = [tvec_RC[0] + tvec_offset, tvec_RC[1] + tvec_offset, tvec_RC[2] + tvec_offset, rvec_RC[0] + 0.0001, rvec_RC[1] + 0.0001, rvec_RC[2] + 0.0001,
                tvec_GT[0] + tvec_offset, tvec_GT[1] + tvec_offset, tvec_GT[2] + tvec_offset, rvec_GT[0] + 0.0001, rvec_GT[1] + 0.0001, rvec_GT[2] + 0.0001]
        print("initial parameters", init_params)
        init_value = self.reprojection_error_ls(init_params, camera_matrix, dist_coef, points_3d, points_2d, list_T_RG)
        print("initial value", np.sum(init_value**2))
        res = least_squares(self.reprojection_error_ls, # the minimization function
                        init_params, # the initial parameters
                        args=(camera_matrix, dist_coef, points_3d, points_2d, list_T_RG), # arguments for the minimization function
                        method='trf', # optimization method
                        bounds=(lower_bounds, upper_bounds),
                        ) # bounds for the parameters
        if res.success:
            print("Optimization successful.")
            print("Optimal value:", np.sum(res.fun**2))
            print("optimal/initial", np.sum(res.fun**2)/np.sum(init_value**2))
            # save the results to a json file
            tvec_RC = res.x[:3]
            rvec_RC = res.x[3:6]
            tvec_GT = res.x[6:9]
            rvec_GT = res.x[9:]
            print("Optimized parameters:", res.x)
            data = {
                "T_rc": {
                    "translation_vector": tvec_RC.tolist(),
                    "rotation_matrix": rvec_RC.tolist()
                },
                "T_gt": {
                    "translation_vector": tvec_GT.tolist(),
                    "rotation_matrix": rvec_GT.tolist()
                }
            }
            with open(MY_RESULT, "w") as f:
                json.dump(data, f, indent=4)

