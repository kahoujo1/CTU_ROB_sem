"""
Module for storing information about the base of the cubes.
"""

import numpy as np
import cv2
import csv
import sys
import os
from typing import List

# appending path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.se3 import SE3  # SE2/3, SO2/3
from lib.so3 import SO3

BASE_SIZE_X = 240  # [mm]
BASE_SIZE_Y = 200  # [mm]
CUBE_SIZE = 40  # [mm]
ARUCO_SIZE = 36  # [mm]
ARUCO2ARUCO_X = 180  # [mm] offset of the aruco markers in x direction
ARUCO2ARUCO_Y = 140  # [mm] offset of the aruco markers in y direction

Z_OFFSET = 20  # [mm] offset of the center of the part holder in z diretction


class Base:
    """
    Class for storing information about the base of the cubes.

    Attributes:
        - aruco_ids: List[int] - List of ArUco IDs
        - cube_centers: List[List[float]] - List of cube centers (list of [x, y])
        - position: SE3 - Tranformation matrix from base to the camera (T_CB)
        - target_transforms: List[SE3] - List of target transformation matrices from the cube positions to the camera (T_CT)
    """

    def __init__(self):
        self.aruco_ids = []  # List of ArUco IDs
        self.cube_centers = []  # List of cube centers (list of [x, y])
        self.position = (
            None  # TODO: Tranformation matrix from base to the camera (T_CB)
        )
        self.target_transforms = (
            []
        )  # List of target transformation matrices from target to camera (T_CT)

    def csv2base(self, filename: str) -> None:
        """
        Loads base data from a CSV file.

        Args:
            - filename: str - path to the CSV file describing the base
        """
        with open(filename, mode="r") as file:
            csv_reader = csv.reader(file)
            # Read the first line for ArUco IDs
            self.aruco_ids = list(map(int, next(csv_reader)))
            # Read the subsequent lines for cube centers
            self.cube_centers = [list(map(float, row)) for row in csv_reader]

    def set_position(self, position: SE3) -> None:
        """
        Sets the position of the base.

        Args:
            - position: SE3 - Tranformation matrix from base to the camera (T_CB) (calculated using base_pose_estimator)
        """
        self.position = position
        if not isinstance(position, SE3):
            raise ValueError("Position is not a valid SE3 transformation matrix.")
        # calculate
        print("Position", self.position)
        self.target_transforms = []
        for center in self.cube_centers:
            tmp_transform = position * SE3(
                [center[0], center[1], Z_OFFSET], SO3()
            )  # T_CB * T_BT
            # because the flange has z-axis pointing down, we need to rotate the target by 180 degrees around x
            # also rotate around the z-axis for 180 to have the same default orientation
            tmp_transform = (
                tmp_transform
                * SE3(rotation=SO3.rx(np.pi))
                * SE3(rotation=SO3.rz(-np.pi / 2))
            )
            # tranform to the camera frame and append to the list
            self.target_transforms.append(tmp_transform)

    def get_target_transforms_from_camera(self) -> List[SE3]:
        """
        Return the list of target transformation matrices from the cube positions to camera (T_CT).

        Returns:
            - List[SE3] - List of target transformation matrices from cube positions to camera
        """
        if self.position is None:
            raise ValueError("Position is not set.")
        return self.target_transforms

    def get_target_transforms_from_robot(self, T_RC: SE3) -> List[SE3]:
        """
        Return the list of target transformation matrices from cube positions to robot (T_RT).

        Args:
            - T_RC: SE3 - Transformation matrix from camera to robot (T_RC)

        Returns:
            - List[SE3] - List of target transformation matrices from cube positions to robot
        """
        if self.position is None:
            raise ValueError("Position is not set.")
        # for T in self.target_transforms:
        #     T.translation /= 1000
        ans = []
        for T_CT in self.target_transforms:
            T_CT.translation /= 1000
            ans.append(T_RC * T_CT)
            T_CT.translation *= 1000
        return ans
