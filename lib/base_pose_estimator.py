"""
Module for estimating the pose of the base of the cubes in relation to the camera.
"""

import numpy as np
import cv2
import sys
import os

# appending a path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from lib.base import (
    Base,
    BASE_SIZE_X,
    BASE_SIZE_Y,
    CUBE_SIZE,
    ARUCO_SIZE,
    ARUCO2ARUCO_X,
    ARUCO2ARUCO_Y,
)
from lib.se3 import SE3  # SE2/3, SO2/3
from lib.so3 import SO3
NUM_OF_IMAGES = 15


class BasePoseEstimator:
    """
    This class serves for estimating the pose of the base of the cubes in relation to the camera.

    Attributes:
        - camera_matrix: np.ndarray - camera matrix
        - distortion_coefficients: np.ndarray - distortion coefficients
        - aruco_size: float - size of the aruco markers
        - base: Base - base object for which the pose is estimated
    """

    def __init__(
        self,
        camera_matrix: np.ndarray = None,
        distortion_coefficients: np.ndarray = None,
    ) -> None:
        """
        Constructor of the BasePoseEstimator class.

        Args:
            - camera_matrix: np.ndarray - camera matrix
            - distortion_coefficients: np.ndarray - distortion coefficients
        """

        self.camera_matrix = camera_matrix
        self.distortion_coefficients = distortion_coefficients
        self.aruco_size = 0.36  # size of the aruco markers on bases
        self.base = None

    def set_base(self, base: Base) -> None:
        """
        Sets the base object.

        Args:
            - base: Base - base object
        """
        self.base = base

    def set_camera_matrix(self, camera_matrix: np.ndarray) -> None:
        """
        Sets the camera matrix.

        Args:
            - camera_matrix: np.ndarray - camera matrix
        """
        self.camera_matrix = camera_matrix

    def set_distortion_coefficients(self, distortion_coefficients: np.ndarray) -> None:
        """
        Sets the distortion coefficients.

        Args:
            - distortion_coefficients: np.ndarray - distortion coefficients
        """
        self.distortion_coefficients = distortion_coefficients

    def calculate_pose_normal(self, img: np.ndarray) -> bool:
        """
        Calculates the pose of the base in relation to the camera using perspective-n-point algorithm.

        Args:
            - img: np.ndarray - image from the camera

        Returns:
            - bool - True if the pose was calculated, False if the algorithm failed
        """
        assert self.base is not None, "Base object not set."
        assert self.camera_matrix is not None, "Camera matrix not set."
        assert (
            self.distortion_coefficients is not None
        ), "Distortion coefficients not set."

        # get the aruco markers from image
        aruco = cv2.aruco
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        # detect markers
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )
        # print("My IDS",self.base.aruco_ids)
        # print("IDS",ids)
        if not ((self.base.aruco_ids[0] in ids) and (self.base.aruco_ids[1] in ids)):
            raise Exception("Wanted ArUco markers not detected.")
        # get points on the base plane
        base_points = np.array(
            [
                # bottom left aruco marker
                [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top left corner
                [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top right corner
                [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom right corner
                [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom left corner
            ]
        )
        # add top right aruco marker
        base_points = np.concatenate((base_points, base_points), axis=0)
        base_points[4:] += np.array([ARUCO2ARUCO_X, ARUCO2ARUCO_Y, 0])
        world_points = np.zeros((8, 2))
        # print("corners:",corners, "ids:",ids)
        # print("base points:",base_points, "aruco_ids", self.base.aruco_ids)
        for i in range(len(ids)):  # assign the corners to the correct aruco markers
            if ids[i] == self.base.aruco_ids[0]:
                world_points[:4] = corners[i]
            elif ids[i] == self.base.aruco_ids[1]:
                world_points[4:] = corners[i]
        # print("world points:",world_points)
        # calculate the pose
        ret, rvec, tvec = cv2.solvePnP(
            base_points,
            world_points,
            self.camera_matrix,
            self.distortion_coefficients,
            False,
            cv2.SOLVEPNP_IPPE_SQUARE,
        )
        # print("rvec:",rvec, "tvec:",tvec, "ret",ret)
        if ret:
            self.base.set_position(SE3(tvec.flatten(), SO3.exp(rvec.flatten())))
            return True
        return False

    def calculate_pose(self, camera) -> bool:
        images = []
        for i in range(NUM_OF_IMAGES):
            img = camera.grab_image()

            if img is None or img.size == 0:
                print("No image captured")
                continue
            
            images.append(img)
        return self.calculate_pose_normal(images[0])
        #return self.calculate_pose_multiple_photos(images)
    
    def calculate_pose_multiple_photos(self, images: list) -> bool:
        """
        Calculates the pose of the base in relation to the camera using perspective-n-point algorithm.

        Args:
            - img: list - list of images from the camera

        Returns:
            - bool - True if the pose was calculated, False if the algorithm failed
        """
        assert self.base is not None, "Base object not set."
        assert self.camera_matrix is not None, "Camera matrix not set."
        assert (
            self.distortion_coefficients is not None
        ), "Distortion coefficients not set."
        
        ret = True
        rvecs = []
        tvecs = []

        # get the aruco markers from image
        aruco = cv2.aruco
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        parameters = aruco.DetectorParameters()
        # detect markers
        for img in images:
            if not ret:
                return False
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters
            )
            # print("My IDS",self.base.aruco_ids)
            # print("IDS",ids)
            if not ((self.base.aruco_ids[0] in ids) and (self.base.aruco_ids[1] in ids)):
                raise Exception("Wanted ArUco markers not detected.")
            # get points on the base plane
            base_points = np.array(
                [
                    # bottom left aruco marker
                    [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top left corner
                    [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top right corner
                    [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom right corner
                    [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom left corner
                ]
            )
            # add top right aruco marker
            base_points = np.concatenate((base_points, base_points), axis=0)
            base_points[4:] += np.array([ARUCO2ARUCO_X, ARUCO2ARUCO_Y, 0])
            world_points = np.zeros((8, 2))
            # print("corners:",corners, "ids:",ids)
            # print("base points:",base_points, "aruco_ids", self.base.aruco_ids)
            for i in range(len(ids)):  # assign the corners to the correct aruco markers
                if ids[i] == self.base.aruco_ids[0]:
                    world_points[:4] = corners[i]
                elif ids[i] == self.base.aruco_ids[1]:
                    world_points[4:] = corners[i]
            # print("world points:",world_points)
            # calculate the pose
            ret, rvec, tvec = cv2.solvePnP(
                base_points,
                world_points,
                self.camera_matrix,
                self.distortion_coefficients,
                False,
                cv2.SOLVEPNP_IPPE_SQUARE,
            )

            # take average of 5 photos mean
            rvecs.append(rvec)
            tvecs.append(tvec)


        # Calculate the mean of the vectors
        final_rvec = np.median(np.array(rvecs).reshape(-1, 3), axis=0)
        final_tvec = np.median(np.array(tvecs).reshape(-1, 3), axis=0)

        if ret:
            self.base.set_position(SE3(final_tvec.flatten(), SO3.exp(final_rvec.flatten())))
            self.set_rotation(self.base, final_tvec, final_rvec)
            return True
        return False
    
    def set_rotation(self, base, tvec, rvec):
        T_RC = SE3(translation = [
                0.4709404392331024,
                -0.001012245901015851,
                1.1588433705611427
            ], rotation = SO3.exp([
                2.2453229541584725,
                2.189286312669311,
                -0.04055677484668194
            ]))
        # calculate the normal vector of the base plane
        tansf = base.get_target_transforms_from_robot(T_RC)
        # get the normal vector of the plane
        normal_base = np.cross(tansf[0].translation - tvec, tansf[2].translation - tvec)
        # normal vector of the robot plane
        nvec_robot = np.array([0, 0, 1])
        # calculate the rotation axis and the angle
        rot_axis = np.cross(normal_base, nvec_robot)    
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        angle = np.arccos(np.dot(normal_base, nvec_robot) / (np.linalg.norm(normal_base) * np.linalg.norm(nvec_robot)))
        # clip angle to either 0, +-7 or +-15 degrees
        valid_angles = np.deg2rad(np.array([0, 7, -7, 15, -15, 173, -173, 165, -165, 179.5, -179.5]))
        closest_angle = valid_angles[np.argmin(np.abs(valid_angles - angle))]
        # create the rotation matrix
        rot_axis = closest_angle * rot_axis
        print("Rotation axis:", rot_axis)
        print("Angle:", closest_angle,"rad",np.rad2deg(closest_angle))
        print("prior angle:", angle, "rad", np.rad2deg(angle))
        print("rvec,", rvec, "rot_axis,",rot_axis)
        #base.set_position(SE3(tvec.flatten(), SO3.exp(rot_axis.flatten())))

    def get_updated_base(self) -> Base:
        """
        Returns the base object with the updated pose. Can be called after calculate_pose.

        Returns:
            - Base - base object with the updated pose.
        """
        if self.base.position is None:
            raise Exception("Base position not calculated.")
        return self.base
