from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
import os
import json
from typing import List, Tuple
from lib.so3 import SO3
from lib.se3 import SE3


# utils for conversions
def vector_to_se3(tvec, rvec) -> SE3:
    """
    Converts translation and rotation (Euler) vectors to SE3 transformation class

    Args:
        tvec: ArrayLike - translation vector
        rvec: ArrayLike - rotation vector

    Returns:
        SE3: SE3 object
    """
    print("tvec", tvec, "rvec", rvec)
    ret = SE3()
    ret.rotation = SO3.exp(rvec)
    ret.translation = tvec
    return ret


def robot_gripper_to_SE3(robot) -> SE3:
    """
    Gets transformation from gripper to robot using robot joint angles

    Args:
        robot: Robot - robot object

    Returns:
        SE3: SE3 object representing transformation from gripper to robot
    """
    fk = robot.get_gripper_pos()
    return SE3(fk[:3, 3], SO3(fk[:3, :3]))


def get_R(roll, pitch, yaw) -> np.ndarray:
    """
    Returns the rotation matrix from roll, pitch, yaw angles

    Args:
        roll: float - roll angle
        pitch: float - pitch angle
        yaw: float - yaw angle

    Returns:
        np.ndarray: rotation matrix
    """
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    roll = np.radians(roll)

    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )

    R_y = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )

    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )

    return R_z @ R_y @ R_x


def load_camera_calibration_data(
    filename: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads camera calibration data from a file

    Args:
        filename: str - filename of the calibration data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: camera matrix, distortion coefficients, new camera matrix
    """
    with open(filename, "r") as f:
        calibration_data = json.load(f)

    camera_matrix = np.array(calibration_data["camera_matrix"])
    dist_coeff = np.array(calibration_data["dist_coeff"])
    camera_matrix_new = np.array(calibration_data["new_camera_matrix"])

    return camera_matrix, dist_coeff, camera_matrix_new


def get_images_from_dir(img_path: str) -> list:
    """Loads all images from a directory.

    Args:
        img_path (str): The path to the directory containing the images.

    Returns:
        list: A list of paths to the images.
    """
    image_files = []
    extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
    for root, _, files in os.walk(img_path):
        for file in files:
            if file.lower().endswith(extensions):
                image_files.append(os.path.join(root, file))
    return image_files


def write_data_to_json(file_path, **data):
    """
    Writes multiple lists or dictionaries to a JSON file.

    Args:
        file_path (str): The path to the JSON file.
        **data: Arbitrary keyword arguments representing the data to be saved.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def read_data_from_json(file_path):
    """
    Reads multiple lists or dictionaries from a JSON file.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: A dictionary containing the data.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
