"""Gets the camera image and detects the Aruco markers on it."""
import cv2
import copy
import sys
import os
import numpy as np


# appending a path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from lib.camera_calibration import CameraCalib

"""
# For checkerboard calibration, the following code can be used:
def main():
    camera_calibration = CameraCalib()
    image_path = os.path.dirname(__file__) + "/../../calibration_images/"
    print(image_path)
    camera_calibration.set_image_path(image_path)
    camera_calibration.calibrate("test.json")
"""
# For charuco calibration, the following code can be used:
def main():
    camera_calibration = CameraCalib()
    image_path = os.path.dirname(__file__) + "/../../charuco_calibration_images/"
    camera_calibration.set_image_path(image_path)
    camera_calibration.calibrate_using_charuco("camera_matrix_4.json")


if __name__ == "__main__":
    main()