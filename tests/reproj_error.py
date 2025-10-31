"""
Test to check the reprojection error of the camera model.
"""

import numpy as np
import cv2
import sys
import os
import json
import time

# appending a path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.hand_eye_calib import HandEyeCalib

CAMERA_PATH = 'lib/camera_matrix_4.json'
HANDEYE_CALIB_DATA_PATH = 'src/calibration/handeye_cal_data.json'
HANDEYE_CALIB_RESULTS_PATH = "src/calibration/handeye_cal_results.json"

def main():
    calib = HandEyeCalib()
    calib.reprojection_optimization(CAMERA_PATH, HANDEYE_CALIB_DATA_PATH, HANDEYE_CALIB_RESULTS_PATH)

if __name__ == "__main__":
    main()