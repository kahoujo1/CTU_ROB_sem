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
import json



def main():
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
    camera.start()
    
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

    # Take one image from the camera
    img = camera.grab_image()
    img_undistorted = cv2.undistort(img, camera_matrix, dist_coeff, None, camera_matrix_new)

    if (img is not None) and (img.size > 0):
        estimator.calculate_pose(img)
        T = estimator.base.position
        [rvec,tvec] = [T.rotation.log(), T.translation]
        base = estimator.get_updated_base()
        translations = base.get_target_transforms_from_camera()
        print("rvec",rvec,"tvec",tvec)
        cv2.drawFrameAxes(img, camera_matrix_new, dist_coeff, rvec, tvec, length=20)
        for t in translations:
            [rvec,tvec] = [t.rotation.log(), t.translation]
            cv2.drawFrameAxes(img, camera_matrix_new, dist_coeff, rvec, tvec, length=20)
        cv2.namedWindow('Camera image', cv2.WINDOW_NORMAL)
        cv2.imshow('Camera image', img)
        cv2.waitKey(0)
    else:
        print("The image was not captured.")
    camera.close()

if __name__ == '__main__':
    main()