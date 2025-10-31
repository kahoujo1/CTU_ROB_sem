import cv2
import copy
import sys
import os
import json
import numpy as np

CALIBRATION_IMAGES_DIR = "charuco_calibration_images"

# appending a path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
# Our Basler camera interface
from lib.basler_camera import BaslerCamera

def initialise_camera() -> BaslerCamera:
    """initialises the camera and returns the camera object

    Returns:
        BaslerCamera: camera object
    """
    camera: BaslerCamera = BaslerCamera()
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
    return camera


def camera_take_image(camera: BaslerCamera, dirname: str):
    """Takes images from the camera and saves them in the directory given, until the user presses 'x' to exit.
    Numbers the images sequentially, never overwriting an existing image. Only accepts detected chessboard images.

    Args:
        camera (BaslerCamera): camera object
        dirname (str): directory to save the images
    """
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()

    key = input("Press 'x' to exit or any other key to take a picture: ")
    idx = 1
    saved = False
    while key != "x":
        img = camera.grab_image()
        if (img is not None) and (img.size > 0):  # img acquired

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            corners, ids, rejected = aruco.detectMarkers(
                img, aruco_dict, parameters=parameters
            )

            if ids is not None:  # if corners are found, save the image in the directory
                while not saved:
                    img_name = str("calibration_image" + str(idx) + ".jpg")
                    img_filename = os.path.join(dirname, img_name)
                    if not os.path.exists(dirname):
                        print("Directory does not exist, creating directory: ", dirname)
                        os.makedirs(dirname)
                    if os.path.exists(img_filename):
                        idx += 1
                        continue
                    else:
                        cv2.imwrite(img_filename, img)
                        print("Image saved to: ", img_filename)
                        saved = True
            else:
                print("Not one Aruco marker found, try again.")
        else:
            print("The image was not captured, try again.")
        key = input("Press 'X' to exit or any other key to take a picture: ")
        if saved:
            idx += 1
            saved = False


def main():
    camera = initialise_camera()
    camera_take_image(camera, CALIBRATION_IMAGES_DIR)
    camera.close()


if __name__ == "__main__":
    main()