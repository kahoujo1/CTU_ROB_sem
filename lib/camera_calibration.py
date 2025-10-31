import cv2
import copy
import numpy as np
import json
import os
import sys

# pitomec je to
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import lib.utils

CHESSBOARD_SIZE = (7, 4)
CHESSBOARD_SQUARE_SIZE = 50  # [mm]
ARUCO_MARKERS_DETECTED = 4

CHARUCOBOARD_SIZE = (7, 9)
CHARUCO_SQUARE_LENGTH = 0.030
CHARUCO_MARKER_SIZE = 0.022


def blur_laplacian(image, threshold=250):
    """Check if an image is blurry based on the Laplacian variance.

    Args:
        image (np.ndarray): input image to analyze
        threshold (float, optional): threshold for the Laplacian variance. Defaults to 250.

    Returns:
        tuple[bool, float]: is_blurry (bool, True if the image is blurry), laplacian_var (float, Laplacian variance)
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, laplacian_var


def edge_sharpness(image: np.ndarray, gradient_threshold=300) -> tuple[bool, float]:
    """Analyze edge sharpness based on gradient size, and disqualify blurry images.

    Args:
        image (np.ndarray): input image to analyze
        gradient_threshold (_type_, optional): . Defaults to 300.

    Returns:
        tuple[bool, float]: is_blurry (bool, True if the image is blurry), avg_gradient (float, average gradient)
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 1: Detect edges using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Step 2: Calculate gradient magnitudes
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

    # Step 3: Filter gradients only along edges
    edge_gradients = gradient_magnitude[edges > 0]

    # Step 5: Check sharpness criteria
    avg_gradient = np.mean(edge_gradients)

    is_blurry = avg_gradient < gradient_threshold
    return is_blurry, avg_gradient


class CameraCalib:
    """class for calibrating the camera using a chessboard or a Charuco board."""

    def __init__(self) -> None:
        self.objpoints = []
        self.imgpoints = []
        self.img_path = None

    def set_image_path(self, img_path: str) -> None:
        self.img_path = img_path

    def get_images(self) -> list:
        image_files = []
        extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")
        for root, _, files in os.walk(self.img_path):
            for file in files:
                if file.lower().endswith(extensions):
                    image_files.append(os.path.join(root, file))
        return image_files

    def calibrate(self, json_filename: str) -> None:
        """Calibrate the camera. Save the calibration data to a json file.

        Args:
            json_filename (str): Name of the json file to save the calibration data.
        """
        assert self.img_path is not None, "Image path not set."
        images = self.get_images()
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = (
            np.mgrid[0 : CHESSBOARD_SIZE[0], 0 : CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
            * CHESSBOARD_SQUARE_SIZE
        )
        print("objp:", objp)
        print(images)
        for img_path in images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
            if ret == True:
                self.objpoints.append(objp)
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )  # finds corners with subpixel accuracy
                self.imgpoints.append(corners2)

                # draw and display the corners
                img = cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
                cv2.imshow("img", img)
                cv2.waitKey(10000)
        cv2.destroyAllWindows()
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None
        )
        K, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, gray.shape[::-1], 1, gray.shape[::-1]
        )
        if ret:
            # Create a dictionary to store the calibration data
            calibration_data = {
                "camera_matrix": mtx.tolist(),
                "dist_coeff": dist.tolist(),
                "new_camera_matrix": K.tolist(),
            }

            # Define the path to the JSON file
            calibration_file = os.path.join(os.path.dirname(__file__), json_filename)

            # Write the calibration data to the JSON file
            with open(calibration_file, "w") as f:
                json.dump(calibration_data, f, indent=4)
            print("Camera calibration successful.")
            # print("Camera matrix:\n", mtx)
            # print("Distortion coefficients:\n", dist)
        else:
            print("Camera calibration failed.")

    def calibrate_using_charuco(self, json_filename: str) -> None:
        """Calibrate the camera using a Charuco board. Save the calibration data to a json file.

        Args:
            json_filename (str): Name of the json file to save the calibration data.
        """

        assert self.img_path is not None, "Image path not set."
        images = self.get_images()
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters()
        charuco_board = cv2.aruco.CharucoBoard(
            CHARUCOBOARD_SIZE,
            squareLength=CHARUCO_SQUARE_LENGTH,  # [m]
            markerLength=CHARUCO_MARKER_SIZE,  # [m]
            dictionary=aruco_dict,
        )
        # arrays to store object points and image points from all the images
        all_charuco_corners = []
        all_charuco_ids = []
        gray = None  # Initialize gray to ensure it is defined

        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # go through each image
        for img_path in images:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=parameters
            )  # detect aruco markers
            # cv2.aruco.drawDetectedMarkers(gray,corners,ids)

            # cv2.imshow('gray', gray)
            # cv2.waitKey(5000)
            # additional disqualification criteria

            # Check if the image is blurry
            is_blurry, avg_gradient = edge_sharpness(img)

            """ if is_blurry:
                print(f"Image {img_path} is blurry. Avg gradient: {avg_gradient} Skipping.")
                continue """
            # print(len(ids))
            # Find the corners and ids of the charuco board
            if len(ids) >= ARUCO_MARKERS_DETECTED:
                x, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, charuco_board
                )
                # print(x)
                # print((charuco_corners))
                if charuco_corners is not None and len(charuco_corners) >= 8:
                    all_charuco_corners.append(charuco_corners)
                    all_charuco_ids.append(charuco_ids)
                    # Draw and display the corners
                    img = cv2.aruco.drawDetectedCornersCharuco(
                        img, charuco_corners, charuco_ids
                    )
                    # cv2.imshow('img', img)
                    # cv2.waitKey(5000)
        cv2.destroyAllWindows()

        # Calibrate the camera using the detected corners
        print(
            "Count charuco corners detected:",
            len(all_charuco_corners),
            "Count charuco ids detected:",
            len(all_charuco_ids),
        )
        ret, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=charuco_board,
            imageSize=gray.shape[::-1],
            cameraMatrix=None,
            distCoeffs=None,
            # flags=cv2.CALIB_RATIONAL_MODEL # added for bigger distortion model
        )

        K, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, gray.shape[::-1], 1, gray.shape[::-1]
        )
        if ret:
            # Create a dictionary to store the calibration data
            calibration_data = {
                "camera_matrix": mtx.tolist(),
                "dist_coeff": dist.tolist(),
                "new_camera_matrix": K.tolist(),
            }

            # Define the path to the JSON file
            calibration_file = os.path.join(os.path.dirname(__file__), json_filename)

            # Write the calibration data to the JSON file
            with open(calibration_file, "w") as f:
                json.dump(calibration_data, f, indent=4)
            print("Camera calibration successful.")
            print("RMS error:", ret)
            # print("Camera matrix:\n", mtx)
            # print("Distortion coefficients:\n", dist)
        else:
            print("Camera calibration failed.")
