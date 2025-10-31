import cv2
import copy
import sys
import os
import json
import numpy as np
import time
from typing import List

CALIBRATION_IMAGES_DIR = "handeye_cal_img"
ROBOT_POSITIONS = "robot_positions.json"  # expected format is list of [x,y,z] positions in robot "coordinates"
ROBOT_HAND_ARUCO_SIZE = (
    44.6  # size of the aruco marker in mm THIS WAS MEASURED BY CALIPERS!
)
CALIBRATION_DATA_DIR = (
    "handeye_cal_data.json"  # filename for the hand eye calibration json file
)
CALIBRATION_RESULTS_DIR = "handeye_cal_results.json"  # filename for the hand eye calibration results json file
CAMERA_MATRIX_FILE = "../../lib/camera_matrix_4.json"
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from lib.robot_wrapper import Robot
from lib.se3 import SE3
from lib.so3 import SO3
from lib.basler_camera import BaslerCamera
from lib import utils

MIN_X = 0.10  # [m]
MAX_X = 0.53  # [m]
DELTA_X = 0.05  # [m]

MIN_Z = 0.05  # [m] was 0.15
MAX_Z = 0.16  # [m] was 0.25
DELTA_Z = 0.05  # [m]

MIN_Y = -0.40  # [q in rad]
MAX_Y = 0.41  # [q in rad]
DELTA_Y = 0.1  # [q in rad]

Q5_MIN = 0
Q5_MAX = np.pi / 12 + 0.01
Q5_DELTA = np.pi / 12

Q6_MIN = -np.pi / 2
Q6_MAX = np.pi / 2 + 0.01
Q6_DELTA = np.pi / 8


def initialise_camera() -> BaslerCamera:
    """initialises the camera and returns the camera object

    Returns:
        BaslerCamera: camera object
    
    """
    camera: BaslerCamera = BaslerCamera()
    camera.connect_by_name("camera-crs97")
    # Open the communication with the camera
    camera.open()
    # Set capturing parameters from the camera object.
    # The default parameters (set by constructor) are OK.
    camera.set_parameters()
    # Starts capturing images by camera.

    return camera


def generate_robot_positions(robot: Robot):
    """
    Generate a list of robot positions (in)

    Args:
        robot (Robot): robot object

    Returns:   
        list: list of robot positions
    """
    robot_positions = []
    for x in np.arange(MIN_X, MAX_X, DELTA_X):
        for z in np.arange(MIN_Z, MAX_Z, DELTA_Z):
            # calculate the inverse kinematics for the robot
            tvec = np.array([x, 0, z])
            rvec = np.array([0, 160, 0])
            robot_pos = utils.vector_to_se3(tvec, rvec)
            ik = robot.return_best_ik(robot_pos.homogeneous())
            if ik is not None:
                for y in np.arange(MIN_Y, MAX_Y, DELTA_Y):  # go through y values
                    for q5 in np.arange(Q5_MIN, Q5_MAX, Q5_DELTA):  # second last joint
                        for q6 in np.arange(Q6_MIN, Q6_MAX, Q6_DELTA): # last joint
                            pos = ik + [y, 0, 0, 0, q5, q6]
                            if robot.in_limits(pos):
                                robot_positions.append(pos)
    print("Number of robot positions: ", len(robot_positions))
    return robot_positions


def generate_robot_positions_new(robot: Robot) -> List:
    """ Generate a list of robot positions

    Args:
        robot (Robot): robot object

    Returns:
        List: list of robot positions
    """
    robot_positions = []
    minx = 0.1
    maxx = 0.5
    miny = -0.3
    maxy = 0.3
    minz = 0.05
    maxz = 0.15
    default_rvec = np.array([np.pi, 0, 0])
    num_positions = 500
    for _ in range(num_positions):
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        z = np.random.uniform(minz, maxz)
        q5 = np.random.uniform(Q5_MIN, Q5_MAX)
        q6 = np.random.uniform(Q6_MIN, Q6_MAX)
        
        tvec = np.array([x, y, z])
        rvec_diff = np.random.uniform(-np.pi/4, np.pi/4, 3)
        rvec = default_rvec + rvec_diff
        robot_pos = utils.vector_to_se3(tvec, rvec)
        ik = robot.return_best_ik(robot_pos.homogeneous())
        
        if ik is not None:
            pos = ik + [0, 0, 0, 0, q5, q6]
            if robot.in_limits(pos):
                robot_positions.append(pos)
    
    print("Number of robot positions: ", len(robot_positions))
    return robot_positions


def hand_eye_take_image(dirname:str) -> List:   
    """
    Cycles through robot positions and takes images at each position.
    Format of robot positions is a list of robot SE3s.

    Args:
        dirname (str): directory to save the images

    Returns:
        list: list of SE3s of the robot
    """
    camera = initialise_camera()
    robot = Robot(True)
    robot.gripper_drop()
    robot_positions = generate_robot_positions(robot)
    # initialise aruco dict for checking valid pictures
    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters = aruco.DetectorParameters()

    hand_positions = []  # list of SE3s of the hand aruco marker
    hand_aruco_corners = []  # list of aruco corners of each position
    valid_robot_positions = []  # list of SE3s of the robot

    for i in range(len(robot_positions)):
        print("Moving to position: ", robot_positions[i])
        robot.move_to_q(robot_positions[i])
        robot.wait_for_motion_stop()
        time.sleep(.2)  # make sure the robot is stable before taking the picture

        img = camera.grab_image()

        if img.size == 0 or img is None:
            print("Failed capture img")
            continue

        # detect markers - check if the image is usable
        corners, ids, rejected = aruco.detectMarkers(
            img, aruco_dict, parameters=parameters
        )
        print("IDS found:", ids, "CORNERS found:", corners)

        # if corners are found, save the image in the directory
        if ids is not None:
            # save the image
            img_name = str("calibration_image" + str(i) + ".jpg")
            img_filename = os.path.join(dirname, img_name)
            if not os.path.exists(dirname):  # create directory if it does not exist
                print("Directory does not exist, creating directory: ", dirname)
                os.makedirs(dirname)
            cv2.imwrite(img_filename, img)
            print("Image saved to: ", img_filename)

            # find the hand pose vector
            hand_pose = find_aruco_id_vec(corners, ids, ROBOT_HAND_ARUCO_SIZE)
            print("Hand pose: ", hand_pose)

            # save all the data
            hand_positions.append(hand_pose)
            valid_robot_positions.append(utils.robot_gripper_to_SE3(robot))
            hand_aruco_corners.append(corners[0].tolist())
        else:
            print(
                "Aruco marker of hand NOT found in position"
                + str(i)
                + "cycling to next position"
            )
            continue
    print("Finished taking images")
    camera.close()
    robot.close()

    # Convert SE3 objects to dictionaries
    valid_robot_positions_dicts = [
        calid_robot_pos.to_dict() for calid_robot_pos in valid_robot_positions
    ]
    hand_positions_dicts = [
        hand_position.to_dict() for hand_position in hand_positions
    ]

    # Save the dictionary to a JSON file
    calibration_file = os.path.join(os.path.dirname(__file__), CALIBRATION_DATA_DIR)
    utils.write_data_to_json(
        calibration_file,
        hand_positions=hand_positions_dicts,
        valid_robot_positions=valid_robot_positions_dicts,
        hand_aruco_corners=hand_aruco_corners,
    )
    print("Calibration data saved to: ", calibration_file)
    return valid_robot_positions, hand_positions


def find_aruco_id_vec(corners, ids, ARUCO_SIZE: int) -> SE3:
    """
    Find the vectors of the wanted aruco id in the image.

    Args:
        corners (list): list of aruco corners
        ids (list): list of aruco ids
        ARUCO_SIZE (int): size of the aruco marker

    Returns:
        SE3: SE3 object of the aruco marker
    """
    calibration_file_path = os.path.join(
        os.path.dirname(__file__), CAMERA_MATRIX_FILE
    )
    with open(calibration_file_path, "r") as f:
        calibration_data = json.load(f)

    camera_matrix = np.array(calibration_data["camera_matrix"])
    dist_coeff = np.array(calibration_data["dist_coeff"])
    camera_matrix_new = np.array(calibration_data["new_camera_matrix"])

    base_points = np.array(
        [
            # bottom left aruco marker
            [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top left corner
            [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],  # top right corner
            [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom right corner
            [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],  # bottom left corner
        ]
    )
    world_points = corners[0]
    ret, rvec, tvec = cv2.solvePnP(
        base_points,
        world_points,
        camera_matrix,
        dist_coeff,
        False,
        cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if ret:
        rvec = np.reshape(rvec, (3,))
        tvec = np.reshape(tvec, (3,))/1000
        return utils.vector_to_se3(tvec, rvec)
    else:
        print("Error in finding vectors for ARUCO")


def solve_AX_YB(a, b):
    """Solve A^iX=YB^i, return X, Y

    Args:
        a: list of SE3 objects
        b: list of SE3 objects

    Returns:
        X, Y: SE3 objects
    """
    rvec_a = [T.rotation.log() for T in a]
    tvec_a = [T.translation for T in a]
    rvec_b = [T.rotation.log() for T in b]
    tvec_b = [T.translation for T in b]

    Rx, tx, Ry, ty = cv2.calibrateRobotWorldHandEye(rvec_a, tvec_a, rvec_b, tvec_b)
    # return T_gripper_target, T_robot_camera
    return SE3(tx[:, 0], SO3(Rx)), SE3(ty[:, 0], SO3(Ry))


def main():
    hand_pos, eye_pos = hand_eye_take_image(CALIBRATION_IMAGES_DIR)

    T_gt, T_rc = solve_AX_YB(hand_pos, eye_pos)
    print("gripper2target", T_gt.rotation.log(), T_gt.translation)
    print("robot2camera", T_rc.rotation.log(), T_rc.translation)

    data = {"T_gt": T_gt.to_dict(), "T_rc": T_rc.to_dict()}
    with open(
        os.path.join(os.path.dirname(__file__), CALIBRATION_RESULTS_DIR), "w"
    ) as f:
        json.dump(data, f, indent=4)
    print("Calibration successful, results saved to: ", CALIBRATION_RESULTS_DIR)


if __name__ == "__main__":
    main()
