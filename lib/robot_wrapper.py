import numpy as np 
import sys 
import os
import time
from ctu_crs import CRS97
# appending a path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from lib.se3 import SE3
from lib.so3 import SO3
from lib.base import Base

DEFAULT_GRIPPER_OFFSET_MATRIX = SE3([-0.0, 0., -0.035])
# old transform SE3([-0.025, -0.013, -0.01]) # !!!!! Warning it is in the robot coordinates, multiply from the left
OPEN_GRIPEPR_VALUE = 0.45 # the relative position which is beeing fed into the gripper
CLOSED_GRIPPER_VALUE = 0.8
DEFAULT_POS = np.deg2rad([0, -30, -90, 0, -45, -10])
ADAPTIVE_OFFSET = True

class Robot(CRS97):
    def __init__(self, robotHome:bool=True):
        """Class that wraps the CRS97 robot. It is a child class of the CRS97 robot.
        Has updates of the functions to be more user friendly, as control of robot limits and 
        return of the best ik solution.

        Args:
            robotHome (bool, optional): It sets the home variable, true is for initialization rutine
                false is to skip initialization rutine. Defaults to False.
        """
        super().__init__()
        self.initialize(home = robotHome)
        self.name = "Milasek"
        self.cam_to_rob = SE3()
        self.gripper_offset_matrix = DEFAULT_GRIPPER_OFFSET_MATRIX

    def setup(self, gripper_offset_matrix: SE3 | None = None, cam_to_rob_matrix: SE3 | None = None,):
        """sets up transformation matrix from camera to robot base. also sets up the 
        offset matrix

        Args:
            cam_to_rob_matrix (SE3): transformation matrix from camera to robot base.
            grippper_offset_matrix (SE3 | None, optional): offset matrix to custom gripper. Defaults to None.
        """
        self.cam_to_rob = cam_to_rob_matrix
        if gripper_offset_matrix is None:
            self.gripper_offset_matrix = DEFAULT_GRIPPER_OFFSET_MATRIX
        else:
            self.gripper_offset_matrix = gripper_offset_matrix

    def get_name(self) -> str:
        """returns name of the robot.

        Returns:
            str: name of the robot
        """
        return self.name

    def move_to_gate(self) -> bool:
        """This function moves the robot to the gate position, which makes 
        easier access to the gripper for making changes. 

        Returns:
            bool: True if the robot moved to the gate position, False otherwise.
        """
        self.soft_home()
        q = self.get_q()
        self.move_to_q(q + [-np.pi/4, -np.pi/6, 0, 0, 0, 0])
        self.wait_for_motion_stop()
        self.gripper.control_position_relative(0)
        print(f"Robot {self.name} moved to gate position {q}")
        return True
    
    def move_to_camera_capture_pos(self) -> bool:
        """
        This function moves the robot to a position in which shadows are minimized in the camera view.

        Returns:
            bool: True if the robot moved to the gate position, False otherwise.
        """
        self.soft_home()
        q = self.get_q()
        self.move_to_q(np.deg2rad([-90, 0, -90, 0, -45, 0]))
        self.wait_for_motion_stop()
        self.gripper.control_position_relative(0)
        print(f"Robot {self.name} moved to outside of camera view.")
        return True
    
    def move_under_camera(self) -> bool:
        """Moves the robot under a camera.

        Returns:
            bool: True if the robot moved under the camera, False otherwise.
        """
        self.soft_home()
        q = self.get_q()
        self.move_to_q(q + [0, -np.pi/4, 0, 0, 0, 0])
        self.wait_for_motion_stop()
        print(f"Robot {self.name} moved under camera position {q}")
        return True
    
    def get_flange_pos(self) -> np.ndarray:
        """This function returns the position of the flange (not the gripper) in the robot base coordinates.

        Returns:
            np.ndarray: (x, y, z) coordinates of the flange.
        """
        return self.fk_flange_pos(self.get_q())
    
    def get_gripper_pos(self) -> np.ndarray:
        """This function returns the T transformation matrix of the gripper.

        Returns:
            np.ndarray: T matrix of the flange.
        """
        return self.fk(self.get_q())
    
    def move_to_coordinates(self, pos: np.ndarray) -> bool:
        """This function moves the robot to the given position.

        Args:
            pos (np.ndarray): Transformation matrix of the gripper. T matrix.

        Returns:
            bool: True if the robot moved to the position, False otherwise.
        """
        ik_solutions = self.ik(pos)
        if len(ik_solutions) <= 0:
            print("No solutions found")
            return False
        # pick solutions in limits
        solutions_in_limits = []
        for sol in ik_solutions:
            if self.in_limits(sol):
                solutions_in_limits.append(sol)
        
        if len(solutions_in_limits) > 0:
            clossest_solution = min(solutions_in_limits, key=lambda q:np.linalg.norm(q - self.get_q()))
            self.move_to_q(clossest_solution)
            self.wait_for_motion_stop()
        else:
            print("Outside of joint limits")
            return False

        return True

    def wiggle_move_to_coords(self, pos: np.ndarray) -> bool:
        """This function moves the robot to the given position while wiggling.

        Args:
            pos (np.ndarray): Transformation matrix of the gripper. T matrix.

        Returns:
            bool: True if the robot moved to the position, False otherwise.
        """
        ik_solutions = self.ik(pos)
        if len(ik_solutions) <= 0:
            print("No solutions found")
            return False
        # pick solutions in limits
        solutions_in_limits = []
        for sol in ik_solutions:
            if self.in_limits(sol):
                solutions_in_limits.append(sol)
        
        if len(solutions_in_limits) > 0:
            clossest_solution = min(solutions_in_limits, key=lambda q:np.linalg.norm(q - self.get_q()))
            print(np.linalg.norm(clossest_solution - self.get_q()))
            if np.linalg.norm(clossest_solution - self.get_q()) > 0.2:
                return False
            self.move_to_q(clossest_solution)
            self.wait_for_motion_stop()
            #print(f"Robot {self.name} moved to coordinates {pos}")
        else:
            print("Outside of joint limits")
            return False

        return True
        
    
    def return_best_ik(self, pos: np.ndarray) -> np.ndarray:
        """This function returns the best ik solution for the robot to reach the given position.

        Args:
            pos (np.ndarray): Transformation matrix of the gripper.

        Returns:
            np.ndarray: Best ik solution for the robot.
        """
        ik_solutions = self.ik(pos)
        if len(ik_solutions) <= 0:
            print("No solutions found")
            return None
        # pick solutions in limits
        solutions_in_limits = []
        for sol in ik_solutions:
            if self.in_limits(sol):
                solutions_in_limits.append(sol)
        if len(solutions_in_limits) > 0:
            clossest_solution = min(solutions_in_limits, key=lambda q:np.linalg.norm(q - self.get_q()))
            return clossest_solution
        else:
            print("Outside of joint limits")
            return None

    def move_to_angles(self, q:np.ndarray) -> bool:
        """moves the robot to the given position based on a given joint angles.

        Args:
            q (np.ndarray): joint angles of the robot

        Returns:
            bool: True if the robot moved to the position, False otherwise.
        """
        
        if self.in_limits(q):
            self.move_to_q(q)
            self.wait_for_motion_stop()
        else: 
            print("Outside of limits")
            return False

        return True

    def get_desired_pos_of_custom_gripper(self, pos: SE3)-> SE3:
        """ This function returns the desired position of the robot in the robot base coordinates, 
            for the custom gripper.

        Args:
            pos (SE3): position of the robot in the camera coordinates.

        Returns:
            SE3: Position of the custom gripper in the robot base coordinates, or None if the position is incorrect.
        """
        if pos is not None:
            pos_r = pos*self.gripper_offset_matrix
            return pos_r

        else: 
            return None

    
    def move_custom_gripper(self, pos: SE3):
        """ This function moves the robot to the given position in robot coordinates.
            moves the custom gripper to a given position.

        Args:
            pos (SE3): position to which the custom gripper will be moved

        Returns:
            bool: True if the robot moved to the position, False otherwise.
        """
        ret = 0
        #print("poss without offset: ", pos.homogeneous())
        pos_r = self.get_desired_pos_of_custom_gripper(pos)
        #print(" pos after offset: ", pos_r.homogeneous())
        if ADAPTIVE_OFFSET:
            pos_r = self.position_with_adaptive_offset(pos_r.homogeneous())
            ret = self.move_to_coordinates(pos_r)
        else:
            ret = self.move_to_coordinates(pos_r.homogeneous())
        return ret
    
    def position_with_adaptive_offset(self, pos: np.ndarray) -> np.ndarray:
        """Transforms the position using adaptive offset
        
        Args:
            pos (np.ndarray): Position to be transformed

        Returns:
            np.ndarray: Transformed position
        """
        # parameters for the offset
        # x limits
        x_min = 0.35
        x_max = 0.55
        # x values
        x_min_value = 0.002
        x_max_value = 0.005
        x_bias = 0.002
        # y limits
        y_min = -0.25
        y_max = 0.20
        # y values
        y_min_value = 0.007
        y_max_value = -0.007
        y_bias = 0.002
        tvec = pos[:3, 3]
        x = tvec[0]
        y = tvec[1]
        # calculate the offsets
        x_offset = (x_max_value - x_min_value) * (x - x_min)/(x_max - x_min) + x_min_value + x_bias
        if y>0:
            x_offset = 0
        y_offset = (y_max_value - y_min_value) * (y - y_min)/(y_max - y_min) + y_min_value + y_bias
        z_offset = 0
        if y < 0:
            z_offset = 0#-0.005
        # apply the offsets
        pos[0,3] += x_offset
        pos[1,3] += y_offset
        pos[2,3] += z_offset
        return pos
        
        
    def gripper_pick(self):
        """
        Opens the gripper. 
        """
        self.gripper.control_position_relative(OPEN_GRIPEPR_VALUE)
        time.sleep(1)

    def gripper_drop(self):
        """
        Closes the gripper using the learned strength.
        """
        self.gripper.control_position_relative(CLOSED_GRIPPER_VALUE)
        time.sleep(1)
        print(f"Gripper of robot {self.name} closed")

    def pick_drop_part(self, base: Base, index_of_target: int, pick_part: bool):
        """This function picks or drops a part from a given base.

        Args:
            base (Base): Base object from which the part will be picked or dropped.
            index_of_target (int): Index of the target part.
            pick_part (bool): True if the part will be picked, False if the part will be dropped.
        """
        #assert self.cam_to_rob.translation != [0,0,0], "Camera to robot transformation is not set"
        # get the target transform
        T_RT = base.get_target_transforms_from_robot(self.cam_to_rob)[index_of_target]
        # move to target
        self.move_to_angles(DEFAULT_POS)
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.05]))
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.01]))
        self.move_custom_gripper(T_RT)
        
        if pick_part:
            self.gripper_pick()
        else:
            self.gripper_drop()
        # move back
        self.move_custom_gripper(T_RT)
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.01]))
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.05]))
        # move to default position
        self.move_to_angles(DEFAULT_POS)

    def wiggle_wiggle_wiggle(self):
        """
        This function moves the robot in a wiggle pattern.

        Serves as a failsafe when the coordinates of the part are not precise enough. This wiggle position should make the part fall into the holder.
        """
        wiggle_x = 0.007
        wiggle_y = 0.005
        fk = self.get_gripper_pos()
        for i in [-1,1]:
            self.wiggle_move_to_coords(fk @ SE3([i*wiggle_x,0, 0]).homogeneous())
            self.wiggle_move_to_coords(fk)
            self.wiggle_move_to_coords(fk @ SE3([0, i*wiggle_y, 0]).homogeneous())
            self.wiggle_move_to_coords(fk)


    def pick_drop_part_with_wiggle(self, base: Base, index_of_target: int, pick_part: bool):
        """
        This function picks or drops a part from a given base and wiggles to make sure the part falls into the holder.

        Args:
            base (Base): Base object from which the part will be picked or dropped.
            index_of_target (int): Index of the target part.
            pick_part (bool): True if the part will be picked, False if the part will be dropped.
        """
        # get the target transform
        if pick_part:
            self.gripper_drop()
        else:
            self.gripper_pick()
        T_RT = base.get_target_transforms_from_robot(self.cam_to_rob)[index_of_target]
        # move to target
        print("target transform")
        print(T_RT.homogeneous())
        self.move_to_angles(DEFAULT_POS)
        time.sleep(1)
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.05]))
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.01]))
        self.move_custom_gripper(T_RT)
        time.sleep(1)
        if pick_part:
            self.wiggle_wiggle_wiggle()
            self.gripper_pick()
        else:
            self.gripper_drop()
            self.wiggle_wiggle_wiggle()
        # move back
        self.move_custom_gripper(T_RT)
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.01]))
        self.move_custom_gripper(T_RT * SE3(translation=[0, 0, -0.05]))
        # move to default position
        self.move_to_angles(DEFAULT_POS)