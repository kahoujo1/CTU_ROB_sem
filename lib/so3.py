#!/usr/bin/env python
#
# Copyright (c) CTU -- All Rights Reserved
# Created on: 2023-07-4
#     Author: Vladimir Petrik <vladimir.petrik@cvut.cz>
#

"""Module for representing 3D rotation."""

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


class SO3:
    """This class represents an SO3 rotations internally represented by rotation
    matrix."""

    def __init__(self, rotation_matrix: ArrayLike | None = None) -> None:
        """Creates a rotation transformation from rot_vector."""
        super().__init__()
        self.rot: np.ndarray = (
            np.asarray(rotation_matrix) if rotation_matrix is not None else np.eye(3)
        )

    @staticmethod
    def exp(rot_vector: ArrayLike) -> SO3:
        """Compute SO3 transformation from a given rotation vector, i.e. exponential
        representation of the rotation."""
        v = np.asarray(rot_vector)
        assert v.shape == (3,)
        t = SO3()
        # todo HW01: implement Rodrigues' formula, t.rot = ...
        # get the angle and the rotation vector from the exponential representation
        angle = np.linalg.norm(v)
        v = v / angle
        skew_v = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        t.rot += np.sin(angle)*skew_v + (1-np.cos(angle))* (skew_v @ skew_v)# tp.rot is already by default np.eye(3)
        return t

    def log(self) -> np.ndarray:
        """Compute rotation vector from this SO3"""
        # todo HW01: implement computation of rotation vector from this SO3
        v = np.zeros(3)
        ident_transform = SO3()
        if self.__eq__(ident_transform):
            pass # intentionally do nothing
        elif self.rot.trace() == -1:
            if self.rot[2][2] != -1:
                v = 1/np.sqrt(2*(1+self.rot[2][2])) * np.array([self.rot[0][2], self.rot[1][2], 1+self.rot[2][2]])
            elif self.rot[1][1] != -1:
                v = 1/np.sqrt(2*(1+self.rot[1][1])) * np.array([self.rot[0][1], 1+self.rot[1][1], self.rot[2][1]])
            else:
                v = 1/np.sqrt(2*(1+self.rot[0][0])) * np.array([1+self.rot[0][0], self.rot[1][0], self.rot[2][0]])
            v *= np.pi
        else:
            angle = np.arccos((np.trace(self.rot)-1)/2)
            skew_v = (self.rot - self.rot.T)/(2*np.sin(angle))
            v = angle * np.array([skew_v[2][1], skew_v[0][2], skew_v[1][0]])
        return v

    def __mul__(self, other: SO3) -> SO3:
        """Compose two rotations, i.e., self * other"""
        # todo: HW01: implement composition of two rotation.
        res = SO3()
        res.rot = self.rot @ other.rot
        return res

    def inverse(self) -> SO3:
        """Return inverse of the transformation."""
        # todo: HW01: implement inverse, do not use np.linalg.inverse()
        res = SO3()
        res.rot = self.rot.T
        return res

    def act(self, vector: ArrayLike) -> np.ndarray:
        """Rotate given vector by this transformation."""
        v = np.asarray(vector)
        assert v.shape == (3,)
        return self.rot @ v

    def __eq__(self, other: SO3) -> bool:
        """Returns true if two transformations are almost equal."""
        return np.allclose(self.rot, other.rot)

    @staticmethod
    def rx(angle: float) -> SO3:
        """Return rotation matrix around x axis."""
        # todo: HW1opt: implement rx
        res = SO3()
        res.rot = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)], 
            [0, np.sin(angle), np.cos(angle)]
                            ])
        return res
    
    @staticmethod
    def ry(angle: float) -> SO3:
        """Return rotation matrix around y axis."""
        # todo: HW1opt: implement ry
        res = SO3()
        res.rot = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0], 
            [-np.sin(angle), 0, np.cos(angle)]
                            ])
        return res

    @staticmethod
    def rz(angle: float) -> SO3:
        """Return rotation matrix around z axis."""
        # todo: HW1opt: implement rz
        res = SO3()
        res.rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
                            ])
        return res

    @staticmethod
    def from_quaternion(q: ArrayLike) -> SO3:
        """Compute rotation from quaternion in a form [qx, qy, qz, qw]."""
        # todo: HW1opt: implement from quaternion
        res = SO3()
        q_xyz = q[:3]
        res = res.exp(2*np.arccos(q[3])*q_xyz/np.linalg.norm(q_xyz))
        return res

    def to_quaternion(self) -> np.ndarray:
        """Compute quaternion from self."""
        # todo: HW1opt: implement to quaternion
        q = np.ndarray(4)
        q[3] = 1/2*np.sqrt(1+self.rot.trace())
        q[:3] = 1/(4*q[3])*np.array([self.rot[2][1]-self.rot[1][2], self.rot[0][2]-self.rot[2][0], self.rot[1][0]-self.rot[0][1]])
        return q

    @staticmethod
    def from_angle_axis(angle: float, axis: ArrayLike) -> SO3:
        """Compute rotation from angle axis representation."""
        # todo: HW1opt: implement from angle axis
        w = angle*axis/np.linalg.norm(axis)
        res = SO3()
        res = res.exp(w)
        return res


    def to_angle_axis(self) -> tuple[float, np.ndarray]:
        """Compute angle axis representation from self."""
        # todo: HW1opt: implement to angle axis
        w = self.log()
        angle = np.linalg.norm(w)
        axis = w/angle
        return angle, axis

    @staticmethod
    def from_euler_angles(angles: ArrayLike, seq: list[str]) -> SO3:
        """Compute rotation from euler angles defined by a given sequence.
        angles: is a three-dimensional array of angles
        seq: is a list of axis around which angles rotate, e.g. 'xyz', 'xzx', etc.
        """
        # todo: HW1opt: implement from euler angles
        res = SO3()
        for i in range(3):
            if seq[i] == 'x':
                res = res * SO3.rx(angles[i])
            elif seq[i] == 'y':
                res = res * SO3.ry(angles[i])
            else:
                res = res * SO3.rz(angles[i])
        return res

    def __hash__(self):
        return id(self)
