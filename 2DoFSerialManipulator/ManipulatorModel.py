"""
Author: CTC 2801320287@qq.com
Date: 2024-05-14 15:17:42
LastEditors: CTC 2801320287@qq.com
LastEditTime: 2024-05-14 15:29:21
Description: Kinematical and Dynamical information about the 2-DoF Manipulator
Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import numpy as np
import math


class TwoDoFManipulator:
    def __init__(self, DYNAMICS, KINEMATICS):
        # DYNAMICS is a TUPLE, whose elements are namely (m1,m2,g,lc1,lc2,Jc1,Jc2)
        # KINEMATICS is a TUPLE, whose elements are namely (l1,l2)
        self.m1 = DYNAMICS[0]
        self.m2 = DYNAMICS[1]
        self.g = DYNAMICS[2]
        self.lc1 = DYNAMICS[3]
        self.lc2 = DYNAMICS[4]
        self.Jc1 = DYNAMICS[5]
        self.Jc2 = DYNAMICS[6]
        self.l1 = KINEMATICS[0]
        self.l2 = KINEMATICS[1]

    def T_c1_0(self, THETA):
        theta_1 = THETA[0, 0]
        lc1 = self.lc1
        return np.array(
            [
                [math.cos(theta_1), -math.sin(theta_1), 0, lc1 * math.cos(theta_1)],
                [math.sin(theta_1), math.cos(theta_1), 0, lc1 * math.sin(theta_1)],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def T_c2_0(self, THETA):
        theta_1 = THETA[0, 0]
        theta_2 = THETA[2, 0]
        lc2 = self.lc2
        l1 = self.l1
        return np.array(
            [
                [
                    math.cos(theta_1 + theta_2),
                    -math.sin(theta_1 + theta_2),
                    0,
                    l1 * math.cos(theta_1) + lc2 * math.cos(theta_1 + theta_2),
                ],
                [
                    math.sin(theta_1 + theta_2),
                    math.cos(theta_1 + theta_2),
                    0,
                    l1 * math.sin(theta_1) + lc2 * math.sin(theta_1 + theta_2),
                ],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def Jacobian_c1(self, THETA):
        theta_1 = THETA[0, 0]
        lc1 = self.lc1
        return np.array(
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [-lc1 * math.sin(theta_1), 0],
                [lc1 * math.cos(theta_1), 0],
                [0, 0],
            ]
        )

    def Jacobian_c2(self, THETA):
        theta_1 = THETA[0, 0]
        theta_2 = THETA[2, 0]
        lc2 = self.lc2
        l1 = self.l1
        return np.array(
            [
                [0, 0],
                [0, 0],
                [1, 1],
                [
                    -l1 * math.sin(theta_1) - lc2 * math.sin(theta_1 + theta_2),
                    -lc2 * math.sin(theta_1 + theta_2),
                ],
                [
                    l1 * math.cos(theta_1) + lc2 * math.cos(theta_1 + theta_2),
                    lc2 * math.cos(theta_1 + theta_2),
                ],
                [0, 0],
            ]
        )

    def M_Matrix(self, THETA):
        theta_1 = THETA[0, 0]
        theta_2 = THETA[1, 0]
        lc1 = self.lc1
        lc2 = self.lc2
        l1 = self.l1
        m1 = self.m1
        m2 = self.m2
        Jc1 = self.Jc1
        Jc2 = self.Jc2
        return np.array(
            [
                [
                    Jc1
                    + Jc2
                    + l1**2 * m2
                    + 2 * l1 * lc2 * m2 * math.cos(theta_2)
                    + lc1**2 * m1
                    + lc2**2 * m2,
                    Jc2 + l1 * lc2 * m2 * math.cos(theta_2) + lc2**2 * m2,
                ],
                [
                    Jc2 + l1 * lc2 * m2 * math.cos(theta_2) + lc2**2 * m2,
                    Jc2 + lc2**2 * m2,
                ],
            ]
        )

    def G_Vector(self, THETA):
        theta_1 = THETA[0, 0]
        theta_2 = THETA[1, 0]
        lc1 = self.lc1
        lc2 = self.lc2
        l1 = self.l1
        m1 = self.m1
        m2 = self.m2
        g = self.g
        return np.array(
            [
                [
                    g
                    * (
                        l1 * m2 * math.cos(theta_1)
                        + lc1 * m1 * math.cos(theta_1)
                        + lc2 * m2 * math.cos(theta_1 + theta_2)
                    )
                ],
                [g * lc2 * m2 * math.cos(theta_1 + theta_2)],
            ]
        )

    def C_Vector(self, THETA, dTHETA):
        theta_1 = THETA[0, 0]
        theta_2 = THETA[1, 0]
        dtheta_1 = dTHETA[0, 0]
        dtheta_2 = dTHETA[1, 0]
        lc1 = self.lc1
        lc2 = self.lc2
        l1 = self.l1
        m1 = self.m1
        m2 = self.m2
        return np.array(
            [
                [
                    -2 * dtheta_1 * dtheta_2 * l1 * lc2 * m2 * math.sin(theta_2)
                    - dtheta_2**2 * l1 * lc2 * m2 * math.sin(theta_2)
                ],
                [dtheta_1**2 * l1 * lc2 * m2 * math.sin(theta_2)],
            ]
        )

    def TorqueNoFriction(self, THETA, dTHETA, ddTHETA):
        return (
            self.M_Matrix(THETA) @ ddTHETA
            + self.C_Vector(THETA, dTHETA)
            + self.G_Vector(THETA)
        )
