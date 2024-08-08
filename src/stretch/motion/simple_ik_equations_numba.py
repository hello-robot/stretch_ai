# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

###########################################################################
# The following license applies to simple_ik.py and
# simple_ik_equations_numba.py (the "Files"), which contain software
# for use with the Stretch mobile manipulators, which are robots
# produced and sold by Hello Robot Inc.

# Copyright 2024 Hello Robot Inc.

# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# v3.0 (GNU LGPLv3) as published by the Free Software Foundation.

# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License v3.0 (GNU LGPLv3) for more details,
# which can be found via the following link:

# https://www.gnu.org/licenses/lgpl-3.0.en.html

# For further information about the Files including inquiries about
# dual licensing, please contact Hello Robot Inc.
###########################################################################

import math

import numpy as np
from numba import njit


@njit
def idealized_fk_with_rotary_base(base_angle, lift_distance, arm_distance, b1):
    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground

    # lift (lambda)
    L = lift_distance
    # arm (alpha)
    A = arm_distance
    # base rotation (theta)
    T = base_angle

    x = (-math.cos(T) * b1) + (math.sin(T) * A)
    y = (-math.sin(T) * b1) - (math.cos(T) * A)
    z = L

    return (x, y, z)


@njit
def calibrated_fk_with_rotary_base(base_angle, lift_distance, arm_distance, b1, l_vector, a_vector):
    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground

    # unit vector in the direction of positive lift motion
    l1, l2, l3 = l_vector
    # unit vector in the direction of positive arm extension
    a1, a2, a3 = a_vector

    # lift (lambda)
    L = lift_distance
    # arm (alpha)
    A = arm_distance
    # base rotation (theta)
    T = base_angle

    term1 = (A * a1) + (L * l1) + b1
    term2 = (A * a2) + (L * l2)

    x = (math.cos(T) * term1) - (math.sin(T) * term2)
    y = (math.sin(T) * term1) + (math.cos(T) * term2)
    z = (A * a3) + (L * l3)

    return (x, y, z)


@njit
def idealized_ik_with_rotary_base(wrist_position, b1):
    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground
    x, y, z = wrist_position

    # lift (lambda)
    L = z
    # arm (alpha)
    A = math.sqrt(x**2 + y**2 - b1**2)
    # base rotation (theta)
    T = -2 * math.atan((A + y) / (b1 - x))

    return (T, L, A)


@njit
def calibrated_ik_with_rotary_base(wrist_position, b1, l_vector, a_vector):
    x, y, z = wrist_position

    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground

    # unit vector in the direction of positive lift motion
    l1, l2, l3 = l_vector
    # unit vector in the direction of positive arm extension
    a1, a2, a3 = a_vector

    # arm (alpha)
    A = (
        (
            (
                (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                + (-2 * a3 * (a1 * l1 + a2 * l2) * l3 + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2))
            )
        )
        ** (-1)
        * ((-1 * l3 * (a1 * x + a2 * y) + a3 * (l1 * x + l2 * y))) ** (-1)
        * (
            (a3) ** (2) * (l1 * x + l2 * y) * (b1 * l1 * l3 + ((l1) ** (2) + (l2) ** (2)) * z)
            + (
                -1
                * a3
                * l3
                * (
                    a2 * b1 * l1 * l3 * y
                    + (
                        a1 * b1 * l3 * (2 * l1 * x + l2 * y)
                        + (
                            a1 * (2 * (l1) ** (2) * x + ((l2) ** (2) * x + l1 * l2 * y)) * z
                            + a2 * (l1 * l2 * x + ((l1) ** (2) * y + 2 * (l2) ** (2) * y)) * z
                        )
                    )
                )
                + l3
                * (
                    (a1) ** (2) * b1 * (l3) ** (2) * x
                    + (
                        a1 * a2 * b1 * (l3) ** (2) * y
                        + (
                            (a1) ** (2) * l1 * l3 * x * z
                            + (
                                a1 * a2 * l2 * l3 * x * z
                                + (
                                    a1 * a2 * l1 * l3 * y * z
                                    + (
                                        (a2) ** (2) * l2 * l3 * y * z
                                        + (
                                            ((l3 * (a1 * x + a2 * y) + -1 * a3 * (l1 * x + l2 * y)))
                                            ** (2)
                                            * (
                                                (a3) ** (2)
                                                * (
                                                    -1 * (b1) ** (2) * (l2) ** (2)
                                                    + ((l1) ** (2) + (l2) ** (2))
                                                    * ((x) ** (2) + (y) ** (2))
                                                )
                                                + (
                                                    2 * a1 * a2 * l2 * z * (b1 * l3 + l1 * z)
                                                    + (
                                                        (a2) ** (2)
                                                        * (
                                                            -1 * (b1) ** (2) * (l3) ** (2)
                                                            + (
                                                                (l3) ** (2)
                                                                * ((x) ** (2) + (y) ** (2))
                                                                + (
                                                                    -2 * b1 * l1 * l3 * z
                                                                    + -1 * (l1) ** (2) * (z) ** (2)
                                                                )
                                                            )
                                                        )
                                                        + (
                                                            (a1) ** (2)
                                                            * (
                                                                (l3) ** (2)
                                                                * ((x) ** (2) + (y) ** (2))
                                                                + -1 * (l2) ** (2) * (z) ** (2)
                                                            )
                                                            + 2
                                                            * a3
                                                            * (
                                                                a2
                                                                * l2
                                                                * (
                                                                    (b1) ** (2) * l3
                                                                    + (
                                                                        -1
                                                                        * l3
                                                                        * ((x) ** (2) + (y) ** (2))
                                                                        + b1 * l1 * z
                                                                    )
                                                                )
                                                                + -1
                                                                * a1
                                                                * (
                                                                    l1
                                                                    * l3
                                                                    * ((x) ** (2) + (y) ** (2))
                                                                    + b1 * (l2) ** (2) * z
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                        ** (1 / 2)
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    if A < 0.0:

        # arm (alpha)
        A = (
            (
                (
                    (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                    + (
                        -2 * a3 * (a1 * l1 + a2 * l2) * l3
                        + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2)
                    )
                )
            )
            ** (-1)
            * ((-1 * l3 * (a1 * x + a2 * y) + a3 * (l1 * x + l2 * y))) ** (-1)
            * (
                (a3) ** (2) * (l1 * x + l2 * y) * (b1 * l1 * l3 + ((l1) ** (2) + (l2) ** (2)) * z)
                + (
                    -1
                    * a3
                    * l3
                    * (
                        a2 * b1 * l1 * l3 * y
                        + (
                            a1 * b1 * l3 * (2 * l1 * x + l2 * y)
                            + (
                                a1 * (2 * (l1) ** (2) * x + ((l2) ** (2) * x + l1 * l2 * y)) * z
                                + a2 * (l1 * l2 * x + ((l1) ** (2) * y + 2 * (l2) ** (2) * y)) * z
                            )
                        )
                    )
                    + l3
                    * (
                        (a1) ** (2) * b1 * (l3) ** (2) * x
                        + (
                            a1 * a2 * b1 * (l3) ** (2) * y
                            + (
                                (a1) ** (2) * l1 * l3 * x * z
                                + (
                                    a1 * a2 * l2 * l3 * x * z
                                    + (
                                        a1 * a2 * l1 * l3 * y * z
                                        + (
                                            (a2) ** (2) * l2 * l3 * y * z
                                            + -1
                                            * (
                                                (
                                                    (
                                                        l3 * (a1 * x + a2 * y)
                                                        + -1 * a3 * (l1 * x + l2 * y)
                                                    )
                                                )
                                                ** (2)
                                                * (
                                                    (a3) ** (2)
                                                    * (
                                                        -1 * (b1) ** (2) * (l2) ** (2)
                                                        + ((l1) ** (2) + (l2) ** (2))
                                                        * ((x) ** (2) + (y) ** (2))
                                                    )
                                                    + (
                                                        2 * a1 * a2 * l2 * z * (b1 * l3 + l1 * z)
                                                        + (
                                                            (a2) ** (2)
                                                            * (
                                                                -1 * (b1) ** (2) * (l3) ** (2)
                                                                + (
                                                                    (l3) ** (2)
                                                                    * ((x) ** (2) + (y) ** (2))
                                                                    + (
                                                                        -2 * b1 * l1 * l3 * z
                                                                        + -1
                                                                        * (l1) ** (2)
                                                                        * (z) ** (2)
                                                                    )
                                                                )
                                                            )
                                                            + (
                                                                (a1) ** (2)
                                                                * (
                                                                    (l3) ** (2)
                                                                    * ((x) ** (2) + (y) ** (2))
                                                                    + -1 * (l2) ** (2) * (z) ** (2)
                                                                )
                                                                + 2
                                                                * a3
                                                                * (
                                                                    a2
                                                                    * l2
                                                                    * (
                                                                        (b1) ** (2) * l3
                                                                        + (
                                                                            -1
                                                                            * l3
                                                                            * (
                                                                                (x) ** (2)
                                                                                + (y) ** (2)
                                                                            )
                                                                            + b1 * l1 * z
                                                                        )
                                                                    )
                                                                    + -1
                                                                    * a1
                                                                    * (
                                                                        l1
                                                                        * l3
                                                                        * ((x) ** (2) + (y) ** (2))
                                                                        + b1 * (l2) ** (2) * z
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                            ** (1 / 2)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # lift (lambda)
        L = (
            (
                (
                    (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                    + (
                        -2 * a3 * (a1 * l1 + a2 * l2) * l3
                        + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2)
                    )
                )
            )
            ** (-1)
            * ((-1 * l3 * (a1 * x + a2 * y) + a3 * (l1 * x + l2 * y))) ** (-1)
            * (
                -1 * (a3) ** (3) * b1 * l1 * (l1 * x + l2 * y)
                + (
                    -1 * ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2) * (a1 * x + a2 * y) * z
                    + (
                        (a3) ** (2)
                        * (
                            a2 * b1 * l1 * l3 * y
                            + (
                                a1 * b1 * l3 * (2 * l1 * x + l2 * y)
                                + (
                                    -1 * a1 * l1 * (l1 * x + l2 * y) * z
                                    + -1 * a2 * l2 * (l1 * x + l2 * y) * z
                                )
                            )
                        )
                        + a3
                        * (
                            -1 * (a1) ** (2) * b1 * (l3) ** (2) * x
                            + (
                                -1 * a1 * a2 * b1 * (l3) ** (2) * y
                                + (
                                    2 * (a1) ** (2) * l1 * l3 * x * z
                                    + (
                                        (a2) ** (2) * l1 * l3 * x * z
                                        + (
                                            a1 * a2 * l2 * l3 * x * z
                                            + (
                                                a1 * a2 * l1 * l3 * y * z
                                                + (
                                                    (a1) ** (2) * l2 * l3 * y * z
                                                    + (
                                                        2 * (a2) ** (2) * l2 * l3 * y * z
                                                        + (
                                                            (
                                                                (
                                                                    l3 * (a1 * x + a2 * y)
                                                                    + -1 * a3 * (l1 * x + l2 * y)
                                                                )
                                                            )
                                                            ** (2)
                                                            * (
                                                                (a3) ** (2)
                                                                * (
                                                                    -1 * (b1) ** (2) * (l2) ** (2)
                                                                    + ((l1) ** (2) + (l2) ** (2))
                                                                    * ((x) ** (2) + (y) ** (2))
                                                                )
                                                                + (
                                                                    2
                                                                    * a1
                                                                    * a2
                                                                    * l2
                                                                    * z
                                                                    * (b1 * l3 + l1 * z)
                                                                    + (
                                                                        (a2) ** (2)
                                                                        * (
                                                                            -1
                                                                            * (b1) ** (2)
                                                                            * (l3) ** (2)
                                                                            + (
                                                                                (l3) ** (2)
                                                                                * (
                                                                                    (x) ** (2)
                                                                                    + (y) ** (2)
                                                                                )
                                                                                + (
                                                                                    -2
                                                                                    * b1
                                                                                    * l1
                                                                                    * l3
                                                                                    * z
                                                                                    + -1
                                                                                    * (l1) ** (2)
                                                                                    * (z) ** (2)
                                                                                )
                                                                            )
                                                                        )
                                                                        + (
                                                                            (a1) ** (2)
                                                                            * (
                                                                                (l3) ** (2)
                                                                                * (
                                                                                    (x) ** (2)
                                                                                    + (y) ** (2)
                                                                                )
                                                                                + -1
                                                                                * (l2) ** (2)
                                                                                * (z) ** (2)
                                                                            )
                                                                            + 2
                                                                            * a3
                                                                            * (
                                                                                a2
                                                                                * l2
                                                                                * (
                                                                                    (b1) ** (2) * l3
                                                                                    + (
                                                                                        -1
                                                                                        * l3
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + b1
                                                                                        * l1
                                                                                        * z
                                                                                    )
                                                                                )
                                                                                + -1
                                                                                * a1
                                                                                * (
                                                                                    l1
                                                                                    * l3
                                                                                    * (
                                                                                        (x) ** (2)
                                                                                        + (y) ** (2)
                                                                                    )
                                                                                    + b1
                                                                                    * (l2) ** (2)
                                                                                    * z
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                        ** (1 / 2)
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # base rotation (theta)
        TArcTanArg1 = (
            (
                (
                    (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                    + (
                        -2 * a3 * (a1 * l1 + a2 * l2) * l3
                        + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2)
                    )
                )
            )
            ** (-1)
            * (((x) ** (2) + (y) ** (2))) ** (-1)
            * (
                (a3) ** (2) * b1 * (l2) ** (2) * x
                + (
                    -2 * a2 * a3 * b1 * l2 * l3 * x
                    + (
                        (a2) ** (2) * b1 * (l3) ** (2) * x
                        + (
                            -1 * (a3) ** (2) * b1 * l1 * l2 * y
                            + (
                                a2 * a3 * b1 * l1 * l3 * y
                                + (
                                    a1 * a3 * b1 * l2 * l3 * y
                                    + (
                                        -1 * a1 * a2 * b1 * (l3) ** (2) * y
                                        + (
                                            -1 * a2 * a3 * l1 * l2 * x * z
                                            + (
                                                a1 * a3 * (l2) ** (2) * x * z
                                                + (
                                                    (a2) ** (2) * l1 * l3 * x * z
                                                    + (
                                                        -1 * a1 * a2 * l2 * l3 * x * z
                                                        + (
                                                            a2 * a3 * (l1) ** (2) * y * z
                                                            + (
                                                                -1 * a1 * a3 * l1 * l2 * y * z
                                                                + (
                                                                    -1 * a1 * a2 * l1 * l3 * y * z
                                                                    + (
                                                                        (a1) ** (2)
                                                                        * l2
                                                                        * l3
                                                                        * y
                                                                        * z
                                                                        + (
                                                                            (
                                                                                (
                                                                                    l3
                                                                                    * (
                                                                                        a1 * x
                                                                                        + a2 * y
                                                                                    )
                                                                                    + -1
                                                                                    * a3
                                                                                    * (
                                                                                        l1 * x
                                                                                        + l2 * y
                                                                                    )
                                                                                )
                                                                            )
                                                                            ** (2)
                                                                            * (
                                                                                (a3) ** (2)
                                                                                * (
                                                                                    -1
                                                                                    * (b1) ** (2)
                                                                                    * (l2) ** (2)
                                                                                    + (
                                                                                        (l1) ** (2)
                                                                                        + (l2)
                                                                                        ** (2)
                                                                                    )
                                                                                    * (
                                                                                        (x) ** (2)
                                                                                        + (y) ** (2)
                                                                                    )
                                                                                )
                                                                                + (
                                                                                    2
                                                                                    * a1
                                                                                    * a2
                                                                                    * l2
                                                                                    * z
                                                                                    * (
                                                                                        b1 * l3
                                                                                        + l1 * z
                                                                                    )
                                                                                    + (
                                                                                        (a2) ** (2)
                                                                                        * (
                                                                                            -1
                                                                                            * (b1)
                                                                                            ** (2)
                                                                                            * (l3)
                                                                                            ** (2)
                                                                                            + (
                                                                                                (l3)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + (
                                                                                                    -2
                                                                                                    * b1
                                                                                                    * l1
                                                                                                    * l3
                                                                                                    * z
                                                                                                    + -1
                                                                                                    * (
                                                                                                        l1
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    * (
                                                                                                        z
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                        + (
                                                                                            (a1)
                                                                                            ** (2)
                                                                                            * (
                                                                                                (l3)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + -1
                                                                                                * (
                                                                                                    l2
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                * (
                                                                                                    z
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                            )
                                                                                            + 2
                                                                                            * a3
                                                                                            * (
                                                                                                a2
                                                                                                * l2
                                                                                                * (
                                                                                                    (
                                                                                                        b1
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    * l3
                                                                                                    + (
                                                                                                        -1
                                                                                                        * l3
                                                                                                        * (
                                                                                                            (
                                                                                                                x
                                                                                                            )
                                                                                                            ** (
                                                                                                                2
                                                                                                            )
                                                                                                            + (
                                                                                                                y
                                                                                                            )
                                                                                                            ** (
                                                                                                                2
                                                                                                            )
                                                                                                        )
                                                                                                        + b1
                                                                                                        * l1
                                                                                                        * z
                                                                                                    )
                                                                                                )
                                                                                                + -1
                                                                                                * a1
                                                                                                * (
                                                                                                    l1
                                                                                                    * l3
                                                                                                    * (
                                                                                                        (
                                                                                                            x
                                                                                                        )
                                                                                                        ** (
                                                                                                            2
                                                                                                        )
                                                                                                        + (
                                                                                                            y
                                                                                                        )
                                                                                                        ** (
                                                                                                            2
                                                                                                        )
                                                                                                    )
                                                                                                    + b1
                                                                                                    * (
                                                                                                        l2
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    * z
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                        ** (1 / 2)
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        TArcTanArg2 = (
            (
                (
                    (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                    + (
                        -2 * a3 * (a1 * l1 + a2 * l2) * l3
                        + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2)
                    )
                )
            )
            ** (-1)
            * (((x) ** (2) + (y) ** (2))) ** (-1)
            * ((-1 * l3 * (a1 * x + a2 * y) + a3 * (l1 * x + l2 * y))) ** (-1)
            * (
                (a3) ** (3) * b1 * l2 * ((l1 * x + l2 * y)) ** (2)
                + (
                    -1
                    * (a3) ** (2)
                    * (l1 * x + l2 * y)
                    * (
                        a2 * b1 * l3 * (l1 * x + 3 * l2 * y)
                        + (
                            a2 * l1 * (l1 * x + l2 * y) * z
                            + -1 * a1 * l2 * (-2 * b1 * l3 * x + (l1 * x * z + l2 * y * z))
                        )
                    )
                    + (
                        -1
                        * l3
                        * (
                            (a1) ** (2) * a2 * b1 * (l3) ** (2) * (x) ** (2)
                            + (
                                2 * a1 * (a2) ** (2) * b1 * (l3) ** (2) * x * y
                                + (
                                    (a2) ** (3) * b1 * (l3) ** (2) * (y) ** (2)
                                    + (
                                        (a1) ** (2) * a2 * l1 * l3 * (x) ** (2) * z
                                        + (
                                            -1 * (a1) ** (3) * l2 * l3 * (x) ** (2) * z
                                            + (
                                                2 * a1 * (a2) ** (2) * l1 * l3 * x * y * z
                                                + (
                                                    -2 * (a1) ** (2) * a2 * l2 * l3 * x * y * z
                                                    + (
                                                        (a2) ** (3) * l1 * l3 * (y) ** (2) * z
                                                        + (
                                                            -1
                                                            * a1
                                                            * (a2) ** (2)
                                                            * l2
                                                            * l3
                                                            * (y) ** (2)
                                                            * z
                                                            + (
                                                                -1
                                                                * a2
                                                                * x
                                                                * (
                                                                    (
                                                                        (
                                                                            l3 * (a1 * x + a2 * y)
                                                                            + -1
                                                                            * a3
                                                                            * (l1 * x + l2 * y)
                                                                        )
                                                                    )
                                                                    ** (2)
                                                                    * (
                                                                        (a3) ** (2)
                                                                        * (
                                                                            -1
                                                                            * (b1) ** (2)
                                                                            * (l2) ** (2)
                                                                            + (
                                                                                (l1) ** (2)
                                                                                + (l2) ** (2)
                                                                            )
                                                                            * (
                                                                                (x) ** (2)
                                                                                + (y) ** (2)
                                                                            )
                                                                        )
                                                                        + (
                                                                            2
                                                                            * a1
                                                                            * a2
                                                                            * l2
                                                                            * z
                                                                            * (b1 * l3 + l1 * z)
                                                                            + (
                                                                                (a2) ** (2)
                                                                                * (
                                                                                    -1
                                                                                    * (b1) ** (2)
                                                                                    * (l3) ** (2)
                                                                                    + (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + (
                                                                                            -2
                                                                                            * b1
                                                                                            * l1
                                                                                            * l3
                                                                                            * z
                                                                                            + -1
                                                                                            * (l1)
                                                                                            ** (2)
                                                                                            * (z)
                                                                                            ** (2)
                                                                                        )
                                                                                    )
                                                                                )
                                                                                + (
                                                                                    (a1) ** (2)
                                                                                    * (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + -1
                                                                                        * (l2)
                                                                                        ** (2)
                                                                                        * (z) ** (2)
                                                                                    )
                                                                                    + 2
                                                                                    * a3
                                                                                    * (
                                                                                        a2
                                                                                        * l2
                                                                                        * (
                                                                                            (b1)
                                                                                            ** (2)
                                                                                            * l3
                                                                                            + (
                                                                                                -1
                                                                                                * l3
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + b1
                                                                                                * l1
                                                                                                * z
                                                                                            )
                                                                                        )
                                                                                        + -1
                                                                                        * a1
                                                                                        * (
                                                                                            l1
                                                                                            * l3
                                                                                            * (
                                                                                                (x)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                + (
                                                                                                    y
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                            )
                                                                                            + b1
                                                                                            * (l2)
                                                                                            ** (2)
                                                                                            * z
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                                ** (1 / 2)
                                                                + a1
                                                                * y
                                                                * (
                                                                    (
                                                                        (
                                                                            l3 * (a1 * x + a2 * y)
                                                                            + -1
                                                                            * a3
                                                                            * (l1 * x + l2 * y)
                                                                        )
                                                                    )
                                                                    ** (2)
                                                                    * (
                                                                        (a3) ** (2)
                                                                        * (
                                                                            -1
                                                                            * (b1) ** (2)
                                                                            * (l2) ** (2)
                                                                            + (
                                                                                (l1) ** (2)
                                                                                + (l2) ** (2)
                                                                            )
                                                                            * (
                                                                                (x) ** (2)
                                                                                + (y) ** (2)
                                                                            )
                                                                        )
                                                                        + (
                                                                            2
                                                                            * a1
                                                                            * a2
                                                                            * l2
                                                                            * z
                                                                            * (b1 * l3 + l1 * z)
                                                                            + (
                                                                                (a2) ** (2)
                                                                                * (
                                                                                    -1
                                                                                    * (b1) ** (2)
                                                                                    * (l3) ** (2)
                                                                                    + (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + (
                                                                                            -2
                                                                                            * b1
                                                                                            * l1
                                                                                            * l3
                                                                                            * z
                                                                                            + -1
                                                                                            * (l1)
                                                                                            ** (2)
                                                                                            * (z)
                                                                                            ** (2)
                                                                                        )
                                                                                    )
                                                                                )
                                                                                + (
                                                                                    (a1) ** (2)
                                                                                    * (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + -1
                                                                                        * (l2)
                                                                                        ** (2)
                                                                                        * (z) ** (2)
                                                                                    )
                                                                                    + 2
                                                                                    * a3
                                                                                    * (
                                                                                        a2
                                                                                        * l2
                                                                                        * (
                                                                                            (b1)
                                                                                            ** (2)
                                                                                            * l3
                                                                                            + (
                                                                                                -1
                                                                                                * l3
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + b1
                                                                                                * l1
                                                                                                * z
                                                                                            )
                                                                                        )
                                                                                        + -1
                                                                                        * a1
                                                                                        * (
                                                                                            l1
                                                                                            * l3
                                                                                            * (
                                                                                                (x)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                + (
                                                                                                    y
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                            )
                                                                                            + b1
                                                                                            * (l2)
                                                                                            ** (2)
                                                                                            * z
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                                ** (1 / 2)
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        + a3
                        * (
                            (a1) ** (2) * l2 * l3 * x * (b1 * l3 * x + -2 * (l1 * x + l2 * y) * z)
                            + (
                                (a2) ** (2)
                                * l3
                                * y
                                * (
                                    b1 * l3 * (2 * l1 * x + 3 * l2 * y)
                                    + 2 * l1 * (l1 * x + l2 * y) * z
                                )
                                + (
                                    2
                                    * a1
                                    * a2
                                    * l3
                                    * (
                                        b1 * l3 * x * (l1 * x + 2 * l2 * y)
                                        + ((l1) ** (2) * (x) ** (2) + -1 * (l2) ** (2) * (y) ** (2))
                                        * z
                                    )
                                    + (-1 * l2 * x + l1 * y)
                                    * (
                                        ((l3 * (a1 * x + a2 * y) + -1 * a3 * (l1 * x + l2 * y)))
                                        ** (2)
                                        * (
                                            (a3) ** (2)
                                            * (
                                                -1 * (b1) ** (2) * (l2) ** (2)
                                                + ((l1) ** (2) + (l2) ** (2))
                                                * ((x) ** (2) + (y) ** (2))
                                            )
                                            + (
                                                2 * a1 * a2 * l2 * z * (b1 * l3 + l1 * z)
                                                + (
                                                    (a2) ** (2)
                                                    * (
                                                        -1 * (b1) ** (2) * (l3) ** (2)
                                                        + (
                                                            (l3) ** (2) * ((x) ** (2) + (y) ** (2))
                                                            + (
                                                                -2 * b1 * l1 * l3 * z
                                                                + -1 * (l1) ** (2) * (z) ** (2)
                                                            )
                                                        )
                                                    )
                                                    + (
                                                        (a1) ** (2)
                                                        * (
                                                            (l3) ** (2) * ((x) ** (2) + (y) ** (2))
                                                            + -1 * (l2) ** (2) * (z) ** (2)
                                                        )
                                                        + 2
                                                        * a3
                                                        * (
                                                            a2
                                                            * l2
                                                            * (
                                                                (b1) ** (2) * l3
                                                                + (
                                                                    -1
                                                                    * l3
                                                                    * ((x) ** (2) + (y) ** (2))
                                                                    + b1 * l1 * z
                                                                )
                                                            )
                                                            + -1
                                                            * a1
                                                            * (
                                                                l1 * l3 * ((x) ** (2) + (y) ** (2))
                                                                + b1 * (l2) ** (2) * z
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    ** (1 / 2)
                                )
                            )
                        )
                    )
                )
            )
        )

    else:

        # lift (lambda)
        L = (
            -1
            * (
                (
                    (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                    + (
                        -2 * a3 * (a1 * l1 + a2 * l2) * l3
                        + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2)
                    )
                )
            )
            ** (-1)
            * ((-1 * l3 * (a1 * x + a2 * y) + a3 * (l1 * x + l2 * y))) ** (-1)
            * (
                (a3) ** (3) * b1 * l1 * (l1 * x + l2 * y)
                + (
                    ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2) * (a1 * x + a2 * y) * z
                    + (
                        -1
                        * (a3) ** (2)
                        * (
                            a2 * b1 * l1 * l3 * y
                            + (
                                a1 * b1 * l3 * (2 * l1 * x + l2 * y)
                                + (
                                    -1 * a1 * l1 * (l1 * x + l2 * y) * z
                                    + -1 * a2 * l2 * (l1 * x + l2 * y) * z
                                )
                            )
                        )
                        + a3
                        * (
                            (a1) ** (2) * b1 * (l3) ** (2) * x
                            + (
                                a1 * a2 * b1 * (l3) ** (2) * y
                                + (
                                    -2 * (a1) ** (2) * l1 * l3 * x * z
                                    + (
                                        -1 * (a2) ** (2) * l1 * l3 * x * z
                                        + (
                                            -1 * a1 * a2 * l2 * l3 * x * z
                                            + (
                                                -1 * a1 * a2 * l1 * l3 * y * z
                                                + (
                                                    -1 * (a1) ** (2) * l2 * l3 * y * z
                                                    + (
                                                        -2 * (a2) ** (2) * l2 * l3 * y * z
                                                        + (
                                                            (
                                                                (
                                                                    l3 * (a1 * x + a2 * y)
                                                                    + -1 * a3 * (l1 * x + l2 * y)
                                                                )
                                                            )
                                                            ** (2)
                                                            * (
                                                                (a3) ** (2)
                                                                * (
                                                                    -1 * (b1) ** (2) * (l2) ** (2)
                                                                    + ((l1) ** (2) + (l2) ** (2))
                                                                    * ((x) ** (2) + (y) ** (2))
                                                                )
                                                                + (
                                                                    2
                                                                    * a1
                                                                    * a2
                                                                    * l2
                                                                    * z
                                                                    * (b1 * l3 + l1 * z)
                                                                    + (
                                                                        (a2) ** (2)
                                                                        * (
                                                                            -1
                                                                            * (b1) ** (2)
                                                                            * (l3) ** (2)
                                                                            + (
                                                                                (l3) ** (2)
                                                                                * (
                                                                                    (x) ** (2)
                                                                                    + (y) ** (2)
                                                                                )
                                                                                + (
                                                                                    -2
                                                                                    * b1
                                                                                    * l1
                                                                                    * l3
                                                                                    * z
                                                                                    + -1
                                                                                    * (l1) ** (2)
                                                                                    * (z) ** (2)
                                                                                )
                                                                            )
                                                                        )
                                                                        + (
                                                                            (a1) ** (2)
                                                                            * (
                                                                                (l3) ** (2)
                                                                                * (
                                                                                    (x) ** (2)
                                                                                    + (y) ** (2)
                                                                                )
                                                                                + -1
                                                                                * (l2) ** (2)
                                                                                * (z) ** (2)
                                                                            )
                                                                            + 2
                                                                            * a3
                                                                            * (
                                                                                a2
                                                                                * l2
                                                                                * (
                                                                                    (b1) ** (2) * l3
                                                                                    + (
                                                                                        -1
                                                                                        * l3
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + b1
                                                                                        * l1
                                                                                        * z
                                                                                    )
                                                                                )
                                                                                + -1
                                                                                * a1
                                                                                * (
                                                                                    l1
                                                                                    * l3
                                                                                    * (
                                                                                        (x) ** (2)
                                                                                        + (y) ** (2)
                                                                                    )
                                                                                    + b1
                                                                                    * (l2) ** (2)
                                                                                    * z
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                            )
                                                        )
                                                        ** (1 / 2)
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        # base rotation (theta)
        TArcTanArg1 = (
            (
                (
                    (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                    + (
                        -2 * a3 * (a1 * l1 + a2 * l2) * l3
                        + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2)
                    )
                )
            )
            ** (-1)
            * (((x) ** (2) + (y) ** (2))) ** (-1)
            * (
                (a3) ** (2) * b1 * (l2) ** (2) * x
                + (
                    -2 * a2 * a3 * b1 * l2 * l3 * x
                    + (
                        (a2) ** (2) * b1 * (l3) ** (2) * x
                        + (
                            -1 * (a3) ** (2) * b1 * l1 * l2 * y
                            + (
                                a2 * a3 * b1 * l1 * l3 * y
                                + (
                                    a1 * a3 * b1 * l2 * l3 * y
                                    + (
                                        -1 * a1 * a2 * b1 * (l3) ** (2) * y
                                        + (
                                            -1 * a2 * a3 * l1 * l2 * x * z
                                            + (
                                                a1 * a3 * (l2) ** (2) * x * z
                                                + (
                                                    (a2) ** (2) * l1 * l3 * x * z
                                                    + (
                                                        -1 * a1 * a2 * l2 * l3 * x * z
                                                        + (
                                                            a2 * a3 * (l1) ** (2) * y * z
                                                            + (
                                                                -1 * a1 * a3 * l1 * l2 * y * z
                                                                + (
                                                                    -1 * a1 * a2 * l1 * l3 * y * z
                                                                    + (
                                                                        (a1) ** (2)
                                                                        * l2
                                                                        * l3
                                                                        * y
                                                                        * z
                                                                        + -1
                                                                        * (
                                                                            (
                                                                                (
                                                                                    l3
                                                                                    * (
                                                                                        a1 * x
                                                                                        + a2 * y
                                                                                    )
                                                                                    + -1
                                                                                    * a3
                                                                                    * (
                                                                                        l1 * x
                                                                                        + l2 * y
                                                                                    )
                                                                                )
                                                                            )
                                                                            ** (2)
                                                                            * (
                                                                                (a3) ** (2)
                                                                                * (
                                                                                    -1
                                                                                    * (b1) ** (2)
                                                                                    * (l2) ** (2)
                                                                                    + (
                                                                                        (l1) ** (2)
                                                                                        + (l2)
                                                                                        ** (2)
                                                                                    )
                                                                                    * (
                                                                                        (x) ** (2)
                                                                                        + (y) ** (2)
                                                                                    )
                                                                                )
                                                                                + (
                                                                                    2
                                                                                    * a1
                                                                                    * a2
                                                                                    * l2
                                                                                    * z
                                                                                    * (
                                                                                        b1 * l3
                                                                                        + l1 * z
                                                                                    )
                                                                                    + (
                                                                                        (a2) ** (2)
                                                                                        * (
                                                                                            -1
                                                                                            * (b1)
                                                                                            ** (2)
                                                                                            * (l3)
                                                                                            ** (2)
                                                                                            + (
                                                                                                (l3)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + (
                                                                                                    -2
                                                                                                    * b1
                                                                                                    * l1
                                                                                                    * l3
                                                                                                    * z
                                                                                                    + -1
                                                                                                    * (
                                                                                                        l1
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    * (
                                                                                                        z
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                        + (
                                                                                            (a1)
                                                                                            ** (2)
                                                                                            * (
                                                                                                (l3)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + -1
                                                                                                * (
                                                                                                    l2
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                * (
                                                                                                    z
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                            )
                                                                                            + 2
                                                                                            * a3
                                                                                            * (
                                                                                                a2
                                                                                                * l2
                                                                                                * (
                                                                                                    (
                                                                                                        b1
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    * l3
                                                                                                    + (
                                                                                                        -1
                                                                                                        * l3
                                                                                                        * (
                                                                                                            (
                                                                                                                x
                                                                                                            )
                                                                                                            ** (
                                                                                                                2
                                                                                                            )
                                                                                                            + (
                                                                                                                y
                                                                                                            )
                                                                                                            ** (
                                                                                                                2
                                                                                                            )
                                                                                                        )
                                                                                                        + b1
                                                                                                        * l1
                                                                                                        * z
                                                                                                    )
                                                                                                )
                                                                                                + -1
                                                                                                * a1
                                                                                                * (
                                                                                                    l1
                                                                                                    * l3
                                                                                                    * (
                                                                                                        (
                                                                                                            x
                                                                                                        )
                                                                                                        ** (
                                                                                                            2
                                                                                                        )
                                                                                                        + (
                                                                                                            y
                                                                                                        )
                                                                                                        ** (
                                                                                                            2
                                                                                                        )
                                                                                                    )
                                                                                                    + b1
                                                                                                    * (
                                                                                                        l2
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    * z
                                                                                                )
                                                                                            )
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                        ** (1 / 2)
                                                                    )
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )

        TArcTanArg2 = (
            (
                (
                    (a3) ** (2) * ((l1) ** (2) + (l2) ** (2))
                    + (
                        -2 * a3 * (a1 * l1 + a2 * l2) * l3
                        + ((a1) ** (2) + (a2) ** (2)) * (l3) ** (2)
                    )
                )
            )
            ** (-1)
            * (((x) ** (2) + (y) ** (2))) ** (-1)
            * ((-1 * l3 * (a1 * x + a2 * y) + a3 * (l1 * x + l2 * y))) ** (-1)
            * (
                (a3) ** (3) * b1 * l2 * ((l1 * x + l2 * y)) ** (2)
                + (
                    -1
                    * (a3) ** (2)
                    * (l1 * x + l2 * y)
                    * (
                        a2 * b1 * l3 * (l1 * x + 3 * l2 * y)
                        + (
                            a2 * l1 * (l1 * x + l2 * y) * z
                            + -1 * a1 * l2 * (-2 * b1 * l3 * x + (l1 * x * z + l2 * y * z))
                        )
                    )
                    + (
                        -1
                        * l3
                        * (
                            (a1) ** (2) * a2 * b1 * (l3) ** (2) * (x) ** (2)
                            + (
                                2 * a1 * (a2) ** (2) * b1 * (l3) ** (2) * x * y
                                + (
                                    (a2) ** (3) * b1 * (l3) ** (2) * (y) ** (2)
                                    + (
                                        (a1) ** (2) * a2 * l1 * l3 * (x) ** (2) * z
                                        + (
                                            -1 * (a1) ** (3) * l2 * l3 * (x) ** (2) * z
                                            + (
                                                2 * a1 * (a2) ** (2) * l1 * l3 * x * y * z
                                                + (
                                                    -2 * (a1) ** (2) * a2 * l2 * l3 * x * y * z
                                                    + (
                                                        (a2) ** (3) * l1 * l3 * (y) ** (2) * z
                                                        + (
                                                            -1
                                                            * a1
                                                            * (a2) ** (2)
                                                            * l2
                                                            * l3
                                                            * (y) ** (2)
                                                            * z
                                                            + (
                                                                a2
                                                                * x
                                                                * (
                                                                    (
                                                                        (
                                                                            l3 * (a1 * x + a2 * y)
                                                                            + -1
                                                                            * a3
                                                                            * (l1 * x + l2 * y)
                                                                        )
                                                                    )
                                                                    ** (2)
                                                                    * (
                                                                        (a3) ** (2)
                                                                        * (
                                                                            -1
                                                                            * (b1) ** (2)
                                                                            * (l2) ** (2)
                                                                            + (
                                                                                (l1) ** (2)
                                                                                + (l2) ** (2)
                                                                            )
                                                                            * (
                                                                                (x) ** (2)
                                                                                + (y) ** (2)
                                                                            )
                                                                        )
                                                                        + (
                                                                            2
                                                                            * a1
                                                                            * a2
                                                                            * l2
                                                                            * z
                                                                            * (b1 * l3 + l1 * z)
                                                                            + (
                                                                                (a2) ** (2)
                                                                                * (
                                                                                    -1
                                                                                    * (b1) ** (2)
                                                                                    * (l3) ** (2)
                                                                                    + (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + (
                                                                                            -2
                                                                                            * b1
                                                                                            * l1
                                                                                            * l3
                                                                                            * z
                                                                                            + -1
                                                                                            * (l1)
                                                                                            ** (2)
                                                                                            * (z)
                                                                                            ** (2)
                                                                                        )
                                                                                    )
                                                                                )
                                                                                + (
                                                                                    (a1) ** (2)
                                                                                    * (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + -1
                                                                                        * (l2)
                                                                                        ** (2)
                                                                                        * (z) ** (2)
                                                                                    )
                                                                                    + 2
                                                                                    * a3
                                                                                    * (
                                                                                        a2
                                                                                        * l2
                                                                                        * (
                                                                                            (b1)
                                                                                            ** (2)
                                                                                            * l3
                                                                                            + (
                                                                                                -1
                                                                                                * l3
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + b1
                                                                                                * l1
                                                                                                * z
                                                                                            )
                                                                                        )
                                                                                        + -1
                                                                                        * a1
                                                                                        * (
                                                                                            l1
                                                                                            * l3
                                                                                            * (
                                                                                                (x)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                + (
                                                                                                    y
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                            )
                                                                                            + b1
                                                                                            * (l2)
                                                                                            ** (2)
                                                                                            * z
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                                ** (1 / 2)
                                                                + -1
                                                                * a1
                                                                * y
                                                                * (
                                                                    (
                                                                        (
                                                                            l3 * (a1 * x + a2 * y)
                                                                            + -1
                                                                            * a3
                                                                            * (l1 * x + l2 * y)
                                                                        )
                                                                    )
                                                                    ** (2)
                                                                    * (
                                                                        (a3) ** (2)
                                                                        * (
                                                                            -1
                                                                            * (b1) ** (2)
                                                                            * (l2) ** (2)
                                                                            + (
                                                                                (l1) ** (2)
                                                                                + (l2) ** (2)
                                                                            )
                                                                            * (
                                                                                (x) ** (2)
                                                                                + (y) ** (2)
                                                                            )
                                                                        )
                                                                        + (
                                                                            2
                                                                            * a1
                                                                            * a2
                                                                            * l2
                                                                            * z
                                                                            * (b1 * l3 + l1 * z)
                                                                            + (
                                                                                (a2) ** (2)
                                                                                * (
                                                                                    -1
                                                                                    * (b1) ** (2)
                                                                                    * (l3) ** (2)
                                                                                    + (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + (
                                                                                            -2
                                                                                            * b1
                                                                                            * l1
                                                                                            * l3
                                                                                            * z
                                                                                            + -1
                                                                                            * (l1)
                                                                                            ** (2)
                                                                                            * (z)
                                                                                            ** (2)
                                                                                        )
                                                                                    )
                                                                                )
                                                                                + (
                                                                                    (a1) ** (2)
                                                                                    * (
                                                                                        (l3) ** (2)
                                                                                        * (
                                                                                            (x)
                                                                                            ** (2)
                                                                                            + (y)
                                                                                            ** (2)
                                                                                        )
                                                                                        + -1
                                                                                        * (l2)
                                                                                        ** (2)
                                                                                        * (z) ** (2)
                                                                                    )
                                                                                    + 2
                                                                                    * a3
                                                                                    * (
                                                                                        a2
                                                                                        * l2
                                                                                        * (
                                                                                            (b1)
                                                                                            ** (2)
                                                                                            * l3
                                                                                            + (
                                                                                                -1
                                                                                                * l3
                                                                                                * (
                                                                                                    (
                                                                                                        x
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                    + (
                                                                                                        y
                                                                                                    )
                                                                                                    ** (
                                                                                                        2
                                                                                                    )
                                                                                                )
                                                                                                + b1
                                                                                                * l1
                                                                                                * z
                                                                                            )
                                                                                        )
                                                                                        + -1
                                                                                        * a1
                                                                                        * (
                                                                                            l1
                                                                                            * l3
                                                                                            * (
                                                                                                (x)
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                                + (
                                                                                                    y
                                                                                                )
                                                                                                ** (
                                                                                                    2
                                                                                                )
                                                                                            )
                                                                                            + b1
                                                                                            * (l2)
                                                                                            ** (2)
                                                                                            * z
                                                                                        )
                                                                                    )
                                                                                )
                                                                            )
                                                                        )
                                                                    )
                                                                )
                                                                ** (1 / 2)
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                        + a3
                        * (
                            (a1) ** (2) * l2 * l3 * x * (b1 * l3 * x + -2 * (l1 * x + l2 * y) * z)
                            + (
                                (a2) ** (2)
                                * l3
                                * y
                                * (
                                    b1 * l3 * (2 * l1 * x + 3 * l2 * y)
                                    + 2 * l1 * (l1 * x + l2 * y) * z
                                )
                                + (
                                    2
                                    * a1
                                    * a2
                                    * l3
                                    * (
                                        b1 * l3 * x * (l1 * x + 2 * l2 * y)
                                        + ((l1) ** (2) * (x) ** (2) + -1 * (l2) ** (2) * (y) ** (2))
                                        * z
                                    )
                                    + (l2 * x + -1 * l1 * y)
                                    * (
                                        ((l3 * (a1 * x + a2 * y) + -1 * a3 * (l1 * x + l2 * y)))
                                        ** (2)
                                        * (
                                            (a3) ** (2)
                                            * (
                                                -1 * (b1) ** (2) * (l2) ** (2)
                                                + ((l1) ** (2) + (l2) ** (2))
                                                * ((x) ** (2) + (y) ** (2))
                                            )
                                            + (
                                                2 * a1 * a2 * l2 * z * (b1 * l3 + l1 * z)
                                                + (
                                                    (a2) ** (2)
                                                    * (
                                                        -1 * (b1) ** (2) * (l3) ** (2)
                                                        + (
                                                            (l3) ** (2) * ((x) ** (2) + (y) ** (2))
                                                            + (
                                                                -2 * b1 * l1 * l3 * z
                                                                + -1 * (l1) ** (2) * (z) ** (2)
                                                            )
                                                        )
                                                    )
                                                    + (
                                                        (a1) ** (2)
                                                        * (
                                                            (l3) ** (2) * ((x) ** (2) + (y) ** (2))
                                                            + -1 * (l2) ** (2) * (z) ** (2)
                                                        )
                                                        + 2
                                                        * a3
                                                        * (
                                                            a2
                                                            * l2
                                                            * (
                                                                (b1) ** (2) * l3
                                                                + (
                                                                    -1
                                                                    * l3
                                                                    * ((x) ** (2) + (y) ** (2))
                                                                    + b1 * l1 * z
                                                                )
                                                            )
                                                            + -1
                                                            * a1
                                                            * (
                                                                l1 * l3 * ((x) ** (2) + (y) ** (2))
                                                                + b1 * (l2) ** (2) * z
                                                            )
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                    ** (1 / 2)
                                )
                            )
                        )
                    )
                )
            )
        )

    T = np.arctan2(TArcTanArg2, TArcTanArg1)

    return (T, L, A)


@njit
def idealized_fk_with_prismatic_base(base_distance, lift_distance, arm_distance, b1):
    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground

    # lift (lambda)
    L = lift_distance
    # arm (alpha)
    A = arm_distance
    # base translation (mu)
    M = base_distance

    x = M - b1
    y = -A
    z = L

    return (x, y, z)


@njit
def calibrated_fk_with_prismatic_base(
    base_distance, lift_distance, arm_distance, b1, l_vector, a_vector
):
    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground

    # unit vector in the direction of positive lift motion
    l1, l2, l3 = l_vector
    # unit vector in the direction of positive arm extension
    a1, a2, a3 = a_vector

    # lift (lambda)
    L = lift_distance
    # arm (alpha)
    A = arm_distance
    # base translation (mu)
    M = base_distance

    x = A * a1 + L * l1 + M + b1
    y = A * a2 + L * l2
    z = A * a3 + L * l3

    return (x, y, z)


@njit
def idealized_ik_with_prismatic_base(wrist_position, b1):
    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground

    x, y, z = wrist_position

    # base translation (mu)
    M = x + b1
    # lift (lambda)
    L = z
    # arm (alpha)
    A = -y

    return (M, L, A)


@njit
def calibrated_ik_with_prismatic_base(wrist_position, b1, l_vector, a_vector):
    # b1 is the distance in the x direction between the mobile base rotational axis to the lift on the ground

    x, y, z = wrist_position

    # unit vector in the direction of positive lift motion
    l1, l2, l3 = l_vector
    # unit vector in the direction of positive arm extension
    a1, a2, a3 = a_vector

    denominator = (a2 * l3) - (a3 * l2)

    # base translation (mu)
    M = (
        a1 * ((l2 * z) - (l3 * y))
        + a2 * ((-b1 * l3) - (l1 * z) + (l3 * x))
        + a3 * ((b1 * l2) + (l1 * y) - (l2 * x))
    ) / denominator
    # lift (lambda)
    L = ((a2 * z) - (a3 * y)) / denominator
    # arm (alpha)
    A = ((-l2 * z) + (l3 * y)) / denominator

    return (M, L, A)
