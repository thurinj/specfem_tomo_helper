#!/usr/bin/env python3
"""Test the change of basis functionality in specfem_tomo_helper."""

import numpy as np
from specfem_tomo_helper.utils.change_of_basis import transform

# These test stiffness values are taken from Brown et al. 2016
C_VEC_BROWN = np.array([
    68.3, 32.2, 30.4, 4.9, -2.3, -0.9,
    184.3, 5.0, -4.4, -7.8, -6.4,
    180.0, -9.2, 7.5, -9.4,
    25.0, -2.4, -7.2,
    26.9, 0.6,
    33.6
])

C_VEC_NORTH_EAST_DOWN_1 = C_VEC_BROWN

C_VEC_EAST_NORTH_UP_1 = np.array([
    184.3, 32.2,  5.0,   7.8,  4.4, -6.4,
    68.3, 30.4, 2.3, -4.9, -0.9,
    180.0, -7.5, 9.2, -9.4,
    26.9, -2.4, -0.6,
    25.0, 7.2,
    33.6
])

# These test stiffness values are taken from Igel et al. 1995
C_VEC_IGEL = np.array([
    10.0, 3.5, 2.5, -5.0, 0.1, 0.3,
    8.0, 1.5, 0.2, -0.1, -0.15,
    6.0, 1.0, 0.4, 0.24,
    5.0, 0.35, 0.525,
    4.0, -1.0,
    3.0
])

C_VEC_NORTH_EAST_DOWN_2 = C_VEC_IGEL

C_VEC_EAST_NORTH_UP_2 = np.array([
    8.0, 3.5, 1.5, 0.1, -0.2, -0.15,
    10.0, 2.5, -0.1, 5.0, 0.3,
    6.0, -0.4, -1.0, 0.24,
    4.0, 0.35, 1.0,
    5.0, -0.525,
    3.0
])

def test_basis_change_1():
    assert np.allclose(
        transform(C_VEC_NORTH_EAST_DOWN_1, 'north_east_down'),
        C_VEC_EAST_NORTH_UP_1,
    ), "basis change from 'north_east_down' to 'east_north_up' for C_VEC_BROWN failed"

def test_basis_change_2():
    assert np.allclose(
        transform(C_VEC_NORTH_EAST_DOWN_2, 'north_east_down'),
        C_VEC_EAST_NORTH_UP_2,
    ), "basis change from 'north_east_down' to 'east_north_up' for C_VEC_IGEL failed"