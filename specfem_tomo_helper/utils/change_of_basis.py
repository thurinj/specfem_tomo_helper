#!/usr/bin/env python3
"""
Transform between different bases for 3D cartesian coordinate systems.

This module provides functions to:
- Obtain reference coordinates for geographical directions.
- Retrieve basis matrices for different basis systems.
- Compute transformation matrices between different bases.
- Transform symmetric stiffness matrices (represented as vectors) from one
basis to another.

Notes
-----
All transformations assume right-handed coordinate systems.
"""

import numpy as np

def _check_elasticmapper_available():
    """Check if elasticmapper is available and raise an error if not."""
    try:
        import elasticmapper.core.rotation
        import elasticmapper.mapper_utils.change_of_basis
        import elasticmapper.mapper_utils.utilities
        return True
    except ImportError:
        raise ImportError(
            "The 'elasticmapper' package is required for anisotropic/elastic mapping features. "
            "This package is only needed when using anisotropic tomography models. "
            "Please install it with:\n"
            "pip install 'elasticmapper @ git+https://github.com/uafgeotools/elasticmapper.git'\n\n"
            "If you only need isotropic tomography features, this dependency is not required."
        )

def reference_coordinates(direction):
    """Coordinates for a vector pointing in a given geographical direction.

    Function to return the reference coordinates for a vector pointing
    in a given geographical direction with respect to a reference basis
    defined as x -> south, y -> east, z -> up

    Parameters
    ----------
    direction: str
        "east", "west", "north", "south", "up", "down"

    Returns
    -------
    b_vec: ndarray of shape (3,)
        coordinates for the given direction in the reference basis

    Raises
    ------
    AssertionError
        If `direction` is not one of the supported values.
        Error message: "Error: requested direction invalid"
    """
    assert direction in ["east", "west", "north", "south", "up", "down"],\
        "Error: requested direction invalid"

    if direction == "east":
        b_vec = np.array([0, 1, 0])
    elif direction == "west":
        b_vec = np.array([0, -1, 0])
    elif direction == "north":
        b_vec = np.array([-1, 0, 0])
    elif direction == "south":
        b_vec = np.array([1, 0, 0])
    elif direction == "up":
        b_vec = np.array([0, 0, 1])
    elif direction == "down":
        b_vec = np.array([0, 0, -1])

    return b_vec


def get_basis(basis):
    """Construct basis matrix for a 3D cartesian coordinate system.

    Function to get the basis matrix for a 3D cartesian coordinate system
    representing geographical directions with respect to a reference basis
    defined as x -> south, y -> east, z -> up.

    Parameters
    ----------
    basis: str
        string defining the basis; should be a right-handed system defined by
        an ordered set of 3 directions - east, west, north, south, up, down;
        combined using underscores into a string,
        e.g., 'east_north_up', 'north_east_down', 'south_east_up'

    Returns
    -------
    b_mat: ndarray of shape (3, 3)
        matrix of basis elements for the chosen basis, organized in columns;
        each column corresponds to a basis vector.

    Raises
    ------
    AssertionError
        If `basis` is not one of the supported values.
        Error message: "Error: requested basis '<basis>' is invalid"
    """

    assert basis in [
    "east_north_up",   "east_south_down",   "east_up_south", "east_down_north",
    "north_west_up",   "north_east_down",   "north_up_east", "north_down_west",
    "west_south_up",   "west_north_down",   "west_up_north", "west_down_south",
    "south_east_up",   "south_west_down",   "south_up_west", "south_down_east",
    "up_south_east",     "up_north_west",   "up_east_north",   "up_west_south",
    "down_north_east", "down_south_west", "down_east_south", "down_west_north"
    ], f"Error: requested basis '{basis}' is invalid"

    b_mat = np.zeros((3, 3))
    directions = basis.split('_')
    for i, direction in enumerate(directions):
        b_mat[:, i] = reference_coordinates(direction)
    return b_mat


def get_transformation_matrix(basis_2, basis_1):
    """Construct transformation matrix.

    Function to return matrix that transforms a 3D vector from one basis
    to another.

    Parameters
    ----------
    basis_2: str
        final basis
    basis_1: str
        initial basis

    Returns
    -------
    transformation_mat: ndarray of shape (3, 3)
        transformation matrix
    """
    b_mat_1 = get_basis(basis_1)
    b_mat_2 = get_basis(basis_2)
    transformation_mat = np.dot(np.linalg.inv(b_mat_2), b_mat_1)
    return transformation_mat


def transform(c_vec_1, basis_1):
    """Transform a stiffness vector from one basis to another.

    Function to transform vector representation of the symmetric stiffness
    matrix from one reference basis to another.

    Parameters
    ----------
    c_vec_1: ndarray of shape (21,)
        vector to be transformed, represented in the initial basis
    basis_1: str
        initial basis of the vector, should be a right-handed system defined
        by an ordered set of 3 directions - east, west, north, south, up, down;
        combined using underscores into a string,
        e.g., 'east_north_up', 'north_east_down', 'south_east_up'

    Returns
    -------
    c_vec_2: ndarray of shape (21,)
        vector transformed to the new basis, represented in the final basis
    """
    _check_elasticmapper_available()
    
    from elasticmapper.core.rotation import rotation_mat_6d
    from elasticmapper.mapper_utils.change_of_basis import c_mat_of_t_mat, t_mat_of_c_mat
    from elasticmapper.mapper_utils.utilities import v2sm, sm2v
    
    basis_2 = "east_north_up" # Basis definition in SPECFEM3D_cartesian

    u_mat = get_transformation_matrix(basis_2, basis_1)
    u_mat_6d = rotation_mat_6d(u_mat)

    c_mat_1 = v2sm(c_vec_1)
    t_mat_1 = t_mat_of_c_mat(c_mat_1)

    t_mat_2 = np.dot(np.dot(u_mat_6d, t_mat_1), u_mat_6d.T)

    c_mat_2 = c_mat_of_t_mat(t_mat_2)
    c_vec_2 = sm2v(c_mat_2)

    return c_vec_2