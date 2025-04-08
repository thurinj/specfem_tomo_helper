#!/usr/bin/env python
import numpy as np
import pandas as pd

def write_tomo_file(tomo, interpolator, path, float_format="%.1f"):
    """ Write the tomo file at path+tomography_file.xyz
    Runs through two independant functions for 1) the header and 2) the body.

    It requires the tomo array with the following structure:
    x  y  z  vp  vs  rho
    .  .  .   .   .   .
    .  .  .   .   .   .
    .  .  .   .   .   .

    """
    # Check path format
    if not path.endswith('/'):
        path=path+'/'
    # Create the header
    HEADER = _write_header(tomo[:,0], tomo[:,1], tomo[:,2], tomo[:,3], tomo[:,4], tomo[:,5], interpolator.x_interp_coordinates, interpolator.y_interp_coordinates, interpolator.z_interp_coordinates)
    # Create the body of the model file
    MODEL_BODY = _create_tomo_df(tomo[:,0], tomo[:,1], tomo[:,2], tomo[:,3], tomo[:,4], tomo[:,5])
    # Append the two text files
    TOMO_XYZ_DF = pd.concat([HEADER, MODEL_BODY], ignore_index=True)
    # Write file
    TOMO_XYZ_DF.to_csv(path+'tomography_model.xyz', index=False, header=False, sep=" ", float_format=float_format)


def _write_header(interp_X,interp_Y,interp_Z,interp_vp,interp_vs, interp_rho, x_interp_coordinates, y_interp_coordinates, z_interp_coordinates):
    print('FILE HEADER:')
    X_MIN = np.min(interp_X)
    X_MAX = np.max(interp_X)
    Y_MIN = np.min(interp_Y)
    Y_MAX = np.max(interp_Y)
    Z_MIN = np.min(interp_Z)
    Z_MAX = np.max(interp_Z)
    print(X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX)
    HEAD1 = np.asarray([X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX])
    # -----------------------------------------------------------
    SPACING_X = np.abs(np.diff(x_interp_coordinates[0:2]))[0]
    SPACING_Y = np.abs(np.diff(y_interp_coordinates[0:2]))[0]
    SPACING_Z = np.abs(np.diff(z_interp_coordinates[0:2]))[0]
    print(SPACING_X, SPACING_Y, SPACING_Z)
    HEAD2 = np.asarray([SPACING_X, SPACING_Y, SPACING_Z])
    # -----------------------------------------------------------
    NX = len(x_interp_coordinates)
    NY = len(y_interp_coordinates)
    NZ = len(z_interp_coordinates)
    print(NX, NY, NZ)
    HEAD3 = np.asarray([NX, NY, NZ])
    # -----------------------------------------------------------
    VP_MIN = np.min(interp_vp)
    VP_MAX = np.max(interp_vp)
    VS_MIN = np.min(interp_vs)
    VS_MAX = np.max(interp_vs)
    RHO_MIN = np.min(interp_rho)
    RHO_MAX = np.max(interp_rho)
    print(VP_MIN*1000, VP_MAX*1000, VS_MIN*1000, VS_MAX*1000, RHO_MIN, RHO_MAX)
    HEAD4 = np.asarray([VP_MIN*1000, VP_MAX*1000, VS_MIN*1000, VS_MAX*1000, RHO_MIN, RHO_MAX])
    # -----------------------------------------------------------
    HEADER = pd.DataFrame(data=[HEAD1, HEAD2, HEAD3, HEAD4])
    return(HEADER)

def _create_tomo_df(interp_X,interp_Y,interp_Z,vp_values, vs_values, rho_values):
    GRID_DIM = np.prod(np.asarray(np.shape(interp_X)))
    X_COORDS = interp_X.reshape(1, GRID_DIM).T
    Y_COORDS = interp_Y.reshape(1, GRID_DIM).T
    Z_COORDS = interp_Z.reshape(1, GRID_DIM).T
    VP = vp_values.reshape(1, GRID_DIM).T
    VS = vs_values.reshape(1, GRID_DIM).T
    RHO = rho_values.reshape(1, GRID_DIM).T
    TOMO_DF = np.hstack((X_COORDS, Y_COORDS, Z_COORDS, VP*1000, VS*1000, RHO))
    return(pd.DataFrame(data=TOMO_DF))


def write_anisotropic_tomo_file(tomo, interpolator, path, float_format="%.1f"):
    """ Write a tomography file with full anisotropic stiffness tensor (cij) + density (rho).
    Format: x y z c11 c12 c13 c14 c15 c16 c22 ... c66 rho

    The 4th line of the header is somehow ad-hoc and might need to be revised in the future.

    This assume that the Cij values in tomo are in GPa and rho in kg/m^3.
    The output file is in the format required by SPECFEM3D as per https://github.com/SPECFEM/specfem3d/pull/1435
    """
    if not path.endswith('/'):
        path += '/'

    HEADER = _write_header(
        tomo[:, 0], tomo[:, 1], tomo[:, 2],
        np.sqrt(tomo[:, 3]/tomo[:,-1]*1e-3)*1e3,  # np.sqrt(c11/rho)
        np.sqrt(tomo[:, 18]/tomo[:,-1]*1e-3)*1e3,  # np.sqrt(c44/rho)
        tomo[:, -1],  # rho
        interpolator.x_interp_coordinates,
        interpolator.y_interp_coordinates,
        interpolator.z_interp_coordinates,
    )
    
    # Convert GPa to Pa for all Cij values before saving
    tomo[:, 3:-1] *= 1e9

    MODEL_BODY = pd.DataFrame(tomo)
    TOMO_XYZ_DF = pd.concat([HEADER, MODEL_BODY], ignore_index=True)
    TOMO_XYZ_DF.to_csv(path + 'tomography_model.xyz', index=False, header=False, sep=" ", float_format=float_format)
