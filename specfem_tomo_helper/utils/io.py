#!/usr/bin/env python
import numpy as np
import pandas as pd


def write_tomo_file(
    tomo,
    interpolator,
    path,
    float_format="%.1f",
    attenuation=False
):
    """
    Write an isotropic tomography file.

    Without attenuation:
        x y z vp vs rho

    With attenuation:
        x y z vp vs rho qp qs
    """

    if not path.endswith('/'):
        path += '/'

    if attenuation:
        # x y z vp vs rho qp qs
        HEADER = _write_header(
            tomo[:, 0],
            tomo[:, 1],
            tomo[:, 2],
            tomo[:, 3],
            tomo[:, 4],
            tomo[:, 5],
            interpolator.x_interp_coordinates,
            interpolator.y_interp_coordinates,
            interpolator.z_interp_coordinates
        )

        MODEL_BODY = _create_tomo_df(
            tomo[:, 0],
            tomo[:, 1],
            tomo[:, 2],
            tomo[:, 3],
            tomo[:, 4],
            tomo[:, 5],
            tomo[:, 6],
            tomo[:, 7]
        )

    else:
        # x y z vp vs rho
        HEADER = _write_header(
            tomo[:, 0],
            tomo[:, 1],
            tomo[:, 2],
            tomo[:, 3],
            tomo[:, 4],
            tomo[:, 5],
            interpolator.x_interp_coordinates,
            interpolator.y_interp_coordinates,
            interpolator.z_interp_coordinates
        )

        MODEL_BODY = _create_tomo_df(
            tomo[:, 0],
            tomo[:, 1],
            tomo[:, 2],
            tomo[:, 3],
            tomo[:, 4],
            tomo[:, 5]
        )

    TOMO_XYZ_DF = pd.concat(
        [HEADER, MODEL_BODY],
        ignore_index=True
    )

    TOMO_XYZ_DF.to_csv(
        path + 'tomography_model.xyz',
        index=False,
        header=False,
        sep=" ",
        float_format=float_format
    )


def _write_header(
    interp_X,
    interp_Y,
    interp_Z,
    interp_vp,
    interp_vs,
    interp_rho,
    x_interp_coordinates,
    y_interp_coordinates,
    z_interp_coordinates
):
    print('FILE HEADER:')

    X_MIN = np.min(interp_X)
    X_MAX = np.max(interp_X)
    Y_MIN = np.min(interp_Y)
    Y_MAX = np.max(interp_Y)
    Z_MIN = np.min(interp_Z)
    Z_MAX = np.max(interp_Z)

    print(X_MIN, Y_MIN, Z_MIN, X_MAX, Y_MAX, Z_MAX)

    HEAD1 = np.asarray([
        X_MIN, Y_MIN, Z_MIN,
        X_MAX, Y_MAX, Z_MAX
    ])

    # -----------------------------------------------------------

    SPACING_X = np.abs(
        np.diff(x_interp_coordinates[0:2])
    )[0]

    SPACING_Y = np.abs(
        np.diff(y_interp_coordinates[0:2])
    )[0]

    z_diff = np.abs(np.diff(z_interp_coordinates))

    if np.allclose(z_diff, z_diff[0]):
        # Regular vertical spacing
        SPACING_Z = z_diff[0]
    else:
        # Variable vertical spacing
        SPACING_Z = z_diff[0]
        print("WARNING: Variable depth spacing detected.")
        print(f"Vertical spacings range from {z_diff.min()} to {z_diff.max()} m")


    print(SPACING_X, SPACING_Y, SPACING_Z)

    HEAD2 = np.asarray([
        SPACING_X,
        SPACING_Y,
        SPACING_Z
    ])

    # -----------------------------------------------------------

    NX = int(len(x_interp_coordinates))
    NY = int(len(y_interp_coordinates))
    NZ = int(len(z_interp_coordinates))

    print(NX, NY, NZ)

    HEAD3 = [
        str(NX),
        str(NY),
        str(NZ)
    ]

    # -----------------------------------------------------------

    VP_MIN = np.min(interp_vp)
    VP_MAX = np.max(interp_vp)

    VS_MIN = np.min(interp_vs)
    VS_MAX = np.max(interp_vs)

    RHO_MIN = np.min(interp_rho)
    RHO_MAX = np.max(interp_rho)

    print(
        VP_MIN * 1000,
        VP_MAX * 1000,
        VS_MIN * 1000,
        VS_MAX * 1000,
        RHO_MIN,
        RHO_MAX
    )

    HEAD4 = np.asarray([
        VP_MIN * 1000,
        VP_MAX * 1000,
        VS_MIN * 1000,
        VS_MAX * 1000,
        RHO_MIN,
        RHO_MAX
    ])

    # -----------------------------------------------------------

    HEADER = pd.DataFrame(
        data=[
            HEAD1,
            HEAD2,
            HEAD3,
            HEAD4
        ]
    )

    return HEADER


def _create_tomo_df(
    interp_X,
    interp_Y,
    interp_Z,
    vp_values,
    vs_values,
    rho_values,
    qp_values=None,
    qs_values=None
):
    """
    Create isotropic tomography body.

    Without attenuation:
        x y z vp vs rho

    With attenuation:
        x y z vp vs rho qp qs
    """

    GRID_DIM = np.prod(
        np.asarray(np.shape(interp_X))
    )

    X_COORDS = interp_X.reshape(1, GRID_DIM).T
    Y_COORDS = interp_Y.reshape(1, GRID_DIM).T
    Z_COORDS = interp_Z.reshape(1, GRID_DIM).T

    VP = vp_values.reshape(1, GRID_DIM).T
    VS = vs_values.reshape(1, GRID_DIM).T
    RHO = rho_values.reshape(1, GRID_DIM).T

    if qp_values is not None and qs_values is not None:

        QP = qp_values.reshape(1, GRID_DIM).T
        QS = qs_values.reshape(1, GRID_DIM).T

        TOMO_DF = np.hstack((
            X_COORDS,
            Y_COORDS,
            Z_COORDS,
            VP * 1000,
            VS * 1000,
            RHO,
            QP,
            QS
        ))

    else:

        TOMO_DF = np.hstack((
            X_COORDS,
            Y_COORDS,
            Z_COORDS,
            VP * 1000,
            VS * 1000,
            RHO
        ))

    return pd.DataFrame(data=TOMO_DF)


def write_anisotropic_tomo_file(
    tomo,
    interpolator,
    path,
    float_format="%.1f",
    attenuation=False
):
    """
    Write an anisotropic tomography file.

    Without attenuation:
        x y z c11 ... c66 rho

    With attenuation:
        x y z c11 ... c66 rho qp qs

    Input:
        Cij = GPa
        rho = kg/m^3
        qp, qs = dimensionless

    Output:
        Cij = Pa
        rho, qp, qs unchanged
    """

    if not path.endswith('/'):
        path += '/'

    # -----------------------------------------------------------
    # Column layout
    #
    # x y z = 0:3
    # c11...c66 = 3:24  (21 values)
    # rho = 24
    #
    # attenuation:
    # qp = 25
    # qs = 26
    # -----------------------------------------------------------

    C11_IDX = 3
    C44_IDX = 18
    RHO_IDX = 24

    # -----------------------------------------------------------
    # Header
    # -----------------------------------------------------------

    HEADER = _write_header(
        tomo[:, 0],
        tomo[:, 1],
        tomo[:, 2],

        # Vp = sqrt(c11 / rho)
        np.sqrt(
            tomo[:, C11_IDX] /
            tomo[:, RHO_IDX] *
            1e-3
        ) * 1e3,

        # Vs = sqrt(c44 / rho)
        np.sqrt(
            tomo[:, C44_IDX] /
            tomo[:, RHO_IDX] *
            1e-3
        ) * 1e3,

        # IMPORTANT:
        # rho is always column 24,
        # regardless of attenuation.
        tomo[:, RHO_IDX],

        interpolator.x_interp_coordinates,
        interpolator.y_interp_coordinates,
        interpolator.z_interp_coordinates
    )

    # -----------------------------------------------------------
    # Convert ONLY Cij values from GPa -> Pa
    #
    # 3:24 = exactly 21 stiffness coefficients.
    #
    # Do NOT convert rho, qp or qs.
    # -----------------------------------------------------------

    tomo[:, 3:24] *= 1e9

    # -----------------------------------------------------------
    # Write complete model
    # -----------------------------------------------------------

    MODEL_BODY = pd.DataFrame(tomo)

    TOMO_XYZ_DF = pd.concat(
        [HEADER, MODEL_BODY],
        ignore_index=True
    )

    TOMO_XYZ_DF.to_csv(
        path + 'tomography_model.xyz',
        index=False,
        header=False,
        sep=" ",
        float_format=float_format
    )

