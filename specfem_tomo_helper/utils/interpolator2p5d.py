#!/usr/bin/env python
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.interpolate import interp1d

class linear_interpolator1d2d:
    """
    DEPRECATED: This 1D+2D interpolator is no longer supported. Please use the 3D trilinear_interpolator instead.
    Any attempt to instantiate this class will raise an ImportError.
    """
    def __init__(self, *args, **kwargs):
        raise ImportError(
            "linear_interpolator1d2d is deprecated. Use trilinear_interpolator from interpolator3d.py instead."
        )