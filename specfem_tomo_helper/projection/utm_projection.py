#!/usr/bin/env python

from pyproj import Proj
import pyproj
import pyproj.proj

def define_utm_projection(utm_zone: int, hemisphere: str) -> pyproj.proj.Proj:
    """ Define lat-lon to utm projection based on pyproj. Meant to streamline
    the usage of pyproj.proj function by simplifying the inputs.

    Parameters
    ----------

    utm_zone : int
        The desired utm projection zone
    hemisphere : str
        The projection hemisphere. Only takes `N` or `S` as inut argument

    Notes
    ----------
    Using this function should be the preffered way to define the projection
    from lat/lon to utm easting and northing. If desired, users can handle the
    projection themselves by defining a custom pyproj.proj.Proj object.

    Returns
    ----------
    proj : pyproj.proj.Proj
        The projection object.

    """
    if hemisphere == 'N':
        proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', datum='WGS84', units='m')
    elif hemisphere == 'S':
        proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', datum='WGS84', units='m', south=True)
    else:
        raise ValueError('Hemisphere must be either N or S')
    return proj
