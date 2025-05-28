#!/usr/bin/env python

from pyproj import Proj
import pyproj
import pyproj.proj

def auto_determine_utm_zone(longitude, latitude):
    """
    Automatically determine UTM zone and hemisphere from coordinates.
    
    Parameters
    ----------
    longitude : float
        Longitude in decimal degrees
    latitude : float  
        Latitude in decimal degrees
        
    Returns
    -------
    tuple
        (utm_zone, hemisphere) where utm_zone is int and hemisphere is 'N' or 'S'
    """
    # Calculate UTM zone from longitude
    utm_zone = int((longitude + 180) / 6) + 1
    
    # Determine hemisphere from latitude
    hemisphere = 'N' if latitude >= 0 else 'S'
    
    return utm_zone, hemisphere

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
