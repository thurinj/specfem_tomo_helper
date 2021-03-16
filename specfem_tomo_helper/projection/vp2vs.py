#!/usr/bin/env python

def vp2vs(vp):
    """ Math function definitions to convert vp to vs from Brocher T. 2005, https://doi.org/10.1785/0120050077

    Parameters
    ----------

    vp : float
        P-wave velocity (km/s)

    """
    if vp >= 15:
        raise AssertionError('vp values must be in km/s')
    vspeed = vp
    vs = 0.7858 - 1.2344*vspeed + 0.7949*vspeed**2 - 0.1238*vp**3 + 0.0064*vp**4
    return vs
