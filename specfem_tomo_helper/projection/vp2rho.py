#!/usr/bin/env python

def vp2rho(vp):
    """ Math function definitions to convert vp to rho from Brocher T. 2005, https://doi.org/10.1785/0120050077

    Parameters
    ----------

    vp : float
        P-wave velocity (km/s)

    """

    if any(vp) >= 15:
        raise AssertionError('vp values must be in km/s')

    rho = 1.6612*vp - 0.4721*vp**2 + 0.0671*vp**3 - 0.0043*vp**4 + 0.000106*vp**5
    return rho*1000
