#!/usr/bin/env python

from specfem_tomo_helper.projection import vp2rho, vp2vs, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, bilinear_interpolator, write_tomo_file

path = '../data/SCEC_socal/SCEC-CVM-H-v15-1-n4.nc'

# Two modes examples here. First is direct input.
latitude_min = 3537939.0
latitude_max = 3937939.0
dy = 1000  # in m
longitude_min = 330704.1
longitude_max = 830704.1
dx = 1000  # in m
z_min = -50  # in km
z_max = 4  # in km
# In the 2D interpolation scheme, each depth value of the model is used as depth value.

# load netCDF model
nc_model = Nc_model(path)
# extract coordinates
lon, lat, depth = nc_model.load_coordinates()
# fill_nan is set to True here, as the shallow layers of the model contain nan values
vp = nc_model.load_variable('vp', fill_nan=True)
vs = nc_model.load_variable('vs', fill_nan=True)
rho = nc_model.load_variable('rho', fill_nan=True)

# define pyproj custom projection
myProj = define_utm_projection(11, 'N')

interpolator = bilinear_interpolator(nc_model, myProj)
interpolator.interpolation_parameters(longitude_min, longitude_max, dx,
                                      latitude_min, latitude_max, dy,
                                      z_min, z_max)
tomo = interpolator.interpolate([vp, vs, rho])


# Second mode is here, with graphical area selection for interpolation.
param_holder = maptool(nc_model, myProj)
interpolator = bilinear_interpolator(nc_model, myProj)
interpolator.interpolation_parameters(
    param_holder.extent[0], param_holder.extent[1], dx,
    param_holder.extent[2], param_holder.extent[3], dy,
    z_min, z_max)

tomo = interpolator.interpolate([vp, vs, rho])

write_tomo_file(tomo, interpolator, './')
