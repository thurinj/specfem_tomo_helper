#!/usr/bin/env python

from specfem_tomo_helper.projection import vp2rho, vp2vs, vs2vp, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, trilinear_interpolator, write_tomo_file
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/julienthurin/Documents/Dev/IRIS_Model_converter/CSEM_Europe/csem-europe-2019.12.01.nc'
dx = 10000
dy = 10000
z_min = -250  # in km
z_max = 0  # in km
dz = 10000 # in m

# load netCDF model
nc_model = Nc_model(path)
# extract coordinates
lon, lat, depth = nc_model.load_coordinates()
# vsv = nc_model.load_variable('vsv', fill_nan=True)
vsv = nc_model.load_variable('vsv', fill_nan=False)
# vsh = nc_model.load_variable('vsh', fill_nan=True)

# define pyproj custom projection
myProj = define_utm_projection(11, 'N')

# Second mode is here, with graphical area selection for interpolation.
# The graphical selection tool takes an initial projection as argument. It can be modified using the GUI if it was not correctly specified.
holder = maptool(nc_model, myProj)
interpolator = trilinear_interpolator(nc_model, holder.projection)
interpolator.interpolation_parameters(holder.extent[0], holder.extent[1], dx,
                                      holder.extent[2], holder.extent[3], dy,
                                      z_min, z_max, dz)
tomo = interpolator.interpolate([vsv])

coordinates = tomo[:,0:3]
vsv = tomo[:,3]
vp = vs2vp(vsv)
rho = vp2rho(vp)
param = np.vstack((vp, vsv, rho)).T
tomo = np.hstack((coordinates, param))

write_tomo_file(tomo, interpolator, './')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tomo[:,0], tomo[:,1], tomo[:,2], c=tomo[:,3])
plt.show()
