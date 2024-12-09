#!/usr/bin/env python
# example provided for 2.5d interpolation for 
from specfem_tomo_helper.projection import vp2rho, vp2vs, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import write_tomo_file,linear_interpolator1d2d
import subprocess
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from specfem_tomo_helper.utils.pyqt_gui import MainWindow
from PyQt5.QtWidgets import QApplication

path = '../data/Alaska.JointInversion-RF+Vph+HV-1.Berg.2020-nc4.nc'

if not os.path.isfile(path):
    subprocess.call(['wget', '-P', '../data/', 'http://ds.iris.edu/files/products/emc/emc-files/Alaska.JointInversion-RF+Vph+HV-1.Berg.2020-nc4.nc'])

# Mandatory inputs:
dy = 3000  # in m
dx = 3000  # in m
# In the 2D interpolation scheme, each depth value of the model is used as depth value. Therefore there is no dz parameters.
z_min = -34 # in km
z_max = -8 # in km
dz=3000 # in meters
# load netCDF model
nc_model = Nc_model(path)
# extract coordinates
lon, lat, depth = nc_model.load_coordinates()
# Load the 3 model parameters from the netCDF model
vp = nc_model.load_variable('vpfinal', fill_nan='vertical')
vs = nc_model.load_variable('vsfinal', fill_nan='vertical')
rho = nc_model.load_variable('rhofinal', fill_nan='vertical')

# fill_nan is set to True here, as the shallow layers of the model contain nan values

# define pyproj custom projection. 
myProj = define_utm_projection(6, 'N')

# Here are two example modes possible (1 or 2)
# 1 for direct input, 2 for GUI map input
mode = 1
if mode == 1:
    # Direct input akin to specfem Mesh_Par_File
    latitude_min                    = 7003782.0
    latitude_max                    = 7353782.0
    longitude_min                   = 194682.2
    longitude_max                   = 594682.2
    # Initialize interpolator with model and UTM projection
    interpolator = linear_interpolator1d2d(nc_model, myProj)
    interpolator.interpolation_parameters(longitude_min, longitude_max, dx,
                                          latitude_min, latitude_max, dy,
                                          z_min, z_max,dz)

if mode == 2:
    # Second mode with graphical area selection for interpolation.
    # The GUI window might freeze during the interpolation, instead of closing. Don't panic!
    app = QApplication([])
    gui_parameters = MainWindow()
    gui_parameters.show()
    app.exec_()
    interpolator = linear_interpolator1d2d(nc_model, myProj)
    interpolator.interpolation_parameters(
        gui_parameters.extent[0], gui_parameters.extent[1], dx,
        gui_parameters.extent[2], gui_parameters.extent[3], dy,
        z_min, z_max,dz)

# Compute the tomography array and perform 2.5d interpolation
tomo = interpolator.interpolate([vp, vs, rho])



# Write the tomography_file.xyz in "./" directory. It uses the tomography array and the interpolator parameters to produce the HEADER.
write_tomo_file(tomo, interpolator, './')
