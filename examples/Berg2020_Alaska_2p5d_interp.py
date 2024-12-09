#!/usr/bin/env python

from specfem_tomo_helper.projection import vp2rho, vp2vs, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, linear_interpolator1d2d, write_tomo_file
import subprocess
import os
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

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
vp = nc_model.load_variable('vpfinal', fill_nan=True)
vs = nc_model.load_variable('vsfinal', fill_nan=True)
rho = nc_model.load_variable('rhofinal', fill_nan=True)
# flip the model 
vp.values=np.flipud(vp.values)
vs.values=np.flipud(vs.values)
rho.values=np.flipud(rho.values*1000)
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
    # latitude_min                    = 5862104.0
    # latitude_max                    = 8462104.0
    # longitude_min                   = -928831.7
    # longitude_max                   = 1671168.3
    # Initialize interpolator with model and UTM projection
    interpolator = linear_interpolator1d2d(nc_model, myProj)
    interpolator.interpolation_parameters(longitude_min, longitude_max, dx,
                                          latitude_min, latitude_max, dy,
                                          z_min, z_max, dz)

if mode == 2:
    # Second mode with graphical area selection for interpolation.
    # The GUI window might freeze during the interpolation, instead of closing. Don't panic!
    gui_parameters = maptool(nc_model, myProj)
    interpolator = linear_interpolator1d2d(nc_model, myProj)
    interpolator.interpolation_parameters(
        gui_parameters.extent[0], gui_parameters.extent[1], dx,
        gui_parameters.extent[2], gui_parameters.extent[3], dy,
        z_min, z_max, dz)

# Compute the tomography array
tomo = interpolator.interpolate([vp, vs, rho])
# 2.5d interpolation
zdir_intep=1
if zdir_intep==1:
    #process the layers
    nlay=len(interpolator.x_interp_coordinates)*len(interpolator.y_interp_coordinates)
    dep_int=np.unique(interpolator.z_interp_coordinates)*1000
    tomo_lay=np.empty((nlay,6,len(dep_int)))
    for i,dep,in enumerate(dep_int):
        tomo_lay[:,:,i]=tomo[i*nlay:(i+1)*nlay,:]
    tomo_lay=np.swapaxes(tomo_lay,1,2)
    #interp with new depth
    z_interp=np.flip(np.arange(z_max*1e3, z_min*1e3, -dz))
    tomo=np.empty((nlay,len(z_interp),6))
    f_dep_interp=interp1d(dep_int,tomo_lay,axis=1)
    for i,zint in enumerate(z_interp):
        tomo[:,i,:]=f_dep_interp(zint)
    plot_flag=0
    if plot_flag==1:
        dep_ind=np.where(z_interp==-6400)[0][0]
        plt.figure()
        plt.scatter(tomo[:,dep_ind,0], tomo[:,dep_ind,1], c=tomo[:,dep_ind,4],cmap='RdBu')
        #plt.xticks(lon)
        plt.colorbar(label='vsfinal(km/s)')
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        rectangle = plt.Rectangle((294682.2,7103782.0),200000,100000, fc='none',ec='k')
        plt.gca().add_patch(rectangle)
        rectangle = plt.Rectangle((194682.2,7028782.0),400000,300000, fc='none',ec='k')
        plt.gca().add_patch(rectangle)
        plt.title('depth '+str(z_interp[dep_ind]/1000)+' km')
    #reshape the interped tomo model for using write_tomo_file
    tomo= tomo.reshape(-1, tomo.shape[-1],order='F')
    interpolator.z_interp_coordinates=z_interp
# Write the tomography_file.xyz in "./" directory. It uses the tomography array and the interpolator parameters to produce the HEADER.
write_tomo_file(tomo, interpolator, './')
