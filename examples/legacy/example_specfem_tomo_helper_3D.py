#!/usr/bin/env python

from specfem_tomo_helper.projection import vp2rho, vp2vs, vs2vp, define_utm_projection
from specfem_tomo_helper.fileimport import Nc_model
from specfem_tomo_helper.utils import maptool, trilinear_interpolator, write_tomo_file, TopographyProcessor, MeshProcessor
from specfem_tomo_helper.utils.download import download_if_missing
import matplotlib.pyplot as plt
import numpy as np
import os

# --- Model and grid parameters ---
path = '../data/csem-europe-2019.12.01.nc'
model_url = 'https://ds.iris.edu/files/products/emc/emc-files/csem-europe-2019.12.01.nc'
if not os.path.isfile(path):
    download_if_missing(path, model_url)

# Grid spacing (in meters)
dx = 5000
dy = 5000
# Depth range (in km, negative down)
z_min = -250  # shallowest depth (top)
z_max = 0     # deepest depth (bottom)
dz = 5000     # vertical grid spacing (in meters)

output_dir = './topography_CSEM'  # Set output directory for mesh/topography

# --- Loading, setting up and interpolation ... ---
# Load the netCDF model
nc_model = Nc_model(path)
# Grab the coordinate arrays
lon, lat, depth = nc_model.load_coordinates()
# Load the model parameter you want to interpolate (here: vsv)
vsv = nc_model.load_variable('vsv', fill_nan='vertical')

# Set up the map projection (UTM zone 34N by default, tweak if needed)
myProj = define_utm_projection(34, 'N')
# Use the GUI to select the interpolation area (projection can be changed in the GUI)
gui_parameters = maptool(nc_model, myProj)
# Set up the trilinear interpolator with the chosen projection
interpolator = trilinear_interpolator(nc_model, gui_parameters.projection)
# Define the interpolation grid using the GUI-selected area and your spacing
interpolator.interpolation_parameters(gui_parameters.extent[0], gui_parameters.extent[1], dx,
                                      gui_parameters.extent[2], gui_parameters.extent[3], dy,
                                      z_min, z_max, dz)
# Interpolate the model (returns [x, y, z, vsv])
tomo = interpolator.interpolate([vsv])

# Convert Vs to Vp and Rho using empirical relationships
coordinates = tomo[:, 0:3]
vsv = tomo[:, 3]
vp = vs2vp(vsv)  # Vs to Vp
rho = vp2rho(vp)  # Vp to Rho
param = np.vstack((vp, vsv, rho)).T  # Stack Vp, Vs, Rho for output

# Combine coordinates and parameters into final tomography array
# Format: [lon, lat, depth, vp, vs, rho] (needed for SPECFEM)
tomo = np.hstack((coordinates, param))
# Write the tomography file (outputs to current directory)
write_tomo_file(tomo, interpolator, output_dir)

# The following steps are entirely optional and are only required
# if you want to generate a mesh and interfaces for SPECFEM3D to
# use the tomography model. If you already have a mesh and know what you're doing
# you can skip this part.

# --- Mesh_Par_File and interfaces + topography generation ---
# Set mesh parameters (these control the SPECFEM3D mesh, not the tomography grid)
max_depth = 250.0  # in km (total mesh depth)

# Set up the mesh processor and suggest horizontal mesh configs
mesh = MeshProcessor(interpolated_tomography=tomo, projection=gui_parameters.projection)
mesh.suggest_horizontal_configs(dx_target_km=5.0, max_cpu=64, mode='choice', n_doublings=2)  # mesh element size target (in km)

# Generate the mesh config, including vertical doubling layers (depths in km, positive down)
doubling_layers_km = [31, 80]  # in km, positive down
# Convert to negative for SPECFEM3D convention
neg_doubling_layers_km = [-dl for dl in doubling_layers_km]
doubling_layers_m = [dl * 1000.0 for dl in neg_doubling_layers_km]
mesh.generate_dynamic_mesh_config(dz_target_km=5.0, max_depth=max_depth, doubling_layers=doubling_layers_m)

# Topography and interface generation using the mesh config
topo = TopographyProcessor(interpolator, gui_parameters.projection, save_dir=output_dir, mesh_processor=mesh)
topo.write_all_outputs(slope_thresholds=[10, 15, 20])

# Write the mesh parameter file for SPECFEM
mesh.write_parfile_easy(output_dir=output_dir)

# --- End of Mesh_Par_File and interfaces generation ---

# --- Visual check of the interpolated model ---

# Get bounds for plotting the outer shell
x_bounds = {np.min(tomo[:, 0]), np.max(tomo[:, 0])}
y_bounds = {np.min(tomo[:, 1]), np.max(tomo[:, 1])}
z_bounds = {np.min(tomo[:, 2]), np.max(tomo[:, 2])}

# Filter the outer shell of the model for plotting
outer_shell = tomo[np.isin(tomo[:, 0], list(x_bounds)) |
                   np.isin(tomo[:, 1], list(y_bounds)) |
                   np.isin(tomo[:, 2], list(z_bounds))]

# Display the filtered model (color by Vp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(outer_shell[:, 0], outer_shell[:, 1], outer_shell[:, 2], c=outer_shell[:, 3])
plt.show()
