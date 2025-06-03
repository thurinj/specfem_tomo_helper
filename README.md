# specfem_tomo_helper

`specfem_tomo_helper` is a tool designed to generate external tomography files, mesh files, and topography files required by SPECFEM3D from netCDF4 Earth models available at [IRIS EMC](http://ds.iris.edu/ds/products/emc/). 

The tool provides a streamlined, configuration-file-driven workflow for:

- Tomography file generation -- converts netCDF models into SPECFEM3D-compatible text files, with support for both isotropic and anisotropic media (up to 21 $C_{ij}$ elastic parameters)
- Mesh generation -- generates complete SPECFEM3D mesh configurations, including automated doubling layer placement and optimized CPU distribution
- Topography processing -- produces surface geometry layers to accommodate complex terrain and internal boundaries

The main workflow is based on model re-gridding (via trilinear interpolation), an interactive GUI for region selection, and a streamlined workflow based entirely on editable configuration files. All of these automation features are geared toward begginers. 

## Installation

### Using Conda/Mamba (Recommended)

To set up the environment and install the package using Conda or Mamba, follow these steps:

1. Ensure you have Conda or Mamba installed. If not, download and install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

2. Create the environment and install the package using the `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:

   ```bash
   conda activate specfem_tomo_helper
   ```

This will set up the environment and install all dependencies, including the `specfem_tomo_helper` package.

### Using pip

If you prefer to use pip, follow these steps:

1. Ensure you have Python 3.8 or later installed.

2. Install the required dependencies and the package:

   ```bash
   pip install -e .
   ```

## Quick Start Guide

### 1. Create a Configuration File

Generate a template configuration file to get started:

```bash
# For isotropic models (Vp, Vs, density)
tomo-helper --create-config --output my_config.yaml

# For anisotropic models (21-parameter elastic tensor)
tomo-helper --create-config --anisotropic --output my_anisotropic_config.yaml
```

### 2. Edit the Configuration

Open the generated configuration file and modify the key parameters:

```yaml
# Path to your NetCDF model
data_path: 'path/to/your/model.nc'

# Grid spacing in meters (interval spacing for interpolation)
dx: 5000
dy: 5000  
dz: 5000

# Elevation range in km (negative below sea level)
z_min: -40
z_max: 0

# Variable(s) to interpolate
variable: ['vp', 'vs', 'rho']  # or single variable like 'vs'
# You should be able to see what parameters are defined in the netCDF model
# by using ncdump -h <data_path> beforehand or looking at the metadata on IRIS EMC.

# Area selection (set to null to use interactive GUI)
utm_zone: null
utm_hemisphere: null
extent: null
use_gui: true

```

By default, some fields are set to ensure the Meshing and Topography generation helpers are run, but you can disable them by setting `generate_mesh: false` or `generate_topography: false`.

### 3. Run the Tool

Execute the tool with your configuration file:

```bash
tomo-helper --config my_config.yaml --verbose
```

## Command Line Interface

### Basic Usage

```bash
# Run with configuration file
tomo-helper -c my_config.yaml

# Enable verbose output for detailed logging
tomo-helper -c my_config.yaml --verbose

# Show help and available options
tomo-helper --help

# Create a new configuration file template
tomo-helper --create-config --output my_config.yaml

# Create a new configuration file template for anisotropic models
tomo-helper --create-config --anisotropic --output my_config_ani.yaml

# Specify an output path
tomo-helper --create-config --output /path/to/my_config.yaml

```

## Workflow Overview

The tool follows an automated workflow that generates some of the key files needed for SPECFEM3D:

1. **Model Loading** - Reads netCDF4 Earth models from IRIS EMC
2. **Area Selection** - Interactive GUI or predefined UTM coordinates
3. **3D Interpolation** - Trilinear interpolation onto regular grid
4. **Tomography Files** - Generate SPECFEM3D-compatible `.xyz` files (raw text file)

Optionally, the workflow can include:

5. **Topography Processing** - Extract and smooth surface topography 
6. **Mesh Generation** - Create complete SPECFEM3D parameter files
7. **Visualization** - Generate plots for validation and quality control

### Contribution ideas
The code recieved a major overhaul for version 0.2, yet, there are two features that would make the code more complete and would be great additions. Any help contributing to the code, or specifically with these two items are welcome:

#### Rotated domain
As of now, the regular interpolation grid can only be along the N-E directions. It would be desirable to be able to interpolate onto any rotated cartesian grid, such that the model size can be optimized to save on numerical modeling costs afterward (say, if you need to create a simulation domain that aligns with NE-SW).

#### Multiple tomographic files
The current interpolator only outputs a single tomographic file with regular x, y and z sampling. However, on very dense models and large modeling domains, the `tomography_file.xyz` can rapidly grow in size and become impractical to work with.
To mitigate that issue, the model domain can be split in several vertical chunks with varying spatial sampling (with the chunks becoming coarser with depth), and read as several distinct tomographic files by specfem3D.
This is currently doable by playing with the config files, but it would be nice to have an optional flag that automatically generates coarser models when doubling are used.
