# specfem_tomo_helper

`specfem_tomo_helper` is a program to generate the external tomography files required by specfem3D, from netCDF4 models available at [IRIS EMC](http://ds.iris.edu/ds/products/emc/). The intent of this code is to streamline the generation of these tomographic files.
The Area of Interest (AoI) of these tomographic files can be selected by direct input (in UTM coordinates as in specfem), or with a graphical user interface to select the AoI.

>***:warning: This package is work in progress and NOT YET READY FOR PRODUCTIVE USE.***:

## Installation
### Installing Python and Dependencies

If you are well versed in Python, then the installation should not prove an issue; otherwise, please follow the advice here and download the [Anaconda Python distribution](https://www.anaconda.com/products/individual) for your system. The code is working with the current version of Python (3.9 at the moment) but should be working with most of the older 3.x versions.

After downloading and installing Anaconda, update it with

```
$ conda update conda
```
Then create a new environment. You don’t have to, but using a new environment grants a clean separation between Python installations.

```
 $ conda create --name sth
 ```


 This will create a new conda environment based on the latest Python released (3.9) named `sth`. Activate it with

 ```
 $ conda activate sth
 ```

Remember to activate it every time you want to use `specfem_tomo_helper`. You can quit the `sth` environment with `conda deactivate`.

 Now install pip and cartopy with
 ```
 $ conda install pip cartopy
 ```

 ### Installing specfem_tomo_helper

 Now that the environment has been set up, you can download and install `specfem_tomo_helper` with

 ```
 $ git clone https://github.com/thurinj/specfem_tomo_helper.git
 $ cd specfem_tomo_helper/
 $ pip install -e .
 ```

Pip should install a few more dependencies while installing `specfem_tomo_helper` (namely `numpy`, `scipy`, `pyproj`, `pandas` ,and `netCDF4`).


### Running specfem_tomo_helper
From here, you can move to the examples folder, where you can try 2 examples for 2D and 3D interpolation.
>***:warning: Instructions for the 2D example will be provided soon.***:

The first should be used for very dense models where trilinear interpolation would be too expensive to use. The included example is the SCEC south Californian model [EMC-CVM_H_v15_1](http://ds.iris.edu/ds/products/emc-cvm_h_v15_1/). This example performs the interpolation over a cartesian grid of each depth slice in the model. It could benefit from parallelization.

The latter example is the option to favor for sparse models. It was tested on 1° lat/lon and 0.5° lat/lon resolution models and proved to be perfectly appropriate.
The test model included is the [CSEM_Europe_2019.12.01](http://ds.iris.edu/ds/products/emc-csem_europe/).

You can run the trilinear interpolator code with
```
$ python example_specfem_tomo_helper_3D.py
```
