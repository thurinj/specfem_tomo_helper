import setuptools
from setuptools import find_packages, setup, Extension

setuptools.setup(
    name="specfem_tomo_helper", # Replace with your own username
    version="0.1",
    license="GPLv3",
    description="Helper package for specfem3D tomography files",
    author="Julien Thurin",
    author_email="jthurin@alaska.edu",
    url="https://github.com/thurinj/specfem_tomo_helper",
    packages=find_packages(),
    classifiers=[
    # complete classifier list:
    # http://pypi.python.org/pypi?%3Aaction=list_classifiers
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GPLv3",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Unix",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics"
    ],

    keywords=[
        "seismology"
    ],
    python_requires='~=3.6',
    install_requires=[
        "numpy", "scipy", "pyproj",
        "pandas", "cartopy", "netCDF4"
    ],
)
