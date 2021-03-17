import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="specfem_tomo_helper", # Replace with your own username
    version="0.0.1",
    author="Julien Thurin",
    author_email="jthurin@alaska.edu",
    description="Helper package for specfem3D tomography files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/thurinj/specfem_tomo_helper",
    project_urls={
        "Bug Tracker": "https://github.com/thurinj/specfem_tomo_helper/issues",
    },
    classifiers=[
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: GPLv3",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Unix",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Physics"
    ],
    package_dir={"": "specfem_tomo_helper"},
    packages=setuptools.find_packages(where="specfem_tomo_helper"),
    python_requires=">=3.8",
)
