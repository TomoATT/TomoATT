# Real case of regional tomography in central California near Parkfield

This is a real case to invert traveltimes for velocity heterogeneity and azimuthal anisotropy in central California near Parkfield

Reference:

[1] J. Chen, G. Chen, M. Nagaso, and P. Tong, Adjoint-state traveltime tomography for azimuthally anisotropic media in spherical coordinates. Geophys. J. Int., 234 (2023), pp. 712-736. 
https://doi.org/10.1093/gji/ggad093

[2] J. Chen, M. Nagaso, M. Xu, and P. Tong, TomoATT: An open-source package for Eikonal equation-based adjoint-state traveltime tomography for seismic velocity and azimuthal anisotropy, submitted.
https://doi.org/10.48550/arXiv.2412.00031


Python modules are required to initiate the inversion and to plot final results:
- h5py
- PyTomoAT
- Pygmt
- gmt

Run this example:

1. Run bash script `bash run_this_example.sh` to execute the test.

2. After inversion, run `plot_output.py` to plot the results.

The imaging results:

![](img/imaging_result.jpg)


