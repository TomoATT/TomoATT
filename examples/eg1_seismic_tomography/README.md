# seismic tomography

This is a toy model to invert traveltimes for Vp and anisotropy (Figure 8c.)

Reference:
[1] J. Chen, M. Nagaso, M. Xu, and P. Tong, TomoATT: An open-source package for Eikonal equation-based adjoint-state traveltime tomography for seismic velocity and azimuthal anisotropy, submitted.
https://doi.org/10.48550/arXiv.2412.00031

Python modules are required to initiate the inversion and to plot final results:
- h5py
- PyTomoAT
- Pygmt
- gmt

Run this example:

1. Run bash script `bash run_this_example.sh` to execute the test.

2. After inversion, run `plot_output.py` to plot the results.

The initial and true models:

![](img/model_setting.jpg)

The inversion result:

![](img/model_inv.jpg)


