# Real case of teleseismic tomography in Thailand and adjacent areas

This is a real case to invert common-source differential arrival times for velocity heterogeneity in Thailand and adjacent areas

Reference:

[1] J. Chen, S. Wu, M. Xu, M. Nagaso, J. Yao, K. Wang, T. Li, Y. Bai, and P. Tong, Adjoint-state teleseismic traveltime tomography: method and application to Thailand in Indochina Peninsula. J.Geophys. Res. Solid Earth, 128(2023), e2023JB027348.
https://doi.org/10.1029/2023JB027348

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


