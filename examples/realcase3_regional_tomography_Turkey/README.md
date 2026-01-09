# Real case of regional tomography in central California near Parkfield

This is a real case to invert traveltimes for velocity heterogeneity and azimuthal anisotropy around the Eastern Anatolian Fault in Turkey.

Reference:

[1] J. Chen, M. Xu, Y. Bai, S. Wu, X. Xiao, S. Hao, M. Nagaso, H. Yang, and P. Tong, High normal stress promoted supershear rupture during the 2023 Mw 7.8 Kahramanmaraş earthquake. Nat. Geosci.(2026).
https://doi.org/10.1038/s41561-025-01893-z


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


