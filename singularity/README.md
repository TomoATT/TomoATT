# Singularity image of TomoATT project

[Singularity](https://docs.sylabs.io/guides/3.10/user-guide/introduction.html) is a container technology that prepare a pre-compiled project's executable with an environment required dependencies (this is called a container image).
This is very useful for the users who are not familiar with the process of compiling the project's executable and installing the required dependencies, and also for the users who are not allowed to install the required dependencies on their system.

Once the container image is prepared, this image can be uploaded on singularity-hub.org, and the users can download the image and run the project's executable on their system instantly.

[Docker](https://www.docker.com/) is another container technology similar to Singularity, which is rather popular in the Information Technology (IT) community. However, Docker may not access the MPI library on the native system, which Singularity can do, and this cause degradation of the performance of the application.

Therefore, Singularity is more popular and suitable in scientific computing community and often used in the HPC environment.


This directry includes a singularity receipe file (*.def) to build a container image of TomoATT project.
To build the container image,
``` bash
sudo singularity build tomoatt.sif tomoatt.def
```

To run the TomoATT executable on the container image,
``` bash
singularity run tomoatt.sif TOMOATT
```

To run the TomoATT executable on the container image with native MPI library,
``` bash
mpirun -n 8 singularity run tomoatt.sif TOMOATT -i input_params.yml
```
For utilizing the native MPI library, the version number of OpenMPI should be the same between the native system and the container image.
The two variables in tomoatt.def file need to be modified for this purpose.
``` bash
    export OMPI_VERSOIN_MAJOR=3.0
    export OMPI_VERSION_FULL=3.0.1
```


# For running the container image on the HPC with old kernel environment

Some HPC systems have old kernel environment, e.g. CentOS 6, RHEL 6, and Ubuntu 16.04.
In this case, the container image cannot be run on the HPC system. (The error message is "FATAL: kernel too old")
To create a container image that can be run on the HPC system, this image needs to be built on the old kernel environment as well.
For this purpose, we use [docker](https://www.docker.com/) for virtualy creating an old kernel environment, and then build the container image on it.

To build the container image on the old kernel environment,
``` bash
docker build -t singularity_on_centos6 -f Dockerfile_singularity_on_centos6 .
```

To build the container image on the old kernel environment,
``` bash
docker run --privileged -v "(full path to)/TomoATT:/TomoATT" singularity_on_centos6 singularity build /TomoATT/singularity/tomoatt.sif /TomoATT/singularity/tomoatt.def
```