#ifndef CUDA_INITIALIZE_H
#define CUDA_INITIALIZE_H


#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

//#include "config.h"
#include "cuda_constants.cuh"

void get_free_memory(double* free_db, double* used_db, double* total_db) {

    // gets memory usage in byte
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(EXIT_FAILURE);
    }

    *free_db = (double)free_byte ;
    *total_db = (double)total_byte ;
    *used_db = *total_db - *free_db ;
    return;
}


// setup cuda constants and variables by reading device properties
void initialize_cuda(){

    std::cout << "Initializing CUDA..." << std::endl;

    int ncuda_device;
    int device;

    // count number of devices
    cudaGetDeviceCount(&ncuda_device);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount returned error code %d after %d devices\n", err, ncuda_device);
        exit(1);
    }

    if (ncuda_device == 0)
    {
        printf("There is no device supporting CUDA\n");
        exit(1);
    }

    // set the active device
    if (ncuda_device >= 1){
        cudaDeviceReset();

        device = world_rank % ncuda_device;
        cudaSetDevice(device);

        cudaFree(0);

        // check device is set
        cudaGetDevice(&device);
        if (device != world_rank % ncuda_device){
            printf("Error: Could not set device to %d\n", world_rank % ncuda_device);
            exit(1);
        }
    } // end if ncuda_device >= 1

    cudaGetDevice(&device);

    // get device properties
    cudaDeviceProp deviceProp; // in cuda_constants
    cudaGetDeviceProperties(&deviceProp, device);

    // exit if machine has no cuda enable device
    if (deviceProp.major == 9999 && deviceProp.minor == 9999){
        printf("Error: No CUDA device found\n");
        exit(1);
    }

    // print device properties
    char filename[256];

    if (world_rank == 0){
        sprintf(filename, "cuda_device_info.txt");
        FILE *fp = fopen(filename, "w");

        if(fp == NULL){
            printf("Error: Could not open file %s\n", filename);
            exit(1);
        }

        // display device properties
        fprintf(fp,"Device Name = %s\n",deviceProp.name);
        fprintf(fp,"memory:\n");
        fprintf(fp,"  totalGlobalMem (in MB): %f\n",(unsigned long) deviceProp.totalGlobalMem / (1024.f * 1024.f));
        fprintf(fp,"  totalGlobalMem (in GB): %f\n",(unsigned long) deviceProp.totalGlobalMem / (1024.f * 1024.f * 1024.f));
        fprintf(fp,"  totalConstMem (in bytes): %lu\n",(unsigned long) deviceProp.totalConstMem);
        fprintf(fp,"  Maximum 1D texture size (in bytes): %lu\n",(unsigned long) deviceProp.maxTexture1D);
        fprintf(fp,"  sharedMemPerBlock (in bytes): %lu\n",(unsigned long) deviceProp.sharedMemPerBlock);
        fprintf(fp,"  regsPerBlock (in bytes): %lu\n",(unsigned long) deviceProp.regsPerBlock);
        fprintf(fp,"blocks:\n");
        fprintf(fp,"  Maximum number of threads per block: %d\n",deviceProp.maxThreadsPerBlock);
        fprintf(fp,"  Maximum size of each dimension of a block: %d x %d x %d\n",
                deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
        fprintf(fp,"  Maximum sizes of each dimension of a grid: %d x %d x %d\n",
                deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);
        fprintf(fp,"features:\n");
        fprintf(fp,"  Compute capability of the device = %d.%d\n", deviceProp.major, deviceProp.minor);
        fprintf(fp,"  multiProcessorCount: %d\n",deviceProp.multiProcessorCount);
        // CUDA 13 removed some cudaDeviceProp fields (e.g. deviceOverlap);
        // read them via the attribute API which is stable across versions.
        {
            int _attr = 0;
            if (cudaDeviceGetAttribute(&_attr, cudaDevAttrCanMapHostMemory, device) == cudaSuccess && _attr){
                fprintf(fp,"  canMapHostMemory: TRUE\n");
            }else{
                fprintf(fp,"  canMapHostMemory: FALSE\n");
            }
            _attr = 0;
            if (cudaDeviceGetAttribute(&_attr, cudaDevAttrGpuOverlap, device) == cudaSuccess && _attr){
                fprintf(fp,"  deviceOverlap: TRUE\n");
            }else{
                fprintf(fp,"  deviceOverlap: FALSE\n");
            }
            _attr = 0;
            if (cudaDeviceGetAttribute(&_attr, cudaDevAttrConcurrentKernels, device) == cudaSuccess && _attr){
                fprintf(fp,"  concurrentKernels: TRUE\n");
            }else{
                fprintf(fp,"  concurrentKernels: FALSE\n");
            }
        }
        // outputs initial memory infos via cudaMemGetInfo()
        double free_db,used_db,total_db;
        get_free_memory(&free_db,&used_db,&total_db);
        fprintf(fp,"memory usage:\n");
        fprintf(fp,"  rank %d: GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",myrank,
                used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);

        // closes output file
        fclose(fp);
    }

}


void finalize_cuda(){
    cudaDeviceReset();
}

#endif // CUDA_INITIALIZE_H