cmake_minimum_required(VERSION 3.10)

# set the project name
set(CMAKE_PROJECT_NAME "TOMOATT")
project(${CMAKE_PROJECT_NAME} VERSION 1.0.2 LANGUAGES C CXX )

# set install directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# check debug or release
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Build type: Debug")
else()
  message(STATUS "Build type: Release")
endif()

# check compiler type
message(STATUS "Compiler type: ${CMAKE_CXX_COMPILER_ID}")
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CXX_ADDITIONAL_FLAGS "-Wall -pedantic -g -O0")
  else()
    set(CXX_ADDITIONAL_FLAGS "-Wall -pedantic -O3 -funroll-loops -ffast-math -ftree-vectorize")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CXX_ADDITIONAL_FLAGS "-Wall -pedantic -g -O0 -lm -lstdc++fs")
  else()
    set(CXX_ADDITIONAL_FLAGS "-Wall -pedantic -lm -O3 -funroll-loops -ffast-math -ftree-vectorize -lstdc++fs")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CXX_ADDITIONAL_FLAGS "-diag-disable=10441,2012,2015,2017,2047,2304,2305,3868,10193,10315,11074,11076 -Wall -pedantic -g -O0 -lm -lstdc++fs")
  else()
    set(CXX_ADDITIONAL_FLAGS "-diag-disable=10441,2012,2015,2017,2047,2304,2305,3868,10193,10315,11074,11076 -Wall -pedantic -O3 -funroll-loops -ffast-math -lm  -ftree-vectorize -lstdc++fs")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Fujitsu")
  MESSAGE(FATAL_ERROR "Fujitsu trad compiler is not supported. Please use clang mode.")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "FujitsuClang")
  MESSAGE("Compiler type: FujitsuClang")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CXX_ADDITIONAL_FLAGS "-Nclang -g -O0 -std=c++17 -mcpu=a64fx+sve -march=armv8-a+sve")
  else()
    set(CXX_ADDITIONAL_FLAGS "-Nclang -Ofast -std=c++17 -mcpu=a64fx+sve -march=armv8-a+sve")
  endif()
else()
  MESSAGE(FATAL_ERROR "Compiler type: Unknown")
endif()
# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

set(BUILD_TESTING OFF)
set(Force_download_externel_libs OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_ADDITIONAL_FLAGS}")

# find installed MPI
message(STATUS "Running CMAKE FindMPI.cmake...")
find_package(MPI)

message(STATUS "MPI_FOUND: ${MPI_FOUND}")
message(STATUS "MPI_VERSION: ${MPI_VERSION}")

# find openmp ## WE DO NOT USE OPENMP BUT KEEP THIS FOR FUTURE USE
###find_package(OpenMP)
###if(OPENMP_FOUND)
###  message(STATUS "OpenMP found")
###  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
###  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
###  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
###  add_definitions(-DUSE_OMP)
###endif()

# find HDF5 parallel TODO: check parallel io is enable or not
message(STATUS "Running CMAKE FindHDF5.cmake...")
# set parallel HDF5 default
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5)
if(HDF5_FOUND)
  message(STATUS "HDF5_FOUND: ${HDF5_FOUND}")
  add_definitions(-DUSE_HDF5)
  # check if HD5 PARALLEL is available
  if(HDF5_IS_PARALLEL)
    message(STATUS "HDF5 parallel is available.")
  else()
    message(FATAL "TomoATT requires HDF5 compiled with parallel IO option.")
  endif()
endif()

# use collective io for HDF5: should be faster than independent io,
#but there will be a memory overhead and may not work on some systems.
#If you have a problem, please comment out this line.
add_definitions(-DUSE_HDF5_IO_COLLECTIVE)

# precision setting (uncomment for single precision. default is double precision.)
#add_definitions(-DSINGLE_PRECISION)

# use SIMD (SSE/AVX/AVX2/AVX512) for vectorization, which is faster than the default but use a little more memory.
if (USE_SIMD)
  message(STATUS "TomoATT is compiled with SIMD.")
  add_definitions(-DUSE_SIMD)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mfma")
endif()

# find cuda package if USE_CUDA is defined
if(USE_CUDA)
  message(STATUS "Running CMAKE FindCUDA.cmake...")
  enable_language(CUDA)
  find_package(CUDA)
else()
  message(STATUS "TomoATT is compiled without cuda, because -DUSE_CUDA=True is not defined")
endif()

if(CUDA_FOUND) # TODO : add HIP here in the future
  message(STATUS "CUDA_FOUND: ${CUDA_FOUND}")
  add_definitions(-DUSE_CUDA)
  set(CUDA_LIBRARY_NAME "TOMOATT_CUDA")

  # list of source and header files for cuda
  file(GLOB SOURCES_CUDA "cuda/*.cu")
  file(GLOB HEADERS_CUDA "cuda/*.cuh")

  # cuda flag
  #
  # for production
  set(CMAKE_CUDA_FLAGS "-fPIC -O3 -use_fast_math -extra-device-vectorization -gencode arch=compute_61,code=sm_61")
  #
  # for debugging
  #set(CMAKE_CUDA_FLAGS "-fPIC -lineinfo -g -G -O0 -gencode arch=compute_61,code=sm_61")
  set(CMAKE_CUDA_STANDARD "11")
  message(STATUS, "TomoATT will be compiled with cuda.")
else()
  message(STATUS "TomoATT will be compiled without cuda, because cuda is not found or -DUSE_CUDA=True was not specified.")
endif()

# synchronize the adjuscent ghost layers for each direction oft the sweep
# which is more frequent than the referred paper but necessary
add_definitions(-DFREQ_SYNC_GHOST)

# find BLAS # WE DO NOT USE BLAS BUT KEEP THIS FOR FUTURE USE
#find_package(BLAS)
#if(BLAS_FOUND)
#  message(STATUS "BLAS_FOUND: ${BLAS_FOUND} at ${BLAS_LIBRARIES}, ${BLAS_INCLUDE_DIRS}")
#  add_definitions(-DUSE_BLAS)
#  find_path(BLAS_INCLUDE_DIRS cblas.h
#    /usr/include
#    /usr/local/include
#    /usr/local/include/openblas)
#endif()

# submodules
# yaml parser
find_package(yaml-cpp 0.8 QUIET)
if (yaml-cpp_FOUND AND NOT ${Force_download_externel_libs})
  message(STATUS "yaml-cpp found")
  message(STATUS "YAML_CPP_INCLUDE_DIR: ${YAML_CPP_INCLUDE_DIR}")
  message(STATUS "YAML_CPP_LIBRARIES: ${YAML_CPP_LIBRARIES}")
else()
  message(STATUS "yaml-cpp not found. Using external_libs/yaml-cpp ...")
  add_subdirectory(external_libs)
  set(YAML_CPP_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/external_libs/yaml-cpp/include)
  set(YAML_CPP_LIBRARIES yaml-cpp)
endif()

# add include directory
include_directories(include cuda)

configure_file(
  ${PROJECT_SOURCE_DIR}/include/version.h.in
  ${PROJECT_SOURCE_DIR}/include/version.h
)

# list of source files
file(GLOB SOURCES "src/*.cpp")
if(CUDA_FOUND)
  file(GLOB HEADERS "include/*.h" "cuda/*.cuh")
else()
  file(GLOB HEADERS "include/*.h")
endif()
file(GLOB SOURCES_EXT_XML "external_libs/tinyxml2/*.cpp")

# compile cuda code
if (CUDA_FOUND)
  include_directories(${CUDA_INCLUDE_DIRS})
  add_library(${CUDA_LIBRARY_NAME} STATIC ${SOURCES_CUDA} ${HEADERS_CUDA} )
  target_include_directories(${CUDA_LIBRARY_NAME} PUBLIC ${YAML_CPP_INCLUDE_DIR})
  target_include_directories(${CUDA_LIBRARY_NAME} PUBLIC ${HDF5_INCLUDE_DIRS})
  target_include_directories(${CUDA_LIBRARY_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/external_libs/tinyxml2)
endif()


#
# compile the executables
# all the files with the name *.cxx is compiled as an executable
#
#file( GLOB APP_SOURCES src/*.cxx )

# add one by one
set(APP_SOURCES
  src/TOMOATT.cxx
  #src/TOMOATT_solver_only.cxx
  #src/TOMOATT_2d_precalc.cxx
  #src/SrcRecWeight.cxx
  )

# if BUILD_TESTING is defined, make APP_TEST and append to APP_SOURCES list
if(BUILD_TESTING)
  # use all the cxx files in tests/ directory
  file( GLOB APP_TEST tests/*.cxx)
  # or you can specify the test files one by one
  #set(APP_TEST
  #  tests/read_write_srcrec.cxx
  #)
  list(APPEND APP_SOURCES ${APP_TEST})
endif()

foreach( execsourcefile ${APP_SOURCES} )
  # get app name from file name
  get_filename_component(EXEC_NAME ${execsourcefile} NAME_WE)

  # add the executable
  add_executable(${EXEC_NAME} ${execsourcefile} ${SOURCES} ${HEADERS} ${SOURCES_EXT_XML})

  # set include path
  target_include_directories(${EXEC_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/cuda
            ${PROJECT_SOURCE_DIR}/external_libs/tinyxml2)

  # link mpi
  target_link_libraries(${EXEC_NAME} PUBLIC MPI::MPI_CXX)

  # link yaml-app:
  target_link_libraries(${EXEC_NAME} PUBLIC ${YAML_CPP_LIBRARIES})
  target_include_directories(${EXEC_NAME} PUBLIC ${YAML_CPP_INCLUDE_DIR})

  # link HDF5
  if(HDF5_FOUND)
    target_link_libraries(${EXEC_NAME} PUBLIC ${HDF5_LIBRARIES})
    target_include_directories(${EXEC_NAME} PUBLIC ${HDF5_INCLUDE_DIRS})
  endif()

  # link blas
  if(BLAS_FOUND)
    target_link_libraries(${EXEC_NAME} PUBLIC ${BLAS_LIBRARIES})
    target_include_directories(${EXEC_NAME} PUBLIC ${BLAS_INCLUDE_DIRS})
  endif()

  # link cuda
  if (CUDA_FOUND)

    #set_target_properties(${CUDA_LIBRARY_NAME} PROPERTIES CUDA_ARCHITECTURES "35;50;72")
    set_target_properties(${CUDA_LIBRARY_NAME} PROPERTIES CUDA_ARCHITECTURES "61")
    set_property(TARGET ${CUDA_LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES 61)

    target_link_libraries(${EXEC_NAME} PRIVATE ${CUDA_LIBRARY_NAME})
    target_link_libraries(${CUDA_LIBRARY_NAME} PUBLIC MPI::MPI_CXX)
    target_link_libraries(${CUDA_LIBRARY_NAME} PUBLIC yaml-cpp)
    target_link_libraries(${CUDA_LIBRARY_NAME} PUBLIC ${HDF5_LIBRARIES})

  endif()


endforeach( execsourcefile ${APP_SOURCES} )

# install
install(TARGETS TOMOATT DESTINATION bin)

# test

# We check if this is the main file
# you don't usually want users of your library to
# execute tests as part of their build
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR} AND BUILD_TESTING)
    include(CTest)

    # loop over APP_TEST
    foreach( execsourcefile ${APP_TEST} )
      add_test(NAME ${execsourcefile} COMMAND ${EXEC_NAME} ${execsourcefile})
    endforeach( execsourcefile ${APP_TEST} )
endif ()

enable_testing()
