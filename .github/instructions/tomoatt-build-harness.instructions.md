---
description: "Use when doing harness engineering, CMake/CI work, shell-script updates, or reproducing TomoATT build and test failures. Covers MPI/HDF5 dependencies, clean build directories, and the regression harness flow."
applyTo:
  - CMakeLists.txt
  - .github/workflows/**
  - .github/scripts/**
  - tests/**
  - test/**
---
# TomoATT Build And Test Harness

- `CMakeLists.txt` expects MPI and attempts to use parallel HDF5. `yaml-cpp` can come from the system install or the vendored fallback under `external_libs/yaml-cpp`.
- Treat local build directories as disposable. `build_test/` and `build_test2/` can carry stale absolute Homebrew paths for MPI or HDF5, so use a fresh configure directory when reproducing failures instead of trusting an old cache.
- On macOS local repro, prefer configuring with the active MPI wrappers and current Homebrew prefixes for `open-mpi` and `hdf5-mpi` rather than hard-coded cellar paths.
- `BUILD_TESTING` only adds the small executable tests under `tests/`. The more representative regression harness is `.github/scripts/run_test.sh`, which drives `test/inversion_small/` and validates the generated final model.
- Keep the compiler matrix intact across Clang, GCC, and Intel oneAPI paths in `CMakeLists.txt`. Changes that only work on one compiler branch are likely to break CI.
- If dependency detection changes, fail configuration explicitly for unsupported states instead of printing a message and continuing into a partial build.
- Use `.github/scripts/run_install.sh` and `.github/scripts/run_test.sh` as the canonical CI harness behavior before adding new ad hoc setup or validation scripts.
- Runtime tests depend on Python-side tooling such as `pta`, `numpy`, `h5py`, and `pytomoatt`, so build-only fixes and runtime-harness fixes are often different problems and should be diagnosed separately.