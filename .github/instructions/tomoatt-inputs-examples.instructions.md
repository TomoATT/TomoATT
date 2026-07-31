---
description: "Use when editing TomoATT YAML input parsing, source-receiver handling, or example and regression fixtures. Covers rank-0 parsing, MPI broadcast behavior, units, and keeping parser changes aligned with examples."
applyTo:
  - include/input_params.h
  - src/input_params.cpp
  - examples/**
  - test/**
---
# TomoATT Inputs And Examples

- `InputParams` parses YAML on rank 0 and then broadcasts values through the MPI helper layer. Preserve that contract when adding or refactoring configuration fields.
- The YAML structure is stable around `domain`, `source`, `model`, `parallel`, `output_setting`, `run_mode`, and `model_update`. New fields should fit clearly into that layout instead of introducing redundant top-level keys.
- Be explicit about required versus optional fields. Required keys usually stop execution early, while optional keys silently keep defaults; keep error messages and defaults consistent with the existing style.
- Example cases under `examples/` and regression cases under `test/` are also fixtures for parser behavior. When parser expectations change, update the affected example or test inputs in the same change.
- Source and receiver handling is tightly coupled to `SrcRecInfo`, `src_rec` files, simulation-group broadcasting, and name-to-id ordering. Preserve ordering and broadcast semantics unless the change intentionally updates the data model.
- Keep units obvious and consistent: YAML latitude and longitude are generally entered in degrees and converted internally to radians, while depth values are converted to radius for the solver.
- Test scripts such as `test/inversion_small/run_this_example.sh` mutate YAML values in place with `pta`, so parser changes should remain compatible with that edit-and-run workflow when possible.