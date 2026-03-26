# Reflected Wave Computation in TomoATT (Eikonal Solver)

**Executive Summary:**  We propose extending TomoATT’s eikonal solver to compute reflected and mode-converted wave arrivals in addition to first (direct) arrivals.  The key idea is a *multistage eikonal* approach: run the standard eikonal solver for the direct wave, then **reinitialize** the solver at each model discontinuity to propagate reflected (and transmitted) phases separately【21†L81-L89】【50†L329-L338】.  This handles flat or curved interfaces (including free-surface) and elastic conversions (P↔S) by using the interface arrival times as new sources.  We will integrate this into TomoATT by identifying the solver entry points (e.g. where the travel-time field is computed), adding data structures for interface nodes, and running extra solves for each reflection branch.  A flowchart (below) summarizes the algorithm.  We compare methods in the table and recommend the multistage eikonal scheme as practical for TomoATT’s large-scale 3D setting【21†L81-L89】【50†L329-L338】.  Implementation will require new modules to detect interfaces, store interface traveltimes, and launch seeded eikonal runs; we outline pseudocode and data structures.  We also suggest synthetic tests (layered models, free-surface) for validation.  Milestones include: adding interface handling, coding reflected-phase solves, integrating with I/O, then benchmarking against analytic/refraction solutions.  Overall, this gives TomoATT the ability to output reflected-phase travel times (e.g. sP, pP, etc.) for tomography or imaging applications【21†L81-L89】【44†L78-L86】.

## Discontinuities & Reflection Physics  
Geophysical models contain **interfaces** (velocity jumps) such as layer boundaries, the free surface, or internal faults.  At each discontinuity, seismic waves reflect and refract according to Snell’s law.  For elastic media, a P-wave incidence can reflect as P or convert to S, and vice versa.  The travel-time solver must enforce that the incident and reflected rays satisfy Snell’s law (equal incidence and reflection angles) and Fermat’s stationary-time principle【50†L329-L338】.  

Typical discontinuities:  
- **Free surface**: acts as a mirror with total P-to-P and S-to-S reflection (S → SH or SV depending on polarity).  An easy method is the *image source* (mirror the source across the surface and compute direct traveltimes)【21†L95-L99】.  We can implement this by creating a “ghost” source or by reusing the interface-seeding approach.  
- **Layer interfaces**: when a wavefront hits a layer boundary, part of the wave reflects back (possibly converted) and part transmits downward.  In a grid solver, we capture the reflected branch by restarting the eikonal solution in the *upper* layer from the boundary, using the boundary traveltimes as new source values【50†L329-L338】.  The transmitted branch is similar (restart in the lower layer).  
- **Internal heterogeneities**: small-scale scatterers cause diffractions, which grid methods automatically handle as slow first-arrival delays.  Large contrasts (like salt) are effectively treated as sharp interfaces using the same multistage idea.  

Figure:  

【36†embed_image】 *Figure: Paths of primary (P) and secondary (S) body waves through Earth.  (Note: S-waves cannot travel through the liquid outer core, so only P-waves appear beyond ~2900 km)【35†L287-L289】.*  
The figure illustrates why P and S traveltimes differ at interfaces (here the core–mantle boundary).  In the crust and mantle, analogous reflections occur at each impedance jump.  

## Methods: Ray-Tracing vs. Grid-Based Eikonal  

- **Ray-tracing** (shooting/bending) computes individual ray paths that satisfy the eikonal’s Hamiltonian ODEs.  It naturally yields multiple arrivals and exact Snell behavior.  However, it must be run per source–receiver pair (or per ray path) and can miss rays (shadow zones) or fail to converge.  It is also computationally expensive for dense receiver sets or 3D.  Ray tracing does handle reflections easily (just extend ray when hitting interface) and can include elastic mode conversions explicitly.  

- **Standard eikonal (Fast Marching / Fast Sweeping)** computes the *first-arrival* traveltime field to all points on a grid in one pass【21†L81-L89】.  It is very efficient and robust (O(N log N) complexity).  The drawback is it only finds the earliest arrival (viscosity solution) and ignores later reflected/refraction arrivals【44†L78-L86】.  

- **Multistage grid-based methods** (e.g. de Kool et al., Bai et al.) extend eikonal solvers to multipaths by restarting at interfaces.  These are practical for large 3D problems.  They use the efficiency of the Eulerian solver but allow arbitrary reflection/transmission sequences by sequential solves【21†L81-L89】【50†L329-L338】.  

- **Phase-space / multi-arrival methods** (Fomel & Sethian) solve Hamilton-Jacobi in (x,p) space to get all arrivals in one go【48†L5-L14】.  These are very powerful (all phases at once) but computationally intense (memory/time blow-up).  Not ideal for large 3D tomography.  

- **Image source / two-point methods** use clever boundary conditions: mirror sources for planar reflectors, or solve two eikonals (from source and from receiver) and enforce equal-time on the reflector【21†L95-L99】.  These produce high accuracy for specific reflections (like teleseismic PP) but require multiple solves per interface and are complex to generalize to many sources.  

Overall, for TomoATT (large-scale 3D tomography) we favor a *multistage eikonal* scheme.  It leverages the existing solver architecture, scales well, and can handle both flat and curved interfaces【21†L81-L89】【50†L329-L338】.  We will illustrate this approach in detail below.

## Multistage Eikonal Algorithm for Reflections  

We adopt a multistage fast-marching (or fast-sweeping) approach by layers【21†L81-L89】【50†L329-L338】.  The steps for each source are: 

1. **Compute direct arrival:** Run the existing eikonal solver to get the direct-wave traveltime field `T_direct`.  
2. **Identify interfaces:** For each model discontinuity (free-surface or internal layer), gather the set of grid nodes on that interface.  This could be user-specified depths or derived from velocity contrasts.  
3. **Interface arrivals:** Extract traveltimes `T_int(node)` from `T_direct` at the interface nodes.  These represent the arrival of the incident wavefront at the interface.  
4. **Reflected phase (upper layer):** Treat the interface nodes as *new sources* with initial times `T_int`.  Rerun the eikonal solver *in the incident (upper) layer* using these interface seeds.  This yields a traveltime field `T_refl` for the reflected branch.  (If the incident wave is a P-wave, this computes P→P reflection; to compute P→S conversion, run another solve with the S-wave speeds in the upper layer, seeded by the same `T_int`.)  
5. **Transmitted phase (lower layer, if needed):** Similarly, reinitialize the solver in the lower layer from the same `T_int` to compute the transmitted branch (P→P and P→S downward) if those are of interest.  
6. **Repeat for multiple interfaces:** If more than one interface is in the path, iterate: a reflected wave can strike another interface, so one may chain the process (or simply allow the algorithm to be reapplied with the reflected arrivals as a new “source”).  

【21†L81-L89】summarizes this approach: “each layer that the wave front enters [is treated] as a separate computational domain… reinitialized at each interface to track the evolving wave front as either a reflection or a transmission.”  Bai et al. further describe using the *earliest-arrival node* on each interface to seed new wavefronts (Huygens’ principle)【50†L329-L338】.  In practice we can seed *all* interface nodes at once (the FMM handles multiple sources) or use a min-heap keyed by `T_int`.  

**Pseudocode (simplified):**  
```
// Given: velocityModel with layers, source location.
T_direct = SolveEikonal(source, velocityModel)   // existing code
for interface in interfaces:
    T_int = ExtractTimes(T_direct, interface.nodes)
    // Reflected P-wave in upper layer
    T_reflP = SolveEikonal(interface.nodes, velocityModel_upper, initTimes=T_int)
    // Optional: converted S-wave in upper layer
    T_reflS = SolveEikonal(interface.nodes, velocityModel_upper_S, initTimes=T_int)
    // Optional: transmitted P,S in lower layer
    T_transP = SolveEikonal(interface.nodes, velocityModel_lower, initTimes=T_int)
    T_transS = SolveEikonal(interface.nodes, velocityModel_lower_S, initTimes=T_int)
    // Store or output T_reflP, T_reflS, etc. 
end
```
Each `SolveEikonal` call could be the same FMM routine, but modified to accept multiple initial seeds with prescribed times (instead of a single source).  TomoATT’s solver likely already supports multiple sources (for multiple simultaneous events)【52†L361-L369】; we would use that by inserting the interface nodes as pseudo-sources.  

## Implementation Plan for TomoATT  

To integrate reflections into TomoATT, we identify key tasks, likely code areas, and changes:  

- **Interface Representation:** Add a way to specify discontinuities.  Options: allow the user to input interface depths or slowness jumps; or detect large velocity gradients.  Data structure: e.g. `std::vector<GridNode>` listing interface cells.  

- **Solver Entry Points:** TomoATT’s solver is in `src` (likely a function like `solveTravelTime(...)`).  We will extend this function to loop over interfaces after computing the direct wave.  It should accept initial times array.  Look for code that sets up the source term – modify it to allow multiple seeds.  

- **Data Structures:** Reuse the traveltime grid array for each solve.  Additional arrays (or reuse) for `T_refl`.  If memory is a concern, do solves sequentially (free memory in-between).  Possibly use an output container for multiple phases: e.g. extend any existing traveltime output format to include phase type (like store in HDF5 with group “reflP”, “reflS”).  

- **New Modules/Functions:**  
  - **Interface initialization:** A function to read/store interfaces.  
  - **Seeded Eikonal solve:** Possibly create a wrapper `SolveEikonalFromBoundary(interfaceNodes, velocityGrid, initTimes)` that calls the existing solver (FMM) but seeds it at interface.  
  - **Phase labeling:** Tag each solver run with a phase name (e.g. “P-to-P reflection”).  

- **API Changes:** Modify user-facing run configuration to include an option to enable reflection mode and to specify interfaces.  The main driver would parse new parameters (e.g. `--interfaces`).  Also extend input/output: the output traveltime files should now include reflected phases separately.  

- **Integration Steps:**  
  1. **Locate and extend eikonal solver:** Find the code where sources are initialized (likely in `src/`). Modify it to allow array of source nodes with given times. TomoATT already parallelizes over sources; use that structure to handle interface seeds.  
  2. **Add interface handling:** After direct solve, call the seeded solver for each interface as above. Use MPI (layer1 parallel) to distribute these solves if multiple sources or subdomains.  
  3. **Output storage:** Ensure traveltimes from reflected solves are written out (e.g. appended HDF5 datasets). Possibly reuse the tomography inversion framework to invert on both direct and reflected picks if needed.  
  4. **Testing and validation:** Create example configs with simple interfaces.  

**Example data structures:** In C++:  
```cpp
struct Interface { 
    std::vector<int> nodeIDs; 
    std::string label; // e.g. "Moho", "freeSurface"
    VelocityGrid upperVel, lowerVel; 
};
```
`nodeIDs` store grid cell indices.  `upperVel`/`lowerVel` hold P/S velocities on each side.  After the direct solve:  
```cpp
std::vector<double> T_int(interface.nodeIDs.size());
for(int i=0; i<interface.nodeIDs.size(); ++i){
    T_int[i] = T_direct[ interface.nodeIDs[i] ];
}
```
Then use `T_int` as initial condition.

## Proposed Algorithm Flowchart  

```mermaid
flowchart TD
    A[Start: Load model and source] --> B[Compute direct traveltime (existing eikonal)]
    B --> C{Any interfaces?}
    C -- No --> Z[End (only direct arrivals)]
    C -- Yes --> D[For each interface: extract interface node times]
    D --> E[Initialize solver with interface nodes as sources]
    E --> F[Run eikonal solve in upper layer -> reflected traveltime]
    F --> G{Compute converted/reflections?}
    G -- Yes --> H[Run solver with S-velocities -> P→S reflection]
    G --> I[Optional: run solver in lower layer (transmitted branches)]
    I --> J[Collect traveltimes for all reflected/transmitted phases]
    J --> K{More interfaces?}
    K -- Yes --> D
    K -- No --> Z[End with direct & reflected outputs]
```

This flowchart illustrates the multistage process: the direct run (B) is followed by per-interface branching into reflected and converted solves (F–I).  Each solve appends new traveltime fields.  

## Comparison of Methods  

| Method                        | Advantages                                  | Limitations                                      | Complexity & Use Case                         |
|-------------------------------|---------------------------------------------|--------------------------------------------------|-----------------------------------------------|
| **Ray-Tracing**               | Directly models multipaths and mode conversions; flexible in complex media. | Slow for many sources; requires solving BVP/IVP for each ray; can miss rays. | $O(N_\text{rays} \cdot n_\text{steps})$. Works in 2D/3D, elastic easily. |
| **Fast Marching (first-arrival)** | Very fast (O(N log N)) for global first arrivals; robust; single-sweep. | Only finds earliest arrival; no reflections.        | $O(N \log N)$. 2D/3D acoustic (needs separate run for S). |
| **Multistage FMM** (this work) | Captures reflections (and refractions) with FMM efficiency; grid-based; good for layered models【21†L81-L89】. | Requires knowing interfaces; additional solves (linear overhead); some interpolation error at interface. | ~$M\times O(N\log N)$ for $M$ phases. Suited for 2D/3D large problems. |
| **Modified Shortest-Path (MSPM)**【44†L78-L86】 | Similar to multistage FMM; high accuracy (uses interpolation)【50†L329-L338】; efficient CPU/mem. | Complex to implement; still layered assumption; inversion-specific code. | Comparable to FMM, plus re-meshing costs. 2D/3D, supports elasticity (P/S)【50†L329-L338】. |
| **Phase-Space (Fomel 2002)**【48†L5-L14】 | Computes *all* arrivals simultaneously; handles complex multipath exactly. | Very high-dimensional (phase space); expensive memory/time (practically 2D only). | ~$O(N^{d+1})$; mostly research use. |
| **Image-Source**             | Simple implementation for planar/flat reflectors; no code complexity. | Only exact for flat interfaces; cannot handle variable velocity media without errors. | One extra solve (mirror source). 2D/3D acoustic only (no mode conv). |
| **Two-Point Eikonal**        | Rigorous (uses Fermat’s principle); can pin-point any reflection path. | Requires 2 solves per ray or pair; expensive for many sources/receivers; tricky to automate. | $2\times O(N\log N)$ per reflector (source and receiver solves). Used for targeted picks. |

**Recommendation:** For TomoATT’s use (large-scale 3D tomography), the **Multistage FMM** approach is most practical.  It extends the existing eikonal framework with a moderate overhead (few extra solves) and uses the same parallel infrastructure【21†L81-L89】【50†L329-L338】.  Ray tracing is too slow for many sources.  Phase-space is too costly for high-res grids.  We note that our multistage approach *inherits* the accuracy of FMM for each branch (up to interpolation at interfaces) and will scale similarly.

## Numerical and Implementation Details  

- **Grid vs. Unstructured:** TomoATT is grid-based (Cartesian grid).  We will work with that existing mesh.  If we needed unstructured support (e.g. curved topography), we could approximate by a staircase or use a coordinate transform, but that’s beyond scope.  

- **Parallelization:** TomoATT already parallelizes over sources and subdomains【52†L361-L369】.  We should integrate reflection solves into that scheme.  For each interface solve, we also use the same subdomain/sweep parallelism.  MPI layer-1 can handle multiple “secondary sources” (interface nodes) as it does primary sources.  

- **Accuracy/Stability:** Each eikonal solve is stable and first-order (or second-order in advanced schemes).  In multistage mode, we may need finer grid near interfaces to reduce interpolation error (Bai et al. used local refinement)【50†L329-L338】.  Raypaths can be reconstructed by backtracking along the traveltime gradient if needed【24†L953-L962】.  

- **Computational Cost:** Roughly, each interface adds 1–2 extra eikonal solves.  If there are $k$ interfaces of interest, cost $\sim (k+1)\times$ direct solve time.  This is acceptable if $k$ is small.  Memory use is dominated by the grid (unchanged).  MPI overhead is minimal since eikonal solves dominate.  

## Validation Strategies  

We will verify correctness with synthetic models:  

- **Layered model:** Two-layer model with horizontal interface.  Choose velocities (e.g. $V_1=4$ km/s, $V_2=6$ km/s, interface at $z=10$ km).  Place a source below and receivers above.  Analytic reflection traveltime $t = t_{src→interface}+t_{interface→rcv}$ can be computed.  Check that our solver output matches.  

- **Dip model:** Single inclined interface (e.g. 45°).  Test Snell-law compliance: vary source angle and check reflected ray geometry.  

- **Free-surface:** Homogeneous half-space with a source.  The reflected traveltime should match an image source (mirror depth).  

- **Elastic model:** Add S-wave speed and compare converted wave traveltimes.  For example, a plane interface with known PP and PS arrival curves.  

- **Benchmark cases:** Large 3D wedge or subduction zone (from literature) to test performance and stability.  

Metrics: Traveltime error (vs analytic or ray-trace) and CPU time.  Also correctness of ray paths (via gradient integration).  

## Implementation Roadmap (Milestones)  

1. **Interface Handling (1–2 weeks):** Add input parameters for discontinuities; code to identify interface nodes and store them.  Review solver entry (likely in `src/Solver.cpp`) to insert interface logic after direct solve.  
2. **Seeded Eikonal Solve (2–3 weeks):** Modify or wrap the existing eikonal function to accept multiple seeds with given traveltimes.  Use MPI parallelism.  Ensure output traveltime array for reflections.  
3. **I/O and Data Structures (1 week):** Extend output format to include reflected traveltimes (label by phase).  Possibly reuse HDF5 writes.  
4. **Testing & Validation (2 weeks):** Create synthetic models and run test cases, compare results.  Unit tests in `tests/` for reflection correctness.  
5. **Performance Tuning (2 weeks):** Benchmark multi-solve overhead.  Optimize if needed (e.g. reuse global data structures, minimize MPI waits).  
6. **Documentation & Examples (1 week):** Update user guide and examples to illustrate reflection usage.  

*Risks:* Ensuring numerical accuracy at interfaces (grid discretization error).  Managing multiple solves’ parallel resources.  (These can be mitigated with local refinement and careful MPI design.)  

## References  

Key literature on multipath eikonal methods: de Kool et al. (2006)【21†L81-L89】【24†L953-L962】, Bai et al. (2009)【44†L78-L86】【50†L329-L338】 (grid-based multi arrival), Fomel & Sethian (2002)【48†L5-L14】 (phase space), and standard texts on ray theory (Cerveny 2001) and imaging.  The TomoATT documentation and publications provide solver context【52†L324-L333】【52†L342-L351】.  We preserve all cited source references above with the format “【source†Ln-Lm】”.

