#ifndef INTERFACE_H
#define INTERFACE_H

#include <string>
#include <vector>
#include <map>
#include <iostream>

#include "config.h"

// Wave type enumeration
enum WaveType {
    P_WAVE = 0,
    S_WAVE = 1
};

// Direction of wave propagation leg
enum LegDirection {
    LEG_DOWN = 0,   // wave traveling downward (increasing depth / decreasing r)
    LEG_UP   = 1    // wave traveling upward (decreasing depth / increasing r)
};

// A single leg in a phase path
struct PhaseLeg {
    WaveType      wave_type;   // P or S
    LegDirection  direction;   // up or down
    std::string   interface_label; // interface hit at the end of this leg ("" if none)
};

// Parsed phase definition from standard seismological naming
struct PhaseDefinition {
    std::string          name;  // original name, e.g. "PmP", "pP", "sP"
    std::vector<PhaseLeg> legs; // sequence of legs
    bool                 valid = false;
};

// A node on an interface
struct InterfaceNode {
    int        i;              // longitude index
    int        j;              // latitude index
    int        k;              // radius index
    CUSTOMREAL arrival_time;   // traveltime of incident wave at this node
};

// Interface (discontinuity) definition
struct InterfaceDefinition {
    std::string label;                    // e.g. "Moho", "freeSurface"
    CUSTOMREAL  depth_km;                 // nominal depth in km (-1 for auto-detected)
    bool        is_free_surface = false;  // true for the top boundary
    int         k_interface     = -1;     // closest grid index in r-direction (set during identification)
    std::vector<InterfaceNode> nodes;     // grid nodes on this interface (local to this MPI rank)
};

// Parse a seismological phase name into a PhaseDefinition
// Supported formats:
//   Direct: "P", "S"
//   Free-surface reflections: "pP", "sP", "pS", "sS"
//   Named-interface reflections: "PmP", "SmS", "PmS", "SmP"
//     where 'm' refers to the first non-surface interface (conventionally "Moho")
//   General interface: "P{label}P", "P{label}S" etc.
PhaseDefinition parse_phase_name(const std::string& phase_name,
                                 const std::vector<InterfaceDefinition>& interfaces);

// Identify interface grid nodes on the local subdomain
void identify_interface_nodes(InterfaceDefinition& iface,
                              const CUSTOMREAL* r_loc_1d,
                              int loc_I, int loc_J, int loc_K,
                              CUSTOMREAL dr);

// Auto-detect interfaces from velocity contrast
std::vector<InterfaceDefinition> auto_detect_interfaces(
    const CUSTOMREAL* fun_loc,
    const CUSTOMREAL* r_loc_1d,
    int loc_I, int loc_J, int loc_K,
    CUSTOMREAL threshold,
    int k_start_loc, int k_end_loc);

// Find the interface definition matching a label
const InterfaceDefinition* find_interface_by_label(
    const std::vector<InterfaceDefinition>& interfaces,
    const std::string& label);

#endif // INTERFACE_H
