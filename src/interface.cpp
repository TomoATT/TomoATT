#include "interface.h"
#include "utils.h"

#include <cmath>
#include <algorithm>
#include <set>


PhaseDefinition parse_phase_name(const std::string& phase_name,
                                 const std::vector<InterfaceDefinition>& interfaces) {
    PhaseDefinition pdef;
    pdef.name = phase_name;
    pdef.valid = false;

    if (phase_name.empty()) return pdef;

    // Find the free-surface interface label
    std::string free_surface_label;
    // Find the first non-surface interface label (conventionally "Moho" or the 'm' interface)
    std::string moho_label;

    for (const auto& iface : interfaces) {
        if (iface.is_free_surface) {
            free_surface_label = iface.label;
        } else if (moho_label.empty()) {
            // first non-surface interface is the "m" interface
            moho_label = iface.label;
        }
    }

    // Simple cases: single letter
    if (phase_name == "P") {
        // Direct P wave - no reflection needed
        pdef.valid = true;
        return pdef;
    }
    if (phase_name == "S") {
        // Direct S wave - no reflection needed
        pdef.valid = true;
        return pdef;
    }

    // Free-surface reflections: pP, sP, pS, sS
    // Convention: lowercase first letter = upgoing from source, uppercase second = reflected
    if (phase_name.length() == 2) {
        char c0 = phase_name[0];
        char c1 = phase_name[1];

        // Check format: lowercase + uppercase (free-surface reflection)
        if ((c0 == 'p' || c0 == 's') && (c1 == 'P' || c1 == 'S')) {
            if (free_surface_label.empty()) {
                std::cerr << "Warning: Phase '" << phase_name
                          << "' requires a free-surface interface but none defined." << std::endl;
                return pdef;
            }

            // Leg 1: upgoing from source to free surface
            PhaseLeg leg1;
            leg1.wave_type = (c0 == 'p') ? P_WAVE : S_WAVE;
            leg1.direction = LEG_UP;
            leg1.interface_label = free_surface_label;
            pdef.legs.push_back(leg1);

            // Leg 2: downgoing from free surface (reflected)
            PhaseLeg leg2;
            leg2.wave_type = (c1 == 'P') ? P_WAVE : S_WAVE;
            leg2.direction = LEG_DOWN;
            leg2.interface_label = "";
            pdef.legs.push_back(leg2);

            pdef.valid = true;
            return pdef;
        }
    }

    // Named-interface reflections: PmP, SmS, PmS, SmP
    // Format: [P|S] + 'm' + [P|S]  (where 'm' is the Moho/first deep interface)
    if (phase_name.length() == 3 && phase_name[1] == 'm') {
        char c0 = phase_name[0];
        char c2 = phase_name[2];

        if ((c0 == 'P' || c0 == 'S') && (c2 == 'P' || c2 == 'S')) {
            if (moho_label.empty()) {
                std::cerr << "Warning: Phase '" << phase_name
                          << "' requires a deep interface (Moho) but none defined." << std::endl;
                return pdef;
            }

            // Leg 1: downgoing from source to Moho
            PhaseLeg leg1;
            leg1.wave_type = (c0 == 'P') ? P_WAVE : S_WAVE;
            leg1.direction = LEG_DOWN;
            leg1.interface_label = moho_label;
            pdef.legs.push_back(leg1);

            // Leg 2: upgoing from Moho (reflected)
            PhaseLeg leg2;
            leg2.wave_type = (c2 == 'P') ? P_WAVE : S_WAVE;
            leg2.direction = LEG_UP;
            leg2.interface_label = "";
            pdef.legs.push_back(leg2);

            pdef.valid = true;
            return pdef;
        }
    }

    // General interface reflections: P{label}P, S{label}S, P{label}S, S{label}P
    // Format: [P|S] + {interface_label} + [P|S]
    if (phase_name.length() >= 3) {
        char c_first = phase_name[0];
        char c_last  = phase_name[phase_name.length() - 1];

        if ((c_first == 'P' || c_first == 'S') && (c_last == 'P' || c_last == 'S')) {
            std::string iface_label = phase_name.substr(1, phase_name.length() - 2);

            // Check if this interface label exists
            const InterfaceDefinition* iface = find_interface_by_label(interfaces, iface_label);
            if (iface != nullptr) {
                // Determine direction: if interface is below source → down then up
                // Default assumption: source is above the interface
                PhaseLeg leg1;
                leg1.wave_type = (c_first == 'P') ? P_WAVE : S_WAVE;
                leg1.direction = iface->is_free_surface ? LEG_UP : LEG_DOWN;
                leg1.interface_label = iface_label;
                pdef.legs.push_back(leg1);

                PhaseLeg leg2;
                leg2.wave_type = (c_last == 'P') ? P_WAVE : S_WAVE;
                leg2.direction = iface->is_free_surface ? LEG_DOWN : LEG_UP;
                leg2.interface_label = "";
                pdef.legs.push_back(leg2);

                pdef.valid = true;
                return pdef;
            }
        }
    }

    std::cerr << "Warning: Could not parse phase name '" << phase_name << "'." << std::endl;
    return pdef;
}


void identify_interface_nodes(InterfaceDefinition& iface,
                              const CUSTOMREAL* r_loc_1d,
                              int loc_I, int loc_J, int loc_K,
                              CUSTOMREAL dr) {
    iface.nodes.clear();

    // For all interfaces (including free surface): find closest grid level
    // using depth_km. depth_km is in km, r_loc_1d is in Earth radius units.
    CUSTOMREAL target_r = depth2radius(iface.depth_km);

    // Find closest k-index
    int best_k = -1;
    CUSTOMREAL best_dist = 1e30;
    for (int k = 0; k < loc_K; k++) {
        CUSTOMREAL dist = std::abs(r_loc_1d[k] - target_r);
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }

    // Check if the interface falls within this subdomain (within 1 grid spacing)
    if (best_k >= 0 && best_dist <= dr) {
        iface.k_interface = best_k;
        for (int j = 0; j < loc_J; j++) {
            for (int i = 0; i < loc_I; i++) {
                InterfaceNode node;
                node.i = i;
                node.j = j;
                node.k = best_k;
                node.arrival_time = 0.0;
                iface.nodes.push_back(node);
            }
        }
    }
    // If interface is not in this subdomain, nodes stays empty
}


std::vector<InterfaceDefinition> auto_detect_interfaces(
    const CUSTOMREAL* fun_loc,
    const CUSTOMREAL* r_loc_1d,
    int loc_I, int loc_J, int loc_K,
    CUSTOMREAL threshold,
    int k_start_loc, int k_end_loc) {

    std::vector<InterfaceDefinition> detected;

    // Scan in the r-direction for velocity jumps
    // Only check interior nodes (excluding ghost layers)
    std::set<int> detected_k_levels;

    for (int k_r = k_start_loc; k_r < k_end_loc; k_r++) {
        bool jump_found = false;
        for (int j_lat = 0; j_lat < loc_J && !jump_found; j_lat++) {
            for (int i_lon = 0; i_lon < loc_I && !jump_found; i_lon++) {
                CUSTOMREAL vel_bottom = _1_CR / fun_loc[I2V(i_lon, j_lat, k_r)];
                CUSTOMREAL vel_top    = _1_CR / fun_loc[I2V(i_lon, j_lat, k_r + 1)];

                if (vel_top > vel_bottom * (1.0 + threshold) ||
                    vel_top < vel_bottom * (1.0 - threshold)) {
                    detected_k_levels.insert(k_r);
                    jump_found = true;
                }
            }
        }
    }

    // Create InterfaceDefinition for each detected level
    int iface_count = 0;
    for (int k_level : detected_k_levels) {
        InterfaceDefinition iface;
        iface.label = "auto_" + std::to_string(iface_count);
        iface.depth_km = radius2depth(r_loc_1d[k_level]);
        iface.is_free_surface = false;
        iface.k_interface = k_level;

        // Populate nodes at this level
        for (int j = 0; j < loc_J; j++) {
            for (int i = 0; i < loc_I; i++) {
                InterfaceNode node;
                node.i = i;
                node.j = j;
                node.k = k_level;
                node.arrival_time = 0.0;
                iface.nodes.push_back(node);
            }
        }

        detected.push_back(iface);
        iface_count++;
    }

    return detected;
}


const InterfaceDefinition* find_interface_by_label(
    const std::vector<InterfaceDefinition>& interfaces,
    const std::string& label) {

    for (const auto& iface : interfaces) {
        if (iface.label == label) {
            return &iface;
        }
    }
    return nullptr;
}
