#include "iterator_wrapper.cuh"

__device__ const CUSTOMREAL PLUS = 1.0;
__device__ const CUSTOMREAL MINUS = -1.0;
__device__ const CUSTOMREAL v_eps = 1e-12;

__device__ const CUSTOMREAL _0_CR     = 0.0;
__device__ const CUSTOMREAL _0_25_CR  = 0.25;
__device__ const CUSTOMREAL _0_5_CR   = 0.5;
__device__ const CUSTOMREAL _1_CR     = 1.0;
__device__ const CUSTOMREAL _2_CR     = 2.0;
__device__ const CUSTOMREAL _3_CR     = 3.0;
__device__ const CUSTOMREAL _4_CR     = 4.0;

__device__ CUSTOMREAL my_square_cu(CUSTOMREAL const& x) {
    return x*x;
}

__device__ CUSTOMREAL calc_stencil_1st(CUSTOMREAL const& a, CUSTOMREAL const& b, CUSTOMREAL const& Dinv){
    return Dinv*(a-b);
}

__device__ CUSTOMREAL calc_stencil_3rd(CUSTOMREAL const& a, CUSTOMREAL const& b, CUSTOMREAL const& c, CUSTOMREAL const& d, CUSTOMREAL const& Dinv_half, CUSTOMREAL const& sign){
    CUSTOMREAL tmp1 = v_eps + my_square_cu(a-_2_CR*b+c);
    CUSTOMREAL tmp2 = v_eps + my_square_cu(d-_2_CR*a+b);
    CUSTOMREAL ww   = _1_CR/(_1_CR+_2_CR*my_square_cu(tmp1/tmp2));
    return sign*((_1_CR-ww)* (b-d)*Dinv_half + ww*(-_3_CR*a+_4_CR*b-c)*Dinv_half);
}

__device__ CUSTOMREAL cuda_calc_LF_Hamiltonian( \
                                            CUSTOMREAL const& fac_a_, \
                                            CUSTOMREAL const& fac_b_, \
                                            CUSTOMREAL const& fac_c_, \
                                            CUSTOMREAL const& fac_f_, \
                                            CUSTOMREAL const& T0r_, \
                                            CUSTOMREAL const& T0t_, \
                                            CUSTOMREAL const& T0p_, \
                                            CUSTOMREAL const& T0v_, \
                                            CUSTOMREAL& tau_, \
                                            CUSTOMREAL const& pp1, CUSTOMREAL& pp2, \
                                            CUSTOMREAL const& pt1, CUSTOMREAL& pt2, \
                                            CUSTOMREAL const& pr1, CUSTOMREAL& pr2 \
                                            ) {
    // LF Hamiltonian for T = T0 * tau
    return sqrt(
              fac_a_ * my_square_cu(T0r_ * tau_ + T0v_ * (pr1+pr2)/_2_CR) \
    +         fac_b_ * my_square_cu(T0t_ * tau_ + T0v_ * (pt1+pt2)/_2_CR) \
    +         fac_c_ * my_square_cu(T0p_ * tau_ + T0v_ * (pp1+pp2)/_2_CR) \
    -   _2_CR*fac_f_ * (T0t_ * tau_ + T0v_ * (pt1+pt2)/_2_CR) \
                     * (T0p_ * tau_ + T0v_ * (pp1+pp2)/_2_CR) \
    );
}

__global__ void cuda_do_sweep_level_kernel_1st(\
    const int i__j__k__[],\
    const int ip1j__k__[],\
    const int im1j__k__[],\
    const int i__jp1k__[],\
    const int i__jm1k__[],\
    const int i__j__kp1[],\
    const int i__j__km1[],\
    const CUSTOMREAL fac_a[], \
    const CUSTOMREAL fac_b[], \
    const CUSTOMREAL fac_c[], \
    const CUSTOMREAL fac_f[], \
    const CUSTOMREAL T0v[], \
    const CUSTOMREAL T0r[], \
    const CUSTOMREAL T0t[], \
    const CUSTOMREAL T0p[], \
    const CUSTOMREAL fun[], \
    const bool changed[], \
    CUSTOMREAL tau[], \
    const int loc_I, \
    const int loc_J, \
    const int loc_K, \
    const CUSTOMREAL dr, \
    const CUSTOMREAL dt, \
    const CUSTOMREAL dp, \
    const int n_nodes_this_level, \
    const int i_start \
){

    unsigned int i_node = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i_node >= n_nodes_this_level) return;

    i_node += i_start;

    //if (i_node >= loc_I*loc_J*loc_K) return;

    if (changed[i_node] != true) return;

    CUSTOMREAL sigr = _1_CR*sqrt(fac_a[i_node])*T0v[i_node];
    CUSTOMREAL sigt = _1_CR*sqrt(fac_b[i_node])*T0v[i_node];
    CUSTOMREAL sigp = _1_CR*sqrt(fac_c[i_node])*T0v[i_node];
    CUSTOMREAL coe  = _1_CR/((sigr/dr)+(sigt/dt)+(sigp/dp));

    CUSTOMREAL pp1 = calc_stencil_1st(tau[i__j__k__[i_node]],tau[im1j__k__[i_node]], _1_CR/dp);
    CUSTOMREAL pp2 = calc_stencil_1st(tau[ip1j__k__[i_node]],tau[i__j__k__[i_node]], _1_CR/dp);

    CUSTOMREAL pt1 = calc_stencil_1st(tau[i__j__k__[i_node]],tau[i__jm1k__[i_node]], _1_CR/dt);
    CUSTOMREAL pt2 = calc_stencil_1st(tau[i__jp1k__[i_node]],tau[i__j__k__[i_node]], _1_CR/dt);

    CUSTOMREAL pr1 = calc_stencil_1st(tau[i__j__k__[i_node]],tau[i__j__km1[i_node]], _1_CR/dr);
    CUSTOMREAL pr2 = calc_stencil_1st(tau[i__j__kp1[i_node]],tau[i__j__k__[i_node]], _1_CR/dr);

    // LF Hamiltonian
    CUSTOMREAL Htau = cuda_calc_LF_Hamiltonian(\
                                               fac_a[i_node], \
                                               fac_b[i_node], \
                                               fac_c[i_node], \
                                               fac_f[i_node], \
                                               T0r[i_node], \
                                               T0t[i_node], \
                                               T0p[i_node], \
                                               T0v[i_node], \
                                               tau[i__j__k__[i_node]], \
                                               pp1, pp2, pt1, pt2, pr1, pr2);

    tau[i__j__k__[i_node]] += coe*((fun[i_node] - Htau) \
                                  +(sigr*(pr2-pr1) \
                                  + sigt*(pt2-pt1) \
                                  + sigp*(pp2-pp1))/_2_CR);

}

__global__ void cuda_do_sweep_level_kernel_3rd(\
    const int i__j__k__[],\
    const int ip1j__k__[],\
    const int im1j__k__[],\
    const int i__jp1k__[],\
    const int i__jm1k__[],\
    const int i__j__kp1[],\
    const int i__j__km1[],\
    const int ip2j__k__[],\
    const int im2j__k__[],\
    const int i__jp2k__[],\
    const int i__jm2k__[],\
    const int i__j__kp2[],\
    const int i__j__km2[],\
    const CUSTOMREAL fac_a[], \
    const CUSTOMREAL fac_b[], \
    const CUSTOMREAL fac_c[], \
    const CUSTOMREAL fac_f[], \
    const CUSTOMREAL T0v[], \
    const CUSTOMREAL T0r[], \
    const CUSTOMREAL T0t[], \
    const CUSTOMREAL T0p[], \
    const CUSTOMREAL fun[], \
    const bool changed[], \
    CUSTOMREAL tau[], \
    const int loc_I, \
    const int loc_J, \
    const int loc_K, \
    const CUSTOMREAL dr, \
    const CUSTOMREAL dt, \
    const CUSTOMREAL dp,  \
    const int n_nodes_this_level, \
    const int i_start \
){

    CUSTOMREAL pp1, pp2, pt1, pt2, pr1, pr2;

    unsigned int i_node = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (i_node >= n_nodes_this_level) return;

    i_node += i_start;
    //if (i_node >= loc_I*loc_J*loc_K) return;

    if (changed[i_node] != true) return;

    int k =  i__j__k__[i_node] / (loc_I*loc_J);
    int j = (i__j__k__[i_node] - k*loc_I*loc_J)/loc_I;
    int i =  i__j__k__[i_node] - k*loc_I*loc_J - j*loc_I;


    CUSTOMREAL DRinv = _1_CR/dr;
    CUSTOMREAL DTinv = _1_CR/dt;
    CUSTOMREAL DPinv = _1_CR/dp;
    CUSTOMREAL DRinv_half = DRinv*_0_5_CR;
    CUSTOMREAL DTinv_half = DTinv*_0_5_CR;
    CUSTOMREAL DPinv_half = DPinv*_0_5_CR;

    CUSTOMREAL sigr = _1_CR*sqrt(fac_a[i_node])*T0v[i_node];
    CUSTOMREAL sigt = _1_CR*sqrt(fac_b[i_node])*T0v[i_node];
    CUSTOMREAL sigp = _1_CR*sqrt(fac_c[i_node])*T0v[i_node];
    CUSTOMREAL coe  = _1_CR/((sigr/dr)+(sigt/dt)+(sigp/dp));

    // direction p
    if (i == 1) {
        pp1 = calc_stencil_1st(tau[i__j__k__[i_node]],tau[im1j__k__[i_node]],DPinv);
        pp2 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[ip1j__k__[i_node]],tau[ip2j__k__[i_node]],tau[im1j__k__[i_node]],DPinv_half, PLUS);
    } else if (i == loc_I-2) {
        pp1 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[im1j__k__[i_node]],tau[im2j__k__[i_node]],tau[ip1j__k__[i_node]],DPinv_half, MINUS);
        pp2 = calc_stencil_1st(tau[ip1j__k__[i_node]],tau[i__j__k__[i_node]],DPinv);
    } else {
        pp1 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[im1j__k__[i_node]],tau[im2j__k__[i_node]],tau[ip1j__k__[i_node]],DPinv_half, MINUS);
        pp2 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[ip1j__k__[i_node]],tau[ip2j__k__[i_node]],tau[im1j__k__[i_node]],DPinv_half, PLUS);
    }

    // direction t
    if (j == 1) {
        pt1 = calc_stencil_1st(tau[i__j__k__[i_node]],tau[i__jm1k__[i_node]],DTinv);
        pt2 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__jp1k__[i_node]],tau[i__jp2k__[i_node]],tau[i__jm1k__[i_node]],DTinv_half, PLUS);
    } else if (j == loc_J-2) {
        pt1 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__jm1k__[i_node]],tau[i__jm2k__[i_node]],tau[i__jp1k__[i_node]],DTinv_half, MINUS);
        pt2 = calc_stencil_1st(tau[i__jp1k__[i_node]],tau[i__j__k__[i_node]],DTinv);
    } else {
        pt1 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__jm1k__[i_node]],tau[i__jm2k__[i_node]],tau[i__jp1k__[i_node]],DTinv_half, MINUS);
        pt2 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__jp1k__[i_node]],tau[i__jp2k__[i_node]],tau[i__jm1k__[i_node]],DTinv_half, PLUS);
    }

    // direction r
    if (k == 1) {
        pr1 = calc_stencil_1st(tau[i__j__k__[i_node]],tau[i__j__km1[i_node]],DRinv);
        pr2 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__j__kp1[i_node]],tau[i__j__kp2[i_node]],tau[i__j__km1[i_node]],DRinv_half, PLUS);
    } else if (k == loc_K-2) {
        pr1 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__j__km1[i_node]],tau[i__j__km2[i_node]],tau[i__j__kp1[i_node]],DRinv_half, MINUS);
        pr2 = calc_stencil_1st(tau[i__j__kp1[i_node]],tau[i__j__k__[i_node]],DRinv);
    } else {
        pr1 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__j__km1[i_node]],tau[i__j__km2[i_node]],tau[i__j__kp1[i_node]],DRinv_half, MINUS);
        pr2 = calc_stencil_3rd(tau[i__j__k__[i_node]],tau[i__j__kp1[i_node]],tau[i__j__kp2[i_node]],tau[i__j__km1[i_node]],DRinv_half, PLUS);
    }

    CUSTOMREAL Htau = cuda_calc_LF_Hamiltonian(\
                                               fac_a[i_node], \
                                               fac_b[i_node], \
                                               fac_c[i_node], \
                                               fac_f[i_node], \
                                               T0r[i_node], \
                                               T0t[i_node], \
                                               T0p[i_node], \
                                               T0v[i_node], \
                                               tau[i__j__k__[i_node]], \
                                               pp1, pp2, pt1, pt2, pr1, pr2);

    tau[i__j__k__[i_node]] += coe*((fun[i_node] - Htau) \
                                  +(sigr*(pr2-pr1) \
                                  + sigt*(pt2-pt1) \
                                  + sigp*(pp2-pp1))/_2_CR);


}


// ============================================================================
// UPWIND solver GPU kernel
// Ports calculate_stencil_1st_order_upwind to GPU.
// 26 cases: 8 tetrahedron (3D) + 12 triangle (2D: r-t, r-p, t-p planes) + 6 line (1D, 2 per axis)
// Each case solves a quadratic for tau and checks causality.
// Final: take minimum valid candidate.
// ============================================================================

__device__ inline CUSTOMREAL cuda_calc_upwind_Hamiltonian(
    CUSTOMREAL const& fac_a, CUSTOMREAL const& fac_b, CUSTOMREAL const& fac_c,
    CUSTOMREAL const& T0r, CUSTOMREAL const& T0t, CUSTOMREAL const& T0p,
    CUSTOMREAL const& T0v,
    CUSTOMREAL const& pp1, CUSTOMREAL const& pp2,
    CUSTOMREAL const& pt1, CUSTOMREAL const& pt2,
    CUSTOMREAL const& pr1, CUSTOMREAL const& pr2)
{
    // LF Hamiltonian for T = T0 * tau (same as LF solver)
    return sqrt(
        fac_a * my_square_cu(T0r * T0v + T0v * (pr1+pr2)/_2_CR)  // Note: this is the LF Hamiltonian, used as fallback
      + fac_b * my_square_cu(T0t * T0v + T0v * (pt1+pt2)/_2_CR)
      + fac_c * my_square_cu(T0p * T0v + T0v * (pp1+pp2)/_2_CR)
      - _2_CR * fac_c * (T0t * T0v + T0v * (pt1+pt2)/_2_CR)
                     * (T0p * T0v + T0v * (pp1+pp2)/_2_CR)
    );
}

// UPWIND 1st order kernel - implements the full upwind solver on GPU
__global__ void cuda_do_sweep_level_kernel_upwind(
    const int i__j__k__[],    // current node index
    const int ip1j__k__[],    // i+1 neighbor
    const int im1j__k__[],    // i-1 neighbor
    const int i__jp1k__[],    // j+1 neighbor
    const int i__jm1k__[],    // j-1 neighbor
    const int i__j__kp1[],    // k+1 neighbor
    const int i__j__km1[],    // k-1 neighbor
    const CUSTOMREAL fac_a[],
    const CUSTOMREAL fac_b[],
    const CUSTOMREAL fac_c[],
    const CUSTOMREAL fac_f[],
    const CUSTOMREAL T0v[],
    const CUSTOMREAL T0r[],
    const CUSTOMREAL T0t[],
    const CUSTOMREAL T0p[],
    const CUSTOMREAL fun[],
    const bool changed[],
    CUSTOMREAL tau[],
    const CUSTOMREAL T0v_glob[],   // global (full-grid) T0v indexed by flattened global index, for neighbor causality
    const int loc_I,
    const int loc_J,
    const int loc_K,
    const CUSTOMREAL dr,
    const CUSTOMREAL dt,
    const CUSTOMREAL dp,
    const int n_nodes_this_level,
    const int i_start)
{
    unsigned int i_node = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i_node >= n_nodes_this_level) return;
    i_node += i_start;
    if (!changed[i_node]) return;

    // Get 3D indices from the current node
    int ii = i__j__k__[i_node];
    int k = ii / (loc_I * loc_J);
    int j = (ii - k * loc_I * loc_J) / loc_I;
    int i = ii - k * loc_I * loc_J - j * loc_I;

    // Boundary checks
    int np = loc_I, nt = loc_J, nr = loc_K;

    // Prepare forward/backward partial derivative coefficients
    // T_p = (T0*tau)_p = T0p*tau + T0v*tau_p = ap*tau + bp
    CUSTOMREAL ap1 = 0, bp1 = 0, ap2 = 0, bp2 = 0;
    CUSTOMREAL at1 = 0, bt1 = 0, at2 = 0, bt2 = 0;
    CUSTOMREAL ar1 = 0, br1 = 0, ar2 = 0, br2 = 0;

    if (i > 0) {
        ap1 = T0p[i_node] + T0v[i_node] / dp;
        bp1 = -T0v[i_node] / dp * tau[im1j__k__[i_node]];
    }
    if (i < np - 1) {
        ap2 = T0p[i_node] - T0v[i_node] / dp;
        bp2 = T0v[i_node] / dp * tau[ip1j__k__[i_node]];
    }
    if (j > 0) {
        at1 = T0t[i_node] + T0v[i_node] / dt;
        bt1 = -T0v[i_node] / dt * tau[i__jm1k__[i_node]];
    }
    if (j < nt - 1) {
        at2 = T0t[i_node] - T0v[i_node] / dt;
        bt2 = T0v[i_node] / dt * tau[i__jp1k__[i_node]];
    }
    if (k > 0) {
        ar1 = T0r[i_node] + T0v[i_node] / dr;
        br1 = -T0v[i_node] / dr * tau[i__j__km1[i_node]];
    }
    if (k < nr - 1) {
        ar2 = T0r[i_node] - T0v[i_node] / dr;
        br2 = T0v[i_node] / dr * tau[i__j__kp1[i_node]];
    }

    CUSTOMREAL fun_loc_sq = fun[i_node] * fun[i_node];
    CUSTOMREAL bc_f2 = fac_b[i_node] * fac_c[i_node] - fac_f[i_node] * fac_f[i_node];
    CUSTOMREAL bc_over_b = bc_f2 / fac_b[i_node];
    CUSTOMREAL bc_over_c = bc_f2 / fac_c[i_node];

    // Candidate solutions (max 52 candidates from 26 cases × 2 solutions)
    CUSTOMREAL cand[52];
    int count_cand = 0;

    // First catalog: 8 tetrahedron cases (3D volume)
    for (int i_case = 0; i_case < 8; i_case++) {
        CUSTOMREAL ap = 0, bp = 0, at = 0, bt = 0, ar = 0, br = 0;

        switch (i_case) {
            case 0: if (i == 0 || j == 0 || k == 0) continue; ap = ap1; bp = bp1; at = at1; bt = bt1; ar = ar1; br = br1; break;
            case 1: if (i == 0 || j == 0 || k == nr-1) continue; ap = ap1; bp = bp1; at = at1; bt = bt1; ar = ar2; br = br2; break;
            case 2: if (i == 0 || j == nt-1 || k == 0) continue; ap = ap1; bp = bp1; at = at2; bt = bt2; ar = ar1; br = br1; break;
            case 3: if (i == 0 || j == nt-1 || k == nr-1) continue; ap = ap1; bp = bp1; at = at2; bt = bt2; ar = ar2; br = br2; break;
            case 4: if (i == np-1 || j == 0 || k == 0) continue; ap = ap2; bp = bp2; at = at1; bt = bt1; ar = ar1; br = br1; break;
            case 5: if (i == np-1 || j == 0 || k == nr-1) continue; ap = ap2; bp = bp2; at = at1; bt = bt1; ar = ar2; br = br2; break;
            case 6: if (i == np-1 || j == nt-1 || k == 0) continue; ap = ap2; bp = bp2; at = at2; bt = bt2; ar = ar1; br = br1; break;
            case 7: if (i == np-1 || j == nt-1 || k == nr-1) continue; ap = ap2; bp = bp2; at = at2; bt = bt2; ar = ar2; br = br2; break;
        }

        // Solve quadratic: a*(ar*tau+br)^2 + b*(at*tau+bt)^2 + c*(ap*tau+bp)^2 - 2*f*(at*tau+bt)*(ap*tau+bp) = s^2
        CUSTOMREAL eqn_a = fac_a[i_node] * ar*ar + fac_b[i_node] * at*at
                         + fac_c[i_node] * ap*ap - _2_CR * fac_f[i_node] * at * ap;
        CUSTOMREAL eqn_b = _2_CR * fac_a[i_node] * ar * br + _2_CR * fac_b[i_node] * at * bt
                         + _2_CR * fac_c[i_node] * ap * bp - _2_CR * fac_f[i_node] * (at*bp + bt*ap);
        CUSTOMREAL eqn_c = fac_a[i_node] * br*br + fac_b[i_node] * bt*bt
                         + fac_c[i_node] * bp*bp - _2_CR * fac_f[i_node] * bt * bp
                         - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_tau;
                if (i_solution == 0) tmp_tau = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_tau = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                // Check causality
                CUSTOMREAL T_r = ar * tmp_tau + br;
                CUSTOMREAL T_t = at * tmp_tau + bt;
                CUSTOMREAL T_p = ap * tmp_tau + bp;
                CUSTOMREAL charact_r = fac_a[i_node] * T_r;
                CUSTOMREAL charact_t = fac_b[i_node] * T_t - fac_f[i_node] * T_p;
                CUSTOMREAL charact_p = fac_c[i_node] * T_p - fac_f[i_node] * T_t;

                bool is_causality = false;
                switch (i_case) {
                    case 0: if (charact_p >= 0 && charact_t >= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 1: if (charact_p >= 0 && charact_t >= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                    case 2: if (charact_p >= 0 && charact_t <= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 3: if (charact_p >= 0 && charact_t <= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                    case 4: if (charact_p <= 0 && charact_t >= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 5: if (charact_p <= 0 && charact_t >= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                    case 6: if (charact_p <= 0 && charact_t <= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 7: if (charact_p <= 0 && charact_t <= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_tau;
                }
            }
        }
    }

    // Second catalog: 12 triangle cases (2D surfaces)
    // r-t plane (cases 0-3): force H_p3 = c*T_p - f*T_t = 0 -> a*T_r^2 + (bc-f^2)/c*T_t^2 = s^2
    for (int i_case = 0; i_case < 4; i_case++) {
        CUSTOMREAL at = 0, bt = 0, ar = 0, br = 0;
        switch (i_case) {
            case 0: if (j == 0 || k == 0) continue; at = at1; bt = bt1; ar = ar1; br = br1; break;
            case 1: if (j == 0 || k == nr-1) continue; at = at1; bt = bt1; ar = ar2; br = br2; break;
            case 2: if (j == nt-1 || k == 0) continue; at = at2; bt = bt2; ar = ar1; br = br1; break;
            case 3: if (j == nt-1 || k == nr-1) continue; at = at2; bt = bt2; ar = ar2; br = br2; break;
        }

        // Solve: a*(ar*tau+br)^2 + (bc-f^2)/c*(at*tau+bt)^2 = s^2
        CUSTOMREAL eqn_a = fac_a[i_node] * ar*ar + bc_over_c * at*at;
        CUSTOMREAL eqn_b = _2_CR * fac_a[i_node] * ar * br + _2_CR * bc_over_c * at * bt;
        CUSTOMREAL eqn_c = fac_a[i_node] * br*br + bc_over_c * bt*bt - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_tau;
                if (i_solution == 0) tmp_tau = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_tau = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                CUSTOMREAL T_r = ar * tmp_tau + br;
                CUSTOMREAL T_t = at * tmp_tau + bt;
                CUSTOMREAL charact_r = fac_a[i_node] * T_r;
                CUSTOMREAL charact_t = bc_over_c * T_t;

                bool is_causality = false;
                switch (i_case) {
                    case 0: if (charact_t >= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 1: if (charact_t >= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                    case 2: if (charact_t <= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 3: if (charact_t <= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_tau;
                }
            }
        }
    }

    // r-p plane (cases 4-7): force H_p2 = b*T_t - f*T_p = 0 -> a*T_r^2 + (bc-f^2)/b*T_p^2 = s^2
    for (int i_case = 4; i_case < 8; i_case++) {
        CUSTOMREAL ap = 0, bp = 0, ar = 0, br = 0;
        switch (i_case) {
            case 4: if (i == 0 || k == 0) continue; ap = ap1; bp = bp1; ar = ar1; br = br1; break;
            case 5: if (i == 0 || k == nr-1) continue; ap = ap1; bp = bp1; ar = ar2; br = br2; break;
            case 6: if (i == np-1 || k == 0) continue; ap = ap2; bp = bp2; ar = ar1; br = br1; break;
            case 7: if (i == np-1 || k == nr-1) continue; ap = ap2; bp = bp2; ar = ar2; br = br2; break;
        }

        // Solve: a*(ar*tau+br)^2 + (bc-f^2)/b*(ap*tau+bp)^2 = s^2
        CUSTOMREAL eqn_a = fac_a[i_node] * ar*ar + bc_over_b * ap*ap;
        CUSTOMREAL eqn_b = _2_CR * fac_a[i_node] * ar * br + _2_CR * bc_over_b * ap * bp;
        CUSTOMREAL eqn_c = fac_a[i_node] * br*br + bc_over_b * bp*bp - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_tau;
                if (i_solution == 0) tmp_tau = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_tau = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                CUSTOMREAL T_r = ar * tmp_tau + br;
                CUSTOMREAL T_p = ap * tmp_tau + bp;
                CUSTOMREAL charact_r = fac_a[i_node] * T_r;
                CUSTOMREAL charact_p = bc_over_b * T_p;

                bool is_causality = false;
                switch (i_case) {
                    case 4: if (charact_p >= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 5: if (charact_p >= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                    case 6: if (charact_p <= 0 && charact_r >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 7: if (charact_p <= 0 && charact_r <= 0 && tmp_tau > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_tau;
                }
            }
        }
    }

    // t-p plane (cases 8-11): force H_p1 = T_r = 0 -> b*T_t^2 + c*T_p^2 - 2f*T_t*T_p = s^2
    for (int i_case = 8; i_case < 12; i_case++) {
        CUSTOMREAL ap = 0, bp = 0, at = 0, bt = 0;
        switch (i_case) {
            case 8:  if (i == 0 || j == 0) continue; ap = ap1; bp = bp1; at = at1; bt = bt1; break;
            case 9:  if (i == 0 || j == nt-1) continue; ap = ap1; bp = bp1; at = at2; bt = bt2; break;
            case 10: if (i == np-1 || j == 0) continue; ap = ap2; bp = bp2; at = at1; bt = bt1; break;
            case 11: if (i == np-1 || j == nt-1) continue; ap = ap2; bp = bp2; at = at2; bt = bt2; break;
        }

        // Solve: b*(at*tau+bt)^2 + c*(ap*tau+bp)^2 - 2f*(at*tau+bt)*(ap*tau+bp) = s^2
        CUSTOMREAL eqn_a = fac_b[i_node] * at*at
                         + fac_c[i_node] * ap*ap - _2_CR * fac_f[i_node] * at * ap;
        CUSTOMREAL eqn_b = _2_CR * fac_b[i_node] * at * bt
                         + _2_CR * fac_c[i_node] * ap * bp - _2_CR * fac_f[i_node] * (at*bp + bt*ap);
        CUSTOMREAL eqn_c = fac_b[i_node] * bt*bt
                         + fac_c[i_node] * bp*bp - _2_CR * fac_f[i_node] * bt * bp - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_tau;
                if (i_solution == 0) tmp_tau = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_tau = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                CUSTOMREAL T_t = at * tmp_tau + bt;
                CUSTOMREAL T_p = ap * tmp_tau + bp;
                CUSTOMREAL charact_t = fac_b[i_node] * T_t - fac_f[i_node] * T_p;
                CUSTOMREAL charact_p = fac_c[i_node] * T_p - fac_f[i_node] * T_t;

                bool is_causality = false;
                switch (i_case) {
                    case 8:  if (charact_p >= 0 && charact_t >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 9:  if (charact_p >= 0 && charact_t <= 0 && tmp_tau > 0) is_causality = true; break;
                    case 10: if (charact_p <= 0 && charact_t >= 0 && tmp_tau > 0) is_causality = true; break;
                    case 11: if (charact_p <= 0 && charact_t <= 0 && tmp_tau > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_tau;
                }
            }
        }
    }

    // Third catalog: 6 line cases (1D, 2 per axis)
    // r-axis (cases 0-1)
    for (int i_case = 0; i_case < 2; i_case++) {
        CUSTOMREAL ar = 0, br = 0;
        switch (i_case) {
            case 0: if (k == 0) continue; ar = ar1; br = br1; break;
            case 1: if (k == nr-1) continue; ar = ar2; br = br2; break;
        }

        CUSTOMREAL fun_loc_sqrt = sqrt(fun_loc_sq / fac_a[i_node]);
        CUSTOMREAL one_over_a = _1_CR / ar;
        for (int i_solution = 0; i_solution < 2; i_solution++) {
            CUSTOMREAL tmp_tau;
            if (i_solution == 0) tmp_tau = (fun_loc_sqrt - br) * one_over_a;
            else                 tmp_tau = (-fun_loc_sqrt - br) * one_over_a;

            // Check causality: compare traveltime with neighbor (global-indexed T0v)
            int ii_mr = i__j__km1[i_node]; // k-1 neighbor
            int ii_pr = i__j__kp1[i_node]; // k+1 neighbor
            bool is_causality = false;
            if (i_case == 0) {
                if (tmp_tau * T0v[i_node] > tau[ii_mr] * T0v_glob[ii_mr]
                    && _2_CR * tmp_tau > tau[ii_mr] && tmp_tau > 0)
                    is_causality = true;
            } else {
                if (tmp_tau * T0v[i_node] > tau[ii_pr] * T0v_glob[ii_pr]
                    && _2_CR * tmp_tau > tau[ii_pr] && tmp_tau > 0)
                    is_causality = true;
            }

            if (is_causality && count_cand < 52) {
                cand[count_cand++] = tmp_tau;
            }
        }
    }

    // t-axis (cases 2-3)
    for (int i_case = 2; i_case < 4; i_case++) {
        CUSTOMREAL at = 0, bt = 0;
        switch (i_case) {
            case 2: if (j == 0) continue; at = at1; bt = bt1; break;
            case 3: if (j == nt-1) continue; at = at2; bt = bt2; break;
        }

        CUSTOMREAL fun_loc_sqrt = sqrt(fun_loc_sq * fac_c[i_node] / bc_f2);
        CUSTOMREAL one_over_a = _1_CR / at;
        for (int i_solution = 0; i_solution < 2; i_solution++) {
            CUSTOMREAL tmp_tau;
            if (i_solution == 0) tmp_tau = (fun_loc_sqrt - bt) * one_over_a;
            else                 tmp_tau = (-fun_loc_sqrt - bt) * one_over_a;

            bool is_causality = false;
            if (i_case == 2) {
                int ii_mt = i__jm1k__[i_node];
                if (tmp_tau * T0v[i_node] > tau[ii_mt] * T0v_glob[ii_mt]
                    && _2_CR * tmp_tau > tau[ii_mt] && tmp_tau > 0)
                    is_causality = true;
            } else {
                int ii_pt = i__jp1k__[i_node];
                if (tmp_tau * T0v[i_node] > tau[ii_pt] * T0v_glob[ii_pt]
                    && _2_CR * tmp_tau > tau[ii_pt] && tmp_tau > 0)
                    is_causality = true;
            }

            if (is_causality && count_cand < 52) {
                cand[count_cand++] = tmp_tau;
            }
        }
    }

    // p-axis (cases 4-5)
    for (int i_case = 4; i_case < 6; i_case++) {
        CUSTOMREAL ap = 0, bp = 0;
        switch (i_case) {
            case 4: if (i == 0) continue; ap = ap1; bp = bp1; break;
            case 5: if (i == np-1) continue; ap = ap2; bp = bp2; break;
        }

        CUSTOMREAL fun_loc_sqrt = sqrt(fun_loc_sq * fac_b[i_node] / bc_f2);
        CUSTOMREAL one_over_a = _1_CR / ap;
        for (int i_solution = 0; i_solution < 2; i_solution++) {
            CUSTOMREAL tmp_tau;
            if (i_solution == 0) tmp_tau = (fun_loc_sqrt - bp) * one_over_a;
            else                 tmp_tau = (-fun_loc_sqrt - bp) * one_over_a;

            bool is_causality = false;
            if (i_case == 4) {
                int ii_mp = im1j__k__[i_node];
                if (tmp_tau * T0v[i_node] > tau[ii_mp] * T0v_glob[ii_mp]
                    && _2_CR * tmp_tau > tau[ii_mp] && tmp_tau > 0)
                    is_causality = true;
            } else {
                int ii_pp = ip1j__k__[i_node];
                if (tmp_tau * T0v[i_node] > tau[ii_pp] * T0v_glob[ii_pp]
                    && _2_CR * tmp_tau > tau[ii_pp] && tmp_tau > 0)
                    is_causality = true;
            }

            if (is_causality && count_cand < 52) {
                cand[count_cand++] = tmp_tau;
            }
        }
    }

    // Final: take minimum candidate as updated value
    for (int i_cand = 0; i_cand < count_cand; i_cand++) {
        tau[ii] = min(tau[ii], cand[i_cand]);
    }
}

// ============================================================================
// ADJOINT solver GPU kernel
// Ports Iterator::calculate_stencil_adj (adjoint transport sweeping) to GPU.
// Linear update (no quadratic): reads the frozen traveltime field T, anisotropy
// (zeta/xi/eta), the adjoint source (tau_old), and spherical coordinate factors;
// the adjoint field itself is stored in `tau` (same memory reuse as the CPU code).
// Boundary nodes on the physical domain edge are zeroed (calculate_boundary_nodes_adj).
// ============================================================================

__global__ void cuda_do_sweep_level_kernel_adj(
    const int i__j__k__[],       // global flat index of each node on this level
    const CUSTOMREAL T[],        // full-grid traveltime field
    const CUSTOMREAL zeta[],
    const CUSTOMREAL xi[],
    const CUSTOMREAL eta[],
    const CUSTOMREAL tau_old[],  // adjoint source (delta) field
    const CUSTOMREAL oor[],      // one_over_r_loc_1d      [loc_K]
    const CUSTOMREAL oor_sq[],   // one_over_r_loc_1d_sq   [loc_K]
    const CUSTOMREAL ooc[],      // one_over_cos_t_loc     [loc_J]
    const CUSTOMREAL ooc_sq[],   // one_over_cos_t_loc_sq  [loc_J]
    const CUSTOMREAL sint[],     // sin_t_loc              [loc_J]
    const CUSTOMREAL cosm[],     // cos_t_loc_m0p5         [loc_J]
    const CUSTOMREAL cosp[],     // cos_t_loc_p0p5         [loc_J]
    CUSTOMREAL tau[],            // adjoint field (in/out)
    const int loc_I, const int loc_J, const int loc_K,
    const CUSTOMREAL dr, const CUSTOMREAL dt, const CUSTOMREAL dp,
    const int ifirst, const int ilast,
    const int jfirst, const int jlast,
    const int kfirst, const int klast,
    const int n_nodes_this_level,
    const int i_start)
{
    unsigned int i_node = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i_node >= n_nodes_this_level) return;
    i_node += i_start;

    int ii = i__j__k__[i_node];
    int np = loc_I, nt = loc_J, nr = loc_K;

    int k = ii / (loc_I * loc_J);
    int j = (ii - k * loc_I * loc_J) / loc_I;
    int i = ii - k * loc_I * loc_J - j * loc_I;

    // physical boundary: zero adjoint field on domain-edge boundary nodes
    if (i == 0 || j == 0 || k == 0 || i == np-1 || j == nt-1 || k == nr-1) {
        if (   (i == 0    && ifirst) || (i == np-1 && ilast)
            || (j == 0    && jfirst) || (j == nt-1 && jlast)
            || (k == 0    && kfirst) || (k == nr-1 && klast)) {
            tau[ii] = _0_CR;
        }
        return;
    }

    const CUSTOMREAL dr_inv = _1_CR / dr;
    const CUSTOMREAL dt_inv = _1_CR / dt;
    const CUSTOMREAL dp_inv = _1_CR / dp;

    int ii_mp    = ii - 1;
    int ii_pp    = ii + 1;
    int ii_mt    = ii - loc_I;
    int ii_pt    = ii + loc_I;
    int ii_mr    = ii - loc_I * loc_J;
    int ii_pr    = ii + loc_I * loc_J;
    int ii_pp_pt = ii + 1 + loc_I;
    int ii_mp_pt = ii - 1 + loc_I;
    int ii_pp_mt = ii + 1 - loc_I;
    int ii_mp_mt = ii - 1 - loc_I;

    CUSTOMREAL one_over_r_loc_1d_kkr    = oor[k];
    CUSTOMREAL one_over_r_loc_1d_kkr_sq = oor_sq[k];
    CUSTOMREAL one_over_cos_t_loc_jjt   = ooc[j];
    CUSTOMREAL sin_t_loc_jjt            = sint[j];
    CUSTOMREAL one_over_r_loc_cos_t_loc_sq = one_over_r_loc_1d_kkr_sq * ooc_sq[j];

    CUSTOMREAL cos_t_loc_m0p5_jjt = cosm[j];
    CUSTOMREAL cos_t_loc_p0p5_jjt = cosp[j];
    CUSTOMREAL tmp_T_ip = T[ii_pp] - T[ii_mp];
    CUSTOMREAL tmp_T_jt = T[ii_pt] - T[ii_mt];

    // a1(-0.5r) = - (1 + 2*zeta) & T_r
    CUSTOMREAL a1  = - (_1_CR+zeta[ii_mr]+zeta[ii]) * (T[ii]-T[ii_mr]) * dr_inv;
    CUSTOMREAL a1m = (a1 - fabs(a1));
    CUSTOMREAL a1p = (a1 + fabs(a1));

    // a2(+0.5r) = - (1 + 2*zeta) & T_r
    CUSTOMREAL a2  = - (_1_CR+zeta[ii]+zeta[ii_pr]) * (T[ii_pr]-T[ii]) * dr_inv;
    CUSTOMREAL a2m = (a2 - fabs(a2));
    CUSTOMREAL a2p = (a2 + fabs(a2));

    // b1(-0.5t) = - (1-2xi) * (1/r^2) * T_t - 2*eta * (1/r^2cos(t)) * T_p
    CUSTOMREAL b1  = - (_1_CR-xi[ii_mt]-xi[ii]) * one_over_r_loc_1d_kkr_sq * (T[ii]-T[ii_mt]) * dt_inv
                     - (      eta[ii_mt]+eta[ii]) * one_over_r_loc_1d_kkr_sq * cos_t_loc_m0p5_jjt * _0_25_CR * dp_inv
                     * ((T[ii_pp_mt] - T[ii_mp_mt]) + (T[ii_pp] - T[ii_mp]));
    CUSTOMREAL b1m = (b1 - fabs(b1));
    CUSTOMREAL b1p = (b1 + fabs(b1));

    // b2(+0.5t) = - (1-2xi) * (1/r^2) * T_t - 2*eta * (1/r^2cos(t)) * T_p
    CUSTOMREAL b2  = - (_1_CR-xi[ii]-xi[ii_pt]) * one_over_r_loc_1d_kkr_sq * (T[ii_pt]-T[ii]) * dt_inv
                     - (      eta[ii]+eta[ii_pt]) * one_over_r_loc_1d_kkr_sq * cos_t_loc_p0p5_jjt * _0_25_CR * dp_inv
                     * (tmp_T_ip + (T[ii_pp_pt]-T[ii_mp_pt]));
    CUSTOMREAL b2m = (b2 - fabs(b2));
    CUSTOMREAL b2p = (b2 + fabs(b2));

    // c1(-0.5p) = - (1 + 2*xi) * (1/r^2cos(t)) * T_p - 2*eta * (1/(r*cos(t))^2) * T_t
    CUSTOMREAL c1  = - (_1_CR+xi[ii_mp]+xi[ii]) * one_over_r_loc_cos_t_loc_sq
                      * (T[ii]-T[ii_mp]) * dp_inv
                      - (eta[ii_mp]+eta[ii]) * one_over_r_loc_cos_t_loc_sq * _0_25_CR * dt_inv
                      * ((T[ii_mp_pt]-T[ii_mp_mt]) + (T[ii_pt]-T[ii_mt]));
    CUSTOMREAL c1m = (c1 - fabs(c1));
    CUSTOMREAL c1p = (c1 + fabs(c1));

    // c2(+0.5p) = - (1 + 2*xi) * (1/r^2cos(t)) * T_p - 2*eta * (1/(r*cos(t))^2) * T_t
    CUSTOMREAL c2  = - (_1_CR+xi[ii]+xi[ii_pp]) * one_over_r_loc_cos_t_loc_sq
                      * (T[ii_pp]-T[ii]) * dp_inv
                      - (eta[ii]+eta[ii_pp]) * one_over_r_loc_cos_t_loc_sq * _0_25_CR * dt_inv
                      * (tmp_T_jt + (T[ii_pp_pt]-T[ii_pp_mt]));
    CUSTOMREAL c2m = (c2 - fabs(c2));
    CUSTOMREAL c2p = (c2 + fabs(c2));

    // additional divergence terms in spherical coordinates
    CUSTOMREAL d   = - (_1_CR+_2_CR*zeta[ii]) * one_over_r_loc_1d_kkr
                     * (T[ii_pr]-T[ii_mr]) * dr_inv
                     + (_0_5_CR-xi[ii]) * sin_t_loc_jjt * one_over_r_loc_1d_kkr_sq * one_over_cos_t_loc_jjt
                     * tmp_T_jt * dt_inv
                     + eta[ii] * sin_t_loc_jjt * one_over_r_loc_cos_t_loc_sq
                     * tmp_T_ip * dp_inv;

    // coe  (half factor folded in, as in the CPU code)
    CUSTOMREAL coe = _0_5_CR * ( (a2p-a1m) * dr_inv + (b2p-b1m) * dt_inv + (c2p-c1m) * dp_inv );

    if (fabs(coe) < (CUSTOMREAL)1.e-6) {   // isZeroAdj: epsAdj = 1e-6 (include/config.h)
        tau[ii] = _0_CR;
    } else {
        // hamiltonian
        CUSTOMREAL Hadj = _0_5_CR * ( (a1p*tau[ii_mr] - a2m*tau[ii_pr]) * dr_inv
                        + (b1p*tau[ii_mt] - b2m*tau[ii_pt]) * dt_inv
                        + (c1p*tau[ii_mp] - c2m*tau[ii_pp]) * dp_inv );

        tau[ii] = (tau_old[ii] + Hadj) / (coe + d);
    }
}

// Run ADJOINT iteration on GPU (one kernel launch per level, same as cuda_run_iteration_upwind)
void cuda_run_iteration_adjoint(Grid_on_device* grid_dv, int const& iswp,
                                int const& ifirst, int const& ilast,
                                int const& jfirst, int const& jlast,
                                int const& kfirst, int const& klast) {

    initialize_sweep_params(grid_dv);

    int block_size = CUDA_SWEEPING_BLOCK_SIZE;
    int num_blocks_x, num_blocks_y;
    int i_node_offset = 0;

    for (size_t i_level = 0; i_level < grid_dv->n_levels_host; i_level++) {
        get_block_xy(ceil(grid_dv->n_nodes_on_levels_host[i_level] / block_size + 0.5), &num_blocks_x, &num_blocks_y);
        dim3 grid_each(num_blocks_x, num_blocks_y);
        dim3 threads_each(block_size, 1, 1);

        int id_stream = i_level;

        const int* level_nodes;
        switch (iswp) {
            case 0: level_nodes = grid_dv->vv_i__j__k___0; break;
            case 1: level_nodes = grid_dv->vv_i__j__k___1; break;
            case 2: level_nodes = grid_dv->vv_i__j__k___2; break;
            case 3: level_nodes = grid_dv->vv_i__j__k___3; break;
            case 4: level_nodes = grid_dv->vv_i__j__k___4; break;
            case 5: level_nodes = grid_dv->vv_i__j__k___5; break;
            case 6: level_nodes = grid_dv->vv_i__j__k___6; break;
            default: level_nodes = grid_dv->vv_i__j__k___7; break;
        }

        void* kernelArgs[] = {
            (void*)&level_nodes,
            &(grid_dv->T_glob), &(grid_dv->zeta_glob), &(grid_dv->xi_glob), &(grid_dv->eta_glob),
            &(grid_dv->tau_old_glob),
            &(grid_dv->adj_one_over_r), &(grid_dv->adj_one_over_r_sq),
            &(grid_dv->adj_one_over_cos_t), &(grid_dv->adj_one_over_cos_t_sq),
            &(grid_dv->adj_sin_t), &(grid_dv->adj_cos_t_m0p5), &(grid_dv->adj_cos_t_p0p5),
            &(grid_dv->tau),
            &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
            &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
            (void*)&ifirst, (void*)&ilast, (void*)&jfirst, (void*)&jlast, (void*)&kfirst, (void*)&klast,
            &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
        };
        print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_adj, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 41000+iswp);
        // levels are sequential (same as CPU): wait for this level before the next
        print_CUDA_error_if_any(cudaStreamSynchronize(grid_dv->level_streams[id_stream]), 41100+iswp);

        i_node_offset += grid_dv->n_nodes_on_levels_host[i_level];
    }

    finalize_sweep_params(grid_dv);
}

// Run UPWIND iteration on GPU (same orchestration as cuda_run_iteration_forward)
void cuda_run_iteration_upwind(Grid_on_device* grid_dv, int const& iswp) {

    initialize_sweep_params(grid_dv);

    int block_size = CUDA_SWEEPING_BLOCK_SIZE;
    int num_blocks_x, num_blocks_y;
    int i_node_offset = 0;

    for (size_t i_level = 0; i_level < grid_dv->n_levels_host; i_level++) {
        get_block_xy(ceil(grid_dv->n_nodes_on_levels_host[i_level] / block_size + 0.5), &num_blocks_x, &num_blocks_y);
        dim3 grid_each(num_blocks_x, num_blocks_y);
        dim3 threads_each(block_size, 1, 1);

        // Launch UPWIND kernel for this level
        int id_stream = i_level;

        if (iswp == 0) {
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___0), &(grid_dv->vv_ip1j__k___0), &(grid_dv->vv_im1j__k___0),
                &(grid_dv->vv_i__jp1k___0), &(grid_dv->vv_i__jm1k___0), &(grid_dv->vv_i__j__kp1_0), &(grid_dv->vv_i__j__km1_0),
                &(grid_dv->vv_fac_a_0), &(grid_dv->vv_fac_b_0), &(grid_dv->vv_fac_c_0), &(grid_dv->vv_fac_f_0),
                &(grid_dv->vv_T0v_0), &(grid_dv->vv_T0r_0), &(grid_dv->vv_T0t_0), &(grid_dv->vv_T0p_0),
                &(grid_dv->vv_fun_0), &(grid_dv->vv_change_0), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40001);
        } else if (iswp == 1) {
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___1), &(grid_dv->vv_ip1j__k___1), &(grid_dv->vv_im1j__k___1),
                &(grid_dv->vv_i__jp1k___1), &(grid_dv->vv_i__jm1k___1), &(grid_dv->vv_i__j__kp1_1), &(grid_dv->vv_i__j__km1_1),
                &(grid_dv->vv_fac_a_1), &(grid_dv->vv_fac_b_1), &(grid_dv->vv_fac_c_1), &(grid_dv->vv_fac_f_1),
                &(grid_dv->vv_T0v_1), &(grid_dv->vv_T0r_1), &(grid_dv->vv_T0t_1), &(grid_dv->vv_T0p_1),
                &(grid_dv->vv_fun_1), &(grid_dv->vv_change_1), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40002);
        } else if (iswp == 2) {
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___2), &(grid_dv->vv_ip1j__k___2), &(grid_dv->vv_im1j__k___2),
                &(grid_dv->vv_i__jp1k___2), &(grid_dv->vv_i__jm1k___2), &(grid_dv->vv_i__j__kp1_2), &(grid_dv->vv_i__j__km1_2),
                &(grid_dv->vv_fac_a_2), &(grid_dv->vv_fac_b_2), &(grid_dv->vv_fac_c_2), &(grid_dv->vv_fac_f_2),
                &(grid_dv->vv_T0v_2), &(grid_dv->vv_T0r_2), &(grid_dv->vv_T0t_2), &(grid_dv->vv_T0p_2),
                &(grid_dv->vv_fun_2), &(grid_dv->vv_change_2), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40003);
        } else if (iswp == 3) {
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___3), &(grid_dv->vv_ip1j__k___3), &(grid_dv->vv_im1j__k___3),
                &(grid_dv->vv_i__jp1k___3), &(grid_dv->vv_i__jm1k___3), &(grid_dv->vv_i__j__kp1_3), &(grid_dv->vv_i__j__km1_3),
                &(grid_dv->vv_fac_a_3), &(grid_dv->vv_fac_b_3), &(grid_dv->vv_fac_c_3), &(grid_dv->vv_fac_f_3),
                &(grid_dv->vv_T0v_3), &(grid_dv->vv_T0r_3), &(grid_dv->vv_T0t_3), &(grid_dv->vv_T0p_3),
                &(grid_dv->vv_fun_3), &(grid_dv->vv_change_3), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40004);
        } else if (iswp == 4) {
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___4), &(grid_dv->vv_ip1j__k___4), &(grid_dv->vv_im1j__k___4),
                &(grid_dv->vv_i__jp1k___4), &(grid_dv->vv_i__jm1k___4), &(grid_dv->vv_i__j__kp1_4), &(grid_dv->vv_i__j__km1_4),
                &(grid_dv->vv_fac_a_4), &(grid_dv->vv_fac_b_4), &(grid_dv->vv_fac_c_4), &(grid_dv->vv_fac_f_4),
                &(grid_dv->vv_T0v_4), &(grid_dv->vv_T0r_4), &(grid_dv->vv_T0t_4), &(grid_dv->vv_T0p_4),
                &(grid_dv->vv_fun_4), &(grid_dv->vv_change_4), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40005);
        } else if (iswp == 5) {
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___5), &(grid_dv->vv_ip1j__k___5), &(grid_dv->vv_im1j__k___5),
                &(grid_dv->vv_i__jp1k___5), &(grid_dv->vv_i__jm1k___5), &(grid_dv->vv_i__j__kp1_5), &(grid_dv->vv_i__j__km1_5),
                &(grid_dv->vv_fac_a_5), &(grid_dv->vv_fac_b_5), &(grid_dv->vv_fac_c_5), &(grid_dv->vv_fac_f_5),
                &(grid_dv->vv_T0v_5), &(grid_dv->vv_T0r_5), &(grid_dv->vv_T0t_5), &(grid_dv->vv_T0p_5),
                &(grid_dv->vv_fun_5), &(grid_dv->vv_change_5), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40006);
        } else if (iswp == 6) {
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___6), &(grid_dv->vv_ip1j__k___6), &(grid_dv->vv_im1j__k___6),
                &(grid_dv->vv_i__jp1k___6), &(grid_dv->vv_i__jm1k___6), &(grid_dv->vv_i__j__kp1_6), &(grid_dv->vv_i__j__km1_6),
                &(grid_dv->vv_fac_a_6), &(grid_dv->vv_fac_b_6), &(grid_dv->vv_fac_c_6), &(grid_dv->vv_fac_f_6),
                &(grid_dv->vv_T0v_6), &(grid_dv->vv_T0r_6), &(grid_dv->vv_T0t_6), &(grid_dv->vv_T0p_6),
                &(grid_dv->vv_fun_6), &(grid_dv->vv_change_6), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40007);
        } else { // iswp == 7
            void* kernelArgs[] = {
                &(grid_dv->vv_i__j__k___7), &(grid_dv->vv_ip1j__k___7), &(grid_dv->vv_im1j__k___7),
                &(grid_dv->vv_i__jp1k___7), &(grid_dv->vv_i__jm1k___7), &(grid_dv->vv_i__j__kp1_7), &(grid_dv->vv_i__j__km1_7),
                &(grid_dv->vv_fac_a_7), &(grid_dv->vv_fac_b_7), &(grid_dv->vv_fac_c_7), &(grid_dv->vv_fac_f_7),
                &(grid_dv->vv_T0v_7), &(grid_dv->vv_T0r_7), &(grid_dv->vv_T0t_7), &(grid_dv->vv_T0p_7),
                &(grid_dv->vv_fun_7), &(grid_dv->vv_change_7), &(grid_dv->tau),
                &(grid_dv->T0v_glob),
                &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
                &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
                &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
            };
            print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 40008);
        }

        i_node_offset += grid_dv->n_nodes_on_levels_host[i_level];
    }

    finalize_sweep_params(grid_dv);
}

// ============================================================================
// TELESEISMIC UPWIND solver GPU kernel
// Port of Iterator::calculate_stencil_1st_order_upwind_tele.
// Same 26 cases (8 tetrahedron + 12 triangle + 6 line) as the main UPWIND solver,
// but solves directly for T (no T0*tau factorization, no T0v anisotropy coeffs):
//   T_p = ap*T + bp with ap=+-1/dp, bp=-/+T[ip-1/1]/dp etc.
// Line-case causality simplifies to  tmp_T >= T[neighbor] && tmp_T > 0.
// The `tau` device array holds the T field (host passes grid.T_loc).
// ============================================================================

__global__ void cuda_do_sweep_level_kernel_upwind_tele(
    const int i__j__k__[],
    const int ip1j__k__[],
    const int im1j__k__[],
    const int i__jp1k__[],
    const int i__jm1k__[],
    const int i__j__kp1[],
    const int i__j__km1[],
    const CUSTOMREAL fac_a[],
    const CUSTOMREAL fac_b[],
    const CUSTOMREAL fac_c[],
    const CUSTOMREAL fac_f[],
    const CUSTOMREAL fun[],
    CUSTOMREAL tau[],            // holds T
    const int loc_I,
    const int loc_J,
    const int loc_K,
    const CUSTOMREAL dr,
    const CUSTOMREAL dt,
    const CUSTOMREAL dp,
    const int n_nodes_this_level,
    const int i_start)
{
    unsigned int i_node = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (i_node >= n_nodes_this_level) return;
    i_node += i_start;

    int ii = i__j__k__[i_node];
    int k = ii / (loc_I * loc_J);
    int j = (ii - k * loc_I * loc_J) / loc_I;
    int i = ii - k * loc_I * loc_J - j * loc_I;

    int np = loc_I, nt = loc_J, nr = loc_K;

    // forward/backward finite-difference coeffs for T: T_p = ap*T + bp etc.
    CUSTOMREAL ap1 = 0, bp1 = 0, ap2 = 0, bp2 = 0;
    CUSTOMREAL at1 = 0, bt1 = 0, at2 = 0, bt2 = 0;
    CUSTOMREAL ar1 = 0, br1 = 0, ar2 = 0, br2 = 0;

    if (i > 0) {
        ap1 =  _1_CR / dp;
        bp1 = -tau[im1j__k__[i_node]] / dp;
    }
    if (i < np - 1) {
        ap2 = -_1_CR / dp;
        bp2 =  tau[ip1j__k__[i_node]] / dp;
    }
    if (j > 0) {
        at1 =  _1_CR / dt;
        bt1 = -tau[i__jm1k__[i_node]] / dt;
    }
    if (j < nt - 1) {
        at2 = -_1_CR / dt;
        bt2 =  tau[i__jp1k__[i_node]] / dt;
    }
    if (k > 0) {
        ar1 =  _1_CR / dr;
        br1 = -tau[i__j__km1[i_node]] / dr;
    }
    if (k < nr - 1) {
        ar2 = -_1_CR / dr;
        br2 =  tau[i__j__kp1[i_node]] / dr;
    }

    CUSTOMREAL fun_loc_sq = fun[i_node] * fun[i_node];
    CUSTOMREAL bc_f2 = fac_b[i_node] * fac_c[i_node] - fac_f[i_node] * fac_f[i_node];
    CUSTOMREAL bc_over_b = bc_f2 / fac_b[i_node];
    CUSTOMREAL bc_over_c = bc_f2 / fac_c[i_node];

    // candidates: 26 cases x 2 solutions
    CUSTOMREAL cand[52];
    int count_cand = 0;

    // ---- catalog 1: tetrahedra (3D volume) ----
    for (int i_case = 0; i_case < 8; i_case++) {
        CUSTOMREAL ap = 0, bp = 0, at = 0, bt = 0, ar = 0, br = 0;
        switch (i_case) {
            case 0: if (i == 0 || j == 0 || k == 0) continue; ap = ap1; bp = bp1; at = at1; bt = bt1; ar = ar1; br = br1; break;
            case 1: if (i == 0 || j == 0 || k == nr-1) continue; ap = ap1; bp = bp1; at = at1; bt = bt1; ar = ar2; br = br2; break;
            case 2: if (i == 0 || j == nt-1 || k == 0) continue; ap = ap1; bp = bp1; at = at2; bt = bt2; ar = ar1; br = br1; break;
            case 3: if (i == 0 || j == nt-1 || k == nr-1) continue; ap = ap1; bp = bp1; at = at2; bt = bt2; ar = ar2; br = br2; break;
            case 4: if (i == np-1 || j == 0 || k == 0) continue; ap = ap2; bp = bp2; at = at1; bt = bt1; ar = ar1; br = br1; break;
            case 5: if (i == np-1 || j == 0 || k == nr-1) continue; ap = ap2; bp = bp2; at = at1; bt = bt1; ar = ar2; br = br2; break;
            case 6: if (i == np-1 || j == nt-1 || k == 0) continue; ap = ap2; bp = bp2; at = at2; bt = bt2; ar = ar1; br = br1; break;
            case 7: if (i == np-1 || j == nt-1 || k == nr-1) continue; ap = ap2; bp = bp2; at = at2; bt = bt2; ar = ar2; br = br2; break;
        }

        CUSTOMREAL eqn_a = fac_a[i_node] * ar*ar + fac_b[i_node] * at*at
                         + fac_c[i_node] * ap*ap - _2_CR * fac_f[i_node] * at * ap;
        CUSTOMREAL eqn_b = _2_CR * fac_a[i_node] * ar * br + _2_CR * fac_b[i_node] * at * bt
                         + _2_CR * fac_c[i_node] * ap * bp - _2_CR * fac_f[i_node] * (at*bp + bt*ap);
        CUSTOMREAL eqn_c = fac_a[i_node] * br*br + fac_b[i_node] * bt*bt
                         + fac_c[i_node] * bp*bp - _2_CR * fac_f[i_node] * bt * bp
                         - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_T;
                if (i_solution == 0) tmp_T = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_T = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                CUSTOMREAL T_r = ar * tmp_T + br;
                CUSTOMREAL T_t = at * tmp_T + bt;
                CUSTOMREAL T_p = ap * tmp_T + bp;
                CUSTOMREAL charact_r = fac_a[i_node] * T_r;
                CUSTOMREAL charact_t = fac_b[i_node] * T_t - fac_f[i_node] * T_p;
                CUSTOMREAL charact_p = fac_c[i_node] * T_p - fac_f[i_node] * T_t;

                bool is_causality = false;
                switch (i_case) {
                    case 0: if (charact_p >= 0 && charact_t >= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 1: if (charact_p >= 0 && charact_t >= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                    case 2: if (charact_p >= 0 && charact_t <= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 3: if (charact_p >= 0 && charact_t <= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                    case 4: if (charact_p <= 0 && charact_t >= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 5: if (charact_p <= 0 && charact_t >= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                    case 6: if (charact_p <= 0 && charact_t <= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 7: if (charact_p <= 0 && charact_t <= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_T;
                }
            }
        }
    }

    // ---- catalog 2: triangles (2D), 12 cases: r-t / r-p / t-p planes ----
    // r-t plane (0-3):  a*T_r^2 + (bc-f^2)/c*T_t^2 = s^2
    for (int i_case = 0; i_case < 4; i_case++) {
        CUSTOMREAL at = 0, bt = 0, ar = 0, br = 0;
        switch (i_case) {
            case 0: if (j == 0 || k == 0) continue; at = at1; bt = bt1; ar = ar1; br = br1; break;
            case 1: if (j == 0 || k == nr-1) continue; at = at1; bt = bt1; ar = ar2; br = br2; break;
            case 2: if (j == nt-1 || k == 0) continue; at = at2; bt = bt2; ar = ar1; br = br1; break;
            case 3: if (j == nt-1 || k == nr-1) continue; at = at2; bt = bt2; ar = ar2; br = br2; break;
        }

        CUSTOMREAL eqn_a = fac_a[i_node] * ar*ar + bc_over_c * at*at;
        CUSTOMREAL eqn_b = _2_CR * fac_a[i_node] * ar * br + _2_CR * bc_over_c * at * bt;
        CUSTOMREAL eqn_c = fac_a[i_node] * br*br + bc_over_c * bt*bt - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_T;
                if (i_solution == 0) tmp_T = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_T = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                CUSTOMREAL T_r = ar * tmp_T + br;
                CUSTOMREAL T_t = at * tmp_T + bt;
                CUSTOMREAL charact_r = fac_a[i_node] * T_r;
                CUSTOMREAL charact_t = bc_over_c * T_t;

                bool is_causality = false;
                switch (i_case) {
                    case 0: if (charact_t >= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 1: if (charact_t >= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                    case 2: if (charact_t <= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 3: if (charact_t <= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_T;
                }
            }
        }
    }

    // r-p plane (4-7):  a*T_r^2 + (bc-f^2)/b*T_p^2 = s^2
    for (int i_case = 4; i_case < 8; i_case++) {
        CUSTOMREAL ap = 0, bp = 0, ar = 0, br = 0;
        switch (i_case) {
            case 4: if (i == 0 || k == 0) continue; ap = ap1; bp = bp1; ar = ar1; br = br1; break;
            case 5: if (i == 0 || k == nr-1) continue; ap = ap1; bp = bp1; ar = ar2; br = br2; break;
            case 6: if (i == np-1 || k == 0) continue; ap = ap2; bp = bp2; ar = ar1; br = br1; break;
            case 7: if (i == np-1 || k == nr-1) continue; ap = ap2; bp = bp2; ar = ar2; br = br2; break;
        }

        CUSTOMREAL eqn_a = fac_a[i_node] * ar*ar + bc_over_b * ap*ap;
        CUSTOMREAL eqn_b = _2_CR * fac_a[i_node] * ar * br + _2_CR * bc_over_b * ap * bp;
        CUSTOMREAL eqn_c = fac_a[i_node] * br*br + bc_over_b * bp*bp - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_T;
                if (i_solution == 0) tmp_T = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_T = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                CUSTOMREAL T_r = ar * tmp_T + br;
                CUSTOMREAL T_p = ap * tmp_T + bp;
                CUSTOMREAL charact_r = fac_a[i_node] * T_r;
                CUSTOMREAL charact_p = bc_over_b * T_p;

                bool is_causality = false;
                switch (i_case) {
                    case 4: if (charact_p >= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 5: if (charact_p >= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                    case 6: if (charact_p <= 0 && charact_r >= 0 && tmp_T > 0) is_causality = true; break;
                    case 7: if (charact_p <= 0 && charact_r <= 0 && tmp_T > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_T;
                }
            }
        }
    }

    // t-p plane (8-11):  b*T_t^2 + c*T_p^2 - 2f*T_t*T_p = s^2
    for (int i_case = 8; i_case < 12; i_case++) {
        CUSTOMREAL ap = 0, bp = 0, at = 0, bt = 0;
        switch (i_case) {
            case 8:  if (i == 0 || j == 0) continue; ap = ap1; bp = bp1; at = at1; bt = bt1; break;
            case 9:  if (i == 0 || j == nt-1) continue; ap = ap1; bp = bp1; at = at2; bt = bt2; break;
            case 10: if (i == np-1 || j == 0) continue; ap = ap2; bp = bp2; at = at1; bt = bt1; break;
            case 11: if (i == np-1 || j == nt-1) continue; ap = ap2; bp = bp2; at = at2; bt = bt2; break;
        }

        CUSTOMREAL eqn_a = fac_b[i_node] * at*at
                         + fac_c[i_node] * ap*ap - _2_CR * fac_f[i_node] * at * ap;
        CUSTOMREAL eqn_b = _2_CR * fac_b[i_node] * at * bt
                         + _2_CR * fac_c[i_node] * ap * bp - _2_CR * fac_f[i_node] * (at*bp + bt*ap);
        CUSTOMREAL eqn_c = fac_b[i_node] * bt*bt
                         + fac_c[i_node] * bp*bp - _2_CR * fac_f[i_node] * bt * bp - fun_loc_sq;
        CUSTOMREAL eqn_Delta = eqn_b*eqn_b - _4_CR * eqn_a * eqn_c;

        if (eqn_Delta >= 0) {
            CUSTOMREAL eqn_Delta_sqrt = sqrt(eqn_Delta);
            CUSTOMREAL one_over_a = _1_CR / (_2_CR * eqn_a);
            for (int i_solution = 0; i_solution < 2; i_solution++) {
                CUSTOMREAL tmp_T;
                if (i_solution == 0) tmp_T = (-eqn_b + eqn_Delta_sqrt) * one_over_a;
                else                 tmp_T = (-eqn_b - eqn_Delta_sqrt) * one_over_a;

                CUSTOMREAL T_t = at * tmp_T + bt;
                CUSTOMREAL T_p = ap * tmp_T + bp;
                CUSTOMREAL charact_t = fac_b[i_node] * T_t - fac_f[i_node] * T_p;
                CUSTOMREAL charact_p = fac_c[i_node] * T_p - fac_f[i_node] * T_t;

                bool is_causality = false;
                switch (i_case) {
                    case 8:  if (charact_p >= 0 && charact_t >= 0 && tmp_T > 0) is_causality = true; break;
                    case 9:  if (charact_p >= 0 && charact_t <= 0 && tmp_T > 0) is_causality = true; break;
                    case 10: if (charact_p <= 0 && charact_t >= 0 && tmp_T > 0) is_causality = true; break;
                    case 11: if (charact_p <= 0 && charact_t <= 0 && tmp_T > 0) is_causality = true; break;
                }

                if (is_causality && count_cand < 52) {
                    cand[count_cand++] = tmp_T;
                }
            }
        }
    }

    // ---- catalog 3: lines (1D), 6 cases ----
    // r-axis (0-1):  a*T_r^2 = s^2
    for (int i_case = 0; i_case < 2; i_case++) {
        CUSTOMREAL ar = 0, br = 0;
        switch (i_case) {
            case 0: if (k == 0) continue; ar = ar1; br = br1; break;
            case 1: if (k == nr-1) continue; ar = ar2; br = br2; break;
        }

        CUSTOMREAL fun_loc_sqrt = sqrt(fun_loc_sq / fac_a[i_node]);
        for (int i_solution = 0; i_solution < 2; i_solution++) {
            CUSTOMREAL tmp_T;
            if (i_solution == 0) tmp_T = ( fun_loc_sqrt - br) / ar;
            else                 tmp_T = (-fun_loc_sqrt - br) / ar;

            bool is_causality = false;
            if (i_case == 0) {
                int nbr = i__j__km1[i_node];
                if (tmp_T >= tau[nbr] && tmp_T > 0) is_causality = true;
            } else {
                int nbr = i__j__kp1[i_node];
                if (tmp_T >= tau[nbr] && tmp_T > 0) is_causality = true;
            }
            if (is_causality && count_cand < 52) {
                cand[count_cand++] = tmp_T;
            }
        }
    }

    // t-axis (2-3):  (bc-f^2)/c * T_t^2 = s^2
    for (int i_case = 2; i_case < 4; i_case++) {
        CUSTOMREAL at = 0, bt = 0;
        switch (i_case) {
            case 2: if (j == 0) continue; at = at1; bt = bt1; break;
            case 3: if (j == nt-1) continue; at = at2; bt = bt2; break;
        }

        CUSTOMREAL fun_loc_sqrt = sqrt(fun_loc_sq * fac_c[i_node] / bc_f2);
        for (int i_solution = 0; i_solution < 2; i_solution++) {
            CUSTOMREAL tmp_T;
            if (i_solution == 0) tmp_T = ( fun_loc_sqrt - bt) / at;
            else                 tmp_T = (-fun_loc_sqrt - bt) / at;

            bool is_causality = false;
            int nbr = (i_case == 2) ? i__jm1k__[i_node] : i__jp1k__[i_node];
            if (tmp_T >= tau[nbr] && tmp_T > 0) is_causality = true;
            if (is_causality && count_cand < 52) {
                cand[count_cand++] = tmp_T;
            }
        }
    }

    // p-axis (4-5):  (bc-f^2)/b * T_p^2 = s^2
    for (int i_case = 4; i_case < 6; i_case++) {
        CUSTOMREAL ap = 0, bp = 0;
        switch (i_case) {
            case 4: if (i == 0) continue; ap = ap1; bp = bp1; break;
            case 5: if (i == np-1) continue; ap = ap2; bp = bp2; break;
        }

        CUSTOMREAL fun_loc_sqrt = sqrt(fun_loc_sq * fac_b[i_node] / bc_f2);
        for (int i_solution = 0; i_solution < 2; i_solution++) {
            CUSTOMREAL tmp_T;
            if (i_solution == 0) tmp_T = ( fun_loc_sqrt - bp) / ap;
            else                 tmp_T = (-fun_loc_sqrt - bp) / ap;

            bool is_causality = false;
            int nbr = (i_case == 4) ? im1j__k__[i_node] : ip1j__k__[i_node];
            if (tmp_T >= tau[nbr] && tmp_T > 0) is_causality = true;
            if (is_causality && count_cand < 52) {
                cand[count_cand++] = tmp_T;
            }
        }
    }

    // final: minimum candidate wins
    for (int i_cand = 0; i_cand < count_cand; i_cand++) {
        tau[ii] = min(tau[ii], cand[i_cand]);
    }
}

// Run TELESEISMIC UPWIND iteration on GPU (same orchestration as cuda_run_iteration_upwind)
void cuda_run_iteration_upwind_tele(Grid_on_device* grid_dv, int const& iswp) {

    initialize_sweep_params(grid_dv);

    int block_size = CUDA_SWEEPING_BLOCK_SIZE;
    int num_blocks_x, num_blocks_y;
    int i_node_offset = 0;

    for (size_t i_level = 0; i_level < grid_dv->n_levels_host; i_level++) {
        get_block_xy(ceil(grid_dv->n_nodes_on_levels_host[i_level] / block_size + 0.5), &num_blocks_x, &num_blocks_y);
        dim3 grid_each(num_blocks_x, num_blocks_y);
        dim3 threads_each(block_size, 1, 1);

        int id_stream = i_level;

        const int*  p_ii;  const int*  p_ip1; const int*  p_im1;
        const int*  p_jp1; const int*  p_jm1; const int*  p_kp1; const int*  p_km1;
        const CUSTOMREAL*  p_fa;   const CUSTOMREAL*  p_fb;   const CUSTOMREAL*  p_fc;
        const CUSTOMREAL*  p_ff;   const CUSTOMREAL*  p_fun;

        switch (iswp) {
#define TATT_TELE_CASE(N) \
            case N: p_ii = grid_dv->vv_i__j__k___##N; p_ip1 = grid_dv->vv_ip1j__k___##N; p_im1 = grid_dv->vv_im1j__k___##N; \
                    p_jp1 = grid_dv->vv_i__jp1k___##N; p_jm1 = grid_dv->vv_i__jm1k___##N; p_kp1 = grid_dv->vv_i__j__kp1_##N; p_km1 = grid_dv->vv_i__j__km1_##N; \
                    p_fa = grid_dv->vv_fac_a_##N; p_fb = grid_dv->vv_fac_b_##N; p_fc = grid_dv->vv_fac_c_##N; \
                    p_ff = grid_dv->vv_fac_f_##N; p_fun = grid_dv->vv_fun_##N; break;
            TATT_TELE_CASE(0)
            TATT_TELE_CASE(1)
            TATT_TELE_CASE(2)
            TATT_TELE_CASE(3)
            TATT_TELE_CASE(4)
            TATT_TELE_CASE(5)
            TATT_TELE_CASE(6)
            TATT_TELE_CASE(7)
#undef TATT_TELE_CASE
            default: return;
        }

        void* kernelArgs[] = {
            (void*)&p_ii, (void*)&p_ip1, (void*)&p_im1,
            (void*)&p_jp1, (void*)&p_jm1, (void*)&p_kp1, (void*)&p_km1,
            (void*)&p_fa, (void*)&p_fb, (void*)&p_fc, (void*)&p_ff,
            (void*)&p_fun,
            &(grid_dv->tau),
            &(grid_dv->loc_I_host), &(grid_dv->loc_J_host), &(grid_dv->loc_K_host),
            &(grid_dv->dr_host), &(grid_dv->dt_host), &(grid_dv->dp_host),
            &(grid_dv->n_nodes_on_levels_host[i_level]), &i_node_offset
        };
        print_CUDA_error_if_any(cudaLaunchKernel((void*)cuda_do_sweep_level_kernel_upwind_tele, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 42000+iswp);

        i_node_offset += grid_dv->n_nodes_on_levels_host[i_level];
    }

    finalize_sweep_params(grid_dv);
}

void initialize_sweep_params(Grid_on_device* grid_dv){

    // check the numBlockPerSm and set the block size accordingly
    //int numBlocksPerSm = 0;
    //int block_size = CUDA_SWEEPING_BLOCK_SIZE;

    //int device;
    //cudaGetDevice(&device);

    //cudaDeviceProp deviceProp;
    //cudaGetDeviceProperties(&deviceProp, device);
    //if(grid_dv->if_3rd_order)
    //    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, cuda_do_sweep_level_kernel_3rd, CUDA_SWEEPING_BLOCK_SIZE, 0);
    //else
    //    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, cuda_do_sweep_level_kernel_1st, CUDA_SWEEPING_BLOCK_SIZE, 0);

    //int max_cooperative_blocks = deviceProp.multiProcessorCount*numBlocksPerSm;

    //grid_dv->threads_sweep_host = dim3(block_size, 1, 1);
    //grid_dv->grid_sweep_host = dim3(max_cooperative_blocks, 1, 1);

    // spawn streams
    //grid_dv->level_streams = (cudaStream_t*)malloc(CUDA_MAX_NUM_STREAMS*sizeof(cudaStream_t));
    //for (int i = 0; i < CUDA_MAX_NUM_STREAMS; i++) {
    grid_dv->level_streams = (cudaStream_t*)malloc(grid_dv->n_levels_host*sizeof(cudaStream_t));
    for (int i = 0; i < grid_dv->n_levels_host; i++) {
        //cudaStreamCreate(&(grid_dv->level_streams[i]));
        // add null
        //cudaStreamCreateWithFlags(&(grid_dv->level_streams[i]), cudaStreamNonBlocking);
        grid_dv->level_streams[i] = nullptr;

    }


}


void finalize_sweep_params(Grid_on_device* grid_on_dv){
    // destroy streams
    //for (int i = 0; i < CUDA_MAX_NUM_STREAMS; i++) {
    //for (int i = 0; i < grid_on_dv->n_levels_host; i++) {
    //    cudaStreamDestroy(grid_on_dv->level_streams[i]);
    //}

    free(grid_on_dv->level_streams);
}


void run_kernel(Grid_on_device* grid_dv, int const& iswp, int& i_node_offset, int const& i_level, \
                dim3& grid_each, dim3& threads_each, int& n_nodes_this_level){

        int id_stream = i_level;// % CUDA_MAX_NUM_STREAMS;

        if (grid_dv->if_3rd_order) {
           if (iswp == 0){
                void *kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___0), \
                    &(grid_dv->vv_ip1j__k___0), \
                    &(grid_dv->vv_im1j__k___0), \
                    &(grid_dv->vv_i__jp1k___0), \
                    &(grid_dv->vv_i__jm1k___0), \
                    &(grid_dv->vv_i__j__kp1_0), \
                    &(grid_dv->vv_i__j__km1_0), \
                    &(grid_dv->vv_ip2j__k___0), \
                    &(grid_dv->vv_im2j__k___0), \
                    &(grid_dv->vv_i__jp2k___0), \
                    &(grid_dv->vv_i__jm2k___0), \
                    &(grid_dv->vv_i__j__kp2_0), \
                    &(grid_dv->vv_i__j__km2_0), \
                    &(grid_dv->vv_fac_a_0    ), \
                    &(grid_dv->vv_fac_b_0    ), \
                    &(grid_dv->vv_fac_c_0    ), \
                    &(grid_dv->vv_fac_f_0    ), \
                    &(grid_dv->vv_T0v_0      ), \
                    &(grid_dv->vv_T0r_0      ), \
                    &(grid_dv->vv_T0t_0      ), \
                    &(grid_dv->vv_T0p_0      ), \
                    &(grid_dv->vv_fun_0      ), \
                    &(grid_dv->vv_change_0   ), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else if (iswp == 1){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___1), \
                    &(grid_dv->vv_i__jp1k___1), \
                    &(grid_dv->vv_i__jm1k___1), \
                    &(grid_dv->vv_i__j__kp1_1), \
                    &(grid_dv->vv_i__j__km1_1), \
                    &(grid_dv->vv_ip1j__k___1), \
                    &(grid_dv->vv_im1j__k___1), \
                    &(grid_dv->vv_ip2j__k___1), \
                    &(grid_dv->vv_im2j__k___1), \
                    &(grid_dv->vv_i__jp2k___1), \
                    &(grid_dv->vv_i__jm2k___1), \
                    &(grid_dv->vv_i__j__kp2_1), \
                    &(grid_dv->vv_i__j__km2_1), \
                    &(grid_dv->vv_fac_a_1    ), \
                    &(grid_dv->vv_fac_b_1    ), \
                    &(grid_dv->vv_fac_c_1    ), \
                    &(grid_dv->vv_fac_f_1    ), \
                    &(grid_dv->vv_T0v_1      ), \
                    &(grid_dv->vv_T0r_1      ), \
                    &(grid_dv->vv_T0t_1      ), \
                    &(grid_dv->vv_T0p_1      ), \
                    &(grid_dv->vv_fun_1      ), \
                    &(grid_dv->vv_change_1   ), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else if (iswp == 2){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___2), \
                    &(grid_dv->vv_i__j__kp1_2), \
                    &(grid_dv->vv_i__j__km1_2), \
                    &(grid_dv->vv_ip1j__k___2), \
                    &(grid_dv->vv_im1j__k___2), \
                    &(grid_dv->vv_i__jp1k___2), \
                    &(grid_dv->vv_i__jm1k___2), \
                    &(grid_dv->vv_ip2j__k___2), \
                    &(grid_dv->vv_im2j__k___2), \
                    &(grid_dv->vv_i__jp2k___2), \
                    &(grid_dv->vv_i__jm2k___2), \
                    &(grid_dv->vv_i__j__kp2_2), \
                    &(grid_dv->vv_i__j__km2_2), \
                    &(grid_dv->vv_fac_a_2), \
                    &(grid_dv->vv_fac_b_2), \
                    &(grid_dv->vv_fac_c_2), \
                    &(grid_dv->vv_fac_f_2), \
                    &(grid_dv->vv_T0v_2), \
                    &(grid_dv->vv_T0r_2), \
                    &(grid_dv->vv_T0t_2), \
                    &(grid_dv->vv_T0p_2), \
                    &(grid_dv->vv_fun_2), \
                    &(grid_dv->vv_change_2), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else if (iswp == 3){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___3), \
                    &(grid_dv->vv_ip1j__k___3), \
                    &(grid_dv->vv_im1j__k___3), \
                    &(grid_dv->vv_i__jp1k___3), \
                    &(grid_dv->vv_i__jm1k___3), \
                    &(grid_dv->vv_i__j__kp1_3), \
                    &(grid_dv->vv_i__j__km1_3), \
                    &(grid_dv->vv_ip2j__k___3), \
                    &(grid_dv->vv_im2j__k___3), \
                    &(grid_dv->vv_i__jp2k___3), \
                    &(grid_dv->vv_i__jm2k___3), \
                    &(grid_dv->vv_i__j__kp2_3), \
                    &(grid_dv->vv_i__j__km2_3), \
                    &(grid_dv->vv_fac_a_3), \
                    &(grid_dv->vv_fac_b_3), \
                    &(grid_dv->vv_fac_c_3), \
                    &(grid_dv->vv_fac_f_3), \
                    &(grid_dv->vv_T0v_3), \
                    &(grid_dv->vv_T0r_3), \
                    &(grid_dv->vv_T0t_3), \
                    &(grid_dv->vv_T0p_3), \
                    &(grid_dv->vv_fun_3), \
                    &(grid_dv->vv_change_3), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else if (iswp == 4){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___4), \
                    &(grid_dv->vv_ip1j__k___4), \
                    &(grid_dv->vv_im1j__k___4), \
                    &(grid_dv->vv_i__jp1k___4), \
                    &(grid_dv->vv_i__jm1k___4), \
                    &(grid_dv->vv_i__j__kp1_4), \
                    &(grid_dv->vv_i__j__km1_4), \
                    &(grid_dv->vv_ip2j__k___4), \
                    &(grid_dv->vv_im2j__k___4), \
                    &(grid_dv->vv_i__jp2k___4), \
                    &(grid_dv->vv_i__jm2k___4), \
                    &(grid_dv->vv_i__j__kp2_4), \
                    &(grid_dv->vv_i__j__km2_4), \
                    &(grid_dv->vv_fac_a_4), \
                    &(grid_dv->vv_fac_b_4), \
                    &(grid_dv->vv_fac_c_4), \
                    &(grid_dv->vv_fac_f_4), \
                    &(grid_dv->vv_T0v_4), \
                    &(grid_dv->vv_T0r_4), \
                    &(grid_dv->vv_T0t_4), \
                    &(grid_dv->vv_T0p_4), \
                    &(grid_dv->vv_fun_4), \
                    &(grid_dv->vv_change_4), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else if (iswp == 5) {
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___5), \
                    &(grid_dv->vv_ip1j__k___5), \
                    &(grid_dv->vv_im1j__k___5), \
                    &(grid_dv->vv_i__jp1k___5), \
                    &(grid_dv->vv_i__jm1k___5), \
                    &(grid_dv->vv_i__j__kp1_5), \
                    &(grid_dv->vv_i__j__km1_5), \
                    &(grid_dv->vv_ip2j__k___5), \
                    &(grid_dv->vv_im2j__k___5), \
                    &(grid_dv->vv_i__jp2k___5), \
                    &(grid_dv->vv_i__jm2k___5), \
                    &(grid_dv->vv_i__j__kp2_5), \
                    &(grid_dv->vv_i__j__km2_5), \
                    &(grid_dv->vv_fac_a_5), \
                    &(grid_dv->vv_fac_b_5), \
                    &(grid_dv->vv_fac_c_5), \
                    &(grid_dv->vv_fac_f_5), \
                    &(grid_dv->vv_T0v_5), \
                    &(grid_dv->vv_T0r_5), \
                    &(grid_dv->vv_T0t_5), \
                    &(grid_dv->vv_T0p_5), \
                    &(grid_dv->vv_fun_5), \
                    &(grid_dv->vv_change_5), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else if (iswp == 6) {
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___6), \
                    &(grid_dv->vv_ip1j__k___6), \
                    &(grid_dv->vv_im1j__k___6), \
                    &(grid_dv->vv_i__jp1k___6), \
                    &(grid_dv->vv_i__jm1k___6), \
                    &(grid_dv->vv_i__j__kp1_6), \
                    &(grid_dv->vv_i__j__km1_6), \
                    &(grid_dv->vv_ip2j__k___6), \
                    &(grid_dv->vv_im2j__k___6), \
                    &(grid_dv->vv_i__jp2k___6), \
                    &(grid_dv->vv_i__jm2k___6), \
                    &(grid_dv->vv_i__j__kp2_6), \
                    &(grid_dv->vv_i__j__km2_6), \
                    &(grid_dv->vv_fac_a_6), \
                    &(grid_dv->vv_fac_b_6), \
                    &(grid_dv->vv_fac_c_6), \
                    &(grid_dv->vv_fac_f_6), \
                    &(grid_dv->vv_T0v_6), \
                    &(grid_dv->vv_T0r_6), \
                    &(grid_dv->vv_T0t_6), \
                    &(grid_dv->vv_T0p_6), \
                    &(grid_dv->vv_fun_6), \
                    &(grid_dv->vv_change_6), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else {
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___7), \
                    &(grid_dv->vv_ip1j__k___7), \
                    &(grid_dv->vv_im1j__k___7), \
                    &(grid_dv->vv_i__jp1k___7), \
                    &(grid_dv->vv_i__jm1k___7), \
                    &(grid_dv->vv_i__j__kp1_7), \
                    &(grid_dv->vv_i__j__km1_7), \
                    &(grid_dv->vv_ip2j__k___7), \
                    &(grid_dv->vv_im2j__k___7), \
                    &(grid_dv->vv_i__jp2k___7), \
                    &(grid_dv->vv_i__jm2k___7), \
                    &(grid_dv->vv_i__j__kp2_7), \
                    &(grid_dv->vv_i__j__km2_7), \
                    &(grid_dv->vv_fac_a_7), \
                    &(grid_dv->vv_fac_b_7), \
                    &(grid_dv->vv_fac_c_7), \
                    &(grid_dv->vv_fac_f_7), \
                    &(grid_dv->vv_T0v_7), \
                    &(grid_dv->vv_T0r_7), \
                    &(grid_dv->vv_T0t_7), \
                    &(grid_dv->vv_T0p_7), \
                    &(grid_dv->vv_fun_7), \
                    &(grid_dv->vv_change_7), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_3rd, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            }
        } else { // 1st order
            if (iswp == 0){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___0), \
                    &(grid_dv->vv_ip1j__k___0), \
                    &(grid_dv->vv_im1j__k___0), \
                    &(grid_dv->vv_i__jp1k___0), \
                    &(grid_dv->vv_i__jm1k___0), \
                    &(grid_dv->vv_i__j__kp1_0), \
                    &(grid_dv->vv_i__j__km1_0), \
                    &(grid_dv->vv_fac_a_0), \
                    &(grid_dv->vv_fac_b_0), \
                    &(grid_dv->vv_fac_c_0), \
                    &(grid_dv->vv_fac_f_0), \
                    &(grid_dv->vv_T0v_0), \
                    &(grid_dv->vv_T0r_0), \
                    &(grid_dv->vv_T0t_0), \
                    &(grid_dv->vv_T0p_0), \
                    &(grid_dv->vv_fun_0), \
                    &(grid_dv->vv_change_0), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30000);

            } else if (iswp == 1){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___1), \
                    &(grid_dv->vv_i__jp1k___1), \
                    &(grid_dv->vv_i__jm1k___1), \
                    &(grid_dv->vv_i__j__kp1_1), \
                    &(grid_dv->vv_i__j__km1_1), \
                    &(grid_dv->vv_ip1j__k___1), \
                    &(grid_dv->vv_im1j__k___1), \
                    &(grid_dv->vv_fac_a_1), \
                    &(grid_dv->vv_fac_b_1), \
                    &(grid_dv->vv_fac_c_1), \
                    &(grid_dv->vv_fac_f_1), \
                    &(grid_dv->vv_T0v_1), \
                    &(grid_dv->vv_T0r_1), \
                    &(grid_dv->vv_T0t_1), \
                    &(grid_dv->vv_T0p_1), \
                    &(grid_dv->vv_fun_1), \
                    &(grid_dv->vv_change_1), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30001);

            } else if (iswp == 2){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___2), \
                    &(grid_dv->vv_i__j__kp1_2), \
                    &(grid_dv->vv_i__j__km1_2), \
                    &(grid_dv->vv_ip1j__k___2), \
                    &(grid_dv->vv_im1j__k___2), \
                    &(grid_dv->vv_i__jp1k___2), \
                    &(grid_dv->vv_i__jm1k___2), \
                    &(grid_dv->vv_fac_a_2), \
                    &(grid_dv->vv_fac_b_2), \
                    &(grid_dv->vv_fac_c_2), \
                    &(grid_dv->vv_fac_f_2), \
                    &(grid_dv->vv_T0v_2), \
                    &(grid_dv->vv_T0r_2), \
                    &(grid_dv->vv_T0t_2), \
                    &(grid_dv->vv_T0p_2), \
                    &(grid_dv->vv_fun_2), \
                    &(grid_dv->vv_change_2), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30002);

            } else if (iswp == 3){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___3), \
                    &(grid_dv->vv_ip1j__k___3), \
                    &(grid_dv->vv_im1j__k___3), \
                    &(grid_dv->vv_i__jp1k___3), \
                    &(grid_dv->vv_i__jm1k___3), \
                    &(grid_dv->vv_i__j__kp1_3), \
                    &(grid_dv->vv_i__j__km1_3), \
                    &(grid_dv->vv_fac_a_3), \
                    &(grid_dv->vv_fac_b_3), \
                    &(grid_dv->vv_fac_c_3), \
                    &(grid_dv->vv_fac_f_3), \
                    &(grid_dv->vv_T0v_3), \
                    &(grid_dv->vv_T0r_3), \
                    &(grid_dv->vv_T0t_3), \
                    &(grid_dv->vv_T0p_3), \
                    &(grid_dv->vv_fun_3), \
                    &(grid_dv->vv_change_3), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30003);

            } else if (iswp == 4){
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___4), \
                    &(grid_dv->vv_ip1j__k___4), \
                    &(grid_dv->vv_im1j__k___4), \
                    &(grid_dv->vv_i__jp1k___4), \
                    &(grid_dv->vv_i__jm1k___4), \
                    &(grid_dv->vv_i__j__kp1_4), \
                    &(grid_dv->vv_i__j__km1_4), \
                    &(grid_dv->vv_fac_a_4), \
                    &(grid_dv->vv_fac_b_4), \
                    &(grid_dv->vv_fac_c_4), \
                    &(grid_dv->vv_fac_f_4), \
                    &(grid_dv->vv_T0v_4), \
                    &(grid_dv->vv_T0r_4), \
                    &(grid_dv->vv_T0t_4), \
                    &(grid_dv->vv_T0p_4), \
                    &(grid_dv->vv_fun_4), \
                    &(grid_dv->vv_change_4), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30004);

            } else if (iswp == 5) {
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___5), \
                    &(grid_dv->vv_ip1j__k___5), \
                    &(grid_dv->vv_im1j__k___5), \
                    &(grid_dv->vv_i__jp1k___5), \
                    &(grid_dv->vv_i__jm1k___5), \
                    &(grid_dv->vv_i__j__kp1_5), \
                    &(grid_dv->vv_i__j__km1_5), \
                    &(grid_dv->vv_fac_a_5), \
                    &(grid_dv->vv_fac_b_5), \
                    &(grid_dv->vv_fac_c_5), \
                    &(grid_dv->vv_fac_f_5), \
                    &(grid_dv->vv_T0v_5), \
                    &(grid_dv->vv_T0r_5), \
                    &(grid_dv->vv_T0t_5), \
                    &(grid_dv->vv_T0p_5), \
                    &(grid_dv->vv_fun_5), \
                    &(grid_dv->vv_change_5), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30005);

            } else if (iswp == 6) {
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___6), \
                    &(grid_dv->vv_ip1j__k___6), \
                    &(grid_dv->vv_im1j__k___6), \
                    &(grid_dv->vv_i__jp1k___6), \
                    &(grid_dv->vv_i__jm1k___6), \
                    &(grid_dv->vv_i__j__kp1_6), \
                    &(grid_dv->vv_i__j__km1_6), \
                    &(grid_dv->vv_fac_a_6), \
                    &(grid_dv->vv_fac_b_6), \
                    &(grid_dv->vv_fac_c_6), \
                    &(grid_dv->vv_fac_f_6), \
                    &(grid_dv->vv_T0v_6), \
                    &(grid_dv->vv_T0r_6), \
                    &(grid_dv->vv_T0t_6), \
                    &(grid_dv->vv_T0p_6), \
                    &(grid_dv->vv_fun_6), \
                    &(grid_dv->vv_change_6), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };
                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30006);


            } else {
                void* kernelArgs[]{\
                    &(grid_dv->vv_i__j__k___7), \
                    &(grid_dv->vv_ip1j__k___7), \
                    &(grid_dv->vv_im1j__k___7), \
                    &(grid_dv->vv_i__jp1k___7), \
                    &(grid_dv->vv_i__jm1k___7), \
                    &(grid_dv->vv_i__j__kp1_7), \
                    &(grid_dv->vv_i__j__km1_7), \
                    &(grid_dv->vv_fac_a_7    ), \
                    &(grid_dv->vv_fac_b_7    ), \
                    &(grid_dv->vv_fac_c_7    ), \
                    &(grid_dv->vv_fac_f_7    ), \
                    &(grid_dv->vv_T0v_7      ), \
                    &(grid_dv->vv_T0r_7      ), \
                    &(grid_dv->vv_T0t_7      ), \
                    &(grid_dv->vv_T0p_7      ), \
                    &(grid_dv->vv_fun_7      ), \
                    &(grid_dv->vv_change_7   ), \
                    &(grid_dv->tau), \
                    &(grid_dv->loc_I_host), \
                    &(grid_dv->loc_J_host), \
                    &(grid_dv->loc_K_host), \
                    &(grid_dv->dr_host), \
                    &(grid_dv->dt_host), \
                    &(grid_dv->dp_host), \
                    &n_nodes_this_level, \
                    &i_node_offset \
                };

                print_CUDA_error_if_any(cudaLaunchKernel((void*) cuda_do_sweep_level_kernel_1st, grid_each, threads_each, kernelArgs, 0, grid_dv->level_streams[id_stream]), 30007);

            }
        }

        // synchronize all streams
        //print_CUDA_error_if_any(cudaStreamSynchronize(grid_dv->level_streams[id_stream]), 30008);
}


// this function calculate all levels of one single sweep direction
void cuda_run_iteration_forward(Grid_on_device* grid_dv, int const& iswp){

    initialize_sweep_params(grid_dv);

    int block_size = CUDA_SWEEPING_BLOCK_SIZE;
    int num_blocks_x, num_blocks_y;
    int i_node_offset=0;
    //get_block_xy(ceil(grid_dv->n_nodes_max_host/block_size+0.5), &num_blocks_x, &num_blocks_y);
    //dim3 grid_each(num_blocks_x, num_blocks_y);
    //dim3 threads_each(block_size, 1, 1);

    for (size_t i_level = 0; i_level < grid_dv->n_levels_host; i_level++){
        get_block_xy(ceil(grid_dv->n_nodes_on_levels_host[i_level]/block_size+0.5), &num_blocks_x, &num_blocks_y);
        dim3 grid_each(num_blocks_x, num_blocks_y);
        dim3 threads_each(block_size, 1, 1);

        run_kernel(grid_dv, iswp, i_node_offset, i_level, grid_each, threads_each, grid_dv->n_nodes_on_levels_host[i_level]);
        //run_kernel(grid_dv, iswp, i_node_offset, i_level, grid_dv->grid_sweep_host, grid_dv->threads_sweep_host, grid_dv->n_nodes_on_levels_host[i_level]);

        i_node_offset += grid_dv->n_nodes_on_levels_host[i_level];
    }

    finalize_sweep_params(grid_dv);

    // check memory leak
    //print_memory_usage();

}