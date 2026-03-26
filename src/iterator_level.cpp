#include "iterator_level.h"

#ifdef USE_SIMD
#include "vectorized_sweep.h"
#endif


Iterator_level::Iterator_level(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                : Iterator(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // do nothing
}


void Iterator_level::do_sweep_adj(int iswp, Grid& grid, InputParams& IP){

    // set sweep direction
    set_sweep_direction(iswp);

    int iip, jjt, kkr;
    int n_levels = ijk_for_this_subproc.size();

#if !defined USE_SIMD

    for (int i_level = 0; i_level < n_levels; i_level++) {
        size_t n_nodes = ijk_for_this_subproc[i_level].size();

        for (size_t i_node = 0; i_node < n_nodes; i_node++) {

            V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

            if (r_dirc < 0) kkr = nr-1-kkr;
            if (t_dirc < 0) jjt = nt-1-jjt;
            if (p_dirc < 0) iip = np-1-iip;

            if(iip < 0 || jjt < 0 || kkr < 0 || iip >= np || jjt >= nt || kkr >= nr) {
                std::cout << "ERROR: iip = " << iip << ", jjt = " << jjt << ", kkr = " << kkr << std::endl;
            }

            //
            // calculate stencils
            //
            if (iip != 0    && jjt != 0    && kkr != 0 \
                && iip != np-1 && jjt != nt-1 && kkr != nr-1) {
                // calculate stencils
                calculate_stencil_adj(grid, iip, jjt, kkr);
            } else {
                calculate_boundary_nodes_adj(grid, iip, jjt, kkr);
            }

        } // end ijk

        // mpi synchronization
        synchronize_all_sub();

    } // end loop i_level

#elif USE_AVX512 || USE_AVX

    // Grid spacing inverse constants
    __mT v_dr_inv   = _mmT_set1_pT(_1_CR / dr);
    __mT v_dt_inv   = _mmT_set1_pT(_1_CR / dt);
    __mT v_dp_inv   = _mmT_set1_pT(_1_CR / dp);
    __mT v_4dp_inv  = _mmT_set1_pT(_1_CR / (_4_CR * dp));
    __mT v_4dt_inv  = _mmT_set1_pT(_1_CR / (_4_CR * dt));
    __mT v_2dr_inv  = _mmT_set1_pT(_1_CR / (_2_CR * dr));
    __mT v_2dt_inv  = _mmT_set1_pT(_1_CR / (_2_CR * dt));
    __mT v_2dp_inv  = _mmT_set1_pT(_1_CR / (_2_CR * dp));
    __mT v_adj_half = _mmT_set1_pT(0.5);
    __mT v_adj_zero = _mmT_set1_pT(0.0);
    __mT v_adj_one  = _mmT_set1_pT(1.0);
    __mT v_adj_two  = _mmT_set1_pT(2.0);
    __mT v_adj_neg1 = _mmT_set1_pT(-1.0);
    __mT v_adj_neg2 = _mmT_set1_pT(-2.0);
    __mT v_eps_adj  = _mmT_set1_pT(1e-6);

    // SIMD abs helper: abs(x) = max(x, -x)
#if USE_AVX512
    #define SIMD_ABS_ADJ(x) _mm512_max_pd(x, _mmT_sub_pT(v_adj_zero, x))
#else
    #define SIMD_ABS_ADJ(x) _mm256_max_pd(x, _mmT_sub_pT(v_adj_zero, x))
#endif

    // Index strides for I2V(i,j,k) = k*loc_I*loc_J + j*loc_I + i
    int stride_p = 1;               // i (phi) direction
    int stride_t = loc_I;           // j (theta) direction
    int stride_r = loc_I * loc_J;   // k (r) direction

    bool adj_i_first = grid.i_first();
    bool adj_i_last  = grid.i_last();
    bool adj_j_first = grid.j_first();
    bool adj_j_last  = grid.j_last();
    bool adj_k_first = grid.k_first();
    bool adj_k_last  = grid.k_last();

    for (int i_level = 0; i_level < n_levels; i_level++) {
        int n_nodes = ijk_for_this_subproc[i_level].size();
        int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

        for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {
            int i_vec = _i_vec * NSIMD;
            int n_valid = std::min(NSIMD, n_nodes - i_vec);

            // ========= SCALAR GATHER: compute center indices and load all values =========
            int    d_center[NSIMD];
            bool   d_is_interior[NSIMD];
            bool   d_is_global_bnd[NSIMD];

            // T_loc at 11 unique positions
            CUSTOMREAL d_T_c[NSIMD],      d_T_km1[NSIMD],    d_T_kp1[NSIMD];
            CUSTOMREAL d_T_jm1[NSIMD],    d_T_jp1[NSIMD],    d_T_im1[NSIMD],    d_T_ip1[NSIMD];
            CUSTOMREAL d_T_ip1jm1[NSIMD], d_T_im1jm1[NSIMD], d_T_ip1jp1[NSIMD], d_T_im1jp1[NSIMD];
            // Anisotropy
            CUSTOMREAL d_zeta_c[NSIMD],   d_zeta_km1[NSIMD], d_zeta_kp1[NSIMD];
            CUSTOMREAL d_xi_c[NSIMD],     d_xi_jm1[NSIMD],   d_xi_jp1[NSIMD],   d_xi_im1[NSIMD], d_xi_ip1[NSIMD];
            CUSTOMREAL d_eta_c[NSIMD],    d_eta_jm1[NSIMD],  d_eta_jp1[NSIMD],  d_eta_im1[NSIMD], d_eta_ip1[NSIMD];
            // tau_loc neighbors + tau_old center
            CUSTOMREAL d_tau_km1[NSIMD],  d_tau_kp1[NSIMD];
            CUSTOMREAL d_tau_jm1[NSIMD],  d_tau_jp1[NSIMD];
            CUSTOMREAL d_tau_im1[NSIMD],  d_tau_ip1[NSIMD];
            CUSTOMREAL d_tau_old_c[NSIMD];
            // Geometric values
            CUSTOMREAL d_r_inv[NSIMD],    d_r2inv[NSIMD];
            CUSTOMREAL d_cos_t_inv[NSIMD],  d_cos2_t_inv[NSIMD];
            CUSTOMREAL d_sin_t[NSIMD];
            CUSTOMREAL d_cos_tmpt1_inv[NSIMD], d_cos_tmpt2_inv[NSIMD];

            for (int i = 0; i < NSIMD; i++) {
                if (i < n_valid) {
                    int ii_, jj_, kk_;
                    V2I(ijk_for_this_subproc[i_level][i_vec+i], ii_, jj_, kk_);
                    int _iip = (p_dirc < 0) ? np-1-ii_ : ii_;
                    int _jjt = (t_dirc < 0) ? nt-1-jj_ : jj_;
                    int _kkr = (r_dirc < 0) ? nr-1-kk_ : kk_;
                    int c = I2V(_iip, _jjt, _kkr);
                    d_center[i] = c;

                    bool is_bnd = (_iip == 0 || _jjt == 0 || _kkr == 0 ||
                                   _iip == np-1 || _jjt == nt-1 || _kkr == nr-1);
                    d_is_interior[i] = !is_bnd;
                    d_is_global_bnd[i] = is_bnd && (
                        (_iip == 0 && adj_i_first) || (_iip == np-1 && adj_i_last) ||
                        (_jjt == 0 && adj_j_first) || (_jjt == nt-1 && adj_j_last) ||
                        (_kkr == 0 && adj_k_first) || (_kkr == nr-1 && adj_k_last));

                    if (!is_bnd) {
                        // T_loc gathers using stride offsets
                        d_T_c[i]      = grid.T_loc[c];
                        d_T_km1[i]    = grid.T_loc[c - stride_r];
                        d_T_kp1[i]    = grid.T_loc[c + stride_r];
                        d_T_jm1[i]    = grid.T_loc[c - stride_t];
                        d_T_jp1[i]    = grid.T_loc[c + stride_t];
                        d_T_im1[i]    = grid.T_loc[c - stride_p];
                        d_T_ip1[i]    = grid.T_loc[c + stride_p];
                        d_T_ip1jm1[i] = grid.T_loc[c + stride_p - stride_t];
                        d_T_im1jm1[i] = grid.T_loc[c - stride_p - stride_t];
                        d_T_ip1jp1[i] = grid.T_loc[c + stride_p + stride_t];
                        d_T_im1jp1[i] = grid.T_loc[c - stride_p + stride_t];
                        // Zeta
                        d_zeta_c[i]   = grid.zeta_loc[c];
                        d_zeta_km1[i] = grid.zeta_loc[c - stride_r];
                        d_zeta_kp1[i] = grid.zeta_loc[c + stride_r];
                        // Xi
                        d_xi_c[i]     = grid.xi_loc[c];
                        d_xi_jm1[i]   = grid.xi_loc[c - stride_t];
                        d_xi_jp1[i]   = grid.xi_loc[c + stride_t];
                        d_xi_im1[i]   = grid.xi_loc[c - stride_p];
                        d_xi_ip1[i]   = grid.xi_loc[c + stride_p];
                        // Eta
                        d_eta_c[i]    = grid.eta_loc[c];
                        d_eta_jm1[i]  = grid.eta_loc[c - stride_t];
                        d_eta_jp1[i]  = grid.eta_loc[c + stride_t];
                        d_eta_im1[i]  = grid.eta_loc[c - stride_p];
                        d_eta_ip1[i]  = grid.eta_loc[c + stride_p];
                        // Tau neighbors (runtime gather - tau changes during sweep)
                        d_tau_km1[i]  = grid.tau_loc[c - stride_r];
                        d_tau_kp1[i]  = grid.tau_loc[c + stride_r];
                        d_tau_jm1[i]  = grid.tau_loc[c - stride_t];
                        d_tau_jp1[i]  = grid.tau_loc[c + stride_t];
                        d_tau_im1[i]  = grid.tau_loc[c - stride_p];
                        d_tau_ip1[i]  = grid.tau_loc[c + stride_p];
                        d_tau_old_c[i] = grid.tau_old_loc[c];
                        // Geometry from 1D coordinate arrays
                        CUSTOMREAL r_val = grid.r_loc_1d[_kkr];
                        CUSTOMREAL t_val = grid.t_loc_1d[_jjt];
                        d_r_inv[i]     = _1_CR / r_val;
                        d_r2inv[i]     = _1_CR / (r_val * r_val);
                        CUSTOMREAL cos_t = std::cos(t_val);
                        CUSTOMREAL sin_t = std::sin(t_val);
                        d_sin_t[i]       = sin_t;
                        d_cos_t_inv[i]   = _1_CR / cos_t;
                        d_cos2_t_inv[i]  = _1_CR / (cos_t * cos_t);
                        CUSTOMREAL t_m1  = grid.t_loc_1d[_jjt - 1];
                        CUSTOMREAL t_p1  = grid.t_loc_1d[_jjt + 1];
                        d_cos_tmpt1_inv[i] = _1_CR / std::cos((t_m1 + t_val) / _2_CR);
                        d_cos_tmpt2_inv[i] = _1_CR / std::cos((t_val + t_p1) / _2_CR);
                    } else {
                        // Zero all values for boundary/padding nodes
                        d_T_c[i] = d_T_km1[i] = d_T_kp1[i] = _0_CR;
                        d_T_jm1[i] = d_T_jp1[i] = d_T_im1[i] = d_T_ip1[i] = _0_CR;
                        d_T_ip1jm1[i] = d_T_im1jm1[i] = d_T_ip1jp1[i] = d_T_im1jp1[i] = _0_CR;
                        d_zeta_c[i] = d_zeta_km1[i] = d_zeta_kp1[i] = _0_CR;
                        d_xi_c[i] = d_xi_jm1[i] = d_xi_jp1[i] = d_xi_im1[i] = d_xi_ip1[i] = _0_CR;
                        d_eta_c[i] = d_eta_jm1[i] = d_eta_jp1[i] = d_eta_im1[i] = d_eta_ip1[i] = _0_CR;
                        d_tau_km1[i] = d_tau_kp1[i] = d_tau_jm1[i] = d_tau_jp1[i] = _0_CR;
                        d_tau_im1[i] = d_tau_ip1[i] = d_tau_old_c[i] = _0_CR;
                        d_r_inv[i] = _1_CR; d_r2inv[i] = _1_CR;
                        d_sin_t[i] = _0_CR; d_cos_t_inv[i] = _1_CR; d_cos2_t_inv[i] = _1_CR;
                        d_cos_tmpt1_inv[i] = _1_CR; d_cos_tmpt2_inv[i] = _1_CR;
                    }
                } else {
                    // Padding past n_valid
                    d_center[i] = 0; d_is_interior[i] = false; d_is_global_bnd[i] = false;
                    d_T_c[i] = d_T_km1[i] = d_T_kp1[i] = _0_CR;
                    d_T_jm1[i] = d_T_jp1[i] = d_T_im1[i] = d_T_ip1[i] = _0_CR;
                    d_T_ip1jm1[i] = d_T_im1jm1[i] = d_T_ip1jp1[i] = d_T_im1jp1[i] = _0_CR;
                    d_zeta_c[i] = d_zeta_km1[i] = d_zeta_kp1[i] = _0_CR;
                    d_xi_c[i] = d_xi_jm1[i] = d_xi_jp1[i] = d_xi_im1[i] = d_xi_ip1[i] = _0_CR;
                    d_eta_c[i] = d_eta_jm1[i] = d_eta_jp1[i] = d_eta_im1[i] = d_eta_ip1[i] = _0_CR;
                    d_tau_km1[i] = d_tau_kp1[i] = d_tau_jm1[i] = d_tau_jp1[i] = _0_CR;
                    d_tau_im1[i] = d_tau_ip1[i] = d_tau_old_c[i] = _0_CR;
                    d_r_inv[i] = _1_CR; d_r2inv[i] = _1_CR;
                    d_sin_t[i] = _0_CR; d_cos_t_inv[i] = _1_CR; d_cos2_t_inv[i] = _1_CR;
                    d_cos_tmpt1_inv[i] = _1_CR; d_cos_tmpt2_inv[i] = _1_CR;
                }
            } // end scalar gather

            // ========= SIMD LOAD =========
            __mT vT_c      = _mmT_loadu_pT(d_T_c);
            __mT vT_km1    = _mmT_loadu_pT(d_T_km1);
            __mT vT_kp1    = _mmT_loadu_pT(d_T_kp1);
            __mT vT_jm1    = _mmT_loadu_pT(d_T_jm1);
            __mT vT_jp1    = _mmT_loadu_pT(d_T_jp1);
            __mT vT_im1    = _mmT_loadu_pT(d_T_im1);
            __mT vT_ip1    = _mmT_loadu_pT(d_T_ip1);
            __mT vT_ip1jm1 = _mmT_loadu_pT(d_T_ip1jm1);
            __mT vT_im1jm1 = _mmT_loadu_pT(d_T_im1jm1);
            __mT vT_ip1jp1 = _mmT_loadu_pT(d_T_ip1jp1);
            __mT vT_im1jp1 = _mmT_loadu_pT(d_T_im1jp1);

            __mT vz_c   = _mmT_loadu_pT(d_zeta_c);
            __mT vz_km1 = _mmT_loadu_pT(d_zeta_km1);
            __mT vz_kp1 = _mmT_loadu_pT(d_zeta_kp1);

            __mT vxi_c   = _mmT_loadu_pT(d_xi_c);
            __mT vxi_jm1 = _mmT_loadu_pT(d_xi_jm1);
            __mT vxi_jp1 = _mmT_loadu_pT(d_xi_jp1);
            __mT vxi_im1 = _mmT_loadu_pT(d_xi_im1);
            __mT vxi_ip1 = _mmT_loadu_pT(d_xi_ip1);

            __mT ve_c   = _mmT_loadu_pT(d_eta_c);
            __mT ve_jm1 = _mmT_loadu_pT(d_eta_jm1);
            __mT ve_jp1 = _mmT_loadu_pT(d_eta_jp1);
            __mT ve_im1 = _mmT_loadu_pT(d_eta_im1);
            __mT ve_ip1 = _mmT_loadu_pT(d_eta_ip1);

            __mT vtau_km1 = _mmT_loadu_pT(d_tau_km1);
            __mT vtau_kp1 = _mmT_loadu_pT(d_tau_kp1);
            __mT vtau_jm1 = _mmT_loadu_pT(d_tau_jm1);
            __mT vtau_jp1 = _mmT_loadu_pT(d_tau_jp1);
            __mT vtau_im1 = _mmT_loadu_pT(d_tau_im1);
            __mT vtau_ip1 = _mmT_loadu_pT(d_tau_ip1);
            __mT vtau_old = _mmT_loadu_pT(d_tau_old_c);

            __mT vr_inv       = _mmT_loadu_pT(d_r_inv);
            __mT vr2inv       = _mmT_loadu_pT(d_r2inv);
            __mT vsin_t       = _mmT_loadu_pT(d_sin_t);
            __mT vcos_t_inv   = _mmT_loadu_pT(d_cos_t_inv);
            __mT vcos2_t_inv  = _mmT_loadu_pT(d_cos2_t_inv);
            __mT vctp1_inv    = _mmT_loadu_pT(d_cos_tmpt1_inv);
            __mT vctp2_inv    = _mmT_loadu_pT(d_cos_tmpt2_inv);

            // ========= SIMD COMPUTE: adjoint stencil =========

            // --- a1: radial backward flux ---
            // a1 = -(1+zeta[c]+zeta[k-1]) * (T[c]-T[k-1]) / dr
            __mT v_a1 = _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                _mmT_add_pT(v_adj_one, _mmT_add_pT(vz_c, vz_km1)),
                _mmT_mul_pT(_mmT_sub_pT(vT_c, vT_km1), v_dr_inv)));
            __mT v_abs_a1 = SIMD_ABS_ADJ(v_a1);
            __mT v_a1m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_a1, v_abs_a1));
            __mT v_a1p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_a1, v_abs_a1));

            // --- a2: radial forward flux ---
            // a2 = -(1+zeta[c]+zeta[k+1]) * (T[k+1]-T[c]) / dr
            __mT v_a2 = _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                _mmT_add_pT(v_adj_one, _mmT_add_pT(vz_c, vz_kp1)),
                _mmT_mul_pT(_mmT_sub_pT(vT_kp1, vT_c), v_dr_inv)));
            __mT v_abs_a2 = SIMD_ABS_ADJ(v_a2);
            __mT v_a2m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_a2, v_abs_a2));
            __mT v_a2p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_a2, v_abs_a2));

            // --- b1: theta backward flux ---
            // b1 = -(1-xi[jm1]-xi[c])/r²*(T[c]-T[jm1])/dt
            //      -(eta[jm1]+eta[c])/(r²*cos(tmpt1))/(4*dp)*cross_b1
            __mT v_cross_b1 = _mmT_add_pT(
                _mmT_sub_pT(vT_ip1jm1, vT_im1jm1),
                _mmT_sub_pT(vT_ip1, vT_im1));
            __mT v_b1 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_sub_pT(_mmT_sub_pT(v_adj_one, vxi_jm1), vxi_c),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(_mmT_sub_pT(vT_c, vT_jm1), v_dt_inv)))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_jm1, ve_c),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vctp1_inv,
                        _mmT_mul_pT(v_4dp_inv, v_cross_b1))))));
            __mT v_abs_b1 = SIMD_ABS_ADJ(v_b1);
            __mT v_b1m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_b1, v_abs_b1));
            __mT v_b1p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_b1, v_abs_b1));

            // --- b2: theta forward flux ---
            // b2 = -(1-xi[c]-xi[jp1])/r²*(T[jp1]-T[c])/dt
            //      -(eta[c]+eta[jp1])/(r²*cos(tmpt2))/(4*dp)*cross_b2
            __mT v_cross_b2 = _mmT_add_pT(
                _mmT_sub_pT(vT_ip1, vT_im1),
                _mmT_sub_pT(vT_ip1jp1, vT_im1jp1));
            __mT v_b2 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_sub_pT(_mmT_sub_pT(v_adj_one, vxi_c), vxi_jp1),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(_mmT_sub_pT(vT_jp1, vT_c), v_dt_inv)))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_c, ve_jp1),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vctp2_inv,
                        _mmT_mul_pT(v_4dp_inv, v_cross_b2))))));
            __mT v_abs_b2 = SIMD_ABS_ADJ(v_b2);
            __mT v_b2m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_b2, v_abs_b2));
            __mT v_b2p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_b2, v_abs_b2));

            // --- c1: phi backward flux ---
            // c1 = -(1+xi[im1]+xi[c])/(r²*cos²t)*(T[c]-T[im1])/dp
            //      -(eta[im1]+eta[c])/(r²*cos(t))/(4*dt)*cross_c1
            __mT v_cross_c1 = _mmT_add_pT(
                _mmT_sub_pT(vT_im1jp1, vT_im1jm1),
                _mmT_sub_pT(vT_jp1, vT_jm1));
            __mT v_c1 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(v_adj_one, _mmT_add_pT(vxi_im1, vxi_c)),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos2_t_inv,
                        _mmT_mul_pT(_mmT_sub_pT(vT_c, vT_im1), v_dp_inv))))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_im1, ve_c),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos_t_inv,
                        _mmT_mul_pT(v_4dt_inv, v_cross_c1))))));
            __mT v_abs_c1 = SIMD_ABS_ADJ(v_c1);
            __mT v_c1m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_c1, v_abs_c1));
            __mT v_c1p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_c1, v_abs_c1));

            // --- c2: phi forward flux ---
            // c2 = -(1+xi[c]+xi[ip1])/(r²*cos²t)*(T[ip1]-T[c])/dp
            //      -(eta[c]+eta[ip1])/(r²*cos(t))/(4*dt)*cross_c2
            __mT v_cross_c2 = _mmT_add_pT(
                _mmT_sub_pT(vT_jp1, vT_jm1),
                _mmT_sub_pT(vT_ip1jp1, vT_ip1jm1));
            __mT v_c2 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(v_adj_one, _mmT_add_pT(vxi_c, vxi_ip1)),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos2_t_inv,
                        _mmT_mul_pT(_mmT_sub_pT(vT_ip1, vT_c), v_dp_inv))))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_c, ve_ip1),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos_t_inv,
                        _mmT_mul_pT(v_4dt_inv, v_cross_c2))))));
            __mT v_abs_c2 = SIMD_ABS_ADJ(v_c2);
            __mT v_c2m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_c2, v_abs_c2));
            __mT v_c2p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_c2, v_abs_c2));

            // --- d: additional spherical divergence term ---
            // d = -2*(1+2*zeta[c])/r * (T[k+1]-T[k-1])/(2*dr)
            //     + (1-2*xi[c])*sin(t)/(r²*cos(t)) * (T[j+1]-T[j-1])/(2*dt)
            //     + 2*eta[c]*sin(t)/(r²*cos²(t)) * (T[i+1]-T[i-1])/(2*dp)
            __mT v_d_t1 = _mmT_mul_pT(v_adj_neg2, _mmT_mul_pT(
                _mmT_add_pT(v_adj_one, _mmT_mul_pT(v_adj_two, vz_c)),
                _mmT_mul_pT(vr_inv, _mmT_mul_pT(_mmT_sub_pT(vT_kp1, vT_km1), v_2dr_inv))));
            __mT v_d_t2 = _mmT_mul_pT(
                _mmT_sub_pT(v_adj_one, _mmT_mul_pT(v_adj_two, vxi_c)),
                _mmT_mul_pT(vsin_t, _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos_t_inv,
                    _mmT_mul_pT(_mmT_sub_pT(vT_jp1, vT_jm1), v_2dt_inv)))));
            __mT v_d_t3 = _mmT_mul_pT(v_adj_two, _mmT_mul_pT(ve_c,
                _mmT_mul_pT(vsin_t, _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos2_t_inv,
                    _mmT_mul_pT(_mmT_sub_pT(vT_ip1, vT_im1), v_2dp_inv))))));
            __mT v_d = _mmT_add_pT(v_d_t1, _mmT_add_pT(v_d_t2, v_d_t3));

            // --- coe = (a2p-a1m)/dr + (b2p-b1m)/dt + (c2p-c1m)/dp ---
            __mT v_coe = _mmT_add_pT(
                _mmT_mul_pT(_mmT_sub_pT(v_a2p, v_a1m), v_dr_inv),
                _mmT_add_pT(
                    _mmT_mul_pT(_mmT_sub_pT(v_b2p, v_b1m), v_dt_inv),
                    _mmT_mul_pT(_mmT_sub_pT(v_c2p, v_c1m), v_dp_inv)));

            // --- Hadj: adjoint Hamiltonian ---
            // Hadj = (a1p*tau[k-1]-a2m*tau[k+1])/dr + (b1p*tau[j-1]-b2m*tau[j+1])/dt
            //       + (c1p*tau[i-1]-c2m*tau[i+1])/dp
            __mT v_Hadj = _mmT_add_pT(
                _mmT_mul_pT(_mmT_sub_pT(
                    _mmT_mul_pT(v_a1p, vtau_km1), _mmT_mul_pT(v_a2m, vtau_kp1)), v_dr_inv),
                _mmT_add_pT(
                    _mmT_mul_pT(_mmT_sub_pT(
                        _mmT_mul_pT(v_b1p, vtau_jm1), _mmT_mul_pT(v_b2m, vtau_jp1)), v_dt_inv),
                    _mmT_mul_pT(_mmT_sub_pT(
                        _mmT_mul_pT(v_c1p, vtau_im1), _mmT_mul_pT(v_c2m, vtau_ip1)), v_dp_inv)));

            // --- result = (tau_old + Hadj) / (coe + d) ---
            __mT v_result = _mmT_div_pT(
                _mmT_add_pT(vtau_old, v_Hadj),
                _mmT_add_pT(v_coe, v_d));

            // --- isZeroAdj: if |coe| < 1e-6, set result to 0 ---
            __mT v_abs_coe = SIMD_ABS_ADJ(v_coe);
#if USE_AVX512
            __mmaskT mask_zero_adj = _mm512_cmp_pT_mask(v_abs_coe, v_eps_adj, _CMP_LT_OQ);
            v_result = _mm512_mask_blend_pT(mask_zero_adj, v_result, v_adj_zero);
#else // USE_AVX
            __mT mask_zero_adj = _mm256_cmp_pT(v_abs_coe, v_eps_adj, _CMP_LT_OQ);
            v_result = _mm256_blendv_pd(v_result, v_adj_zero, mask_zero_adj);
#endif

            // ========= SCALAR SCATTER =========
            _mmT_store_pT(dump_c__, v_result);
            for (int i = 0; i < n_valid; i++) {
                if (d_is_interior[i]) {
                    grid.tau_loc[d_center[i]] = dump_c__[i];
                } else if (d_is_global_bnd[i]) {
                    grid.tau_loc[d_center[i]] = _0_CR;
                }
            }

        } // end i_vec loop

        // mpi synchronization
        synchronize_all_sub();

    } // end i_level loop

    #undef SIMD_ABS_ADJ

#endif // USE_SIMD

}


Iterator_level_tele::Iterator_level_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                : Iterator(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // do nothing
}


void Iterator_level_tele::do_sweep_adj(int iswp, Grid& grid, InputParams& IP){
    // set sweep direction
    set_sweep_direction(iswp);

    int iip, jjt, kkr;
    int n_levels = ijk_for_this_subproc.size();

#if !defined USE_SIMD

    for (int i_level = 0; i_level < n_levels; i_level++) {
        size_t n_nodes = ijk_for_this_subproc[i_level].size();

        for (size_t i_node = 0; i_node < n_nodes; i_node++) {

            V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

            if (r_dirc < 0) kkr = nr-1-kkr;
            if (t_dirc < 0) jjt = nt-1-jjt;
            if (p_dirc < 0) iip = np-1-iip;

            //
            // calculate stencils
            //
            if (iip != 0    && jjt != 0    && kkr != 0 \
             && iip != np-1 && jjt != nt-1 && kkr != nr-1) {
                // calculate stencils
                calculate_stencil_adj(grid, iip, jjt, kkr);
            } else {
                calculate_boundary_nodes_adj(grid, iip, jjt, kkr);
            }
        } // end ijk

        // mpi synchronization
        synchronize_all_sub();

    } // end loop i_level

#elif USE_AVX512 || USE_AVX

    // Grid spacing inverse constants
    __mT v_dr_inv   = _mmT_set1_pT(_1_CR / dr);
    __mT v_dt_inv   = _mmT_set1_pT(_1_CR / dt);
    __mT v_dp_inv   = _mmT_set1_pT(_1_CR / dp);
    __mT v_4dp_inv  = _mmT_set1_pT(_1_CR / (_4_CR * dp));
    __mT v_4dt_inv  = _mmT_set1_pT(_1_CR / (_4_CR * dt));
    __mT v_2dr_inv  = _mmT_set1_pT(_1_CR / (_2_CR * dr));
    __mT v_2dt_inv  = _mmT_set1_pT(_1_CR / (_2_CR * dt));
    __mT v_2dp_inv  = _mmT_set1_pT(_1_CR / (_2_CR * dp));
    __mT v_adj_half = _mmT_set1_pT(0.5);
    __mT v_adj_zero = _mmT_set1_pT(0.0);
    __mT v_adj_one  = _mmT_set1_pT(1.0);
    __mT v_adj_two  = _mmT_set1_pT(2.0);
    __mT v_adj_neg1 = _mmT_set1_pT(-1.0);
    __mT v_adj_neg2 = _mmT_set1_pT(-2.0);
    __mT v_eps_adj  = _mmT_set1_pT(1e-6);

#if USE_AVX512
    #define SIMD_ABS_ADJ(x) _mm512_max_pd(x, _mmT_sub_pT(v_adj_zero, x))
#else
    #define SIMD_ABS_ADJ(x) _mm256_max_pd(x, _mmT_sub_pT(v_adj_zero, x))
#endif

    int stride_p = 1;
    int stride_t = loc_I;
    int stride_r = loc_I * loc_J;

    bool adj_i_first = grid.i_first();
    bool adj_i_last  = grid.i_last();
    bool adj_j_first = grid.j_first();
    bool adj_j_last  = grid.j_last();
    bool adj_k_first = grid.k_first();
    bool adj_k_last  = grid.k_last();

    for (int i_level = 0; i_level < n_levels; i_level++) {
        int n_nodes = ijk_for_this_subproc[i_level].size();
        int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

        for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {
            int i_vec = _i_vec * NSIMD;
            int n_valid = std::min(NSIMD, n_nodes - i_vec);

            int    d_center[NSIMD];
            bool   d_is_interior[NSIMD];
            bool   d_is_global_bnd[NSIMD];

            CUSTOMREAL d_T_c[NSIMD],      d_T_km1[NSIMD],    d_T_kp1[NSIMD];
            CUSTOMREAL d_T_jm1[NSIMD],    d_T_jp1[NSIMD],    d_T_im1[NSIMD],    d_T_ip1[NSIMD];
            CUSTOMREAL d_T_ip1jm1[NSIMD], d_T_im1jm1[NSIMD], d_T_ip1jp1[NSIMD], d_T_im1jp1[NSIMD];
            CUSTOMREAL d_zeta_c[NSIMD],   d_zeta_km1[NSIMD], d_zeta_kp1[NSIMD];
            CUSTOMREAL d_xi_c[NSIMD],     d_xi_jm1[NSIMD],   d_xi_jp1[NSIMD],   d_xi_im1[NSIMD], d_xi_ip1[NSIMD];
            CUSTOMREAL d_eta_c[NSIMD],    d_eta_jm1[NSIMD],  d_eta_jp1[NSIMD],  d_eta_im1[NSIMD], d_eta_ip1[NSIMD];
            CUSTOMREAL d_tau_km1[NSIMD],  d_tau_kp1[NSIMD];
            CUSTOMREAL d_tau_jm1[NSIMD],  d_tau_jp1[NSIMD];
            CUSTOMREAL d_tau_im1[NSIMD],  d_tau_ip1[NSIMD];
            CUSTOMREAL d_tau_old_c[NSIMD];
            CUSTOMREAL d_r_inv[NSIMD],    d_r2inv[NSIMD];
            CUSTOMREAL d_cos_t_inv[NSIMD],  d_cos2_t_inv[NSIMD];
            CUSTOMREAL d_sin_t[NSIMD];
            CUSTOMREAL d_cos_tmpt1_inv[NSIMD], d_cos_tmpt2_inv[NSIMD];

            for (int i = 0; i < NSIMD; i++) {
                if (i < n_valid) {
                    int ii_, jj_, kk_;
                    V2I(ijk_for_this_subproc[i_level][i_vec+i], ii_, jj_, kk_);
                    int _iip = (p_dirc < 0) ? np-1-ii_ : ii_;
                    int _jjt = (t_dirc < 0) ? nt-1-jj_ : jj_;
                    int _kkr = (r_dirc < 0) ? nr-1-kk_ : kk_;
                    int c = I2V(_iip, _jjt, _kkr);
                    d_center[i] = c;

                    bool is_bnd = (_iip == 0 || _jjt == 0 || _kkr == 0 ||
                                   _iip == np-1 || _jjt == nt-1 || _kkr == nr-1);
                    d_is_interior[i] = !is_bnd;
                    d_is_global_bnd[i] = is_bnd && (
                        (_iip == 0 && adj_i_first) || (_iip == np-1 && adj_i_last) ||
                        (_jjt == 0 && adj_j_first) || (_jjt == nt-1 && adj_j_last) ||
                        (_kkr == 0 && adj_k_first) || (_kkr == nr-1 && adj_k_last));

                    if (!is_bnd) {
                        d_T_c[i]      = grid.T_loc[c];
                        d_T_km1[i]    = grid.T_loc[c - stride_r];
                        d_T_kp1[i]    = grid.T_loc[c + stride_r];
                        d_T_jm1[i]    = grid.T_loc[c - stride_t];
                        d_T_jp1[i]    = grid.T_loc[c + stride_t];
                        d_T_im1[i]    = grid.T_loc[c - stride_p];
                        d_T_ip1[i]    = grid.T_loc[c + stride_p];
                        d_T_ip1jm1[i] = grid.T_loc[c + stride_p - stride_t];
                        d_T_im1jm1[i] = grid.T_loc[c - stride_p - stride_t];
                        d_T_ip1jp1[i] = grid.T_loc[c + stride_p + stride_t];
                        d_T_im1jp1[i] = grid.T_loc[c - stride_p + stride_t];
                        d_zeta_c[i]   = grid.zeta_loc[c];
                        d_zeta_km1[i] = grid.zeta_loc[c - stride_r];
                        d_zeta_kp1[i] = grid.zeta_loc[c + stride_r];
                        d_xi_c[i]     = grid.xi_loc[c];
                        d_xi_jm1[i]   = grid.xi_loc[c - stride_t];
                        d_xi_jp1[i]   = grid.xi_loc[c + stride_t];
                        d_xi_im1[i]   = grid.xi_loc[c - stride_p];
                        d_xi_ip1[i]   = grid.xi_loc[c + stride_p];
                        d_eta_c[i]    = grid.eta_loc[c];
                        d_eta_jm1[i]  = grid.eta_loc[c - stride_t];
                        d_eta_jp1[i]  = grid.eta_loc[c + stride_t];
                        d_eta_im1[i]  = grid.eta_loc[c - stride_p];
                        d_eta_ip1[i]  = grid.eta_loc[c + stride_p];
                        d_tau_km1[i]  = grid.tau_loc[c - stride_r];
                        d_tau_kp1[i]  = grid.tau_loc[c + stride_r];
                        d_tau_jm1[i]  = grid.tau_loc[c - stride_t];
                        d_tau_jp1[i]  = grid.tau_loc[c + stride_t];
                        d_tau_im1[i]  = grid.tau_loc[c - stride_p];
                        d_tau_ip1[i]  = grid.tau_loc[c + stride_p];
                        d_tau_old_c[i] = grid.tau_old_loc[c];
                        CUSTOMREAL r_val = grid.r_loc_1d[_kkr];
                        CUSTOMREAL t_val = grid.t_loc_1d[_jjt];
                        d_r_inv[i]     = _1_CR / r_val;
                        d_r2inv[i]     = _1_CR / (r_val * r_val);
                        CUSTOMREAL cos_t = std::cos(t_val);
                        CUSTOMREAL sin_t = std::sin(t_val);
                        d_sin_t[i]       = sin_t;
                        d_cos_t_inv[i]   = _1_CR / cos_t;
                        d_cos2_t_inv[i]  = _1_CR / (cos_t * cos_t);
                        CUSTOMREAL t_m1  = grid.t_loc_1d[_jjt - 1];
                        CUSTOMREAL t_p1  = grid.t_loc_1d[_jjt + 1];
                        d_cos_tmpt1_inv[i] = _1_CR / std::cos((t_m1 + t_val) / _2_CR);
                        d_cos_tmpt2_inv[i] = _1_CR / std::cos((t_val + t_p1) / _2_CR);
                    } else {
                        d_T_c[i] = d_T_km1[i] = d_T_kp1[i] = _0_CR;
                        d_T_jm1[i] = d_T_jp1[i] = d_T_im1[i] = d_T_ip1[i] = _0_CR;
                        d_T_ip1jm1[i] = d_T_im1jm1[i] = d_T_ip1jp1[i] = d_T_im1jp1[i] = _0_CR;
                        d_zeta_c[i] = d_zeta_km1[i] = d_zeta_kp1[i] = _0_CR;
                        d_xi_c[i] = d_xi_jm1[i] = d_xi_jp1[i] = d_xi_im1[i] = d_xi_ip1[i] = _0_CR;
                        d_eta_c[i] = d_eta_jm1[i] = d_eta_jp1[i] = d_eta_im1[i] = d_eta_ip1[i] = _0_CR;
                        d_tau_km1[i] = d_tau_kp1[i] = d_tau_jm1[i] = d_tau_jp1[i] = _0_CR;
                        d_tau_im1[i] = d_tau_ip1[i] = d_tau_old_c[i] = _0_CR;
                        d_r_inv[i] = _1_CR; d_r2inv[i] = _1_CR;
                        d_sin_t[i] = _0_CR; d_cos_t_inv[i] = _1_CR; d_cos2_t_inv[i] = _1_CR;
                        d_cos_tmpt1_inv[i] = _1_CR; d_cos_tmpt2_inv[i] = _1_CR;
                    }
                } else {
                    d_center[i] = 0; d_is_interior[i] = false; d_is_global_bnd[i] = false;
                    d_T_c[i] = d_T_km1[i] = d_T_kp1[i] = _0_CR;
                    d_T_jm1[i] = d_T_jp1[i] = d_T_im1[i] = d_T_ip1[i] = _0_CR;
                    d_T_ip1jm1[i] = d_T_im1jm1[i] = d_T_ip1jp1[i] = d_T_im1jp1[i] = _0_CR;
                    d_zeta_c[i] = d_zeta_km1[i] = d_zeta_kp1[i] = _0_CR;
                    d_xi_c[i] = d_xi_jm1[i] = d_xi_jp1[i] = d_xi_im1[i] = d_xi_ip1[i] = _0_CR;
                    d_eta_c[i] = d_eta_jm1[i] = d_eta_jp1[i] = d_eta_im1[i] = d_eta_ip1[i] = _0_CR;
                    d_tau_km1[i] = d_tau_kp1[i] = d_tau_jm1[i] = d_tau_jp1[i] = _0_CR;
                    d_tau_im1[i] = d_tau_ip1[i] = d_tau_old_c[i] = _0_CR;
                    d_r_inv[i] = _1_CR; d_r2inv[i] = _1_CR;
                    d_sin_t[i] = _0_CR; d_cos_t_inv[i] = _1_CR; d_cos2_t_inv[i] = _1_CR;
                    d_cos_tmpt1_inv[i] = _1_CR; d_cos_tmpt2_inv[i] = _1_CR;
                }
            }

            __mT vT_c      = _mmT_loadu_pT(d_T_c);
            __mT vT_km1    = _mmT_loadu_pT(d_T_km1);
            __mT vT_kp1    = _mmT_loadu_pT(d_T_kp1);
            __mT vT_jm1    = _mmT_loadu_pT(d_T_jm1);
            __mT vT_jp1    = _mmT_loadu_pT(d_T_jp1);
            __mT vT_im1    = _mmT_loadu_pT(d_T_im1);
            __mT vT_ip1    = _mmT_loadu_pT(d_T_ip1);
            __mT vT_ip1jm1 = _mmT_loadu_pT(d_T_ip1jm1);
            __mT vT_im1jm1 = _mmT_loadu_pT(d_T_im1jm1);
            __mT vT_ip1jp1 = _mmT_loadu_pT(d_T_ip1jp1);
            __mT vT_im1jp1 = _mmT_loadu_pT(d_T_im1jp1);

            __mT vz_c   = _mmT_loadu_pT(d_zeta_c);
            __mT vz_km1 = _mmT_loadu_pT(d_zeta_km1);
            __mT vz_kp1 = _mmT_loadu_pT(d_zeta_kp1);
            __mT vxi_c   = _mmT_loadu_pT(d_xi_c);
            __mT vxi_jm1 = _mmT_loadu_pT(d_xi_jm1);
            __mT vxi_jp1 = _mmT_loadu_pT(d_xi_jp1);
            __mT vxi_im1 = _mmT_loadu_pT(d_xi_im1);
            __mT vxi_ip1 = _mmT_loadu_pT(d_xi_ip1);
            __mT ve_c   = _mmT_loadu_pT(d_eta_c);
            __mT ve_jm1 = _mmT_loadu_pT(d_eta_jm1);
            __mT ve_jp1 = _mmT_loadu_pT(d_eta_jp1);
            __mT ve_im1 = _mmT_loadu_pT(d_eta_im1);
            __mT ve_ip1 = _mmT_loadu_pT(d_eta_ip1);

            __mT vtau_km1 = _mmT_loadu_pT(d_tau_km1);
            __mT vtau_kp1 = _mmT_loadu_pT(d_tau_kp1);
            __mT vtau_jm1 = _mmT_loadu_pT(d_tau_jm1);
            __mT vtau_jp1 = _mmT_loadu_pT(d_tau_jp1);
            __mT vtau_im1 = _mmT_loadu_pT(d_tau_im1);
            __mT vtau_ip1 = _mmT_loadu_pT(d_tau_ip1);
            __mT vtau_old = _mmT_loadu_pT(d_tau_old_c);

            __mT vr_inv       = _mmT_loadu_pT(d_r_inv);
            __mT vr2inv       = _mmT_loadu_pT(d_r2inv);
            __mT vsin_t       = _mmT_loadu_pT(d_sin_t);
            __mT vcos_t_inv   = _mmT_loadu_pT(d_cos_t_inv);
            __mT vcos2_t_inv  = _mmT_loadu_pT(d_cos2_t_inv);
            __mT vctp1_inv    = _mmT_loadu_pT(d_cos_tmpt1_inv);
            __mT vctp2_inv    = _mmT_loadu_pT(d_cos_tmpt2_inv);

            // a1 = -(1+zeta[c]+zeta[k-1]) * (T[c]-T[k-1]) / dr
            __mT v_a1 = _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                _mmT_add_pT(v_adj_one, _mmT_add_pT(vz_c, vz_km1)),
                _mmT_mul_pT(_mmT_sub_pT(vT_c, vT_km1), v_dr_inv)));
            __mT v_abs_a1 = SIMD_ABS_ADJ(v_a1);
            __mT v_a1m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_a1, v_abs_a1));
            __mT v_a1p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_a1, v_abs_a1));

            // a2 = -(1+zeta[c]+zeta[k+1]) * (T[k+1]-T[c]) / dr
            __mT v_a2 = _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                _mmT_add_pT(v_adj_one, _mmT_add_pT(vz_c, vz_kp1)),
                _mmT_mul_pT(_mmT_sub_pT(vT_kp1, vT_c), v_dr_inv)));
            __mT v_abs_a2 = SIMD_ABS_ADJ(v_a2);
            __mT v_a2m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_a2, v_abs_a2));
            __mT v_a2p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_a2, v_abs_a2));

            __mT v_cross_b1 = _mmT_add_pT(
                _mmT_sub_pT(vT_ip1jm1, vT_im1jm1),
                _mmT_sub_pT(vT_ip1, vT_im1));
            __mT v_b1 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_sub_pT(_mmT_sub_pT(v_adj_one, vxi_jm1), vxi_c),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(_mmT_sub_pT(vT_c, vT_jm1), v_dt_inv)))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_jm1, ve_c),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vctp1_inv,
                        _mmT_mul_pT(v_4dp_inv, v_cross_b1))))));
            __mT v_abs_b1 = SIMD_ABS_ADJ(v_b1);
            __mT v_b1m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_b1, v_abs_b1));
            __mT v_b1p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_b1, v_abs_b1));

            __mT v_cross_b2 = _mmT_add_pT(
                _mmT_sub_pT(vT_ip1, vT_im1),
                _mmT_sub_pT(vT_ip1jp1, vT_im1jp1));
            __mT v_b2 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_sub_pT(_mmT_sub_pT(v_adj_one, vxi_c), vxi_jp1),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(_mmT_sub_pT(vT_jp1, vT_c), v_dt_inv)))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_c, ve_jp1),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vctp2_inv,
                        _mmT_mul_pT(v_4dp_inv, v_cross_b2))))));
            __mT v_abs_b2 = SIMD_ABS_ADJ(v_b2);
            __mT v_b2m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_b2, v_abs_b2));
            __mT v_b2p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_b2, v_abs_b2));

            __mT v_cross_c1 = _mmT_add_pT(
                _mmT_sub_pT(vT_im1jp1, vT_im1jm1),
                _mmT_sub_pT(vT_jp1, vT_jm1));
            __mT v_c1 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(v_adj_one, _mmT_add_pT(vxi_im1, vxi_c)),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos2_t_inv,
                        _mmT_mul_pT(_mmT_sub_pT(vT_c, vT_im1), v_dp_inv))))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_im1, ve_c),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos_t_inv,
                        _mmT_mul_pT(v_4dt_inv, v_cross_c1))))));
            __mT v_abs_c1 = SIMD_ABS_ADJ(v_c1);
            __mT v_c1m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_c1, v_abs_c1));
            __mT v_c1p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_c1, v_abs_c1));

            __mT v_cross_c2 = _mmT_add_pT(
                _mmT_sub_pT(vT_jp1, vT_jm1),
                _mmT_sub_pT(vT_ip1jp1, vT_ip1jm1));
            __mT v_c2 = _mmT_add_pT(
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(v_adj_one, _mmT_add_pT(vxi_c, vxi_ip1)),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos2_t_inv,
                        _mmT_mul_pT(_mmT_sub_pT(vT_ip1, vT_c), v_dp_inv))))),
                _mmT_mul_pT(v_adj_neg1, _mmT_mul_pT(
                    _mmT_add_pT(ve_c, ve_ip1),
                    _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos_t_inv,
                        _mmT_mul_pT(v_4dt_inv, v_cross_c2))))));
            __mT v_abs_c2 = SIMD_ABS_ADJ(v_c2);
            __mT v_c2m = _mmT_mul_pT(v_adj_half, _mmT_sub_pT(v_c2, v_abs_c2));
            __mT v_c2p = _mmT_mul_pT(v_adj_half, _mmT_add_pT(v_c2, v_abs_c2));

            __mT v_d_t1 = _mmT_mul_pT(v_adj_neg2, _mmT_mul_pT(
                _mmT_add_pT(v_adj_one, _mmT_mul_pT(v_adj_two, vz_c)),
                _mmT_mul_pT(vr_inv, _mmT_mul_pT(_mmT_sub_pT(vT_kp1, vT_km1), v_2dr_inv))));
            __mT v_d_t2 = _mmT_mul_pT(
                _mmT_sub_pT(v_adj_one, _mmT_mul_pT(v_adj_two, vxi_c)),
                _mmT_mul_pT(vsin_t, _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos_t_inv,
                    _mmT_mul_pT(_mmT_sub_pT(vT_jp1, vT_jm1), v_2dt_inv)))));
            __mT v_d_t3 = _mmT_mul_pT(v_adj_two, _mmT_mul_pT(ve_c,
                _mmT_mul_pT(vsin_t, _mmT_mul_pT(vr2inv, _mmT_mul_pT(vcos2_t_inv,
                    _mmT_mul_pT(_mmT_sub_pT(vT_ip1, vT_im1), v_2dp_inv))))));
            __mT v_d = _mmT_add_pT(v_d_t1, _mmT_add_pT(v_d_t2, v_d_t3));

            __mT v_coe = _mmT_add_pT(
                _mmT_mul_pT(_mmT_sub_pT(v_a2p, v_a1m), v_dr_inv),
                _mmT_add_pT(
                    _mmT_mul_pT(_mmT_sub_pT(v_b2p, v_b1m), v_dt_inv),
                    _mmT_mul_pT(_mmT_sub_pT(v_c2p, v_c1m), v_dp_inv)));

            __mT v_Hadj = _mmT_add_pT(
                _mmT_mul_pT(_mmT_sub_pT(
                    _mmT_mul_pT(v_a1p, vtau_km1), _mmT_mul_pT(v_a2m, vtau_kp1)), v_dr_inv),
                _mmT_add_pT(
                    _mmT_mul_pT(_mmT_sub_pT(
                        _mmT_mul_pT(v_b1p, vtau_jm1), _mmT_mul_pT(v_b2m, vtau_jp1)), v_dt_inv),
                    _mmT_mul_pT(_mmT_sub_pT(
                        _mmT_mul_pT(v_c1p, vtau_im1), _mmT_mul_pT(v_c2m, vtau_ip1)), v_dp_inv)));

            __mT v_result = _mmT_div_pT(
                _mmT_add_pT(vtau_old, v_Hadj),
                _mmT_add_pT(v_coe, v_d));

            __mT v_abs_coe = SIMD_ABS_ADJ(v_coe);
#if USE_AVX512
            __mmaskT mask_zero_adj = _mm512_cmp_pT_mask(v_abs_coe, v_eps_adj, _CMP_LT_OQ);
            v_result = _mm512_mask_blend_pT(mask_zero_adj, v_result, v_adj_zero);
#else
            __mT mask_zero_adj = _mm256_cmp_pT(v_abs_coe, v_eps_adj, _CMP_LT_OQ);
            v_result = _mm256_blendv_pd(v_result, v_adj_zero, mask_zero_adj);
#endif

            _mmT_store_pT(dump_c__, v_result);
            for (int i = 0; i < n_valid; i++) {
                if (d_is_interior[i]) {
                    grid.tau_loc[d_center[i]] = dump_c__[i];
                } else if (d_is_global_bnd[i]) {
                    grid.tau_loc[d_center[i]] = _0_CR;
                }
            }

        } // end i_vec loop

        // mpi synchronization
        synchronize_all_sub();

    } // end i_level loop

    #undef SIMD_ABS_ADJ

#endif // USE_SIMD

}


// ERROR index!!!!!!!!!!!!!
Iterator_level_1st_order::Iterator_level_1st_order(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                         : Iterator_level(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}


void Iterator_level_1st_order::do_sweep(int iswp, Grid& grid, InputParams& IP){

    if(!use_gpu){

#if !defined USE_SIMD

        // set sweep direction
        set_sweep_direction(iswp);

        int iip, jjt, kkr;
        int n_levels = ijk_for_this_subproc.size();

        for (int i_level = 0; i_level < n_levels; i_level++) {
            size_t n_nodes = ijk_for_this_subproc[i_level].size();

            for (size_t i_node = 0; i_node < n_nodes; i_node++) {

                V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

                if (r_dirc < 0) kkr = nr-kkr; //kk-1;
                else            kkr = kkr-1;  //nr-kk;
                if (t_dirc < 0) jjt = nt-jjt; //jj-1;
                else            jjt = jjt-1;  //nt-jj;
                if (p_dirc < 0) iip = np-iip; //ii-1;
                else            iip = iip-1;  //np-ii;

                //
                // calculate stencils
                //
                if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                    calculate_stencil_1st_order(grid, iip, jjt, kkr);
                } // is_changed == true
            } // end ijk

            // mpi synchronization
            synchronize_all_sub();

        } // end loop i_level

#elif USE_AVX512 || USE_AVX

        // preload constants
        __mT v_DP_inv      = _mmT_set1_pT(1.0/dp);
        __mT v_DT_inv      = _mmT_set1_pT(1.0/dt);
        __mT v_DR_inv      = _mmT_set1_pT(1.0/dr);
        __mT v_DP_inv_half = _mmT_set1_pT(1.0/dp*0.5);
        __mT v_DT_inv_half = _mmT_set1_pT(1.0/dt*0.5);
        __mT v_DR_inv_half = _mmT_set1_pT(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1;
        __mT v_pp2;
        __mT v_pt1;
        __mT v_pt2;
        __mT v_pr1;
        __mT v_pr2;

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            __mT* v_iip    = (__mT*) vv_iip.at(iswp).at(i_level);
            __mT* v_jjt    = (__mT*) vv_jjt.at(iswp).at(i_level);
            __mT* v_kkr    = (__mT*) vv_kkr.at(iswp).at(i_level);

            __mT* v_fac_a  = (__mT*) vv_fac_a.at(iswp).at(i_level);
            __mT* v_fac_b  = (__mT*) vv_fac_b.at(iswp).at(i_level);
            __mT* v_fac_c  = (__mT*) vv_fac_c.at(iswp).at(i_level);
            __mT* v_fac_f  = (__mT*) vv_fac_f.at(iswp).at(i_level);
            __mT* v_T0v    = (__mT*) vv_T0v.at(iswp).at(i_level);
            __mT* v_T0r    = (__mT*) vv_T0r.at(iswp).at(i_level);
            __mT* v_T0t    = (__mT*) vv_T0t.at(iswp).at(i_level);
            __mT* v_T0p    = (__mT*) vv_T0p.at(iswp).at(i_level);
            __mT* v_fun    = (__mT*) vv_fun.at(iswp).at(i_level);
            __mT* v_change = (__mT*) vv_change.at(iswp).at(i_level);

            // alias for dumped index
            int* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            int* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            int* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            int* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            int* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            int* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            int* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {

                int i_vec = _i_vec * NSIMD;
                __mT v_c__    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijk[i_vec]);
                __mT v_p__    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ip1jk[i_vec]);
                __mT v_m__    = load_mem_gen_to_mTd(grid.tau_loc, &dump_im1jk[i_vec]);
                __mT v__p_    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijp1k[i_vec]);
                __mT v__m_    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijm1k[i_vec]);
                __mT v___p    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijkp1[i_vec]);
                __mT v___m    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijkm1[i_vec]);

                // loop over all nodes in one level
                vect_stencil_1st_pre_simd(v_iip[_i_vec], v_jjt[_i_vec], v_kkr[_i_vec], \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                // calculate updated value on c
                vect_stencil_1st_3rd_apre_simd(v_c__, v_fac_a[_i_vec], v_fac_b[_i_vec], v_fac_c[_i_vec], v_fac_f[_i_vec], \
                                               v_T0v[_i_vec], v_T0p[_i_vec]  , v_T0t[_i_vec]  , v_T0r[_i_vec]  , v_fun[_i_vec]  , v_change[_i_vec], \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // store v_c__ to dump_c__
                _mmT_store_pT(dump_c__, v_c__);


                for (int i = 0; i < NSIMD; i++) {
                    if(i_vec+i>=n_nodes) break;

                    grid.tau_loc[dump_ijk[i_vec+i]] = dump_c__[i];
                }



            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub();

        } // end of i_level loop

#elif USE_ARM_SVE

        svbool_t pg;
        //
        __mT v_DP_inv      = svdup_f64(1.0/dp);
        __mT v_DT_inv      = svdup_f64(1.0/dt);
        __mT v_DR_inv      = svdup_f64(1.0/dr);
        __mT v_DP_inv_half = svdup_f64(1.0/dp*0.5);
        __mT v_DT_inv_half = svdup_f64(1.0/dt*0.5);
        __mT v_DR_inv_half = svdup_f64(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1;
        __mT v_pp2;
        __mT v_pt1;
        __mT v_pt2;
        __mT v_pr1;
        __mT v_pr2;

        __mT v_c__   ;
        __mT v_p__   ;
        __mT v_m__   ;
        __mT v__p_   ;
        __mT v__m_   ;
        __mT v___p   ;
        __mT v___m   ;

        __mT v_iip_   ;
        __mT v_jjt_   ;
        __mT v_kkr_   ;
        __mT v_fac_a_ ;
        __mT v_fac_b_ ;
        __mT v_fac_c_ ;
        __mT v_fac_f_ ;
        __mT v_T0v_   ;
        __mT v_T0r_   ;
        __mT v_T0t_   ;
        __mT v_T0p_   ;
        __mT v_fun_   ;
        __mT v_change_;


        // measure time for only loop
        //auto start = std::chrono::high_resolution_clock::now();

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();
            //std::cout << "n_nodes = " << n_nodes << std::endl;

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            CUSTOMREAL* v_iip    = vv_iip.at(iswp).at(i_level);
            CUSTOMREAL* v_jjt    = vv_jjt.at(iswp).at(i_level);
            CUSTOMREAL* v_kkr    = vv_kkr.at(iswp).at(i_level);

            CUSTOMREAL* v_fac_a  = vv_fac_a.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_b  = vv_fac_b.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_c  = vv_fac_c.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_f  = vv_fac_f.at(iswp).at(i_level);
            CUSTOMREAL* v_T0v    = vv_T0v.at(iswp).at(i_level);
            CUSTOMREAL* v_T0r    = vv_T0r.at(iswp).at(i_level);
            CUSTOMREAL* v_T0t    = vv_T0t.at(iswp).at(i_level);
            CUSTOMREAL* v_T0p    = vv_T0p.at(iswp).at(i_level);
            CUSTOMREAL* v_fun    = vv_fun.at(iswp).at(i_level);
            CUSTOMREAL* v_change = vv_change.at(iswp).at(i_level);

            // alias for dumped index
            uint64_t* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            uint64_t* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            uint64_t* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            uint64_t* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            uint64_t* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            uint64_t* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            uint64_t* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {
                int i_vec = _i_vec * NSIMD;

                pg = svwhilelt_b64(i_vec, n_nodes);

                v_c__    = load_mem_gen_to_mTd(pg, grid.tau_loc,  &dump_ijk[i_vec]);
                v_p__    = load_mem_gen_to_mTd(pg, grid.tau_loc,  &dump_ip1jk[i_vec]);
                v_m__    = load_mem_gen_to_mTd(pg, grid.tau_loc,  &dump_im1jk[i_vec]);
                v__p_    = load_mem_gen_to_mTd(pg, grid.tau_loc,  &dump_ijp1k[i_vec]);
                v__m_    = load_mem_gen_to_mTd(pg, grid.tau_loc,  &dump_ijm1k[i_vec]);
                v___p    = load_mem_gen_to_mTd(pg, grid.tau_loc,  &dump_ijkp1[i_vec]);
                v___m    = load_mem_gen_to_mTd(pg, grid.tau_loc,  &dump_ijkm1[i_vec]);

                // load v_iip, v_jjt, v_kkr
                v_iip_   = svld1_vnum_f64(pg, v_iip   , _i_vec);
                v_jjt_   = svld1_vnum_f64(pg, v_jjt   , _i_vec);
                v_kkr_   = svld1_vnum_f64(pg, v_kkr   , _i_vec);
                v_fac_a_ = svld1_vnum_f64(pg, v_fac_a , _i_vec);
                v_fac_b_ = svld1_vnum_f64(pg, v_fac_b , _i_vec);
                v_fac_c_ = svld1_vnum_f64(pg, v_fac_c , _i_vec);
                v_fac_f_ = svld1_vnum_f64(pg, v_fac_f , _i_vec);
                v_T0v_   = svld1_vnum_f64(pg, v_T0v   , _i_vec);
                v_T0r_   = svld1_vnum_f64(pg, v_T0r   , _i_vec);
                v_T0t_   = svld1_vnum_f64(pg, v_T0t   , _i_vec);
                v_T0p_   = svld1_vnum_f64(pg, v_T0p   , _i_vec);
                v_fun_   = svld1_vnum_f64(pg, v_fun   , _i_vec);
                v_change_= svld1_vnum_f64(pg, v_change, _i_vec);

                // loop over all nodes in one level
                vect_stencil_1st_pre_simd(pg, v_iip_, v_jjt_, v_kkr_, \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                // calculate updated value on c
                vect_stencil_1st_3rd_apre_simd(pg, v_c__, v_fac_a_, v_fac_b_, v_fac_c_, v_fac_f_, \
                                               v_T0v_, v_T0p_, v_T0t_, v_T0r_, v_fun_, v_change_, \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // store v_c__ to dump_c__
                svst1_scatter_u64index_f64(pg, grid.tau_loc, svld1_u64(pg,&dump_ijk[i_vec]), v_c__);


            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub();

        } // end of i_level loop

#endif // ifndef USE_SIMD

    } // end of if !use_gpu
    else { // if use_gpu

#if defined USE_CUDA

        // copy tau to device
        cuda_copy_tau_to_device(gpu_grid, grid.tau_loc);

        // run iteration
        cuda_run_iteration_forward(gpu_grid, iswp);

        // copy tau to host
        cuda_copy_tau_to_host(gpu_grid, grid.tau_loc);

#else // !defiend USE_CUDA
        // exit code
        std::cout << "Error: USE_CUDA is not defined" << std::endl;
        exit(1);
#endif

    } // end of if use_gpu

    // update boundary
    if (subdom_main) {
        calculate_boundary_nodes(grid);
    }
}


// ERROR index!!!!!!!!!!!!!
Iterator_level_3rd_order::Iterator_level_3rd_order(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                         : Iterator_level(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}


void Iterator_level_3rd_order::do_sweep(int iswp, Grid& grid, InputParams& IP){

    if(!use_gpu) {

#if !defined USE_SIMD

        // set sweep direction
        set_sweep_direction(iswp);

        int iip, jjt, kkr;
        int n_levels = ijk_for_this_subproc.size();

        for (int i_level = 0; i_level < n_levels; i_level++) {
            size_t n_nodes = ijk_for_this_subproc[i_level].size();

            for (size_t i_node = 0; i_node < n_nodes; i_node++) {

                V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

                if (r_dirc < 0) kkr = nr-kkr; //kk-1;
                else            kkr = kkr-1;  //nr-kk;
                if (t_dirc < 0) jjt = nt-jjt; //jj-1;
                else            jjt = jjt-1;  //nt-jj;
                if (p_dirc < 0) iip = np-iip; //ii-1;
                else            iip = iip-1;  //np-ii;

                //
                // calculate stencils
                //
                if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                    calculate_stencil_3rd_order(grid, iip, jjt, kkr);
                } // is_changed == true
            } // end ijk

            // mpi synchronization
            synchronize_all_sub();

        } // end loop i_level

#elif USE_AVX512 || USE_AVX

        //
        __mT v_DP_inv      = _mmT_set1_pT(1.0/dp);
        __mT v_DT_inv      = _mmT_set1_pT(1.0/dt);
        __mT v_DR_inv      = _mmT_set1_pT(1.0/dr);
        __mT v_DP_inv_half = _mmT_set1_pT(1.0/dp*0.5);
        __mT v_DT_inv_half = _mmT_set1_pT(1.0/dt*0.5);
        __mT v_DR_inv_half = _mmT_set1_pT(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1 = _mmT_set1_pT(0.0);
        __mT v_pp2 = _mmT_set1_pT(0.0);
        __mT v_pt1 = _mmT_set1_pT(0.0);
        __mT v_pt2 = _mmT_set1_pT(0.0);
        __mT v_pr1 = _mmT_set1_pT(0.0);
        __mT v_pr2 = _mmT_set1_pT(0.0);

        // measure time for only loop
        //auto start = std::chrono::high_resolution_clock::now();

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();
            //std::cout << "n_nodes = " << n_nodes << std::endl;

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            __mT* v_iip    = (__mT*) vv_iip.at(iswp).at(i_level);
            __mT* v_jjt    = (__mT*) vv_jjt.at(iswp).at(i_level);
            __mT* v_kkr    = (__mT*) vv_kkr.at(iswp).at(i_level);

            __mT* v_fac_a  = (__mT*) vv_fac_a.at(iswp).at(i_level);
            __mT* v_fac_b  = (__mT*) vv_fac_b.at(iswp).at(i_level);
            __mT* v_fac_c  = (__mT*) vv_fac_c.at(iswp).at(i_level);
            __mT* v_fac_f  = (__mT*) vv_fac_f.at(iswp).at(i_level);
            __mT* v_T0v    = (__mT*) vv_T0v.at(iswp).at(i_level);
            __mT* v_T0r    = (__mT*) vv_T0r.at(iswp).at(i_level);
            __mT* v_T0t    = (__mT*) vv_T0t.at(iswp).at(i_level);
            __mT* v_T0p    = (__mT*) vv_T0p.at(iswp).at(i_level);
            __mT* v_fun    = (__mT*) vv_fun.at(iswp).at(i_level);
            __mT* v_change = (__mT*) vv_change.at(iswp).at(i_level);

            // alias for dumped index
            int* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            int* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            int* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            int* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            int* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            int* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            int* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);
            int* dump_ip2jk = vv_ip2j__k__.at(iswp).at(i_level);
            int* dump_im2jk = vv_im2j__k__.at(iswp).at(i_level);
            int* dump_ijp2k = vv_i__jp2k__.at(iswp).at(i_level);
            int* dump_ijm2k = vv_i__jm2k__.at(iswp).at(i_level);
            int* dump_ijkp2 = vv_i__j__kp2.at(iswp).at(i_level);
            int* dump_ijkm2 = vv_i__j__km2.at(iswp).at(i_level); /////////////////////////////////////

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {

                int i_vec = _i_vec * NSIMD;
                __mT v_c__    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijk  [i_vec]);
                __mT v_p__    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ip1jk[i_vec]);
                __mT v_m__    = load_mem_gen_to_mTd(grid.tau_loc, &dump_im1jk[i_vec]);
                __mT v__p_    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijp1k[i_vec]);
                __mT v__m_    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijm1k[i_vec]);
                __mT v___p    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijkp1[i_vec]);
                __mT v___m    = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijkm1[i_vec]);
                __mT v_pp____ = load_mem_gen_to_mTd(grid.tau_loc, &dump_ip2jk[i_vec]);
                __mT v_mm____ = load_mem_gen_to_mTd(grid.tau_loc, &dump_im2jk[i_vec]);
                __mT v___pp__ = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijp2k[i_vec]);
                __mT v___mm__ = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijm2k[i_vec]);
                __mT v_____pp = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijkp2[i_vec]);
                __mT v_____mm = load_mem_gen_to_mTd(grid.tau_loc, &dump_ijkm2[i_vec]);

                // loop over all nodes in one level
                vect_stencil_3rd_pre_simd(v_iip[_i_vec], v_jjt[_i_vec], v_kkr[_i_vec], \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                // calculate updated value on c
                vect_stencil_1st_3rd_apre_simd(v_c__, v_fac_a[_i_vec], v_fac_b[_i_vec], v_fac_c[_i_vec], v_fac_f[_i_vec], \
                                               v_T0v[_i_vec], v_T0p[_i_vec]  , v_T0t[_i_vec]  , v_T0r[_i_vec]  , v_fun[_i_vec]  , v_change[_i_vec], \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // store v_c__ to dump_c__
                _mmT_store_pT(dump_c__, v_c__);


                for (int i = 0; i < NSIMD; i++) {
                    if(i_vec+i>=n_nodes) break;

                    grid.tau_loc[dump_ijk[i_vec+i]] = dump_c__[i];
                }



            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub(); // dead lock

        } // end of i_level loop

#elif USE_ARM_SVE
        //
        svbool_t pg;

        __mT v_DP_inv      = svdup_f64(1.0/dp);
        __mT v_DT_inv      = svdup_f64(1.0/dt);
        __mT v_DR_inv      = svdup_f64(1.0/dr);
        __mT v_DP_inv_half = svdup_f64(1.0/dp*0.5);
        __mT v_DT_inv_half = svdup_f64(1.0/dt*0.5);
        __mT v_DR_inv_half = svdup_f64(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1;
        __mT v_pp2;
        __mT v_pt1;
        __mT v_pt2;
        __mT v_pr1;
        __mT v_pr2;

        __mT v_c__   ;
        __mT v_p__   ;
        __mT v_m__   ;
        __mT v__p_   ;
        __mT v__m_   ;
        __mT v___p   ;
        __mT v___m   ;
        __mT v_pp____;
        __mT v_mm____;
        __mT v___pp__;
        __mT v___mm__;
        __mT v_____pp;
        __mT v_____mm;

        __mT v_iip_   ;
        __mT v_jjt_   ;
        __mT v_kkr_   ;
        __mT v_fac_a_ ;
        __mT v_fac_b_ ;
        __mT v_fac_c_ ;
        __mT v_fac_f_ ;
        __mT v_T0v_   ;
        __mT v_T0r_   ;
        __mT v_T0t_   ;
        __mT v_T0p_   ;
        __mT v_fun_   ;
        __mT v_change_;


        // measure time for only loop
        //auto start = std::chrono::high_resolution_clock::now();

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();
            //std::cout << "n_nodes = " << n_nodes << std::endl;

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            CUSTOMREAL* v_iip    = vv_iip.at(iswp).at(i_level);
            CUSTOMREAL* v_jjt    = vv_jjt.at(iswp).at(i_level);
            CUSTOMREAL* v_kkr    = vv_kkr.at(iswp).at(i_level);

            CUSTOMREAL* v_fac_a  = vv_fac_a.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_b  = vv_fac_b.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_c  = vv_fac_c.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_f  = vv_fac_f.at(iswp).at(i_level);
            CUSTOMREAL* v_T0v    = vv_T0v.at(iswp).at(i_level);
            CUSTOMREAL* v_T0r    = vv_T0r.at(iswp).at(i_level);
            CUSTOMREAL* v_T0t    = vv_T0t.at(iswp).at(i_level);
            CUSTOMREAL* v_T0p    = vv_T0p.at(iswp).at(i_level);
            CUSTOMREAL* v_fun    = vv_fun.at(iswp).at(i_level);
            CUSTOMREAL* v_change = vv_change.at(iswp).at(i_level);

            // alias for dumped index
            uint64_t* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            uint64_t* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            uint64_t* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            uint64_t* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            uint64_t* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            uint64_t* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            uint64_t* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);
            uint64_t* dump_ip2jk = vv_ip2j__k__.at(iswp).at(i_level);
            uint64_t* dump_im2jk = vv_im2j__k__.at(iswp).at(i_level);
            uint64_t* dump_ijp2k = vv_i__jp2k__.at(iswp).at(i_level);
            uint64_t* dump_ijm2k = vv_i__jm2k__.at(iswp).at(i_level);
            uint64_t* dump_ijkp2 = vv_i__j__kp2.at(iswp).at(i_level);
            uint64_t* dump_ijkm2 = vv_i__j__km2.at(iswp).at(i_level);

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {
                int i_vec = _i_vec * NSIMD;

                pg = svwhilelt_b64(i_vec, n_nodes);

                v_c__    = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijk  [i_vec]);
                v_p__    = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ip1jk[i_vec]);
                v_m__    = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_im1jk[i_vec]);
                v__p_    = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijp1k[i_vec]);
                v__m_    = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijm1k[i_vec]);
                v___p    = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijkp1[i_vec]);
                v___m    = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijkm1[i_vec]);
                v_pp____ = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ip2jk[i_vec]);
                v_mm____ = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_im2jk[i_vec]);
                v___pp__ = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijp2k[i_vec]);
                v___mm__ = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijm2k[i_vec]);
                v_____pp = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijkp2[i_vec]);
                v_____mm = load_mem_gen_to_mTd(pg, grid.tau_loc, &dump_ijkm2[i_vec]);

                // load v_iip, v_jjt, v_kkr
                v_iip_   = svld1_vnum_f64(pg, v_iip   , _i_vec);
                v_jjt_   = svld1_vnum_f64(pg, v_jjt   , _i_vec);
                v_kkr_   = svld1_vnum_f64(pg, v_kkr   , _i_vec);
                v_fac_a_ = svld1_vnum_f64(pg, v_fac_a , _i_vec);
                v_fac_b_ = svld1_vnum_f64(pg, v_fac_b , _i_vec);
                v_fac_c_ = svld1_vnum_f64(pg, v_fac_c , _i_vec);
                v_fac_f_ = svld1_vnum_f64(pg, v_fac_f , _i_vec);
                v_T0v_   = svld1_vnum_f64(pg, v_T0v   , _i_vec);
                v_T0r_   = svld1_vnum_f64(pg, v_T0r   , _i_vec);
                v_T0t_   = svld1_vnum_f64(pg, v_T0t   , _i_vec);
                v_T0p_   = svld1_vnum_f64(pg, v_T0p   , _i_vec);
                v_fun_   = svld1_vnum_f64(pg, v_fun   , _i_vec);
                v_change_= svld1_vnum_f64(pg, v_change, _i_vec);

                // loop over all nodes in one level
                vect_stencil_3rd_pre_simd(pg, v_iip_, v_jjt_, v_kkr_, \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                //// calculate updated value on c
                vect_stencil_1st_3rd_apre_simd(pg, v_c__, v_fac_a_, v_fac_b_, v_fac_c_, v_fac_f_, \
                                               v_T0v_, v_T0p_, v_T0t_, v_T0r_, v_fun_, v_change_, \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // store v_c__ to dump_c__
                svst1_scatter_u64index_f64(pg, grid.tau_loc, svld1_u64(pg,&dump_ijk[i_vec]), v_c__);
            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub();

        } // end of i_level loop

#endif // ifndef USE_SIMD

    } // end of if !use_gpu
    else { // if use_gpu

#if defined USE_CUDA

        // copy tau to device
        cuda_copy_tau_to_device(gpu_grid, grid.tau_loc);

        // run iteration
        cuda_run_iteration_forward(gpu_grid, iswp);

        // copy tau to host
        cuda_copy_tau_to_host(gpu_grid, grid.tau_loc);

#else // !defiend USE_CUDA
        // exit code
        std::cout << "Error: USE_CUDA is not defined" << std::endl;
        exit(1);
#endif

    } // end of if use_gpu

    // update boundary
    if (subdom_main) {
        calculate_boundary_nodes(grid);
    }

}


Iterator_level_1st_order_upwind::Iterator_level_1st_order_upwind(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                         : Iterator_level(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}

void Iterator_level_1st_order_upwind::do_sweep(int iswp, Grid& grid, InputParams& IP){

    // set sweep direction
    set_sweep_direction(iswp);

    int iip, jjt, kkr;
    int n_levels = ijk_for_this_subproc.size();

    for (int i_level = 0; i_level < n_levels; i_level++) {
        size_t n_nodes = ijk_for_this_subproc[i_level].size();

        for (size_t i_node = 0; i_node < n_nodes; i_node++) {

            V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

            if (r_dirc < 0) kkr = nr-1-kkr;
            if (t_dirc < 0) jjt = nt-1-jjt;
            if (p_dirc < 0) iip = np-1-iip;

            //
            // calculate stencils
            //
            if (grid.is_changed[I2V(iip, jjt, kkr)]) {
                calculate_stencil_1st_order_upwind(grid, iip, jjt, kkr);
            } // is_changed == true
        } // end ijk

        // mpi synchronization
        synchronize_all_sub();

    } // end loop i_level
}

// ERROR index!!!!!!!!!!!!!
Iterator_level_1st_order_tele::Iterator_level_1st_order_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                         : Iterator_level_tele(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}

void Iterator_level_1st_order_tele::do_sweep(int iswp, Grid& grid, InputParams& IP){

    if(!use_gpu) {

#if !defined USE_SIMD

        // set sweep direction
        set_sweep_direction(iswp);

        int iip, jjt, kkr;
        int n_levels = ijk_for_this_subproc.size();

        for (int i_level = 0; i_level < n_levels; i_level++) {
            size_t n_nodes = ijk_for_this_subproc[i_level].size();

            for (size_t i_node = 0; i_node < n_nodes; i_node++) {

                V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

                if (r_dirc < 0) kkr = nr-kkr-1;
                //else            kkr = kkr;
                if (t_dirc < 0) jjt = nt-jjt-1;
                //else            jjt = jjt;
                if (p_dirc < 0) iip = np-iip-1;
                //else            iip = iip;

                //
                // calculate stencils
                //
                if (iip != 0 && iip != np-1 && jjt != 0 && jjt != nt-1 && kkr != 0 && kkr != nr-1) {
                    // calculate stencils
                    calculate_stencil_1st_order_tele(grid, iip, jjt, kkr);
                } else {
                    // update boundary
                    calculate_boundary_nodes_tele(grid, iip, jjt, kkr);
                }
            } // end ijk

            // mpi synchronization
            synchronize_all_sub();

        } // end loop i_level

#elif USE_AVX512 || USE_AVX

        // preload constants
        __mT v_DP_inv      = _mmT_set1_pT(1.0/dp);
        __mT v_DT_inv      = _mmT_set1_pT(1.0/dt);
        __mT v_DR_inv      = _mmT_set1_pT(1.0/dr);
        __mT v_DP_inv_half = _mmT_set1_pT(1.0/dp*0.5);
        __mT v_DT_inv_half = _mmT_set1_pT(1.0/dt*0.5);
        __mT v_DR_inv_half = _mmT_set1_pT(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1;
        __mT v_pp2;
        __mT v_pt1;
        __mT v_pt2;
        __mT v_pr1;
        __mT v_pr2;

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            __mT* v_iip    = (__mT*) vv_iip.at(iswp).at(i_level);
            __mT* v_jjt    = (__mT*) vv_jjt.at(iswp).at(i_level);
            __mT* v_kkr    = (__mT*) vv_kkr.at(iswp).at(i_level);

            __mT* v_fac_a  = (__mT*) vv_fac_a.at(iswp).at(i_level);
            __mT* v_fac_b  = (__mT*) vv_fac_b.at(iswp).at(i_level);
            __mT* v_fac_c  = (__mT*) vv_fac_c.at(iswp).at(i_level);
            __mT* v_fac_f  = (__mT*) vv_fac_f.at(iswp).at(i_level);
            __mT* v_fun    = (__mT*) vv_fun.at(iswp).at(i_level);
            __mT* v_change = (__mT*) vv_change.at(iswp).at(i_level);

            // alias for dumped index
            int* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            int* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            int* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            int* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            int* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            int* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            int* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);
            int* dump_ip2jk = vv_ip2j__k__.at(iswp).at(i_level);
            int* dump_im2jk = vv_im2j__k__.at(iswp).at(i_level);
            int* dump_ijp2k = vv_i__jp2k__.at(iswp).at(i_level);
            int* dump_ijm2k = vv_i__jm2k__.at(iswp).at(i_level);
            int* dump_ijkp2 = vv_i__j__kp2.at(iswp).at(i_level);
            int* dump_ijkm2 = vv_i__j__km2.at(iswp).at(i_level);

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {

                int i_vec = _i_vec * NSIMD;
                __mT v_c__    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijk[i_vec]);
                __mT v_p__    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ip1jk[i_vec]);
                __mT v_m__    = load_mem_gen_to_mTd(grid.T_loc,   &dump_im1jk[i_vec]);
                __mT v__p_    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijp1k[i_vec]);
                __mT v__m_    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijm1k[i_vec]);
                __mT v___p    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijkp1[i_vec]);
                __mT v___m    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijkm1[i_vec]);
                __mT v_pp____ = load_mem_gen_to_mTd(grid.T_loc, &dump_ip2jk[i_vec]);
                __mT v_mm____ = load_mem_gen_to_mTd(grid.T_loc, &dump_im2jk[i_vec]);
                __mT v___pp__ = load_mem_gen_to_mTd(grid.T_loc, &dump_ijp2k[i_vec]);
                __mT v___mm__ = load_mem_gen_to_mTd(grid.T_loc, &dump_ijm2k[i_vec]);
                __mT v_____pp = load_mem_gen_to_mTd(grid.T_loc, &dump_ijkp2[i_vec]);
                __mT v_____mm = load_mem_gen_to_mTd(grid.T_loc, &dump_ijkm2[i_vec]);


                // loop over all nodes in one level (this routine for teleseismic is same with that for local)
                vect_stencil_1st_pre_simd(v_iip[_i_vec], v_jjt[_i_vec], v_kkr[_i_vec], \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                // calculate updated value on c
                vect_stencil_1st_3rd_apre_simd_tele(v_c__, v_fac_a[_i_vec], v_fac_b[_i_vec], v_fac_c[_i_vec], v_fac_f[_i_vec], \
                                               v_fun[_i_vec], v_change[_i_vec], \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // calculate the values on boudaries
                calculate_boundary_nodes_tele_simd(v_iip[_i_vec], v_jjt[_i_vec], v_kkr[_i_vec], \
                                                   v_c__, \
                                                   v_p__,   v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                                   v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                                   v_change[_i_vec], \
                                                   loc_I, loc_J, loc_K);

                // store v_c__ to dump_c__
                _mmT_store_pT(dump_c__, v_c__);

                for (int i = 0; i < NSIMD; i++) {
                    if(i_vec+i>=n_nodes) break;

                    grid.T_loc[dump_ijk[i_vec+i]] = dump_c__[i];
                }



            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub();

        } // end of i_level loop

#elif USE_ARM_SVE

        svbool_t pg;
        //
        __mT v_DP_inv      = svdup_f64(1.0/dp);
        __mT v_DT_inv      = svdup_f64(1.0/dt);
        __mT v_DR_inv      = svdup_f64(1.0/dr);
        __mT v_DP_inv_half = svdup_f64(1.0/dp*0.5);
        __mT v_DT_inv_half = svdup_f64(1.0/dt*0.5);
        __mT v_DR_inv_half = svdup_f64(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1;
        __mT v_pp2;
        __mT v_pt1;
        __mT v_pt2;
        __mT v_pr1;
        __mT v_pr2;

        __mT v_c__   ;
        __mT v_p__   ;
        __mT v_m__   ;
        __mT v__p_   ;
        __mT v__m_   ;
        __mT v___p   ;
        __mT v___m   ;
        __mT v_pp____;
        __mT v_mm____;
        __mT v___pp__;
        __mT v___mm__;
        __mT v_____pp;
        __mT v_____mm;

        __mT v_iip_   ;
        __mT v_jjt_   ;
        __mT v_kkr_   ;
        __mT v_fac_a_ ;
        __mT v_fac_b_ ;
        __mT v_fac_c_ ;
        __mT v_fac_f_ ;
        __mT v_fun_   ;
        __mT v_change_;

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            CUSTOMREAL* v_iip    = vv_iip.at(iswp).at(i_level);
            CUSTOMREAL* v_jjt    = vv_jjt.at(iswp).at(i_level);
            CUSTOMREAL* v_kkr    = vv_kkr.at(iswp).at(i_level);

            CUSTOMREAL* v_fac_a  = vv_fac_a.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_b  = vv_fac_b.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_c  = vv_fac_c.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_f  = vv_fac_f.at(iswp).at(i_level);
            CUSTOMREAL* v_fun    = vv_fun.at(iswp).at(i_level);
            CUSTOMREAL* v_change = vv_change.at(iswp).at(i_level);

            // alias for dumped index
            uint64_t* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            uint64_t* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            uint64_t* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            uint64_t* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            uint64_t* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            uint64_t* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            uint64_t* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);
            uint64_t* dump_ip2jk = vv_ip2j__k__.at(iswp).at(i_level);
            uint64_t* dump_im2jk = vv_im2j__k__.at(iswp).at(i_level);
            uint64_t* dump_ijp2k = vv_i__jp2k__.at(iswp).at(i_level);
            uint64_t* dump_ijm2k = vv_i__jm2k__.at(iswp).at(i_level);
            uint64_t* dump_ijkp2 = vv_i__j__kp2.at(iswp).at(i_level);
            uint64_t* dump_ijkm2 = vv_i__j__km2.at(iswp).at(i_level);

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {
                int i_vec = _i_vec * NSIMD;

                pg = svwhilelt_b64(i_vec, n_nodes);

                v_c__    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijk[i_vec]);
                v_p__    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ip1jk[i_vec]);
                v_m__    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_im1jk[i_vec]);
                v__p_    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijp1k[i_vec]);
                v__m_    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijm1k[i_vec]);
                v___p    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkp1[i_vec]);
                v___m    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkm1[i_vec]);
                v_pp____ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ip2jk[i_vec]);
                v_mm____ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_im2jk[i_vec]);
                v___pp__ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijp2k[i_vec]);
                v___mm__ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijm2k[i_vec]);
                v_____pp = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkp2[i_vec]);
                v_____mm = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkm2[i_vec]);

                // load v_iip, v_jjt, v_kkr
                v_iip_   = svld1_vnum_f64(pg, v_iip   , _i_vec);
                v_jjt_   = svld1_vnum_f64(pg, v_jjt   , _i_vec);
                v_kkr_   = svld1_vnum_f64(pg, v_kkr   , _i_vec);
                v_fac_a_ = svld1_vnum_f64(pg, v_fac_a , _i_vec);
                v_fac_b_ = svld1_vnum_f64(pg, v_fac_b , _i_vec);
                v_fac_c_ = svld1_vnum_f64(pg, v_fac_c , _i_vec);
                v_fac_f_ = svld1_vnum_f64(pg, v_fac_f , _i_vec);
                v_fun_   = svld1_vnum_f64(pg, v_fun   , _i_vec);
                v_change_= svld1_vnum_f64(pg, v_change, _i_vec);

                // loop over all nodes in one level
                vect_stencil_1st_pre_simd(pg, v_iip_, v_jjt_, v_kkr_, \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                // calculate updated value on c
                vect_stencil_1st_3rd_apre_simd_tele(pg, v_c__, v_fac_a_, v_fac_b_, v_fac_c_, v_fac_f_, \
                                               v_fun_, v_change_, \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // calculate the values on boundaries (teleseismic)
                calculate_boundary_nodes_tele_simd(pg, v_iip_, v_jjt_, v_kkr_, \
                                                   v_c__, \
                                                   v_p__,   v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                                   v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                                   v_change_, \
                                                   loc_I, loc_J, loc_K);

                // store v_c__ to T_loc
                svst1_scatter_u64index_f64(pg, grid.T_loc, svld1_u64(pg,&dump_ijk[i_vec]), v_c__);

            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub();

        } // end of i_level loop

#endif // ifndef USE_SIMD

    } // end of if !use_gpu
    else { // if use_gpu

#if defined USE_CUDA

        //// copy tau to device
        //cuda_copy_tau_to_device(gpu_grid, grid.tau_loc);

        //// run iteration
        //cuda_run_iteration_forward_tele(gpu_grid, iswp);

        //// copy tau to host
        //cuda_copy_tau_to_host(gpu_grid, grid.tau_loc);

#else // !defiend USE_CUDA
        // exit code
        std::cout << "Error: USE_CUDA is not defined" << std::endl;
        exit(1);
#endif

    } // end of if use_gpu


}

// ERROR index!!!!!!!!!!!!!
Iterator_level_3rd_order_tele::Iterator_level_3rd_order_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                         : Iterator_level_tele(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}


void Iterator_level_3rd_order_tele::do_sweep(int iswp, Grid& grid, InputParams& IP){

    if(!use_gpu) {

#if !defined USE_SIMD

        // set sweep direction
        set_sweep_direction(iswp);

        int iip, jjt, kkr;
        int n_levels = ijk_for_this_subproc.size();

        for (int i_level = 0; i_level < n_levels; i_level++) {
            size_t n_nodes = ijk_for_this_subproc[i_level].size();

            for (size_t i_node = 0; i_node < n_nodes; i_node++) {

                V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

                if (r_dirc < 0) kkr = nr-kkr-1;
                //else            kkr = kkr;
                if (t_dirc < 0) jjt = nt-jjt-1;
                //else            jjt = jjt;
                if (p_dirc < 0) iip = np-iip-1;
                //else            iip = iip;

                //
                // calculate stencils
                //
                if (iip != 0 && iip != np-1 && jjt != 0 && jjt != nt-1 && kkr != 0 && kkr != nr-1) {
                    // calculate stencils
                    calculate_stencil_3rd_order_tele(grid, iip, jjt, kkr);
                } else {
                    // update boundary
                    calculate_boundary_nodes_tele(grid, iip, jjt, kkr);
                }
            } // end ijk

            // mpi synchronization
            synchronize_all_sub(); ///////////////////////////////////////

        } // end loop i_level

#elif USE_AVX512 || USE_AVX

        // preload constants
        __mT v_DP_inv      = _mmT_set1_pT(1.0/dp);
        __mT v_DT_inv      = _mmT_set1_pT(1.0/dt);
        __mT v_DR_inv      = _mmT_set1_pT(1.0/dr);
        __mT v_DP_inv_half = _mmT_set1_pT(1.0/dp*0.5);
        __mT v_DT_inv_half = _mmT_set1_pT(1.0/dt*0.5);
        __mT v_DR_inv_half = _mmT_set1_pT(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1;
        __mT v_pp2;
        __mT v_pt1;
        __mT v_pt2;
        __mT v_pr1;
        __mT v_pr2;

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            __mT* v_iip    = (__mT*) vv_iip.at(iswp).at(i_level);
            __mT* v_jjt    = (__mT*) vv_jjt.at(iswp).at(i_level);
            __mT* v_kkr    = (__mT*) vv_kkr.at(iswp).at(i_level);

            __mT* v_fac_a  = (__mT*) vv_fac_a.at(iswp).at(i_level);
            __mT* v_fac_b  = (__mT*) vv_fac_b.at(iswp).at(i_level);
            __mT* v_fac_c  = (__mT*) vv_fac_c.at(iswp).at(i_level);
            __mT* v_fac_f  = (__mT*) vv_fac_f.at(iswp).at(i_level);
            __mT* v_fun    = (__mT*) vv_fun.at(iswp).at(i_level);
            __mT* v_change = (__mT*) vv_change.at(iswp).at(i_level);

            // alias for dumped index
            int* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            int* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            int* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            int* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            int* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            int* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            int* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);
            int* dump_ip2jk = vv_ip2j__k__.at(iswp).at(i_level);
            int* dump_im2jk = vv_im2j__k__.at(iswp).at(i_level);
            int* dump_ijp2k = vv_i__jp2k__.at(iswp).at(i_level);
            int* dump_ijm2k = vv_i__jm2k__.at(iswp).at(i_level);
            int* dump_ijkp2 = vv_i__j__kp2.at(iswp).at(i_level);
            int* dump_ijkm2 = vv_i__j__km2.at(iswp).at(i_level);

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {

                int i_vec = _i_vec * NSIMD;
                __mT v_c__    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijk[i_vec]);
                __mT v_p__    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ip1jk[i_vec]);
                __mT v_m__    = load_mem_gen_to_mTd(grid.T_loc,   &dump_im1jk[i_vec]);
                __mT v__p_    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijp1k[i_vec]);
                __mT v__m_    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijm1k[i_vec]);
                __mT v___p    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijkp1[i_vec]);
                __mT v___m    = load_mem_gen_to_mTd(grid.T_loc,   &dump_ijkm1[i_vec]);
                __mT v_pp____ = load_mem_gen_to_mTd(grid.T_loc, &dump_ip2jk[i_vec]);
                __mT v_mm____ = load_mem_gen_to_mTd(grid.T_loc, &dump_im2jk[i_vec]);
                __mT v___pp__ = load_mem_gen_to_mTd(grid.T_loc, &dump_ijp2k[i_vec]);
                __mT v___mm__ = load_mem_gen_to_mTd(grid.T_loc, &dump_ijm2k[i_vec]);
                __mT v_____pp = load_mem_gen_to_mTd(grid.T_loc, &dump_ijkp2[i_vec]);
                __mT v_____mm = load_mem_gen_to_mTd(grid.T_loc, &dump_ijkm2[i_vec]);

                // loop over all nodes in one level
                vect_stencil_3rd_pre_simd(v_iip[_i_vec], v_jjt[_i_vec], v_kkr[_i_vec], \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                // calculate updated value on c (teleseismic variant)
                vect_stencil_1st_3rd_apre_simd_tele(v_c__, v_fac_a[_i_vec], v_fac_b[_i_vec], v_fac_c[_i_vec], v_fac_f[_i_vec], \
                                               v_fun[_i_vec], v_change[_i_vec], \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // calculate the values on boundaries (teleseismic)
                calculate_boundary_nodes_tele_simd(v_iip[_i_vec], v_jjt[_i_vec], v_kkr[_i_vec], \
                                                   v_c__, \
                                                   v_p__,   v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                                   v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                                   v_change[_i_vec], \
                                                   loc_I, loc_J, loc_K);

                // store v_c__ to T_loc
                _mmT_store_pT(dump_c__, v_c__);

                for (int i = 0; i < NSIMD; i++) {
                    if(i_vec+i>=n_nodes) break;

                    grid.T_loc[dump_ijk[i_vec+i]] = dump_c__[i];
                }

            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub();

        } // end of i_level loop

#elif USE_ARM_SVE

        svbool_t pg;

        __mT v_DP_inv      = svdup_f64(1.0/dp);
        __mT v_DT_inv      = svdup_f64(1.0/dt);
        __mT v_DR_inv      = svdup_f64(1.0/dr);
        __mT v_DP_inv_half = svdup_f64(1.0/dp*0.5);
        __mT v_DT_inv_half = svdup_f64(1.0/dt*0.5);
        __mT v_DR_inv_half = svdup_f64(1.0/dr*0.5);

        // store stencil coefs
        __mT v_pp1;
        __mT v_pp2;
        __mT v_pt1;
        __mT v_pt2;
        __mT v_pr1;
        __mT v_pr2;

        __mT v_c__   ;
        __mT v_p__   ;
        __mT v_m__   ;
        __mT v__p_   ;
        __mT v__m_   ;
        __mT v___p   ;
        __mT v___m   ;
        __mT v_pp____;
        __mT v_mm____;
        __mT v___pp__;
        __mT v___mm__;
        __mT v_____pp;
        __mT v_____mm;

        __mT v_iip_   ;
        __mT v_jjt_   ;
        __mT v_kkr_   ;
        __mT v_fac_a_ ;
        __mT v_fac_b_ ;
        __mT v_fac_c_ ;
        __mT v_fac_f_ ;
        __mT v_fun_   ;
        __mT v_change_;

        int n_levels = ijk_for_this_subproc.size();
        for (int i_level = 0; i_level < n_levels; i_level++) {
            int n_nodes = ijk_for_this_subproc.at(i_level).size();

            int num_iter = n_nodes / NSIMD + (n_nodes % NSIMD == 0 ? 0 : 1);

            // make alias to preloaded data
            CUSTOMREAL* v_iip    = vv_iip.at(iswp).at(i_level);
            CUSTOMREAL* v_jjt    = vv_jjt.at(iswp).at(i_level);
            CUSTOMREAL* v_kkr    = vv_kkr.at(iswp).at(i_level);

            CUSTOMREAL* v_fac_a  = vv_fac_a.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_b  = vv_fac_b.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_c  = vv_fac_c.at(iswp).at(i_level);
            CUSTOMREAL* v_fac_f  = vv_fac_f.at(iswp).at(i_level);
            CUSTOMREAL* v_fun    = vv_fun.at(iswp).at(i_level);
            CUSTOMREAL* v_change = vv_change.at(iswp).at(i_level);

            // alias for dumped index
            uint64_t* dump_ijk   = vv_i__j__k__.at(iswp).at(i_level);
            uint64_t* dump_ip1jk = vv_ip1j__k__.at(iswp).at(i_level);
            uint64_t* dump_im1jk = vv_im1j__k__.at(iswp).at(i_level);
            uint64_t* dump_ijp1k = vv_i__jp1k__.at(iswp).at(i_level);
            uint64_t* dump_ijm1k = vv_i__jm1k__.at(iswp).at(i_level);
            uint64_t* dump_ijkp1 = vv_i__j__kp1.at(iswp).at(i_level);
            uint64_t* dump_ijkm1 = vv_i__j__km1.at(iswp).at(i_level);
            uint64_t* dump_ip2jk = vv_ip2j__k__.at(iswp).at(i_level);
            uint64_t* dump_im2jk = vv_im2j__k__.at(iswp).at(i_level);
            uint64_t* dump_ijp2k = vv_i__jp2k__.at(iswp).at(i_level);
            uint64_t* dump_ijm2k = vv_i__jm2k__.at(iswp).at(i_level);
            uint64_t* dump_ijkp2 = vv_i__j__kp2.at(iswp).at(i_level);
            uint64_t* dump_ijkm2 = vv_i__j__km2.at(iswp).at(i_level);

            // load data of all nodes in one level on temporal aligned array
            for (int _i_vec = 0; _i_vec < num_iter; _i_vec++) {
                int i_vec = _i_vec * NSIMD;

                pg = svwhilelt_b64(i_vec, n_nodes);

                v_c__    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijk[i_vec]);
                v_p__    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ip1jk[i_vec]);
                v_m__    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_im1jk[i_vec]);
                v__p_    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijp1k[i_vec]);
                v__m_    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijm1k[i_vec]);
                v___p    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkp1[i_vec]);
                v___m    = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkm1[i_vec]);
                v_pp____ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ip2jk[i_vec]);
                v_mm____ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_im2jk[i_vec]);
                v___pp__ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijp2k[i_vec]);
                v___mm__ = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijm2k[i_vec]);
                v_____pp = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkp2[i_vec]);
                v_____mm = load_mem_gen_to_mTd(pg, grid.T_loc,  &dump_ijkm2[i_vec]);

                // load v_iip, v_jjt, v_kkr
                v_iip_   = svld1_vnum_f64(pg, v_iip   , _i_vec);
                v_jjt_   = svld1_vnum_f64(pg, v_jjt   , _i_vec);
                v_kkr_   = svld1_vnum_f64(pg, v_kkr   , _i_vec);
                v_fac_a_ = svld1_vnum_f64(pg, v_fac_a , _i_vec);
                v_fac_b_ = svld1_vnum_f64(pg, v_fac_b , _i_vec);
                v_fac_c_ = svld1_vnum_f64(pg, v_fac_c , _i_vec);
                v_fac_f_ = svld1_vnum_f64(pg, v_fac_f , _i_vec);
                v_fun_   = svld1_vnum_f64(pg, v_fun   , _i_vec);
                v_change_= svld1_vnum_f64(pg, v_change, _i_vec);

                // loop over all nodes in one level
                vect_stencil_3rd_pre_simd(pg, v_iip_, v_jjt_, v_kkr_, \
                                          v_c__, \
                                          v_p__,    v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                          v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                          v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                          v_DP_inv, v_DT_inv, v_DR_inv, \
                                          v_DP_inv_half, v_DT_inv_half, v_DR_inv_half, \
                                          loc_I, loc_J, loc_K);

                // calculate updated value on c (teleseismic variant)
                vect_stencil_1st_3rd_apre_simd_tele(pg, v_c__, v_fac_a_, v_fac_b_, v_fac_c_, v_fac_f_, \
                                               v_fun_, v_change_, \
                                               v_pp1, v_pp2, v_pt1, v_pt2, v_pr1, v_pr2, \
                                               v_DP_inv, v_DT_inv, v_DR_inv);

                // calculate the values on boundaries (teleseismic)
                calculate_boundary_nodes_tele_simd(pg, v_iip_, v_jjt_, v_kkr_, \
                                                   v_c__, \
                                                   v_p__,   v_m__,    v__p_,    v__m_,    v___p,    v___m, \
                                                   v_pp____, v_mm____, v___pp__, v___mm__, v_____pp, v_____mm, \
                                                   v_change_, \
                                                   loc_I, loc_J, loc_K);

                // store v_c__ to T_loc
                svst1_scatter_u64index_f64(pg, grid.T_loc, svld1_u64(pg,&dump_ijk[i_vec]), v_c__);

            } // end of i_vec loop

            // mpi synchronization
            synchronize_all_sub();

        } // end of i_level loop

#endif // ifndef USE_SIMD

    } // end of if !use_gpu
    else { // if use_gpu

#if defined USE_CUDA

        //// copy tau to device
        //cuda_copy_tau_to_device(gpu_grid, grid.tau_loc);

        //// run iteration
        //cuda_run_iteration_forward_tele(gpu_grid, iswp);

        //// copy tau to host
        //cuda_copy_tau_to_host(gpu_grid, grid.tau_loc);

#else // !defiend USE_CUDA
        // exit code
        std::cout << "Error: USE_CUDA is not defined" << std::endl;
        exit(1);
#endif

    } // end of if use_gpu


}


Iterator_level_1st_order_upwind_tele::Iterator_level_1st_order_upwind_tele(InputParams& IP, Grid& grid, Source& src, IO_utils& io, const std::string& src_name, bool first_init, bool is_teleseismic_in, bool is_second_run_in) \
                         : Iterator_level_tele(IP, grid, src, io, src_name, first_init, is_teleseismic_in, is_second_run_in) {
    // initialization is done in the base class
}

void Iterator_level_1st_order_upwind_tele::do_sweep(int iswp, Grid& grid, InputParams& IP){

    // set sweep direction
    set_sweep_direction(iswp);

    int iip, jjt, kkr;
    int n_levels = ijk_for_this_subproc.size();

    for (int i_level = 0; i_level < n_levels; i_level++) {
        size_t n_nodes = ijk_for_this_subproc[i_level].size();

        for (size_t i_node = 0; i_node < n_nodes; i_node++) {

            V2I(ijk_for_this_subproc[i_level][i_node], iip, jjt, kkr);

            if (r_dirc < 0) kkr = nr-1-kkr;
            if (t_dirc < 0) jjt = nt-1-jjt;
            if (p_dirc < 0) iip = np-1-iip;

            //
            // calculate stencils
            //
            // if (iip != 0 && iip != np-1 && jjt != 0 && jjt != nt-1 && kkr != 0) {   // top layer is not fixed, otherwise, the top layer will be 2000
            //     calculate_stencil_1st_order_upwind_tele(grid, iip, jjt, kkr);       // no need to consider the boundary for upwind scheme
            // }

            calculate_stencil_1st_order_upwind_tele(grid, iip, jjt, kkr);
        } // end ijk

        // mpi synchronization
        synchronize_all_sub();

    } // end loop i_level
}
