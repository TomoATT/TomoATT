#ifndef VECTORIZED_SWEEP_H
#define VECTORIZED_SWEEP_H


#include<vector>
#include "config.h"

#ifdef USE_SIMD // closed at the end of this file
#include "simd_conf.h"

#if USE_AVX || USE_AVX512
__mTd COEF      = _mmT_set1_pd(1.0);
__mTd v_1       = _mmT_set1_pd(1.0);
__mTd v_0       = _mmT_set1_pd(0.0);
__mTd v_half    = _mmT_set1_pd(0.5);
__mTd v_2       = _mmT_set1_pd(2.0);
__mTd v_m2      = _mmT_set1_pd(-2.0);
__mTd v_4       = _mmT_set1_pd(4.0);
__mTd v_m3      = _mmT_set1_pd(-3.0);
__mTd coe_max   = _mmT_set1_pd(1e19);
__mTd v_eps     = _mmT_set1_pd(1e-12);
__mTd COEF_TELE = _mmT_set1_pd(1.05);


// square of __mTd
inline __mTd my_square_v(__mTd const& a){
    return _mmT_mul_pd(a, a);
}

// pp = (a - b) * Dinv
inline __mTd calc_1d_stencil(__mTd const& a, __mTd const& b, __mTd const& Dinv){
    return _mmT_mul_pd(_mmT_sub_pd(a,b),Dinv);
}

/*
    ww  = _1_CR/(_1_CR+_2_CR*my_square_v((eps + my_square_v(      a \
                                                       -_2_CR*b \
                                                       +      c) ) \

                                     / (eps + my_square_v(      d \
                                                       -_2_CR*a \
                                                       +      b) )) );

    pp  = sign  * (_1_CR - ww) * (         b \                  <-- sign = +1 or -1
                                 -         d) * 0.5 * Dinv \
                         + ww  * ( -3.0 * a \
                                 +  4.0 * b \
                                 -  1.0 * c ) * 0.5 * Dinv );
*/
inline __mTd calc_3d_stencil(__mTd const& a, __mTd const& b, __mTd const&c, __mTd const& d, __mTd const& Dinv_half, int const& sign){

#ifdef __FMA__
    // v_eps + square(a - 2.0*b + c)
    __mTd tmp1 = _mmT_add_pd(v_eps,my_square_v(_mmT_add_pd(a,_mmT_fmadd_pd(v_m2,b,c))));
    // v_eps + square(d - 2.0*a + b)
    __mTd tmp2 = _mmT_add_pd(v_eps,my_square_v(_mmT_add_pd(d,_mmT_fmadd_pd(v_m2,a,b))));
    // ww = 1.0/(1.0 + 2.0 * square(tmp1/tmp2))
    __mTd ww = _mmT_div_pd(v_1,_mmT_fmadd_pd(v_2,my_square_v(_mmT_div_pd(tmp1,tmp2)),v_1));
    // pp = sign* ((1.0 - ww) * (b - d) / 2.0 / D
    //                  + ww  * (-3.0* a + 4.0 * b - c) * Dinv_half)
    /*
    */
    return _mmT_mul_pd(_mmT_set1_pd(sign), \
                _mmT_add_pd(\
                            _mmT_mul_pd(_mmT_sub_pd(v_1,ww),_mmT_mul_pd(_mmT_sub_pd(b,d),Dinv_half)),\
                            _mmT_mul_pd(ww,_mmT_mul_pd(_mmT_sub_pd(_mmT_fmadd_pd(v_4,b,_mmT_mul_pd(v_m3,a)),c),Dinv_half))\
                )\
           );
#else
    __mTd tmp1 = _mmT_add_pd(v_eps,my_square_v(_mmT_add_pd(a,_mmT_add_pd(_mmT_mul_pd(v_m2,b),c))));
    __mTd tmp2 = _mmT_add_pd(v_eps,my_square_v(_mmT_add_pd(d,_mmT_add_pd(_mmT_mul_pd(v_m2,a),b))));
    __mTd ww = _mmT_div_pd(v_1,_mmT_add_pd(v_1,_mmT_mul_pd(v_2,my_square_v(_mmT_div_pd(tmp1,tmp2)))));
    return _mmT_mul_pd(_mmT_set1_pd(sign), \
                _mmT_add_pd(\
                            _mmT_mul_pd(_mmT_sub_pd(v_1,ww),_mmT_mul_pd(_mmT_sub_pd(b,d),Dinv_half)),\
                            _mmT_mul_pd(ww,_mmT_mul_pd(_mmT_sub_pd(_mmT_add_pd(_mmT_mul_pd(v_4,b),_mmT_mul_pd(v_m3,a)),c),Dinv_half))\
                )\
           );
#endif

}

#elif USE_ARM_SVE

inline __mTd my_square_v(svbool_t const& pg, __mTd const& a){
    return svmul_f64_z(pg, a, a);
}

inline __mTd calc_1d_stencil(svbool_t const& pg, __mTd const& a, __mTd const& b, __mTd const& Dinv){
    return svmul_f64_z(pg, svsub_f64_z(pg, a, b), Dinv);
}

inline __mTd calc_3d_stencil(svbool_t const& pg, __mTd const& a, __mTd const& b, __mTd const&c, __mTd const& d, __mTd const& Dinv_half, int const& sign){

    __mTd v_1     = svdup_f64(1.0);
    __mTd v_2     = svdup_f64(2.0);
    __mTd v_m2    = svdup_f64(-2.0);
    __mTd v_4     = svdup_f64(4.0);
    __mTd v_m3    = svdup_f64(-3.0);
    __mTd v_eps   = svdup_f64(1e-12);

    // v_eps + square(a - 2.0*b + c)
    //__mTd tmp1 = svadd_f64_z(pg, v_eps, my_square_v(pg, svadd_f64_z(pg, a, svadd_f64_z(pg, svmul_f64_z(pg, v_m2, b), c))));
    __mTd tmp1 = svadd_f64_z(pg, v_eps, my_square_v(pg, svadd_f64_z(pg, a, svmad_f64_z(pg, v_m2, b, c))));
    // v_eps + square(d - 2.0*a + b)
    //__mTd tmp2 = svadd_f64_z(pg, v_eps, my_square_v(pg, svadd_f64_z(pg, d, svadd_f64_z(pg, svmul_f64_z(pg, v_m2, a), b))));
    __mTd tmp2 = svadd_f64_z(pg, v_eps, my_square_v(pg, svadd_f64_z(pg, d, svmad_f64_z(pg, v_m2, a, b))));
    // ww = 1.0/(1.0 + 2.0 * square(tmp1/tmp2))
    //__mTd ww = svdiv_f64_z(pg, v_1, svadd_f64_z(pg, v_1, svmul_f64_z(pg, v_2, my_square_v(pg, svdiv_f64_z(pg, tmp1, tmp2)))));
    __mTd ww = svdiv_f64_z(pg, v_1, svmad_f64_z(pg, v_2, my_square_v(pg, svdiv_f64_z(pg, tmp1, tmp2)), v_1));
    // pp = sign* ((1.0 - ww) * (b - d) / 2.0 / D
    //                  + ww  * (-3.0* a + 4.0 * b - c) * Dinv_half)
    /*
    return svmul_f64_m(pg, svdup_f64(sign), \
            svadd_f64_z(pg, \
                        svmul_f64_z(pg, svsub_f64_z(pg, v_1, ww), svmul_f64_z(pg, svsub_f64_z(pg, b, d), Dinv_half)),\
                        svmul_f64_z(pg, ww, svmul_f64_z(pg, svsub_f64_z(pg, svadd_f64_z(pg, svmul_f64_z(pg, v_4, b), svmul_f64_z(pg, v_m3, a)), c), Dinv_half))\
            )\
       );
    */
    return svmul_f64_m(pg, svdup_f64(sign), \
                svadd_f64_z(pg, \
                            svmul_f64_z(pg, svsub_f64_z(pg, v_1, ww), svmul_f64_z(pg, svsub_f64_z(pg, b, d), Dinv_half)),\
                            svmul_f64_z(pg, ww, svmul_f64_z(pg, svsub_f64_z(pg, svmad_f64_z(pg, v_4, b, svmul_f64_z(pg, v_m3, a)), c), Dinv_half))\
                )\
           );
}


#endif



inline void vect_stencil_1st_pre_simd(
#if USE_ARM_SVE
                                      svbool_t const& pg,
#endif
                                      __mTd const& v_iip, __mTd const& v_jjt, __mTd const& v_kkr,
                                      __mTd const& v_c__,
                                      __mTd const& v_p__, __mTd const& v_m__, __mTd const& v__p_, __mTd const& v__m_, __mTd const& v___p, __mTd const& v___m,
                                      __mTd& v_pp1, __mTd& v_pp2, __mTd& v_pt1, __mTd& v_pt2, __mTd& v_pr1, __mTd& v_pr2,
                                      __mTd const& v_DP_inv, __mTd const& v_DT_inv, __mTd const& v_DR_inv,
                                      __mTd const& v_DP_inv_half, __mTd const& v_DT_inv_half, __mTd const& v_DR_inv_half,
                                      int const& NP, int const& NT, int const& NR){

#if USE_AVX512 || USE_AVX

    v_pp1 = calc_1d_stencil(v_c__, v_m__, v_DP_inv);
    v_pp2 = calc_1d_stencil(v_p__, v_c__, v_DP_inv);
    v_pt1 = calc_1d_stencil(v_c__, v__m_, v_DT_inv);
    v_pt2 = calc_1d_stencil(v__p_, v_c__, v_DT_inv);
    v_pr1 = calc_1d_stencil(v_c__, v___m, v_DR_inv);
    v_pr2 = calc_1d_stencil(v___p, v_c__, v_DR_inv);

#elif USE_ARM_SVE

    v_pp1 = calc_1d_stencil(pg, v_c__, v_m__, v_DP_inv);
    v_pp2 = calc_1d_stencil(pg, v_p__, v_c__, v_DP_inv);
    v_pt1 = calc_1d_stencil(pg, v_c__, v__m_, v_DT_inv);
    v_pt2 = calc_1d_stencil(pg, v__p_, v_c__, v_DT_inv);
    v_pr1 = calc_1d_stencil(pg, v_c__, v___m, v_DR_inv);
    v_pr2 = calc_1d_stencil(pg, v___p, v_c__, v_DR_inv);

#endif

}


inline void vect_stencil_3rd_pre_simd(
#if USE_ARM_SVE
                                      svbool_t const& pg,
#endif
                                      __mTd const& v_iip, __mTd const& v_jjt, __mTd const& v_kkr,
                                      __mTd const& v_c__,
                                      __mTd const& v_p__,     __mTd const& v_m__,     __mTd const& v__p_,    __mTd const& v__m_,    __mTd const& v___p,    __mTd const& v___m,
                                      __mTd const& v_pp____,  __mTd const& v_mm____,  __mTd const& v___pp__, __mTd const& v___mm__, __mTd const& v_____pp, __mTd const& v_____mm,
                                      __mTd& v_pp1,     __mTd& v_pp2,     __mTd& v_pt1,    __mTd& v_pt2,    __mTd& v_pr1,    __mTd& v_pr2,
                                      __mTd const& v_DP_inv,  __mTd const& v_DT_inv,  __mTd const& v_DR_inv,
                                      __mTd const& v_DP_inv_half, __mTd const& v_DT_inv_half, __mTd const& v_DR_inv_half,
                                      int const& NP, int const& NT, int const& NR){

    const int PLUS  = 1;
    const int MINUS = -1;

#if USE_AVX512

    __mmask8 mask_i_eq_1         = _mm512_cmp_pd_mask(v_iip, v_1,_CMP_EQ_OQ);    // if iip == 1
    __mmask8 mask_j_eq_1         = _mm512_cmp_pd_mask(v_jjt, v_1,_CMP_EQ_OQ);    // if jjt == 1
    __mmask8 mask_k_eq_1         = _mm512_cmp_pd_mask(v_kkr, v_1,_CMP_EQ_OQ);    // if kkr == 1
    __mmask8 mask_i_eq_N_minus_2 = _mm512_cmp_pd_mask(v_iip, _mm512_set1_pd(NP-2),_CMP_EQ_OQ); // if iip == N-2
    __mmask8 mask_j_eq_N_minus_2 = _mm512_cmp_pd_mask(v_jjt, _mm512_set1_pd(NT-2),_CMP_EQ_OQ); // if jjt == N-2
    __mmask8 mask_k_eq_N_minus_2 = _mm512_cmp_pd_mask(v_kkr, _mm512_set1_pd(NR-2),_CMP_EQ_OQ); // if kkr == N-2

    // 1 < iip < N-2
    __mmask8 mask_i_else = _kand_mask8(
                               _mm512_cmp_pd_mask(v_iip, v_1                , _CMP_GT_OQ),
                               _mm512_cmp_pd_mask(_mm512_set1_pd(NP-2),  v_iip, _CMP_GT_OQ)
                         );
    // 1 < jjt < N-2
    __mmask8 mask_j_else = _kand_mask8(
                               _mm512_cmp_pd_mask(v_jjt, v_1                , _CMP_GT_OQ),
                               _mm512_cmp_pd_mask(_mm512_set1_pd(NT-2),  v_jjt, _CMP_GT_OQ)
                         );
    // 1 < kkr < N-2
    __mmask8 mask_k_else = _kand_mask8(
                               _mm512_cmp_pd_mask(v_kkr, v_1                , _CMP_GT_OQ),
                               _mm512_cmp_pd_mask(_mm512_set1_pd(NR-2),  v_kkr, _CMP_GT_OQ)
                         );

    // if _i_eq_1 == true
    v_pp1 = _mm512_mask_blend_pd(mask_i_eq_1, calc_1d_stencil(v_c__, v_m__, v_DP_inv)                            , v_pp1 );
    v_pp2 = _mm512_mask_blend_pd(mask_i_eq_1, calc_3d_stencil(v_c__, v_p__, v_pp____, v_m__, v_DP_inv_half, PLUS), v_pp2 );
    // if_i_eq_N_minus_2 == true
    v_pp1 = _mm512_mask_blend_pd(mask_i_eq_N_minus_2, calc_3d_stencil(v_c__, v_m__, v_mm____, v_p__, v_DP_inv_half, MINUS), v_pp1);
    v_pp2 = _mm512_mask_blend_pd(mask_i_eq_N_minus_2, calc_1d_stencil(v_p__, v_c__, v_DP_inv)                             , v_pp2);
    // else
    v_pp1 = _mm512_mask_blend_pd(mask_i_else, calc_3d_stencil(v_c__, v_m__, v_mm____, v_p__, v_DP_inv_half, MINUS), v_pp1);
    v_pp2 = _mm512_mask_blend_pd(mask_i_else, calc_3d_stencil(v_c__, v_p__, v_pp____, v_m__, v_DP_inv_half, PLUS) , v_pp2);

    // if _j_eq_1 == true
    v_pt1 = _mm512_mask_blend_pd(mask_j_eq_1, calc_1d_stencil(v_c__, v__m_, v_DT_inv)                            , v_pt1);
    v_pt2 = _mm512_mask_blend_pd(mask_j_eq_1, calc_3d_stencil(v_c__, v__p_, v___pp__, v__m_, v_DT_inv_half, PLUS), v_pt2);
    // if _j_eq_N_minus_2 == true
    v_pt1 = _mm512_mask_blend_pd(mask_j_eq_N_minus_2, calc_3d_stencil(v_c__, v__m_, v___mm__, v__p_, v_DT_inv_half, MINUS), v_pt1);
    v_pt2 = _mm512_mask_blend_pd(mask_j_eq_N_minus_2, calc_1d_stencil(v__p_, v_c__, v_DT_inv)                             , v_pt2);
    // else
    v_pt1 = _mm512_mask_blend_pd(mask_j_else, calc_3d_stencil(v_c__, v__m_, v___mm__, v__p_, v_DT_inv_half, MINUS), v_pt1);
    v_pt2 = _mm512_mask_blend_pd(mask_j_else, calc_3d_stencil(v_c__, v__p_, v___pp__, v__m_, v_DT_inv_half, PLUS) , v_pt2);

    // if _k_eq_1 == true
    v_pr1 = _mm512_mask_blend_pd(mask_k_eq_1, calc_1d_stencil(v_c__, v___m, v_DR_inv)                            , v_pr1 );
    v_pr2 = _mm512_mask_blend_pd(mask_k_eq_1, calc_3d_stencil(v_c__, v___p, v_____pp, v___m, v_DR_inv_half, PLUS), v_pr2 );
    // if _k_eq_N_minus_2 == true
    v_pr1 = _mm512_mask_blend_pd(mask_k_eq_N_minus_2, calc_3d_stencil(v_c__, v___m, v_____mm, v___p, v_DR_inv_half, MINUS), v_pr1);
    v_pr2 = _mm512_mask_blend_pd(mask_k_eq_N_minus_2, calc_1d_stencil(v___p, v_c__, v_DR_inv)                             , v_pr2);
    // else
    v_pr1 = _mm512_mask_blend_pd(mask_k_else, calc_3d_stencil(v_c__, v___m, v_____mm, v___p, v_DR_inv_half, MINUS), v_pr1);
    v_pr2 = _mm512_mask_blend_pd(mask_k_else, calc_3d_stencil(v_c__, v___p, v_____pp, v___m, v_DR_inv_half, PLUS) , v_pr2);

#elif USE_AVX

    __m256d mask_i_eq_1         = _mm256_cmp_pd(v_iip, v_1,_CMP_EQ_OQ);    // if iip == 1
    __m256d mask_j_eq_1         = _mm256_cmp_pd(v_jjt, v_1,_CMP_EQ_OQ);    // if jjt == 1
    __m256d mask_k_eq_1         = _mm256_cmp_pd(v_kkr, v_1,_CMP_EQ_OQ);    // if kkr == 1
    __m256d mask_i_eq_N_minus_2 = _mm256_cmp_pd(v_iip, _mm256_set1_pd(NP-2),_CMP_EQ_OQ); // if iip == N-2
    __m256d mask_j_eq_N_minus_2 = _mm256_cmp_pd(v_jjt, _mm256_set1_pd(NT-2),_CMP_EQ_OQ); // if jjt == N-2
    __m256d mask_k_eq_N_minus_2 = _mm256_cmp_pd(v_kkr, _mm256_set1_pd(NR-2),_CMP_EQ_OQ); // if kkr == N-2

    // 1 < iip < N-2
    __m256d mask_i_else = _mm256_and_pd(
                                        _mm256_cmp_pd(v_iip, v_1                , _CMP_GT_OQ),
                                        _mm256_cmp_pd(_mm256_set1_pd(NP-2),v_iip, _CMP_GT_OQ)
                         );
    // 1 < jjt < N-2
    __m256d mask_j_else = _mm256_and_pd(
                                        _mm256_cmp_pd(v_jjt, v_1                , _CMP_GT_OQ),
                                        _mm256_cmp_pd(_mm256_set1_pd(NT-2),v_jjt, _CMP_GT_OQ)
                         );
    // 1 < kkr < N-2
    __m256d mask_k_else = _mm256_and_pd(
                                        _mm256_cmp_pd(v_kkr, v_1                , _CMP_GT_OQ),
                                        _mm256_cmp_pd(_mm256_set1_pd(NR-2),v_kkr, _CMP_GT_OQ)
                         );

    // if _i_eq_1 == true
    v_pp1 = _mm256_blendv_pd(v_pp1, calc_1d_stencil(v_c__, v_m__, v_DP_inv),                             mask_i_eq_1);
    v_pp2 = _mm256_blendv_pd(v_pp2, calc_3d_stencil(v_c__, v_p__, v_pp____, v_m__, v_DP_inv_half, PLUS), mask_i_eq_1);
    // if_i_eq_N_minus_2 == true
    v_pp1 = _mm256_blendv_pd(v_pp1, calc_3d_stencil(v_c__, v_m__, v_mm____, v_p__, v_DP_inv_half, MINUS), mask_i_eq_N_minus_2);
    v_pp2 = _mm256_blendv_pd(v_pp2, calc_1d_stencil(v_p__, v_c__, v_DP_inv),                              mask_i_eq_N_minus_2);
    // else
    v_pp1 = _mm256_blendv_pd(v_pp1, calc_3d_stencil(v_c__, v_m__, v_mm____, v_p__, v_DP_inv_half, MINUS), mask_i_else);
    v_pp2 = _mm256_blendv_pd(v_pp2, calc_3d_stencil(v_c__, v_p__, v_pp____, v_m__, v_DP_inv_half, PLUS),  mask_i_else);

    // if _j_eq_1 == true
    v_pt1 = _mm256_blendv_pd(v_pt1, calc_1d_stencil(v_c__, v__m_, v_DT_inv),                             mask_j_eq_1);
    v_pt2 = _mm256_blendv_pd(v_pt2, calc_3d_stencil(v_c__, v__p_, v___pp__, v__m_, v_DT_inv_half, PLUS), mask_j_eq_1);
    // if _j_eq_N_minus_2 == true
    v_pt1 = _mm256_blendv_pd(v_pt1, calc_3d_stencil(v_c__, v__m_, v___mm__, v__p_, v_DT_inv_half, MINUS), mask_j_eq_N_minus_2);
    v_pt2 = _mm256_blendv_pd(v_pt2, calc_1d_stencil(v__p_, v_c__, v_DT_inv),                              mask_j_eq_N_minus_2);
    // else
    v_pt1 = _mm256_blendv_pd(v_pt1, calc_3d_stencil(v_c__, v__m_, v___mm__, v__p_, v_DT_inv_half, MINUS), mask_j_else);
    v_pt2 = _mm256_blendv_pd(v_pt2, calc_3d_stencil(v_c__, v__p_, v___pp__, v__m_, v_DT_inv_half, PLUS),  mask_j_else);

    // if _k_eq_1 == true
    v_pr1 = _mm256_blendv_pd(v_pr1, calc_1d_stencil(v_c__, v___m, v_DR_inv),                             mask_k_eq_1);
    v_pr2 = _mm256_blendv_pd(v_pr2, calc_3d_stencil(v_c__, v___p, v_____pp, v___m, v_DR_inv_half, PLUS), mask_k_eq_1);
    // if _k_eq_N_minus_2 == true
    v_pr1 = _mm256_blendv_pd(v_pr1, calc_3d_stencil(v_c__, v___m, v_____mm, v___p, v_DR_inv_half, MINUS), mask_k_eq_N_minus_2);
    v_pr2 = _mm256_blendv_pd(v_pr2, calc_1d_stencil(v___p, v_c__, v_DR_inv),                              mask_k_eq_N_minus_2);
    // else
    v_pr1 = _mm256_blendv_pd(v_pr1, calc_3d_stencil(v_c__, v___m, v_____mm, v___p, v_DR_inv_half, MINUS), mask_k_else);
    v_pr2 = _mm256_blendv_pd(v_pr2, calc_3d_stencil(v_c__, v___p, v_____pp, v___m, v_DR_inv_half, PLUS),  mask_k_else);

#elif USE_ARM_SVE

    svfloat64_t v_1 = svdup_f64(1.0);

    svfloat64_t v_NP_minus_2 = svdup_f64(NP-2);
    svfloat64_t v_NT_minus_2 = svdup_f64(NT-2);
    svfloat64_t v_NR_minus_2 = svdup_f64(NR-2);

    svbool_t mask_i_eq_1         = svcmpeq_f64(pg, v_iip, v_1);    // if iip == 1
    svbool_t mask_j_eq_1         = svcmpeq_f64(pg, v_jjt, v_1);    // if jjt == 1
    svbool_t mask_k_eq_1         = svcmpeq_f64(pg, v_kkr, v_1);    // if kkr == 1
    svbool_t mask_i_eq_N_minus_2 = svcmpeq_f64(pg, v_iip, v_NP_minus_2); // if iip == N-2
    svbool_t mask_j_eq_N_minus_2 = svcmpeq_f64(pg, v_jjt, v_NT_minus_2); // if jjt == N-2
    svbool_t mask_k_eq_N_minus_2 = svcmpeq_f64(pg, v_kkr, v_NR_minus_2); // if kkr == N-2

    // 1 < iip < N-2
    svbool_t mask_i_else = svand_b_z(pg,
                               svcmpgt_f64(pg, v_iip, v_1),
                               svcmplt_f64(pg, v_iip, v_NP_minus_2)
                         );
    // 1 < jjt < N-2
    svbool_t mask_j_else = svand_b_z(pg,
                               svcmpgt_f64(pg, v_jjt, v_1),
                               svcmplt_f64(pg, v_jjt, v_NT_minus_2)
                         );
    // 1 < kkr < N-2
    svbool_t mask_k_else = svand_b_z(pg,
                               svcmpgt_f64(pg, v_kkr, v_1),
                               svcmplt_f64(pg, v_kkr, v_NR_minus_2)
                         );

    // if _i_eq_1 == true
    v_pp1 = svsel_f64(mask_i_eq_1, calc_1d_stencil(mask_i_eq_1, v_c__, v_m__, v_DP_inv)                            , v_pp1);
    v_pp2 = svsel_f64(mask_i_eq_1, calc_3d_stencil(mask_i_eq_1, v_c__, v_p__, v_pp____, v_m__, v_DP_inv_half, PLUS), v_pp2);
    // if_i_eq_N_minus_2 == true
    v_pp1 = svsel_f64(mask_i_eq_N_minus_2, calc_3d_stencil(mask_i_eq_N_minus_2, v_c__, v_m__, v_mm____, v_p__, v_DP_inv_half, MINUS), v_pp1);
    v_pp2 = svsel_f64(mask_i_eq_N_minus_2, calc_1d_stencil(mask_i_eq_N_minus_2, v_p__, v_c__, v_DP_inv)                             , v_pp2);
    // else
    v_pp1 = svsel_f64(mask_i_else, calc_3d_stencil(mask_i_else, v_c__, v_m__, v_mm____, v_p__, v_DP_inv_half, MINUS), v_pp1);
    v_pp2 = svsel_f64(mask_i_else, calc_3d_stencil(mask_i_else, v_c__, v_p__, v_pp____, v_m__, v_DP_inv_half, PLUS) , v_pp2);

    // if _j_eq_1 == true
    v_pt1 = svsel_f64(mask_j_eq_1, calc_1d_stencil(mask_j_eq_1, v_c__, v__m_, v_DT_inv)                            , v_pt1);
    v_pt2 = svsel_f64(mask_j_eq_1, calc_3d_stencil(mask_j_eq_1, v_c__, v__p_, v___pp__, v__m_, v_DT_inv_half, PLUS), v_pt2);
    // if _j_eq_N_minus_2 == true
    v_pt1 = svsel_f64(mask_j_eq_N_minus_2, calc_3d_stencil(mask_j_eq_N_minus_2, v_c__, v__m_, v___mm__, v__p_, v_DT_inv_half, MINUS), v_pt1);
    v_pt2 = svsel_f64(mask_j_eq_N_minus_2, calc_1d_stencil(mask_j_eq_N_minus_2, v__p_, v_c__, v_DT_inv)                             , v_pt2);
    // else
    v_pt1 = svsel_f64(mask_j_else, calc_3d_stencil(mask_j_else, v_c__, v__m_, v___mm__, v__p_, v_DT_inv_half, MINUS), v_pt1);
    v_pt2 = svsel_f64(mask_j_else, calc_3d_stencil(mask_j_else, v_c__, v__p_, v___pp__, v__m_, v_DT_inv_half, PLUS) , v_pt2);

    // if _k_eq_1 == true
    v_pr1 = svsel_f64(mask_k_eq_1, calc_1d_stencil(mask_k_eq_1, v_c__, v___m, v_DR_inv)                            , v_pr1);
    v_pr2 = svsel_f64(mask_k_eq_1, calc_3d_stencil(mask_k_eq_1, v_c__, v___p, v_____pp, v___m, v_DR_inv_half, PLUS), v_pr2);
    // if _k_eq_N_minus_2 == true
    v_pr1 = svsel_f64(mask_k_eq_N_minus_2, calc_3d_stencil(mask_k_eq_N_minus_2, v_c__, v___m, v_____mm, v___p, v_DR_inv_half, MINUS), v_pr1);
    v_pr2 = svsel_f64(mask_k_eq_N_minus_2, calc_1d_stencil(mask_k_eq_N_minus_2, v___p, v_c__, v_DR_inv)                             , v_pr2);
    // else
    v_pr1 = svsel_f64(mask_k_else, calc_3d_stencil(mask_k_else, v_c__, v___m, v_____mm, v___p, v_DR_inv_half, MINUS), v_pr1);
    v_pr2 = svsel_f64(mask_k_else, calc_3d_stencil(mask_k_else, v_c__, v___p, v_____pp, v___m, v_DR_inv_half, PLUS) , v_pr2);


#endif

}


inline void vect_stencil_1st_3rd_apre_simd(
#if USE_ARM_SVE
                                           svbool_t const& pg,
#endif
                                           __mTd& v_tau,        __mTd const& v_fac_a,__mTd const& v_fac_b, __mTd const& v_fac_c, __mTd const& v_fac_f,
                                           __mTd const& v_T0v,  __mTd const& v_T0p,  __mTd const& v_T0t,   __mTd const& v_T0r,   __mTd const& v_fun, __mTd const& v_change,
                                           __mTd const& v_pp1,  __mTd const& v_pp2,  __mTd const& v_pt1,   __mTd const& v_pt2,   __mTd const& v_pr1, __mTd const& v_pr2,
                                           __mTd const& DP_inv, __mTd const& DT_inv, __mTd const& DR_inv){

#if USE_AVX512 || USE_AVX

    // sigr = COEF * sqrt(v_fac_a)*v_T0v;
    __mTd sigr = _mmT_mul_pd(_mmT_mul_pd(COEF,_mmT_sqrt_pd(v_fac_a)),v_T0v);
    // sigt = COEF * sqrt(v_fac_b)*v_T0v;
    __mTd sigt = _mmT_mul_pd(_mmT_mul_pd(COEF,_mmT_sqrt_pd(v_fac_b)),v_T0v);
    // sigp = COEF * sqrt(v_fac_c)*v_T0v;
    __mTd sigp = _mmT_mul_pd(_mmT_mul_pd(COEF,_mmT_sqrt_pd(v_fac_c)),v_T0v);

    // coe = 1.0 / (sigr/D + sigt/D + sigz/D);
    __mTd coe = _mmT_div_pd(v_1,_mmT_add_pd(_mmT_add_pd(_mmT_mul_pd(sigr,DR_inv),_mmT_mul_pd(sigt,DT_inv)),_mmT_mul_pd(sigp,DP_inv)));
    // coe becomes inf as sig* goes to 0
    // if coe > 1e19, set coe = 1e19
    coe = _mmT_min_pd(coe,coe_max);

    // Htau  = v_fac_a * square(v_T0r * v_tau + v_T0v * (v_pr1 + v_pr2)*0.5);
    __mTd Htau = _mmT_mul_pd(v_fac_a,my_square_v(_mmT_add_pd(_mmT_mul_pd(v_T0r,v_tau),_mmT_mul_pd(v_T0v,_mmT_mul_pd(_mmT_add_pd(v_pr1,v_pr2),v_half)))));

    // Htau += v_fac_b * square(v_T0t * v_tau + v_T0v * (v_pt1 + v_pt2)*0.5))
    Htau = _mmT_add_pd(Htau,_mmT_mul_pd(v_fac_b,my_square_v(_mmT_add_pd(_mmT_mul_pd(v_T0t,v_tau),_mmT_mul_pd(v_T0v,_mmT_mul_pd(_mmT_add_pd(v_pt1,v_pt2), v_half))))));
    // Htau += v_fac_c * square(v_T0p * v_tau + v_T0v * (v_pp1 + v_pp2)*0.5))
    Htau = _mmT_add_pd(Htau,_mmT_mul_pd(v_fac_c,my_square_v(_mmT_add_pd(_mmT_mul_pd(v_T0p,v_tau),_mmT_mul_pd(v_T0v,_mmT_mul_pd(_mmT_add_pd(v_pp1,v_pp2), v_half))))));

    // tmp1 = ( v_T0t * v_tau + v_T0v * (v_pt1 + v_pt2)*0.5)
    __mTd tmp1 = _mmT_add_pd(_mmT_mul_pd(v_T0t,v_tau),_mmT_mul_pd(v_T0v,_mmT_mul_pd(_mmT_add_pd(v_pt1,v_pt2), v_half)));
    // tmp2 = ( v_T0p * v_tau + v_T0v * (v_pp1 + v_pp2)*0.5))
    __mTd tmp2 = _mmT_add_pd(_mmT_mul_pd(v_T0p,v_tau),_mmT_mul_pd(v_T0v,_mmT_mul_pd(_mmT_add_pd(v_pp1,v_pp2), v_half)));
    // tmp3 = -2.0 * v_fac_f * tmp1 * tmp2;
    __mTd tmp3 = _mmT_mul_pd(v_m2,_mmT_mul_pd(v_fac_f,_mmT_mul_pd(tmp1,tmp2)));

    // Htau = sqrt(Htau + tmp3); // # becamse nan as Htau - tmp3 goes to 0
    Htau = _mmT_sqrt_pd(_mmT_add_pd(Htau,tmp3));

    // tmp = (sigr*(v_pr2 - v_pr1) + sigt*(v_pt2 - v_pt1) + sigz*(v_pp2 - v_pp1))*0.5
    __mTd tmp = _mmT_mul_pd(v_half,_mmT_add_pd(_mmT_add_pd(_mmT_mul_pd(sigr,_mmT_sub_pd(v_pr2,v_pr1)),_mmT_mul_pd(sigt,_mmT_sub_pd(v_pt2,v_pt1))),_mmT_mul_pd(sigp,_mmT_sub_pd(v_pp2,v_pp1))));

    // v_tau += coe * ((v_fun - Htau) + tmp) if mask is true
    v_tau = _mmT_add_pd(v_tau,_mmT_mul_pd(coe,_mmT_add_pd(_mmT_sub_pd(v_fun,Htau),tmp)));

#if USE_AVX512
    // mask if v_change != 1.0
    __mmask8 mask = _mm512_cmp_pd_mask(v_change,v_1,_CMP_NEQ_OQ);
    // set 1 if mask is true
    v_tau = _mm512_mask_blend_pd(mask,v_tau,v_1);
#elif USE_AVX
    __m256d mask = _mm256_cmp_pd(v_change, v_1,_CMP_NEQ_OQ);
    v_tau        = _mm256_blendv_pd(v_tau, v_1,mask);
#endif

#elif USE_ARM_SVE

    __mTd COEF    = svdup_f64(1.0);
    __mTd v_1     = svdup_f64(1.0);
    __mTd v_half  = svdup_f64(0.5);
    __mTd v_m2    = svdup_f64(-2.0);
    __mTd coe_max = svdup_f64(1e19);

    // sigr = COEF * sqrt(v_fac_a)*v_T0v;
    __mTd sigr = svmul_f64_z(pg, svmul_f64_z(pg,COEF,svsqrt_f64_z(pg,v_fac_a)), v_T0v);
    // sigt = COEF * sqrt(v_fac_b)*v_T0v;
    __mTd sigt = svmul_f64_z(pg, svmul_f64_z(pg,COEF,svsqrt_f64_z(pg,v_fac_b)), v_T0v);
    // sigp = COEF * sqrt(v_fac_c)*v_T0v;
    __mTd sigp = svmul_f64_z(pg, svmul_f64_z(pg,COEF,svsqrt_f64_z(pg,v_fac_c)), v_T0v);

    // coe = 1.0 / (sigr/D + sigt/D + sigz/D);
    __mTd coe = svdiv_f64_z(pg, v_1, svadd_f64_z(pg, svadd_f64_z(pg, svmul_f64_z(pg, sigr, DR_inv), svmul_f64_z(pg, sigt, DT_inv)), svmul_f64_z(pg, sigp, DP_inv)));
    // coe becomes inf as sig* goes to 0
    // if coe > 1e19, set coe = 1e19
    coe = svmin_f64_z(pg, coe, coe_max);

    // Htau  = v_fac_a * square(v_T0r * v_tau + v_T0v * (v_pr1 + v_pr2)*0.5);
    __mTd Htau = svmul_f64_z(pg, v_fac_a, my_square_v(pg, svadd_f64_z(pg, \
                                                          svmul_f64_z(pg, v_T0r, v_tau), \
                                                          svmul_f64_z(pg, v_T0v, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pr1, v_pr2))))\
                                                   )\
                            );

    // Htau += v_fac_b * square(v_T0t * v_tau + v_T0v * (v_pt1 + v_pt2)*0.5))
    Htau = svadd_f64_z(pg, Htau, svmul_f64_z(pg, v_fac_b, my_square_v(pg, svadd_f64_z(pg, \
                                                                            svmul_f64_z(pg, v_T0t, v_tau), \
                                                                            svmul_f64_z(pg, v_T0v, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pt1, v_pt2))))\
                                                                   )\
                                            )\
                      );
    // Htau += v_fac_c * square(v_T0p * v_tau + v_T0v * (v_pp1 + v_pp2)*0.5))
    Htau = svadd_f64_z(pg, Htau, svmul_f64_z(pg, v_fac_c, my_square_v(pg, svadd_f64_z(pg, \
                                                                            svmul_f64_z(pg, v_T0p, v_tau), \
                                                                            svmul_f64_z(pg, v_T0v, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pp1, v_pp2))))\
                                                                   )\
                                            )\
                      );

    // tmp1 = ( v_T0t * v_tau + v_T0v * (v_pt1 + v_pt2)*0.5)
    __mTd tmp1 = svadd_f64_z(pg, \
                             svmul_f64_z(pg, v_T0t, v_tau), \
                             svmul_f64_z(pg, v_T0v, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pt1, v_pt2)))\
                             );
    // tmp2 = ( v_T0p * v_tau + v_T0v * (v_pp1 + v_pp2)*0.5))
    __mTd tmp2 = svadd_f64_z(pg, \
                             svmul_f64_z(pg, v_T0p, v_tau), \
                             svmul_f64_z(pg, v_T0v, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pp1, v_pp2)))\
                             );
    // tmp3 = -2.0 * v_fac_f * tmp1 * tmp2;
    __mTd tmp3 = svmul_f64_z(pg, v_m2, svmul_f64_z(pg, v_fac_f, svmul_f64_z(pg, tmp1, tmp2)));

    // Htau = sqrt(Htau + tmp3); // # becamse nan as Htau - tmp3 goes to 0
    Htau = svsqrt_f64_z(pg, svadd_f64_z(pg, Htau, tmp3));

    // tmp = (sigr*(v_pr2 - v_pr1) + sigt*(v_pt2 - v_pt1) + sigz*(v_pp2 - v_pp1))*0.5
    __mTd tmp = svmul_f64_z(pg, v_half, svadd_f64_z(pg, \
                                                   svadd_f64_z(pg, \
                                                               svmul_f64_z(pg, sigr, svsub_f64_z(pg, v_pr2, v_pr1)), \
                                                               svmul_f64_z(pg, sigt, svsub_f64_z(pg, v_pt2, v_pt1))\
                                                               ), \
                                                   svmul_f64_z(pg, sigp, svsub_f64_z(pg, v_pp2, v_pp1))\
                                                   )\
                            );

    // v_tau += coe * ((v_fun - Htau) + tmp) if mask is true
    v_tau = svadd_f64_z(pg, v_tau, svmul_f64_z(pg, coe, svadd_f64_z(pg, svsub_f64_z(pg, v_fun, Htau), tmp)));
    // mask = v_change != 1.0
    svbool_t mask = svcmpne_f64(pg, v_change, v_1);
    // v_tau = v_1 if mask is true (!= 1.0)
    v_tau = svsel_f64(mask, v_1, v_tau);

#endif

}



inline void vect_stencil_1st_3rd_apre_simd_tele(
#if USE_ARM_SVE
                                           svbool_t const& pg,
#endif
                                           __mTd& v_tau,        __mTd const& v_fac_a,__mTd const& v_fac_b, __mTd const& v_fac_c, __mTd const& v_fac_f,
                                           __mTd const&  v_fun, __mTd const& v_change,
                                           __mTd const& v_pp1,  __mTd const& v_pp2,  __mTd const& v_pt1,   __mTd const& v_pt2,   __mTd const& v_pr1, __mTd const& v_pr2,
                                           __mTd const& DP_inv, __mTd const& DT_inv, __mTd const& DR_inv){

#if USE_AVX512 || USE_AVX

    // sigr = COEF * sqrt(v_fac_a);
    __mTd sigr = _mmT_mul_pd(COEF_TELE,_mmT_sqrt_pd(v_fac_a));
    // sigt = COEF * sqrt(v_fac_b);
    __mTd sigt = _mmT_mul_pd(COEF_TELE,_mmT_sqrt_pd(v_fac_b));
    // sigp = COEF * sqrt(v_fac_c);
    __mTd sigp = _mmT_mul_pd(COEF_TELE,_mmT_sqrt_pd(v_fac_c));

    // coe = 1.0 / (sigr/D + sigt/D + sigz/D);
    __mTd coe = _mmT_div_pd(v_1,_mmT_add_pd(_mmT_add_pd(_mmT_mul_pd(sigr,DR_inv),_mmT_mul_pd(sigt,DT_inv)),_mmT_mul_pd(sigp,DP_inv)));
    // coe becomes inf as sig* goes to 0
    // if coe > 1e19, set coe = 1e19
    coe = _mmT_min_pd(coe,coe_max);

    // Htau  = v_fac_a * square((v_pr1 + v_pr2)*0.5);
    __mTd Htau = _mmT_mul_pd(v_fac_a,my_square_v(_mmT_mul_pd(_mmT_add_pd(v_pr1,v_pr2),v_half)));

    // Htau += v_fac_b * square((v_pt1 + v_pt2)*0.5))
    Htau = _mmT_add_pd(Htau,_mmT_mul_pd(v_fac_b,my_square_v(_mmT_mul_pd(_mmT_add_pd(v_pt1,v_pt2), v_half))));
    // Htau += v_fac_c * square((v_pp1 + v_pp2)*0.5))
    Htau = _mmT_add_pd(Htau,_mmT_mul_pd(v_fac_c,my_square_v(_mmT_mul_pd(_mmT_add_pd(v_pp1,v_pp2), v_half))));

    // tmp1 = (v_pt1 + v_pt2)*0.5
    __mTd tmp1 = _mmT_mul_pd(_mmT_add_pd(v_pt1,v_pt2), v_half);
    // tmp2 = (v_pp1 + v_pp2)*0.5
    __mTd tmp2 = _mmT_mul_pd(_mmT_add_pd(v_pp1,v_pp2), v_half);
    // tmp3 = -2.0 * v_fac_f * tmp1 * tmp2;
    __mTd tmp3 = _mmT_mul_pd(v_m2,_mmT_mul_pd(v_fac_f,_mmT_mul_pd(tmp1,tmp2)));

    // Htau = sqrt(Htau + tmp3); // # becamse nan as Htau - tmp3 goes to 0
    Htau = _mmT_sqrt_pd(_mmT_add_pd(Htau,tmp3));

    // tmp = (sigr*(v_pr2 - v_pr1) + sigt*(v_pt2 - v_pt1) + sigz*(v_pp2 - v_pp1))*0.5
    __mTd tmp = _mmT_mul_pd(v_half,_mmT_add_pd(_mmT_add_pd(_mmT_mul_pd(sigr,_mmT_sub_pd(v_pr2,v_pr1)),_mmT_mul_pd(sigt,_mmT_sub_pd(v_pt2,v_pt1))),_mmT_mul_pd(sigp,_mmT_sub_pd(v_pp2,v_pp1))));

    // v_tau += coe * ((v_fun - Htau) + tmp) if mask is true
    v_tau = _mmT_add_pd(v_tau,_mmT_mul_pd(coe,_mmT_add_pd(_mmT_sub_pd(v_fun,Htau),tmp)));

#if USE_AVX512
    // mask if v_change != 1.0
    __mmask8 mask = _mm512_cmp_pd_mask(v_change,v_1,_CMP_NEQ_OQ);
    // set 1 if mask is true
    v_tau = _mm512_mask_blend_pd(mask,v_tau,v_1);
#elif USE_AVX
    __m256d mask = _mm256_cmp_pd(v_change, v_1,_CMP_NEQ_OQ);
    v_tau        = _mm256_blendv_pd(v_tau, v_1,mask);
#endif

#elif USE_ARM_SVE

    __mTd COEF_TELE = svdup_f64(1.05);
    __mTd v_1       = svdup_f64(1.0);
    __mTd v_half    = svdup_f64(0.5);
    __mTd v_m2      = svdup_f64(-2.0);
    __mTd coe_max   = svdup_f64(1e19);

    // sigr = COEF * sqrt(v_fac_a);
    __mTd sigr = svmul_f64_z(pg,COEF_TELE,svsqrt_f64_z(pg,v_fac_a));
    // sigt = COEF * sqrt(v_fac_b);
    __mTd sigt = svmul_f64_z(pg,COEF_TELE,svsqrt_f64_z(pg,v_fac_b));
    // sigp = COEF * sqrt(v_fac_c);
    __mTd sigp = svmul_f64_z(pg,COEF_TELE,svsqrt_f64_z(pg,v_fac_c));

    // coe = 1.0 / (sigr/D + sigt/D + sigz/D);
    __mTd coe = svdiv_f64_z(pg, v_1, svadd_f64_z(pg, svadd_f64_z(pg, svmul_f64_z(pg, sigr, DR_inv), svmul_f64_z(pg, sigt, DT_inv)), svmul_f64_z(pg, sigp, DP_inv)));
    // coe becomes inf as sig* goes to 0
    // if coe > 1e19, set coe = 1e19
    coe = svmin_f64_z(pg, coe, coe_max);

    // Htau  = v_fac_a * square((v_pr1 + v_pr2)*0.5);
    __mTd Htau = svmul_f64_z(pg, v_fac_a, my_square_v(pg, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pr1, v_pr2))));
    // Htau += v_fac_b * square((v_pt1 + v_pt2)*0.5))
    Htau = svadd_f64_z(pg, Htau, svmul_f64_z(pg, v_fac_b, my_square_v(pg, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pt1, v_pt2)))));
    // Htau += v_fac_c * square((v_pp1 + v_pp2)*0.5))
    Htau = svadd_f64_z(pg, Htau, svmul_f64_z(pg, v_fac_c, my_square_v(pg, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pp1, v_pp2)))));

    // tmp1 = (v_pt1 + v_pt2)*0.5
    __mTd tmp1 = svadd_f64_z(pg, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pt1, v_pt2)));
    // tmp2 = (v_pp1 + v_pp2)*0.5
    __mTd tmp2 = svadd_f64_z(pg, svmul_f64_z(pg, v_half, svadd_f64_z(pg, v_pp1, v_pp2)));
    // tmp3 = -2.0 * v_fac_f * tmp1 * tmp2;
    __mTd tmp3 = svmul_f64_z(pg, v_m2, svmul_f64_z(pg, v_fac_f, svmul_f64_z(pg, tmp1, tmp2)));

    // Htau = sqrt(Htau + tmp3); // # becamse nan as Htau - tmp3 goes to 0
    Htau = svsqrt_f64_z(pg, svadd_f64_z(pg, Htau, tmp3));

    // tmp = (sigr*(v_pr2 - v_pr1) + sigt*(v_pt2 - v_pt1) + sigz*(v_pp2 - v_pp1))*0.5
    __mTd tmp = svmul_f64_z(pg, v_half, svadd_f64_z(pg, \
                                                   svadd_f64_z(pg, \
                                                               svmul_f64_z(pg, sigr, svsub_f64_z(pg, v_pr2, v_pr1)), \
                                                               svmul_f64_z(pg, sigt, svsub_f64_z(pg, v_pt2, v_pt1))\
                                                               ), \
                                                   svmul_f64_z(pg, sigp, svsub_f64_z(pg, v_pp2, v_pp1))\
                                                   )\
                            );

    // v_tau += coe * ((v_fun - Htau) + tmp) if mask is true
    v_tau = svadd_f64_z(pg, v_tau, svmul_f64_z(pg, coe, svadd_f64_z(pg, svsub_f64_z(pg, v_fun, Htau), tmp)));
    // mask = v_change != 1.0
    svbool_t mask = svcmpne_f64(pg, v_change, v_1);
    // v_tau = v_1 if mask is true (!= 1.0)
    v_tau = svsel_f64(mask, v_1, v_tau);

#endif

}


// function for calculating the values on boundaries
inline void calculate_boundary_nodes_tele_simd(
#if USE_ARM_SVE
    svbool_t const& pg,
#endif
    __mTd const& v_iip, __mTd const& v_jjt, __mTd const& v_kkr,
    __mTd & v_c__,
    __mTd const& v_p__,     __mTd const& v_m__,     __mTd const& v__p_,    __mTd const& v__m_,    __mTd const& v___p,    __mTd const& v___m,
    __mTd const& v_pp____,  __mTd const& v_mm____,  __mTd const& v___pp__, __mTd const& v___mm__, __mTd const& v_____pp, __mTd const& v_____mm,
    __mTd const& v_change,
    int const& loc_I, int const& loc_J, int const& loc_K
){

#if USE_AVX512

    // mask for bottom boundary (v_kkr == 0 && v_change == 1.0)
    __mmask8 mask_bot = _kand_mask8(\
        _mm512_cmp_pd_mask(v_kkr, v_0, _CMP_EQ_OQ), \
        _mm512_cmp_pd_mask(v_change, v_1, _CMP_EQ_OQ));
    // mask for top boundary (v_kkr == loc_K-1 && v_change == 1.0)
    __mmask8 mask_top = _kand_mask8(\
        _mm512_cmp_pd_mask(v_kkr, _mm512_set1_pd(loc_K-1), _CMP_EQ_OQ), \
        _mm512_cmp_pd_mask(v_change, v_1, _CMP_EQ_OQ));
    // mask for south boundary (v_jjt == 0 && v_change == 1.0)
    __mmask8 mask_south = _kand_mask8(\
        _mm512_cmp_pd_mask(v_jjt, v_0, _CMP_EQ_OQ), \
        _mm512_cmp_pd_mask(v_change, v_1, _CMP_EQ_OQ));
    // mask for north boundary (v_jjt == loc_J-1 && v_change == 1.0)
    __mmask8 mask_north = _kand_mask8(\
        _mm512_cmp_pd_mask(v_jjt, _mm512_set1_pd(loc_J-1), _CMP_EQ_OQ), \
        _mm512_cmp_pd_mask(v_change, v_1, _CMP_EQ_OQ));
    // mask for west boundary (v_iip == 0 && v_change == 1.0)
    __mmask8 mask_west = _kand_mask8(\
        _mm512_cmp_pd_mask(v_iip, v_0, _CMP_EQ_OQ), \
        _mm512_cmp_pd_mask(v_change, v_1, _CMP_EQ_OQ));
    // mask for east boundary (v_iip == loc_I-1 && v_change == 1.0)
    __mmask8 mask_east = _kand_mask8(\
        _mm512_cmp_pd_mask(v_iip, _mm512_set1_pd(loc_I-1), _CMP_EQ_OQ), \
        _mm512_cmp_pd_mask(v_change, v_1, _CMP_EQ_OQ));

    // if mask_bot, v_c__ = max(2*v___p - v_____pp, v_____pp)
    v_c__ = _mm512_mask_blend_pd(mask_bot, v_c__, _mm512_max_pd(_mm512_sub_pd(_mm512_mul_pd(v_2, v___p), v_____pp), v_____pp));
    // if mask_top, v_c__ = max(2*v___m - v_____mm, v_____mm)
    v_c__ = _mm512_mask_blend_pd(mask_top, v_c__, _mm512_max_pd(_mm512_sub_pd(_mm512_mul_pd(v_2, v___m), v_____mm), v_____mm));
    // if mask_south, v_c__ = max(2*v__p_ - v___pp__, v___pp__)
    v_c__ = _mm512_mask_blend_pd(mask_south, v_c__, _mm512_max_pd(_mm512_sub_pd(_mm512_mul_pd(v_2, v__p_), v___pp__), v___pp__));
    // if mask_north, v_c__ = max(2*v__m_ - v___mm__, v___mm__)
    v_c__ = _mm512_mask_blend_pd(mask_north, v_c__, _mm512_max_pd(_mm512_sub_pd(_mm512_mul_pd(v_2, v__m_), v___mm__), v___mm__));
    // if mask_west, v_c__ = max(2*v_p__ - v_pp____, v_pp____)
    v_c__ = _mm512_mask_blend_pd(mask_west, v_c__, _mm512_max_pd(_mm512_sub_pd(_mm512_mul_pd(v_2, v_p__), v_pp____), v_pp____));
    // if mask_east, v_c__ = max(2*v_m__ - v_mm____, v_mm____)
    v_c__ = _mm512_mask_blend_pd(mask_east, v_c__, _mm512_max_pd(_mm512_sub_pd(_mm512_mul_pd(v_2, v_m__), v_mm____), v_mm____));
#elif USE_AVX

    // mask for bottom boundary (v_kkr == 0 && v_change == 1.0)
    __m256d mask_bot = _mm256_and_pd(\
        _mm256_cmp_pd(v_kkr, v_0, _CMP_EQ_OQ), \
        _mm256_cmp_pd(v_change, v_1, _CMP_EQ_OQ));
    // mask for top boundary (v_kkr == loc_K-1 && v_change == 1.0)
    __m256d mask_top = _mm256_and_pd(\
        _mm256_cmp_pd(v_kkr, _mm256_set1_pd(loc_K-1), _CMP_EQ_OQ), \
        _mm256_cmp_pd(v_change, v_1, _CMP_EQ_OQ));
    // mask for south boundary (v_jjt == 0 && v_change == 1.0)
    __m256d mask_south = _mm256_and_pd(\
        _mm256_cmp_pd(v_jjt, v_0, _CMP_EQ_OQ), \
        _mm256_cmp_pd(v_change, v_1, _CMP_EQ_OQ));
    // mask for north boundary (v_jjt == loc_J-1 && v_change == 1.0)
    __m256d mask_north = _mm256_and_pd(\
        _mm256_cmp_pd(v_jjt, _mm256_set1_pd(loc_J-1), _CMP_EQ_OQ), \
        _mm256_cmp_pd(v_change, v_1, _CMP_EQ_OQ));
    // mask for west boundary (v_iip == 0 && v_change == 1.0)
    __m256d mask_west = _mm256_and_pd(\
        _mm256_cmp_pd(v_iip, v_0, _CMP_EQ_OQ), \
        _mm256_cmp_pd(v_change, v_1, _CMP_EQ_OQ));
    // mask for east boundary (v_iip == loc_I-1 && v_change == 1.0)
    __m256d mask_east = _mm256_and_pd(\
        _mm256_cmp_pd(v_iip, _mm256_set1_pd(loc_I-1), _CMP_EQ_OQ), \
        _mm256_cmp_pd(v_change, v_1, _CMP_EQ_OQ));
    // if mask_bot, v_c__ = max(2*v___p - v_____pp, v_____pp)
    v_c__ = _mm256_blendv_pd(v_c__, _mm256_max_pd(_mm256_sub_pd(_mm256_mul_pd(v_2, v___p), v_____pp), v_____pp), mask_bot);
    // if mask_top, v_c__ = max(2*v___m - v_____mm, v_____mm)
    v_c__ = _mm256_blendv_pd(v_c__, _mm256_max_pd(_mm256_sub_pd(_mm256_mul_pd(v_2, v___m), v_____mm), v_____mm), mask_top);
    // if mask_south, v_c__ = max(2*v__p_ - v___pp__, v___pp__)
    v_c__ = _mm256_blendv_pd(v_c__, _mm256_max_pd(_mm256_sub_pd(_mm256_mul_pd(v_2, v__p_), v___pp__), v___pp__), mask_south);
    // if mask_north, v_c__ = max(2*v__m_ - v___mm__, v___mm__)
    v_c__ = _mm256_blendv_pd(v_c__, _mm256_max_pd(_mm256_sub_pd(_mm256_mul_pd(v_2, v__m_), v___mm__), v___mm__), mask_north);
    // if mask_west, v_c__ = max(2*v_p__ - v_pp____, v_pp____)
    v_c__ = _mm256_blendv_pd(v_c__, _mm256_max_pd(_mm256_sub_pd(_mm256_mul_pd(v_2, v_p__), v_pp____), v_pp____), mask_west);
    // if mask_east, v_c__ = max(2*v_m__ - v_mm____, v_mm____)
    v_c__ = _mm256_blendv_pd(v_c__, _mm256_max_pd(_mm256_sub_pd(_mm256_mul_pd(v_2, v_m__), v_mm____), v_mm____), mask_east);

#elif USE_ARM_SVE

    svfloat64_t v_0 = svdup_f64(0.0);
    svfloat64_t v_1 = svdup_f64(1.0);
    svfloat64_t v_2 = svdup_f64(2.0);
    svfloat64_t v_loc_I_minus_1 = svdup_f64(loc_I-1);
    svfloat64_t v_loc_J_minus_1 = svdup_f64(loc_J-1);
    svfloat64_t v_loc_K_minus_1 = svdup_f64(loc_K-1);


    // mask for bottom boundary
    svbool_t mask_bot = svand_b_z(pg, \
        svcmpeq_f64(pg, v_kkr, v_0), \
        svcmpeq_b64(pg, v_change, v_1));
    // mask for top boundary
    svbool_t mask_top = svand_b_z(pg, \
        svcmpeq_f64(pg, v_kkr, v_loc_K_minus_1), \
        svcmpeq_b64(pg, v_change, v_1));
    // mask for south boundary
    svbool_t mask_south = svand_b_z(pg, \
        svcmpeq_f64(pg, v_jjt, v_0), \
        svcmpeq_b64(pg, v_change, v_1));
    // mask for north boundary
    svbool_t mask_north = svand_b_z(pg, \
        svcmpeq_f64(pg, v_jjt, v_loc_J_minus_1), \
        svcmpeq_b64(pg, v_change, v_1));
    // mask for west boundary
    svbool_t mask_west = svand_b_z(pg, \
        svcmpeq_f64(pg, v_iip, v_0), \
        svcmpeq_b64(pg, v_change, v_1));
    // mask for east boundary
    svbool_t mask_east = svand_b_z(pg, \
        svcmpeq_f64(pg, v_iip, v_loc_I_minus_1), \
        svcmpeq_b64(pg, v_change, v_1));

    // if mask_bot, v_c__ = max(2*v___p - v_____pp, v_____pp)
    v_c__ = svsel_f64(mask_bot, \
        svmax_f64_z(pg, svsub_f64_z(pg, svmul_f64_z(pg, v_2, v___p), v_____pp), v_____pp), \
        v_c__);
    // if mask_top, v_c__ = max(2*v___m - v_____mm, v_____mm)
    v_c__ = svsel_f64(mask_top, \
        svmax_f64_z(pg, svsub_f64_z(pg, svmul_f64_z(pg, v_2, v___m), v_____mm), v_____mm), \
        v_c__);
    // if mask_south, v_c__ = max(2*v__p_ - v___pp__, v___pp__)
    v_c__ = svsel_f64(mask_south, \
        svmax_f64_z(pg, svsub_f64_z(pg, svmul_f64_z(pg, v_2, v__p_), v___pp__), v___pp__), \
        v_c__);
    // if mask_north, v_c__ = max(2*v__m_ - v___mm__, v___mm__)
    v_c__ = svsel_f64(mask_north, \
        svmax_f64_z(pg, svsub_f64_z(pg, svmul_f64_z(pg, v_2, v__m_), v___mm__), v___mm__), \
        v_c__);
    // if mask_west, v_c__ = max(2*v_p__ - v_pp____, v_pp____)
    v_c__ = svsel_f64(mask_west, \
        svmax_f64_z(pg, svsub_f64_z(pg, svmul_f64_z(pg, v_2, v_p__), v_pp____), v_pp____), \
        v_c__);
    // if mask_east, v_c__ = max(2*v_m__ - v_mm____, v_mm____)
    v_c__ = svsel_f64(mask_east, \
        svmax_f64_z(pg, svsub_f64_z(pg, svmul_f64_z(pg, v_2, v_m__), v_mm____), v_mm____), \
        v_c__);

#endif

}






#if USE_AVX512 || USE_AVX

inline __mTd load_mem_gen_to_mTd(CUSTOMREAL* a, int* ijk){

        CUSTOMREAL dump_[NSIMD];
        for (int i=0; i<NSIMD; i++){
            dump_[i] = a[ijk[i]];
        }

        return  _mmT_loadu_pd(dump_);
}

inline __mTd load_mem_bool_to_mTd(bool* a, int* ijk){

        CUSTOMREAL dump_[NSIMD];
        for (int i=0; i<NSIMD; i++){
            dump_[i] = (CUSTOMREAL) a[ijk[i]];
        }

       return _mmT_loadu_pd(dump_);
}

#elif USE_ARM_SVE

inline __mTd load_mem_gen_to_mTd(svbool_t const& pg, CUSTOMREAL* a, uint64_t* ijk){
    svuint64_t v_ijk = svld1_u64(pg, ijk);
    return svld1_gather_u64index_f64(pg, a, v_ijk);
}

inline __mTd load_mem_bool_to_mTd(svbool_t const& pg, bool* a, int* ijk){
        CUSTOMREAL dump_[NSIMD];

        for (int i=0; i<NSIMD; i++){
            dump_[i] = (CUSTOMREAL) a[ijk[i]];
        }

        return svld1_f64(pg, dump_); // change this to gather load
}

#endif // __ARM_FEATURE_SVE

#endif // USE_SIMD



#endif // VECTORIZATED_SWEEP_H