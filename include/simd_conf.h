#ifndef SIMD_CONF_H
#define SIMD_CONF_H

#ifdef USE_SIMD
#if defined __AVX__ || defined __AVX512F__
#include <immintrin.h>
#elif defined __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif // __
#endif // USE_SIMD

#ifdef USE_SIMD

#ifdef __AVX__
const int ALIGN = 32;
const int NSIMD = 4;
#define __mTd __m256d
#define _mmT_set1_pd _mm256_set1_pd
#define _mmT_loadu_pd _mm256_loadu_pd
#define _mmT_mul_pd _mm256_mul_pd
#define _mmT_sub_pd _mm256_sub_pd
#define _mmT_add_pd _mm256_add_pd
#define _mmT_div_pd _mm256_div_pd
#define _mmT_min_pd _mm256_min_pd
#define _mmT_sqrt_pd _mm256_sqrt_pd
#define _mmT_store_pd _mm256_store_pd
#endif // __AVX__
#ifdef __AVX512F__
const int ALIGN = 64;
const int NSIMD = 8;
#define __mTd __m512d
#define _mmT_set1_pd _mm512_set1_pd
#define _mmT_loadu_pd _mm512_loadu_pd
#define _mmT_mul_pd _mm512_mul_pd
#define _mmT_sub_pd _mm512_sub_pd
#define _mmT_add_pd _mm512_add_pd
#define _mmT_div_pd _mm512_div_pd
#define _mmT_min_pd _mm512_min_pd
#define _mmT_sqrt_pd _mm512_sqrt_pd
#define _mmT_store_pd _mm512_store_pd
#endif // __AVX512F__

#endif // USE_SIMD

#endif // SIMD_CONF_H