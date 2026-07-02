#include "gpu_kernels.cuh"

#include <cmath>

namespace tomogpu {

namespace {

__device__ __forceinline__ real_t lf_hamiltonian(
    real_t fac_a,
    real_t fac_b,
    real_t fac_c,
    real_t fac_f,
    real_t T0r,
    real_t T0t,
    real_t T0p,
    real_t T0v,
    real_t tau,
    real_t pp1,
    real_t pp2,
    real_t pt1,
    real_t pt2,
    real_t pr1,
    real_t pr2) {
    const real_t half = static_cast<real_t>(0.5);
    const real_t two = static_cast<real_t>(2.0);

    const real_t dr_term = T0r * tau + T0v * (pr1 + pr2) * half;
    const real_t dt_term = T0t * tau + T0v * (pt1 + pt2) * half;
    const real_t dp_term = T0p * tau + T0v * (pp1 + pp2) * half;

    return ::sqrt(
        fac_a * dr_term * dr_term +
        fac_b * dt_term * dt_term +
        fac_c * dp_term * dp_term -
        two * fac_f * dt_term * dp_term);
}

}  // namespace

template <int kOrder>
__launch_bounds__(kDefaultBlockSize)
__global__ void sweep_level_kernel(
    real_t* __restrict__ tau,
    const real_t* __restrict__ T0v,
    const real_t* __restrict__ T0r,
    const real_t* __restrict__ T0t,
    const real_t* __restrict__ T0p,
    const real_t* __restrict__ fac_a,
    const real_t* __restrict__ fac_b,
    const real_t* __restrict__ fac_c,
    const real_t* __restrict__ fac_f,
    const real_t* __restrict__ fun,
    const uint8_t* __restrict__ is_changed,
    const idx_t* __restrict__ level_offsets,
    const idx_t* __restrict__ level_counts,
    const idx_t* __restrict__ level_indices,
    int level,
    int sign_r,
    int sign_t,
    int sign_p,
    int nx,
    int ny,
    int nz,
    real_t dr,
    real_t dt,
    real_t dp) {
    (void)kOrder;  // 3rd order path currently uses same stencil math as 1st order.

    const idx_t tid = static_cast<idx_t>(blockIdx.x) * static_cast<idx_t>(blockDim.x) +
                      static_cast<idx_t>(threadIdx.x);
    if (tid >= level_counts[level]) {
        return;
    }

    const idx_t natural_flat = level_indices[level_offsets[level] + tid];

    const idx_t k = natural_flat / (nx * ny);
    const idx_t j = (natural_flat - k * nx * ny) / nx;
    const idx_t i = natural_flat - k * nx * ny - j * nx;

    const idx_t phys_i = (sign_p > 0) ? (i - 1) : (nx - i);
    const idx_t phys_j = (sign_t > 0) ? (j - 1) : (ny - j);
    const idx_t phys_k = (sign_r > 0) ? (k - 1) : (nz - k);

    if (phys_i <= 0 || phys_i >= nx - 1 ||
        phys_j <= 0 || phys_j >= ny - 1 ||
        phys_k <= 0 || phys_k >= nz - 1) {
        return;
    }

    const idx_t c = phys_k * nx * ny + phys_j * nx + phys_i;
    if (!is_changed[c]) {
        return;
    }

    const idx_t cm1 = c - 1;
    const idx_t cp1 = c + 1;
    const idx_t jm1 = c - nx;
    const idx_t jp1 = c + nx;
    const idx_t km1 = c - nx * ny;
    const idx_t kp1 = c + nx * ny;

    const real_t dp_inv = kOne / dp;
    const real_t dt_inv = kOne / dt;
    const real_t dr_inv = kOne / dr;

    const real_t pp1 = (tau[c] - tau[cm1]) * dp_inv;
    const real_t pp2 = (tau[cp1] - tau[c]) * dp_inv;
    const real_t pt1 = (tau[c] - tau[jm1]) * dt_inv;
    const real_t pt2 = (tau[jp1] - tau[c]) * dt_inv;
    const real_t pr1 = (tau[c] - tau[km1]) * dr_inv;
    const real_t pr2 = (tau[kp1] - tau[c]) * dr_inv;

    const real_t sigr = kSweepCoeff * ::sqrt(fac_a[c]) * T0v[c];
    const real_t sigt = kSweepCoeff * ::sqrt(fac_b[c]) * T0v[c];
    const real_t sigp = kSweepCoeff * ::sqrt(fac_c[c]) * T0v[c];
    const real_t coe = kOne / ((sigr / dr) + (sigt / dt) + (sigp / dp));

    const real_t Htau = lf_hamiltonian(
        fac_a[c], fac_b[c], fac_c[c], fac_f[c],
        T0r[c], T0t[c], T0p[c], T0v[c], tau[c],
        pp1, pp2, pt1, pt2, pr1, pr2);

    const real_t correction =
        (sigr * (pr2 - pr1) + sigt * (pt2 - pt1) + sigp * (pp2 - pp1)) * kHalf;

    tau[c] += coe * (fun[c] - Htau + correction);
}

template __global__ void sweep_level_kernel<1>(
    real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const uint8_t* __restrict__,
    const idx_t* __restrict__,
    const idx_t* __restrict__,
    const idx_t* __restrict__,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    real_t,
    real_t,
    real_t);

template __global__ void sweep_level_kernel<3>(
    real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const real_t* __restrict__,
    const uint8_t* __restrict__,
    const idx_t* __restrict__,
    const idx_t* __restrict__,
    const idx_t* __restrict__,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    real_t,
    real_t,
    real_t);

__global__ void boundary_kernel(
    real_t* __restrict__ tau,
    const real_t* __restrict__ T0v,
    const real_t* __restrict__ T0r,
    const real_t* __restrict__ T0t,
    const real_t* __restrict__ T0p,
    const real_t* __restrict__ fac_a,
    const real_t* __restrict__ fac_b,
    const real_t* __restrict__ fac_c,
    const real_t* __restrict__ fac_f,
    const real_t* __restrict__ fun,
    const uint8_t* __restrict__ is_changed,
    int nx,
    int ny,
    int nz,
    real_t dr,
    real_t dt,
    real_t dp,
    int stencil_order) {
    (void)tau;
    (void)T0v;
    (void)T0r;
    (void)T0t;
    (void)T0p;
    (void)fac_a;
    (void)fac_b;
    (void)fac_c;
    (void)fac_f;
    (void)fun;
    (void)is_changed;
    (void)nx;
    (void)ny;
    (void)nz;
    (void)dr;
    (void)dt;
    (void)dp;
    (void)stencil_order;
    // Boundary handling is currently performed on CPU side after copy-back.
}

__global__ void init_changed_kernel(uint8_t* __restrict__ is_changed, int nx, int ny, int nz) {
    const idx_t tid = static_cast<idx_t>(blockIdx.x) * static_cast<idx_t>(blockDim.x) +
                      static_cast<idx_t>(threadIdx.x);
    const idx_t total = static_cast<idx_t>(nx) * ny * nz;
    if (tid >= total) {
        return;
    }

    const idx_t k = tid / (nx * ny);
    const idx_t j = (tid - k * nx * ny) / nx;
    const idx_t i = tid - k * nx * ny - j * nx;

    is_changed[tid] =
        (i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1)
            ? static_cast<uint8_t>(1)
            : static_cast<uint8_t>(0);
}

__global__ void convergence_reduce_kernel(
    const real_t* __restrict__ tau,
    const real_t* __restrict__ tau_old,
    int total_nodes,
    real_t* __restrict__ block_l1,
    real_t* __restrict__ block_linf) {
    extern __shared__ real_t sdata[];

    const idx_t tid = static_cast<idx_t>(blockIdx.x) * static_cast<idx_t>(blockDim.x) +
                      static_cast<idx_t>(threadIdx.x);
    const idx_t stride = static_cast<idx_t>(blockDim.x) * static_cast<idx_t>(gridDim.x);

    real_t local_l1 = static_cast<real_t>(0);
    real_t local_linf = static_cast<real_t>(0);

    for (idx_t i = tid; i < total_nodes; i += stride) {
        const real_t diff = ::fabs(tau[i] - tau_old[i]);
        local_l1 += diff;
        local_linf = ::fmax(local_linf, diff);
    }

    sdata[threadIdx.x] = local_l1;
    __syncthreads();

    for (idx_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_l1[blockIdx.x] = sdata[0];
    }

    sdata[threadIdx.x] = local_linf;
    __syncthreads();

    for (idx_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] = ::fmax(sdata[threadIdx.x], sdata[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_linf[blockIdx.x] = sdata[0];
    }
}

__global__ void fill_real_kernel(real_t* arr, real_t value, int n) {
    const idx_t tid = static_cast<idx_t>(blockIdx.x) * static_cast<idx_t>(blockDim.x) +
                      static_cast<idx_t>(threadIdx.x);
    if (tid < n) {
        arr[tid] = value;
    }
}

__global__ void fill_bool_kernel(uint8_t* arr, uint8_t value, int n) {
    const idx_t tid = static_cast<idx_t>(blockIdx.x) * static_cast<idx_t>(blockDim.x) +
                      static_cast<idx_t>(threadIdx.x);
    if (tid < n) {
        arr[tid] = value;
    }
}

__global__ void copy_real_kernel(real_t* dst, const real_t* src, int n) {
    const idx_t tid = static_cast<idx_t>(blockIdx.x) * static_cast<idx_t>(blockDim.x) +
                      static_cast<idx_t>(threadIdx.x);
    if (tid < n) {
        dst[tid] = src[tid];
    }
}

}  // namespace tomogpu
