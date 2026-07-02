//
// gpu_memory.h — RAII device memory management for the clean-slate GPU backend.
//
// Provides:
//   * `DeviceBuffer<T>` — owning wrapper around a device allocation.
//   * `PinnedHostBuffer<T>` — page-locked host buffer for fast H2D/D2H.
//   * Stream-ordered allocation helpers (`cudaMallocAsync`).
//
// All allocations are stream-ordered so they can be reused across sweeps
// without explicit cudaFree calls, reducing driver overhead.
//
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <utility>

#include "gpu_types.h"

namespace tomogpu {

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
namespace detail {

inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        // Note: using fprintf to avoid <iostream> header cost in kernels
        fprintf(stderr, "CUDA error at %s:%d : %s\n", file, line,
                cudaGetErrorString(err));
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

}  // namespace detail

#define TOMOGPU_CUDA_CHECK(err) \
    ::tomogpu::detail::check_cuda((err), __FILE__, __LINE__)

// ---------------------------------------------------------------------------
// DeviceBuffer<T> — RAII device allocation
// ---------------------------------------------------------------------------
// Owns a contiguous region of device memory allocated via
// `cudaMallocAsync` on a given stream.  The buffer is freed in the
// destructor (also stream-ordered).
//
// Move-only: copying is intentionally disabled to prevent double-free.
//
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    // Allocate `count` elements on `stream`.
    DeviceBuffer(std::size_t count, cudaStream_t stream = 0)
        : count_(count), stream_(stream) {
        if (count == 0) return;
        T* ptr = nullptr;
        TOMOGPU_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr),
                                            count * sizeof(T), stream));
        ptr_ = ptr;
    }

    // Destructor — stream-ordered free.
    ~DeviceBuffer() { reset(); }

    // Move semantics
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_), stream_(other.stream_) {
        other.ptr_   = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_    = other.ptr_;
            count_  = other.count_;
            stream_ = other.stream_;
            other.ptr_   = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Accessors
    T*       get()       noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size()  const noexcept { return count_; }
    bool      empty()    const noexcept { return ptr_ == nullptr; }

    // Release ownership (caller becomes responsible for freeing).
    T* release() noexcept {
        T* tmp = ptr_;
        ptr_   = nullptr;
        count_ = 0;
        return tmp;
    }

    // Reset (free if owning).
    void reset() {
        if (ptr_) {
            cudaFreeAsync(ptr_, stream_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    // Reallocate to a new size (frees old allocation first).
    void resize(std::size_t count, cudaStream_t stream = 0) {
        reset();
        count_  = count;
        stream_ = stream;
        if (count == 0) return;
        T* ptr = nullptr;
        TOMOGPU_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr),
                                            count * sizeof(T), stream));
        ptr_ = ptr;
    }

    // Copy host data into this buffer.
    void copy_from_host(const T* host, cudaStream_t stream = 0) {
        if (count_ == 0) return;
        TOMOGPU_CUDA_CHECK(cudaMemcpyAsync(ptr_, host, count_ * sizeof(T),
                                            cudaMemcpyHostToDevice, stream));
    }

    // Copy this buffer to a host array.
    void copy_to_host(T* host, cudaStream_t stream = 0) const {
        if (count_ == 0) return;
        TOMOGPU_CUDA_CHECK(cudaMemcpyAsync(host, ptr_, count_ * sizeof(T),
                                            cudaMemcpyDeviceToHost, stream));
    }

private:
    T*           ptr_    = nullptr;
    std::size_t  count_  = 0;
    cudaStream_t stream_ = 0;
};

// ---------------------------------------------------------------------------
// PinnedHostBuffer<T> — page-locked host memory
// ---------------------------------------------------------------------------
// Allocated via `cudaMallocHost` to enable async H2D/D2H copies overlapped
// with computation.  Maps host memory into the CUDA address space when
// `cudaHostAllocMapped` is requested.
//
template <typename T>
class PinnedHostBuffer {
public:
    PinnedHostBuffer() = default;

    explicit PinnedHostBuffer(std::size_t count, unsigned int flags = 0)
        : count_(count) {
        if (count == 0) return;
        T* ptr = nullptr;
        TOMOGPU_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                         count * sizeof(T), flags));
        ptr_ = ptr;
    }

    ~PinnedHostBuffer() { reset(); }

    PinnedHostBuffer(PinnedHostBuffer&& other) noexcept
        : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_   = nullptr;
        other.count_ = 0;
    }

    PinnedHostBuffer& operator=(PinnedHostBuffer&& other) noexcept {
        if (this != &other) {
            reset();
            ptr_   = other.ptr_;
            count_ = other.count_;
            other.ptr_   = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    PinnedHostBuffer(const PinnedHostBuffer&)            = delete;
    PinnedHostBuffer& operator=(const PinnedHostBuffer&) = delete;

    T*       get()       noexcept { return ptr_; }
    const T* get() const noexcept { return ptr_; }
    std::size_t size()  const noexcept { return count_; }

    void reset() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        count_ = 0;
    }

    void resize(std::size_t count, unsigned int flags = 0) {
        reset();
        count_ = count;
        if (count == 0) return;
        T* ptr = nullptr;
        TOMOGPU_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                         count * sizeof(T), flags));
        ptr_ = ptr;
    }

private:
    T*          ptr_   = nullptr;
    std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// Convenience type aliases
// ---------------------------------------------------------------------------
using RealDeviceBuffer   = DeviceBuffer<real_t>;
using IndexDeviceBuffer  = DeviceBuffer<idx_t>;
using BoolDeviceBuffer   = DeviceBuffer<uint8_t>;  // bool arrays stored as uint8

}  // namespace tomogpu
