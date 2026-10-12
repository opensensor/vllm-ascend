// SPDX-License-Identifier: Apache-2.0
// Resident vendor GEMM and standalone vector/copy probes. No device resets.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>

namespace {
constexpr int kThreads = 256;
constexpr int kMaxDim = 8192;
constexpr size_t kMaxElements = 64ULL * 1024 * 1024;
constexpr size_t kMaxCopyBytes = 1024ULL * 1024 * 1024;
constexpr float kEpsilon = 1e-6F;
enum Operation { kGemm, kAdd, kExp, kSoftmax, kRmsnorm, kCopy, kH2d, kD2h };
enum Precision { kFp16, kFp32, kBf16, kInt8, kUint8 };

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(status));
}

struct Probe {
  int op, dtype, m, n, k;
  bool pinned;
  cublasHandle_t blas = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  void *x = nullptr, *weight = nullptr, *out = nullptr, *integer_out = nullptr, *host = nullptr;
  size_t output_bytes = 0;
  std::string error;
  ~Probe() {
    cudaFree(x);
    cudaFree(weight);
    cudaFree(out);
    cudaFree(integer_out);
    if (host) {
      if (pinned)
        cudaFreeHost(host);
      else
        std::free(host);
    }
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    if (blas) cublasDestroy(blas);
  }
};

__global__ void int_to_half(const int* input, __half* output, size_t count) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) output[index] = __float2half_rn(__int2float_rn(input[index]));
}
__global__ void elementwise(const float* x, const float* y, float* out, size_t count, int op) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) out[index] = op == kAdd ? __fadd_rn(x[index], y[index]) : expf(x[index]);
}
__global__ void row_math(const float* x, float* out, int width, int op) {
  __shared__ float reduction[kThreads];
  size_t base = size_t(blockIdx.x) * width;
  float value = op == kSoftmax ? -INFINITY : 0;
  for (int column = threadIdx.x; column < width; column += blockDim.x) {
    float v = x[base + column];
    value = op == kSoftmax ? fmaxf(value, v) : __fadd_rn(value, __fmul_rn(v, v));
  }
  reduction[threadIdx.x] = value;
  __syncthreads();
  for (int stride = kThreads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reduction[threadIdx.x] = op == kSoftmax ? fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + stride])
                                              : __fadd_rn(reduction[threadIdx.x], reduction[threadIdx.x + stride]);
    __syncthreads();
  }
  float maximum = reduction[0];
  if (op == kSoftmax) {
    value = 0;
    for (int column = threadIdx.x; column < width; column += blockDim.x) value += expf(x[base + column] - maximum);
    reduction[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kThreads / 2; stride; stride /= 2) {
      if (threadIdx.x < stride) reduction[threadIdx.x] += reduction[threadIdx.x + stride];
      __syncthreads();
    }
  }
  float factor = op == kSoftmax ? reduction[0] : rsqrtf(maximum / width + kEpsilon);
  for (int column = threadIdx.x; column < width; column += blockDim.x)
    out[base + column] = op == kSoftmax ? expf(x[base + column] - maximum) / factor : x[base + column] * factor;
}

void run(Probe* p) {
  if (p->op == kGemm) {
    const float one = 1, zero = 0;
    auto type = p->dtype == kFp16 ? CUDA_R_16F : p->dtype == kBf16 ? CUDA_R_16BF : CUDA_R_32F;
    if (p->dtype == kInt8) {
      const int ione = 1, izero = 0;
      blas_check(cublasGemmEx(p->blas, CUBLAS_OP_T, CUBLAS_OP_N, p->n, p->m, p->k, &ione, p->weight, CUDA_R_8I, p->k,
                              p->x, CUDA_R_8I, p->k, &izero, p->integer_out, CUDA_R_32I, p->n, CUBLAS_COMPUTE_32I,
                              CUBLAS_GEMM_DEFAULT));
      size_t elements = size_t(p->m) * p->n;
      int_to_half<<<(elements + kThreads - 1) / kThreads, kThreads>>>(static_cast<const int*>(p->integer_out),
                                                                      static_cast<__half*>(p->out), elements);
    } else {
      auto compute = p->dtype == kFp32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
      blas_check(cublasGemmEx(p->blas, CUBLAS_OP_T, CUBLAS_OP_N, p->n, p->m, p->k, &one, p->weight, type, p->k, p->x,
                              type, p->k, &zero, p->out, type, p->n, compute, CUBLAS_GEMM_DEFAULT));
    }
  } else if (p->op == kAdd || p->op == kExp) {
    elementwise<<<(p->m + kThreads - 1) / kThreads, kThreads>>>(static_cast<const float*>(p->x),
                                                                static_cast<const float*>(p->weight),
                                                                static_cast<float*>(p->out), p->m, p->op);
  } else if (p->op == kSoftmax || p->op == kRmsnorm) {
    row_math<<<p->m, kThreads>>>(static_cast<const float*>(p->x), static_cast<float*>(p->out), p->n, p->op);
  } else {
    check(cudaMemcpyAsync(p->op == kD2h ? p->host : p->out, p->op == kH2d ? p->host : p->x, p->output_bytes,
                          p->op == kH2d   ? cudaMemcpyHostToDevice
                          : p->op == kD2h ? cudaMemcpyDeviceToHost
                                          : cudaMemcpyDeviceToDevice));
  }
  check(cudaGetLastError());
}

void upload(void** target, const void* input, size_t bytes) {
  check(cudaMalloc(target, bytes));
  check(cudaMemcpy(*target, input, bytes, cudaMemcpyHostToDevice));
}
}  // namespace

extern "C" void* raw_math_create(int op, int dtype, int m, int n, int k, int pinned, const void* x,
                                 const void* weight) {
  auto* p = new Probe();
  p->op = op;
  p->dtype = dtype;
  p->m = m;
  p->n = n;
  p->k = k;
  p->pinned = pinned;
  try {
    if (!x || op < kGemm || op > kD2h || dtype < kFp16 || dtype > kUint8 || m < 1 || pinned < 0 || pinned > 1)
      throw std::runtime_error("invalid raw math arguments");
    bool matrix = op == kGemm;
    bool rows = op == kSoftmax || op == kRmsnorm;
    bool copy = op >= kCopy;
    if ((matrix && (m > kMaxDim || n < 1 || n > kMaxDim || k < 1 || k > kMaxDim || dtype == kUint8 || !weight)) ||
        (rows && (m > kMaxDim || n < 1 || n > kMaxDim || k || dtype != kFp32)) ||
        ((op == kAdd || op == kExp) &&
         (size_t(m) > kMaxElements || n || k || dtype != kFp32 || (op == kAdd && !weight))) ||
        (copy && (size_t(m) > kMaxCopyBytes || n || k || dtype != kUint8)) || (pinned && op != kH2d && op != kD2h))
      throw std::runtime_error("raw math geometry exceeds its contract");
    check(cudaSetDevice(0));
    check(cudaEventCreate(&p->begin));
    check(cudaEventCreate(&p->end));
    size_t element = dtype == kFp32 ? 4 : dtype == kInt8 || dtype == kUint8 ? 1 : 2;
    size_t elements = matrix || rows ? size_t(m) * n : size_t(m);
    p->output_bytes = elements * (dtype == kInt8 ? 2 : element);
    upload(&p->x, x, (matrix ? size_t(m) * k : elements) * element);
    if (matrix || op == kAdd) upload(&p->weight, weight, (matrix ? size_t(n) * k : elements) * element);
    check(cudaMalloc(&p->out, p->output_bytes));
    if (matrix) {
      blas_check(cublasCreate(&p->blas));
      blas_check(cublasSetMathMode(p->blas, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
      if (dtype == kInt8) check(cudaMalloc(&p->integer_out, elements * sizeof(int)));
    }
    if (op == kH2d || op == kD2h) {
      if (pinned)
        check(cudaHostAlloc(&p->host, p->output_bytes, cudaHostAllocDefault));
      else {
        p->host = std::malloc(p->output_bytes);
        if (!p->host) throw std::bad_alloc();
      }
      std::memcpy(p->host, x, p->output_bytes);
    }
    check(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    p->error = e.what();
  }
  return p;
}
extern "C" const char* raw_math_error(void* handle) {
  return handle ? static_cast<Probe*>(handle)->error.c_str() : "null probe";
}
extern "C" int raw_math_run(void* handle, int repeats, float* milliseconds) {
  if (!handle || !milliseconds || repeats < 1 || repeats > 64) return -1;
  auto* p = static_cast<Probe*>(handle);
  if (!p->error.empty()) return -1;
  try {
    check(cudaEventRecord(p->begin));
    for (int i = 0; i < repeats; ++i) run(p);
    check(cudaEventRecord(p->end));
    check(cudaEventSynchronize(p->end));
    check(cudaEventElapsedTime(milliseconds, p->begin, p->end));
  } catch (const std::exception& e) {
    p->error = e.what();
  }
  return p->error.empty() ? 0 : -1;
}
extern "C" int raw_math_read(void* handle, void* host) {
  if (!handle || !host) return -1;
  auto* p = static_cast<Probe*>(handle);
  if (!p->error.empty()) return -1;
  try {
    if (p->op == kD2h)
      std::memcpy(host, p->host, p->output_bytes);
    else
      check(cudaMemcpy(host, p->out, p->output_bytes, cudaMemcpyDeviceToHost));
  } catch (const std::exception& e) {
    p->error = e.what();
  }
  return p->error.empty() ? 0 : -1;
}
extern "C" int raw_math_refresh(void* handle, const void* host) {
  if (!handle || !host) return -1;
  auto* p = static_cast<Probe*>(handle);
  if (!p->error.empty()) return -1;
  size_t element = p->dtype == kFp32 ? 4 : p->dtype == kInt8 || p->dtype == kUint8 ? 1 : 2;
  size_t elements = p->op == kGemm                           ? size_t(p->m) * p->k
                    : p->op == kSoftmax || p->op == kRmsnorm ? size_t(p->m) * p->n
                                                             : size_t(p->m);
  try {
    if (p->op == kH2d)
      std::memcpy(p->host, host, elements * element);
    else
      check(cudaMemcpy(p->x, host, elements * element, cudaMemcpyHostToDevice));
    check(cudaDeviceSynchronize());
  } catch (const std::exception& e) {
    p->error = e.what();
  }
  return p->error.empty() ? 0 : -1;
}
extern "C" void raw_math_destroy(void* handle) { delete static_cast<Probe*>(handle); }
