// SPDX-License-Identifier: Apache-2.0
// Resident arithmetic probes, independent of the serving daemon.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdexcept>
#include <string>

namespace {
constexpr int THREADS = 256;
constexpr int HEAD_DIM = 256;
constexpr int STREAMS = 4;
constexpr int MAX_ROWS = 512;
constexpr int MAX_CHANNELS = 154880;
constexpr int MAX_WIDTH = 16384;
constexpr int HIDDEN_WIDTH = 4096;
constexpr int MAX_ATTENTION_HEADS = 64;
constexpr int MAX_SELECTED_KV = 2048;
constexpr int COPY_BYTES = 256 * 1024 * 1024;

void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void check_blas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status " + std::to_string(status));
}

struct Probe {
  int op, m, n, k;
  bool fp32;
  cublasHandle_t blas = nullptr;
  void *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr, *out = nullptr;
  float* scores = nullptr;
  __half* probabilities = nullptr;
  size_t output_bytes = 0;
  std::string error;
  ~Probe() {
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d);
    cudaFree(out);
    cudaFree(scores);
    cudaFree(probabilities);
    if (blas) cublasDestroy(blas);
  }
};

void upload(void** target, const void* host, size_t size) {
  check(cudaMalloc(target, size));
  check(cudaMemcpy(*target, host, size, cudaMemcpyHostToDevice));
}

__global__ void softmax(const float* x, __half* p, int rows, int cols) {
  extern __shared__ float reductions[];
  int row = blockIdx.x;
  if (row >= rows) return;
  float maximum = -CUDART_INF_F;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) maximum = fmaxf(maximum, x[size_t(row) * cols + col]);
  reductions[threadIdx.x] = maximum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      reductions[threadIdx.x] = fmaxf(reductions[threadIdx.x], reductions[threadIdx.x + stride]);
    __syncthreads();
  }
  maximum = reductions[0];
  float total = 0;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) total += expf(x[size_t(row) * cols + col] - maximum);
  reductions[threadIdx.x] = total;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    __syncthreads();
  }
  total = reductions[0];
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    p[size_t(row) * cols + col] = __float2half(expf(x[size_t(row) * cols + col] - maximum) / total);
}

__global__ void mhc_post(const __half* x, const float* residual, const float* post, const float* comb, float* out,
                         size_t elements, int width) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= elements) return;
  int h = i % width;
  int stream = (i / width) % STREAMS;
  size_t token = i / (STREAMS * width);
  // Disable FMA at build time so the reference's separate additions match.
  float value = 0;
  for (int input = 0; input < STREAMS; ++input)
    value +=
        residual[(token * STREAMS + input) * width + h] * comb[token * STREAMS * STREAMS + input * STREAMS + stream];
  out[i] = value + post[token * STREAMS + stream] * __half2float(x[token * width + h]);
}

__global__ void cache_sweep(const unsigned int* input, unsigned int* output, size_t words) {
  size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < words) output[index] = __ldcg(input + index);
}

void run(Probe* p) {
  const float one = 1, zero = 0;
  if (p->op == 0) {
    auto type = p->fp32 ? CUDA_R_32F : CUDA_R_16F;
    auto compute = p->fp32 ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
    // Row-major output = input @ weight.T, through column-major cuBLAS.
    check_blas(cublasGemmEx(p->blas, CUBLAS_OP_T, CUBLAS_OP_N, p->n, p->m, p->k, &one, p->b, type, p->k, p->a, type,
                            p->k, &zero, p->out, type, p->n, compute, CUBLAS_GEMM_DEFAULT));
  } else if (p->op == 1) {
    float scale = 1.0f / sqrtf(float(HEAD_DIM));
    check_blas(cublasGemmStridedBatchedEx(p->blas, CUBLAS_OP_T, CUBLAS_OP_N, p->k, p->m, HEAD_DIM, &scale, p->b,
                                          CUDA_R_16F, HEAD_DIM, size_t(p->k) * HEAD_DIM, p->a, CUDA_R_16F, HEAD_DIM,
                                          size_t(p->m) * HEAD_DIM, &zero, p->scores, CUDA_R_32F, p->k,
                                          size_t(p->m) * p->k, p->n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    softmax<<<p->n * p->m, THREADS, THREADS * sizeof(float)>>>(p->scores, p->probabilities, p->n * p->m, p->k);
    check_blas(cublasGemmStridedBatchedEx(p->blas, CUBLAS_OP_N, CUBLAS_OP_N, HEAD_DIM, p->m, p->k, &one, p->c,
                                          CUDA_R_16F, HEAD_DIM, size_t(p->k) * HEAD_DIM, p->probabilities, CUDA_R_16F,
                                          p->k, size_t(p->m) * p->k, &zero, p->out, CUDA_R_16F, HEAD_DIM,
                                          size_t(p->m) * HEAD_DIM, p->n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  } else if (p->op == 2) {
    size_t elements = size_t(p->m) * STREAMS * p->k;
    mhc_post<<<(elements + THREADS - 1) / THREADS, THREADS>>>(static_cast<__half*>(p->a), static_cast<float*>(p->b),
                                                              static_cast<float*>(p->c), static_cast<float*>(p->d),
                                                              static_cast<float*>(p->out), elements, p->k);
  } else {
    check(cudaMemcpyAsync(p->out, p->a, p->output_bytes, cudaMemcpyDeviceToDevice));
  }
  check(cudaGetLastError());
}
}  // namespace

extern "C" void* architecture_create(int op, int m, int n, int k, int fp32, const void* a, const void* b, const void* c,
                                     const void* d) {
  auto* p = new Probe();
  p->op = op;
  p->m = m;
  p->n = n;
  p->k = k;
  p->fp32 = fp32;
  try {
    if (op < 0 || op > 3 || !a || (op < 3 && !b) || (op == 1 && !c) || (op == 2 && (!c || !d)))
      throw std::runtime_error("invalid operation or missing input pointer");
    if (op < 3 && (m < 1 || m > MAX_ROWS || n < 1 || n > MAX_CHANNELS || k < 1 || k > MAX_WIDTH))
      throw std::runtime_error("probe exceeds bounded GLM geometry");
    if ((op == 1 && (n > MAX_ATTENTION_HEADS || k > MAX_SELECTED_KV || fp32)) ||
        (op == 2 && (n != STREAMS || k != HIDDEN_WIDTH || !fp32)) || (op == 3 && (m != COPY_BYTES || n || k || fp32)))
      throw std::runtime_error("probe geometry differs from its arithmetic contract");
    check(cudaSetDevice(0));
    check_blas(cublasCreate(&p->blas));
    size_t element = fp32 ? sizeof(float) : sizeof(__half);
    if (op == 0) {
      upload(&p->a, a, size_t(m) * k * element);
      upload(&p->b, b, size_t(n) * k * element);
      p->output_bytes = size_t(m) * n * element;
    } else if (op == 1) {
      upload(&p->a, a, size_t(n) * m * HEAD_DIM * sizeof(__half));
      upload(&p->b, b, size_t(n) * k * HEAD_DIM * sizeof(__half));
      upload(&p->c, c, size_t(n) * k * HEAD_DIM * sizeof(__half));
      check(cudaMalloc(&p->scores, size_t(n) * m * k * sizeof(float)));
      check(cudaMalloc(&p->probabilities, size_t(n) * m * k * sizeof(__half)));
      p->output_bytes = size_t(n) * m * HEAD_DIM * sizeof(__half);
    } else if (op == 2) {
      upload(&p->a, a, size_t(m) * k * sizeof(__half));
      upload(&p->b, b, size_t(m) * STREAMS * k * sizeof(float));
      upload(&p->c, c, size_t(m) * STREAMS * sizeof(float));
      upload(&p->d, d, size_t(m) * STREAMS * STREAMS * sizeof(float));
      p->output_bytes = size_t(m) * STREAMS * k * sizeof(float);
    } else if (op == 3) {
      p->output_bytes = size_t(m);
      upload(&p->a, a, p->output_bytes);
    } else {
      throw std::runtime_error("unknown probe operation");
    }
    check(cudaMalloc(&p->out, p->output_bytes));
  } catch (const std::exception& e) {
    p->error = e.what();
  }
  return p;
}

extern "C" const char* architecture_error(void* handle) {
  if (!handle) return "null probe handle";
  return static_cast<Probe*>(handle)->error.c_str();
}

extern "C" int architecture_run(void* handle, float* milliseconds) {
  if (!handle || !milliseconds) return -1;
  auto* p = static_cast<Probe*>(handle);
  if (!p->error.empty()) return -1;
  cudaEvent_t start = nullptr, end = nullptr;
  try {
    check(cudaEventCreate(&start));
    check(cudaEventCreate(&end));
    check(cudaEventRecord(start));
    run(p);
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    check(cudaEventElapsedTime(milliseconds, start, end));
  } catch (const std::exception& e) {
    p->error = e.what();
  }
  if (start) cudaEventDestroy(start);
  if (end) cudaEventDestroy(end);
  return p->error.empty() ? 0 : -1;
}

extern "C" int architecture_read(void* handle, void* host) {
  if (!handle || !host) return -1;
  auto* p = static_cast<Probe*>(handle);
  if (!p->error.empty()) return -1;
  try {
    check(cudaMemcpy(host, p->out, p->output_bytes, cudaMemcpyDeviceToHost));
  } catch (const std::exception& e) {
    p->error = e.what();
  }
  return p->error.empty() ? 0 : -1;
}

extern "C" void architecture_destroy(void* handle) { delete static_cast<Probe*>(handle); }

extern "C" int architecture_flush(void* destination, const void* source, int bytes) {
  if (!destination || !source || bytes != COPY_BYTES) return int(cudaErrorInvalidValue);
  size_t words = size_t(bytes) / sizeof(unsigned int);
  cache_sweep<<<(words + THREADS - 1) / THREADS, THREADS>>>(static_cast<const unsigned int*>(source),
                                                            static_cast<unsigned int*>(destination), words);
  return int(cudaGetLastError());
}
