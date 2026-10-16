// SPDX-License-Identifier: Apache-2.0
#ifndef SIDECARD_EXECUTION_META_H
#define SIDECARD_EXECUTION_META_H
#include <ATen/ATen.h>
#include <cmath>
#include <tuple>

namespace sidecard_execution {

inline void CheckTensor(const at::Tensor& input, const at::Tensor& owner, at::ScalarType dtype) {
  TORCH_CHECK(input.device() == owner.device() && input.scalar_type() == dtype && input.is_contiguous(),
              "sidecard execution inputs require matching device, dtype and contiguous layout");
}

inline at::Tensor KdaMeta(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                          const at::Tensor& gate, const at::Tensor& beta, at::Tensor& state) {
  TORCH_CHECK(query.dim() == 3 && query.size(0) == 1 && query.size(2) == 128, "sidecard KDA requires [1,H,128]");
  const auto heads = query.size(1);
  TORCH_CHECK(heads >= 8 && heads <= 128 && heads % 8 == 0, "unsupported sidecard KDA head count");
  for (const auto* input : {&query, &key, &value}) {
    CheckTensor(*input, query, at::kHalf);
    TORCH_CHECK(input->sizes() == query.sizes(), "sidecard KDA q/k/v shape mismatch");
  }
  CheckTensor(gate, query, at::kFloat);
  CheckTensor(beta, query, at::kFloat);
  CheckTensor(state, query, at::kFloat);
  TORCH_CHECK(gate.sizes() == query.sizes() && beta.dim() == 2 && beta.size(0) == 1 && beta.size(1) == heads &&
                  state.dim() == 3 && state.size(0) == heads && state.size(1) == 128 && state.size(2) == 128,
              "sidecard KDA gate/beta/carry shape mismatch");
  // The schema, not a dummy data write, describes the in-place FP32 state
  // mutation. Meta retains its shape/dtype and returns an independent FP16
  // output. Only the real device adapter checks read/write data-pointer aliasing.
  return at::empty_like(value);
}

inline std::tuple<at::Tensor, at::Tensor, at::Tensor> PreMeta(const at::Tensor& residual, const at::Tensor& fn,
                                                              const at::Tensor& scale, const at::Tensor& base,
                                                              const at::Tensor& normWeight, double normEps,
                                                              double hcEps) {
  CheckTensor(residual, residual, at::kFloat);
  TORCH_CHECK(residual.dim() == 3 && residual.size(0) >= 1 && residual.size(0) <= 32 && residual.size(1) == 4 &&
                  residual.size(2) == 4096,
              "sidecard mHC pre requires residual [1..32,4,4096]");
  for (const auto* input : {&fn, &scale, &base}) {
    CheckTensor(*input, residual, at::kFloat);
  }
  CheckTensor(normWeight, residual, at::kHalf);
  TORCH_CHECK(fn.dim() == 2 && fn.size(0) == 24 && fn.size(1) == 16384 && scale.dim() == 1 && scale.size(0) == 3 &&
                  base.dim() == 1 && base.size(0) == 24 && normWeight.dim() == 1 && normWeight.size(0) == 4096,
              "sidecard mHC coefficient or norm shape mismatch");
  TORCH_CHECK(std::isfinite(normEps) && normEps > 0 && std::isfinite(hcEps) && hcEps > 0 &&
                  std::isfinite(static_cast<float>(normEps)) && static_cast<float>(normEps) > 0 &&
                  std::isfinite(static_cast<float>(hcEps)) && static_cast<float>(hcEps) > 0,
              "sidecard mHC epsilons must be finite positive FP32");
  const auto rows = residual.size(0);
  return {at::empty({rows, 4, 1}, residual.options()), at::empty({rows, 4, 4}, residual.options()),
          at::empty({rows, 4096}, normWeight.options())};
}

inline at::Tensor PostMeta(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& post,
                           const at::Tensor& comb) {
  CheckTensor(x, x, at::kHalf);
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 32 && x.size(1) == 4096,
              "sidecard mHC post requires x [1..32,4096]");
  for (const auto* input : {&residual, &post, &comb}) {
    CheckTensor(*input, x, at::kFloat);
    TORCH_CHECK(input->dim() == 3 && input->size(0) == x.size(0) && input->size(1) == 4,
                "sidecard mHC residual or mixer rank mismatch");
  }
  TORCH_CHECK(residual.size(2) == 4096 && post.size(2) == 1 && comb.size(2) == 4,
              "sidecard mHC residual or mixer shape mismatch");
  return at::empty_like(residual);
}
}  // namespace sidecard_execution
#endif
