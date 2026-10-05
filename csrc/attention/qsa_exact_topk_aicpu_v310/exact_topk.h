// SPDX-License-Identifier: Apache-2.0
// Experimental Qwen3.8 QSA selector shared by the host gate and AI CPU kernel.
#ifndef QSA_EXACT_TOPK_AICPU_V310_EXACT_TOPK_H
#define QSA_EXACT_TOPK_AICPU_V310_EXACT_TOPK_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace qsa_exact_topk {

struct SelectionShape {
    int64_t queries;
    int64_t groups;
    int64_t topk;
};

// ACLNN AI CPU descriptors may flatten both tensors to one dimension. Use
// their element counts and the explicit top_k attribute to recover the
// logical [queries, groups] and [queries, topk] shapes.
inline bool ResolveSelectionShape(int64_t score_elements, int64_t index_elements,
                                  int64_t topk, SelectionShape* shape) {
    if (shape == nullptr || topk <= 0 || index_elements <= 0 ||
        index_elements % topk != 0) {
        return false;
    }
    const auto queries = index_elements / topk;
    if (score_elements <= 0 || score_elements % queries != 0) {
        return false;
    }
    const auto groups = score_elements / queries;
    if (groups < topk || groups > std::numeric_limits<int32_t>::max()) {
        return false;
    }
    *shape = {queries, groups, topk};
    return true;
}

// Scores are the already-masked FP32 [queries, groups] tensor. The order is
// score descending, then group ID ascending. This matches stable descending
// argsort on the original group order, including ties at the K boundary.
inline bool SelectRows(const float* scores, int64_t queries, int64_t groups,
                       int64_t topk, int32_t* indices) {
    if (scores == nullptr || indices == nullptr || queries < 0 || groups <= 0 ||
        groups > std::numeric_limits<int32_t>::max() ||
        topk <= 0 || topk > groups) {
        return false;
    }
    std::vector<int32_t> order(static_cast<size_t>(groups));
    for (int64_t row = 0; row < queries; ++row) {
        const float* row_scores = scores + row * groups;
        for (int64_t group = 0; group < groups; ++group) {
            if (std::isnan(row_scores[group])) {
                return false;
            }
        }
        std::iota(order.begin(), order.end(), 0);
        const auto better = [row_scores](int32_t lhs, int32_t rhs) {
            if (row_scores[lhs] != row_scores[rhs]) {
                return row_scores[lhs] > row_scores[rhs];
            }
            return lhs < rhs;
        };
        if (topk < groups) {
            std::nth_element(order.begin(), order.begin() + topk, order.end(), better);
        }
        std::sort(order.begin(), order.begin() + topk, better);
        std::copy_n(order.begin(), topk, indices + row * topk);
    }
    return true;
}

}  // namespace qsa_exact_topk

#endif  // QSA_EXACT_TOPK_AICPU_V310_EXACT_TOPK_H
