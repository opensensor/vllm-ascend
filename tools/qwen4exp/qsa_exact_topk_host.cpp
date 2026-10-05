// SPDX-License-Identifier: Apache-2.0
// Host parity and feasibility harness for the experimental AI CPU selector.
#include "../../csrc/attention/qsa_exact_topk_aicpu_v310/exact_topk.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

extern "C" int qsa_exact_topk_host(const float* scores, int64_t queries, int64_t groups, int64_t topk,
                                   int32_t* indices) {
  return qsa_exact_topk::SelectRows(scores, queries, groups, topk, indices) ? 0 : 1;
}

extern "C" int qsa_exact_topk_shape_host(int64_t score_elements, int64_t index_elements, int64_t topk, int64_t* queries,
                                         int64_t* groups) {
  qsa_exact_topk::SelectionShape shape{};
  if (queries == nullptr || groups == nullptr ||
      !qsa_exact_topk::ResolveSelectionShape(score_elements, index_elements, topk, &shape)) {
    return 1;
  }
  *queries = shape.queries;
  *groups = shape.groups;
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: qsa_exact_topk_host QUERIES GROUPS TOPK REPEATS THREADS random|ties|masked\n";
    return 2;
  }
  const auto queries = std::stoll(argv[1]);
  const auto groups = std::stoll(argv[2]);
  const auto topk = std::stoll(argv[3]);
  const auto repeats = std::stoll(argv[4]);
  const auto threads = std::stoll(argv[5]);
  const std::string pattern = argv[6];
  if (queries <= 0 || groups <= 0 || topk <= 0 || topk > groups || repeats <= 0 || threads <= 0 || threads > queries ||
      (pattern != "random" && pattern != "ties" && pattern != "masked")) {
    std::cerr << "invalid benchmark parameters\n";
    return 2;
  }
  std::mt19937 rng(1024);
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  std::vector<float> scores(static_cast<size_t>(queries * groups));
  std::vector<int32_t> indices(static_cast<size_t>(queries * topk));
  for (int64_t row = 0; row < queries; ++row) {
    for (int64_t column = 0; column < groups; ++column) {
      auto& score = scores[static_cast<size_t>(row * groups + column)];
      score = pattern == "ties" ? static_cast<float>(rng() % 16) : distribution(rng);
      if (pattern == "masked" && column >= groups / 2) {
        score = -std::numeric_limits<float>::infinity();
      }
    }
  }
  const auto select = [&]() {
    std::vector<std::thread> workers;
    for (int64_t worker = 0; worker < threads; ++worker) {
      const auto first = queries * worker / threads;
      const auto last = queries * (worker + 1) / threads;
      workers.emplace_back([&, first, last]() {
        qsa_exact_topk::SelectRows(scores.data() + first * groups, last - first, groups, topk,
                                   indices.data() + first * topk);
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
  };
  select();
  std::vector<double> trials_ms;
  trials_ms.reserve(static_cast<size_t>(repeats));
  for (int64_t trial = 0; trial < repeats; ++trial) {
    const auto start = std::chrono::steady_clock::now();
    select();
    const auto stop = std::chrono::steady_clock::now();
    trials_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
  }
  std::sort(trials_ms.begin(), trials_ms.end());
  uint64_t checksum = 0;
  for (auto index : indices) {
    checksum += static_cast<uint32_t>(index);
  }
  std::cout << "{\"queries\":" << queries << ",\"groups\":" << groups << ",\"topk\":" << topk
            << ",\"threads\":" << threads << ",\"pattern\":\"" << pattern
            << "\",\"median_ms\":" << trials_ms[static_cast<size_t>(repeats / 2)] << ",\"checksum\":" << checksum
            << "}\n";
  return 0;
}
