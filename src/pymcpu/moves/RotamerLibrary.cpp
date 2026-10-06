#include "pymcpu/moves/RotamerLibrary.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcpu {

namespace {
constexpr float kHalfLog2Pi = 0.9189385332046727f; // 0.5 * log(2*pi)
} // namespace

void RotamerLibrary::add_residue_type(int amino_idx,
                                       const std::vector<float>& weights,
                                       const std::vector<std::array<float, 4>>& means,
                                       const std::vector<std::array<float, 4>>& sigmas) {
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes) {
        throw std::out_of_range("RotamerLibrary::add_residue_type: amino_idx out of range");
    }
    const size_t k = weights.size();
    if (k == 0) {
        throw std::invalid_argument("RotamerLibrary::add_residue_type: no rotamer rows given");
    }
    if (means.size() != k || sigmas.size() != k) {
        throw std::invalid_argument(
            "RotamerLibrary::add_residue_type: weights/means/sigmas size mismatch");
    }

    std::vector<RotamerComponent> components(k);
    std::vector<float> cumulative(k);
    float running = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        components[i].log_weight = std::log(weights[i]);
        components[i].mean = means[i];
        components[i].sigma = sigmas[i];
        for (size_t c = 0; c < 4; ++c) {
            components[i].log_sigma[c] = std::log(sigmas[i][c]);
        }
        running += weights[i];
        cumulative[i] = running;
    }
    components_per_amino_[static_cast<size_t>(amino_idx)] = std::move(components);
    cumulative_weight_per_amino_[static_cast<size_t>(amino_idx)] = std::move(cumulative);
}

int RotamerLibrary::num_rows(int amino_idx) const {
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes) return 0;
    return static_cast<int>(components_per_amino_[static_cast<size_t>(amino_idx)].size());
}

int RotamerLibrary::sample_row(int amino_idx, float u01) const {
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes) return -1;
    const auto& cumulative = cumulative_weight_per_amino_[static_cast<size_t>(amino_idx)];
    if (cumulative.empty()) return -1;
    for (size_t k = 0; k < cumulative.size(); ++k) {
        if (u01 < cumulative[k]) return static_cast<int>(k);
    }
    // Float-rounding fallthrough (cumulative sum landed a hair under u01, or
    // u01 itself was drawn as exactly 1.0f) -- clamp to the last row rather
    // than reading out of bounds.
    return static_cast<int>(cumulative.size()) - 1;
}

const RotamerComponent& RotamerLibrary::row(int amino_idx, int row_idx) const {
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes) {
        throw std::out_of_range("RotamerLibrary::row: amino_idx out of range");
    }
    const auto& components = components_per_amino_[static_cast<size_t>(amino_idx)];
    if (row_idx < 0 || static_cast<size_t>(row_idx) >= components.size()) {
        throw std::out_of_range("RotamerLibrary::row: row_idx out of range");
    }
    return components[static_cast<size_t>(row_idx)];
}

float RotamerLibrary::log_mixture_density(int amino_idx, int ntorsions,
                                           const std::array<float, 4>& chi) const {
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes) {
        return -std::numeric_limits<float>::infinity();
    }
    const auto& components = components_per_amino_[static_cast<size_t>(amino_idx)];
    if (components.empty()) {
        return -std::numeric_limits<float>::infinity();
    }
    const int nt = std::clamp(ntorsions, 0, 4);

    // The per-component terms live on the stack for every library row count
    // seen in practice (bbind02 has at most 81 rows per type); larger tables
    // fall back to the heap. Same arithmetic in the same order either way.
    constexpr size_t kStackTerms = 128;
    std::array<float, kStackTerms> stack_terms;
    std::vector<float> heap_terms;
    float* log_terms = stack_terms.data();
    if (components.size() > kStackTerms) {
        heap_terms.resize(components.size());
        log_terms = heap_terms.data();
    }
    const size_t n = components.size();
    for (size_t k = 0; k < n; ++k) {
        const RotamerComponent& c = components[k];
        float log_p = c.log_weight;
        for (int i = 0; i < nt; ++i) {
            const size_t ui = static_cast<size_t>(i);
            const float d = wrap_angle_to_pi(chi[ui] - c.mean[ui]);
            const float s = c.sigma[ui];
            log_p += -0.5f * (d / s) * (d / s) - c.log_sigma[ui] - kHalfLog2Pi;
        }
        log_terms[k] = log_p;
    }

    const float max_log = *std::max_element(log_terms, log_terms + n);
    if (!std::isfinite(max_log)) return max_log; // all components degenerate/unreachable
    float sum_exp = 0.0f;
    for (size_t k = 0; k < n; ++k) sum_exp += std::exp(log_terms[k] - max_log);
    return max_log + std::log(sum_exp);
}

} // namespace mcpu
