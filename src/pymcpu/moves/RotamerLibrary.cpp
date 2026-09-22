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

    std::vector<float> log_terms(components.size());
    for (size_t k = 0; k < components.size(); ++k) {
        const RotamerComponent& c = components[k];
        float log_p = c.log_weight;
        for (int i = 0; i < nt; ++i) {
            const float d = wrap_angle_to_pi(
                chi[static_cast<size_t>(i)] - c.mean[static_cast<size_t>(i)]);
            const float s = c.sigma[static_cast<size_t>(i)];
            log_p += -0.5f * (d / s) * (d / s) - std::log(s) - kHalfLog2Pi;
        }
        log_terms[k] = log_p;
    }

    const float max_log = *std::max_element(log_terms.begin(), log_terms.end());
    if (!std::isfinite(max_log)) return max_log; // all components degenerate/unreachable
    float sum_exp = 0.0f;
    for (float lt : log_terms) sum_exp += std::exp(lt - max_log);
    return max_log + std::log(sum_exp);
}

} // namespace mcpu
