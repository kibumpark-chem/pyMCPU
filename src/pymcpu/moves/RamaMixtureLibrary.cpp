#include "pymcpu/moves/RamaMixtureLibrary.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace mcpu {

namespace {
constexpr float kLog2Pi = 1.8378770664093453f;  // log(2*pi)

/// Numerically stable log-sum-exp over a small vector (K components or
/// (2*n_wrap+1)^2 periodic images -- both are small, no need for anything
/// fancier than max-subtract).
float logsumexp(const std::vector<float>& terms) {
    const float max_term = *std::max_element(terms.begin(), terms.end());
    if (!std::isfinite(max_term)) return max_term;  // all -inf (degenerate/unreachable)
    float sum_exp = 0.0f;
    for (float t : terms) sum_exp += std::exp(t - max_term);
    return max_term + std::log(sum_exp);
}
} // namespace

void RamaMixtureLibrary::ensure_capacity(int amino_idx) {
    if (amino_idx < 0) {
        throw std::out_of_range("RamaMixtureLibrary: amino_idx must be >= 0");
    }
    if (static_cast<size_t>(amino_idx) >= components_per_amino_.size()) {
        components_per_amino_.resize(static_cast<size_t>(amino_idx) + 1);
        cumulative_weight_per_amino_.resize(static_cast<size_t>(amino_idx) + 1);
    }
}

void RamaMixtureLibrary::add_residue_type(int amino_idx,
                                           const std::vector<float>& weights,
                                           const std::vector<std::array<float, 2>>& means,
                                           const std::vector<std::array<float, 3>>& covariances) {
    ensure_capacity(amino_idx);
    const size_t k = weights.size();
    if (k == 0) {
        throw std::invalid_argument("RamaMixtureLibrary::add_residue_type: no mixture rows given");
    }
    if (means.size() != k || covariances.size() != k) {
        throw std::invalid_argument(
            "RamaMixtureLibrary::add_residue_type: weights/means/covariances size mismatch");
    }

    std::vector<RamaComponent> components(k);
    std::vector<float> cumulative(k);
    float running = 0.0f;
    for (size_t i = 0; i < k; ++i) {
        RamaComponent& c = components[i];
        c.log_weight = std::log(weights[i]);
        c.mean = means[i];
        c.cov = covariances[i];

        const float c11 = c.cov[0], c12 = c.cov[1], c22 = c.cov[2];
        const float det = c11 * c22 - c12 * c12;
        if (!(c11 > 0.0f) || !(det > 0.0f)) {
            throw std::invalid_argument(
                "RamaMixtureLibrary::add_residue_type: covariance row " + std::to_string(i) +
                " for amino_idx " + std::to_string(amino_idx) +
                " is not symmetric positive-definite (c11=" + std::to_string(c11) +
                ", det=" + std::to_string(det) + ")");
        }

        // Cholesky (for sampling): cov = L L^T, L = [[l11, 0], [l21, l22]].
        c.chol_l11 = std::sqrt(c11);
        c.chol_l21 = c12 / c.chol_l11;
        c.chol_l22 = std::sqrt(c22 - c.chol_l21 * c.chol_l21);

        // Inverse + log-det (for density).
        c.inv_c11 = c22 / det;
        c.inv_c22 = c11 / det;
        c.inv_c12 = -c12 / det;
        c.log_det = std::log(det);

        running += weights[i];
        cumulative[i] = running;
    }
    components_per_amino_[static_cast<size_t>(amino_idx)] = std::move(components);
    cumulative_weight_per_amino_[static_cast<size_t>(amino_idx)] = std::move(cumulative);
}

int RamaMixtureLibrary::num_rows(int amino_idx) const {
    if (amino_idx < 0 || static_cast<size_t>(amino_idx) >= components_per_amino_.size()) return 0;
    return static_cast<int>(components_per_amino_[static_cast<size_t>(amino_idx)].size());
}

int RamaMixtureLibrary::sample_row(int amino_idx, float u01) const {
    if (amino_idx < 0 || static_cast<size_t>(amino_idx) >= cumulative_weight_per_amino_.size()) {
        return -1;
    }
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

const RamaComponent& RamaMixtureLibrary::row(int amino_idx, int row_idx) const {
    if (amino_idx < 0 || static_cast<size_t>(amino_idx) >= components_per_amino_.size()) {
        throw std::out_of_range("RamaMixtureLibrary::row: amino_idx out of range");
    }
    const auto& components = components_per_amino_[static_cast<size_t>(amino_idx)];
    if (row_idx < 0 || static_cast<size_t>(row_idx) >= components.size()) {
        throw std::out_of_range("RamaMixtureLibrary::row: row_idx out of range");
    }
    return components[static_cast<size_t>(row_idx)];
}

float RamaMixtureLibrary::log_mixture_density(int amino_idx,
                                               const std::array<float, 2>& phi_psi) const {
    if (amino_idx < 0 || static_cast<size_t>(amino_idx) >= components_per_amino_.size()) {
        return -std::numeric_limits<float>::infinity();
    }
    const auto& components = components_per_amino_[static_cast<size_t>(amino_idx)];
    if (components.empty()) {
        return -std::numeric_limits<float>::infinity();
    }

    std::vector<float> log_terms(components.size());
    std::vector<float> shift_terms(shifts_.size());
    for (size_t k = 0; k < components.size(); ++k) {
        const RamaComponent& c = components[k];
        for (size_t s = 0; s < shifts_.size(); ++s) {
            const float d0 = phi_psi[0] - (c.mean[0] + shifts_[s][0]);
            const float d1 = phi_psi[1] - (c.mean[1] + shifts_[s][1]);
            const float quad = c.inv_c11 * d0 * d0 + 2.0f * c.inv_c12 * d0 * d1 +
                                c.inv_c22 * d1 * d1;
            shift_terms[s] = -0.5f * (quad + c.log_det + 2.0f * kLog2Pi);
        }
        log_terms[k] = c.log_weight + logsumexp(shift_terms);
    }
    return logsumexp(log_terms);
}

} // namespace mcpu
