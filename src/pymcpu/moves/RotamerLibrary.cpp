#include "pymcpu/moves/RotamerLibrary.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace mcpu {

namespace {
constexpr float kHalfLog2Pi = 0.9189385332046727f; // 0.5 * log(2*pi)
constexpr double kTwoTo32 = 4294967296.0;
} // namespace

void RotamerLibrary::add_residue_type(int amino_idx,
                                       const std::vector<float>& weights,
                                       const std::vector<std::array<float, 4>>& means,
                                       const std::vector<std::array<float, 4>>& sigmas) {
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes) {
        throw std::out_of_range("RotamerLibrary::add_residue_type: amino_idx out of range");
    }
    if (has_truncated_tables_) {
        throw std::logic_error(
            "RotamerLibrary::add_residue_type: this library holds truncated tables; "
            "row tables and truncated tables do not mix");
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
    const int n_truncated = truncated_tables_[static_cast<size_t>(amino_idx)].n_rows;
    if (n_truncated > 0) return n_truncated;
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

void RotamerLibrary::add_truncated_table(int amino_idx, const std::vector<double>& weights,
                                         const std::vector<double>& means,
                                         const std::vector<double>& sigmas,
                                         const std::vector<double>& periods) {
    const auto fail = [](const std::string& why) {
        throw std::invalid_argument("RotamerLibrary::add_truncated_table: " + why);
    };
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes) {
        throw std::out_of_range("RotamerLibrary::add_truncated_table: amino_idx out of range");
    }
    for (const auto& rows : components_per_amino_) {
        if (!rows.empty()) {
            throw std::logic_error(
                "RotamerLibrary::add_truncated_table: this library holds row tables; "
                "row tables and truncated tables do not mix");
        }
    }
    const int k_rows = static_cast<int>(weights.size());
    const int n_chi = static_cast<int>(periods.size());
    if (k_rows < 1 || k_rows > kMaxTableRows) fail("need 1 to 256 rows");
    if (n_chi < 1 || n_chi > 4) fail("need 1 to 4 chi angles");
    const size_t k = static_cast<size_t>(k_rows);
    const size_t n = static_cast<size_t>(n_chi);
    if (means.size() != k * n || sigmas.size() != k * n) {
        fail("means and sigmas must be [row][chi], with one weight per row");
    }

    TruncatedTable table;
    table.n_rows = k_rows;
    table.n_chi = n_chi;
    for (size_t i = 0; i < n; ++i) {
        const double period = periods[i];
        const bool pi_period = std::fabs(period - PI) < 1e-9;
        if (!pi_period && std::fabs(period - 2.0 * PI) >= 1e-9) fail("a period must be pi or 2 pi");
        table.period[i] = pi_period ? PI_F : 2.0f * PI_F;
        table.inv_period[i] = 1.0f / table.period[i];
        table.half_window[i] = 0.5f * table.period[i];
        table.symmetric[i] = pi_period;
    }
    table.block.resize(k * (1 + 2 * n));
    table.thresholds.resize(k - 1);

    double total = 0.0;
    for (size_t r = 0; r < k; ++r) {
        if (!(weights[r] > 0.0) || !std::isfinite(weights[r])) {
            fail("every weight must be positive and finite");
        }
        total += weights[r];
    }
    std::vector<int> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&weights](int a, int b) { return weights[a] > weights[b]; });

    double cumulative = 0.0;
    std::uint64_t previous = 0;
    for (size_t j = 0; j < k; ++j) {
        const size_t r = static_cast<size_t>(order[j]);
        cumulative += weights[r] / total;
        const std::uint64_t threshold = j + 1 < k
            ? static_cast<std::uint64_t>(std::llround(cumulative * kTwoTo32))
            : static_cast<std::uint64_t>(kTwoTo32);
        if (threshold <= previous ||
            (j + 1 < k && threshold >= static_cast<std::uint64_t>(kTwoTo32))) {
            fail("a weight is too small to be drawn with 32-bit thresholds");
        }
        if (j + 1 < k) table.thresholds[j] = static_cast<std::uint32_t>(threshold);
        double c = std::log(static_cast<double>(threshold - previous) / kTwoTo32);
        previous = threshold;
        for (size_t i = 0; i < n; ++i) {
            const double mean = means[r * n + i];
            const double sigma = sigmas[r * n + i];
            if (!std::isfinite(mean) || !(sigma > 0.0) || !std::isfinite(sigma)) {
                fail("means must be finite and sigmas positive");
            }
            const float inv_sigma = static_cast<float>(1.0 / sigma);
            // The draw keeps |z| <= half_window * inv_sigma, computed in float
            // exactly like this; Z is the normal mass inside it.
            const float z_max = table.half_window[i] * inv_sigma;
            const double log_z = std::log(std::erf(static_cast<double>(z_max) / std::sqrt(2.0)));
            c -= -std::log(static_cast<double>(inv_sigma)) + log_z + 0.5 * std::log(2.0 * PI) +
                 (table.symmetric[i] ? std::log(2.0) : 0.0);
            table.block[k * (1 + 2 * i) + j] = wrap_angle_to_pi(static_cast<float>(mean));
            table.block[k * (2 + 2 * i) + j] = inv_sigma;
        }
        table.block[j] = static_cast<float>(c);
    }
    truncated_tables_[static_cast<size_t>(amino_idx)] = std::move(table);
    has_truncated_tables_ = true;
}

const RotamerLibrary::TruncatedTable& RotamerLibrary::truncated_table(int amino_idx) const {
    if (amino_idx < 0 || amino_idx >= kNumAminoTypes ||
        truncated_tables_[static_cast<size_t>(amino_idx)].n_rows == 0) {
        throw std::out_of_range("RotamerLibrary: no truncated table for this amino_idx");
    }
    return truncated_tables_[static_cast<size_t>(amino_idx)];
}

float RotamerLibrary::log_density(int amino_idx, const std::array<float, 4>& chi) const {
    const TruncatedTable& table = truncated_table(amino_idx);
    const int k_rows = table.n_rows;
    const float* block = table.block.data();

    // Structure of arrays, so the per-chi loops over rows vectorise; nearbyint
    // compiles to one rounding instruction (ties go to even: the antipode,
    // where both images give the same density).
    alignas(32) std::array<float, kMaxTableRows> term;
    std::copy(block, block + k_rows, term.begin());
    for (int i = 0; i < table.n_chi; ++i) {
        const float* mu = block + k_rows * (1 + 2 * i);
        const float* inv_sigma = mu + k_rows;
        const float x = chi[static_cast<size_t>(i)];
        const float period = table.period[static_cast<size_t>(i)];
        const float inv_period = table.inv_period[static_cast<size_t>(i)];
        for (int k = 0; k < k_rows; ++k) {
            float d = x - mu[k];
            d -= period * std::nearbyint(d * inv_period);
            const float z = d * inv_sigma[k];
            term[static_cast<size_t>(k)] -= 0.5f * z * z;
        }
    }
    const float max_term = *std::max_element(term.begin(), term.begin() + k_rows);
    // Rows more than 24 nats below the largest add under 4e-11 each; with at
    // most 256 rows that stays below float rounding of the sum.
    float sum = 0.0f;
    for (int k = 0; k < k_rows; ++k) {
        const float e = term[static_cast<size_t>(k)] - max_term;
        if (e > -24.0f) sum += std::exp(e);
    }
    return max_term + std::log(sum);
}

int RotamerLibrary::sample(int amino_idx, std::mt19937& rng,
                           std::normal_distribution<float>& unit_normal,
                           std::array<float, 4>& chi) const {
    const TruncatedTable& table = truncated_table(amino_idx);
    const int k_rows = table.n_rows;
    const float* block = table.block.data();

    const auto x = static_cast<std::uint32_t>(rng());
    int k = 0;
    while (k < k_rows - 1 && x >= table.thresholds[static_cast<size_t>(k)]) ++k;

    for (int i = 0; i < table.n_chi; ++i) {
        const size_t ui = static_cast<size_t>(i);
        const float mu = block[k_rows * (1 + 2 * i) + k];
        const float inv_sigma = block[k_rows * (2 + 2 * i) + k];
        const float z_max = table.half_window[ui] * inv_sigma;
        float z = unit_normal(rng);
        while (std::fabs(z) > z_max) z = unit_normal(rng);
        float target = mu + z / inv_sigma;
        if (table.symmetric[ui] && (rng() & 1u)) target += PI_F;
        chi[ui] = wrap_angle_to_pi(target);
    }
    return k;
}

RotamerLibrary::TableRow RotamerLibrary::table_row(int amino_idx, int k) const {
    const TruncatedTable& table = truncated_table(amino_idx);
    if (k < 0 || k >= table.n_rows) throw std::out_of_range("RotamerLibrary::table_row: k out of range");
    const size_t uk = static_cast<size_t>(k);
    const size_t k_rows = static_cast<size_t>(table.n_rows);
    const double upper = uk + 1 < k_rows ? table.thresholds[uk] : kTwoTo32;
    const double lower = uk > 0 ? table.thresholds[uk - 1] : 0.0;
    TableRow row;
    row.weight = (upper - lower) / kTwoTo32;
    for (size_t i = 0; i < static_cast<size_t>(table.n_chi); ++i) {
        row.mean[i] = table.block[k_rows * (1 + 2 * i) + uk];
        row.sigma[i] = 1.0f / table.block[k_rows * (2 + 2 * i) + uk];
    }
    return row;
}

} // namespace mcpu
