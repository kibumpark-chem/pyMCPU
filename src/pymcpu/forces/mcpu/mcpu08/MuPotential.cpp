#include "pymcpu/forces/mcpu/mcpu08/MuPotential.h"
#include "pymcpu/State.h"
#include "pymcpu/Context.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/CellList.h"
#include "pymcpu/neighbor/OpenCellGrid.h"
#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/neighbor/NeighborFallback.h"
#include "pymcpu/neighbor/PairSearch.h"
#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/CoordView.h"
#include "pymcpu/AtomPermutation.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mcpu::forces::mcpu08 {

namespace {

using mcpu::neighbor::kSpanMaskSlack;

/// MCPU_CONTACT_LIST=0 turns the live contact list off; read once.
bool contact_list_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("MCPU_CONTACT_LIST");
        return !(e && e[0] == '0');
    }();
    return on;
}

/// The atoms the Mu pair loops visit, in increasing order: every atom but the
/// amide hydrogens. Iterating this list visits the same pairs in the same
/// order as testing System::is_amide_h_atom on both atoms of every pair.
void mu_pair_atoms(const System& sys, int n, std::vector<int>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        if (!sys.is_amide_h_atom(i)) out.push_back(i);
}

/// Calls visit(i, j, r2) for every pair i = atoms[a], j = atoms[b] (a < b)
/// whose r2 = cv.dist2(i, j) is not greater than cutoff_sq (NaN counts as
/// kept), in the order of the all-pairs loop over `atoms`, and stops at the
/// first visit that returns true (the result is then true).
///
/// The candidates come from a cell binning of these coordinates, cells wider
/// than the cutoff, so every pair within it lies in neighbouring cells; each
/// atom's later partners are collected in a bitmap and taken in increasing
/// index. Same pairs, same r2, same order as the O(n^2) loop, which is kept
/// for short lists and coordinates that are not finite. The cells start 1%
/// wider than the cutoff (dist2 rounds |dx| by far less) and widen further
/// when the box would need more than max(32768, 16 n) of them, so an
/// extended chain costs O(n) memory. Scratch is per call: full energies are
/// rare, and are also taken on states that are not accepted.
template <class Visit>
bool mu_for_each_near_pair(const CoordView& cv, const std::vector<int>& atoms,
                           float cutoff_sq, Visit&& visit) {
    const size_t n = atoms.size();
    auto all_pairs = [&]() {
        for (size_t a = 0; a < n; ++a) {
            const int i = atoms[a];
            for (size_t b = a + 1; b < n; ++b) {
                const int j = atoms[b];
                const float r2 = cv.dist2(i, j);
                if (r2 > cutoff_sq) continue;
                if (visit(i, j, r2)) return true;
            }
        }
        return false;
    };
    const double cutoff = std::sqrt(static_cast<double>(cutoff_sq));
    if (n < 128 || !(cutoff > 0.0) || !std::isfinite(cutoff)) return all_pairs();
    double lo[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL};
    double hi[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
    for (const int i : atoms) {
        const double p[3] = {cv.x(i), cv.y(i), cv.z(i)};
        for (int d = 0; d < 3; ++d) {
            if (!std::isfinite(p[d])) return all_pairs();
            lo[d] = std::min(lo[d], p[d]);
            hi[d] = std::max(hi[d], p[d]);
        }
    }
    const double max_cells = std::max(32768.0, 16.0 * static_cast<double>(n));
    double cell = cutoff * 1.01 + 1e-3;
    double ncd[3];
    for (;;) {
        double total = 1.0;
        for (int d = 0; d < 3; ++d) {
            ncd[d] = std::floor((hi[d] - lo[d]) / cell) + 1.0;
            total *= ncd[d];
        }
        if (total <= max_cells) break;
        cell *= std::cbrt(total / max_cells) * 1.01;
    }
    const int nc[3] = {static_cast<int>(ncd[0]), static_cast<int>(ncd[1]),
                       static_cast<int>(ncd[2])};
    const size_t n_cells = static_cast<size_t>(nc[0]) * static_cast<size_t>(nc[1]) *
                           static_cast<size_t>(nc[2]);
    auto bin = [&](int d, double v) {
        return std::min(static_cast<int>((v - lo[d]) / cell), nc[d] - 1);
    };
    // Atoms by cell, each cell's list in increasing list position.
    std::vector<int> cell_of(n), start(n_cells + 1, 0), sorted(n);
    for (size_t a = 0; a < n; ++a) {
        const int i = atoms[a];
        cell_of[a] = (bin(0, cv.x(i)) * nc[1] + bin(1, cv.y(i))) * nc[2] + bin(2, cv.z(i));
        ++start[static_cast<size_t>(cell_of[a]) + 1];
    }
    for (size_t c = 0; c < n_cells; ++c) start[c + 1] += start[c];
    {
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (size_t a = 0; a < n; ++a)
            sorted[static_cast<size_t>(fill[static_cast<size_t>(cell_of[a])]++)] =
                static_cast<int>(a);
    }
    std::vector<uint64_t> bits((n + 63) / 64, 0);
    for (size_t a = 0; a + 1 < n; ++a) {
        const int c = cell_of[a];
        const int cz = c % nc[2], cy = (c / nc[2]) % nc[1], cx = c / (nc[1] * nc[2]);
        size_t w_hi = (a + 1) >> 6;
        for (int x = std::max(cx - 1, 0); x <= std::min(cx + 1, nc[0] - 1); ++x)
            for (int y = std::max(cy - 1, 0); y <= std::min(cy + 1, nc[1] - 1); ++y)
                for (int z = std::max(cz - 1, 0); z <= std::min(cz + 1, nc[2] - 1); ++z) {
                    const size_t cc = static_cast<size_t>((x * nc[1] + y) * nc[2] + z);
                    for (int k = start[cc + 1] - 1; k >= start[cc]; --k) {
                        const size_t b = static_cast<size_t>(sorted[static_cast<size_t>(k)]);
                        if (b <= a) break;
                        bits[b >> 6] |= uint64_t{1} << (b & 63);
                        w_hi = std::max(w_hi, b >> 6);
                    }
                }
        const int i = atoms[a];
        for (size_t w = (a + 1) >> 6; w <= w_hi; ++w) {
            uint64_t m = bits[w];
            bits[w] = 0;
            while (m) {
                const size_t b = (w << 6) + static_cast<size_t>(__builtin_ctzll(m));
                m &= m - 1;
                const int j = atoms[b];
                const float r2 = cv.dist2(i, j);
                if (r2 > cutoff_sq) continue;
                if (visit(i, j, r2)) return true;
            }
        }
    }
    return false;
}

}  // namespace

    static constexpr float kHardCorePenalty = 99999.0f;
    /// Absolute parameter sanity bound (Å²). Denselist uses mu_exact_cutoff_.
    static constexpr float kLegacyContactCutoffSq = 6.0f * 6.0f;
    static constexpr float kMuCutoffFallbackA = 6.0f;

    /// Same gate as Integrator::run occ_stencil dump (MCPU_VERBOSE).
    [[nodiscard]] static bool mcpu_verbose_enabled() noexcept {
        static const bool kVerbose = [] {
            const char* e = std::getenv("MCPU_VERBOSE");
            return e && e[0] && e[0] != '0';
        }();
        return kVerbose;
    }

    void MuPotential::apply_mu_denselist_cutoff() {
        float max_contact_r2 = 0.f;
        float max_hard_r2 = 0.f;
        if (!type_params_.empty()) {
            for (const auto& p : type_params_) {
                max_contact_r2 = std::max(max_contact_r2, p.contact_r2);
                max_hard_r2 = std::max(max_hard_r2, p.hard_r2);
            }
        } else {
            if (contact_dist_sq.size() > 0)
                max_contact_r2 = contact_dist_sq.maxCoeff();
            if (hard_core_sq.size() > 0)
                max_hard_r2 = hard_core_sq.maxCoeff();
        }
        // No pair overlaps beyond the largest hard-core radius (see
        // clash_prefilter_r2_): + 0.001 (rounding resolution) + 0.001 (float slack).
        if (max_hard_r2 > 0.f) {
            const float clash_r = std::sqrt(max_hard_r2) + 0.002f;
            clash_prefilter_r2_ = clash_r * clash_r;
        }
        // The near-miss band (see kContactBandA), widened to the next float so
        // that contact_r2 + w covers (contact_r + band)^2 for every type pair.
        if (max_contact_r2 > 0.f) {
            const double cmax = std::sqrt(static_cast<double>(max_contact_r2));
            const double band = static_cast<double>(kContactBandA);
            contact_band_w_r2_ = std::nextafter(
                static_cast<float>(2.0 * cmax * band + band * band),
                std::numeric_limits<float>::infinity());
        }
        const float max_r2 =
            std::max(max_contact_r2 + contact_band_w_r2_, max_hard_r2);
        float exact = kMuCutoffFallbackA;
        if (max_r2 <= 0.f) {
            std::fprintf(stderr,
                "WARN: Mu denselist cutoff: max contact/hard r²=0 — "
                "falling back to %.1f Å.\n",
                kMuCutoffFallbackA);
        } else {
            exact = std::sqrt(max_r2) * 1.0001f;
        }
        mu_exact_cutoff_ = exact;
        contact_cutoff_sq_ = exact * exact;
        // Print once type_params_ is authoritative (skip matrix-only ctor pass).
        if (!type_params_.empty() && mcpu_verbose_enabled()) {
            std::fprintf(stderr,
                "INFO: Mu denselist cutoff=%.6f Å "
                "(max_contact_r2=%.6f + band %.6f, max_hard_r2=%.6f).\n",
                mu_exact_cutoff_, max_contact_r2, contact_band_w_r2_, max_hard_r2);
        }
    }

    // The per-atom walk's skip_mask is 64-bit, indexed by `1ull << m` for a
    // cell slot m < CELL_CAPACITY, so the capacity must fit in it.
    static_assert(::mcpu::OpenCellGrid::CELL_CAPACITY <= 64,
                  "skip_mask is 64-bit; CELL_CAPACITY must fit");

    /// Report the atom pair that trips the hard-core sentinel in the full
    /// recompute (MCPU_CLASH_REPORT=1); see the call site in calculateEnergy.
    static bool clash_report_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("MCPU_CLASH_REPORT");
            return e && e[0] == '1';
        }();
        return on;
    }

    /// Count one Mu pair r2 evaluation (distance check + optional within-rcut).
    /// Increment sites (exactly one hot r2 path per backend):
    ///   1) Cell-grid enumeration  — calculateEnergyChange_fast (dense OpenCellGrid)
    ///   2) Fallback enumeration   — NeighborFallback moved-vs-all (+ explicit moved–moved)
    static inline void note_mu_pair_r2(::mcpu::NeighborStats& nstats, float r2,
                                       float cut2) noexcept {
        ++nstats.mu_num_pair_distance_checks;
        if (r2 <= cut2)
            ++nstats.mu_num_pairs_within_rcut;
    }

    MuPotential::MuPotential(
        Eigen::MatrixXf contact_energies,
        Eigen::MatrixXf contact_dist_sq,
        Eigen::MatrixXf hard_core_sq,
        std::vector<int> atom_types,
        std::vector<int> atom_to_residue
    ) : contact_energies(std::move(contact_energies)),
        contact_dist_sq(std::move(contact_dist_sq)),
        hard_core_sq(std::move(hard_core_sq)),
        atom_types(std::move(atom_types)),
        atom_to_residue(std::move(atom_to_residue)) {
        if (this->contact_dist_sq.size() > 0 &&
            this->contact_dist_sq.maxCoeff() > kLegacyContactCutoffSq) {
            throw std::invalid_argument(
                "MuPotential: contact distance exceeds the 6 A neighbor cutoff");
        }
        if (this->hard_core_sq.size() > 0 &&
            this->hard_core_sq.maxCoeff() > kLegacyContactCutoffSq) {
            throw std::invalid_argument(
                "MuPotential: hard-core distance exceeds the 6 A neighbor cutoff");
        }
        // Exact denselist cutoff from parameter matrices (refined after
        // type_params_ in cache_necessary_data). Default ON.
        apply_mu_denselist_cutoff();
    }

    void MuPotential::permute_atom_indices(const AtomPermutation& perm) {
        if (perm.is_identity()) return;
        const int n = perm.n_atoms();
        if (n != num_atoms_cached_ && num_atoms_cached_ != 0) {
            // Still allow permute before cache if sizes match atom_types.
        }
        if (static_cast<int>(atom_types.size()) != n) {
            throw std::runtime_error("MuPotential::permute_atom_indices: type size mismatch");
        }

        auto permute_vec_int = [&](std::vector<int>& v) {
            std::vector<int> tmp(static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                tmp[static_cast<size_t>(i)] = v[static_cast<size_t>(perm.int_to_ext[static_cast<size_t>(i)])];
            }
            v.swap(tmp);
        };
        permute_vec_int(atom_types);
        permute_vec_int(atom_to_residue);

        auto permute_mat = [&](Eigen::MatrixXf& m) {
            if (m.rows() != n || m.cols() != n) return;
            Eigen::MatrixXf tmp(n, n);
            for (int i = 0; i < n; ++i) {
                const int ei = perm.int_to_ext[static_cast<size_t>(i)];
                for (int j = 0; j < n; ++j) {
                    const int ej = perm.int_to_ext[static_cast<size_t>(j)];
                    tmp(i, j) = m(ei, ej);
                }
            }
            m.swap(tmp);
        };
        permute_mat(contact_energies);
        permute_mat(contact_dist_sq);
        permute_mat(hard_core_sq);

        auto permute_flat = [&](auto& v) {
            if (static_cast<int>(v.size()) != n * n) return;
            using T = typename std::decay_t<decltype(v)>::value_type;
            std::vector<T> tmp(static_cast<size_t>(n) * static_cast<size_t>(n));
            for (int i = 0; i < n; ++i) {
                const int ei = perm.int_to_ext[static_cast<size_t>(i)];
                for (int j = 0; j < n; ++j) {
                    const int ej = perm.int_to_ext[static_cast<size_t>(j)];
                    tmp[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                        v[static_cast<size_t>(ei) * static_cast<size_t>(n) + static_cast<size_t>(ej)];
                }
            }
            v.swap(tmp);
        };
        permute_flat(topo_contact_mask_);
        permute_flat(topo_clash_mask_);
        permute_flat(topo_flag_);
        num_atoms_cached_ = n;
        rebuild_type_params_from_matrices();
        build_compact_topo();
    }

    void MuPotential::build_compact_topo() {
        tf_ready_ = false;
        const int n = num_atoms_cached_;
        const size_t N = static_cast<size_t>(n);
        if (n < 2 || topo_flag_.size() != N * N || atom_to_residue.size() != N ||
            atom_types.size() != N) {
            return;
        }
        int max_t = -1;
        for (int t : atom_types) max_t = std::max(max_t, t);
        if (max_t + 2 > 256) return;
        const int ncls = max_t + 2;
        tf_ncls_ = ncls;
        tf_res_.assign(atom_to_residue.begin(), atom_to_residue.end());
        tf_cls_.resize(N);
        for (size_t i = 0; i < N; ++i)
            tf_cls_[i] = static_cast<uint8_t>(std::max(atom_types[i], -1) + 1);

        // Residue order: rank atoms by (residue, index).
        std::vector<int> order(N);
        for (int i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return tf_res_[static_cast<size_t>(a)] < tf_res_[static_cast<size_t>(b)];
        });
        std::vector<int32_t> res_sorted(N);
        tf_rank_.assign(N, 0);
        for (size_t r = 0; r < N; ++r) {
            tf_rank_[static_cast<size_t>(order[r])] = static_cast<int32_t>(r);
            res_sorted[r] = tf_res_[static_cast<size_t>(order[r])];
        }
        // Band rows: atom i holds the atoms of residues res_i -+ kTopoBand.
        tf_row_.assign(N, 0);
        std::vector<int32_t> lo(N), hi(N);
        size_t total = 0;
        for (size_t i = 0; i < N; ++i) {
            const int32_t r = tf_res_[i];
            lo[i] = static_cast<int32_t>(
                std::lower_bound(res_sorted.begin(), res_sorted.end(),
                                 r - kTopoBand) - res_sorted.begin());
            hi[i] = static_cast<int32_t>(
                std::upper_bound(res_sorted.begin(), res_sorted.end(),
                                 r + kTopoBand) - res_sorted.begin());
            if (total > static_cast<size_t>(std::numeric_limits<int32_t>::max() / 2))
                return;
            tf_row_[i] = static_cast<int32_t>(total) - lo[i];
            total += static_cast<size_t>(hi[i] - lo[i]);
        }
        tf_band_.assign(std::max<size_t>(total, 1), 0);
        for (size_t i = 0; i < N; ++i) {
            for (int32_t r = lo[i]; r < hi[i]; ++r) {
                const size_t j = static_cast<size_t>(order[static_cast<size_t>(r)]);
                tf_band_[static_cast<size_t>(tf_row_[i] + r)] = topo_flag_[i * N + j];
            }
        }
        // Far pairs: each class pair's most common byte; the rest are
        // exceptions.
        const size_t C = static_cast<size_t>(ncls);
        std::vector<uint32_t> votes(C * C * 4, 0);
        for (size_t i = 0; i < N; ++i) {
            const uint8_t* row = topo_flag_.data() + i * N;
            const size_t ci = tf_cls_[i] * C;
            for (size_t j = 0; j < N; ++j) {
                const int d = tf_res_[j] - tf_res_[i];
                if (d >= -kTopoBand && d <= kTopoBand) continue;
                if (row[j] > 3) return;
                ++votes[(ci + tf_cls_[j]) * 4 + row[j]];
            }
        }
        tf_far_.assign(C * C, 0);
        for (size_t k = 0; k < C * C; ++k) {
            uint8_t best = 0;
            for (uint8_t f = 1; f < 4; ++f)
                if (votes[k * 4 + f] > votes[k * 4 + best]) best = f;
            tf_far_[k] = best;
        }
        tf_exc_atom_.assign(N, 0);
        tf_exc_start_.assign(N + 1, 0);
        tf_exc_col_.clear();
        tf_exc_flag_.clear();
        for (size_t i = 0; i < N; ++i) {
            const uint8_t* row = topo_flag_.data() + i * N;
            const size_t ci = tf_cls_[i] * C;
            for (size_t j = 0; j < N; ++j) {
                const int d = tf_res_[j] - tf_res_[i];
                if (d >= -kTopoBand && d <= kTopoBand) continue;
                if (row[j] != tf_far_[ci + tf_cls_[j]]) {
                    tf_exc_col_.push_back(static_cast<int32_t>(j));
                    tf_exc_flag_.push_back(row[j]);
                    tf_exc_atom_[i] = 1;
                }
            }
            tf_exc_start_[i + 1] = static_cast<int32_t>(tf_exc_col_.size());
            if (tf_exc_col_.size() > 8 * N) return;  // not compact: keep N*N
        }
        // Check every entry before switching over.
        tf_ready_ = true;
        for (int i = 0; i < n && tf_ready_; ++i) {
            for (int j = 0; j < n; ++j) {
                if (topo_flag(i, j) !=
                    topo_flag_[static_cast<size_t>(i) * N + static_cast<size_t>(j)]) {
                    tf_ready_ = false;
                    break;
                }
            }
        }
        if (mcpu_verbose_enabled()) {
            std::fprintf(stderr,
                         "INFO: compact topo flags %s: band %.1f KB, %d classes, "
                         "%zu far exceptions\n",
                         tf_ready_ ? "on" : "off (mismatch)", total / 1024.0,
                         ncls, tf_exc_col_.size());
        }
    }

    void MuPotential::rebuild_type_params_from_matrices() {
        const size_t N = static_cast<size_t>(num_atoms_cached_);
        if (N == 0) {
            type_params_.clear();
            n_types_ = 0;
            return;
        }

        int max_t = -1;
        for (int t : atom_types) {
            if (t > max_t) max_t = t;
        }
        n_types_ = max_t + 1;
        const size_t NT = static_cast<size_t>(std::max(n_types_, 0));
        type_params_.assign(NT * NT, TypePairParams{});
        std::vector<uint8_t> filled(NT * NT, 0);

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                const int ti = atom_types[i];
                const int tj = atom_types[j];
                if (ti < 0 || tj < 0 || NT == 0) continue;

                TypePairParams tp;
                tp.hard_r2 = hard_core_sq(static_cast<int>(i), static_cast<int>(j));
                tp.contact_r2 =
                    contact_dist_sq(static_cast<int>(i), static_cast<int>(j));
                tp.energy = contact_energies(static_cast<int>(i), static_cast<int>(j));
                const float hr = (tp.hard_r2 > 0.f) ? std::sqrt(tp.hard_r2) : 0.f;
                tp.hard_tol_r2 = hard_tol_r2_from(hr);
                store_type_pair_params(filled, static_cast<int>(i), static_cast<int>(j), tp);
            }
        }
        apply_mu_denselist_cutoff();
    }

    void MuPotential::store_type_pair_params(
        std::vector<uint8_t>& filled, int i, int j, const TypePairParams& tp)
    {
        const size_t NT = static_cast<size_t>(n_types_);
        const int ti = atom_types[static_cast<size_t>(i)];
        const int tj = atom_types[static_cast<size_t>(j)];
        const size_t key = static_cast<size_t>(ti) * NT + static_cast<size_t>(tj);
        const size_t key_sym = static_cast<size_t>(tj) * NT + static_cast<size_t>(ti);
        if (filled[key]) {
            const TypePairParams& have = type_params_[key];
            if (have.hard_r2 != tp.hard_r2 || have.contact_r2 != tp.contact_r2 ||
                have.energy != tp.energy) {
                throw std::invalid_argument(
                    "MuPotential: atoms " + std::to_string(i) + " and " +
                    std::to_string(j) + " (types " + std::to_string(ti) + " and " +
                    std::to_string(tj) + ") have a different hard-core distance, "
                    "contact distance or energy from an earlier pair of the same "
                    "types. Mu stores one entry per unordered type pair, so every "
                    "atom of a type must have the same radius, and the energy "
                    "matrix must be symmetric and finite.");
            }
            return;
        }
        filled[key] = filled[key_sym] = 1;
        type_params_[key] = tp;
        type_params_[key_sym] = tp;
    }

    void MuPotential::cache_necessary_data(
        const std::vector<int8_t>& topo_contact_mask,
        const std::vector<int8_t>& topo_clash_mask,
        const Eigen::Matrix3Xf& coords
    ) {
        const int num_atoms = coords.cols();
        num_atoms_cached_ = num_atoms;
        const size_t n2 = static_cast<size_t>(num_atoms) * static_cast<size_t>(num_atoms);

        CoordsSoA soa_tmp;
        soa_tmp.load_from_eigen(coords);
        const CoordView cv(soa_tmp);

        topo_contact_mask_.assign(n2, 0);
        topo_clash_mask_.assign(n2, 0);

        topo_flag_.assign(n2, uint8_t{0});

        int max_t = -1;
        for (int t : atom_types) {
            if (t > max_t) max_t = t;
        }
        n_types_ = max_t + 1;
        const size_t NT = static_cast<size_t>(std::max(n_types_, 0));
        type_params_.assign(NT * NT, TypePairParams{});
        std::vector<uint8_t> filled(NT * NT, 0);


        for (int i = 0; i < num_atoms; ++i) {
            for (int j = i + 1; j < num_atoms; ++j) {
                const int matrix_idx = i * num_atoms + j;
                const int matrix_idx_sym = j * num_atoms + i;

                bool check_clash = topo_clash_mask[static_cast<size_t>(matrix_idx)] != 0;
                bool check_contact = topo_contact_mask[static_cast<size_t>(matrix_idx)] != 0;

                // Native-structure exceptions: a pair already under its move
                // cutoff in the structure the force field is built from is
                // exempt for the whole run, since every move that re-decided
                // it would be rejected. The test is the move cutoff, not the
                // exact hard core: a pair between the two clashes under
                // neither, and exempting it would drop its protection for good.
                if (check_clash) {
                    const float dist_sq = cv.dist2(i, j);
                    const float hc = hard_core_sq(i, j);
                    if (is_hard_clash(dist_sq,
                                      hard_tol_r2_from(hc > 0.f ? std::sqrt(hc) : 0.f))) {
                        check_clash = false;
                    }
                }

                topo_clash_mask_[static_cast<size_t>(matrix_idx)] =
                    topo_clash_mask_[static_cast<size_t>(matrix_idx_sym)] =
                        static_cast<uint8_t>(check_clash ? 1 : 0);

                // A zero-energy contact pair is no contact pair.
                const float e_ij = contact_energies(i, j);
                if (check_contact && e_ij == 0.0f) {
                    check_contact = false;
                }
                topo_contact_mask_[static_cast<size_t>(matrix_idx)] =
                    topo_contact_mask_[static_cast<size_t>(matrix_idx_sym)] =
                        static_cast<uint8_t>(check_contact ? 1 : 0);

                const float hc = hard_core_sq(i, j);
                const float cd = contact_dist_sq(i, j);

                uint8_t flag = 0;
                if (check_clash) flag |= 1u;
                if (check_contact) flag |= 2u;
                topo_flag_[static_cast<size_t>(matrix_idx)] =
                    topo_flag_[static_cast<size_t>(matrix_idx_sym)] = flag;

                const int ti = atom_types[static_cast<size_t>(i)];
                const int tj = atom_types[static_cast<size_t>(j)];
                if (ti >= 0 && tj >= 0 && NT > 0) {
                    TypePairParams tp;
                    tp.hard_r2 = hc;
                    tp.contact_r2 = cd;
                    tp.energy = e_ij;
                    const float hr = (hc > 0.f) ? std::sqrt(hc) : 0.f;
                    tp.hard_tol_r2 = hard_tol_r2_from(hr);
                    store_type_pair_params(filled, i, j, tp);
                }
            }
        }
        if (mcpu_verbose_enabled()) {
            std::fprintf(stderr, "INFO: topo_flag_ %.2f MB (N²×1 B)\n",
                         topo_flag_.size() / (1024.0 * 1024.0));
        }
        build_compact_topo();
        // CHANGED: exact denselist cutoff from type_params_ (default ON).
        apply_mu_denselist_cutoff();
    }

    // ---------------------------------------------------------
    // Dispatch
    // ---------------------------------------------------------
    EnergyChangeResult MuPotential::calculateEnergyChange(
        const Context& context,
        const State& old_state,
        const State& new_state,
        const ProposalPatch& patch
    ) const {
        // DEFAULT ON (1.35-1.45x on chignolin/1igd/actin when introduced).
        // MCPU_CONTACT_LIST=0 sends every move to the all-pairs moved-vs-all
        // delta instead: an exact O(n_moved * N) reference, slow, for checks.
        const bool kContactList = contact_list_enabled();
        float delta;
        if (kContactList) {
            // A rigid move adds to the drift budget of the pairs it carries
            // unseen (see the contact-list notes in the header).
            const bool carries =
                neighbor::MoveFootprint::of(patch, context.neighborConfig().skip_rigid_mm)
                .moved_rigid();
            const float carry_bound =
                carries ? context.rigid_carry_bound_A(new_state, patch) : 0.f;
            const float budget = kContactBandA - kContactBandSlackA;
            // The live list is only valid where the dense contiguous grid sees
            // every candidate pair and the carry fits the drift budget. Under
            // an energy mask it holds no masked pair (eval_pair scores them 0
            // and the clash tests still see ClashOnly ones), so it is valid
            // for the mask it was built under; a mask change drops it, or a
            // prebuilt one.
            const System& sys_m = context.getSystem();
            if ((old_state.mu_contacts.ready() ||
                 old_state.mu_contact_list_prebuilt) &&
                old_state.mu_list_mask_epoch != sys_m.energy_mask_epoch()) {
                old_state.mu_contact_invalidate();
            }
            const bool usable =
                context.denseGridsActive() &&
                context.neighbors().muGrid().grid().use_contiguous() &&
                context.trial_in_bounds(new_state, patch) &&
                carry_bound <= budget;
            if (usable) {
                if (!old_state.mu_contacts.ready() &&
                    old_state.mu_contact_list_prebuilt) {
                    // Filled by the last full-energy resync from these same
                    // coordinates: the list rebuild_contact_list would make.
                    old_state.mu_contact_list_prebuilt = false;
                    old_state.mu_contacts.set_ready(true);
                    ++contact_list_rebuilds_;
                }
                if (!old_state.mu_contacts.ready() ||
                    old_state.mu_list_drift + carry_bound > budget) {
                    rebuild_contact_list(context, old_state);
                }
                delta = calculateEnergyChange_clist(
                    context, old_state, new_state, patch, carry_bound);
            } else {
                // The list cannot follow this move. If the move is accepted,
                // the list is dropped, and the next move that can use one
                // rebuilds it (O(N^2)); a rejected trial leaves it as it was.
                // Until then it still says which carried pairs are near
                // their cutoff.
                const bool list_exact =
                    old_state.mu_contacts.ready() &&
                    old_state.mu_list_drift + carry_bound <= budget;
                // Nearly every move that lands here overlaps something (99%
                // of them on actin pivots, 94% on LDH-A), and the all-pairs
                // delta below spends ~2 x n_moved x N distance tests, ~8 ms
                // on actin, to say so. The grid still holds every atom that
                // delta's new half tests against (all unmoved atoms but the
                // fixed amide H, which it skips too), so ask the clash-first
                // question there first: an overlap it finds is one the delta
                // would find, and the move is rejected the same way. A move
                // it clears goes through the delta as before. An energy mask
                // changes neither answer: the overlap test drops the pairs
                // ignore_all switches off and keeps the clashes clash_only
                // keeps, as eval_pair does, and reads the mask afresh.
                int overlap = -1;
                if (context.denseGridsActive() &&
                    context.neighbors().muGrid().grid().use_contiguous() &&
                    !patch.moved_indices.empty()) {
                    overlap = fallback_grid_overlap(context, new_state, patch);
                }
                if (overlap >= 0) {
                    auto& ws = const_cast<mcpu::MuWorkspace&>(
                        context.getMuWorkspace());
                    context.pairScratch().clash_hot.note(overlap);
                    ws.clear();
                    delta = kHardCorePenalty;
                } else {
                    delta = calculateEnergyChange_fast(
                        context, old_state, new_state, patch, list_exact);
                }
                const_cast<mcpu::MuWorkspace&>(context.getMuWorkspace())
                    .pending_list_invalidate = true;
                ++clist_fallbacks_;
                // A developer diagnostic; clist_fallbacks() counts it
                // either way.
                if (mcpu_verbose_enabled() &&
                    (clist_fallbacks_ == 1 || (clist_fallbacks_ % 1000) == 0)) {
                    std::fprintf(stderr,
                        "NOTE: Mu move #%llu that cannot use the contact "
                        "list (it leaves the neighbour grid, or there is "
                        "none). An accepted one costs an O(N^2) list "
                        "rebuild.\n",
                        static_cast<unsigned long long>(clist_fallbacks_));
                }
            }
        } else {
            delta = calculateEnergyChange_fast(
                context, old_state, new_state, patch);
        }
        if (delta >= 0.5f * kHardCorePenalty) {
            return EnergyChangeResult::rejected(delta, RejectReason::StericClash);
        }
        return EnergyChangeResult::finite(delta);
    }

    __attribute__((always_inline)) inline int MuPotential::first_grid_overlap(
            const Context& context, const State& new_state,
            const ProposalPatch& patch, const std::uint8_t* mpc,
            bool hot_only) const {
        const float rq = clash_query_radius();
        if (!(rq > 0.f)) return -1;
        const OpenCellGrid& grid = context.neighbors().muGrid().grid();
        const std::vector<uint8_t>& is_moved = patch.moving_atoms;
        const std::vector<int>& moved = patch.moved_indices;
        const CoordView cnew(new_state.coord_view());
        // Enumerates only the cells rq can reach, not the contact stencil.
        const neighbor::WalkArgs wa{is_moved.data(), is_moved.size(), mpc, rq,
                                    clash_prefilter_r2_ * kSpanMaskSlack};
        return neighbor::hot_then_moved<neighbor::Cells::WithinRadius>(
            grid, cnew, context.pairScratch().clash_hot, moved.data(),
            hot_only ? 0 : static_cast<int>(moved.size()), wa,
            [&](const neighbor::Probe& p, int j, const neighbor::CellSpan& s,
                int m) {
                const float dx = p.x - s.x[m];
                const float dy = p.y - s.y[m];
                const float dz = p.z - s.z[m];
                const float r2 = pair_r2(dx, dy, dz);
                return overlaps_at_move_cutoff(p.i, j, r2)
                           ? neighbor::Visit::Stop
                           : neighbor::Visit::Continue;
            });
    }

    int MuPotential::fallback_grid_overlap(const Context& context,
                                           const State& new_state,
                                           const ProposalPatch& patch) const {
        setup_mask_cache(context.getSystem());
        const OpenCellGrid& grid = context.neighbors().muGrid().grid();
        const std::vector<int>& moved = patch.moved_indices;
        const neighbor::MovedCellScope<OpenCellGrid> moved_cells(
            context.pairScratch().moved[neighbor::kMuGrid], grid, moved.data(),
            static_cast<int>(moved.size()));
        return first_grid_overlap(context, new_state, patch,
                                  moved_cells.counts(), /*hot_only=*/false);
    }


    float MuPotential::delta_moved_vs_all(
        const Context& context,
        const State& old_state,
        const State& new_state,
        const ProposalPatch& patch,
        const std::vector<int>& moved_indices,
        bool list_exact
    ) const {
        auto& ws = const_cast<mcpu::MuWorkspace&>(context.getMuWorkspace());
        const std::vector<uint8_t>& is_moved = patch.moving_atoms;
        // Fallback scans all atoms; skip amide H (not in historical Mu contact set).
        const auto& sys = context.getSystem();
        const int h_begin = sys.getTotalBBAtoms() + sys.getTotalOAtoms() + sys.getTotalSCAtoms();
        auto skip_fixed_h = [&](int j) {
            if (sys.residueContiguousLayout()) {
                return sys.is_amide_h_atom(j) && !is_moved[static_cast<size_t>(j)];
            }
            return j >= h_begin && !is_moved[static_cast<size_t>(j)];
        };
        const bool skip_rigid_mm =
            neighbor::MoveFootprint::of(patch, context.neighborConfig().skip_rigid_mm)
                .moved_rigid();
        // Under an energy mask, a moved atom of a masked residue adds only
        // zeros to the old half (eval_pair scores its pairs 0 in either
        // mode), and under ignore_all to the new half too (its pairs cannot
        // clash). Leave such atoms out of the O(N) scans below: only zero
        // terms go, so the delta is the same to the bit. A masked tail that
        // leaves the grid on its own then costs O(n_moved), not
        // O(n_moved x N). (calculateEnergyChange_fast set the mask cache.)
        std::vector<int> unmasked;
        if (energy_mask_ptr_) {
            unmasked.reserve(moved_indices.size());
            for (int i : moved_indices) {
                if (!energy_mask_ptr_[static_cast<size_t>(
                        atom_to_residue[static_cast<size_t>(i)])]) {
                    unmasked.push_back(i);
                }
            }
        }
        const std::vector<int>& old_scan =
            energy_mask_ptr_ ? unmasked : moved_indices;
        const std::vector<int>& new_scan =
            energy_mask_ptr_ &&
                    energy_mask_mode_cached_ == EnergyMaskMode::IgnoreAll
                ? unmasked
                : moved_indices;
        float delta_E = 0.0f;
        bool clash = false;
        auto& nstats = const_cast<NeighborStats&>(context.neighborStats());
        const float cut2 = contact_cutoff_sq_;
        // A rigid move keeps the distances it carries up to rounding,
        // which can still take a pair across its contact cutoff. With an
        // exact contact list, only the carried pairs it lists can cross,
        // and they are re-decided from the new coordinates as the list
        // path does; without one, every carried pair within the Mu
        // cutoff is (carried_pairs_delta).
        if (skip_rigid_mm) {
            if (list_exact) {
                const CoordView cnew_mm(new_state.coord_view());
                for (int i : moved_indices) {
                    for (const auto& c : old_state.mu_contacts.partners(i)) {
                        const int j = c.j;
                        if (i > j || !is_moved[static_cast<size_t>(j)]) continue;
                        delta_E += listed_contact_energy(i, j, cnew_mm.dist2(i, j)) - c.payload;
                    }
                }
            } else {
                delta_E += carried_pairs_delta(old_state, new_state, moved_indices);
            }
            const std::uint64_t n = static_cast<std::uint64_t>(moved_indices.size());
            nstats.elided_rigid_mm += (n * (n > 0 ? n - 1 : 0)) / 2ull;
        }
        // Old energy: moved at old pos vs all (accepted coords for partners)
        NeighborFallback::for_each_moved_neighbor(
            old_state.coords_soa, old_state.coords_soa, old_scan, is_moved, cut2,
            /*is_rigid=*/skip_rigid_mm,
            [&](int i, int j, float r2) __attribute__((always_inline)) {
                if (skip_fixed_h(j)) return;
                note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                // ClashCutoff::None, written as the state cutoff plus a
                // fix-up for the rare pair under it: with a plain
                // eval_pair<None> GCC 8 makes this scan 1.15x slower (actin,
                // 2881 moved atoms).
                bool lc = false;
                float e = eval_pair<ClashCutoff::State>(i, j, r2, &lc);
                if (lc) e = eval_pair<ClashCutoff::None>(i, j, r2, nullptr);
                delta_E -= e;
            });
        // New energy: moved at new pos vs fixed(old); MM skipped when rigid.
        NeighborFallback::for_each_moved_neighbor(
            new_state.coords_soa, old_state.coords_soa, new_scan, is_moved, cut2,
            /*is_rigid=*/true, // always skip MM here; handled below if needed
            [&](int i, int j, float r2) __attribute__((always_inline)) {
                if (is_moved[static_cast<size_t>(j)]) return; // fixed only here
                if (skip_fixed_h(j)) return;
                note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                bool local_clash = false;
                delta_E += eval_pair(i, j, r2, &local_clash);
                if (local_clash) clash = true;
            });
        if (!clash && !skip_rigid_mm) {
            for (size_t a = 0; a < new_scan.size() && !clash; ++a) {
                const int i = new_scan[a];
                for (size_t b = a + 1; b < new_scan.size(); ++b) {
                    const int j = new_scan[b];
                    const float r2 = new_state.coord_view().dist2(i, j);
                    note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                    bool local_clash = false;
                    delta_E += eval_pair(i, j, r2, &local_clash);
                    if (local_clash) { clash = true; break; }
                }
            }
        }
        if (clash) {
            ws.clear();
            return kHardCorePenalty;
        }
        return delta_E;
    }

    float MuPotential::carried_pairs_delta(const State& old_state,
                                           const State& new_state,
                                           const std::vector<int>& moved) const {
        // The old coordinates of the moved atoms, packed, so that each row
        // of distances is one contiguous loop the compiler can vectorize.
        const size_t n = moved.size();
        std::vector<float> x(n), y(n), z(n), r2_row(n);
        const CoordView cold(old_state.coord_view());
        const CoordView cnew(new_state.coord_view());
        for (size_t a = 0; a < n; ++a) {
            x[a] = cold.x(moved[a]);
            y[a] = cold.y(moved[a]);
            z[a] = cold.z(moved[a]);
        }
        const float cut2 = contact_cutoff_sq_;
        double dE = 0.0;
        for (size_t a = 0; a + 1 < n; ++a) {
            const float xa = x[a], ya = y[a], za = z[a];
            for (size_t b = a + 1; b < n; ++b) {
                const float dx = x[b] - xa, dy = y[b] - ya, dz = z[b] - za;
                r2_row[b] = pair_r2(dx, dy, dz);
            }
            // Beyond the Mu cutoff a pair is out of contact on both sides:
            // the cutoff includes the contact list's 0.05 A band, and a
            // carry is far smaller.
            for (size_t b = a + 1; b < n; ++b) {
                if (r2_row[b] > cut2) continue;
                const int i = moved[a], j = moved[b];
                dE += static_cast<double>(
                    eval_pair<ClashCutoff::None>(i, j, cnew.dist2(i, j), nullptr) -
                    eval_pair<ClashCutoff::None>(i, j, r2_row[b], nullptr));
            }
        }
        return static_cast<float>(dE);
    }

    float MuPotential::calculateEnergyChange_fast(
        const Context& context,
        const State& old_state,
        const State& new_state,
        const ProposalPatch& patch,
        bool list_exact
    ) const {
        setup_mask_cache(context.getSystem());
        auto& ws = const_cast<mcpu::MuWorkspace&>(context.getMuWorkspace());
        ws.clear();
        // Direct delta callers do not pass through Integrator's bounds policy.
        // Derive fallback from the trial itself so an out-of-grid clash cannot
        // be omitted by cell-list enumeration.
        ws.use_trial_fallback = !context.trial_in_bounds(new_state, patch);
        eval_pair_calls_local_ = 0;
        eval_pair_nonzero_local_ = 0;
        auto& nstats_flush = const_cast<NeighborStats&>(context.neighborStats());
        struct EvalCallsFlush {
            NeighborStats* s;
            const MuPotential* self;
            ~EvalCallsFlush() {
                if (s) {
                    s->mu_eval_pair_calls += self->eval_pair_calls_local_;
                    s->mu_eval_pair_nonzero += self->eval_pair_nonzero_local_;
                }
            }
        } eval_flush{&nstats_flush, this};


        const int num_atoms = context.getSystem().getNumAtoms();
        auto& ns = const_cast<NeighborSystem&>(context.neighbors());
        // Mu index (BB+O+SC only). Prefer NeighborSystem candidate API.
        const std::vector<uint8_t>& is_moved = patch.moving_atoms;
        // Prefer patch.moved_indices; fall back to a one-time scan if empty
        std::vector<int> fallback_moved;
        const std::vector<int>* moved_ptr = &patch.moved_indices;
        {
            if (moved_ptr->empty()) {
                fallback_moved.reserve(64);
                for (int i = 0; i < num_atoms; ++i) {
                    if (is_moved[static_cast<size_t>(i)]) fallback_moved.push_back(i);
                }
                moved_ptr = &fallback_moved;
            }
        }
        const std::vector<int>& moved_indices = *moved_ptr;


        float delta_E = 0.0f;
        bool clash = false;

        const bool skip_rigid_mm =
            neighbor::MoveFootprint::of(patch, context.neighborConfig().skip_rigid_mm)
                .moved_rigid();

        // Hot pair loop #3: moved-vs-all, for a trial that leaves the grid
        // (no dense-grid rebuild under AUTO_EXPAND), a run without dense
        // grids, and every move under MCPU_CONTACT_LIST=0, where it is the
        // exact slow reference for the contact-list delta.
        if (ws.use_trial_fallback || !context.denseGridsActive() ||
            !contact_list_enabled()) {
            return delta_moved_vs_all(context, old_state, new_state, patch,
                                      moved_indices, list_exact);
        }

        auto& nstats = const_cast<NeighborStats&>(context.neighborStats());

        CellListMC* moved_grid = nullptr;
        // Rigid pivot: skip moved_new_grid -- its moved-moved pairs are not
        // re-measured on this path (see the contact-list notes in the header).
        if (!moved_indices.empty() && !skip_rigid_mm) {
            ws.ensure_moved_grid(mu_exact_cutoff_, num_atoms);
            moved_grid = ws.moved_new_grid.get();
            // Share contact-grid bounds so inserts index correctly (open, no wrap)
            const auto& gb = ns.muGrid().grid().bounds();
            if (gb.valid) {
                NeighborConfig cfg = context.neighborConfig();
                const float mu_cell = effective_mu_cell_size_A(mu_exact_cutoff_, cfg);
                // Match accepted Mu denselist: r_mu cell/query.
                // CRITICAL: configure() assigns n_cells×CAPACITY packed arrays (~MB).
                // Only reconfigure when geometry changes — was called every SC/KIC
                // denselist step and dominated SC Mu (~80 µs fixed overhead).
                moved_grid->set_cutoff(mu_exact_cutoff_);
                if (!moved_grid->grid().matches_geometry(gb, mu_cell, cfg)) {
                    moved_grid->configure(gb, cfg, mu_cell);
                }
                // Scratch MM grid never uses occupied stencil (Mu denselist only).
                moved_grid->grid().set_occupied_stencil_mode(
                    OccupiedStencilMode::Off);
            }
            // No reset()/clear_cells_keep_shape: ensure_moved_grid already removed
            // prior membership via clear_moved_grid (O(n_moved)).
            moved_grid->ensure_atom_capacity(num_atoms);
            ws.moved_grid_atoms.reserve(moved_indices.size());
            for (int j : moved_indices) {
                moved_grid->insert(j, new_state.coords_soa);
                ws.moved_grid_atoms.push_back(j);
            }
        }

        // Hot pair loop #1: classic denselist OpenCellGrid.
        // CSR pack-all-then-SIMD-r² was tried and reverted (wall ~150→459 µs):
        // packing before clash exit overflows (>98k pairs) and double-walks;
        // cell walk is ~71% of pivot Mu so splitting r² cannot win.
        const CoordView cold(old_state.coord_view());
        const CoordView cnew(new_state.coord_view());

        // Per-moved-atom fused walk of the dense Mu grid. The contact list
        // cannot follow an in-grid move only when the grid lost its
        // contiguous layout after a cell overflow, so that is when this runs.
        const OpenCellGrid& mu_grid_ref = ns.muGrid().grid();
        const bool use_span = mu_grid_ref.use_contiguous();
        for (int i : moved_indices) {
            const float ox = cold.x(i), oy = cold.y(i), oz = cold.z(i);
            const float nx = cnew.x(i), ny = cnew.y(i), nz = cnew.z(i);

            if (use_span) {
                mu_grid_ref.for_each_neighbor_cell_span(
                    ox, oy, oz,
                    [&](const int* __restrict__ cids,
                        const float* __restrict__ cx,
                        const float* __restrict__ cy,
                        const float* __restrict__ cz, int count) {
                        std::uint64_t skip_mask = 0ull;
                        for (int m = 0; m < count; ++m) {
                            const int j = cids[m];
                            if (j == i) {
                                skip_mask |= (1ull << m);
                                continue;
                            }
                            if (is_moved[static_cast<size_t>(j)]) {
                                if (skip_rigid_mm) {
                                    ++nstats.elided_rigid_mm;
                                    skip_mask |= (1ull << m);
                                } else if (i > j) {
                                    skip_mask |= (1ull << m);
                                }
                            }
                        }
                        float r2_buf[OpenCellGrid::CELL_CAPACITY];
#pragma GCC ivdep
                        for (int m = 0; m < count; ++m) {
                            const float dx = ox - cx[m];
                            const float dy = oy - cy[m];
                            const float dz = oz - cz[m];
                            r2_buf[m] = pair_r2(dx, dy, dz);
                        }
                        for (int m = 0; m < count; ++m) {
                            if (skip_mask & (1ull << m)) continue;
                            const float r2 = r2_buf[m];
                            note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                            if (r2 <= contact_cutoff_sq_)
                                delta_E -=
                                    eval_pair<ClashCutoff::None>(i, cids[m], r2, nullptr);
                        }
                    },
                    &nstats.neighbor_num_cell_visits);

                const bool ok_fixed =
                    mu_grid_ref.for_each_neighbor_cell_span_while(
                        nx, ny, nz,
                        [&](const int* __restrict__ cids,
                            const float* __restrict__ cx,
                            const float* __restrict__ cy,
                            const float* __restrict__ cz, int count) {
                            std::uint64_t skip_mask = 0ull;
                            for (int m = 0; m < count; ++m) {
                                const int j = cids[m];
                                if (j == i ||
                                    is_moved[static_cast<size_t>(j)])
                                    skip_mask |= (1ull << m);
                            }
                            float r2_buf[OpenCellGrid::CELL_CAPACITY];
#pragma GCC ivdep
                            for (int m = 0; m < count; ++m) {
                                const float dx = nx - cx[m];
                                const float dy = ny - cy[m];
                                const float dz = nz - cz[m];
                                r2_buf[m] = pair_r2(dx, dy, dz);
                            }
                            for (int m = 0; m < count; ++m) {
                                if (skip_mask & (1ull << m)) continue;
                                const float r2 = r2_buf[m];
                                bool local_clash = false;
                                note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                                if (r2 <= contact_cutoff_sq_) {
                                    delta_E += eval_pair(
                                        i, cids[m], r2, &local_clash);
                                }
                                if (local_clash) {
                                    clash = true;
                                    return false;
                                }
                            }
                            return true;
                        },
                        &nstats.neighbor_num_cell_visits);
                if (!ok_fixed || clash) {
                    clash = true;
                    break;
                }
            } else {
                ns.for_each_mu_candidate(ox, oy, oz, [&](int j) {
                    if (j == i) return;
                    if (is_moved[static_cast<size_t>(j)]) {
                        if (skip_rigid_mm) {
                            ++nstats.elided_rigid_mm;
                            return;
                        }
                        if (i > j) return;
                    }
                    const float r2 = cold.dist2(i, j);
                    note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                    if (r2 <= contact_cutoff_sq_)
                        delta_E -= eval_pair<ClashCutoff::None>(i, j, r2, nullptr);
                });

                {
                    const bool ok_fixed = ns.for_each_mu_candidate_while(
                        nx, ny, nz, [&](int j) {
                            if (j == i) return true;
                            if (is_moved[static_cast<size_t>(j)])
                                return true;
                            const float r2 = cnew.dist2(i, j);
                            bool local_clash = false;
                            note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                            if (r2 <= contact_cutoff_sq_) {
                                delta_E +=
                                    eval_pair(i, j, r2, &local_clash);
                            }
                            if (local_clash) {
                                clash = true;
                                return false;
                            }
                            return true;
                        });
                    if (!ok_fixed || clash) {
                        clash = true;
                        break;
                    }
                }
            }

            if (moved_grid) {
                const bool ok_moved = moved_grid->for_each_neighbor_while(
                    nx, ny, nz, [&](int j) {
                        ++nstats.mu_num_candidates_iterated;
                        if (j == i) return true;
                        if (i > j) return true;
                        const float r2 = cnew.dist2(i, j);
                        bool local_clash = false;
                        note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                        if (r2 <= contact_cutoff_sq_) {
                            delta_E += eval_pair(i, j, r2, &local_clash);
                        }
                        if (local_clash) {
                            clash = true;
                            return false;
                        }
                        return true;
                    });
                if (!ok_moved || clash) {
                    clash = true;
                    break;
                }
            }
        }

        if (clash) {
            ws.clear();
            ws.clear_moved_grid();
            return kHardCorePenalty;
        }

        ws.clear_moved_grid();
        return delta_E;
    }

    // ================================================================
    // LIVE CONTACT LIST  (default; MCPU_CONTACT_LIST=0 turns it off)
    // ================================================================
    // See the header for why this exists. In short: a move's energy change is
    //     dE = (contact energy at the new positions)
    //        - (contact energy at the old positions)
    // and the second term is something we already knew at the end of the
    // previous accepted move. Writing it down means the move only has to
    // measure distances ONCE, at the new positions.

    
    
    void MuPotential::rebuild_contact_list(const Context& context,
                                           const State& state) const {
        const System& sys = context.getSystem();
        setup_mask_cache(sys);
        const int N = sys.getNumAtoms();
        state.mu_contacts.reset(N);
        state.mu_contact_list_prebuilt = false;
        const CoordView cv(state.coord_view());
        std::vector<int> atoms;
        mu_pair_atoms(sys, N, atoms);
        mu_for_each_near_pair(cv, atoms, contact_cutoff_sq_, [&](int i, int j, float r2) {
                const size_t idx = static_cast<size_t>(i) *
                                       static_cast<size_t>(N) +
                                   static_cast<size_t>(j);
                if (!topo_contact_mask_[idx]) return false;
                // No clash test (ClashCutoff::None): a pair that rounding
                // carried under its hard-core cutoff is listed with the
                // contact energy the running energy holds for it.
                bool near = false;
                const float e = eval_pair<ClashCutoff::None>(i, j, r2, nullptr, &near);
                if (e != 0.0f || near) state.mu_contacts.add(i, j, e);
                return false;
        });
        state.mu_contacts.set_ready(true);
        state.mu_list_drift = 0.f;
        state.mu_list_mask_epoch = sys.energy_mask_epoch();
        ++contact_list_rebuilds_;
    }

    
    float MuPotential::calculateEnergyChange_clist(
        const Context& context,
        const State& old_state,
        const State& new_state,
        const ProposalPatch& patch,
        float carry_bound
    ) const {
        setup_mask_cache(context.getSystem());
        auto& ws = const_cast<mcpu::MuWorkspace&>(context.getMuWorkspace());
        ws.clear();

        auto& ns = const_cast<NeighborSystem&>(context.neighbors());
        const OpenCellGrid& grid = ns.muGrid().grid();
        const std::vector<uint8_t>& is_moved = patch.moving_atoms;
        const std::vector<int>& moved = patch.moved_indices;
        const CoordView cnew(new_state.coord_view());
        const CoordView cold(old_state.coord_view());
        const bool skip_mm =
            neighbor::MoveFootprint::of(patch, context.neighborConfig().skip_rigid_mm)
                .moved_rigid();

        double dE = 0.0;
        bool clash = false;

        // Both new-position walks below skip every moved atom they meet: the
        // grid holds accepted coordinates, so a moved atom's packed position
        // is stale. On a pivot most of the cells they visit hold nothing else
        // (measured on actin: 71% of the contact walk's cell visits, 78% of
        // the clash pass's, ~75% of all slots), so count the moved atoms each
        // cell lists and skip a cell whose atoms all moved. Same pairs, same
        // order: only cells that contribute nothing are dropped. O(n_moved)
        // to fill, and the guard zeroes it again on every return.
#ifndef NDEBUG
        // The skip test (count == moved_per_cell[c]) is only right when each
        // moved atom is listed once: a duplicate makes a cell that still holds
        // an unmoved atom look fully moved. See ProposalPatch::mark_moved.
        {
            std::size_t n_marked = 0;
            for (std::uint8_t m : is_moved) n_marked += (m != 0);
            assert(n_marked == moved.size() &&
                   "moved_indices must list each atom moving_atoms marks exactly once");
        }
#endif
        const neighbor::MovedCellScope<OpenCellGrid> moved_cells(
            context.pairScratch().moved[neighbor::kMuGrid], grid, moved.data(),
            static_cast<int>(moved.size()));
        const std::uint8_t* const mpc = moved_cells.counts();

        // ---- PASS 0: answer "does this move overlap?" on its own ----
        // About a third of the actin step is contact energy computed for pivot
        // moves that are then discarded for a hard-core overlap (measured:
        // 34.5% of the step, results/clash_split_summary.md). The overlap
        // question needs only r < ~2.8 A, while the contact walk sweeps ~5.1 A
        // cells -- a box ~6x larger in volume. Asking it first, with a stencil
        // sized to the question, rejects those moves without doing any contact
        // work at all. Costs one extra tight pass on moves that do NOT overlap.
        // It measured a further 1.16x on actin / 1.11x on 1igd on top of the
        // contact list, and neutral on chignolin thanks to the moved-atom
        // gate below. See first_grid_overlap().
        const bool clash_first =
            static_cast<int>(moved.size()) >=
            context.neighborConfig().clash_first_min_moved;
        // By default the pass tests only the clash_hot atoms. They catch
        // 98.6-99.4% of the pivots that overlap (actin, LDH-A, PGK1), while
        // testing every moved atom cost the pivots that do NOT overlap a
        // full extra pass, 6-9% of a pivot-only step. The contact walk below
        // rejects the few overlaps the hot atoms miss, and feeds clash_hot.
        int clash_atom = -1;
        if (clash_first) {
            const int found = first_grid_overlap(context, new_state, patch, mpc,
                                                 /*hot_only=*/true);
            if (found >= 0) {
                context.pairScratch().clash_hot.note(found);
                ws.clear();
                return kHardCorePenalty;
            }
        }

        // ---- OLD half: read it off the list. No distances, no cell walk. ----
        // Every pair a moved atom has on the list is a pair this move is about
        // to re-decide, so all of them come off the books here; the NEW half
        // puts back the ones that are still in contact or in the band.
        for (int i : moved) {
            for (const auto& c : old_state.mu_contacts.partners(i)) {
                const int j = c.j;
                if (is_moved[static_cast<size_t>(j)]) {
                    if (i > j) continue;  // once per pair, by the lower index
                    if (skip_mm) {
                        // A rigid move keeps a carried distance up to
                        // rounding, which can still take a pair near its
                        // cutoff across it. Re-decide it here, from the new
                        // coordinates, and update the entry if it flipped.
                        const float e =
                            listed_contact_energy(i, j, cnew.dist2(i, j));
                        if (e != c.payload) {
                            dE += static_cast<double>(e) -
                                  static_cast<double>(c.payload);
                            ws.pending_contacts.drop.push_back(
                                mcpu::MuWorkspace::PendingContact{i, j, c.payload});
                            ws.pending_contacts.add.push_back(
                                mcpu::MuWorkspace::PendingContact{i, j, e});
                        }
                        continue;
                    }
                }
                dE -= static_cast<double>(c.payload);
                ws.pending_contacts.drop.push_back(
                    mcpu::MuWorkspace::PendingContact{i, j, c.payload});
            }
        }

        // ---- NEW half: one cell walk, one distance per candidate. ----
        // The grid holds ACCEPTED coordinates, so a moved partner's packed
        // position is stale: the walk skips moved partners, and moved-moved
        // pairs are done below from the trial coordinates. The walk hands
        // over the distance its prefilter computed and applies the exact
        // cutoff itself (moved_vs_static_r2).
        const neighbor::WalkArgs contact_wa{
            is_moved.data(), is_moved.size(), mpc, 0.f,
            contact_cutoff_sq_};
        const bool walked = neighbor::moved_vs_static_r2(
            grid, cnew, moved.data(), static_cast<int>(moved.size()),
            contact_wa,
            [&](const neighbor::Probe& p, int j, float r2) {
                bool local_clash = false, near = false;
                const float e = eval_pair(p.i, j, r2, &local_clash, &near);
                if (local_clash) {
                    clash_atom = p.i;
                    return neighbor::Visit::Stop;
                }
                if (e != 0.0f || near) {
                    dE += static_cast<double>(e);
                    ws.pending_contacts.add.push_back(
                        mcpu::MuWorkspace::PendingContact{p.i, j, e});
                }
                return neighbor::Visit::Continue;
            });
        if (!walked) clash = true;

        // ---- moved-moved pairs ----
        // A rigid move keeps every moved-moved distance up to rounding: the
        // listed ones were re-decided in the OLD half, and the rest cannot
        // cross (State::mu_list_drift). A flexible move re-decides them here.
        if (!clash && moved.size() > 1 && !skip_mm) {
            clash = !neighbor::moved_vs_moved(
                cnew, moved.data(), static_cast<int>(moved.size()),
                contact_cutoff_sq_, [&](int i, int j, float r2) {
                    bool local_clash = false, near = false;
                    const float e = eval_pair(i, j, r2, &local_clash, &near);
                    if (local_clash) return neighbor::Visit::Stop;
                    if (e != 0.0f || near) {
                        dE += static_cast<double>(e);
                        ws.pending_contacts.add.push_back(
                            mcpu::MuWorkspace::PendingContact{i, j, e});
                    }
                    return neighbor::Visit::Continue;
                });
        }

        if (clash) {
            if (clash_first && clash_atom >= 0) context.pairScratch().clash_hot.note(clash_atom);
            ws.clear();
            return kHardCorePenalty;
        }
        ws.pending_list_drift = carry_bound;
        return static_cast<float>(dE);
    }

    float MuPotential::calculateEnergy(const Context& context, const State& state) const {
        return full_energy(context, state, /*resync=*/false);
    }

    float MuPotential::resyncEnergy(const Context& context, const State& state) const {
        return full_energy(context, state, /*resync=*/true);
    }

    float MuPotential::full_energy(
        const Context& context, const State& state, bool resync
    ) const {
        // Full energy for an arbitrary State must use that state's coordinates.
        // NeighborSystem Mu index reflects accepted coords only — do not query it here
        // for proposed/trial states (PhysicsVerifier), so the pairs come from a
        // cell binning of this state's own coordinates (mu_for_each_near_pair).
        const System& sys = context.getSystem();
        setup_mask_cache(sys);
        float total_energy = 0.0f;
        const int num_atoms = sys.getNumAtoms();

        // A resync (Potential::resyncEnergy) also rewrites the state's live
        // contact list, near misses included, from this pass and resets its
        // drift budget, so the list and the running energy are reset
        // together. Masked pairs score 0 here and are not listed, as in
        // rebuild_contact_list, so the refilled list belongs to the current
        // mask. Clearing in place keeps each atom's allocation.
        //
        // A list that is not ready (dropped by set_positions, a restore or a
        // move that could not follow it) is filled too, and kept as prebuilt:
        // the first move that can use a list adopts it instead of paying the
        // O(N^2) rebuild after every replica swap. A clash, or a ClashOnly
        // overlap, drops it.
        bool refill_contacts = false;
        bool prebuild = false;
        if (resync && state.mu_contacts.ready()) {
            refill_contacts = true;
            state.mu_contacts.clear_rows(num_atoms);
            state.mu_list_drift = 0.f;
            state.mu_list_mask_epoch = sys.energy_mask_epoch();
        } else if (resync && contact_list_enabled()) {
            refill_contacts = true;
            prebuild = true;
            state.mu_contacts.reset(num_atoms);
            state.mu_contact_list_prebuilt = false;
            state.mu_list_drift = 0.f;
            state.mu_list_mask_epoch = sys.energy_mask_epoch();
        }
        const CoordView cv(state.coord_view());
        std::vector<int> atoms;
        mu_pair_atoms(sys, num_atoms, atoms);
        // No pair clashes, makes a contact or is a near miss beyond the Mu
        // cutoff (see mu_exact_cutoff_), so only pairs within it are visited.
        const bool clashed = mu_for_each_near_pair(
            cv, atoms, contact_cutoff_sq_, [&](int i, int j, float dist_sq) {
                const int matrix_idx = i * num_atoms + j;

                if (!topo_contact_mask_[static_cast<size_t>(matrix_idx)] &&
                    !topo_clash_mask_[static_cast<size_t>(matrix_idx)]) {
                    return false;
                }
                bool local_clash = false, near = false;
                // The state cutoff: see ClashCutoff.
                const float e = eval_pair<ClashCutoff::State>(
                    i, j, dist_sq, &local_clash, refill_contacts ? &near : nullptr);
                if (local_clash) {
                    // ClashOnly: full energy stays contact-only (legacy
                    // CLASH_WEIGHT=0). Delta path still StericClash-rejects.
                    if (energy_mask_mode_cached_ == EnergyMaskMode::ClashOnly) {
                        // rebuild_contact_list would list an unmasked one of
                        // these with its contact energy; leave that to it.
                        if (refill_contacts) {
                            state.mu_contact_invalidate();
                            refill_contacts = false;
                            prebuild = false;
                        }
                        return false;
                    }
                    // DIAGNOSTIC (MCPU_CLASH_REPORT=1): identify the pair that
                    // trips the sentinel. No move can put a pair under the
                    // move cutoff, and rounding carries one at most a few
                    // 1e-6 A further, so on an accepted state this means the
                    // coordinates came from outside (set_positions, a restore)
                    // or a delta path missed the pair. Reports the geometry
                    // plus the Mu cutoff, so a pair that sits outside the
                    // cell-grid's enumeration radius (the out-of-grid case the
                    // ws.use_trial_fallback comment in calculateEnergyChange
                    // guards against) is identifiable.
                    if (clash_report_enabled()) {
                        const size_t NT = static_cast<size_t>(n_types_);
                        const int ti = atom_types[static_cast<size_t>(i)];
                        const int tj = atom_types[static_cast<size_t>(j)];
                        float hard_r2 = 0.f, contact_r2 = 0.f;
                        if (ti >= 0 && tj >= 0 && NT > 0) {
                            const TypePairParams& g =
                                type_params_[static_cast<size_t>(ti) * NT +
                                             static_cast<size_t>(tj)];
                            hard_r2 = g.hard_r2;
                            contact_r2 = g.contact_r2;
                        }
                        const auto ri = atom_to_residue[static_cast<size_t>(i)];
                        const auto rj = atom_to_residue[static_cast<size_t>(j)];
                        std::fprintf(stderr,
                            "[clash-report] i=%d j=%d res_i=%d res_j=%d "
                            "type_i=%d type_j=%d r=%.6f hard_r=%.6f "
                            "contact_r=%.6f mu_cutoff=%.6f masked_i=%d "
                            "masked_j=%d topo_clash=%d topo_contact=%d\n",
                            i, j, static_cast<int>(ri), static_cast<int>(rj),
                            ti, tj, std::sqrt(dist_sq), std::sqrt(hard_r2),
                            std::sqrt(contact_r2), mu_exact_cutoff_,
                            energy_mask_ptr_
                                ? int(energy_mask_ptr_[static_cast<size_t>(ri)]) : 0,
                            energy_mask_ptr_
                                ? int(energy_mask_ptr_[static_cast<size_t>(rj)]) : 0,
                            int(topo_clash_mask_[static_cast<size_t>(matrix_idx)]),
                            int(topo_contact_mask_[static_cast<size_t>(matrix_idx)]));
                    }
                    // A clashing state's list is half rewritten; drop it, and
                    // the next move rebuilds it.
                    if (refill_contacts) state.mu_contact_invalidate();
                    return true;
                }
                if (e != 0.0f) total_energy += e;
                if (refill_contacts && (e != 0.0f || near)) state.mu_contacts.add(i, j, e);
                return false;
            });
        if (clashed) return kHardCorePenalty;
        if (prebuild) state.mu_contact_list_prebuilt = true;

        return total_energy;
    }

    bool MuPotential::clashesAtMoveCutoff(
        const Context& context, const State& proposed_state,
        const ProposalPatch& patch
    ) const {
        const System& sys = context.getSystem();
        setup_mask_cache(sys);
        const int num_atoms = sys.getNumAtoms();
        const bool skip_carried =
            neighbor::MoveFootprint::of(patch, context.neighborConfig().skip_rigid_mm)
                .moved_rigid();
        const std::vector<uint8_t>& is_moved = patch.moving_atoms;
        // Same moved set as the delta path: moved_indices, or a scan of
        // moving_atoms when a hand-built patch left it empty.
        std::vector<int> scanned;
        if (patch.moved_indices.empty()) {
            for (int a = 0; a < static_cast<int>(is_moved.size()); ++a) {
                if (is_moved[static_cast<size_t>(a)]) scanned.push_back(a);
            }
        }
        const std::vector<int>& moved =
            patch.moved_indices.empty() ? scanned : patch.moved_indices;
        const CoordView cv(proposed_state.coord_view());
        for (int i : moved) {
            if (sys.is_amide_h_atom(i)) continue;
            for (int j = 0; j < num_atoms; ++j) {
                if (j == i || sys.is_amide_h_atom(j)) continue;
                if (is_moved[static_cast<size_t>(j)] && (skip_carried || j < i)) {
                    continue;
                }
                bool clash = false;
                eval_pair(i, j, cv.dist2(i, j), &clash);
                if (clash) return true;
            }
        }
        return false;
    }

} // namespace mcpu::forces::mcpu08
