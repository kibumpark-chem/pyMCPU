#include "pymcpu/forces/mcpu/mcpu08/MuPotential.h"
#include "pymcpu/State.h"
#include "pymcpu/Context.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/CellList.h"
#include "pymcpu/neighbor/OpenCellGrid.h"
#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/neighbor/NeighborFallback.h"
#include "pymcpu/neighbor/VerletList.h"
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

#if defined(MCPU_CP_BREAKDOWN)
#include <x86intrin.h>
#endif

namespace mcpu::forces::mcpu08 {

#if defined(MCPU_CP_BREAKDOWN)
// ---------------------------------------------------------------------------
// Cell-pair phase breakdown (DIAGNOSTIC BUILD ONLY -- -DMCPU_CP_BREAKDOWN).
//
// The cp_* fields in NeighborStats have always been declared, copied into
// StepStats and exposed through bindings, but NOTHING EVER WROTE THEM, so
// MCPU_CELL_PAIR_BREAKDOWN=1 silently reported zeros. This wires them.
//
// Compile-gated AND sampled, because the phases sit inside the per-(moved
// atom, neighbour-cell) loop -- ~860 cell visits/step at actin. Timing every
// visit with 3 rdtsc pairs would add ~6k rdtsc/step ~= 40 us on a 120 us step:
// a 34% observer effect that would invalidate the split we are trying to
// measure. Sampling one move in kCpSample keeps overhead ~0.5% while still
// collecting hundreds of timed moves over a 20k-step run.
//
// CONSEQUENCE: per-step averages must be taken over cp_n_steps (the number of
// SAMPLED moves), never over total steps.
//
// rdtsc rather than steady_clock (~20-25 ns/call through the vDSO) because it
// is ~6-20 cycles. Converted at the 2.9 GHz nominal clock of the Xeon 8268
// these nodes run, so absolute ns drift under turbo -- but the phase SHARES,
// which is the whole point, do not.
static constexpr std::uint64_t kCpSample = 64;
static constexpr double kCpCyclesPerNs = 2.9;

struct CpTimer {
    std::uint64_t* sink;
    std::uint64_t t0;
    bool live;
    CpTimer(std::uint64_t* s, bool on) noexcept
        : sink(s), t0(on ? __rdtsc() : 0), live(on) {}
    ~CpTimer() noexcept {
        if (live) {
            const std::uint64_t d = __rdtsc() - t0;
            *sink += static_cast<std::uint64_t>(static_cast<double>(d) / kCpCyclesPerNs);
        }
    }
};
#define CP_CAT2(a, b) a##b
#define CP_CAT(a, b) CP_CAT2(a, b)
#define CP_SCOPE(field, on) CpTimer CP_CAT(cp_t_, __LINE__)(&(field), (on))
#else
#define CP_SCOPE(field, on) ((void)0)
#endif

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
        // Bound for the rigid-MM clash-guard prefilter (see MuPotential.h).
        // Global max hard_r + 0.001 (rounding resolution) + 0.001 (float slack).
        if (max_hard_r2 > 0.f) {
            const float mm_r = std::sqrt(max_hard_r2) + 0.002f;
            mm_guard_prefilter_r2_ = mm_r * mm_r;
        }
        const float max_r2 = std::max(max_contact_r2, max_hard_r2);
        float exact = kMuCutoffFallbackA;
        if (max_r2 <= 0.f) {
            std::fprintf(stderr,
                "WARN: Mu denselist cutoff: max contact/hard r²=0 — "
                "falling back to %.1f Å.\n",
                kMuCutoffFallbackA);
        } else {
            exact = std::sqrt(max_r2) * 1.0001f;
        }
        const float computed = exact;
        if (const char* e = std::getenv("MCPU_MU_CUTOFF_OVERRIDE")) {
            char* end = nullptr;
            const float v = std::strtof(e, &end);
            if (end != e && v > 0.f) {
                std::fprintf(stderr,
                    "WARN: MCPU_MU_CUTOFF_OVERRIDE=%.4f overrides "
                    "computed exact cutoff %.6f Å.\n",
                    v, computed);
                exact = v;
            }
        }
        mu_exact_cutoff_ = exact;
        contact_cutoff_sq_ = exact * exact;
        // Print once type_params_ is authoritative (skip matrix-only ctor pass).
        // CHANGED: gated behind MCPU_VERBOSE
        if (!type_params_.empty() && mcpu_verbose_enabled()) {
            std::fprintf(stderr, // CHANGED: gated behind MCPU_VERBOSE
                "INFO: Mu denselist cutoff=%.6f Å "
                "(max_contact_r2=%.6f max_hard_r2=%.6f). "
                "MCPU_MU_CUTOFF_OVERRIDE=6.0 restores legacy denselist.\n",
                mu_exact_cutoff_, max_contact_r2, max_hard_r2);
        }
    }

    /// Sort new-side neighbor cells by descending occupancy (MCPU_CLASH_ORDER_BY_DENSITY=1).
    static bool clash_order_by_density_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("MCPU_CLASH_ORDER_BY_DENSITY");
            return e && e[0] == '1';
        }();
        return on;
    }

    // FIXED: moved_bits / skip_mask are 64-bit because they are indexed by
    // `1ull << m` for m < n_static, and n_static can reach CELL_CAPACITY.
    // They were std::uint32_t: for m >= 32 `1u << m` is undefined behaviour, and
    // on x86 the shift count is masked to 5 bits, so slot 32 silently ALIASED
    // slot 0 -- corrupting the moved/skip masks, which can drop a pair or miss a
    // steric clash. Unreachable at default occupancy (actin max ~22) but
    // reachable at MCPU_MU_CELL_SCALE >= 1.2 (measured max_occ 31/46), and a
    // same-residue moved-moved clash escaped the delta path in production
    // (p18.8.7 sce, atoms 811/2180 of residue 270) which is exactly this
    // failure mode. Same bug class as the moved_ids[32] -> [CELL_CAPACITY] fix
    // in Context.h. This assert makes the coupling explicit.
    static_assert(::mcpu::OpenCellGrid::CELL_CAPACITY <= 64,
                  "moved_bits/skip_mask are 64-bit; CELL_CAPACITY must fit");

    /// Report the atom pair that trips the hard-core sentinel in the full
    /// recompute (MCPU_CLASH_REPORT=1). Diagnostic for the delta-path detection
    /// gap: Integrator.cpp rejects any proposal whose delta path reports
    /// StericClash, so a clash surviving into an accepted state means the
    /// grid-based delta enumeration never examined this pair.
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
    ///   2) Verlet enumeration     — calculateEnergyChange_fast (CSR VerletList)
    ///   3) Fallback enumeration  — NeighborFallback moved-vs-all (+ explicit moved–moved)
    /// MCPU_HOT_COUNTERS=0 compiles the hot-loop diagnostic counters out of the
    /// measured path. They are ON by default and therefore present in EVERY
    /// production run and every published pyMCPU timing. The legacy baseline
    /// (fold_potential_mpi_asshipped) carries no equivalent instrumentation, so
    /// leaving them on makes any legacy-vs-pyMCPU ratio asymmetric. Set this to 0
    /// for publication numbers.
    /// MCPU_UNIFORM_SKIPMASK=0 restores the per-(moved atom, cell) skip_mask loop.
    static bool uniform_skipmask_enabled() {
        static const bool on = [] {
            const char* e = std::getenv("MCPU_UNIFORM_SKIPMASK");
            return !(e && e[0] == '0');
        }();
        return on;
    }

    static bool hot_counters_enabled() {
        // DEFAULT OFF. These are diagnostics, they cost a measured 2.3% of the
        // step, and the legacy baseline carries no equivalent instrumentation,
        // so leaving them on made every published pyMCPU-vs-legacy ratio
        // asymmetric. MCPU_HOT_COUNTERS=1 turns them back on.
        static const bool on = [] {
            const char* e = std::getenv("MCPU_HOT_COUNTERS");
            return e && e[0] == '1';
        }();
        return on;
    }

    static inline void note_mu_pair_r2(::mcpu::NeighborStats& nstats, float r2,
                                       float cut2) noexcept {
        ++nstats.mu_num_pair_distance_checks;
        if (r2 <= cut2)
            ++nstats.mu_num_pairs_within_rcut;
    }

    /// Build MovedCellGroups from moved indices + coords. O(n_moved).
    /// Returns false if any atom is OOB or group capacity exceeded (caller falls back).
    static bool build_moved_cell_groups(
        const std::vector<int>& moved_indices,
        const CoordsSoA& coords,
        const OpenCellGrid& grid,
        MuWorkspace::MovedCellGroups& out,
        std::vector<int>& cell_to_group)
    {
        out.n_groups = 0;
        const int n_cells = static_cast<int>(grid.num_cells());
        if (static_cast<int>(cell_to_group.size()) < n_cells)
            cell_to_group.assign(static_cast<size_t>(n_cells), -1);

        for (int i : moved_indices) {
            const int cell = grid.cell_of(coords.x[static_cast<size_t>(i)],
                                         coords.y[static_cast<size_t>(i)],
                                         coords.z[static_cast<size_t>(i)]);
            if (cell < 0) {
                std::fprintf(stderr,
                    "WARN: cell_of OOB for moved atom %d — falling back to "
                    "per-atom denselist walk\n",
                    i);
                for (int g = 0; g < out.n_groups; ++g)
                    cell_to_group[static_cast<size_t>(out.groups[g].cell_id)] = -1;
                out.n_groups = 0;
                return false;
            }
            int gi = cell_to_group[static_cast<size_t>(cell)];
            if (gi < 0) {
                if (out.n_groups >= MuWorkspace::MovedCellGroups::MAX_GROUPS) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        std::fprintf(stderr,
                            "WARN: MovedCellGroups overflow n_groups=%d "
                            "(MAX_GROUPS=%d) — denselist fallback "
                            "(further overflows suppressed)\n",
                            out.n_groups,
                            MuWorkspace::MovedCellGroups::MAX_GROUPS);
                    }
                    for (int g = 0; g < out.n_groups; ++g)
                        cell_to_group[static_cast<size_t>(out.groups[g].cell_id)] =
                            -1;
                    out.n_groups = 0;
                    return false;
                }
                gi = out.n_groups++;
                cell_to_group[static_cast<size_t>(cell)] = gi;
                out.groups[gi].cell_id = cell;
                out.groups[gi].count = 0;
            }
            auto& g = out.groups[gi];
            if (g.count >= OpenCellGrid::CELL_CAPACITY) {
                std::fprintf(stderr,
                    "WARN: MovedCellGroups cell %d count overflow — fallback\n",
                    cell);
                for (int g2 = 0; g2 < out.n_groups; ++g2)
                    cell_to_group[static_cast<size_t>(out.groups[g2].cell_id)] =
                        -1;
                out.n_groups = 0;
                return false;
            }
            g.moved_ids[g.count++] = i;
        }
        for (int g = 0; g < out.n_groups; ++g)
            cell_to_group[static_cast<size_t>(out.groups[g].cell_id)] = -1;
        return true;
    }

    /// Whether cell-pair denselist is enabled (config + env + moved size). O(1).
    /// Env ``MCPU_USE_CELL_PAIR=1`` forces on; ``=0`` forces off.
    /// ``MCPU_CELL_PAIR_MIN_MOVED`` overrides ``cell_pair_min_moved`` when set.
    static bool cell_pair_enabled(const NeighborConfig& cfg, int n_moved) {
        static const int kEnv = [] {
            const char* e = std::getenv("MCPU_USE_CELL_PAIR");
            if (e && e[0] == '0') return 0;
            if (e && e[0] == '1') return 1;
            return -1;
        }();
        static const int kMinEnv = [] {
            const char* e = std::getenv("MCPU_CELL_PAIR_MIN_MOVED");
            if (!e || !e[0]) return -1;
            return std::atoi(e);
        }();
        if (kEnv == 0) return false;
        if (kEnv == 1) return true;
        if (!cfg.use_cell_pair) return false;
        const int min_m =
            (kMinEnv >= 0) ? kMinEnv : cfg.cell_pair_min_moved;
        return n_moved >= min_m;
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
#if MCPU_FAST_MU_DELTA
        // Exact denselist cutoff from parameter matrices (refined after
        // type_params_ in cache_necessary_data). Default ON.
        apply_mu_denselist_cutoff();
        // ADDED: MM clash margin for rigid elision path
        if (const char* e = std::getenv("MCPU_MM_CLASH_MARGIN")) {
            char* end = nullptr;
            const float v = std::strtof(e, &end);
            if (end != e) mm_clash_margin_ = v;
        }
        // ADDED: double MM boundary clash check
        if (const char* e = std::getenv("MCPU_MM_DOUBLE_BOUNDARY")) {
            mm_double_boundary_ = (e[0] == '1');
        }
        // ADDED: three-layer eval — layered path is always active.
        // ADDED: topo_flag_ path (layered v2). Default on; =0 forces v1 branches.
        if (const char* e = std::getenv("MCPU_TOPO_FLAGS")) {
            use_topo_flags_ = (e[0] != '0');
        }
#endif
    }

    void MuPotential::set_topology_atom_meta(
        std::vector<int32_t> res_index,
        std::vector<uint8_t> is_sidechain,
        std::vector<uint8_t> atom_role,
        std::vector<uint8_t> res_class
    ) {
#if MCPU_FAST_MU_DELTA
        const size_t n = res_index.size();
        if (is_sidechain.size() != n || atom_role.size() != n ||
            res_class.size() != n) {
            throw std::invalid_argument(
                "MuPotential::set_topology_atom_meta: size mismatch");
        }
        if (!atom_types.empty() && atom_types.size() != n) {
            throw std::invalid_argument(
                "MuPotential::set_topology_atom_meta: size != atom_types");
        }
        res_index_ = std::move(res_index);
        is_sc_ = std::move(is_sidechain);
        atom_role_ = std::move(atom_role);
        res_class_ = std::move(res_class);
        layer1_meta_ready_ = true;
#else
        (void)res_index;
        (void)is_sidechain;
        (void)atom_role;
        (void)res_class;
#endif
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

#if MCPU_FAST_MU_DELTA
        if (layer1_meta_ready_ &&
            static_cast<int>(res_index_.size()) == n) {
            auto permute_i32 = [&](std::vector<int32_t>& v) {
                std::vector<int32_t> tmp(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    tmp[static_cast<size_t>(i)] =
                        v[static_cast<size_t>(perm.int_to_ext[static_cast<size_t>(i)])];
                }
                v.swap(tmp);
            };
            auto permute_u8 = [&](std::vector<uint8_t>& v) {
                std::vector<uint8_t> tmp(static_cast<size_t>(n));
                for (int i = 0; i < n; ++i) {
                    tmp[static_cast<size_t>(i)] =
                        v[static_cast<size_t>(perm.int_to_ext[static_cast<size_t>(i)])];
                }
                v.swap(tmp);
            };
            permute_i32(res_index_);
            permute_u8(is_sc_);
            permute_u8(atom_role_);
            permute_u8(res_class_);
        }
#endif

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
#if MCPU_FAST_MU_DELTA
        permute_flat(topo_flag_);
        num_atoms_cached_ = n;
        rebuild_type_params_from_matrices();
        if (layer1_meta_ready_) {
            build_clash_exceptions();
        }
#endif
#if !MCPU_FAST_MU_DELTA
        permute_flat(contact_cache);
#endif
        num_atoms_cached_ = n;
    }

#if MCPU_FAST_MU_DELTA
    /// Build Layer 3 CSR from final clash masks + Layer 1 rules. O(N²).
    void MuPotential::build_clash_exceptions() {
        const size_t N = static_cast<size_t>(num_atoms_cached_);
        if (!layer1_meta_ready_ || res_index_.size() != N ||
            topo_clash_mask_.size() != N * N) {
            clash_exceptions_ = SparseClashExceptions{};
            return;
        }

        std::vector<std::vector<int32_t>> exc_rows(N);
        size_t n_exceptions = 0;
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = i + 1; j < N; ++j) {
                if (topo_clash_mask_[i * N + j] != 0) continue;

                bool topo_clash = false, topo_contact = false;
                topology_pair_flags(
                    static_cast<int>(i), static_cast<int>(j), topo_clash,
                    topo_contact);
                (void)topo_contact;
                if (!topo_clash) continue;

                exc_rows[i].push_back(static_cast<int32_t>(j));
                exc_rows[j].push_back(static_cast<int32_t>(i));
                ++n_exceptions;
            }
        }

        clash_exceptions_.row_start.assign(N + 1, 0);
        clash_exceptions_.col.clear();
        clash_exceptions_.col.reserve(n_exceptions * 2);
        for (size_t i = 0; i < N; ++i) {
            std::sort(exc_rows[i].begin(), exc_rows[i].end());
            clash_exceptions_.row_start[i + 1] =
                clash_exceptions_.row_start[i] +
                static_cast<int32_t>(exc_rows[i].size());
            for (int32_t j : exc_rows[i])
                clash_exceptions_.col.push_back(j);
        }

        // CHANGED: gated behind MCPU_VERBOSE
        if (mcpu_verbose_enabled()) {
            std::fprintf(
                stderr, // CHANGED: gated behind MCPU_VERBOSE
                "INFO: Layer 3 sparse exceptions: %zu pairs (%.1f KB CSR)\n",
                n_exceptions,
                (clash_exceptions_.row_start.size() * 4 +
                 clash_exceptions_.col.size() * 4) /
                    1024.0);
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
                // ADDED: precompute hard_r = sqrt(hard_r2) for 3-decimal clash guard
                const float hr = (tp.hard_r2 > 0.f) ? std::sqrt(tp.hard_r2) : 0.f;
                tp.hard_tol_r2 = hard_tol_r2_from(hr);

                const size_t key =
                    static_cast<size_t>(ti) * NT + static_cast<size_t>(tj);
                const size_t key_sym =
                    static_cast<size_t>(tj) * NT + static_cast<size_t>(ti);
                type_params_[key] = tp;
                type_params_[key_sym] = tp;
            }
        }
        apply_mu_denselist_cutoff();
    }

    void MuPotential::bench_eval_pair_only(int n_iter) const {
        const int N = num_atoms_cached_;
        if (N < 2 || topo_flag_.empty()) {
            std::fprintf(stderr, "bench_eval_pair_only: not ready\n");
            return;
        }
        // Diverse pairs covering Layer 1 branches + distant contacts.
        struct Pair {
            int i, j;
            float r2;
        };
        std::vector<Pair> test_pairs;
        test_pairs.push_back({0, 1, 10.0f});  // likely same/near residue
        if (N > 6) test_pairs.push_back({0, 5, 10.0f});
        if (N > 51) test_pairs.push_back({0, 50, 20.0f});
        if (N > 70) test_pairs.push_back({10, 60, 15.0f});
        if (N > 200) test_pairs.push_back({100, 180, 12.0f});
        if (N > 500) {
            test_pairs.push_back({200, 450, 8.0f});
            test_pairs.push_back({300, 301, 4.0f});
            test_pairs.push_back({0, N - 1, 25.0f});
        }
        // Pad with scattered distant pairs
        for (int k = 0; k < 16 && N > 100; ++k) {
            const int i = (k * 97) % (N - 1);
            const int j = (i + 50 + k * 13) % N;
            if (i != j) test_pairs.push_back({i, j, 10.0f + static_cast<float>(k)});
        }

        // Warmup
        float sink = 0.f;
        bool clash = false;
        for (int w = 0; w < 1000; ++w) {
            for (const auto& p : test_pairs)
                sink += eval_pair(p.i, p.j, p.r2, &clash);
        }

        const auto t0 = std::chrono::steady_clock::now();
        for (int iter = 0; iter < n_iter; ++iter) {
            for (const auto& p : test_pairs)
                sink += eval_pair(p.i, p.j, p.r2, &clash);
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        const double n_calls =
            static_cast<double>(n_iter) * static_cast<double>(test_pairs.size());
        const char* mode =
            use_topo_flags_ && !topo_flag_.empty() ? "layered_v2" : "layered_v1";
        std::fprintf(
            stderr,
            "eval_pair (%s): %.2f ns/call  n_pairs=%zu n_iter=%d sink=%.1f\n",
            mode, ns / n_calls, test_pairs.size(), n_iter, sink);
    }

    void MuPotential::verify_layered_eval_consistency() const {
        if (!layer1_meta_ready_ || type_params_.empty()) {
            std::fprintf(
                stderr,
                "ERROR: verify_layered_eval_consistency: missing Layer1 / "
                "type_params_\n");
            return;
        }
        if (topo_flag_.empty()) {
            std::fprintf(
                stderr,
                "ERROR: verify_layered_eval_consistency: no topo_flag_\n");
            return;
        }
        size_t mismatches = 0;
        const size_t N = static_cast<size_t>(num_atoms_cached_);
        const float tests[] = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
        const bool saved = use_topo_flags_;
        for (float test_r2 : tests) {
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = i + 1; j < N; ++j) {
                    bool c1 = false, c2 = false;
                    const_cast<MuPotential*>(this)->use_topo_flags_ = false;
                    const float e1 = eval_pair_layered_v1(
                        static_cast<int>(i), static_cast<int>(j), test_r2,
                        &c1);
                    const_cast<MuPotential*>(this)->use_topo_flags_ = true;
                    const float e2 = eval_pair_layered_v2(
                        static_cast<int>(i), static_cast<int>(j), test_r2,
                        &c2);
                    if (std::fabs(e1 - e2) > 1e-5f || c1 != c2) {
                        ++mismatches;
                        if (mismatches <= 5) {
                            std::fprintf(
                                stderr,
                                "LAYERED_MISMATCH (v1 vs v2) r2=%.1f "
                                "i=%zu j=%zu v1=(%.4f,%d) v2=(%.4f,%d)\n",
                                test_r2, i, j, e1, static_cast<int>(c1), e2,
                                static_cast<int>(c2));
                        }
                    }
                }
            }
        }
        const_cast<MuPotential*>(this)->use_topo_flags_ = saved;
        // CHANGED: gated behind MCPU_VERBOSE
        if (mismatches == 0) {
            if (mcpu_verbose_enabled()) {
                std::fprintf(
                    stderr, // CHANGED: gated behind MCPU_VERBOSE
                    "INFO: eval_pair_layered consistency (v1 vs v2): PASS\n");
            }
        } else
            std::fprintf(
                stderr, "ERROR: eval_pair_layered: %zu mismatches\n",
                mismatches);
    }
#endif

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

#if MCPU_FAST_MU_DELTA
        topo_flag_.assign(n2, uint8_t{0});

        int max_t = -1;
        for (int t : atom_types) {
            if (t > max_t) max_t = t;
        }
        n_types_ = max_t + 1;
        const size_t NT = static_cast<size_t>(std::max(n_types_, 0));
        type_params_.assign(NT * NT, TypePairParams{});
#endif

#if !MCPU_FAST_MU_DELTA
        contact_cache.resize(n2);
#endif

        for (int i = 0; i < num_atoms; ++i) {
            for (int j = i + 1; j < num_atoms; ++j) {
                const int matrix_idx = i * num_atoms + j;
                const int matrix_idx_sym = j * num_atoms + i;

                bool check_clash = topo_clash_mask[static_cast<size_t>(matrix_idx)] != 0;
                bool check_contact = topo_contact_mask[static_cast<size_t>(matrix_idx)] != 0;

                if (check_clash) {
                    const float dist_sq = cv.dist2(i, j);
#if MCPU_FAST_MU_DELTA
                    if (dist_sq < hard_core_sq(i, j)) {
#else
                    if (dist_sq < hard_core_sq(matrix_idx)) {
#endif
                        check_clash = false;
                    }
                }

                topo_clash_mask_[static_cast<size_t>(matrix_idx)] =
                    topo_clash_mask_[static_cast<size_t>(matrix_idx_sym)] =
                        static_cast<uint8_t>(check_clash ? 1 : 0);

#if MCPU_FAST_MU_DELTA
                // Zero-energy contacts are treated as disabled (matches legacy ContactData)
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
                    const size_t key =
                        static_cast<size_t>(ti) * NT + static_cast<size_t>(tj);
                    const size_t key_sym =
                        static_cast<size_t>(tj) * NT + static_cast<size_t>(ti);
                    TypePairParams tp;
                    tp.hard_r2 = hc;
                    tp.contact_r2 = cd;
                    tp.energy = e_ij;
                    // ADDED: precompute hard_r for 3-decimal MM clash guard
                    const float hr = (hc > 0.f) ? std::sqrt(hc) : 0.f;
                    tp.hard_tol_r2 = hard_tol_r2_from(hr);
                    type_params_[key] = tp;
                    type_params_[key_sym] = tp;
                }
#else
                ContactData& cd = contact_cache[static_cast<size_t>(matrix_idx)];
                cd.check_clash = check_clash;
                cd.check_contact = check_contact
                    ? (contact_energies(matrix_idx) != 0.0f)
                    : false;
                cd.energy = cd.check_contact ? contact_energies(matrix_idx) : 0.0f;
                cd.contact_dist_sq = contact_dist_sq(matrix_idx);
                cd.hard_core_sq = hard_core_sq(matrix_idx);
                contact_cache[static_cast<size_t>(matrix_idx_sym)] = cd;

                topo_contact_mask_[static_cast<size_t>(matrix_idx)] =
                    topo_contact_mask_[static_cast<size_t>(matrix_idx_sym)] =
                        static_cast<uint8_t>(cd.check_contact ? 1 : 0);
#endif
            }
        }
#if MCPU_FAST_MU_DELTA
        if (layer1_meta_ready_) {
            build_clash_exceptions();
        } else {
            std::fprintf(
                stderr,
                "WARNING: set_topology_atom_meta was not called — layered "
                "eval Layer 1 / Layer 3 will misbehave.\n");
        }
        // CHANGED: gated behind MCPU_VERBOSE
        if (mcpu_verbose_enabled()) {
            std::fprintf(
                stderr, // CHANGED: gated behind MCPU_VERBOSE
                "INFO: topo_flag_ %.2f MB (N²×1 B) use_topo_flags=%d\n",
                topo_flag_.size() / (1024.0 * 1024.0),
                static_cast<int>(use_topo_flags_));
        }
        // CHANGED: exact denselist cutoff from type_params_ (default ON).
        apply_mu_denselist_cutoff();
#endif
    }

    float MuPotential::calculateEnergyBrute(const Context& context, const State& state) const {
        setup_mask_cache(context.getSystem());
        float total_energy = 0.0f;
        const int num_atoms = context.getSystem().getNumAtoms();
        const CoordView cv(state.coord_view());

        for (int i = 0; i < num_atoms; ++i) {
            for (int j = i + 1; j < num_atoms; ++j) {
#if MCPU_FAST_MU_DELTA
                const float dist_sq = cv.dist2(i, j);
                bool clash = false;
                const float e = eval_pair(i, j, dist_sq, &clash);
                if (clash) {
                    if (energy_mask_mode_cached_ == EnergyMaskMode::ClashOnly) {
                        continue;
                    }
                    return kHardCorePenalty;
                }
                total_energy += e;
#else
                const int matrix_idx = i * num_atoms + j;
                const auto& contact_info = contact_cache[static_cast<size_t>(matrix_idx)];
                if (!contact_info.check_contact && !contact_info.check_clash) continue;

                const float dist_sq = cv.dist2(i, j);
                if (contact_info.check_clash &&
                    is_hard_clash(dist_sq, contact_info.hard_core_sq)) {
                    return kHardCorePenalty;
                }
                if (contact_info.check_contact && dist_sq <= contact_info.contact_dist_sq) {
                    total_energy += contact_info.energy;
                }
#endif
            }
        }
        return total_energy;
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
#if MCPU_FAST_MU_DELTA
        // DEFAULT ON. Measured 1.35-1.45x with bit-identical trajectories on
        // chignolin/1igd/actin. MCPU_CONTACT_LIST=0 restores the re-measure path.
        static const bool kContactList = [] {
            const char* e = std::getenv("MCPU_CONTACT_LIST");
            return !(e && e[0] == '0');
        }();
        float delta;
        if (kContactList) {
            // The live list is only valid where the dense contiguous grid sees
            // every candidate pair and no residue is energy-masked.
            const bool usable =
                context.denseGridsActive() &&
                context.neighbors().muGrid().grid().use_contiguous() &&
                context.trial_in_bounds(new_state, patch) &&
                !context.getSystem().has_energy_mask();
            if (usable) {
                if (!old_state.mu_contact_list_ready) {
                    rebuild_contact_list(context, old_state);
                }
                delta = calculateEnergyChange_clist(
                    context, old_state, new_state, patch);
            } else {
                // Cannot maintain the list through this move -- rebuild it
                // from the accepted state afterwards rather than let it drift.
                delta = calculateEnergyChange_fast(
                    context, old_state, new_state, patch);
                old_state.mu_contact_invalidate();
                ++clist_fallbacks_;
                if (clist_fallbacks_ == 1 || (clist_fallbacks_ % 1000) == 0) {
                    std::fprintf(stderr,
                        "NOTE: contact-list rebuild #%llu (move outside the "
                        "dense grid). Each rebuild is O(N^2).\n",
                        static_cast<unsigned long long>(clist_fallbacks_));
                }
            }
        } else {
            delta = calculateEnergyChange_fast(
                context, old_state, new_state, patch);
        }
#else
        const float delta =
            calculateEnergyChange_legacy(context, old_state, new_state, patch);
#endif
        if (delta >= 0.5f * kHardCorePenalty) {
            return EnergyChangeResult::rejected(delta, RejectReason::StericClash);
        }
        return EnergyChangeResult::finite(delta);
    }

#if MCPU_FAST_MU_DELTA
    float MuPotential::calculateEnergyChange_fast(
        const Context& context,
        const State& old_state,
        const State& new_state,
        const ProposalPatch& patch
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
        // Fallback scans all atoms; skip amide H (not in historical Mu contact set).
        const auto& sys = context.getSystem();
        const int h_begin = sys.getTotalBBAtoms() + sys.getTotalOAtoms() + sys.getTotalSCAtoms();
        auto skip_fixed_h = [&](int j) {
            if (sys.residueContiguousLayout()) {
                return sys.is_amide_h_atom(j) && !is_moved[static_cast<size_t>(j)];
            }
            return j >= h_begin && !is_moved[static_cast<size_t>(j)];
        };

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

        ws.ensure_stamp_capacity(num_atoms); // kept for evaluate_neighbor_grids / legacy paths

        const bool skip_rigid_mm =
            patch.is_rigid && context.neighborConfig().skip_rigid_mm;

        // Hot pair loop #3: Fallback moved-vs-all (out-of-box / no dense grid).
        // Out-of-bounds trial under AUTO_EXPAND: no dense-grid rebuild; moved-vs-all fallback.
        if (ws.use_trial_fallback || !context.denseGridsActive()) {
            auto& nstats = const_cast<NeighborStats&>(context.neighborStats());
            const float cut2 = contact_cutoff_sq_;
            // Old energy: moved at old pos vs all (accepted coords for partners)
            NeighborFallback::for_each_moved_neighbor(
                old_state.coords_soa, old_state.coords_soa, moved_indices, is_moved, cut2,
                /*is_rigid=*/skip_rigid_mm,
                [&](int i, int j, float r2) {
                    if (skip_fixed_h(j)) return;
                    note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                    delta_E -= eval_pair(i, j, r2, nullptr);
                });
            // New energy: moved at new pos vs fixed(old); MM skipped when rigid.
            NeighborFallback::for_each_moved_neighbor(
                new_state.coords_soa, old_state.coords_soa, moved_indices, is_moved, cut2,
                /*is_rigid=*/true, // always skip MM here; handled below if needed
                [&](int i, int j, float r2) {
                    if (is_moved[static_cast<size_t>(j)]) return; // fixed only here
                    if (skip_fixed_h(j)) return;
                    note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                    bool local_clash = false;
                    delta_E += eval_pair(i, j, r2, &local_clash);
                    if (local_clash) clash = true;
                });
            if (!clash && !skip_rigid_mm) {
                for (size_t a = 0; a < moved_indices.size() && !clash; ++a) {
                    const int i = moved_indices[a];
                    for (size_t b = a + 1; b < moved_indices.size(); ++b) {
                        const int j = moved_indices[b];
                        const float r2 = new_state.coord_view().dist2(i, j);
                        note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                        bool local_clash = false;
                        delta_E += eval_pair(i, j, r2, &local_clash);
                        if (local_clash) { clash = true; break; }
                    }
                }
            } else if (skip_rigid_mm) {
                // Energy MM elided (ΔE≈0); still clash-check new MM geometry.
                // FIXED: 3-decimal rounding — PDB precision MM clash guard
                const std::uint64_t n =
                    static_cast<std::uint64_t>(moved_indices.size());
                nstats.elided_rigid_mm += (n * (n > 0 ? n - 1 : 0)) / 2ull;
                const CoordView cnew_mm(new_state.coord_view());
                for (size_t a = 0; a < moved_indices.size() && !clash; ++a) {
                    const int i = moved_indices[a];
                    for (size_t b = a + 1; b < moved_indices.size(); ++b) {
                        const int j = moved_indices[b];
                        if (rigid_mm_pair_clashes(i, j, cnew_mm.dist2(i, j))) {
                            clash = true;
                            break;
                        }
                    }
                }
            }
            if (clash) {
                ws.clear();
                return kHardCorePenalty;
            }
            return delta_E;
        }

        // Optional Verlet path for KIC/SC when Integrator marked VerletPreferred and list is usable
        auto& nstats = const_cast<NeighborStats&>(context.neighborStats());
        const bool use_verlet =
            (ws.neighbor_mode == NeighborMode::VerletPreferred)
            && (context.neighborConfig().skin > 0.f)
            && context.verletContact().trial_usable(
                   moved_indices, old_state.coords_soa, new_state.coords_soa);
        if (use_verlet) ++nstats.num_verlet_used();

        CellListMC* moved_grid = nullptr;
        // Rigid pivot: skip moved_new_grid — MM ΔE is identically zero.
        if (!moved_indices.empty() && !use_verlet && !skip_rigid_mm) {
            ws.ensure_moved_grid(mu_exact_cutoff_, num_atoms);
            moved_grid = ws.moved_new_grid.get();
            // Share contact-grid bounds so inserts index correctly (open, no wrap)
            const auto& gb = ns.muGrid().grid().bounds();
            if (gb.valid) {
                NeighborConfig cfg = context.neighborConfig();
                const float mu_cell = effective_mu_cell_size_A(
                    mu_exact_cutoff_, /*skin=*/0.f, cfg);
                // Match accepted Mu denselist: r_mu cell/query (not skin-inflated).
                // CRITICAL: configure() assigns n_cells×CAPACITY packed arrays (~MB).
                // Only reconfigure when geometry changes — was called every SC/KIC
                // denselist step and dominated SC Mu (~80 µs fixed overhead).
                moved_grid->set_cutoff(mu_exact_cutoff_);
                if (!moved_grid->grid().matches_geometry(gb, mu_cell, cfg)) {
                    moved_grid->configure(gb, cfg, /*skin=*/0.f, mu_cell);
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

        if (use_verlet) {
            // Hot pair loop #2: Verlet CSR enumeration (moved atoms × undirected neighbors).
            const VerletList& vl = context.verletContact();
            const CoordView cold(old_state.coord_view());
            const CoordView cnew(new_state.coord_view());
            for (int i : moved_indices) {
                {
                    const int a = vl.offsets[static_cast<size_t>(i)];
                    const int b = vl.offsets[static_cast<size_t>(i) + 1];
                    for (int k = a; k < b; ++k) {
                        const int j = vl.neighbors[static_cast<size_t>(k)];
                        ++nstats.mu_num_candidates_iterated;
                        if (is_moved[static_cast<size_t>(j)]) {
                            if (skip_rigid_mm) {
                                ++nstats.elided_rigid_mm;
                                continue;
                            }
                            if (i > j) continue;
                        }
                        const float r2 = cold.dist2(i, j);
                        note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                        // CHANGED: gate like denselist — hard_core/contact ≤ 6 Å.
                        if (r2 <= contact_cutoff_sq_)
                            delta_E -= eval_pair(i, j, r2, nullptr);
                    }
                }
                {
                    const int a = vl.offsets[static_cast<size_t>(i)];
                    const int b = vl.offsets[static_cast<size_t>(i) + 1];
                    for (int k = a; k < b && !clash; ++k) {
                        const int j = vl.neighbors[static_cast<size_t>(k)];
                        ++nstats.mu_num_candidates_iterated;
                        if (is_moved[static_cast<size_t>(j)]) {
                            if (skip_rigid_mm) {
                                ++nstats.elided_rigid_mm;
                                // FIXED: 3-decimal rounding — PDB precision MM clash guard
                                if (i < j &&
                                    rigid_mm_pair_clashes(
                                        i, j, cnew.dist2(i, j))) {
                                    clash = true;
                                }
                                continue;
                            }
                            if (i > j) continue;
                            const float r2 = cnew.dist2(i, j);
                            bool local_clash = false;
                            note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                            if (r2 <= contact_cutoff_sq_) {
                                delta_E += eval_pair(i, j, r2, &local_clash);
                            }
                            if (local_clash) clash = true;
                        } else {
                            const float r2 = cnew.dist2(i, j);
                            bool local_clash = false;
                            note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                            if (r2 <= contact_cutoff_sq_) {
                                delta_E += eval_pair(i, j, r2, &local_clash);
                            }
                            if (local_clash) clash = true;
                        }
                    }
                }
                if (clash) break;
            }
            if (clash) {
                ws.clear();
                return kHardCorePenalty;
            }
            return delta_E;
        }

        // Hot pair loop #1: classic denselist OpenCellGrid (skin=0 default).
        // CSR pack-all-then-SIMD-r² was tried and reverted (wall ~150→459 µs):
        // packing before clash exit overflows (>98k pairs) and double-walks;
        // cell walk is ~71% of pivot Mu so splitting r² cannot win.
        const CoordView cold(old_state.coord_view());
        const CoordView cnew(new_state.coord_view());

        // Optional diagnostic: O(n_moved²) rigid MM ΔE (should be ~0).
        static const bool kDebugMmDelta = [] {
            const char* e = std::getenv("MCPU_DEBUG_MM_DELTA");
            return e && e[0] == '1';
        }();
        if (kDebugMmDelta && skip_rigid_mm && moved_indices.size() > 1) {
            float mm = 0.f;
            for (size_t a = 0; a < moved_indices.size(); ++a) {
                const int i = moved_indices[a];
                for (size_t b = a + 1; b < moved_indices.size(); ++b) {
                    const int j = moved_indices[b];
                    const float r2o = cold.dist2(i, j);
                    const float r2n = cnew.dist2(i, j);
                    if (r2o <= contact_cutoff_sq_)
                        mm -= eval_pair(i, j, r2o, nullptr);
                    if (r2n <= contact_cutoff_sq_) {
                        bool lc = false;
                        mm += eval_pair(i, j, r2n, &lc);
                        (void)lc;
                    }
                }
            }
            std::fprintf(stderr, "mm_delta=%.8g n_moved=%zu\n",
                         static_cast<double>(mm), moved_indices.size());
        }

        // Diagnostic: phased denselist for pivot Mu (MCPU_PIVOT_MU_BREAKDOWN=1).
        // Collect → r² filter → eval_pair; measures each phase. Not production.
        static const bool kPivotMuBreakdown = [] {
            const char* e = std::getenv("MCPU_PIVOT_MU_BREAKDOWN");
            return e && e[0] == '1';
        }();
        const bool do_pivot_breakdown =
            kPivotMuBreakdown &&
            (ws.move_kind == MoveKind::Pivot || patch.is_rigid);

        if (do_pivot_breakdown) {
            using Clock = std::chrono::steady_clock;
            const auto t_setup0 = Clock::now();

            auto& cands = ws.pivot_cand_scratch;
            auto& incut = ws.pivot_incut_scratch;
            cands.clear();
            incut.clear();
            // Per-atom staging so we can early-exit on clash like the fused path.
            cands.reserve(512);
            incut.reserve(64);

            std::uint64_t ns_walk = 0, ns_r2 = 0, ns_eval = 0;
            std::uint64_t n_cand = 0, n_incut = 0;
            auto to_ns = [](Clock::time_point a, Clock::time_point b) {
                return static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(b - a)
                        .count());
            };

            const auto t0 = Clock::now();
            // Per-atom phased denselist: walk → r² → eval, matching fused early-exit.
            // Diagnostic path (MCPU_PIVOT_MU_BREAKDOWN=1); production uses packed spans below.
            // O(n_moved × avg_neighbors) total; abort on hard-core clash.
            for (int i : moved_indices) {
                const float ox = cold.x(i), oy = cold.y(i), oz = cold.z(i);
                const float nx = cnew.x(i), ny = cnew.y(i), nz = cnew.z(i);

                // --- Old subtract ---
                cands.clear();
                auto tw0 = Clock::now();
                ns.for_each_mu_candidate(ox, oy, oz, [&](int j) {
                    if (j == i) return;
                    if (is_moved[static_cast<size_t>(j)]) {
                        if (skip_rigid_mm) {
                            ++nstats.elided_rigid_mm;
                            return;
                        }
                        if (i > j) return;
                    }
                    cands.push_back(
                        MuWorkspace::PivotCand{i, j, /*is_new=*/0});
                });
                auto tw1 = Clock::now();
                ns_walk += to_ns(tw0, tw1);

                incut.clear();
                auto tr0 = Clock::now();
                for (const auto& c : cands) {
                    const float r2 = cold.dist2(c.i, c.j);
                    note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                    if (r2 <= contact_cutoff_sq_) {
                        incut.push_back(
                            MuWorkspace::PivotInCut{c.i, c.j, r2, c.is_new});
                    }
                }
                auto tr1 = Clock::now();
                ns_r2 += to_ns(tr0, tr1);
                n_cand += cands.size();
                n_incut += incut.size();

                auto te0 = Clock::now();
                for (const auto& p : incut) {
                    delta_E -= eval_pair(p.i, p.j, p.r2, nullptr);
                }
                auto te1 = Clock::now();
                ns_eval += to_ns(te0, te1);

                // --- New add (static + optional moved_grid) ---
                cands.clear();
                tw0 = Clock::now();
                ns.for_each_mu_candidate(nx, ny, nz, [&](int j) {
                    if (j == i) return;
                    if (is_moved[static_cast<size_t>(j)]) return;
                    cands.push_back(
                        MuWorkspace::PivotCand{i, j, /*is_new=*/1});
                });
                if (moved_grid) {
                    moved_grid->for_each_neighbor(nx, ny, nz, [&](int j) {
                        ++nstats.mu_num_candidates_iterated;
                        if (j == i) return;
                        if (i > j) return;
                        cands.push_back(
                            MuWorkspace::PivotCand{i, j, /*is_new=*/1});
                    });
                }
                tw1 = Clock::now();
                ns_walk += to_ns(tw0, tw1);

                incut.clear();
                tr0 = Clock::now();
                for (const auto& c : cands) {
                    const float r2 = cnew.dist2(c.i, c.j);
                    note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                    if (r2 <= contact_cutoff_sq_) {
                        incut.push_back(
                            MuWorkspace::PivotInCut{c.i, c.j, r2, c.is_new});
                    }
                }
                tr1 = Clock::now();
                ns_r2 += to_ns(tr0, tr1);
                n_cand += cands.size();
                n_incut += incut.size();

                te0 = Clock::now();
                for (const auto& p : incut) {
                    bool local_clash = false;
                    delta_E += eval_pair(p.i, p.j, p.r2, &local_clash);
                    if (local_clash) {
                        clash = true;
                        break;
                    }
                }
                te1 = Clock::now();
                ns_eval += to_ns(te0, te1);
                if (clash) break;
            }
            const auto t_end = Clock::now();

            nstats.pivot_mu_overhead_ns += to_ns(t_setup0, t0);
            nstats.pivot_mu_cell_walk_ns += ns_walk;
            nstats.pivot_mu_r2_filter_ns += ns_r2;
            nstats.pivot_mu_eval_pair_ns += ns_eval;
            // Unaccounted gap inside the timed region (timer call overhead, etc.).
            const auto accounted = ns_walk + ns_r2 + ns_eval;
            const auto span = to_ns(t0, t_end);
            if (span > accounted)
                nstats.pivot_mu_overhead_ns += (span - accounted);
            nstats.pivot_mu_candidates += n_cand;
            nstats.pivot_mu_in_cutoff += n_incut;
            ++nstats.pivot_mu_n_steps;

            if (clash) {
                ws.clear();
                ws.clear_moved_grid();
                return kHardCorePenalty;
            }
            ws.clear_moved_grid();
            return delta_E;
        }

        // CHANGED: cell-pair inversion (optional) + fused packed denselist fallback.
        const OpenCellGrid& mu_grid_ref = ns.muGrid().grid();
        const bool use_span = mu_grid_ref.use_contiguous();
        const bool want_cell_pair =
            use_span &&
            cell_pair_enabled(context.neighborConfig(),
                              static_cast<int>(moved_indices.size()));
        static constexpr bool kPrefetchLayered = true;

        bool used_cell_pair = false;
        if (want_cell_pair) {
#if defined(MCPU_CP_BREAKDOWN)
            // One move in kCpSample is timed; see the note at the top of file.
            static std::uint64_t cp_move_seq = 0;
            const bool cp_on = ((cp_move_seq++ % kCpSample) == 0);
            if (cp_on) ++nstats.cp_n_steps;
#else
            constexpr bool cp_on = false;
#endif
            // MEASURED AND REJECTED: legacy's old-side-skip gate does not port.
            //
            // Legacy gates its entire old-cell pass on one pointer compare
            // (contacts.h:278, comparing the new and old stencils' self cell) and
            // skips it when the atom did not change cell. Two reasons that cannot
            // be copied here:
            //
            // 1. STRUCTURAL. Legacy can skip the old side because it keeps a
            //    persistent per-pair contact-bit cache (data[][], O(N^2) memory --
            //    ~512 MB at 4000 atoms, its scaling wall) and computes the delta by
            //    comparing the new distance against the cached bit. It never needs
            //    E_old. pyMCPU has no such cache by design, so it must evaluate the
            //    old neighbourhood whatever the cell membership. The most available
            //    here is MERGING the two walks when the cell sets coincide, which
            //    saves the walk but not the evaluation.
            // 2. STATISTICAL. Measured fraction of cell-pair moves whose moved-atom
            //    cell membership is unchanged: chignolin 20.4%, barnase 10.4%,
            //    sce 10.4% (MCPU_CELLGATE_PROBE, 60k steps each). So the merge would
            //    apply to ~1 move in 5 at best, saving at most half the stencil walk
            //    on those -- a few percent of the step -- against restructuring a
            //    loop documented below as having cost 13% wall the last time it was
            //    reorganised. Negative expected value.
            //
            // CHANGED: cell-pair inversion — group moved atoms by old/new cell.
            const bool ok_old = build_moved_cell_groups(
                moved_indices, old_state.coords_soa, mu_grid_ref,
                ws.old_moved_groups, ws.cell_to_group_scratch);
            const bool ok_new = ok_old && build_moved_cell_groups(
                moved_indices, new_state.coords_soa, mu_grid_ref,
                ws.new_moved_groups, ws.cell_to_group_scratch);
            if (ok_old && ok_new) {
                used_cell_pair = true;
                ++nstats.pivot_mu_cell_pair_evals;
                nstats.pivot_mu_n_groups +=
                    static_cast<std::uint64_t>(ws.old_moved_groups.n_groups) +
                    static_cast<std::uint64_t>(ws.new_moved_groups.n_groups);
                for (int g = 0; g < ws.old_moved_groups.n_groups; ++g)
                    nstats.pivot_mu_group_atoms +=
                        static_cast<std::uint64_t>(
                            ws.old_moved_groups.groups[g].count);
                for (int g = 0; g < ws.new_moved_groups.n_groups; ++g)
                    nstats.pivot_mu_group_atoms +=
                        static_cast<std::uint64_t>(
                            ws.new_moved_groups.groups[g].count);

                auto eval_groups =
                    [&](const MuWorkspace::MovedCellGroups& groups,
                        const CoordView& cview, bool is_new_side,
                        double& acc) -> bool {
                    // Timed from inside the lambda so the call sites stay
                    // byte-for-byte original. Wrapping them instead cost 13%
                    // wall with the timers compiled OUT -- the restructuring,
                    // not the rdtsc, was the observer effect.
                    CP_SCOPE(is_new_side ? nstats.cp_new_walk_ns
                                         : nstats.cp_old_walk_ns,
                             cp_on);
                    const bool order_dense =
                        is_new_side && clash_order_by_density_enabled();
                    const bool hot_cnt = hot_counters_enabled();
                    // ADDED: one-shot nc-sharing diagnostic (MCPU_NC_SHARE_DIAG=1)
                    static const bool kNcShareDiag = [] {
                        const char* e = std::getenv("MCPU_NC_SHARE_DIAG");
                        return e && e[0] == '1';
                    }();
                    if (kNcShareDiag && is_new_side &&
                        groups.n_groups >= 40) {
                        static bool printed = false;
                        if (!printed) {
                            printed = true;
                            thread_local std::vector<int> visit;
                            const int n_cells =
                                static_cast<int>(mu_grid_ref.num_cells());
                            if (static_cast<int>(visit.size()) < n_cells)
                                visit.assign(static_cast<size_t>(n_cells), 0);
                            int unique = 0;
                            long long sum = 0;
                            for (int gi = 0; gi < groups.n_groups; ++gi) {
                                mu_grid_ref.for_each_neighbor_cell_of(
                                    groups.groups[gi].cell_id, [&](int nc) {
                                        if (visit[static_cast<size_t>(nc)] ==
                                            0)
                                            ++unique;
                                        ++visit[static_cast<size_t>(nc)];
                                    });
                            }
                            for (int c = 0; c < n_cells; ++c) {
                                if (visit[static_cast<size_t>(c)] > 0) {
                                    sum += visit[static_cast<size_t>(c)];
                                    visit[static_cast<size_t>(c)] = 0;
                                }
                            }
                            const double avg =
                                unique > 0
                                    ? static_cast<double>(sum) / unique
                                    : 0.0;
                            // CHANGED: gated behind MCPU_VERBOSE
                            if (mcpu_verbose_enabled()) {
                                std::fprintf(stderr, // CHANGED: gated behind MCPU_VERBOSE
                                    "INFO: nc-share diag (new-side pivot): "
                                    "n_groups=%d unique_nc=%d "
                                    "avg_groups_per_nc=%.2f\n",
                                    groups.n_groups, unique, avg);
                            }
                        }
                    }
                    // Span prefetch / cell-outer were measured neutral → removed.
                    for (int iter = 0; iter < groups.n_groups; ++iter) {
                        int gi = iter;
                        const auto& mc = groups.groups[gi];
                        // MEASURED AND REVERTED: point-to-cell distance cull.
                        // A neighbour cell whose nearest point is beyond the cutoff
                        // cannot hold a partner in range, so it can be dropped
                        // bit-identically (verified: 0 in-cutoff pairs lost, same
                        // energy to the last digit, same coordinates, on all three
                        // systems). It removed 24.8% of cell visits and 21.8% of r2
                        // checks at sce, 21.3%/20.9% at barnase.
                        //
                        // It was still SLOWER: -1.4% sce, -0.9% barnase, -3.0%
                        // chignolin (7 interleaved repeats, spread <=0.4%). Placing
                        // the test inside the ki/a loops was worse still (-1.9% /
                        // -1.6% / -11.3%), because a ~25%-taken branch in the
                        // innermost loop mispredicts on every (moved atom, cell)
                        // pair. Hoisting it into this list build removed the branch
                        // and it STILL did not pay.
                        //
                        // Conclusion: cell visits and r2 checks are not what Mu's
                        // time is made of, so removing a fifth of them buys nothing.
                        // See mu_span_slots_scanned vs mu_num_pair_distance_checks
                        // for where it actually goes. Do not re-attempt geometric
                        // stencil culling without first moving that ratio.
                        //
                        // Local stencil copy (≤27); optional density sort for new-side.
                        int nc_buf[NeighborCellList::kCap];
                        int n_nc = 0;
                        mu_grid_ref.for_each_neighbor_cell_of(
                            mc.cell_id, [&](int nc) {
                                if (n_nc < NeighborCellList::kCap)
                                    nc_buf[n_nc++] = nc;
                            });
                        if (order_dense && n_nc > 1) {
                            // Insertion sort by cell_atom_count descending.
                            for (int i = 1; i < n_nc; ++i) {
                                const int key = nc_buf[i];
                                const int key_c =
                                    mu_grid_ref.cell_atom_count(key);
                                int j = i - 1;
                                while (j >= 0 &&
                                       mu_grid_ref.cell_atom_count(
                                           nc_buf[j]) < key_c) {
                                    nc_buf[j + 1] = nc_buf[j];
                                    --j;
                                }
                                nc_buf[j + 1] = key;
                            }
                        }
                        // i-side block: gather this group's moved-atom coordinates
                        // ONCE instead of re-fetching them inside every neighbour
                        // cell. mc has ~6 atoms and a group visits ~4.3 cells, so
                        // the scattered SoA loads were ~4.3x redundant. This is the
                        // one piece of the GROMACS i-cluster idea that transfers
                        // directly to a rebuild-every-step MC delta. Bit-identical:
                        // same values, fetched once.
                        float gx[OpenCellGrid::CELL_CAPACITY];
                        float gy[OpenCellGrid::CELL_CAPACITY];
                        float gz[OpenCellGrid::CELL_CAPACITY];
                        for (int a = 0; a < mc.count; ++a) {
                            const int ia = mc.moved_ids[a];
                            gx[a] = cview.x(ia);
                            gy[a] = cview.y(ia);
                            gz[a] = cview.z(ia);
                        }
                        for (int ki = 0; ki < n_nc; ++ki) {
                            const int nc = nc_buf[ki];
                            if (clash) break;
                            const int* cids;
                            int n_static;
                            const float* __restrict__ cx;
                            const float* __restrict__ cy;
                            const float* __restrict__ cz;
                            const auto sp =
                                mu_grid_ref.cell_atoms_span(nc);
                            cids = sp.first;
                            n_static = sp.second;
                            if (n_static == 0) {
                                if (hot_cnt) ++nstats.pivot_mu_cell_pairs_empty;
                                continue;
                            }
                            cx = mu_grid_ref.cell_x_span(nc);
                            cy = mu_grid_ref.cell_y_span(nc);
                            cz = mu_grid_ref.cell_z_span(nc);
                            if (hot_cnt) {
                                ++nstats.pivot_mu_cell_pairs;
                                ++nstats.neighbor_num_cell_visits;
                            }

                                // CHANGED: cell-pair inversion — moved mask once per nc
                                std::uint64_t moved_bits = 0ull;
                                {
                                    CP_SCOPE(nstats.cp_movedbits_ns, cp_on);
                                    for (int m = 0; m < n_static; ++m) {
                                        if (is_moved[static_cast<size_t>(cids[m])])
                                            moved_bits |= (1ull << m);
                                    }
                                }

                                // ADDED: the old-side skip_mask is provably
                                // INDEPENDENT of `a` whenever skip_rigid_mm is on.
                                // In that branch every moved slot gets masked, and
                                // the j == i case is subsumed: `i` comes from
                                // mc.moved_ids, so is_moved[i] is true, so if i
                                // occupies slot m then moved_bits already carries bit
                                // m. The O(n_static) per-(moved atom, cell) loop
                                // below therefore evaluates to exactly `moved_bits`
                                // every single time -- the same simplification
                                // already made on the new side. Only the
                                // !skip_rigid_mm branch really depends on `a`,
                                // through the `i > j` keep-once tiebreak.
                                const bool uniform_skip =
                                    (is_new_side || skip_rigid_mm) &&
                                    uniform_skipmask_enabled();
                                // i lies in this cell only when it is the group's own
                                // cell, and then exactly once.
                                const int self_here = (nc == mc.cell_id) ? 1 : 0;
                                for (int a = 0; a < mc.count && !clash; ++a) {
                                    const int i = mc.moved_ids[a];
                                    const float xi = gx[a];
                                    const float yi = gy[a];
                                    const float zi = gz[a];

                                    std::uint64_t skip_mask = 0ull;
                                    {
                                    CP_SCOPE(nstats.cp_skipmask_ns, cp_on);
                                    if (uniform_skip) {
                                        skip_mask = moved_bits;
                                        if (!is_new_side && hot_cnt) {
                                            // keep the counter's exact meaning:
                                            // moved slots here, minus self
                                            nstats.elided_rigid_mm +=
                                                static_cast<std::uint64_t>(
                                                    __builtin_popcountll(moved_bits) -
                                                    self_here);
                                        }
                                    } else if (is_new_side) {
                                        // New: skip all moved + self.
                                        //
                                        // FIXED: the self-scan below was dead. `i`
                                        // always comes from mc.moved_ids, so it is a
                                        // moved atom; if it also appears in this cell
                                        // at slot m then is_moved[cids[m]] is true and
                                        // moved_bits already carries bit m. The loop
                                        // therefore only ever re-set a bit that was
                                        // set. Removing it drops an O(n_static) scan
                                        // per (moved atom, neighbour cell) -- and the
                                        // skip-mask phase is 17% of new-side cell-pair
                                        // time at actin. Provably bit-identical.
                                        skip_mask = moved_bits;
                                    } else {
                                        // Old: per-i self / MM keep-once.
                                        for (int m = 0; m < n_static; ++m) {
                                            const int j = cids[m];
                                            if (j == i) {
                                                skip_mask |= (1ull << m);
                                                continue;
                                            }
                                            if (moved_bits & (1ull << m)) {
                                                if (skip_rigid_mm) {
                                                    if (hot_cnt) ++nstats.elided_rigid_mm;
                                                    skip_mask |= (1ull << m);
                                                } else if (i > j) {
                                                    skip_mask |= (1ull << m);
                                                }
                                            }
                                        }
                                    }
                                    }  // end skip_mask timing scope

                                    // CHANGED: r2 is no longer computed eagerly for
                                    // the whole span here -- it is computed inside the
                                    // collect loop below, for the slots that survive
                                    // skip_mask only.
                                    //
                                    // The eager `#pragma GCC ivdep` loop filled all
                                    // n_static slots, but r2_buf is read ONLY through
                                    // in_cut[], which the collect loop populates from
                                    // non-skipped slots; the MM clash guard between
                                    // them deliberately uses cnew.dist2 instead (see
                                    // its comment). So every skipped slot's r2 was
                                    // computed and discarded. Measured share of span
                                    // slots discarded: 6.8% chignolin, 51.5% barnase,
                                    // 63.3% sce -- it grows with the molecule because
                                    // a pivot moves a spatially compact chain segment,
                                    // so a moved atom's neighbour cells are themselves
                                    // largely moved, and skip_mask == moved_bits on
                                    // the new side.
                                    //
                                    // Bit-identical: __builtin_ctzll walks set bits in
                                    // increasing m, the same order the old `for m`
                                    // loop used, computing the same expression on the
                                    // same slots. Only slots whose r2 was never read
                                    // are dropped.
                                    float r2_buf[OpenCellGrid::CELL_CAPACITY];
                                    if (hot_cnt)
                                        nstats.mu_span_slots_scanned +=
                                            static_cast<std::uint64_t>(n_static);
                                    // A/B switch for the change described above.
                                    // MCPU_LIVE_R2=0 restores the eager full-span
                                    // loop so the two can be interleaved on one node.
                                    static const bool kLiveR2 = [] {
                                        const char* e = std::getenv("MCPU_LIVE_R2");
                                        return !(e && e[0] == '0');
                                    }();
                                    if (!kLiveR2) {
#pragma GCC ivdep
                                        for (int m = 0; m < n_static; ++m) {
                                            const float dx = xi - cx[m];
                                            const float dy = yi - cy[m];
                                            const float dz = zi - cz[m];
                                            r2_buf[m] =
                                                dx * dx + dy * dy + dz * dz;
                                        }
                                    }
#if defined(MCPU_CP_BREAKDOWN)
                                    // count only on SAMPLED moves so the
                                    // denominator matches cp_n_steps
                                    if (cp_on) {
                                        if (is_new_side)
                                            nstats.cp_new_r2_checks +=
                                                static_cast<std::uint64_t>(n_static);
                                        else
                                            nstats.cp_old_r2_checks +=
                                                static_cast<std::uint64_t>(n_static);
                                    }
#endif

                                    // When skip_rigid_mm, energy MM is elided but
                                    // clash must still be detected (float32 rotation
                                    // can push a near-boundary MM pair under hard_r).
                                    // FIXED: 3-decimal rounding — PDB precision MM clash guard
                                    // (optional margin / double-boundary remain additive).
                                    const bool check_mm_clash =
                                        is_new_side && skip_rigid_mm;
                                    if (check_mm_clash) {
                                        CP_SCOPE(nstats.cp_mmguard_ns, cp_on);
                                        for (int m = 0;
                                             m < n_static && !clash; ++m) {
                                            if (!(moved_bits & (1ull << m)))
                                                continue;
                                            const int j = cids[m];
                                            if (j == i || i > j) continue;
                                            if (mm_double_boundary_) {
                                                // ADDED: double MM boundary
                                                const float r2_old =
                                                    cold.dist2(i, j);
                                                const size_t pidx =
                                                    static_cast<size_t>(i) *
                                                        static_cast<size_t>(
                                                            num_atoms_cached_) +
                                                    static_cast<size_t>(j);
                                                const uint8_t flag =
                                                    topo_flag_[pidx];
                                                if (!(flag & 1u)) continue;
                                                const int ti =
                                                    atom_types[static_cast<
                                                        size_t>(i)];
                                                const int tj =
                                                    atom_types[static_cast<
                                                        size_t>(j)];
                                                if (ti < 0 || tj < 0 ||
                                                    n_types_ <= 0)
                                                    continue;
                                                const auto& g =
                                                    type_params_
                                                        [static_cast<size_t>(ti) *
                                                             static_cast<size_t>(
                                                                 n_types_) +
                                                         static_cast<size_t>(
                                                             tj)];
                                                const float hard = g.hard_r2;
                                                const bool near =
                                                    r2_old >=
                                                        hard -
                                                            mm_double_boundary_sq_ &&
                                                    r2_old <=
                                                        hard +
                                                            mm_double_boundary_sq_;
                                                const float r2_new =
                                                    cnew.dist2(i, j);
                                                if (near) {
                                                    if (rigid_mm_clash_3decimal(
                                                            r2_new, g.hard_tol_r2)) {
                                                        clash = true;
                                                    }
                                                } else if (rigid_mm_clash_3decimal(
                                                               r2_new,
                                                               g.hard_tol_r2)) {
                                                    clash = true;
                                                }
                                            } else if (mm_clash_margin_ > 0.f) {
                                                // ADDED: MM clash margin on NEW MM
                                                const float r2_new =
                                                    cnew.dist2(i, j);
                                                const float r2_adj =
                                                    r2_new - mm_clash_margin_;
                                                bool lc = false;
                                                (void)eval_pair(
                                                    i, j,
                                                    r2_adj > 0.f ? r2_adj : 0.f,
                                                    &lc);
                                                if (lc) clash = true;
                                            } else {
                                                // Default: 3-decimal hard_r guard
                                                // FIXED: use cnew.dist2 — r2_buf is
                                                // new_i vs old_j and false-positives.
                                                const float r2_new =
                                                    cnew.dist2(i, j);
                                                if (rigid_mm_pair_clashes(
                                                        i, j, r2_new)) {
                                                    clash = true;
                                                }
                                            }
                                        }
                                        if (clash) break;
                                    }

                                    // Collect in-cutoff partners; prefetch layered tables.
                                    CP_SCOPE(is_new_side ? nstats.cp_new_eval_ns
                                                         : nstats.cp_old_eval_ns,
                                             cp_on);
                                    int in_cut[OpenCellGrid::CELL_CAPACITY];
                                    int n_in = 0;
                                    {
                                        CP_SCOPE(is_new_side
                                                     ? nstats.cp_new_r2_ns
                                                     : nstats.cp_old_r2_ns,
                                                 cp_on);
                                        std::uint64_t live = ~skip_mask;
                                        if (n_static < 64)
                                            live &= (1ull << n_static) - 1ull;
                                        while (live) {
                                            const int m = __builtin_ctzll(live);
                                            live &= live - 1ull;
                                            float r2;
                                            if (kLiveR2) {
                                                const float dx = xi - cx[m];
                                                const float dy = yi - cy[m];
                                                const float dz = zi - cz[m];
                                                r2 = dx * dx + dy * dy + dz * dz;
                                                r2_buf[m] = r2;
                                            } else {
                                                r2 = r2_buf[m];
                                            }
                                            if (hot_cnt)
                                                note_mu_pair_r2(
                                                    nstats, r2,
                                                    contact_cutoff_sq_);
                                            if (r2 > contact_cutoff_sq_) continue;
                                            in_cut[n_in++] = m;
                                        }
                                    }
                                    for (int t = 0; t < n_in; ++t) {
                                        if (kPrefetchLayered && t + 2 < n_in &&
                                            use_topo_flags_ &&
                                            !topo_flag_.empty()) {
                                            const int j_next = cids[in_cut[t + 2]];
                                            const size_t pidx =
                                                static_cast<size_t>(i) *
                                                    static_cast<size_t>(
                                                        num_atoms_cached_) +
                                                static_cast<size_t>(j_next);
                                            __builtin_prefetch(
                                                &topo_flag_[pidx], 0, 1);
                                            const int ti =
                                                atom_types[static_cast<size_t>(
                                                    i)];
                                            const int tj =
                                                atom_types[static_cast<size_t>(
                                                    j_next)];
                                            if (ti >= 0 && tj >= 0 &&
                                                n_types_ > 0) {
                                                __builtin_prefetch(
                                                    &type_params_
                                                        [static_cast<size_t>(
                                                             ti) *
                                                             static_cast<
                                                                 size_t>(
                                                                 n_types_) +
                                                         static_cast<size_t>(
                                                             tj)],
                                                    0, 1);
                                            }
                                        }
                                        const int m = in_cut[t];
                                        const float r2 = r2_buf[m];
                                        if (is_new_side) {
                                            bool local_clash = false;
                                            acc += static_cast<double>(eval_pair(
                                                i, cids[m], r2, &local_clash));
                                            if (local_clash) {
                                                clash = true;
                                                break;
                                            }
                                        } else {
                                            acc -= static_cast<double>(eval_pair(
                                                i, cids[m], r2, nullptr));
                                        }
                                    }
                                }
                        }
                        if (clash) return false;
                    }
                    return true;
                };

                // CHANGED: cell-pair — NEW side first (clash early-exit), then OLD.
                // Pair sets match per-atom; double acc reduces FP order sensitivity.
                double cell_pair_acc = 0.0;
                // NOTE: cp_new_walk_ns / cp_old_walk_ns hold the TOTAL time in
                // eval_groups for that side. The walk proper (stencil
                // enumeration, span fetch, moved_bits/skip_mask bookkeeping,
                // and the rigid-MM clash guard) is therefore
                //     walk = total - r2 - eval
                // computed by the reporter. Bracketing the walk directly would
                // need timers inside the per-cell loop, tripling the rdtsc
                // count for no extra information.
                const bool new_ok = eval_groups(
                    ws.new_moved_groups, cnew, /*is_new=*/true, cell_pair_acc);
                if (!new_ok || clash) {
                    clash = true;
                } else if (!eval_groups(ws.old_moved_groups, cold,
                                        /*is_new=*/false, cell_pair_acc) ||
                           clash) {
                    clash = true;
                } else {
                    delta_E += static_cast<float>(cell_pair_acc);
                }

                // Non-rigid: moved–moved at new positions via moved_grid.
                if (!clash && moved_grid) {
                    for (int i : moved_indices) {
                        const float nx = cnew.x(i), ny = cnew.y(i),
                                    nz = cnew.z(i);
                        const bool ok_moved = moved_grid->for_each_neighbor_while(
                            nx, ny, nz, [&](int j) {
                                ++nstats.mu_num_candidates_iterated;
                                if (j == i) return true;
                                if (i > j) return true;
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
                        if (!ok_moved || clash) {
                            clash = true;
                            break;
                        }
                    }
                }
            }
        }

        if (!used_cell_pair && !clash) {
            // Per-moved-atom fused denselist (legacy / fallback).
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
                                r2_buf[m] = dx * dx + dy * dy + dz * dz;
                            }
                            for (int m = 0; m < count; ++m) {
                                if (skip_mask & (1ull << m)) continue;
                                const float r2 = r2_buf[m];
                                note_mu_pair_r2(nstats, r2, contact_cutoff_sq_);
                                if (r2 <= contact_cutoff_sq_)
                                    delta_E -=
                                        eval_pair(i, cids[m], r2, nullptr);
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
                                std::uint64_t moved_bits = 0ull;
                                for (int m = 0; m < count; ++m) {
                                    const int j = cids[m];
                                    if (j == i ||
                                        is_moved[static_cast<size_t>(j)])
                                        skip_mask |= (1ull << m);
                                    if (is_moved[static_cast<size_t>(j)])
                                        moved_bits |= (1ull << m);
                                }
                                float r2_buf[OpenCellGrid::CELL_CAPACITY];
#pragma GCC ivdep
                                for (int m = 0; m < count; ++m) {
                                    const float dx = nx - cx[m];
                                    const float dy = ny - cy[m];
                                    const float dz = nz - cz[m];
                                    r2_buf[m] = dx * dx + dy * dy + dz * dz;
                                }
                                // FIXED: 3-decimal rounding — PDB precision MM clash guard
                                // Use cnew.dist2 — cell pack coords are pre-move.
                                if (skip_rigid_mm && moved_bits != 0ull) {
                                    for (int m = 0; m < count; ++m) {
                                        if (!(moved_bits & (1ull << m))) continue;
                                        const int j = cids[m];
                                        if (j == i || i > j) continue;
                                        if (rigid_mm_pair_clashes(
                                                i, j, cnew.dist2(i, j))) {
                                            clash = true;
                                            return false;
                                        }
                                    }
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
                            delta_E -= eval_pair(i, j, r2, nullptr);
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
                        // FIXED: 3-decimal rounding — PDB precision MM clash guard
                        if (skip_rigid_mm && !clash) {
                            for (int j : moved_indices) {
                                if (j <= i) continue;
                                if (rigid_mm_pair_clashes(
                                        i, j, cnew.dist2(i, j))) {
                                    clash = true;
                                    break;
                                }
                            }
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
        }

        if (clash) {
            ws.clear();
            ws.clear_moved_grid();
            return kHardCorePenalty;
        }

        ws.clear_moved_grid();
        return delta_E;
    }
#else // !MCPU_FAST_MU_DELTA
    float MuPotential::calculateEnergyChange_fast(
        const Context&, const State&, const State&, const ProposalPatch&
    ) const {
        return 0.0f;
    }
#endif // MCPU_FAST_MU_DELTA

    // ---------------------------------------------------------
    // Legacy delta (MCPU_FAST_MU_DELTA=0) — behaviour unchanged; profile optional
    // ---------------------------------------------------------

    // ================================================================
    // LIVE CONTACT LIST  (MCPU_CONTACT_LIST=1)
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
        state.mu_contact_list.assign(static_cast<size_t>(N), {});
        const CoordView cv(state.coord_view());
        for (int i = 0; i < N; ++i) {
            if (sys.is_amide_h_atom(i)) continue;
            for (int j = i + 1; j < N; ++j) {
                if (sys.is_amide_h_atom(j)) continue;
                const size_t idx = static_cast<size_t>(i) *
                                       static_cast<size_t>(N) +
                                   static_cast<size_t>(j);
                if (!topo_contact_mask_[idx]) continue;
                const float r2 = cv.dist2(i, j);
                if (r2 > contact_cutoff_sq_) continue;
                bool lc = false;
                const float e = eval_pair(i, j, r2, &lc);
                if (e != 0.0f) state.mu_contact_add(i, j, e);
            }
        }
        state.mu_contact_list_ready = true;
    }

    
    float MuPotential::calculateEnergyChange_clist(
        const Context& context,
        const State& old_state,
        const State& new_state,
        const ProposalPatch& patch
    ) const {
        setup_mask_cache(context.getSystem());
        auto& ws = const_cast<mcpu::MuWorkspace&>(context.getMuWorkspace());
        ws.clear();

        const System& sys = context.getSystem();
        auto& ns = const_cast<NeighborSystem&>(context.neighbors());
        const OpenCellGrid& grid = ns.muGrid().grid();
        const std::vector<uint8_t>& is_moved = patch.moving_atoms;
        const std::vector<int>& moved = patch.moved_indices;
        const CoordView cnew(new_state.coord_view());
        const CoordView cold(old_state.coord_view());
        const bool skip_mm =
            patch.is_rigid && context.neighborConfig().skip_rigid_mm;

        double dE = 0.0;
        bool clash = false;

        // ---- OPTIONAL PASS 0: answer "does this move overlap?" on its own ----
        // About a third of the actin step is contact energy computed for pivot
        // moves that are then discarded for a hard-core overlap (measured:
        // 34.5% of the step, results/clash_split_summary.md). The overlap
        // question needs only r < ~2.8 A, while the contact walk sweeps 5.08 A
        // cells -- a box ~5.8x larger in volume. Asking it first, with a stencil
        // sized to the question, rejects those moves without doing any contact
        // work at all. Costs one extra tight pass on moves that do NOT overlap.
        // MCPU_CLASH_FIRST=1.
        // DEFAULT 2 (direct reachable-cell enumeration). Measured a further
        // 1.16x on actin / 1.11x on 1igd on top of the contact list, and
        // neutral on chignolin thanks to the moved-atom gate below.
        // 0 = off, 1 = the 27-offset + point-to-box cull variant (MEASURED
        // SLOWER, kept only so the comparison can be reproduced).
        static const int kClashFirst = [] {
            const char* e = std::getenv("MCPU_CLASH_FIRST");
            if (!e || !e[0]) return 2;
            if (e[0] == '0') return 0;
            if (e[0] == '1') return 1;
            return 2;
        }();
        // Only worth it for moves big enough that skipping the contact walk pays
        // for the extra pass. A small move examines few pairs anyway, so the
        // pre-pass is nearly pure overhead -- measured -10% on chignolin, whose
        // largest move touches ~25 atoms, against +14% on actin, whose pivots
        // touch ~740. Threshold is on MOVED ATOMS, not system size, so one
        // number covers every move kind and every protein.
        // Threshold now lives on NeighborConfig (settable from Python via
        // Context::set_clash_first_min_moved); the env var stays as an override
        // so an A/B sweep can be driven without touching the caller.
        static const int kClashFirstEnv = [] {
            const char* e = std::getenv("MCPU_CLASH_FIRST_MIN_MOVED");
            if (!e || !e[0]) return -1;
            return std::atoi(e);
        }();
        const int kClashFirstMinMoved =
            (kClashFirstEnv >= 0) ? kClashFirstEnv
                                  : context.neighborConfig().clash_first_min_moved;
        if (kClashFirst &&
            static_cast<int>(moved.size()) >= kClashFirstMinMoved) {
            const float rq = clash_query_radius();
            if (rq > 0.f) {
                for (int i : moved) {
                    const float nx = cnew.x(i), ny = cnew.y(i), nz = cnew.z(i);
                    auto clash_cell = [&](const int* __restrict__ cids,
                            const float* __restrict__ cx,
                            const float* __restrict__ cy,
                            const float* __restrict__ cz, int count) {
                            for (int m = 0; m < count; ++m) {
                                const int j = cids[m];
                                if (j == i) continue;
                                if (is_moved[static_cast<size_t>(j)]) continue;
                                const float dx = nx - cx[m];
                                const float dy = ny - cy[m];
                                const float dz = nz - cz[m];
                                const float r2 = dx * dx + dy * dy + dz * dz;
                                if (rigid_mm_pair_clashes(i, j, r2)) return false;
                            }
                            return true;
                        };
                    const bool ok =
                        (kClashFirst == 2)
                            ? grid.for_each_cell_span_within_fast(
                                  nx, ny, nz, rq, clash_cell)
                            : grid.for_each_neighbor_cell_span_while_within(
                                  nx, ny, nz, rq, clash_cell);
                    if (!ok) { clash = true; break; }
                }
                if (clash) {
                    ws.clear();
                    return kHardCorePenalty;
                }
            }
        }

        // ---- OLD half: read it off the list. No distances, no cell walk. ----
        // Every contact a moved atom currently has is a contact that this move
        // is about to re-decide, so all of them come off the books here; the
        // NEW half puts back the ones that survive.
        for (int i : moved) {
            for (const auto& c : old_state.mu_contact_list[static_cast<size_t>(i)]) {
                const int j = c.j;
                if (is_moved[static_cast<size_t>(j)]) {
                    // A rigid move cannot change a moved-moved distance, so
                    // those contacts survive untouched and must NOT be
                    // re-decided. For a flexible move they are re-decided
                    // once, by the lower-indexed partner.
                    if (skip_mm) continue;
                    if (i > j) continue;
                }
                dE -= static_cast<double>(c.energy);
                ws.pending_contact_drop.push_back(
                    mcpu::MuWorkspace::PendingContact{i, j, c.energy});
            }
        }

        // ---- NEW half: one cell walk, one distance per candidate. ----
        for (int i : moved) {
            const float nx = cnew.x(i), ny = cnew.y(i), nz = cnew.z(i);
            const bool ok = grid.for_each_neighbor_cell_span_while(
                nx, ny, nz,
                [&](const int* __restrict__ cids,
                    const float* __restrict__ cx,
                    const float* __restrict__ cy,
                    const float* __restrict__ cz, int count) {
                    for (int m = 0; m < count; ++m) {
                        const int j = cids[m];
                        if (j == i) continue;
                        // The grid holds ACCEPTED coordinates, so a moved
                        // partner's packed position is stale. Moved-moved
                        // pairs are done below from the trial coordinates.
                        if (is_moved[static_cast<size_t>(j)]) continue;
                        const float dx = nx - cx[m];
                        const float dy = ny - cy[m];
                        const float dz = nz - cz[m];
                        const float r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 > contact_cutoff_sq_) continue;
                        bool local_clash = false;
                        const float e = eval_pair(i, j, r2, &local_clash);
                        if (local_clash) { clash = true; return false; }
                        if (e != 0.0f) {
                            dE += static_cast<double>(e);
                            ws.pending_contact_add.push_back(
                                mcpu::MuWorkspace::PendingContact{i, j, e});
                        }
                    }
                    return true;
                });
            if (!ok || clash) { clash = true; break; }
        }

        // ---- moved-moved pairs ----
        if (!clash && moved.size() > 1) {
            if (skip_mm) {
                // Energy is unchanged by construction (a rigid transform cannot
                // change a moved-moved distance); this only guards against a
                // float32 rotation nudging a borderline pair under the hard core.
                //
                // FIXED: this used to be an O(n_moved^2) double loop -- 725
                // moved atoms on an actin pivot is 263,000 distance
                // computations, to catch a ~1e-6 A rounding effect. Two atoms
                // can only be closer than the hard core if they are within the
                // clash radius of each other, so the grid can enumerate the
                // candidates instead. Same predicate, same pairs that can
                // possibly fire; O(n_moved x few cells).
                //
                // MCPU_MM_GUARD_N2=1 restores the quadratic loop so the two can
                // be interleaved on one node and checked for identical accepts.
                // DEFAULT ON (the quadratic loop). Counter-intuitive but
                // measured: 725 moved atoms is 263k pairs, yet they are
                // contiguous in memory so the loop vectorises and beats a grid
                // query (actin 80.6 vs 82.5 us/step, chignolin 15.9 vs 18.0).
                // CAVEAT: it is O(n_moved^2), so for chains much longer than
                // actin (~377 res) the grid form should eventually win.
                // MCPU_MM_GUARD_N2=0 selects the grid form.
                static const bool kN2Guard = [] {
                    const char* e = std::getenv("MCPU_MM_GUARD_N2");
                    return !(e && e[0] == '0');
                }();
                const float rq = clash_query_radius();
                if (kN2Guard || rq <= 0.f) {
                    for (size_t a = 0; a < moved.size() && !clash; ++a) {
                        const int i = moved[a];
                        for (size_t b = a + 1; b < moved.size(); ++b) {
                            const int j = moved[b];
                            if (rigid_mm_pair_clashes(i, j, cnew.dist2(i, j))) {
                                clash = true;
                                break;
                            }
                        }
                    }
                } else {
                    for (int i : moved) {
                        // Query at the OLD position: the grid stores accepted
                        // coordinates, and a rigid move leaves every
                        // moved-moved distance unchanged to ~1e-6 A, so the
                        // candidate set is the same either way -- but only the
                        // old position agrees with where the grid put things.
                        const float qx = cold.x(i), qy = cold.y(i), qz = cold.z(i);
                        const bool ok =
                            grid.for_each_neighbor_cell_span_while_within(
                                qx, qy, qz, rq,
                                [&](const int* __restrict__ cids,
                                    const float* __restrict__,
                                    const float* __restrict__,
                                    const float* __restrict__, int count) {
                                    for (int m = 0; m < count; ++m) {
                                        const int j = cids[m];
                                        if (j <= i) continue;
                                        if (!is_moved[static_cast<size_t>(j)])
                                            continue;
                                        if (rigid_mm_pair_clashes(
                                                i, j, cnew.dist2(i, j)))
                                            return false;
                                    }
                                    return true;
                                });
                        if (!ok) { clash = true; break; }
                    }
                }
            } else {
                for (size_t a = 0; a < moved.size() && !clash; ++a) {
                    const int i = moved[a];
                    for (size_t b = a + 1; b < moved.size(); ++b) {
                        const int j = moved[b];
                        const float r2 = cnew.dist2(i, j);
                        if (r2 > contact_cutoff_sq_) continue;
                        bool local_clash = false;
                        const float e = eval_pair(i, j, r2, &local_clash);
                        if (local_clash) { clash = true; break; }
                        if (e != 0.0f) {
                            dE += static_cast<double>(e);
                            ws.pending_contact_add.push_back(
                                mcpu::MuWorkspace::PendingContact{i, j, e});
                        }
                    }
                }
            }
        }

        if (clash) {
            ws.clear();
            return kHardCorePenalty;
        }
        return static_cast<float>(dE);
    }

    float MuPotential::calculateEnergyChange_legacy(
        const Context& context,
        const State& old_state,
        const State& new_state,
        const ProposalPatch& patch
    ) const {
#if MCPU_FAST_MU_DELTA
        (void)context; (void)old_state; (void)new_state; (void)patch;
        return 0.0f;
#else
        setup_mask_cache(context.getSystem());
        auto& ws = const_cast<mcpu::MuWorkspace&>(context.getMuWorkspace());
        ws.clear();


        float delta_E = 0.0f;
        bool clash = false;
        int num_atoms = context.getSystem().getNumAtoms();
        const auto& ns = context.neighbors();
        const std::vector<uint8_t>& is_moved = patch.moving_atoms;

        std::vector<int> moved_indices;
        {
            if (!patch.moved_indices.empty()) {
                moved_indices = patch.moved_indices;
            } else {
                moved_indices.reserve(static_cast<size_t>(num_atoms));
                for (int i = 0; i < num_atoms; ++i) {
                    if (is_moved[static_cast<size_t>(i)]) moved_indices.push_back(i);
                }
            }
        }


        const CoordView cold_legacy(old_state.coord_view());
        const CoordView cnew_legacy(new_state.coord_view());

        auto evaluate_against_static = [&](int i) {
            const float ox = cold_legacy.x(i), oy = cold_legacy.y(i), oz = cold_legacy.z(i);
            const float nx = cnew_legacy.x(i), ny = cnew_legacy.y(i), nz = cnew_legacy.z(i);

            {
                ns.for_each_mu_candidate(ox, oy, oz, [&](int j) {
                    if (is_moved[static_cast<size_t>(j)]) return;

                    int matrix_idx = i * num_atoms + j;
                    const auto& cinfo = contact_cache[static_cast<size_t>(matrix_idx)];
                    if (old_state.is_contact_cache[static_cast<size_t>(matrix_idx)]) {
                        delta_E -= cinfo.energy;
                        ws.pending_updates.push_back({matrix_idx, false});
                        ws.pending_updates.push_back({j * num_atoms + i, false});
                    }
                });
            }

            {
                ns.for_each_mu_candidate_while(nx, ny, nz, [&](int j) {
                    if (is_moved[static_cast<size_t>(j)]) return true;

                    int matrix_idx = i * num_atoms + j;
                    const auto& cinfo = contact_cache[static_cast<size_t>(matrix_idx)];
                    if (!cinfo.check_contact && !cinfo.check_clash) {
                        return true;
                    }

                    float dist_sq = cnew_legacy.dist2(i, j);
                    if (cinfo.check_clash &&
                        is_hard_clash(dist_sq, cinfo.hard_core_sq)) {
                        clash = true;
                        return false;
                    }
                    if (cinfo.check_contact && dist_sq <= cinfo.contact_dist_sq) {
                        delta_E += cinfo.energy;
                        ws.pending_updates.push_back({matrix_idx, true});
                        ws.pending_updates.push_back({j * num_atoms + i, true});
                    }
                    return true;
                });
            }
        };

        for (int i : moved_indices) {
            evaluate_against_static(i);
            if (clash) break;
        }

        if (!patch.is_rigid) {
            for (size_t idx_i = 0; idx_i < moved_indices.size(); ++idx_i) {
                int i = moved_indices[idx_i];

                for (size_t idx_j = idx_i + 1; idx_j < moved_indices.size(); ++idx_j) {
                    int j = moved_indices[idx_j];
                    int matrix_idx = i * num_atoms + j;
                    const auto& cinfo = contact_cache[static_cast<size_t>(matrix_idx)];
                    if (!cinfo.check_contact && !cinfo.check_clash) {
                        continue;
                    }

                    if (old_state.is_contact_cache[static_cast<size_t>(matrix_idx)]) {
                        delta_E -= cinfo.energy;
                        ws.pending_updates.push_back({matrix_idx, false});
                        ws.pending_updates.push_back({j * num_atoms + i, false});
                    }

                    const float dist_sq = cnew_legacy.dist2(i, j);

                    if (cinfo.check_clash &&
                        is_hard_clash(dist_sq, cinfo.hard_core_sq)) {
                        clash = true;
                        break;
                    } else if (cinfo.check_contact && dist_sq <= cinfo.contact_dist_sq) {
                        delta_E += cinfo.energy;
                        ws.pending_updates.push_back({matrix_idx, true});
                        ws.pending_updates.push_back({j * num_atoms + i, true});
                    }
                }
                if (clash) break;
            }
        }

        if (clash) {
            ws.clear();
            return kHardCorePenalty;
        }


        return delta_E;
#endif
    }


    float MuPotential::calculateEnergy(const Context& context, const State& state) const {
        // Full energy for an arbitrary State must use that state's coordinates.
        // NeighborSystem Mu index reflects accepted coords only — do not query it here
        // for proposed/trial states (PhysicsVerifier). Production total energy is
        // called on the accepted state; O(n_mu^2) is acceptable for that path.
        const System& sys = context.getSystem();
        setup_mask_cache(sys);
        float total_energy = 0.0f;
        const int num_atoms = sys.getNumAtoms();

        const CoordView cv(state.coord_view());
        for (int i = 0; i < num_atoms; ++i) {
            if (sys.is_amide_h_atom(i)) continue;
            for (int j = i + 1; j < num_atoms; ++j) {
                if (sys.is_amide_h_atom(j)) continue;
                const int matrix_idx = i * num_atoms + j;

#if MCPU_FAST_MU_DELTA
                if (!topo_contact_mask_[static_cast<size_t>(matrix_idx)] &&
                    !topo_clash_mask_[static_cast<size_t>(matrix_idx)]) {
                    continue;
                }
                const float dist_sq = cv.dist2(i, j);
                bool local_clash = false;
                const float e = eval_pair(i, j, dist_sq, &local_clash);
                if (local_clash) {
                    // ClashOnly: full energy stays contact-only (legacy
                    // CLASH_WEIGHT=0). Delta path still StericClash-rejects.
                    if (energy_mask_mode_cached_ == EnergyMaskMode::ClashOnly) {
                        continue;
                    }
                    // DIAGNOSTIC (MCPU_CLASH_REPORT=1): identify the pair that
                    // trips the sentinel. Integrator.cpp rejects any proposal
                    // whose delta path reports StericClash, so reaching here on
                    // an accepted state means the delta path MISSED this pair.
                    // Reports the geometry plus the Mu cutoff, so a pair that
                    // sits outside the cell-grid's enumeration radius (the
                    // out-of-grid case the ws.use_trial_fallback comment in
                    // calculateEnergyChange guards against) is identifiable.
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
                    return kHardCorePenalty;
                }
                if (e != 0.0f) {
                    total_energy += e;
                    if (state.has_contact_cache()) {
                        state.is_contact_cache[static_cast<size_t>(matrix_idx)] = true;
                        state.is_contact_cache[static_cast<size_t>(j * num_atoms + i)] = true;
                    }
                } else if (topo_contact_mask_[static_cast<size_t>(matrix_idx)]) {
                    if (state.has_contact_cache()) {
                        state.is_contact_cache[static_cast<size_t>(matrix_idx)] = false;
                        state.is_contact_cache[static_cast<size_t>(j * num_atoms + i)] = false;
                    }
                }
#else
                const auto& contact_info = contact_cache[static_cast<size_t>(matrix_idx)];
                if (!contact_info.check_contact && !contact_info.check_clash) {
                    continue;
                }
                const float dist_sq = cv.dist2(i, j);
                if (contact_info.check_clash &&
                    is_hard_clash(dist_sq, contact_info.hard_core_sq)) {
                    return kHardCorePenalty;
                }
                if (contact_info.check_contact && dist_sq <= contact_info.contact_dist_sq) {
                    total_energy += contact_info.energy;
                    if (state.has_contact_cache()) {
                        state.is_contact_cache[static_cast<size_t>(matrix_idx)] = true;
                        state.is_contact_cache[static_cast<size_t>(j * num_atoms + i)] = true;
                    }
                } else if (contact_info.check_contact) {
                    if (state.has_contact_cache()) {
                        state.is_contact_cache[static_cast<size_t>(matrix_idx)] = false;
                        state.is_contact_cache[static_cast<size_t>(j * num_atoms + i)] = false;
                    }
                }
#endif
            }
        }

        return total_energy;
    }

    float MuPotential::calculateEnergy(
        const Context& context, const State& state, bool /*update_cache*/
    ) const {
        // Delegate to the 2-arg override explicitly (avoid overload ambiguity)
        return static_cast<const Potential*>(this)->calculateEnergy(context, state);
    }

} // namespace mcpu::forces::mcpu08
