#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace mcpu {

enum class MoveKind : int {
    Pivot = 0,
    KIC = 1,
    Sidechain = 2,
    Other = 3
};

struct NeighborConfig {
    float max_atom_displacement_hard = 12.f;

    bool rebuild_from_aabb = true;
    float margin_angstrom = -1.f; // <0 => 2*r_cut
    int margin_cells = -1;

    std::uint64_t max_cells_total = 2'000'000ull;
    int max_nx = 0;
    int max_ny = 0;
    int max_nz = 0;


    /// If true (default), the pairs a rigid pivot (``patch.is_rigid``) carries
    /// -- both atoms moved -- are not re-measured: a rigid rotation keeps their
    /// distances, so they cannot start to overlap and only a pair on its
    /// contact cutoff can change energy, by rounding. Mu re-decides just the
    /// carried pairs on its contact list (see MuPotential), and the KORP CA-CA
    /// guard skips them. The H-bond term likewise keeps the energy of a
    /// donor-acceptor pair whose backbone geometry (residues r-1 to r+1 on both
    /// sides) moved as one body and carries its ledger entry (see
    /// HBondPotential). False evaluates them all exactly, as a reference.
    bool skip_rigid_mm = true;

    /// If true (default), denselist Mu uses cell-pair inversion (group moved by
    /// cell). Set false / ``MCPU_USE_CELL_PAIR=0`` for per-atom walks (parity).
    bool use_cell_pair = true;

    /// Minimum moved-atom count to use cell-pair inversion. Below this, per-atom
    /// denselist is cheaper (SC ~3.5 atoms). Default 20 ≈ KIC size.
    int cell_pair_min_moved = 20;

    /// Minimum moved-atom count for the clash-first pass (MuPotential). Below
    /// this the pass is nearly pure overhead: a small move examines few pairs
    /// anyway, so skipping the contact walk saves little while the extra pass
    /// still costs. Measured ungated: -9.6% on chignolin (largest move ~25
    /// atoms) against +14% on actin (pivots ~740 atoms). Swept over
    /// chignolin/1igd/actin; 50 is neutral on chignolin and within noise of
    /// 100/300 on actin. The true crossover lies between 25 and 50 and is
    /// unmeasured.
    int clash_first_min_moved = 50;


    /// Mu denselist cell size relative to the Mu cutoff. Default 1.0.
    float mu_cell_size_scale = 1.f;
    /// Absolute Mu cell size (Å). If &gt; 0, overrides scale.
    float mu_cell_size_angstrom = -1.f;
    /// Extra lower bound (Å) on the cell; the cell is never below the Mu
    /// cutoff anyway, so only a value above it does anything. 0 = none.
    float mu_cell_size_min_angstrom = 0.f;
    /// Denselist query/cell cutoff (Å). Filled by Context::sync_geometry from
    /// MuPotential::mu_exact_cutoff(). &lt;0 → NeighborSystem falls back to 6.0.
    float mu_denselist_cutoff_A = -1.f;


    float effective_margin(float r_cut) const {
        if (margin_angstrom >= 0.f) return margin_angstrom;
        if (margin_cells >= 0) return static_cast<float>(margin_cells) * r_cut;
        return 2.f * r_cut;
    }
};

/// Apply ``MCPU_MU_CELL_SCALE`` (Mu denselist cell size relative to the
/// cutoff). O(1).
inline void apply_neighbor_env_overrides(NeighborConfig& cfg) noexcept {
    if (const char* e = std::getenv("MCPU_MU_CELL_SCALE")) {
        char* end = nullptr;
        const float v = std::strtof(e, &end);
        if (end != e && v > 0.f) {
            cfg.mu_cell_size_scale = v;
            cfg.mu_cell_size_angstrom = -1.f;
        }
    }
}

/// Effective Mu denselist cell size (Å): scale*r_cut or absolute, never
/// below ``r_cut``. Cell may exceed ``r_cut`` (scale>1): stencil radius stays
/// ``ceil(query/cell)`` = 1 and still finds all pairs within cutoff. Upper
/// clamp to ``list`` was removed so ``mu_cell_size_scale>1`` and absolute sizes
/// > cutoff are usable for occupancy / AVX tuning.
inline float effective_mu_cell_size_A(float r_cut,
                                     const NeighborConfig& cfg) noexcept {
    float cell = (cfg.mu_cell_size_angstrom > 0.f)
                     ? cfg.mu_cell_size_angstrom
                     : r_cut * (cfg.mu_cell_size_scale > 0.f ? cfg.mu_cell_size_scale
                                                            : 1.f);
    // FIXED: never below the cutoff. The Mu grid code walks a one-cell stencil
    // (27 cells, NeighborCellList::kCap); a smaller cell needs a wider stencil,
    // which overflowed those buffers: set_positions crashed, and the cell-pair
    // path silently dropped cells.
    float mn = cfg.mu_cell_size_min_angstrom > 0.f ? cfg.mu_cell_size_min_angstrom : 0.f;
    if (mn < r_cut) mn = r_cut;
    if (cell < mn) cell = mn;
    // A cell above the cutoff (scale>1 / absolute) is allowed. Correctness:
    // OpenCellGrid uses R=ceil(query_radius/cell_size); query stays at r_cut.
    return cell;
}

/// Per-step neighbour work proxies derived from raw NeighborStats counters.
///
/// - avg_mu_pair_checks_per_step: r2 computations / step (primary work proxy).
/// - avg_mu_pairs_within_rcut_per_step: pairs with r2 <= r_mu^2 / step.
/// - avg_mu_candidates_per_step: cell-list visits before filters.
struct NeighborProxyReport {
    std::uint64_t total_steps = 0;
    double avg_mu_pair_checks_per_step = 0.0;
    double avg_mu_pairs_within_rcut_per_step = 0.0;
    /// Deprecated alias of avg_mu_pair_checks_per_step (historical name).
    double avg_mu_pairs_per_step = 0.0;
    double avg_mu_candidates_per_step = 0.0;
    double avg_cell_visits_per_step = 0.0;
    double avg_hbond_geom_checks_per_step = 0.0;
};

struct NeighborStats {
    // --- lifecycle / policy ---
    std::uint64_t num_trial_fallback = 0;
    std::uint64_t num_dense_cap_fallback = 0;
    std::uint64_t num_reject_hard_disp = 0;
    std::uint64_t num_aabb_rebuild_accept = 0;

    // --- performance proxy counters ---
    /// Neighbor-list / cell-list visits (before moved/dedup filters). Distinct from
    /// pair distance checks.
    std::uint64_t mu_num_candidates_iterated = 0;
    /// Every time Mu delta computes r2 for a candidate pair (i,j).
    std::uint64_t mu_num_pair_distance_checks = 0;
    /// Subset of distance checks with r2 <= r_mu^2 (6 Å cell cutoff).
    std::uint64_t mu_num_pairs_within_rcut = 0;
    /// Deprecated synonym for mu_num_pair_distance_checks (kept for older scripts).
    std::uint64_t& mu_num_pairs_evaluated() noexcept { return mu_num_pair_distance_checks; }
    std::uint64_t mu_num_pairs_evaluated() const noexcept { return mu_num_pair_distance_checks; }
    std::uint64_t hbond_num_candidates_iterated = 0;
    std::uint64_t hbond_num_geom_checks = 0;
    std::uint64_t neighbor_num_cell_visits = 0;
    /// Span slots the cell-pair r2 loop actually scans, i.e. sum of n_static
    /// over every (moved atom, neighbour cell). Compare against
    /// mu_num_pair_distance_checks -- which counts only the slots that survive
    /// skip_mask -- to see how much r2 work is computed and then discarded.
    std::uint64_t mu_span_slots_scanned = 0;
    /// (moved atom, stencil cell) pairs skipped by the distance cull.
    std::uint64_t mu_stencil_cells_culled = 0;

    /// Mu denselist stencil (variable radius).
    std::uint64_t neighbor_offsets_count = 0;

    /// Pair-eval counter flushed from MuPotential (useful denselist proxy).
    std::uint64_t mu_eval_pair_calls = 0;
    /// Subset of eval_pair calls that returned a non-zero energy contribution.
    std::uint64_t mu_eval_pair_nonzero = 0;
    /// Moved–moved denselist candidates skipped for rigid pivots.
    std::uint64_t elided_rigid_mm = 0;

    /// Cell-pair inversion diagnostics (production denselist when use_cell_pair).
    std::uint64_t pivot_mu_cell_pairs = 0;     ///< (mc,nc) nonempty iterations
    std::uint64_t pivot_mu_cell_pairs_empty = 0; ///< stencil nc with count==0
    std::uint64_t pivot_mu_n_groups = 0;       ///< sum of old+new group counts
    std::uint64_t pivot_mu_group_atoms = 0;    ///< sum of group.counts (for avg)
    std::uint64_t pivot_mu_cell_pair_evals = 0; ///< denselist steps using cell-pair

    /// Cell-pair path phase timers (MCPU_CELL_PAIR_BREAKDOWN=1, pivot/rigid only).
    std::uint64_t cp_group_build_ns = 0;
    std::uint64_t cp_new_walk_ns = 0;
    std::uint64_t cp_new_r2_ns = 0;
    std::uint64_t cp_new_eval_ns = 0;
    std::uint64_t cp_old_walk_ns = 0;
    std::uint64_t cp_old_r2_ns = 0;
    std::uint64_t cp_old_eval_ns = 0;
    std::uint64_t cp_movedbits_ns = 0;  ///< per-cell moved mask (random is_moved[] loads)
    std::uint64_t cp_skipmask_ns = 0;   ///< per-(moved atom, cell) skip mask
    std::uint64_t cp_clash_aborts = 0;
    std::uint64_t cp_full_evals = 0;
    std::uint64_t cp_new_r2_checks = 0;
    std::uint64_t cp_old_r2_checks = 0;
    std::uint64_t cp_new_eval_calls = 0;
    std::uint64_t cp_old_eval_calls = 0;
    std::uint64_t cp_n_steps = 0;  ///< pivot/rigid denselist steps timed
    std::uint64_t cp_new_n_groups = 0;
    std::uint64_t cp_new_group_atoms = 0;

    /// MC steps counted in the current Integrator::run (or manual increments).
    std::uint64_t num_steps_executed = 0;


    void reset() { *this = NeighborStats{}; }

    NeighborProxyReport derive(std::uint64_t steps) const {
        NeighborProxyReport r;
        const std::uint64_t den_steps = steps > 0 ? steps : 1ull;
        r.total_steps = steps;
        r.avg_mu_pair_checks_per_step =
            static_cast<double>(mu_num_pair_distance_checks) /
            static_cast<double>(den_steps);
        r.avg_mu_pairs_within_rcut_per_step =
            static_cast<double>(mu_num_pairs_within_rcut) /
            static_cast<double>(den_steps);
        r.avg_mu_pairs_per_step = r.avg_mu_pair_checks_per_step; // deprecated alias
        r.avg_mu_candidates_per_step = static_cast<double>(mu_num_candidates_iterated) /
                                       static_cast<double>(den_steps);
        r.avg_cell_visits_per_step = static_cast<double>(neighbor_num_cell_visits) /
                                     static_cast<double>(den_steps);
        r.avg_hbond_geom_checks_per_step = static_cast<double>(hbond_num_geom_checks) /
                                           static_cast<double>(den_steps);
        return r;
    }

    /// Print raw counters + derived per-step metrics to stderr.
    void print(const char* tag = "neighbor-proxy") const {
        const char* t = tag ? tag : "neighbor-proxy";
        const std::uint64_t steps = num_steps_executed;
        const NeighborProxyReport r = derive(steps);

        std::fprintf(stderr,
            "[%s] steps=%llu | mu_cand=%llu mu_r2=%llu mu_rcut=%llu "
            "hb_cand=%llu hb_geom=%llu cell_visits=%llu | "
            "aabb_rebuild=%llu trial_fb=%llu\n",
            t,
            static_cast<unsigned long long>(steps),
            static_cast<unsigned long long>(mu_num_candidates_iterated),
            static_cast<unsigned long long>(mu_num_pair_distance_checks),
            static_cast<unsigned long long>(mu_num_pairs_within_rcut),
            static_cast<unsigned long long>(hbond_num_candidates_iterated),
            static_cast<unsigned long long>(hbond_num_geom_checks),
            static_cast<unsigned long long>(neighbor_num_cell_visits),
            static_cast<unsigned long long>(num_aabb_rebuild_accept),
            static_cast<unsigned long long>(num_trial_fallback));

        std::fprintf(stderr,
            "[%s] derived: avg_mu_r2/step=%.1f avg_mu_rcut/step=%.1f "
            "avg_mu_cand/step=%.1f avg_cell_visits/step=%.1f "
            "avg_hb_geom/step=%.1f\n",
            t,
            r.avg_mu_pair_checks_per_step,
            r.avg_mu_pairs_within_rcut_per_step,
            r.avg_mu_candidates_per_step,
            r.avg_cell_visits_per_step,
            r.avg_hbond_geom_checks_per_step);
    }
};

} // namespace mcpu
