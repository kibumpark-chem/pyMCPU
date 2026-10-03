#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace mcpu {

enum class BoxPolicy : int {
    Fixed = 0,
    AutoExpand = 1,
    AutoRecenter = 2
};

enum class NeighborMode : int {
    CellOnly = 0,         // Pivot: never Verlet
    VerletPreferred = 1   // KIC/SC: Verlet when valid
};

enum class MoveKind : int {
    Pivot = 0,
    KIC = 1,
    Sidechain = 2,
    Other = 3
};

struct NeighborConfig {
    BoxPolicy box_policy = BoxPolicy::AutoExpand;
    /// Mu Verlet skin (Å). Default 0 = denselist CellOnly (no Verlet CSR).
    /// skin=1.0 + partial rebuild is parity-clean but wall-regresses on actin
    /// (CSR pack + pivot-triggered full rebuilds). Keep 0; opt-in via set_mu_skin
    /// or MCPU_MU_SKIN.
    float skin = 0.f;
    float max_atom_displacement_hard = 12.f;

    bool rebuild_from_aabb = true;
    float margin_angstrom = -1.f; // <0 => 2*(r_cut+skin)
    int margin_cells = -1;

    std::uint64_t max_cells_total = 2'000'000ull;
    int max_nx = 0;
    int max_ny = 0;
    int max_nz = 0;

    bool pivot_uses_verlet = false;
    /// Legacy: force-dirty Mu Verlet on every pivot/rigid accept.
    /// Default false: pivot tracks displacement like other moves (Option A).
    bool invalidate_verlet_on_pivot_accept = false;

    /// Runtime Verlet enable (skin value may stay >0 while gate disables use).
    bool mu_verlet_enabled = true;

    /// If true (default), the pairs a rigid pivot (``patch.is_rigid``) carries
    /// -- both atoms moved -- are not re-measured: a rigid rotation keeps their
    /// distances, so they cannot start to overlap and only a pair on its
    /// contact cutoff can change energy, by rounding. Mu re-decides just the
    /// carried pairs on its contact list (see MuPotential), and the KORP CA-CA
    /// guard skips them. False evaluates them all exactly, as a reference.
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
    /// unmeasured. Env override: MCPU_CLASH_FIRST_MIN_MOVED.
    int clash_first_min_moved = 50;

    /// If moved atom count exceeds this, force CellOnly even for KIC/SC. O(1) check.
    int verlet_moved_threshold = 50;
    /// Partial Verlet CSR rebuild on SC/KIC accept when n_moved ≤ this.
    int verlet_partial_threshold = 50;

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

    /// DEPRECATED: auto-print from Integrator::run() was removed (vNext).
    /// Kept for ABI/config layout only; set_proxy_print_every() is a no-op.
    int proxy_print_every = -1;

    // Verlet health warning thresholds (skin>0 only; one-time per run).
    float verlet_warn_min_use_rate = 0.20f;
    float verlet_warn_max_rebuild_rate = 0.05f;

    float effective_margin(float r_cut) const {
        if (margin_angstrom >= 0.f) return margin_angstrom;
        const float list = r_cut + skin;
        if (margin_cells >= 0) return static_cast<float>(margin_cells) * list;
        return 2.f * list;
    }
};

/// Apply ``MCPU_MU_SKIN`` / ``MCPU_VERLET_PARTIAL_THRESHOLD`` /
/// ``MCPU_MU_CELL_SCALE``. O(1).
inline void apply_neighbor_env_overrides(NeighborConfig& cfg) noexcept {
    if (const char* e = std::getenv("MCPU_MU_SKIN")) {
        char* end = nullptr;
        const float v = std::strtof(e, &end);
        if (end != e) {
            cfg.skin = v;
            cfg.mu_verlet_enabled = (v > 0.f);
        }
    }
    if (const char* e = std::getenv("MCPU_VERLET_PARTIAL_THRESHOLD")) {
        char* end = nullptr;
        const long v = std::strtol(e, &end, 10);
        if (end != e && v >= 0)
            cfg.verlet_partial_threshold = static_cast<int>(v);
    }
    // ADDED: Mu denselist cell size scale (1.0 = cell≈cutoff).
    if (const char* e = std::getenv("MCPU_MU_CELL_SCALE")) {
        char* end = nullptr;
        const float v = std::strtof(e, &end);
        if (end != e && v > 0.f) {
            cfg.mu_cell_size_scale = v;
            cfg.mu_cell_size_angstrom = -1.f;
        }
    }
}

/// Effective Mu denselist cell size (Å): scale*(r_cut+skin) or absolute, never
/// below ``r_cut``. Cell may exceed ``r_cut`` (scale>1): stencil radius stays
/// ``ceil(query/cell)`` = 1 and still finds all pairs within cutoff. Upper
/// clamp to ``list`` was removed so ``mu_cell_size_scale>1`` and absolute sizes
/// > cutoff are usable for occupancy / AVX tuning.
inline float effective_mu_cell_size_A(float r_cut, float skin,
                                     const NeighborConfig& cfg) noexcept {
    const float sk = skin > 0.f ? skin : 0.f;
    const float list = r_cut + sk;
    float cell = (cfg.mu_cell_size_angstrom > 0.f)
                     ? cfg.mu_cell_size_angstrom
                     : list * (cfg.mu_cell_size_scale > 0.f ? cfg.mu_cell_size_scale
                                                           : 1.f);
    // FIXED: never below the cutoff. The Mu grid code walks a one-cell stencil
    // (27 cells, NeighborCellList::kCap); a smaller cell needs a wider stencil,
    // which overflowed those buffers: set_positions crashed, and the cell-pair
    // path silently dropped cells.
    float mn = cfg.mu_cell_size_min_angstrom > 0.f ? cfg.mu_cell_size_min_angstrom : 0.f;
    if (mn < r_cut) mn = r_cut;
    if (cell < mn) cell = mn;
    // CHANGED: allow cell > list (scale>1 / absolute > cutoff). Correctness:
    // OpenCellGrid uses R=ceil(query_radius/cell_size); query stays at r_cut.
    return cell;
}

/// Derived Verlet / neighbor health metrics from raw NeighborStats counters.
///
/// Interpretation (skin>0):
/// - verlet_use_rate: fraction of KIC/SC Verlet-eligible trials that actually used
///   the Verlet list (vs cell fallback). High is good (~>0.5 for small SC moves).
/// - rebuild_rate_per_step: Verlet full rebuilds per MC step. High means skin too
///   small or moves too large (thrash). Aim << 0.05 for healthy skin.
/// - avg_mu_pair_checks_per_step: r2 computations / step (primary work proxy for skin).
/// - avg_mu_pairs_within_rcut_per_step: pairs with r2 <= r_mu^2 / step.
/// - avg_mu_candidates_per_step: neighbor-list / cell-list visits before filters.
struct NeighborProxyReport {
    std::uint64_t total_steps = 0;
    float mu_skin = 0.f;
    double verlet_use_rate = 0.0;
    double rebuild_rate_per_step = 0.0;
    double avg_mu_pair_checks_per_step = 0.0;
    double avg_mu_pairs_within_rcut_per_step = 0.0;
    /// Deprecated alias of avg_mu_pair_checks_per_step (historical name).
    double avg_mu_pairs_per_step = 0.0;
    double avg_mu_candidates_per_step = 0.0;
    double avg_cell_visits_per_step = 0.0;
    double avg_hbond_geom_checks_per_step = 0.0;
    bool low_use_rate = false;
    bool high_rebuild_rate = false;
};

struct NeighborStats {
    // --- lifecycle / policy ---
    std::uint64_t num_trial_fallback = 0;
    std::uint64_t num_dense_cap_fallback = 0;
    std::uint64_t num_verlet_rebuilds = 0;
    std::uint64_t num_verlet_partial_rebuilds = 0;
    std::uint64_t num_verlet_partial_affected_sum = 0;
    std::uint64_t num_verlet_invalidate_pivot_accept = 0;
    /// Pivot-accept Verlet policy (Option A displacement tracking).
    std::uint64_t num_pivot_accepts = 0;
    std::uint64_t num_pivot_accepts_keep_verlet_valid = 0;
    std::uint64_t num_pivot_accepts_dirty_verlet = 0;
    /// Rebuild attribution (incremented inside maybe_rebuild_mu_verlet).
    std::uint64_t num_verlet_rebuild_due_to_pivot_accept = 0;
    std::uint64_t num_verlet_rebuild_due_to_disp_acc_exceeded = 0;
    std::uint64_t num_verlet_rebuild_due_to_dirty_flag = 0;
    std::uint64_t num_verlet_rebuild_due_to_autoexpand_accept = 0;
    /// Last rebuilt undirected CSR size (unique edges = directed/2).
    std::uint64_t verlet_edges_total = 0;
    double verlet_avg_degree = 0.0;
    /// CSR buffer growth (should be ~0 after first rebuild).
    std::uint64_t num_verlet_neigh_reallocs = 0;
    std::uint64_t num_verlet_offsets_reallocs = 0;
    std::uint64_t num_delta_cell_pivot = 0;
    std::uint64_t num_delta_verlet_kic_sc = 0;       // alias: num_verlet_used
    std::uint64_t num_delta_cell_kic_sc_fallback = 0; // alias: num_verlet_fallback_cell
    std::uint64_t num_reject_out_of_box = 0;
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
    /// Moved–moved denselist/Verlet candidates skipped for rigid pivots.
    std::uint64_t elided_rigid_mm = 0;

    /// Pivot denselist Mu sub-timers (MCPU_PIVOT_MU_BREAKDOWN=1 diagnostic path).
    /// Phased collect → r² → eval; not used on the default fused loop.
    std::uint64_t pivot_mu_cell_walk_ns = 0;
    std::uint64_t pivot_mu_r2_filter_ns = 0;
    std::uint64_t pivot_mu_eval_pair_ns = 0;
    std::uint64_t pivot_mu_overhead_ns = 0;
    std::uint64_t pivot_mu_candidates = 0;   ///< collected (i,j) before r² cutoff
    std::uint64_t pivot_mu_in_cutoff = 0;    ///< pairs with r² ≤ 36
    std::uint64_t pivot_mu_n_steps = 0;
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

    /// Walk sub-probe (MCPU_PIVOT_MU_BREAKDOWN=1): stencil empty vs nonempty.
    std::uint64_t pivot_walk_empty_cells = 0;
    std::uint64_t pivot_walk_nonempty_cells = 0;
    std::uint64_t pivot_walk_atom_visits = 0;  ///< sum of cell_count in nonempty
    std::uint64_t pivot_walk_probe_ns = 0;     ///< time for occupancy probe only
    std::uint64_t pivot_walk_oob_cells = 0;

    /// MC steps counted in the current Integrator::run (or manual increments).
    std::uint64_t num_steps_executed = 0;

    /// One-time health warning guard for the current run (cleared by reset()).
    bool verlet_health_warning_emitted = false;

    std::uint64_t& num_verlet_used() noexcept { return num_delta_verlet_kic_sc; }
    std::uint64_t num_verlet_used() const noexcept { return num_delta_verlet_kic_sc; }
    std::uint64_t& num_verlet_fallback_cell() noexcept { return num_delta_cell_kic_sc_fallback; }
    std::uint64_t num_verlet_fallback_cell() const noexcept { return num_delta_cell_kic_sc_fallback; }

    void reset() { *this = NeighborStats{}; }

    NeighborProxyReport derive(std::uint64_t steps, float skin,
                               float warn_min_use = 0.20f,
                               float warn_max_rebuild = 0.05f) const {
        NeighborProxyReport r;
        const std::uint64_t den_steps = steps > 0 ? steps : 1ull;
        r.total_steps = steps;
        r.mu_skin = skin;
        const std::uint64_t used = num_verlet_used();
        const std::uint64_t fb = num_verlet_fallback_cell();
        const std::uint64_t verlet_trials = used + fb;
        r.verlet_use_rate = static_cast<double>(used) /
                            static_cast<double>(verlet_trials > 0 ? verlet_trials : 1ull);
        r.rebuild_rate_per_step = static_cast<double>(num_verlet_rebuilds) /
                                  static_cast<double>(den_steps);
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
        if (skin > 0.f) {
            // Only flag low use when there were Verlet-eligible trials to judge.
            r.low_use_rate = (verlet_trials > 0) && (r.verlet_use_rate < warn_min_use);
            r.high_rebuild_rate = r.rebuild_rate_per_step > warn_max_rebuild;
        }
        return r;
    }

    /// Print raw counters + derived metrics. Emits one-time health warnings when skin>0.
    void print(const char* tag, float skin, const NeighborConfig* cfg = nullptr) {
        const std::uint64_t steps = num_steps_executed;
        const float warn_use = cfg ? cfg->verlet_warn_min_use_rate : 0.20f;
        const float warn_rb = cfg ? cfg->verlet_warn_max_rebuild_rate : 0.05f;
        const NeighborProxyReport r = derive(steps, skin, warn_use, warn_rb);

        std::fprintf(stderr,
            "[%s] steps=%llu skin=%.3f | mu_cand=%llu mu_r2=%llu mu_rcut=%llu "
            "hb_cand=%llu hb_geom=%llu cell_visits=%llu | verlet_used=%llu "
            "verlet_fb_cell=%llu verlet_rebuilds=%llu verlet_inv_pivot=%llu "
            "aabb_rebuild=%llu trial_fb=%llu\n",
            tag ? tag : "neighbor-proxy",
            static_cast<unsigned long long>(steps),
            static_cast<double>(skin),
            static_cast<unsigned long long>(mu_num_candidates_iterated),
            static_cast<unsigned long long>(mu_num_pair_distance_checks),
            static_cast<unsigned long long>(mu_num_pairs_within_rcut),
            static_cast<unsigned long long>(hbond_num_candidates_iterated),
            static_cast<unsigned long long>(hbond_num_geom_checks),
            static_cast<unsigned long long>(neighbor_num_cell_visits),
            static_cast<unsigned long long>(num_verlet_used()),
            static_cast<unsigned long long>(num_verlet_fallback_cell()),
            static_cast<unsigned long long>(num_verlet_rebuilds),
            static_cast<unsigned long long>(num_verlet_invalidate_pivot_accept),
            static_cast<unsigned long long>(num_aabb_rebuild_accept),
            static_cast<unsigned long long>(num_trial_fallback));

        std::fprintf(stderr,
            "[%s] derived: verlet_use_rate=%.3f rebuild_rate/step=%.4f "
            "avg_mu_r2/step=%.1f avg_mu_rcut/step=%.1f avg_mu_cand/step=%.1f "
            "avg_cell_visits/step=%.1f avg_hb_geom/step=%.1f\n",
            tag ? tag : "neighbor-proxy",
            r.verlet_use_rate,
            r.rebuild_rate_per_step,
            r.avg_mu_pair_checks_per_step,
            r.avg_mu_pairs_within_rcut_per_step,
            r.avg_mu_candidates_per_step,
            r.avg_cell_visits_per_step,
            r.avg_hbond_geom_checks_per_step);

        maybe_emit_verlet_health_warning(skin, r, warn_use, warn_rb);
    }

    /// Backward-compatible print without skin (derived skin treated as 0 → no warnings).
    void print(const char* tag = "neighbor-proxy") {
        print(tag, 0.f, nullptr);
    }

    void maybe_emit_verlet_health_warning(float skin, const NeighborProxyReport& r,
                                          float warn_min_use, float warn_max_rebuild) {
        if (skin <= 0.f || verlet_health_warning_emitted) return;
        if (!r.low_use_rate && !r.high_rebuild_rate) return;
        verlet_health_warning_emitted = true;
        std::fprintf(stderr,
            "[neighbor-proxy-WARN] Mu Verlet may not be helping (skin=%.3f): "
            "verlet_use_rate=%.3f (warn<%.2f) rebuild_rate/step=%.4f (warn>%.2f). "
            "Suggestions: reduce move amplitude / step_size_rad; "
            "increase skin slightly if rebuild thrash; "
            "or set skin=0 if use rate stays low.\n",
            static_cast<double>(skin),
            r.verlet_use_rate, static_cast<double>(warn_min_use),
            r.rebuild_rate_per_step, static_cast<double>(warn_max_rebuild));
    }
};

inline NeighborMode resolve_neighbor_mode(MoveKind kind, bool is_rigid,
                                          const NeighborConfig& cfg,
                                          int n_moved = -1) {
    // Pivot / rigid: CellOnly unless explicitly overridden.
    if ((kind == MoveKind::Pivot || is_rigid) && !cfg.pivot_uses_verlet)
        return NeighborMode::CellOnly;
    // Gate / skin=0: treat as cell-only algorithmically (skin value may stay >0).
    if (cfg.skin <= 0.f || !cfg.mu_verlet_enabled)
        return NeighborMode::CellOnly;
    // Large moved sets: Verlet list reuse does not pay for itself.
    if (n_moved >= 0 && n_moved > cfg.verlet_moved_threshold)
        return NeighborMode::CellOnly;
    if (kind == MoveKind::KIC || kind == MoveKind::Sidechain)
        return NeighborMode::VerletPreferred;
    return NeighborMode::CellOnly;
}

} // namespace mcpu
