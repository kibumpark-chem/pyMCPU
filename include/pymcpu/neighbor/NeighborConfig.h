#pragma once
#include <cstdint>
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
    float margin_angstrom = -1.f; // <0 => 2*r_cut
    int margin_cells = -1;

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

    /// Minimum moved-atom count for the clash-first pass (MuPotential). Below
    /// this the pass is nearly pure overhead: a small move examines few pairs
    /// anyway, so skipping the contact walk saves little while the extra pass
    /// still costs. Measured ungated: -9.6% on chignolin (largest move ~25
    /// atoms) against +14% on actin (pivots ~740 atoms). Swept over
    /// chignolin/1igd/actin; 50 is neutral on chignolin and within noise of
    /// 100/300 on actin. The true crossover lies between 25 and 50 and is
    /// unmeasured.
    int clash_first_min_moved = 50;


    /// Denselist query/cell cutoff (Å). Filled by Context::sync_geometry from
    /// MuPotential::mu_exact_cutoff(). &lt;0 → NeighborSystem falls back to 6.0.
    float mu_denselist_cutoff_A = -1.f;


    float effective_margin(float r_cut) const {
        if (margin_angstrom >= 0.f) return margin_angstrom;
        if (margin_cells >= 0) return static_cast<float>(margin_cells) * r_cut;
        return 2.f * r_cut;
    }
};

/// Per-step neighbour work proxies derived from raw NeighborStats counters.
///
/// - avg_mu_pair_checks_per_step: r2 computations / step (primary work proxy).
/// - avg_mu_pairs_within_rcut_per_step: pairs with r2 <= r_mu^2 / step.
struct NeighborProxyReport {
    std::uint64_t total_steps = 0;
    double avg_mu_pair_checks_per_step = 0.0;
    double avg_mu_pairs_within_rcut_per_step = 0.0;
    /// Deprecated alias of avg_mu_pair_checks_per_step (historical name).
    double avg_mu_pairs_per_step = 0.0;
    double avg_cell_visits_per_step = 0.0;
    double avg_hbond_geom_checks_per_step = 0.0;
};

struct NeighborStats {
    // --- lifecycle / policy ---
    /// Times a cell of the Mu grid / an H-bond grid was asked to hold more
    /// atoms than its capacity; the grid then goes inactive until a rebuild
    /// fits (NeighborSystem::note_overflow_).
    std::uint64_t mu_grid_overflows = 0;
    std::uint64_t hbond_grid_overflows = 0;
    std::uint64_t num_reject_hard_disp = 0;
    /// Full grid rebuilds (rebuild_from_accepted_state): set_positions and
    /// the rebuild after a cell overflow.
    std::uint64_t num_grid_rebuilds = 0;

    // --- performance proxy counters ---
    /// Every time Mu delta computes r2 for a candidate pair (i,j).
    std::uint64_t mu_num_pair_distance_checks = 0;
    /// Subset of distance checks with r2 <= r_mu^2 (6 Å cell cutoff).
    std::uint64_t mu_num_pairs_within_rcut = 0;
    std::uint64_t hbond_num_candidates_iterated = 0;
    std::uint64_t hbond_num_geom_checks = 0;
    std::uint64_t neighbor_num_cell_visits = 0;

    /// Mu denselist stencil (variable radius).
    std::uint64_t neighbor_offsets_count = 0;

    /// Pair-eval counter flushed from MuPotential (useful denselist proxy).
    std::uint64_t mu_eval_pair_calls = 0;
    /// Subset of eval_pair calls that returned a non-zero energy contribution.
    std::uint64_t mu_eval_pair_nonzero = 0;
    /// Moved–moved denselist candidates skipped for rigid pivots.
    std::uint64_t elided_rigid_mm = 0;


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
        r.avg_cell_visits_per_step = static_cast<double>(neighbor_num_cell_visits) /
                                     static_cast<double>(den_steps);
        r.avg_hbond_geom_checks_per_step = static_cast<double>(hbond_num_geom_checks) /
                                           static_cast<double>(den_steps);
        return r;
    }
};

} // namespace mcpu
