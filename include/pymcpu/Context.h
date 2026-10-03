#pragma once
#include <memory>
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <string>
#include "pymcpu/System.h"
#include "pymcpu/State.h"
#include "pymcpu/CellList.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/EnergyWeights.h"
#include "pymcpu/AtomPermutation.h"
#include "pymcpu/AtomReorder.h"
#include "pymcpu/utils/geometry_utils.h"
#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/neighbor/NeighborSystem.h"
#include "pymcpu/neighbor/VerletList.h"
#include "pymcpu/utils/CoordSyncStats.h"

namespace mcpu {

namespace forces::mcpu08 {
class MuPotential;
}

struct MuWorkspace {
    struct ContactUpdate {
        int index;
        bool new_value;
    };
    std::vector<ContactUpdate> pending_updates;

    /// Contacts that begin / end if the pending move is accepted. Lives here,
    /// on the per-Context workspace, so replicas sharing one System (and hence
    /// one MuPotential) cannot tread on each other. Applied by
    /// Context::commit_accepted_move; simply discarded on rejection, because
    /// the next delta call clears them.
    struct PendingContact {
        std::int32_t i;
        std::int32_t j;
        float energy;
    };
    std::vector<PendingContact> pending_contact_drop;
    std::vector<PendingContact> pending_contact_add;

    std::vector<uint32_t> seen_stamp;
    uint32_t cur_stamp = 0;

    std::unique_ptr<CellListMC> moved_new_grid;
    std::vector<int> moved_grid_atoms;

    NeighborMode neighbor_mode = NeighborMode::CellOnly;
    bool use_trial_fallback = false;
    MoveKind move_kind = MoveKind::Other;

    /// Scratch for MCPU_PIVOT_MU_BREAKDOWN phased denselist diagnostic. O(1) reuse.
    struct PivotCand {
        int i = 0;
        int j = 0;
        std::uint8_t is_new = 0;  ///< 0 = old subtract, 1 = new add
    };
    struct PivotInCut {
        int i = 0;
        int j = 0;
        float r2 = 0.f;
        std::uint8_t is_new = 0;
    };
    std::vector<PivotCand> pivot_cand_scratch;
    std::vector<PivotInCut> pivot_incut_scratch;

    /// Per-side grouping of moved atoms by cell (cell-pair denselist). O(1) reuse.
    struct MovedCellGroups {
        struct CellGroup {
            int cell_id = -1;
            // FIXED: was int[32] while build_moved_cell_groups bounds-checks
            // g.count against OpenCellGrid::CELL_CAPACITY, which was raised
            // from 32 to 48. Counts 32..47 therefore wrote past this array
            // into `count` and the next CellGroup. Unreachable at production
            // occupancy (actin peaks ~22) but reachable at
            // MCPU_MU_CELL_SCALE>=1.2, which measured max_occ 31/46 -- i.e.
            // this made that knob unsafe to even benchmark. Bind the size to
            // the capacity that is actually enforced.
            int moved_ids[OpenCellGrid::CELL_CAPACITY]{};
            int count = 0;
        };
        // 512: exact denselist (~5.08 Å cells) can exceed 256 unique moved
        // cells on large pivots; 256 was sized for 6 Å cells.
        static constexpr int MAX_GROUPS = 512;
        CellGroup groups[MAX_GROUPS]{};
        int n_groups = 0;
    };
    MovedCellGroups old_moved_groups;
    MovedCellGroups new_moved_groups;
    /// Scratch: cell_id → group index; size = n_cells. Filled with -1. O(1) reuse.
    std::vector<int> cell_to_group_scratch;

    void clear() {
        pending_updates.clear();
        pending_contact_drop.clear();
        pending_contact_add.clear();
    }

    void ensure_stamp_capacity(int num_atoms) {
        if (static_cast<int>(seen_stamp.size()) < num_atoms) {
            seen_stamp.assign(static_cast<size_t>(num_atoms), 0u);
            cur_stamp = 0;
        }
    }

    uint32_t next_stamp() {
        if (++cur_stamp == 0) {
            std::fill(seen_stamp.begin(), seen_stamp.end(), 0u);
            cur_stamp = 1;
        }
        return cur_stamp;
    }


    void ensure_moved_grid(float cutoff, int num_atoms) {
        if (!moved_new_grid) {
            moved_new_grid = std::make_unique<CellListMC>(cutoff, num_atoms);
        } else {
            clear_moved_grid();
        }
        moved_new_grid->ensure_atom_capacity(num_atoms);
    }

    void clear_moved_grid() {
        if (!moved_new_grid) return;
        for (int a : moved_grid_atoms) {
            moved_new_grid->remove(a);
        }
        moved_grid_atoms.clear();
    }
};

struct QBiasWorkspace {
    struct PairUpdate {
        int index;
        bool new_value;
    };
    std::vector<PairUpdate> pending_updates;

    void clear() { pending_updates.clear(); }
};

/// Persistent scratch for HBond ΔE (no per-call unordered_set / unbounded alloc).
///
/// Pair-key duplicates arise because the same (r_don, r_acc) can be discovered from:
///   - donor queries at both old and new H positions
///   - acceptor queries at both old and new O positions
///   - dual donor+acceptor enumeration of the same pair
///   - affected×affected dual-move pass overlapping grid discoveries
/// Key = r_don * n_res + r_acc (residue-pair stamp, not N_atom²).
struct HBondWorkspace {
    std::vector<uint32_t> pair_stamp;
    uint32_t cur_stamp = 1;
    int n_res = 0;

    std::vector<uint8_t> res_affected;
    std::vector<int> aff_list;

    void ensure_capacity(int num_residues) {
        if (num_residues == n_res &&
            static_cast<int>(pair_stamp.size()) == num_residues * num_residues) {
            return;
        }
        n_res = num_residues;
        const size_t n2 = static_cast<size_t>(num_residues) * static_cast<size_t>(num_residues);
        pair_stamp.assign(n2, 0u);
        cur_stamp = 1;
        res_affected.assign(static_cast<size_t>(num_residues), 0);
        aff_list.clear();
        aff_list.reserve(64);
    }

    /// Begin one ΔE evaluation: bump stamp; overflow clears the stamp vector.
    void begin_call(int num_residues) {
        ensure_capacity(num_residues);
        if (++cur_stamp == 0u) {
            std::fill(pair_stamp.begin(), pair_stamp.end(), 0u);
            cur_stamp = 1;
        }
        std::fill(res_affected.begin(), res_affected.end(), static_cast<uint8_t>(0));
        aff_list.clear();
    }

    /// Returns true if this is the first time (r_don, r_acc) is seen this call.
    [[nodiscard]] bool mark_pair(int r_don, int r_acc) noexcept {
        const size_t key =
            static_cast<size_t>(r_don) * static_cast<size_t>(n_res) + static_cast<size_t>(r_acc);
        if (pair_stamp[key] == cur_stamp) return false;
        pair_stamp[key] = cur_stamp;
        return true;
    }

    void clear() {
        // Stamp bump makes pair_stamp reusable; keep buffers.
        aff_list.clear();
    }
};

class Context {
private:
    void sync_geometry();
    void computeTorsions();
    void reinitialize_q_pair_cache();

    std::shared_ptr<System> system;
    State state;

    /// Sole owner of Mu + HBond spatial indices (single lifecycle).
    NeighborSystem neighbors_;

    /// Reusable scratch for Verlet accumulate_accept (moved atoms only). CHANGED: sparse.
    CoordsSoA commit_old_coords_scratch_;

    mutable MuWorkspace mu_workspace_;
    mutable QBiasWorkspace q_bias_workspace_;
    mutable HBondWorkspace hbond_workspace_;
    float q_bias_k_ = 0.0f;
    float q_bias_target_ = 0.0f;

    /// Outer energy weights (legacy defaults). Applied in System::getTotalEnergy /
    /// getDeltaEnergy. Raw per-potential energies remain accessible via energyBreakdown /
    /// calculate_total_energy_raw.
    EnergyWeights energy_weights_;

    /// Optional per-energy-group ΔE timers (enabled by Integrator instrumentation).
    mutable bool energy_delta_timing_enabled_ = false;
    mutable std::uint64_t energy_delta_ns_[8] = {};

    AtomReorderMode atom_reorder_mode_ = AtomReorderMode::Off;
    AtomPermutation atom_perm_ = AtomPermutation::identity(0);
    bool output_internal_order_ = false;
    bool positions_set_ = false;
    bool reorder_applied_ = false;
    RejectReason last_total_reject_reason_ = RejectReason::None;

    void maybe_apply_init_only_reorder_();

public:
    float contactCutoffA() const noexcept { return neighbors_.mu_cutoff_A(); }
    static constexpr float hbondCutoffA() noexcept { return NeighborSystem::kHBondCutoffA; }

    explicit Context(std::shared_ptr<System> sys);
    void setPositions(const Eigen::Matrix3Xf& new_coords);
    /// Throws if another Context reordered this System's atoms after this one
    /// was created: this one's coordinates are then in the wrong order. A
    /// Context created on an already reordered System adopts its permutation.
    void require_current_atom_order() const;
    void commit_accepted_move(const State& proposed_state, const ProposalPatch& patch,
                              MoveKind move_kind = MoveKind::Other);
    float calculate_total_energy(int target_group = -1);
    float calculate_total_energy_raw(int target_group = -1) const;
    float calculate_delta_energy(const State& proposed_state, const ProposalPatch& patch) const;
    [[nodiscard]] bool has_hard_constraint_violation() const noexcept {
        return last_total_reject_reason_ != RejectReason::None;
    }
    [[nodiscard]] bool has_steric_clash() const noexcept {
        return last_total_reject_reason_ == RejectReason::StericClash;
    }


    EnergyBreakdown energy_breakdown() const;
    void setQBias(float k_bias, float n_target);
    /// Alias: harmonic umbrella on hard native-contact count N (``N0 = n_target``).
    void setNativeContactsBias(float k_bias, float n_target) { setQBias(k_bias, n_target); }

    EnergyWeights& energyWeights() noexcept { return energy_weights_; }
    const EnergyWeights& energyWeights() const noexcept { return energy_weights_; }
    void set_use_legacy_weights(bool on) noexcept { energy_weights_.set_use_legacy_weights(on); }
    bool use_legacy_weights() const noexcept { return energy_weights_.use_legacy_weights; }
    void set_energy_weight(int group_id, float w) { energy_weights_.set_energy_weight(group_id, w); }

    void set_atom_reorder_mode(AtomReorderMode mode);
    void set_atom_reorder_mode(const std::string& mode);
    [[nodiscard]] AtomReorderMode atom_reorder_mode() const noexcept { return atom_reorder_mode_; }
    [[nodiscard]] std::string get_atom_reorder_mode() const {
        return atom_reorder_mode_to_string(atom_reorder_mode_);
    }
    [[nodiscard]] const AtomPermutation& atom_permutation() const noexcept { return atom_perm_; }

    /// Snapshot for Python / perf packets.
    struct AtomPermutationInfo {
        bool enabled = false;
        std::string mode = "off";
        std::uint64_t permutation_checksum = 0;
        int n_atoms = 0;
        int n_res = 0;
    };
    [[nodiscard]] AtomPermutationInfo atom_permutation_info() const {
        AtomPermutationInfo info;
        info.enabled = reorder_applied_ && !atom_perm_.is_identity();
        info.mode = atom_reorder_mode_to_string(atom_reorder_mode_);
        info.permutation_checksum = atom_perm_.checksum();
        info.n_atoms = system ? system->getNumAtoms() : 0;
        info.n_res = system ? system->getNumResidues() : 0;
        return info;
    }
    void set_output_internal_order(bool on) noexcept { output_internal_order_ = on; }
    [[nodiscard]] bool output_internal_order() const noexcept { return output_internal_order_; }

    /// Python/IO: coords in external order unless output_internal_order.
    [[nodiscard]] Eigen::Matrix3Xf coords_for_python() const;
    void set_coords_from_python(const Eigen::Matrix3Xf& coords);

    float getQBiasK() const noexcept { return q_bias_k_; }
    float getQBiasTarget() const noexcept { return q_bias_target_; }

    NeighborSystem& neighbors() noexcept { return neighbors_; }
    const NeighborSystem& neighbors() const noexcept { return neighbors_; }

    NeighborConfig& neighborConfig() noexcept { return neighbors_.config(); }
    const NeighborConfig& neighborConfig() const noexcept { return neighbors_.config(); }
    NeighborStats& neighborStats() const noexcept { return neighbors_.stats(); }
    void reset_neighbor_proxy_stats() {
        neighbors_.stats().reset();
    }
    [[deprecated(
        "Neighbor-list tuning only. Auto-print was removed from Integrator::run(). "
        "Will be removed in a future release.")]]
    void print_neighbor_proxy_stats(const char* tag = "neighbor-proxy") const;
    [[deprecated(
        "Auto-print was removed from Integrator::run(). This setter has no effect. "
        "Will be removed in a future release.")]]
    void set_proxy_print_every(int n);
    void set_mu_skin(float skin) {
        neighbors_.config().skin = skin;
        if (skin <= 0.f) {
            neighbors_.config().mu_verlet_enabled = false;
            neighbors_.muVerlet().invalidate();
        } else {
            neighbors_.config().mu_verlet_enabled = true;
        }
        if (positions_set_) {
            // Skin changes the Mu grid query radius, stencil, AABB margin, and
            // Verlet shell. Rebuild the complete accepted-state index.
            neighbors_.rebuild_from_accepted_state(state.coords_soa);
        }
    }
    float mu_skin() const noexcept { return neighbors_.config().skin; }
    bool mu_verlet_enabled() const noexcept {
        return neighbors_.config().mu_verlet_enabled;
    }
    /// Force CellOnly when n_moved > threshold (0 ⇒ always CellOnly for valid moves).
    void set_verlet_moved_threshold(int n) noexcept {
        neighbors_.config().verlet_moved_threshold = n;
    }
    int verlet_moved_threshold() const noexcept {
        return neighbors_.config().verlet_moved_threshold;
    }
    /// Partial Verlet rebuild threshold (default 50). O(1).
    void set_verlet_partial_threshold(int n) noexcept {
        neighbors_.config().verlet_partial_threshold = n;
    }
    int verlet_partial_threshold() const noexcept {
        return neighbors_.config().verlet_partial_threshold;
    }
    /// Gate Verlet use without changing skin / denselist geometry.
    void set_mu_verlet_enabled(bool on) noexcept {
        neighbors_.config().mu_verlet_enabled = on;
        if (!on) neighbors_.muVerlet().invalidate();
    }
    /// Skip the pairs a rigid pivot carries (default true); see
    /// NeighborConfig::skip_rigid_mm. O(1) flag.
    void set_skip_rigid_mm(bool on) noexcept {
        neighbors_.config().skip_rigid_mm = on;
    }
    bool skip_rigid_mm() const noexcept {
        return neighbors_.config().skip_rigid_mm;
    }

    /// Cell-pair denselist Mu (default true). O(1) flag.
    void set_use_cell_pair(bool on) noexcept {
        neighbors_.config().use_cell_pair = on;
    }
    bool use_cell_pair() const noexcept {
        return neighbors_.config().use_cell_pair;
    }
    /// Min moved atoms for cell-pair (default 20). O(1).
    void set_cell_pair_min_moved(int n) noexcept {
        neighbors_.config().cell_pair_min_moved = n;
    }
    int cell_pair_min_moved() const noexcept {
        return neighbors_.config().cell_pair_min_moved;
    }
    /// Min moved atoms for the Mu clash-first pass (default 50). O(1).
    void set_clash_first_min_moved(int n) noexcept {
        neighbors_.config().clash_first_min_moved = n;
    }
    int clash_first_min_moved() const noexcept {
        return neighbors_.config().clash_first_min_moved;
    }

    /// First MuPotential (nullptr if none). For diagnostics / verify.
    [[nodiscard]] forces::mcpu08::MuPotential* mu_potential();
    [[nodiscard]] const forces::mcpu08::MuPotential* mu_potential() const;
    /// Legacy opt-in: restore unconditional pivot→Verlet invalidate (default off).
    void set_invalidate_verlet_on_pivot_accept(bool on) noexcept {
        neighbors_.config().invalidate_verlet_on_pivot_accept = on;
    }
    bool invalidate_verlet_on_pivot_accept() const noexcept {
        return neighbors_.config().invalidate_verlet_on_pivot_accept;
    }


    /// Mu denselist cell size = scale * Mu cutoff, never below the cutoff (so
    /// scale < 1 acts as 1). Default 1.0. The skin does not enter it.
    /// Rebuilds the Mu denselist when positions are already set (scale must be
    /// set before ``setPositions`` / init_only reorder for matching locality).
    void set_mu_cell_size_scale(float scale) noexcept {
        neighbors_.config().mu_cell_size_scale = scale;
        neighbors_.config().mu_cell_size_angstrom = -1.f;
        if (positions_set_) {
            neighbors_.rebuild_from_accepted_state(state.coords_soa);
        }
    }
    float mu_cell_size_scale() const noexcept {
        return neighbors_.config().mu_cell_size_scale;
    }
    /// Absolute Mu denselist cell size (Å), raised to the Mu cutoff if smaller.
    /// &lt;=0 clears absolute override.
    void set_mu_cell_size_angstrom(float angstrom) noexcept {
        neighbors_.config().mu_cell_size_angstrom = angstrom;
        if (positions_set_) {
            neighbors_.rebuild_from_accepted_state(state.coords_soa);
        }
    }
    float mu_cell_size_angstrom() const noexcept {
        return neighbors_.config().mu_cell_size_angstrom;
    }
    /// Extra lower bound on the cell (Å); only a value above the Mu cutoff
    /// has an effect. &lt;=0 clears it.
    void set_mu_cell_size_min_angstrom(float angstrom) noexcept {
        neighbors_.config().mu_cell_size_min_angstrom =
            angstrom > 0.f ? angstrom : 0.f;
        if (positions_set_) {
            neighbors_.rebuild_from_accepted_state(state.coords_soa);
        }
    }
    float mu_cell_size_min_angstrom() const noexcept {
        return neighbors_.config().mu_cell_size_min_angstrom;
    }
    float effective_mu_cell_size_A() const noexcept {
        // Denselist cell ignores skin (Verlet widens stencil only at rebuild).
        const float r_cut = neighbors_.mu_cutoff_A();
        return ::mcpu::effective_mu_cell_size_A(
            r_cut, /*skin=*/0.f, neighbors_.config());
    }

    const BoxBounds& boxBounds() const noexcept { return neighbors_.bounds(); }
    VerletList& verletContact() noexcept { return neighbors_.muVerlet(); }
    const VerletList& verletContact() const noexcept { return neighbors_.muVerlet(); }
    bool denseGridsActive() const noexcept { return neighbors_.denseActive(); }

    bool trial_in_bounds(const State& proposal, const ProposalPatch& patch) const {
        return neighbors_.trial_in_bounds(proposal.coords_soa, patch);
    }
    static float max_moved_displacement(const State& accepted, const State& proposal,
                                        const ProposalPatch& patch) {
        return NeighborSystem::max_moved_displacement(
            accepted.coords_soa, proposal.coords_soa, patch);
    }

    void maybe_rebuild_verlet() {
        neighbors_.maybe_rebuild_mu_verlet(state.coords_soa);
    }
    void invalidate_verlet_pivot_accept() {
        neighbors_.invalidate_mu_verlet_pivot();
    }


    // NOTE: coord_sync_stats() is a single process-wide global (see
    // CoordSyncStats.h), not per-Context state. Calling these on any one
    // Context reads/resets the SAME counters shared by every other Context
    // instance (e.g. all walkers in a multi-walker/replica run).
    void reset_coord_sync_stats() noexcept { coord_sync_stats().reset(); }
    CoordSyncStats& coordSyncStats() noexcept { return coord_sync_stats(); }
    const CoordSyncStats& coordSyncStats() const noexcept { return coord_sync_stats(); }

    MuWorkspace& getWorkspace() noexcept { return mu_workspace_; }
    const MuWorkspace& getMuWorkspace() const { return mu_workspace_; }
    QBiasWorkspace& getQBiasWorkspace() noexcept { return q_bias_workspace_; }
    const QBiasWorkspace& getQBiasWorkspace() const { return q_bias_workspace_; }
    HBondWorkspace& getHBondWorkspace() noexcept { return hbond_workspace_; }
    const HBondWorkspace& getHBondWorkspace() const { return hbond_workspace_; }
    friend class MCIntegrator;
    friend class System;

    /// Enable/disable per-energy-group ScopedTimers inside evaluateDeltaEnergy.
    void set_energy_delta_timing(bool on) const noexcept {
        energy_delta_timing_enabled_ = on;
        if (!on) {
            for (auto& v : energy_delta_ns_) v = 0;
        }
    }
    [[nodiscard]] bool energy_delta_timing() const noexcept {
        return energy_delta_timing_enabled_;
    }
    /// Pointer to accumulator for energy group ``g`` (1..7), or nullptr if off/OOB.
    [[nodiscard]] std::uint64_t* energy_delta_ns_slot(int group) const noexcept {
        if (!energy_delta_timing_enabled_ || group < 0 || group >= 8) return nullptr;
        return &energy_delta_ns_[static_cast<size_t>(group)];
    }
    void copy_energy_delta_ns_into(std::uint64_t out[8]) const noexcept {
        for (int i = 0; i < 8; ++i) out[i] = energy_delta_ns_[static_cast<size_t>(i)];
    }
    void reset_energy_delta_ns() const noexcept {
        for (auto& v : energy_delta_ns_) v = 0;
    }

    [[nodiscard]] const State&      getState()  const { return state; }
    [[nodiscard]] State&            getState() { return state; }
    [[nodiscard]] const System&     getSystem() const { return *system; }
    [[nodiscard]] System&           getSystem() { return *system; }

    /// Mu index only (BB+O+SC). Prefer neighbors().for_each_mu_candidate.
    [[nodiscard]] const CellListMC& getGridContact() const { return neighbors_.muGrid(); }

};

} // namespace mcpu
