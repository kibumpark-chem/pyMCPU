#pragma once
#include <memory>
#include <optional>
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
#include "pymcpu/neighbor/Footprint.h"
#include "pymcpu/neighbor/MovedCells.h"
#include "pymcpu/neighbor/PairLedger.h"
#include "pymcpu/neighbor/PairScratch.h"
#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/neighbor/NeighborSystem.h"
#include "pymcpu/utils/CoordSyncStats.h"

namespace mcpu {

namespace forces::mcpu08 {
class MuPotential;
}

struct MuWorkspace {
    /// Contact-list entries that go / come if the pending move is accepted
    /// (energy 0 is a listed near miss). Lives here, on the per-Context
    /// workspace, so replicas sharing one System (and hence one MuPotential)
    /// cannot tread on each other. Applied by Context::commit_accepted_move;
    /// simply discarded on rejection, because the next delta call clears them.
    using PendingContact = neighbor::PendingPairs<float>::Pair;
    neighbor::PendingPairs<float> pending_contacts;
    /// The pending rigid move's bound on carried-distance change, added to
    /// State::mu_list_drift if it is accepted.
    double pending_list_drift = 0.0;
    /// The pending move could not use the contact list, so the list is
    /// dropped if the move is accepted.
    bool pending_list_invalidate = false;

    // The moved-cell counts and the clash_hot list live in Context's
    // neighbor::PairScratch, shared by every term's pair walks.

    void clear() {
        pending_contacts.clear();
        pending_list_drift = 0.0;
        pending_list_invalidate = false;
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

    // Per residue: kAffected when a pair with it as donor or acceptor can
    // change energy; plus kBackboneMoved when its N, CA, C, O (and explicit
    // amide H) all moved, and kRigidSite when that holds for r-1, r and r+1
    // under a rigid move, so every atom its donor or acceptor geometry reads
    // moved with one rigid transform. Pairs of two rigid sites keep their
    // energy and are skipped (NeighborConfig::skip_rigid_mm).
    static constexpr uint8_t kAffected = 1, kBackboneMoved = 2, kRigidSite = 4;
    std::vector<uint8_t> res_affected;
    std::vector<int> aff_list;

    // Grid walks for the affected sites: atom_aff flags the O (and explicit
    // H) atoms of the affected residues, which the walks skip as partners;
    // o_sites / h_sites list the ids the O and H grids hold for them, so a
    // MovedCellScope can skip cells that hold nothing else. All zero / empty
    // between calls.
    std::vector<uint8_t> atom_aff;
    std::vector<int> o_sites, h_sites;
    neighbor::MovedCellCounts o_cells, h_cells;

    // Old and new O coordinates of the affected acceptors, packed for the
    // affected-donor x affected-acceptor scan; padded to a multiple of 8.
    std::vector<int> acc_res;
    std::vector<float> acc_old_x, acc_old_y, acc_old_z;
    std::vector<float> acc_new_x, acc_new_y, acc_new_z;

    // What the last calculateEnergyChange would fold into the accepted
    // state's HBondStateCache: the new energy of every evaluated pair that is
    // nonzero, against the cache generation it read. aff_list (above) names
    // the residues whose pairs it replaces. Checked and consumed by
    // HBondPotential::commitAcceptedMove.
    struct PendingPair { int d, a; float e, fresh_until; };
    std::vector<PendingPair> pending;
    // The ledger drift the proposed state would have if accepted.
    double pending_drift = 0.0;
    bool pending_valid = false;
    const void* pending_old_state = nullptr;
    const void* pending_proposed_state = nullptr;
    std::size_t pending_num_moved = 0;
    std::uint64_t pending_generation = 0;

    // Debug check (off by default; Context.set_hbond_ledger_check): on every
    // delta, recompute both states over all pairs that touch an affected
    // residue and count pairs whose nonzero-delta status or old energy
    // disagrees with the ledger path.
    bool ledger_check = false;
    std::uint64_t ledger_checks = 0;
    std::uint64_t ledger_mismatches = 0;

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

    mutable MuWorkspace mu_workspace_;
    /// Scratch for the shared pair walks (moved-cell counts per grid, clash
    /// order hints). Per Context, so replicas never share it.
    mutable neighbor::PairScratch pair_scratch_;
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
    // atom_perm_.is_identity(), kept in step with atom_perm_ so the move
    // hot path does not walk the permutation on every pivot.
    bool atom_perm_identity_ = true;
    bool output_internal_order_ = false;
    bool positions_set_ = false;
    bool reorder_applied_ = false;
    RejectReason last_total_reject_reason_ = RejectReason::None;
    /// User coordinates = engine coordinates + frame_offset_ (see
    /// utils/FrameOffset.h). Chosen at the first placement, kept after that.
    Eigen::Vector3d frame_offset_ = Eigen::Vector3d::Zero();

    void maybe_apply_init_only_reorder_();
    /// User-frame coordinates (any order) to the engine frame. Chooses the
    /// frame offset on the first placement, or takes frame_offset if given.
    Eigen::Matrix3Xf enter_frame_(const Eigen::Matrix3Xd& coords,
                                  const std::optional<Eigen::Vector3d>& frame_offset);

public:
    float contactCutoffA() const noexcept { return neighbors_.mu_cutoff_A(); }
    static constexpr float hbondCutoffA() noexcept { return NeighborSystem::kHBondCutoffA; }

    explicit Context(std::shared_ptr<System> sys);
    /// Places the atoms, in build order and the user's frame. The first
    /// placement fixes the frame offset (utils/FrameOffset.h) unless
    /// frame_offset is given, which replaces it.
    void setPositions(const Eigen::Matrix3Xd& new_coords,
                      const std::optional<Eigen::Vector3d>& frame_offset = std::nullopt);
    /// Engine coordinates (get_state()) = user coordinates - frame_offset().
    [[nodiscard]] const Eigen::Vector3d& frame_offset() const noexcept { return frame_offset_; }
    /// Throws if another Context reordered this System's atoms after this one
    /// was created: this one's coordinates are then in the wrong order. A
    /// Context created on an already reordered System adopts its permutation.
    void require_current_atom_order() const;
    void commit_accepted_move(const State& proposed_state, const ProposalPatch& patch);
    double calculate_total_energy(int target_group = -1);
    double calculate_total_energy_raw(int target_group = -1) const;
    double calculate_delta_energy(const State& proposed_state, const ProposalPatch& patch) const;
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
    [[nodiscard]] bool atom_permutation_is_identity() const noexcept { return atom_perm_identity_; }

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

    /// Python/IO: coords in the user's frame, in external order unless
    /// output_internal_order. Double, so engine + offset is exact.
    [[nodiscard]] Eigen::Matrix3Xd coords_for_python() const;
    void set_coords_from_python(const Eigen::Matrix3Xd& coords);

    float getQBiasK() const noexcept { return q_bias_k_; }
    float getQBiasTarget() const noexcept { return q_bias_target_; }

    NeighborSystem& neighbors() noexcept { return neighbors_; }
    const NeighborSystem& neighbors() const noexcept { return neighbors_; }

    NeighborConfig& neighborConfig() noexcept { return neighbors_.config(); }
    const NeighborConfig& neighborConfig() const noexcept { return neighbors_.config(); }
    /// Most a rigid move can change a carried atom-atom distance, in A:
    /// sqrt(3) float steps of the largest coordinate the dense grid holds,
    /// +1% for the double-precision rotation itself. Every coordinate is
    /// the double image of a float rounded once (CoordsSoA::rotate_atoms).
    /// The Mu contact list and the H-bond ledger budget their carries by it.
    float rigid_carry_bound_A(const State& new_state, const ProposalPatch& patch) const;
    NeighborStats& neighborStats() const noexcept { return neighbors_.stats(); }
    void reset_neighbor_proxy_stats() {
        neighbors_.stats().reset();
    }
    [[deprecated(
        "Neighbor-list tuning only. Auto-print was removed from Integrator::run(). "
        "Will be removed in a future release.")]]
    void print_neighbor_proxy_stats(const char* tag = "neighbor-proxy") const;
    /// Skip the pairs a rigid pivot carries (default true); see
    /// NeighborConfig::skip_rigid_mm. O(1) flag.
    void set_skip_rigid_mm(bool on) noexcept {
        neighbors_.config().skip_rigid_mm = on;
    }
    bool skip_rigid_mm() const noexcept {
        return neighbors_.config().skip_rigid_mm;
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

    const BoxBounds& boxBounds() const noexcept { return neighbors_.bounds(); }
    bool denseGridsActive() const noexcept { return neighbors_.denseActive(); }
    /// Bring the Mu grid membership up to the System's energy mask after a
    /// mask change (NeighborSystem::sync_energy_mask). Until then the grid
    /// reads as inactive and Mu takes the exact all-pairs path.
    void sync_energy_mask() {
        if (positions_set_) neighbors_.sync_energy_mask(state.coords_soa);
    }

    bool trial_in_bounds(const State& proposal, const ProposalPatch& patch) const {
        return neighbors_.trial_in_bounds(proposal.coords_soa, patch);
    }
    static float max_moved_displacement(const State& accepted, const State& proposal,
                                        const ProposalPatch& patch) {
        return NeighborSystem::max_moved_displacement(
            accepted.coords_soa, proposal.coords_soa, patch);
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
    neighbor::PairScratch& pairScratch() const noexcept { return pair_scratch_; }
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

    /// Mu index only (BB+O+SC).
    [[nodiscard]] const CellListMC& getGridContact() const { return neighbors_.muGrid(); }

};

} // namespace mcpu
