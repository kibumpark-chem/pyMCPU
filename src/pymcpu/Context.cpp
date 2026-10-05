#include "pymcpu/Context.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include "pymcpu/forces/bias/QBiasPotential.h"
#include "pymcpu/forces/mcpu/mcpu08/MuPotential.h"
#include "pymcpu/AtomReorder.h"
#include "pymcpu/utils/geometry_utils.h"
#include "pymcpu/utils/sidechain_torsion_utils.h"
#include "pymcpu/utils/FrameOffset.h"
#include <atomic>
#include <cstdint>
#include <cmath>
#include <utility>
#include <cstdio>
#include "pymcpu/utils/numbers_compat.h"


namespace mcpu {

Context::Context(std::shared_ptr<System> sys)
    : system(std::move(sys))
    , state(system->getNumAtoms(), system->getNumResidues())
    , atom_perm_(AtomPermutation::identity(system->getNumAtoms()))
{
    // A System another Context reordered (REMD replicas share one) is in
    // storage order: take its permutation, so coordinates given in build
    // order are mapped like that Context's.
    if (system->atoms_reordered()) {
        atom_perm_ = system->applied_atom_permutation();
        atom_perm_identity_ = atom_perm_.is_identity();
        atom_reorder_mode_ = AtomReorderMode::InitOnly;
        reorder_applied_ = true;
    }
    neighbors_.init(*system);
}

void Context::require_current_atom_order() const {
    if (system->atoms_reordered() && !reorder_applied_) {
        throw std::runtime_error(
            "this Context was created before another Context reordered its "
            "System's atoms (init_only), so its coordinates are in the old "
            "order; create the Context again.");
    }
}

void Context::set_atom_reorder_mode(AtomReorderMode mode) {
    atom_reorder_mode_ = mode;
    if (mode == AtomReorderMode::Off) {
        // Keep identity; do not undo an already-applied permutation in this PR.
        return;
    }
    require_current_atom_order();
    if (positions_set_ && !reorder_applied_) {
        maybe_apply_init_only_reorder_();
    }
}

void Context::set_atom_reorder_mode(const std::string& mode) {
    set_atom_reorder_mode(atom_reorder_mode_from_string(mode));
}

void Context::maybe_apply_init_only_reorder_() {
    if (atom_reorder_mode_ != AtomReorderMode::InitOnly) return;
    if (reorder_applied_) return;
    if (!positions_set_) return;
    require_current_atom_order();

    std::vector<BlockIndices> new_blocks;
    const NeighborConfig& ncfg = neighbors_.config();
    const float mu_cell = ::mcpu::effective_mu_cell_size_A(
        neighbors_.mu_cutoff_A(), ncfg.skin, ncfg);
    atom_perm_ = compute_init_only_atom_permutation(
        *system, state.coords_soa, ncfg.skin, mu_cell, &new_blocks);
    atom_perm_identity_ = atom_perm_.is_identity();
    if (atom_perm_identity_) {
        reorder_applied_ = true;
        return;
    }

    permute_coords_soa(state.coords_soa, atom_perm_);
    // Also remaps every potential and records the permutation on the System.
    system->apply_residue_contiguous_blocks(std::move(new_blocks), atom_perm_);
    // Pair indices change meaning under a permutation -- drop the contact list.
    state.invalidate_coordinate_caches();
    // NeighborSystem caches donor bb_starts from System — re-init + rebuild.
    // Registered subset grids name atoms by their pre-reorder ids.
    neighbors_.remap_subset_members(atom_perm_.ext_to_int);
    neighbors_.init(*system, neighbors_.config());
    sync_geometry();
    computeTorsions();
    if (q_bias_k_ > 0.0f) {
        reinitialize_q_pair_cache();
    }
    reorder_applied_ = true;
}

Eigen::Matrix3Xd Context::coords_for_python() const {
    Eigen::Matrix3Xd out =
        (output_internal_order_ || atom_perm_.is_identity())
            ? state.coords_as_eigen().cast<double>().eval()
            : scatter_internal_to_external(state.coords_soa, atom_perm_).cast<double>().eval();
    // Back to the user's frame. Exact in double unless an engine coordinate
    // lies within about 2^-29 |offset| of zero. An axis without an offset is
    // left alone, so a -0.0 keeps its sign.
    for (int d = 0; d < 3; ++d) {
        if (frame_offset_[d] != 0.0) out.row(d).array() += frame_offset_[d];
    }
    return out;
}

namespace {
// Coordinates must cover every atom slot. load_from_eigen would otherwise
// resize the state silently, and a file written with a different atom layout
// would load with every atom after the first difference shifted.
void require_atom_count(const Eigen::Matrix3Xd& coords, int expected) {
    if (coords.cols() != expected) {
        throw std::invalid_argument(
            "got coordinates for " + std::to_string(coords.cols()) +
            " atoms, but the system has " + std::to_string(expected) +
            ". A checkpoint or restart file written with a different atom "
            "layout (for example by an older pyMCPU) cannot be loaded into "
            "this system.");
    }
}
}  // namespace

Eigen::Matrix3Xf Context::enter_frame_(
        const Eigen::Matrix3Xd& coords,
        const std::optional<Eigen::Vector3d>& frame_offset) {
    if (frame_offset) {
        // Past 2^24 A no float32 coordinate survives the shift.
        if (!frame_offset->allFinite() ||
            frame_offset->cwiseAbs().maxCoeff() >= 16777216.0) {
            throw std::invalid_argument(
                "frame_offset must be finite and under 2^24 A in magnitude");
        }
        frame_offset_ = *frame_offset;
    } else if (!positions_set_) {
        frame_offset_ = choose_frame_offset(coords);
    }
    // Exact for float32 input with the offset chosen from it (see
    // choose_frame_offset). Without an offset this is the plain cast to float.
    Eigen::Matrix3Xf engine(3, coords.cols());
    float far = 0.f;
    for (int d = 0; d < 3; ++d) {
        const double c = frame_offset_[d];
        for (Eigen::Index i = 0; i < coords.cols(); ++i) {
            const double u = coords(d, i);
            engine(d, i) = static_cast<float>(c != 0.0 ? u - c : u);
            if (std::isfinite(engine(d, i))) far = std::max(far, std::fabs(engine(d, i)));
        }
    }
    static std::atomic<bool> noted{false};
    if (far >= kFarFrameNoteA && !noted.exchange(true)) {
        std::fprintf(stderr,
            "NOTE: coordinates reach %.0f A from the origin even in the engine "
            "frame (frame_offset %.0f %.0f %.0f A). One float32 step there is "
            "%.1e A, and every move rounds the atoms it moves to it, so bond "
            "lengths drift and KIC moves fail more often than near the origin. "
            "The engine shifts a structure toward the origin only along axes "
            "it does not straddle, and keeps an explicit frame_offset as "
            "given. (Printed once per process.)\n",
            static_cast<double>(far), frame_offset_[0], frame_offset_[1],
            frame_offset_[2],
            static_cast<double>(std::nextafter(far, 2.f * far) - far));
    }
    return engine;
}

void Context::set_coords_from_python(const Eigen::Matrix3Xd& coords) {
    require_atom_count(coords, system->getNumAtoms());
    require_current_atom_order();
    if (atom_perm_.is_identity()) {
        setPositions(coords);
        return;
    }
    // The array is in whatever order coords_for_python() returns: storage
    // order under set_output_internal_order(true), build order otherwise.
    // setPositions always assumes build order, so it cannot be used here.
    // As in setPositions, the live contact list describes the old coordinates.
    const Eigen::Matrix3Xf engine = enter_frame_(coords, std::nullopt);
    state.invalidate_coordinate_caches();
    if (output_internal_order_) {
        state.coords_soa.load_from_eigen(engine);
    } else {
        gather_external_to_internal(engine, atom_perm_, state.coords_soa);
    }
    sync_geometry();
    computeTorsions();
    if (q_bias_k_ > 0.0f) {
        reinitialize_q_pair_cache();
    }
    positions_set_ = true;
}

void Context::reinitialize_q_pair_cache() {
    for (const auto& potential : system->getPotentials()) {
        if (auto* q_bias = dynamic_cast<const forces::QBiasPotential*>(potential.get())) {
            q_bias->initializePairCache(state, state.q_pair_cache);
            return;
        }
    }
    state.q_pair_cache.clear();
}

void Context::setQBias(float k_bias, float n_target) {
    // ``n_target`` is the native-contact count N0 (not fraction Q*).
    q_bias_k_ = k_bias;
    q_bias_target_ = n_target;
    if (q_bias_k_ > 0.0f) {
        reinitialize_q_pair_cache();
    } else {
        state.q_pair_cache.clear();
    }
}

// --- INITIALIZATION ---
void Context::setPositions(const Eigen::Matrix3Xd& new_coords,
                           const std::optional<Eigen::Vector3d>& frame_offset) {
    require_atom_count(new_coords, system->getNumAtoms());
    require_current_atom_order();
    const Eigen::Matrix3Xf engine = enter_frame_(new_coords, frame_offset);
    // Coordinates are being replaced wholesale (load, REMD swap, restart), so
    // the live contact list describes a conformation that no longer exists.
    state.invalidate_coordinate_caches();
    // Incoming coords are always treated as *external* (build) order.
    if (!reorder_applied_ || atom_perm_.is_identity()) {
        state.coords_soa.load_from_eigen(engine);
    } else {
        gather_external_to_internal(engine, atom_perm_, state.coords_soa);
    }
    positions_set_ = true;
    sync_geometry();
    computeTorsions();
    if (q_bias_k_ > 0.0f) {
        reinitialize_q_pair_cache();
    }
    maybe_apply_init_only_reorder_();
}

void Context::print_neighbor_proxy_stats(const char* tag) const {
    neighbors_.stats().print(tag, neighbors_.config().skin, &neighbors_.config());
}

void Context::set_proxy_print_every(int n) {
    std::fprintf(stderr,
        "[MCPU WARNING] set_proxy_print_every() is deprecated and has no effect.\n"
        "  Auto-print was removed from Integrator::run() to prevent log spam.\n"
        "  Call print_neighbor_proxy_stats() explicitly if needed.\n");
    (void)n;
}

void Context::computeTorsions() {
    const auto& blocks = system->getBlockIndices();
    for (size_t r = 1; r < state.backbone_torsions.size() - 1; ++r) {
        const int n   = blocks[r].bb_start;
        const int ca  = blocks[r].ca_atom();
        const int c   = blocks[r].c_atom();
        const int o   = blocks[r].o_start;
        const int n_prev = blocks[r - 1].bb_start;
        const int ca_prev = blocks[r - 1].ca_atom();
        const int c_prev = blocks[r - 1].c_atom();
        const int o_prev = blocks[r - 1].o_start;
        const int n_next = blocks[r + 1].bb_start;
        const int ca_next = blocks[r + 1].ca_atom();
        const int c_next = blocks[r + 1].c_atom();
        const int o_next = blocks[r + 1].o_start;

        float phi = GeometryUtils::calculate_dihedral(
            state.atom_pos(c_prev),
            state.atom_pos(n),
            state.atom_pos(ca),
            state.atom_pos(c)
        );
        float psi = GeometryUtils::calculate_dihedral(
            state.atom_pos(n),
            state.atom_pos(ca),
            state.atom_pos(c),
            state.atom_pos(n_next)
        );
        float pca = GeometryUtils::calculate_a_PCA(
            state.atom_pos(n_prev),
            state.atom_pos(ca_prev),
            state.atom_pos(o_prev),
            state.atom_pos(n_next),
            state.atom_pos(ca_next),
            state.atom_pos(o_next)
        );
        float bca = GeometryUtils::calculate_a_bCA(
            state.atom_pos(n_prev),
            state.atom_pos(ca_prev),
            state.atom_pos(o_prev),
            state.atom_pos(n_next),
            state.atom_pos(ca_next),
            state.atom_pos(o_next)
        );
        state.backbone_torsions[r] = BackboneTorsionAngles(phi, psi, pca, bca);

        compute_sidechain_chi_angles(state, *system, static_cast<int>(r));
    }
}

void Context::sync_geometry() {
    // CHANGED: always propagate exact denselist cutoff from MuPotential.
    if (forces::mcpu08::MuPotential* mu = mu_potential()) {
        neighbors_.config().mu_denselist_cutoff_A = mu->mu_exact_cutoff();
    }
    neighbors_.rebuild_from_accepted_state(state.coords_soa);
}

void Context::commit_accepted_move(const State& proposed_state, const ProposalPatch& patch,
                                   MoveKind move_kind) {
    auto& mu_ws = mu_workspace_;

    auto& q_ws = q_bias_workspace_;
    for (const auto& update : q_ws.pending_updates) {
        state.q_pair_cache[static_cast<size_t>(update.index)] =
            static_cast<uint8_t>(update.new_value);
    }

    state.current_energy = proposed_state.getEnergy();

    // Fold this accepted move into the live Mu contact list. The pending
    // entries carry their own energies, so this needs nothing from the potential.
    if (mu_ws.pending_list_invalidate) {
        state.mu_contact_invalidate();
    } else if (state.mu_contacts.ready()) {
        mu_ws.pending_contacts.commit_into(state.mu_contacts);
        state.mu_list_drift += mu_ws.pending_list_drift;
    } else if (state.mu_contact_list_prebuilt) {
        // A move that did not adopt the prebuilt list changed the coordinates
        // it was measured from.
        state.mu_contact_invalidate();
    }
    mu_ws.clear();

    for (const auto& potential : system->getPotentials()) {
        potential->commitAcceptedMove(*this, state, proposed_state, patch);
    }

    // CHANGED: sparse — Verlet (skin>0) needs old xyz for moved atoms only.
    // NeighborSystem::commit_accepted_move ignores coords_old. Default skin=0
    // skips Verlet entirely, so avoid the O(N) SoA clone.
    const bool need_old_for_verlet =
        neighbors_.config().skin > 0.f && neighbors_.denseActive() &&
        neighbors_.config().mu_verlet_enabled;
    if (need_old_for_verlet) {
        if (commit_old_coords_scratch_.n != state.coords_soa.n) {
            commit_old_coords_scratch_.resize(state.coords_soa.n);
        }
        // O(n_moved): gather pre-accept positions for accumulate_accept.
        for (int i : patch.moved_indices) {
            commit_old_coords_scratch_.copy_atom_from(state.coords_soa, i, i);
        }
    }

    // Copy accepted trial coordinates into master state (no grid mutation here).
    // moved_indices lists exactly the atoms moving_atoms marks (mark_moved
    // keeps both; reset_for_step relies on the same invariant), so walking it
    // copies the same atoms in O(n_moved) instead of scanning all N.
    {
#ifndef NDEBUG
        std::size_t n_marked = 0;
        for (std::uint8_t m : patch.moving_atoms) n_marked += (m != 0);
        assert(n_marked == patch.moved_indices.size() &&
               "moved_indices must list exactly the atoms moving_atoms marks");
#endif
        for (int atom_idx : patch.moved_indices) {
            state.coords_soa.copy_atom_from(proposed_state.coords_soa, atom_idx, atom_idx);
        }

        for (int r : patch.distorted_bb_residues) {
            state.backbone_torsions[r] = proposed_state.backbone_torsions[r];
        }
        for (int r : patch.distorted_sc_residues) {
            state.sidechain_torsions[r] = proposed_state.sidechain_torsions[r];
        }

#if !defined(NDEBUG)
        // Torsion invariant: arrays should match geometry after accept.
        // Moves currently update coords only; proposal torsion copies are often stale.
        {
            const float pi = mcpu::PI_F;
            auto ang_err = [pi](float a, float b) {
                float d = std::fabs(a - b);
                if (d > pi) d = 2.f * pi - d;
                return d;
            };
            const auto& blocks = system->getBlockIndices();
            const int n_res = system->getNumResidues();
            for (int r : patch.distorted_bb_residues) {
                if (r < 1 || r >= n_res - 1) continue;
                const auto& br = blocks[static_cast<size_t>(r)];
                const auto& bp = blocks[static_cast<size_t>(r - 1)];
                const auto& bn = blocks[static_cast<size_t>(r + 1)];
                const float phi = GeometryUtils::calculate_dihedral(
                    state.atom_pos(bp.c_atom()), state.atom_pos(br.bb_start),
                    state.atom_pos(br.ca_atom()), state.atom_pos(br.c_atom()));
                const float psi = GeometryUtils::calculate_dihedral(
                    state.atom_pos(br.bb_start), state.atom_pos(br.ca_atom()),
                    state.atom_pos(br.c_atom()), state.atom_pos(bn.bb_start));
                const float phi_err = ang_err(phi, state.backbone_torsions[static_cast<size_t>(r)].phi);
                const float psi_err = ang_err(psi, state.backbone_torsions[static_cast<size_t>(r)].psi);
                if (phi_err > 0.01f || psi_err > 0.01f) {
                    std::fprintf(stderr,
                        "TORSION_MISMATCH res=%d phi_err=%.4f psi_err=%.4f\n",
                        r, phi_err, psi_err);
                }
            }
            for (int r : patch.distorted_sc_residues) {
                if (r < 0 || r >= n_res) continue;
                const int n_chi = system->getTorsionsPerResidue()[static_cast<size_t>(r)];
                if (n_chi < 1) continue;
                // Reads the same resolved atom-index table computeTorsions()/
                // recompute_sidechain_torsion() use (see
                // sidechain_torsion_utils.h), not a positional sc/sc+1
                // recomputation -- keeps this debug check from drifting out
                // of sync with the real chi definition.
                const auto& a = system->getChiAtomIndices()[static_cast<size_t>(r)][0];
                if (a[0] < 0) continue;
                const float chi0 = GeometryUtils::calculate_dihedral(
                    state.atom_pos(a[0]), state.atom_pos(a[1]),
                    state.atom_pos(a[2]), state.atom_pos(a[3]));
                const float chi_err = ang_err(
                    chi0, state.sidechain_torsions[static_cast<size_t>(r)].chi_angles[0]);
                if (chi_err > 0.01f) {
                    std::fprintf(stderr,
                        "TORSION_MISMATCH res=%d chi0_err=%.4f\n", r, chi_err);
                }
            }
        }
#endif
    }

    // Single lifecycle update for ALL indices (Mu + HBond).
    // coords_old is unused by NeighborSystem; pass scratch (or state) as placeholder.
    {
        const CoordsSoA& coords_old_arg =
            need_old_for_verlet ? commit_old_coords_scratch_ : state.coords_soa;
        neighbors_.commit_accepted_move(
            patch, coords_old_arg, state.coords_soa, move_kind, patch.is_rigid);
    }

    if (need_old_for_verlet) {
        const bool pivot_like =
            (move_kind == MoveKind::Pivot || patch.is_rigid);
        if (pivot_like) {
            ++neighbors_.stats().num_pivot_accepts;
        }

        const bool legacy_force_dirty =
            pivot_like && neighbors_.config().invalidate_verlet_on_pivot_accept;
        if (legacy_force_dirty) {
            // NeighborSystem already invalidated; attribute dirty counter.
            ++neighbors_.stats().num_pivot_accepts_dirty_verlet;
        } else if (!neighbors_.muVerlet().dirty) {
            // SC/KIC accumulate may already have run inside commit (partial path).
            // Pivot still accumulates here.
            if (pivot_like) {
                neighbors_.muVerlet().accumulate_accept(
                    patch.moved_indices, commit_old_coords_scratch_, state.coords_soa);
                if (neighbors_.muVerlet().dirty) {
                    neighbors_.muVerlet().dirty_cause =
                        VerletList::DirtyCause::PivotAccept;
                    ++neighbors_.stats().num_pivot_accepts_dirty_verlet;
                } else {
                    ++neighbors_.stats().num_pivot_accepts_keep_verlet_valid;
                }
            } else if (!neighbors_.last_commit_did_partial_verlet()) {
                // Non-pivot: commit already accumulated when list was clean.
                // If commit skipped (e.g. already dirty), nothing to do.
            }
        }

        // Lazy Verlet rebuild: Integrator::run rebuilds when a KIC/SC step
        // actually needs VerletPreferred. Eager rebuild here made commit dominate
        // wall time (full CSR after nearly every large pivot accept).
        // if (neighbors_.muVerlet().dirty) maybe_rebuild...  — intentionally omitted
    }
}


// --- PHYSICS EVALUATION ---
float Context::calculate_total_energy(int target_group) {
    require_current_atom_order();
    // Asks the System to loop through all its Potentials and calculate baseline energy
    // (legacy-weighted by default via energy_weights_).
    // The whole energy is what current_energy gets reset to, so it is a
    // resync: terms rebuild their incremental bookkeeping in the same pass.
    const TotalEnergyResult result = system->evaluateTotalEnergy(
        *this, state, target_group, /*resync=*/target_group == -1);
    const float e = result.energy;

    if (target_group == -1) {
        last_total_reject_reason_ = result.reject_reason;
        // Do NOT cache a rejection sentinel as if it were a physical energy.
        // System::evaluateTotalEnergy folds Mu's kHardCorePenalty (99999) into
        // the WEIGHTED sum, so on rejection `e` is ~weight*99999 (~+39750 with
        // the legacy Mu weight) -- a number that no longer looks like a sentinel
        // but is not an energy either. current_energy is read straight into
        // Metropolis criteria by the REMD exchange
        // (mpi_replica_exchange.py e_i_total/e_j_total), which silently accepted
        // that value; on p19.14.3 the top ladder rung carried it for 41% of
        // cycles. Keep the last good value here and let callers branch on
        // has_steric_clash() -- which they now do, by raising.
        if (result.reject_reason == RejectReason::None) {
            state.current_energy = e; // Cache total energy for later reference
        }
    }
    return e;
}

float Context::calculate_total_energy_raw(int target_group) const {
    require_current_atom_order();
    return system->getTotalEnergyRaw(*this, state, target_group);
}

float Context::calculate_delta_energy(const State& proposed_state, const ProposalPatch& patch) const {
    // Asks the System to loop through all Potentials and calculate the CHANGE in energy.
    // Because this method is CONST, it guarantees the master grids are not touched!
    return system->getDeltaEnergy(*this, state, proposed_state, patch);
}

EnergyBreakdown Context::energy_breakdown() const {
    require_current_atom_order();
    return system->energyBreakdown(*this, state);
}

forces::mcpu08::MuPotential* Context::mu_potential() {
    for (auto& f : system->getPotentials()) {
        if (auto* mu = dynamic_cast<forces::mcpu08::MuPotential*>(f.get())) {
            return mu;
        }
    }
    return nullptr;
}

const forces::mcpu08::MuPotential* Context::mu_potential() const {
    for (const auto& f : system->getPotentials()) {
        if (const auto* mu = dynamic_cast<const forces::mcpu08::MuPotential*>(f.get())) {
            return mu;
        }
    }
    return nullptr;
}

} // namespace mcpu
