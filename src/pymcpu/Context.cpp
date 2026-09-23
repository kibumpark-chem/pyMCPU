#include "pymcpu/Context.h"
#include "pymcpu/forces/bias/QBiasPotential.h"
#include "pymcpu/forces/mcpu/mcpu08/MuPotential.h"
#include "pymcpu/AtomReorder.h"
#include "pymcpu/utils/geometry_utils.h"
#include "pymcpu/utils/sidechain_torsion_utils.h"
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
    neighbors_.init(*system);
}

void Context::set_atom_reorder_mode(AtomReorderMode mode) {
    atom_reorder_mode_ = mode;
    if (mode == AtomReorderMode::Off) {
        // Keep identity; do not undo an already-applied permutation in this PR.
        return;
    }
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

    std::vector<BlockIndices> new_blocks;
    const NeighborConfig& ncfg = neighbors_.config();
    const float mu_cell = ::mcpu::effective_mu_cell_size_A(
        neighbors_.mu_cutoff_A(), ncfg.skin, ncfg);
    atom_perm_ = compute_init_only_atom_permutation(
        *system, state.coords_soa, ncfg.skin, mu_cell, &new_blocks);
    if (atom_perm_.is_identity()) {
        reorder_applied_ = true;
        return;
    }

    permute_coords_soa(state.coords_soa, atom_perm_);
    system->apply_residue_contiguous_blocks(std::move(new_blocks), atom_perm_);
    for (auto& potential : system->getPotentials()) {
        potential->permute_atom_indices(atom_perm_);
    }
    // Contact cache is indexed by atom pairs in storage order — invalidate.
    state.is_contact_cache.clear();
    // Pair indices change meaning under a permutation -- drop the list too.
    state.mu_contact_invalidate();
    // NeighborSystem caches donor bb_starts from System — re-init + rebuild.
    neighbors_.init(*system, neighbors_.config());
    sync_geometry();
    computeTorsions();
    if (q_bias_k_ > 0.0f) {
        reinitialize_q_pair_cache();
    }
    reorder_applied_ = true;
}

Eigen::Matrix3Xf Context::coords_for_python() const {
    if (output_internal_order_ || atom_perm_.is_identity()) {
        return state.coords_as_eigen();
    }
    return scatter_internal_to_external(state.coords_soa, atom_perm_);
}

void Context::set_coords_from_python(const Eigen::Matrix3Xf& coords_external) {
    if (output_internal_order_ || atom_perm_.is_identity()) {
        setPositions(coords_external);
        return;
    }
    gather_external_to_internal(coords_external, atom_perm_, state.coords_soa);
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
void Context::setPositions(const Eigen::Matrix3Xf& new_coords) {
    // Coordinates are being replaced wholesale (load, REMD swap, restart), so
    // the live contact list describes a conformation that no longer exists.
    state.mu_contact_invalidate();
    // Incoming coords are always treated as *external* (build) order.
    if (!reorder_applied_ || atom_perm_.is_identity()) {
        state.coords_soa.load_from_eigen(new_coords);
    } else {
        gather_external_to_internal(new_coords, atom_perm_, state.coords_soa);
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
    for (const auto& update : mu_ws.pending_updates) {
        state.is_contact_cache[update.index] = static_cast<uint8_t>(update.new_value);
    }

    auto& q_ws = q_bias_workspace_;
    for (const auto& update : q_ws.pending_updates) {
        state.q_pair_cache[static_cast<size_t>(update.index)] =
            static_cast<uint8_t>(update.new_value);
    }

    state.current_energy = proposed_state.getEnergy();

    // Fold this accepted move into the live Mu contact list. The pending
    // entries carry their own energies, so this needs nothing from the potential.
    if (state.mu_contact_list_ready) {
        for (const auto& p : mu_ws.pending_contact_drop)
            state.mu_contact_remove(p.i, p.j);
        for (const auto& p : mu_ws.pending_contact_add)
            state.mu_contact_add(p.i, p.j, p.energy);
    }
    mu_ws.pending_contact_drop.clear();
    mu_ws.pending_contact_add.clear();

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
    {
        for (size_t atom_idx = 0; atom_idx < patch.moving_atoms.size(); ++atom_idx) {
            if (patch.moving_atoms[atom_idx]) {
                state.coords_soa.copy_atom_from(
                    proposed_state.coords_soa,
                    static_cast<int>(atom_idx),
                    static_cast<int>(atom_idx));
            }
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

        if (neighbors_.config().box_policy == BoxPolicy::AutoRecenter) {
            state.coords_soa.recenter();
        }
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
    // Asks the System to loop through all its Potentials and calculate baseline energy
    // (legacy-weighted by default via energy_weights_).
    const TotalEnergyResult result =
        system->evaluateTotalEnergy(*this, state, target_group);
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
    return system->getTotalEnergyRaw(*this, state, target_group);
}

float Context::calculate_delta_energy(const State& proposed_state, const ProposalPatch& patch) const {
    // Asks the System to loop through all Potentials and calculate the CHANGE in energy.
    // Because this method is CONST, it guarantees the master grids are not touched!
    return system->getDeltaEnergy(*this, state, proposed_state, patch);
}

EnergyBreakdown Context::energy_breakdown() const {
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

void Context::set_mm_clash_margin(float margin_r2) {
#if MCPU_FAST_MU_DELTA
    for (auto& f : system->getPotentials()) {
        if (auto* mu = dynamic_cast<forces::mcpu08::MuPotential*>(f.get())) {
            mu->set_mm_clash_margin(margin_r2);
        }
    }
#else
    (void)margin_r2;
#endif
}

float Context::mm_clash_margin() const {
#if MCPU_FAST_MU_DELTA
    for (const auto& f : system->getPotentials()) {
        if (const auto* mu = dynamic_cast<const forces::mcpu08::MuPotential*>(f.get())) {
            return mu->mm_clash_margin();
        }
    }
#endif
    return 0.f;
}

void Context::set_mm_double_boundary(bool on) {
#if MCPU_FAST_MU_DELTA
    for (auto& f : system->getPotentials()) {
        if (auto* mu = dynamic_cast<forces::mcpu08::MuPotential*>(f.get())) {
            mu->set_mm_double_boundary(on);
        }
    }
#else
    (void)on;
#endif
}

bool Context::mm_double_boundary() const {
#if MCPU_FAST_MU_DELTA
    for (const auto& f : system->getPotentials()) {
        if (const auto* mu = dynamic_cast<const forces::mcpu08::MuPotential*>(f.get())) {
            if (mu->mm_double_boundary()) return true;
        }
    }
#endif
    return false;
}

} // namespace mcpu
