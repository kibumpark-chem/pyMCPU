#include "pymcpu/testing/PhysicsVerifier.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "pymcpu/Integrator.h"

namespace mcpu {

namespace {

bool is_clash_energy(double energy) {
    return energy >= PhysicsVerifier::kHardCorePenalty * 0.5f;
}

void clear_workspaces(Context& ctx) {
    ctx.getWorkspace().clear();
    ctx.getQBiasWorkspace().clear();
    ctx.getHBondWorkspace().clear();
}

}  // namespace

PotentialDeltaCheck PhysicsVerifier::verify_potential_delta(
    const Context& ctx,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch,
    int energy_group,
    float atol)
{
    PotentialDeltaCheck result;
    result.energy_group = energy_group;

    if (!patch.is_valid) {
        result.passed = true;
        result.message = "skipped: invalid patch";
        return result;
    }

    const Potential* target = nullptr;
    for (const auto& potential : ctx.getSystem().getPotentials()) {
        if (potential->getEnergyGroup() == energy_group) {
            target = potential.get();
            break;
        }
    }

    if (target == nullptr) {
        result.passed = true;
        result.message = "skipped: energy group not present";
        return result;
    }

    State old_copy = old_state;
    State proposed_copy = proposed_state;


    auto& mutable_ctx = const_cast<Context&>(ctx);
    clear_workspaces(mutable_ctx);
    result.delta_incremental =
        target->calculateEnergyChange(ctx, old_copy, proposed_copy, patch)
            .delta_energy;

    clear_workspaces(mutable_ctx);
    const double e_old = target->calculateEnergy(ctx, old_copy);

    if (energy_group == 6) {
        proposed_copy.q_pair_cache.clear();
    }

    const double e_new = target->calculateEnergy(ctx, proposed_copy);
    result.delta_direct = e_new - e_old;

    // Only a term that actually hard-rejects can emit a clash sentinel. Other
    // groups (QBias group 6, say) can carry large but finite harmonic energies
    // that must not be mistaken for hard-core penalties.
    //
    // This asks the potential rather than testing `energy_group == 1`, which is
    // what it used to do. That was correct only while MuPotential was the sole
    // hard-rejecting term, and it silently failed any other one: a clash-guard
    // term in a second force field returns its sentinel from the incremental
    // path, the direct path returns the same sentinel, and the group test sent
    // both down the plain numeric-comparison branch that a sentinel cannot pass.
    if (target->canHardReject()) {
        const bool incremental_clash = is_clash_energy(result.delta_incremental);
        const bool proposed_clash = is_clash_energy(e_new);

        if (incremental_clash) {
            // Mu returns a fixed clash sentinel for delta_E, not E_new - E_old.
            // Under ClashOnly energy masking, full energy stays contact-only
            // (finite) while delta still StericClash-rejects — that is OK.
            const bool clash_only_contact =
                ctx.getSystem().has_energy_mask() &&
                ctx.getSystem().energy_mask_mode() == EnergyMaskMode::ClashOnly &&
                std::isfinite(e_new) && !proposed_clash;
            // A move is tested against a cutoff kStateClashBufferA tighter
            // than the one calculateEnergy judges a state by, so it can be
            // rejected for a pair the proposed state would still allow.
            const bool move_cutoff_only =
                !proposed_clash && !clash_only_contact &&
                target->clashesAtMoveCutoff(ctx, proposed_copy, patch);
            result.passed = proposed_clash || clash_only_contact || move_cutoff_only;
            result.message = proposed_clash
                ? "clash: incremental sentinel matches proposed clash energy"
                : (clash_only_contact
                       ? "clash: incremental sentinel with ClashOnly contact-only energy"
                       : (move_cutoff_only
                              ? "clash: incremental sentinel; a re-evaluated pair is "
                                "under the move cutoff but not the state cutoff"
                              : "clash: incremental sentinel but proposed energy is finite"));
            return result;
        }

        if (proposed_clash) {
            result.passed = false;
            result.message =
                "clash: proposed energy is clash but incremental delta is finite";
            return result;
        }
    }

    result.passed =
        std::abs(result.delta_incremental - result.delta_direct) <= atol;
    if (!result.passed) {
        std::ostringstream oss;
        oss << "|delta_inc - delta_direct|="
            << std::abs(result.delta_incremental - result.delta_direct)
            << " > atol=" << atol;
        result.message = oss.str();
    } else {
        result.message = "ok";
    }
    return result;
}

std::vector<PotentialDeltaCheck> PhysicsVerifier::verify_all_potential_deltas(
    const Context& ctx,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch,
    float atol)
{
    std::vector<PotentialDeltaCheck> results;
    std::unordered_set<int> seen_groups;

    for (const auto& potential : ctx.getSystem().getPotentials()) {
        const int group = potential->getEnergyGroup();
        if (!seen_groups.insert(group).second) {
            continue;
        }
        results.push_back(
            verify_potential_delta(ctx, old_state, proposed_state, patch, group, atol));
    }
    return results;
}

void PhysicsVerifier::verify_mc_energy_consistency(
    MCIntegrator& integrator,
    Context& ctx,
    int num_steps,
    float atol)
{
    integrator.verify_physics_consistency(ctx, num_steps, atol);
}

}  // namespace mcpu
