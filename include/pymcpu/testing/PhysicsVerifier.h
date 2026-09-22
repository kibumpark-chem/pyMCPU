#pragma once

#include <string>
#include <vector>

#include "pymcpu/Context.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/State.h"

namespace mcpu {

class MCIntegrator;

struct PotentialDeltaCheck {
    int energy_group = -1;
    float delta_incremental = 0.0f;
    float delta_direct = 0.0f;
    bool passed = false;
    std::string message;
};

class PhysicsVerifier {
public:
    static constexpr float kHardCorePenalty = 99999.0f;

    static PotentialDeltaCheck verify_potential_delta(
        const Context& ctx,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& patch,
        int energy_group,
        float atol = 1e-3f);

    static std::vector<PotentialDeltaCheck> verify_all_potential_deltas(
        const Context& ctx,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& patch,
        float atol = 1e-3f);

    static void verify_mc_energy_consistency(
        MCIntegrator& integrator,
        Context& ctx,
        int num_steps,
        float atol = 1e-3f);
};

}  // namespace mcpu
