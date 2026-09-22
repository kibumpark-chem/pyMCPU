#pragma once
#include "pymcpu/State.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/AtomPermutation.h"
#include "pymcpu/EnergyChangeResult.h"

namespace mcpu {
class Context;

class Potential {
protected:
    int energy_group = 0;
    bool enabled_ = true;

public:
    virtual ~Potential() = default;

    void setEnergyGroup(int group) { energy_group = group; }
    int getEnergyGroup() const { return energy_group; }

    /// When false, System skips this potential in total/delta energy (bench/audit only).
    void setEnabled(bool on) noexcept { enabled_ = on; }
    bool isEnabled() const noexcept { return enabled_; }

    /// Optional: remap any stored atom indices after init-only locality permutation.
    virtual void permute_atom_indices(const AtomPermutation& /*perm*/) {}

    virtual float calculateEnergy(
        const Context& context,
        const State& state
    ) const = 0;

    /// CHANGED: return EnergyChangeResult to carry hard-rejection reason.
    virtual EnergyChangeResult calculateEnergyChange(
        const Context& context,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& patch
    ) const = 0;

    /// Override when this potential can hard-reject via a clash energy sentinel.
    virtual bool canHardReject() const noexcept { return false; }

    /// Map a total-energy sentinel to a RejectReason (used by evaluateTotalEnergy).
    virtual RejectReason rejectionForEnergy(float /*energy*/) const noexcept {
        return RejectReason::None;
    }
};
} // namespace mcpu
