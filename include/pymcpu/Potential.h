#pragma once
#include <stdexcept>
#include <string>
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
    std::string name_;

public:
    virtual ~Potential() = default;

    void setEnergyGroup(int group) { energy_group = group; }
    int getEnergyGroup() const { return energy_group; }

    /// The term's name, e.g. "mu" or "korp_6d", set by the force field that
    /// creates it. It labels the term in energy_breakdown()["by_name"], the
    /// energy CSV header and step_stats(). Empty means unnamed, reported as
    /// "group_<n>".
    void setName(const std::string& name) {
        validateName(name);
        name_ = name;
    }
    const std::string& getName() const noexcept { return name_; }

    /// Names must be CSV-safe ([a-z][a-z0-9_]*) and must not collide with the
    /// energy CSV's own columns.
    static void validateName(const std::string& name) {
        if (name.empty()) return;
        bool ok = name[0] >= 'a' && name[0] <= 'z';
        for (char c : name) {
            ok = ok && ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_');
        }
        if (!ok) {
            throw std::invalid_argument(
                "energy term name must match [a-z][a-z0-9_]*, got '" + name + "'");
        }
        auto ends_with = [&](const char* suffix) {
            const std::string s(suffix);
            return name.size() >= s.size() &&
                   name.compare(name.size() - s.size(), s.size(), s) == 0;
        };
        bool reserved = name == "step" || name == "total" || name == "walker_id" ||
                        ends_with("_accepted") || ends_with("_attempted");
        if (name.rfind("group_", 0) == 0 && name.size() > 6 &&
            name.find_first_not_of("0123456789", 6) == std::string::npos) {
            reserved = true;
        }
        if (reserved) {
            throw std::invalid_argument(
                "energy term name '" + name + "' is reserved for the energy CSV's "
                "own columns or for unnamed terms");
        }
    }

    /// When false, System skips this potential in total/delta energy (bench/audit only).
    void setEnabled(bool on) noexcept { enabled_ = on; }
    bool isEnabled() const noexcept { return enabled_; }

    /// Remap every stored atom id from build order to storage order after the
    /// init_only locality permutation. Pure virtual so a new term cannot skip
    /// it silently; a term that stores no atom ids overrides it with an empty
    /// body and says so.
    virtual void permute_atom_indices(const AtomPermutation& perm) = 0;

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
