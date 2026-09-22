#pragma once

#include <cstdint>

namespace mcpu {

/// Reason for hard rejection of a proposed MC move / total-energy state.
/// Used by Potential::calculateEnergyChange and System::evaluateTotalEnergy.
enum class RejectReason : std::uint8_t {
    None = 0,        ///< no hard rejection
    StericClash = 1, ///< hard-core overlap / clash sentinel
};

/// Result of an incremental energy-change calculation.
/// Carries both ΔE and an optional hard-rejection reason for Metropolis.
struct EnergyChangeResult {
    float delta_energy = 0.f;
    RejectReason reject_reason = RejectReason::None;

    [[nodiscard]] bool hard_reject() const noexcept {
        return reject_reason != RejectReason::None;
    }

    /// No hard rejection — Metropolis uses ``delta_energy``.
    [[nodiscard]] static EnergyChangeResult finite(float de) noexcept {
        return EnergyChangeResult{de, RejectReason::None};
    }

    /// Hard rejection — Integrator consumes one RNG draw and rejects.
    [[nodiscard]] static EnergyChangeResult rejected(
        float de, RejectReason reason) noexcept
    {
        return EnergyChangeResult{de, reason};
    }
};

/// Result of a full (non-incremental) total-energy evaluation.
struct TotalEnergyResult {
    float energy = 0.f;
    RejectReason reject_reason = RejectReason::None;
};

} // namespace mcpu
