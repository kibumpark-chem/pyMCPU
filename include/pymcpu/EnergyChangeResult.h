#pragma once

#include <cstdint>

namespace mcpu {

/// Reason for hard rejection of a proposed MC move / total-energy state.
/// Used by Potential::calculateEnergyChange and System::evaluateTotalEnergy.
enum class RejectReason : std::uint8_t {
    None = 0,        ///< no hard rejection
    StericClash = 1, ///< hard-core overlap / clash sentinel
};

/// Precision rule for the whole engine: accumulators, totals, beta and dE
/// are double; per-pair values, caches and coordinates are float. Every
/// energy sum (per-term deltas and full sums, the results below, System's
/// breakdown, State::current_energy, the Metropolis input) and every running
/// budget (State::mu_list_drift, the H-bond ledger drift) is double, so the
/// running total stays within ~1e-10 of a full recompute (3e-11 after 1e7
/// actin steps, with no growth). Per-pair terms are computed in float, and
/// a difference of two of them is taken in double before it is added.
///
/// Result of an incremental energy-change calculation.
/// Carries both ΔE and an optional hard-rejection reason for Metropolis.
struct EnergyChangeResult {
    double delta_energy = 0.0;
    RejectReason reject_reason = RejectReason::None;

    [[nodiscard]] bool hard_reject() const noexcept {
        return reject_reason != RejectReason::None;
    }

    /// No hard rejection — Metropolis uses ``delta_energy``.
    [[nodiscard]] static EnergyChangeResult finite(double de) noexcept {
        return EnergyChangeResult{de, RejectReason::None};
    }

    /// Hard rejection — Integrator consumes one RNG draw and rejects.
    [[nodiscard]] static EnergyChangeResult rejected(
        double de, RejectReason reason) noexcept
    {
        return EnergyChangeResult{de, reason};
    }
};

/// Result of a full (non-incremental) total-energy evaluation.
struct TotalEnergyResult {
    double energy = 0.0;
    RejectReason reject_reason = RejectReason::None;
};

} // namespace mcpu
