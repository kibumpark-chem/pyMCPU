#pragma once
#include <array>
#include <cmath>
#include <vector>
#include "pymcpu/utils/numbers_compat.h"

namespace mcpu {

/// Wraps an angle (radians) into (-PI, PI]. Used throughout the rotamer-library
/// move since chi angles are periodic and both proposal deltas and the mixture
/// log-density must use the minimal (wrapped) angular difference.
inline float wrap_angle_to_pi(float a) noexcept {
    constexpr float kTwoPi = 2.0f * mcpu::PI_F;
    return a - kTwoPi * std::round(a / kTwoPi);
}

/// One Dunbrack-style backbone-independent rotamer "row" for a residue type:
/// a weight plus a per-chi (mean, sigma) pair, all in radians. `log_weight` is
/// precomputed once so the log-sum-exp hot path never re-calls log() on it.
struct RotamerComponent {
    float log_weight = 0.0f;
    std::array<float, 4> mean{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> sigma{1.0f, 1.0f, 1.0f, 1.0f};
};

/// Per-amino-acid-type table of rotamer rows (bbind02.May.lib, parsed
/// Python-side by RotamerLibraryBuilder) plus the Gaussian-mixture proposal
/// density evaluator needed for the rotamer-library sidechain move's
/// Metropolis-Hastings correction (see MCIntegrator::apply_rotamer_at).
///
/// Indexed by the legacy pdb_util.h GetAminoNumber() alphabetical amino-acid
/// index (0-19, see pymcpu.forcefields.builders.hbond_builder.AMINO_INDEX) --
/// the same indexing System::amino_index()/HBondPotential's seq-dep table use.
/// GLY/ALA (no chi angles) and PRO (ring geometry -- its library rows are
/// ring-pucker parameters, not chi rotamers) are never populated; the move
/// logic excludes them before ever querying this class.
class RotamerLibrary {
public:
    static constexpr int kNumAminoTypes = 20;

    RotamerLibrary()
        : components_per_amino_(kNumAminoTypes), cumulative_weight_per_amino_(kNumAminoTypes) {}

    /// Registers residue type `amino_idx`'s rotamer table. `weights` must be
    /// non-negative and is used as-is for cumulative-weight sampling (the
    /// Python builder renormalizes to sum to 1 before calling this); `means`/
    /// `sigmas` must already be in radians. All three vectors must have the
    /// same (non-zero) length K = number of rotamer rows for this residue.
    void add_residue_type(int amino_idx,
                           const std::vector<float>& weights,
                           const std::vector<std::array<float, 4>>& means,
                           const std::vector<std::array<float, 4>>& sigmas);

    /// Number of rotamer rows for `amino_idx`, or 0 if never registered.
    [[nodiscard]] int num_rows(int amino_idx) const;

    /// Draws a rotamer row index for `amino_idx` from a caller-supplied
    /// U(0,1) value `u01` via a cumulative-weight walk (exactly one RNG draw
    /// at the call site -- see apply_rotamer_at). Clamps to the last row on
    /// float-rounding fallthrough. Returns -1 if `amino_idx` has no rows.
    [[nodiscard]] int sample_row(int amino_idx, float u01) const;

    /// Read-only access to one rotamer row (for building the per-chi Gaussian
    /// proposal at the call site). Throws std::out_of_range if invalid.
    [[nodiscard]] const RotamerComponent& row(int amino_idx, int row_idx) const;

    /// log q(chi) for the independence-sampler proposal density: a
    /// K-component Gaussian mixture over `amino_idx`'s rotamer rows,
    /// restricted to the first `ntorsions` chi angles (unused chi slots are
    /// never read), using wrapped angular differences and a numerically
    /// stable log-sum-exp. Returns -inf if `amino_idx` has no rows.
    ///
    /// The single-nearest-wrap approximation (rather than a full periodic /
    /// von-Mises sum over all 2*pi*k image copies) is deliberate: this
    /// library's sigmas are at most a few tens of degrees, so the next-
    /// nearest image is tens of standard deviations away and contributes a
    /// relative density on the order of exp(-2*pi^2/sigma^2) or smaller --
    /// utterly negligible. Do not "fix" this into an infinite sum.
    [[nodiscard]] float log_mixture_density(int amino_idx, int ntorsions,
                                             const std::array<float, 4>& chi) const;

private:
    std::vector<std::vector<RotamerComponent>> components_per_amino_;
    /// Running cumulative sum of `weights` per amino type, for sample_row().
    std::vector<std::vector<float>> cumulative_weight_per_amino_;
};

} // namespace mcpu
