#pragma once
#include <array>
#include <cmath>
#include <vector>
#include "pymcpu/moves/RotamerLibrary.h"  // reuse wrap_angle_to_pi

namespace mcpu {

/// One wrapped-bivariate-normal component of a per-residue-category (phi,
/// psi) mixture. `cov` = {c11, c12, c22} is the RAW (already-regularized by
/// the offline Python fitter, which is not part of this distribution) 2x2
/// covariance in radians^2. The Cholesky factor
/// (sampling) and inverse+log-det (density) are precomputed ONCE at
/// add_residue_type() time from this SAME stored `cov` -- never
/// recomputed on the MH hot path, and load-bearing that both derive from
/// the identical covariance: sample() and log_pdf() must correspond to
/// exactly the same distribution, or the Metropolis-Hastings correction is
/// inconsistent with what was actually proposed.
struct RamaComponent {
    float log_weight = 0.0f;
    std::array<float, 2> mean{0.0f, 0.0f};   // (phi, psi), radians, wrapped to (-pi, pi]
    std::array<float, 3> cov{1.0f, 0.0f, 1.0f};  // c11, c12, c22

    // Precomputed at add_residue_type() time:
    float chol_l11 = 1.0f, chol_l21 = 0.0f, chol_l22 = 1.0f;  // Cholesky, for sampling
    float inv_c11 = 1.0f, inv_c12 = 0.0f, inv_c22 = 1.0f;      // cov^-1, for density
    float log_det = 0.0f;                                      // log|cov|, for density
};

/// Per-residue-category table of (phi, psi) mixture rows plus the
/// wrapped-Gaussian-mixture proposal density evaluator needed for the
/// knowledge-based backbone pivot move's Metropolis-Hastings correction
/// (see MCIntegrator::apply_rama_pivot_at).
///
/// Indexed by an opaque integer category key. Populated today with the
/// legacy pdb_util.h GetAminoNumber() alphabetical amino-acid index (0-19,
/// see pymcpu.forcefields.builders.hbond_builder.AMINO_INDEX) -- the same
/// indexing System::amino_index()/RotamerLibrary use -- but add_residue_type
/// auto-grows internal storage for any key >= its current size, so a future
/// reserved category (e.g. a "pre-proline" row, conditioned on the NEXT
/// residue being PRO rather than on this residue's own identity) is a
/// one-line addition at the call site that resolves which category index to
/// look up (see apply_rama_pivot_at's resolve_rama_category), not a change
/// to this class.
///
/// Density uses a proper (2*n_wrap+1)^2-cell periodic sum in log-space
/// (logsumexp), NOT RotamerLibrary's single-nearest-image shortcut --
/// RotamerLibrary's shortcut is justified only because chi sigmas are "at
/// most a few tens of degrees" (see RotamerLibrary.h); this mixture's fitted
/// sigmas run up to ~1.3 rad for some low-weight background components, far
/// outside that regime. `n_wrap` must match whatever the fit used
/// (WrappedNormalMixture's n_wrap field, wrapped_normal_mixture.py) -- do
/// not hardcode a fixed image count at call sites, read n_wrap().
class RamaMixtureLibrary {
public:
    static constexpr int kDefaultNumCategories = 20;

    explicit RamaMixtureLibrary(int n_wrap = 1)
        : n_wrap_(n_wrap),
          components_per_amino_(kDefaultNumCategories),
          cumulative_weight_per_amino_(kDefaultNumCategories) {
        const int span = 2 * n_wrap_ + 1;
        shifts_.reserve(static_cast<size_t>(span) * static_cast<size_t>(span));
        for (int dx = -n_wrap_; dx <= n_wrap_; ++dx) {
            for (int dy = -n_wrap_; dy <= n_wrap_; ++dy) {
                shifts_.push_back({static_cast<float>(dx) * 2.0f * mcpu::PI_F,
                                    static_cast<float>(dy) * 2.0f * mcpu::PI_F});
            }
        }
    }

    /// Registers category `amino_idx`'s (phi, psi) mixture. `weights` must be
    /// non-negative (the Python builder renormalizes to sum to 1 before
    /// calling this); `means` in radians; `covariances[k] = {c11, c12, c22}`
    /// already includes whatever regularization the fitter applied (e.g.
    /// reg_covar * I) -- not re-jittered here, but validated to be symmetric
    /// positive-definite (throws otherwise, converting a silent
    /// NaN/garbage sampler into a loud, debuggable load-time failure).
    /// Auto-grows internal storage if `amino_idx` is beyond the default
    /// 20-category span.
    void add_residue_type(int amino_idx,
                           const std::vector<float>& weights,
                           const std::vector<std::array<float, 2>>& means,
                           const std::vector<std::array<float, 3>>& covariances);

    /// Number of mixture rows for `amino_idx`, or 0 if never registered.
    [[nodiscard]] int num_rows(int amino_idx) const;

    /// Draws a mixture-row index for `amino_idx` from a caller-supplied
    /// U(0,1) value `u01` via a cumulative-weight walk (exactly one RNG draw
    /// at the call site). Clamps to the last row on float-rounding
    /// fallthrough. Returns -1 if `amino_idx` has no rows.
    [[nodiscard]] int sample_row(int amino_idx, float u01) const;

    /// Read-only access to one mixture row (for building the bivariate
    /// Gaussian proposal at the call site). Throws std::out_of_range if
    /// invalid.
    [[nodiscard]] const RamaComponent& row(int amino_idx, int row_idx) const;

    /// log q(phi, psi): a K-component wrapped-bivariate-Gaussian mixture
    /// over `amino_idx`'s rows, using a proper (2*n_wrap+1)^2-cell periodic
    /// sum (not a nearest-image shortcut -- see class docs) and a
    /// numerically stable log-sum-exp. Returns -inf if `amino_idx` has no
    /// rows. `phi_psi` need not be pre-wrapped to (-pi, pi] -- each periodic
    /// image shift is applied internally.
    [[nodiscard]] float log_mixture_density(int amino_idx,
                                             const std::array<float, 2>& phi_psi) const;

    [[nodiscard]] int n_wrap() const noexcept { return n_wrap_; }

private:
    int n_wrap_;
    std::vector<std::array<float, 2>> shifts_;  // (2*n_wrap+1)^2 entries, each a multiple of 2*pi
    std::vector<std::vector<RamaComponent>> components_per_amino_;
    /// Running cumulative sum of `weights` per category, for sample_row().
    std::vector<std::vector<float>> cumulative_weight_per_amino_;

    void ensure_capacity(int amino_idx);
};

} // namespace mcpu
