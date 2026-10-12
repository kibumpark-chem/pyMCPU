#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>
#include "pymcpu/utils/numbers_compat.h"

namespace mcpu {

/// Wraps an angle (radians) into (-PI, PI]. Used throughout the rotamer-library
/// move since chi angles are periodic and both proposal deltas and the mixture
/// log-density must use the minimal (wrapped) angular difference.
///
/// The quotient is rounded half away from zero, as std::round does, but via
/// std::trunc plus a fix-up: GCC calls libm for std::round while it inlines
/// std::trunc as one roundss on SSE4.1. q - trunc(q) is exact, so the result
/// matches std::round bit for bit, including signed zeros, NaN and infinity.
inline float wrap_angle_to_pi(float a) noexcept {
    constexpr float kTwoPi = 2.0f * mcpu::PI_F;
    const float q = a / kTwoPi;
    float n = std::trunc(q);
    // Branch-free: adding a zero signed like q leaves n (and its sign) as is.
    n += std::copysign(std::fabs(q - n) >= 0.5f ? 1.0f : 0.0f, q);
    return a - kTwoPi * n;
}

/// One Dunbrack-style backbone-independent rotamer "row" for a residue type:
/// a weight plus a per-chi (mean, sigma) pair, all in radians. `log_weight` is
/// precomputed once so the log-sum-exp hot path never re-calls log() on it;
/// `log_sigma` likewise holds log(sigma[i]), filled by add_residue_type.
struct RotamerComponent {
    float log_weight = 0.0f;
    std::array<float, 4> mean{0.0f, 0.0f, 0.0f, 0.0f};
    std::array<float, 4> sigma{1.0f, 1.0f, 1.0f, 1.0f};
    std::array<float, 4> log_sigma{0.0f, 0.0f, 0.0f, 0.0f};
};

/// Per-amino-acid-type rotamer tables for the rotamer-library sidechain move,
/// plus the proposal density its Metropolis-Hastings correction needs (see
/// MCIntegrator::apply_rotamer_at). Built Python-side from Dunbrack's 2002
/// backbone-independent library (bbind02.May.lib) by
/// pymcpu.forcefields.builders.rotamer_builder. Holds one of two kinds of
/// table:
///
/// * Truncated tables (add_truncated_table), "dunbrack2002_sym", the default:
///   each chi is drawn from a normal truncated to one period window around
///   the row mean, and a chi with period pi (the two-fold symmetric ASP chi2,
///   GLU chi3, PHE chi2, TYR chi2) then gets a random 180 degree image, so
///   both of its halves are proposed. sample() and log_density() draw and
///   evaluate exactly the same density (see add_truncated_table).
/// * Row tables (add_residue_type), "dunbrack2002": the rows as legacy MCPU
///   draws them (an untruncated normal per chi, the symmetric chi only on the
///   half the rows cover), corrected by the Gaussian mixture density at the
///   nearest image. sample_row, row and log_mixture_density serve this kind
///   only, and the move draws and evaluates it exactly as before
///   dunbrack2002_sym existed.
///
/// Indexed by the legacy pdb_util.h GetAminoNumber() alphabetical amino-acid
/// index (0-19, see pymcpu.forcefields.builders.hbond_builder.AMINO_INDEX) --
/// the same indexing System::amino_index()/HBondPotential's seq-dep table use.
/// GLY/ALA (no chi angles) and PRO (ring geometry -- its library rows are
/// ring-pucker parameters, not chi rotamers) are never moved; the move logic
/// excludes them before ever querying this class.
class RotamerLibrary {
public:
    static constexpr int kNumAminoTypes = 20;
    /// Largest number of rows per residue type in a truncated table.
    static constexpr int kMaxTableRows = 256;

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

    /// Number of rotamer rows for `amino_idx` (either kind of table), or 0 if
    /// never registered.
    [[nodiscard]] int num_rows(int amino_idx) const;

    /// Draws a rotamer row index for `amino_idx` from a caller-supplied
    /// U(0,1) value `u01` via a cumulative-weight walk (exactly one RNG draw
    /// at the call site -- see apply_rotamer_at). Clamps to the last row on
    /// float-rounding fallthrough. Returns -1 if `amino_idx` has no rows.
    /// Row tables only.
    [[nodiscard]] int sample_row(int amino_idx, float u01) const;

    /// Read-only access to one rotamer row (for building the per-chi Gaussian
    /// proposal at the call site). Throws std::out_of_range if invalid.
    /// Row tables only.
    [[nodiscard]] const RotamerComponent& row(int amino_idx, int row_idx) const;

    /// log q(chi) for the independence-sampler proposal density: a
    /// K-component Gaussian mixture over `amino_idx`'s rotamer rows,
    /// restricted to the first `ntorsions` chi angles (unused chi slots are
    /// never read), using wrapped angular differences and a numerically
    /// stable log-sum-exp. Returns -inf if `amino_idx` has no rows.
    /// Row tables only.
    ///
    /// The single-nearest-wrap approximation (rather than a full periodic /
    /// von-Mises sum over all 2*pi*k image copies) is deliberate: this
    /// library's sigmas are at most a few tens of degrees, so the next-
    /// nearest image is tens of standard deviations away and contributes a
    /// relative density on the order of exp(-2*pi^2/sigma^2) or smaller --
    /// utterly negligible. Do not "fix" this into an infinite sum.
    [[nodiscard]] float log_mixture_density(int amino_idx, int ntorsions,
                                             const std::array<float, 4>& chi) const;

    /// Registers residue type `amino_idx`'s truncated table: K = weights.size()
    /// rows (1 to kMaxTableRows) and n = periods.size() chi (1-4). `weights`
    /// are positive (normalised here), `means` and `sigmas` are [row][chi] in
    /// radians, and `periods` is 2 pi or pi per chi.
    ///
    /// Rows are stored by weight, largest first. The row is drawn by comparing
    /// one raw 32-bit rng() value with integer thresholds
    /// T_k = round(2^32 cum_k), and the density uses the probabilities
    /// (T_k - T_k-1) / 2^32 of exactly that draw; the normals' means, widths
    /// and truncation windows are likewise the float values sample() uses, and
    /// each normal carries its mass inside the window as its normaliser. So
    /// log_density is the density of sample() up to the rounding of its own
    /// evaluation.
    void add_truncated_table(int amino_idx, const std::vector<double>& weights,
                             const std::vector<double>& means,
                             const std::vector<double>& sigmas,
                             const std::vector<double>& periods);

    /// True once a truncated table has been added (the library then holds no
    /// row tables).
    [[nodiscard]] bool has_truncated_tables() const noexcept { return has_truncated_tables_; }

    /// log q(chi) of a truncated table: the density sample() draws from, on
    /// the first n chi of `chi`.
    [[nodiscard]] float log_density(int amino_idx, const std::array<float, 4>& chi) const;

    /// Draws chi from a truncated table into the first n slots of `chi` (each
    /// wrapped into (-pi, pi]) and returns the row drawn. RNG use: one rng()
    /// for the row, one or more unit_normal draws per chi (a draw outside the
    /// row's window is redrawn) and one rng() per chi of period pi for its
    /// image.
    int sample(int amino_idx, std::mt19937& rng, std::normal_distribution<float>& unit_normal,
               std::array<float, 4>& chi) const;

    /// One stored row of a truncated table as sample() uses it: its draw
    /// probability, and per chi the mean and width (1 / the stored inverse
    /// width).
    struct TableRow {
        double weight = 0.0;
        std::array<float, 4> mean{0.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, 4> sigma{0.0f, 0.0f, 0.0f, 0.0f};
    };
    [[nodiscard]] TableRow table_row(int amino_idx, int k) const;

    /// What the library is, for checkpoints: set by the Python builder (name,
    /// data checksum and builder version); empty for a hand-built library.
    [[nodiscard]] const std::string& fingerprint() const noexcept { return fingerprint_; }
    void set_fingerprint(std::string fingerprint) { fingerprint_ = std::move(fingerprint); }

private:
    /// One residue type's truncated table, structure of arrays over its K
    /// rows: `block` holds c[K] (the row constant: log of the draw probability
    /// minus the normalisers of its truncated normals), then mu_i[K] and
    /// inv_sigma_i[K] per chi i < n_chi; `thresholds` holds K - 1 row
    /// thresholds (the last row takes what is left).
    struct TruncatedTable {
        int n_rows = 0;
        int n_chi = 0;
        std::array<float, 4> period{};
        std::array<float, 4> inv_period{};
        std::array<float, 4> half_window{};  // half the period: the truncation window
        std::array<bool, 4> symmetric{};     // period pi: random 180 degree image
        std::vector<float> block;
        std::vector<std::uint32_t> thresholds;
    };
    const TruncatedTable& truncated_table(int amino_idx) const;

    std::vector<std::vector<RotamerComponent>> components_per_amino_;
    /// Running cumulative sum of `weights` per amino type, for sample_row().
    std::vector<std::vector<float>> cumulative_weight_per_amino_;
    std::array<TruncatedTable, kNumAminoTypes> truncated_tables_{};
    bool has_truncated_tables_ = false;
    std::string fingerprint_;
};

} // namespace mcpu
