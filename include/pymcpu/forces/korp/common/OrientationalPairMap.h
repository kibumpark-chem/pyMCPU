#pragma once
/// Engine-side view of a KORP 6D knowledge-based energy map.
///
/// The file is NOT parsed here. `pymcpu.forcefields.korp_map` parses and
/// validates it in Python, and hands the pieces over: the tiny tessellation
/// arrays are copied into this object, and the ~83M-float table is referenced
/// in place. Keeping one parser rather than two is deliberate -- the Python one
/// is checked against the reference `korpe` binary to <= 5e-9 relative, and a
/// second implementation of a format with no version field and no magic number
/// would be a second thing to keep right.
///
/// The table is normally a numpy memmap, so the pages are shared by the OS
/// across every rank on a node instead of each holding its own ~316 MiB. Its
/// lifetime belongs to the caller: the binding layer keeps the Python object
/// alive for as long as this map is.
///
/// Binning transcribed from `contact2bins` (chaconlab/Korp, korpe.cpp).

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "pymcpu/forces/korp/common/ResidueFrame.h"

namespace mcpu::forces {

/// Where a pair's six coordinates land in the table.
struct PairBins {
    int shell;
    int cell_a;
    int cell_b;
    int chi;
};

class OrientationalPairMap {
public:
    /// Sequence separations at or above this are all treated as non-bonding
    /// (`BONDING_THR` in korpe.h). `smapping` is indexed 0..9.
    static constexpr int kSeparationTableSize = 10;
    /// Frame model 10 interacts 20 residue types with one frame each.
    static constexpr int kNumTypes = 20;

    OrientationalPairMap(
        float cutoff, float min_r, int nslices,
        std::vector<float> br,                  // nr + 1
        std::vector<int> shell_ncells,          // nr
        std::vector<int> shell_nchi,            // nr
        std::vector<float> shell_dchi,          // nr
        std::vector<std::int64_t> shell_offset, // nr + 1, float offsets in a block
        std::vector<int> ring_offset,           // nr + 1, index into the ring arrays
        std::vector<float> ring_theta,          // sum(nring)
        std::vector<float> ring_dpsi,           // sum(nring)
        std::vector<int> ring_ncells,           // sum(nring)
        std::vector<int> ring_first_cell,       // sum(nring)
        std::vector<std::int8_t> smapping,      // 10
        std::vector<float> fmapping,            // 10
        const float* table, std::size_t table_size);

    [[nodiscard]] float cutoff() const noexcept { return cutoff_; }
    [[nodiscard]] float cutoff_sq() const noexcept { return cutoff_sq_; }
    [[nodiscard]] float min_r() const noexcept { return min_r_; }
    [[nodiscard]] int num_shells() const noexcept { return nr_; }
    [[nodiscard]] int num_slices() const noexcept { return nslices_; }

    /// Slice for a sequence separation, or -1 when the pair is excluded.
    /// Cross-chain callers pass 0, which upstream treats as non-bonding.
    [[nodiscard]] int slice_for_separation(int separation) const noexcept {
        if (separation >= kSeparationTableSize) separation = 0;
        if (separation < 0) separation = 0;
        return smapping_[static_cast<std::size_t>(separation)];
    }

    [[nodiscard]] float slice_weight(int slice) const noexcept {
        return fmapping_[static_cast<std::size_t>(slice)];
    }

    /// Bin the six coordinates.
    ///
    /// The caller MUST have checked `min_r < d < cutoff` already. Upstream's
    /// radial loop has no upper bound and walks off the end of the boundary
    /// array otherwise, and this port keeps the same shape rather than adding a
    /// branch to the hot path.
    [[nodiscard]] PairBins bins(const PairCoordinates& pc) const noexcept {
        int shell = 0;
        while (pc.d > br_[static_cast<std::size_t>(shell) + 1]) ++shell;

        PairBins out;
        out.shell = shell;
        out.cell_a = angular_cell(shell, pc.cos_theta_a, pc.psi_a);
        out.cell_b = angular_cell(shell, pc.cos_theta_b, pc.psi_b);

        const int nchi = shell_nchi_[static_cast<std::size_t>(shell)];
        int ic = static_cast<int>(pc.chi / shell_dchi_[static_cast<std::size_t>(shell)]);
        if (ic >= nchi) ic = nchi - 1;   // upstream: "rarely... but it does"
        if (ic < 0) ic = 0;
        out.chi = ic;
        return out;
    }

    /// Raw (unweighted) table entry.
    /// Index into the table of the entry for (slice, type_a, type_b, bins).
    [[nodiscard]] std::size_t entry_index(int slice, int type_a, int type_b,
                                          const PairBins& b) const noexcept
    {
        const std::size_t ncells =
            static_cast<std::size_t>(shell_ncells_[static_cast<std::size_t>(b.shell)]);
        const std::size_t nchi =
            static_cast<std::size_t>(shell_nchi_[static_cast<std::size_t>(b.shell)]);
        return block_base(slice, type_a, type_b)
            + static_cast<std::size_t>(shell_offset_[static_cast<std::size_t>(b.shell)])
            + (static_cast<std::size_t>(b.cell_a) * ncells
               + static_cast<std::size_t>(b.cell_b)) * nchi
            + static_cast<std::size_t>(b.chi);
    }

    [[nodiscard]] float entry(std::size_t index) const noexcept { return table_[index]; }

    /// Address of an entry, for a prefetch. The table is ~330 MB and mapped
    /// from the file, so nearly every lookup misses the TLB and the caches.
    [[nodiscard]] const float* entry_address(std::size_t index) const noexcept {
        return table_ + index;
    }

    [[nodiscard]] float lookup(int slice, int type_a, int type_b,
                               const PairBins& b) const noexcept
    {
        return table_[entry_index(slice, type_a, type_b, b)];
    }

private:
    [[nodiscard]] std::size_t block_base(int slice, int a, int b) const noexcept {
        return ((static_cast<std::size_t>(slice) * kNumTypes + static_cast<std::size_t>(a))
                * kNumTypes + static_cast<std::size_t>(b))
               * static_cast<std::size_t>(block_stride_);
    }

    [[nodiscard]] int angular_cell(int shell, double cos_theta, double psi) const noexcept {
        const std::size_t lo = static_cast<std::size_t>(ring_offset_[static_cast<std::size_t>(shell)]);
        const std::size_t hi = static_cast<std::size_t>(ring_offset_[static_cast<std::size_t>(shell) + 1]);
        const int nring = static_cast<int>(hi - lo);

        // `theta > ring_theta[i]`, rewritten with cosines: acos decreases, so
        // the inequality flips. Exactly the same ring, two fewer acos calls.
        int ring = 0;
        while (cos_theta < ring_cos_theta_[lo + static_cast<std::size_t>(ring)]
               && ring < nring - 1) ++ring;

        const std::size_t k = lo + static_cast<std::size_t>(ring);
        int ip = static_cast<int>(psi / ring_dpsi_[k]);
        if (ip >= ring_ncells_[k]) ip = ring_ncells_[k] - 1;
        if (ip < 0) ip = 0;
        return ring_first_cell_[k] + ip;
    }

    float cutoff_, cutoff_sq_, min_r_;
    int nr_, nslices_;
    std::int64_t block_stride_;

    std::vector<float> br_;
    std::vector<int> shell_ncells_, shell_nchi_;
    std::vector<float> shell_dchi_;
    std::vector<std::int64_t> shell_offset_;
    std::vector<int> ring_offset_;
    std::vector<float> ring_theta_, ring_dpsi_;
    /// cos(ring_theta_), precomputed once so the hot loop compares cosines.
    std::vector<double> ring_cos_theta_;
    std::vector<int> ring_ncells_, ring_first_cell_;
    std::vector<std::int8_t> smapping_;
    std::vector<float> fmapping_;

    const float* table_;
    std::size_t table_size_;
};

} // namespace mcpu::forces
