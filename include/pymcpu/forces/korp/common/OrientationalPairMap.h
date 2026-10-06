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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__SSE__)
#include <immintrin.h>
#endif

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
    [[nodiscard]] PairBins bins(const PairVectors& pv) const noexcept {
        int shell = 0;
        if (padded_) {
            // Count of inner boundaries below d: the same shell as the loop
            // below for increasing boundaries, without its unpredictable exit.
            for (int s = 1; s < kMaxShells; ++s) shell += pv.d > br_pad_[s];
        } else {
            while (pv.d > br_[static_cast<std::size_t>(shell) + 1]) ++shell;
        }
        PairBins out;
        out.shell = shell;
        out.cell_a = angular_cell(shell, pv.cos_theta_a, pv.psi_a_y, pv.psi_a_x);
        out.cell_b = angular_cell(shell, pv.cos_theta_b, pv.psi_b_y, pv.psi_b_x);
        const std::size_t us = static_cast<std::size_t>(shell);
        out.chi = angle_bin(pv.chi_y, pv.chi_x, shell_dchi_[us], shell_inv_dchi_[us],
                            shell_nchi_[us], &chi_edge_[static_cast<std::size_t>(chi_edge_offset_[us])]);
        return out;
    }

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

    /// Raw (unweighted) table entry.
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

    /// Direction of a bin edge: the edge at angle e (pi + atan2 convention)
    /// is the unit vector at atan2-angle e - pi.
    struct Edge { double c, s; };

    /// Bin of pi + atan2(y, x) in bins of `width`, clamped to [0, n - 1]:
    /// upstream's `(int)(angle / width)`, found without atan2.
    ///
    /// A rough float angle (octant reduction and a cubic in min/max, with an
    /// approximate reciprocal; error below 2.5e-3 rad) picks a bin, and the
    /// vector is then tested against that bin's two edges, in double, by the
    /// sign of a cross product, which moves it to the neighbouring bin when
    /// the rough guess was off. The guess is off only within 2.5e-3 rad of an
    /// edge, and the narrowest bin is 0.63 rad, so one step either way is
    /// always enough. The edge tests make the result exact up to the rounding
    /// of a cross product, a few ulp of the angle, where the atan2 form decided
    /// by its own rounding. The constructor guarantees bins of at most pi/2
    /// whenever n > 1, so each edge tested is within pi of the vector and the
    /// sign of the cross product says which side of the edge it is on.
    [[nodiscard]] static int angle_bin(double y, double x, float width, float inv_width,
                                       int n, const Edge* edge) noexcept {
        if (n == 1) return 0;
        if (y == 0.0 && x == 0.0) {
            // atan2(0, 0) = 0: the angle is exactly pi. Never seen in practice
            // (a zero projection), and the edge tests cannot place it.
            int i = static_cast<int>(M_PI / static_cast<double>(width));
            return i >= n ? n - 1 : (i < 0 ? 0 : i);
        }
        const float fy = static_cast<float>(y);
        const float fx = static_cast<float>(x);
        const float ay = std::fabs(fy);
        const float ax = std::fabs(fx);
        const float hi = ay > ax ? ay : ax;
        const float lo = ay > ax ? ax : ay;
#if defined(__SSE__)
        const float t = hi > 0.f ? lo * _mm_cvtss_f32(_mm_rcp_ss(_mm_set_ss(hi))) : 0.f;
#else
        const float t = hi > 0.f ? lo / hi : 0.f;
#endif
        float r = t * (0.78539816f + (1.f - t) * (0.2447f + 0.0663f * t));
        r = ay > ax ? 1.57079633f - r : r;
        r = fx < 0.f ? 3.14159265f - r : r;
        r = fy < 0.f ? -r : r;   // -0 counts as +0, as atan2's angle did
        int i = static_cast<int>((r + 3.14159265f) * inv_width);
        i = i >= n ? n - 1 : (i < 0 ? 0 : i);
        // below(k): the vector lies before edge k, i.e. angle < k * width.
        const bool down = (i > 0) & (edge[i].c * y - edge[i].s * x < 0.0);
        const bool up = (i < n - 1) & !(edge[i + 1].c * y - edge[i + 1].s * x < 0.0);
        return i - static_cast<int>(down) + static_cast<int>(up);
    }

    [[nodiscard]] __attribute__((always_inline)) int angular_cell(int shell, double cos_theta, double y,
                                   double x) const noexcept {
        const std::size_t lo = static_cast<std::size_t>(ring_offset_[static_cast<std::size_t>(shell)]);
        // `theta > ring_theta[i]`, rewritten with cosines: acos decreases, so
        // the inequality flips. Exactly the same ring, two fewer acos calls.
        int ring = 0;
        if (padded_) {
            // The rings' cosines decrease, so the loop below stops at the first
            // boundary not above cos_theta: counting the boundaries above it
            // gives the same ring. Unused slots hold -inf and never count.
            const double* rc = &ring_cos_pad_[static_cast<std::size_t>(shell) * kMaxRings];
            for (int i = 0; i < kMaxRings; ++i) ring += cos_theta < rc[i];
        } else {
            const int nring = ring_offset_[static_cast<std::size_t>(shell) + 1]
                              - ring_offset_[static_cast<std::size_t>(shell)];
            while (cos_theta < ring_cos_theta_[lo + static_cast<std::size_t>(ring)]
                   && ring < nring - 1) ++ring;
        }
        const std::size_t k = lo + static_cast<std::size_t>(ring);
        const int ip = angle_bin(y, x, ring_dpsi_[k], ring_inv_dpsi_[k], ring_ncells_[k],
                                 &psi_edge_[static_cast<std::size_t>(psi_edge_offset_[k])]);
        return ring_first_cell_[k] + ip;
    }

    /// Fixed-size, padded copies of the shell and ring boundaries, so the
    /// shell and ring are counted without data-dependent loop exits. Used
    /// when the map fits (`padded_`); otherwise the loops run as upstream's.
    static constexpr int kMaxShells = 16;
    static constexpr int kMaxRings = 8;
    bool padded_ = false;
    double br_pad_[kMaxShells];
    std::vector<double> ring_cos_pad_;   // nr * kMaxRings

    /// Bin-edge directions: ring k's psi edges start at psi_edge_offset_[k]
    /// (ring_ncells + 1 of them), shell s's chi edges at chi_edge_offset_[s].
    std::vector<Edge> psi_edge_, chi_edge_;
    std::vector<int> psi_edge_offset_, chi_edge_offset_;
    std::vector<float> ring_inv_dpsi_, shell_inv_dchi_;

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
