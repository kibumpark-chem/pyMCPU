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
/// The table is the numpy array load_korp_map returned: by default a private
/// copy on 2 MiB pages (the lookups are TLB bound on 4 KiB pages), or with
/// mmap=True a memmap whose pages the OS shares across every rank on a node.
/// Its lifetime belongs to the caller: the binding layer keeps the Python
/// object alive for as long as this map is.
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
#if defined(__AVX2__) && defined(__FMA__)
        if (padded_) return bins_avx2(pv);
#endif
        return bins_scalar(pv);
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
#if defined(__AVX2__) && defined(__FMA__)
    /// c*y - s*x, four lanes, rounded as edge_below rounds it.
    static __m256d edge_cross(__m256d c, __m256d s, __m256d y, __m256d x) noexcept {
#if defined(MCPU_FP_CONTRACT_OFF)
        return _mm256_sub_pd(_mm256_mul_pd(c, y), _mm256_mul_pd(s, x));
#else
        return _mm256_fmsub_pd(c, y, _mm256_mul_pd(s, x));
#endif
    }

    /// bins() for a padded map, with the shell and ring counts as vector
    /// compares and the three angle bins (psi_a, psi_b, chi) found together,
    /// one per lane, with no data-dependent branch. Same bins as bins_scalar
    /// in every build mode: the counts are the same comparisons, and the edge
    /// tests round c*y - s*x the same way edge_below does (fused by default,
    /// unfused under MCPU_FP_CONTRACT=off), so the rough guess only has to be
    /// within one bin, which it is (see angle_bin).
    [[nodiscard]] __attribute__((always_inline)) PairBins bins_avx2(
        const PairVectors& pv) const noexcept
    {
        const __m256d vy = _mm256_setr_pd(pv.psi_a_y, pv.psi_b_y, pv.chi_y, pv.chi_y);
        const __m256d vx = _mm256_setr_pd(pv.psi_a_x, pv.psi_b_x, pv.chi_x, pv.chi_x);
        const __m256d zero = _mm256_setzero_pd();
        // A zero (y, x) is angle_bin's special case; it never happens in
        // practice, and the scalar path handles it.
        if (__builtin_expect(_mm256_movemask_pd(_mm256_and_pd(
                _mm256_cmp_pd(vy, zero, _CMP_EQ_OQ),
                _mm256_cmp_pd(vx, zero, _CMP_EQ_OQ))) != 0, 0)) {
            return bins_scalar(pv);
        }
        const __m256d vd = _mm256_set1_pd(pv.d);
        unsigned above = 0;
        for (int q = 0; q < kMaxShells / 4; ++q) {
            above |= static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(
                         vd, _mm256_loadu_pd(br_pad_ + 4 * q), _CMP_GT_OQ))) << (4 * q);
        }
        const int shell = __builtin_popcount(above & ~1u);   // boundaries 1..15
        const std::size_t us = static_cast<std::size_t>(shell);
        const double* rc = &ring_cos_pad_[us * kMaxRings];
        const __m256d rc0 = _mm256_loadu_pd(rc);
        const __m256d rc1 = _mm256_loadu_pd(rc + 4);
        const auto ring_of = [&](double c) noexcept {
            const __m256d vc = _mm256_set1_pd(c);
            return __builtin_popcount(
                static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(vc, rc0, _CMP_LT_OQ)))
                | (static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(vc, rc1, _CMP_LT_OQ)))
                   << 4));
        };
        const std::size_t ka = static_cast<std::size_t>(ring_offset_[us] + ring_of(pv.cos_theta_a));
        const std::size_t kb = static_cast<std::size_t>(ring_offset_[us] + ring_of(pv.cos_theta_b));

        // Rough angles, as angle_bin computes them, four lanes at once.
        const __m128 fy = _mm256_cvtpd_ps(vy);
        const __m128 fx = _mm256_cvtpd_ps(vx);
        const __m128 sign = _mm_set1_ps(-0.f);
        const __m128 ay = _mm_andnot_ps(sign, fy);
        const __m128 ax = _mm_andnot_ps(sign, fx);
        const __m128 hi = _mm_max_ps(ay, ax);
        const __m128 lo = _mm_min_ps(ay, ax);
        const __m128 t = _mm_and_ps(_mm_cmpgt_ps(hi, _mm_setzero_ps()),
                                    _mm_mul_ps(lo, _mm_rcp_ps(hi)));
        const __m128 one = _mm_set1_ps(1.f);
        __m128 r = _mm_mul_ps(t, _mm_add_ps(_mm_set1_ps(0.78539816f),
                       _mm_mul_ps(_mm_sub_ps(one, t),
                                  _mm_add_ps(_mm_set1_ps(0.2447f),
                                             _mm_mul_ps(_mm_set1_ps(0.0663f), t)))));
        r = _mm_blendv_ps(r, _mm_sub_ps(_mm_set1_ps(1.57079633f), r), _mm_cmpgt_ps(ay, ax));
        r = _mm_blendv_ps(r, _mm_sub_ps(_mm_set1_ps(3.14159265f), r),
                          _mm_cmplt_ps(fx, _mm_setzero_ps()));
        r = _mm_blendv_ps(r, _mm_xor_ps(r, sign), _mm_cmplt_ps(fy, _mm_setzero_ps()));
        const __m128 inv_width = _mm_setr_ps(ring_inv_dpsi_[ka], ring_inv_dpsi_[kb],
                                             shell_inv_dchi_[us], shell_inv_dchi_[us]);
        const int na = ring_ncells_[ka], nb = ring_ncells_[kb], nc = shell_nchi_[us];
        const __m128i nmax = _mm_setr_epi32(na - 1, nb - 1, nc - 1, nc - 1);
        __m128i iv = _mm_cvttps_epi32(
            _mm_mul_ps(_mm_add_ps(r, _mm_set1_ps(3.14159265f)), inv_width));
        iv = _mm_min_epi32(_mm_max_epi32(iv, _mm_setzero_si128()), nmax);
        const int ia = _mm_extract_epi32(iv, 0);
        const int ib = _mm_extract_epi32(iv, 1);
        const int ic = _mm_extract_epi32(iv, 2);

        // Edge tests: below(k) is c_k*y - s_k*x < 0, rounded as edge_below's is.
        const Edge* ea = &psi_edge_[static_cast<std::size_t>(psi_edge_offset_[ka] + ia)];
        const Edge* eb = &psi_edge_[static_cast<std::size_t>(psi_edge_offset_[kb] + ib)];
        const Edge* ec = &chi_edge_[static_cast<std::size_t>(chi_edge_offset_[us] + ic)];
        const __m256d c0 = _mm256_setr_pd(ea[0].c, eb[0].c, ec[0].c, ec[0].c);
        const __m256d s0 = _mm256_setr_pd(ea[0].s, eb[0].s, ec[0].s, ec[0].s);
        const __m256d c1 = _mm256_setr_pd(ea[1].c, eb[1].c, ec[1].c, ec[1].c);
        const __m256d s1 = _mm256_setr_pd(ea[1].s, eb[1].s, ec[1].s, ec[1].s);
        const unsigned below0 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(
            edge_cross(c0, s0, vy, vx), zero, _CMP_LT_OQ)));
        const unsigned below1 = static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(
            edge_cross(c1, s1, vy, vx), zero, _CMP_LT_OQ)));
        const unsigned pos = static_cast<unsigned>(_mm_movemask_ps(
            _mm_castsi128_ps(_mm_cmpgt_epi32(iv, _mm_setzero_si128()))));
        const unsigned under = static_cast<unsigned>(_mm_movemask_ps(
            _mm_castsi128_ps(_mm_cmpgt_epi32(nmax, iv))));
        const unsigned down = below0 & pos;
        const unsigned up = ~below1 & under;
        PairBins out;
        out.shell = shell;
        out.cell_a = ring_first_cell_[ka] + ia - static_cast<int>(down & 1u)
                     + static_cast<int>(up & 1u);
        out.cell_b = ring_first_cell_[kb] + ib - static_cast<int>((down >> 1) & 1u)
                     + static_cast<int>((up >> 1) & 1u);
        out.chi = ic - static_cast<int>((down >> 2) & 1u) + static_cast<int>((up >> 2) & 1u);
        return out;
    }
#endif

    [[nodiscard]] PairBins bins_scalar(const PairVectors& pv) const noexcept {
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

    [[nodiscard]] std::size_t block_base(int slice, int a, int b) const noexcept {
        return ((static_cast<std::size_t>(slice) * kNumTypes + static_cast<std::size_t>(a))
                * kNumTypes + static_cast<std::size_t>(b))
               * static_cast<std::size_t>(block_stride_);
    }

    /// Direction of a bin edge: the edge at angle e (pi + atan2 convention)
    /// is the unit vector at atan2-angle e - pi.
    struct Edge { double c, s; };

    /// below(k): the vector (x, y) lies before edge e, i.e. c*y - s*x < 0.
    /// The rounding is spelled out so that it does not depend on how the
    /// compiler contracts: one fused multiply-subtract when FMA is available
    /// (the form GCC and clang already chose for the plain expression), two
    /// rounded products under MCPU_FP_CONTRACT=off or without FMA.
    /// bins_avx2's edge_cross rounds the same way.
    [[nodiscard]] static bool edge_below(const Edge& e, double y, double x) noexcept {
#if defined(__FMA__) && !defined(MCPU_FP_CONTRACT_OFF)
        return std::fma(e.c, y, -(e.s * x)) < 0.0;
#else
        return e.c * y - e.s * x < 0.0;
#endif
    }

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
        const bool down = (i > 0) & edge_below(edge[i], y, x);
        const bool up = (i < n - 1) & !edge_below(edge[i + 1], y, x);
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
