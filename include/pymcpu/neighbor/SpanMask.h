#pragma once
/// The 8-slot distance prefilter over a packed cell span, and the slot walk
/// built on it. Header-only: every function here is always inlined into the
/// term that calls it, so it adds no call per cell or per pair.
#include <algorithm>
#include <cstdint>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mcpu::neighbor {

/// Relative slack of span_mask8's cutoff over the exact one; see there.
inline constexpr float kSpanMaskSlack = 1.0f + 1.0e-4f;

/// What a pair callback tells the walk that called it.
enum class Visit : unsigned char { Continue, Stop };

/// One packed cell of an OpenCellGrid in contiguous mode: `count` atoms,
/// their ids and their ACCEPTED coordinates. The storage behind each
/// pointer holds whole 8-slot blocks (see span_mask8).
struct CellSpan {
    const int* __restrict__ ids;
    const float* __restrict__ x;
    const float* __restrict__ y;
    const float* __restrict__ z;
    int count;
};

/// The site a walk is querying for, at its trial position.
struct Probe {
    int i;
    float x, y, z;
};

/// Bit k of the result is set when slot m0+k (k < 8, m0+k < count) of a
/// packed cell span holds an atom other than `self` whose r2 to (nx,ny,nz)
/// is not greater than `lim2` (NaN counts as kept). A pre-filter only:
/// callers take the set bits in increasing order, recompute r2 in scalar
/// and apply the exact original test, so the pairs and their order are
/// unchanged. lim2 carries kSpanMaskSlack over the real cutoff because FMA
/// contraction can make the scalar r2 differ from this one by a few ulp.
///
/// Why: the walks over a moved atom's new neighbours were bound by branch
/// mispredicts, one data-dependent branch per slot, and most slots fail the
/// cutoff. Eight distances per AVX2 vector and one branch per surviving slot
/// take ~12% off actin's default move mix on top of skipping fully moved
/// cells. The AVX2 path loads 8 slots from m0 even when fewer remain: a span
/// is CELL_CAPACITY slots, a multiple of 8, so the load stays inside the
/// cell's storage and the lanes past count are masked off. Builds without
/// AVX2 take the scalar loop, which keeps the same bits.
[[gnu::always_inline]] inline unsigned span_mask8(
        float nx, float ny, float nz, const int* cids, const float* cx,
        const float* cy, const float* cz, int m0, int count, int self,
        float lim2) noexcept {
#if defined(__AVX2__)
    const __m256 dx = _mm256_sub_ps(_mm256_set1_ps(nx), _mm256_loadu_ps(cx + m0));
    const __m256 dy = _mm256_sub_ps(_mm256_set1_ps(ny), _mm256_loadu_ps(cy + m0));
    const __m256 dz = _mm256_sub_ps(_mm256_set1_ps(nz), _mm256_loadu_ps(cz + m0));
    const __m256 r2 = _mm256_add_ps(
        _mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
        _mm256_mul_ps(dz, dz));
    unsigned keep = static_cast<unsigned>(_mm256_movemask_ps(
        _mm256_cmp_ps(r2, _mm256_set1_ps(lim2), _CMP_NGT_UQ)));
    const __m256i ids =
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cids + m0));
    const unsigned is_self = static_cast<unsigned>(_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(ids, _mm256_set1_epi32(self)))));
    keep &= ~is_self;
    // left <= 0 (a block past the span, see probe_static_collect) keeps none.
    // Clamped with min/max rather than a conditional: GCC turned the
    // conditional into a branch on the cell's occupancy, which mispredicted
    // on 14-18% of all mispredicts of a Mu move (actin, PGK1).
    const int left = std::min(std::max(count - m0, 0), 8);
    const unsigned valid = (1u << left) - 1u;
    return keep & valid;
#else
    unsigned keep = 0;
    const int n = std::min(8, count - m0);
    for (int k = 0; k < n; ++k) {
        const float dx = nx - cx[m0 + k];
        const float dy = ny - cy[m0 + k];
        const float dz = nz - cz[m0 + k];
        const float r2 = dx * dx + dy * dy + dz * dz;
        if (!(r2 > lim2) && cids[m0 + k] != self) keep |= 1u << k;
    }
    return keep;
#endif
}

#if defined(__AVX2__)
/// span_mask8 that also returns the block's ids and distances, for callers
/// that take r2 from here instead of recomputing it, so a caller can test
/// the exact cutoff here (no slack). With FMA and the default contraction,
/// r2 is fma(dz, dz, fma(dx, dx, dy * dy)), the contraction GCC 8.5 emits
/// for the scalar dx * dx + dy * dy + dz * dz at -march=haswell. Under
/// MCPU_FP_CONTRACT_OFF, or without FMA, r2 is the unfused sum, which is
/// the scalar form exactly. A compiler that contracts the scalar form
/// differently can move a pair on its cutoff by an ulp.
[[gnu::always_inline]] inline unsigned span_hits8(
        float nx, float ny, float nz, const int* cids, const float* cx,
        const float* cy, const float* cz, int m0, int count, int self,
        float cut2, __m256i& ids, __m256& r2) noexcept {
    const __m256 dx = _mm256_sub_ps(_mm256_set1_ps(nx), _mm256_loadu_ps(cx + m0));
    const __m256 dy = _mm256_sub_ps(_mm256_set1_ps(ny), _mm256_loadu_ps(cy + m0));
    const __m256 dz = _mm256_sub_ps(_mm256_set1_ps(nz), _mm256_loadu_ps(cz + m0));
#if defined(__FMA__) && !defined(MCPU_FP_CONTRACT_OFF)
    r2 = _mm256_fmadd_ps(dz, dz, _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy)));
#else
    r2 = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dx, dx), _mm256_mul_ps(dy, dy)),
                       _mm256_mul_ps(dz, dz));
#endif
    unsigned keep = static_cast<unsigned>(_mm256_movemask_ps(
        _mm256_cmp_ps(r2, _mm256_set1_ps(cut2), _CMP_NGT_UQ)));
    ids = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cids + m0));
    const unsigned is_self = static_cast<unsigned>(_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(ids, _mm256_set1_epi32(self)))));
    keep &= ~is_self;
    const int left = std::min(std::max(count - m0, 0), 8);
    const unsigned valid = (1u << left) - 1u;
    return keep & valid;
}
#endif

/// Calls slot_fn(m) for every slot m of `s` that span_mask8 keeps for the
/// probe (r2 <= lim2, not the probe itself), in increasing slot order.
/// Returns false as soon as slot_fn returns Visit::Stop, true otherwise.
/// slot_fn must apply the exact test itself: this is the prefilter only.
template <class SlotFn>
[[gnu::always_inline]] inline bool for_each_span_hit(
        const CellSpan& s, const Probe& p, float lim2, SlotFn&& slot_fn) noexcept {
    for (int m0 = 0; m0 < s.count; m0 += 8) {
        unsigned bits = span_mask8(p.x, p.y, p.z, s.ids, s.x, s.y, s.z, m0,
                                   s.count, p.i, lim2);
        while (bits) {
            const int m = m0 + __builtin_ctz(bits);
            bits &= bits - 1u;
            if (slot_fn(m) == Visit::Stop) return false;
        }
    }
    return true;
}

}  // namespace mcpu::neighbor
