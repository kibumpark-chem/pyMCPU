#pragma once
/// Walk drivers for a move's new pairs: moved sites against the accepted
/// grid (moved_vs_static, hot_then_moved) and moved sites against each
/// other (moved_vs_moved).
///
/// Every driver is an always-inlined template that takes its callback as a
/// forwarding reference, so a term's hot function stays the only frame: no
/// std::function, no virtual call and no call per pair
/// (scripts/check_inlining.py checks the built library for this).
///
/// The layer finds candidates; the term decides. A grid callback gets the
/// probe (the moved site i at its trial position), the partner j and the
/// cell span slot m holding j, and recomputes the exact r2 itself from
/// p.x - s.x[m] etc., so FMA contraction of that expression is the term's
/// and cannot change with the layer. The prefilter here carries
/// kSpanMaskSlack over the term's cutoff for the same reason.
///
/// Order: probes in the order given (or as documented per driver), cells in
/// the grid walk's order, slots in increasing order. Same input, same
/// pairs in the same order.
///
/// Grid walks visit nothing when the grid is not configured or not in
/// contiguous mode (the walk returns true without calling back), as the
/// OpenCellGrid walks do; callers check that before choosing this path.
#include <cstddef>
#include <cstdint>

#include "pymcpu/neighbor/MovedCells.h"
#include "pymcpu/neighbor/SpanMask.h"

namespace mcpu::neighbor {

/// Which cells a grid walk visits around a probe.
enum class Cells : unsigned char {
    /// The grid's neighbour stencil; cells whose atoms all moved are skipped.
    Stencil,
    /// Only cells that can hold a point within `radius` of the probe
    /// (reachable-cell enumeration); cells whose atoms all moved are skipped.
    WithinRadius,
    /// The stencil, minus cells beyond `radius`; no moved-cell skip.
    StencilWithinRadius,
    /// Chosen per call from WalkArgs::cells (one of the three above).
    FromArgs,
};

/// Order in which moved_vs_static takes the moved sites.
enum class Order : unsigned char { Forward, Reverse };

struct WalkArgs {
    const std::uint8_t* is_moved;      // per site, nonzero if the move carries it
    std::size_t n_sites;               // size of is_moved
    const std::uint8_t* moved_counts;  // MovedCellScope::counts() for this grid
    float radius;                      // WithinRadius / StencilWithinRadius
    float lim2;                        // prefilter r2 (cutoff^2 * kSpanMaskSlack)
    Cells cells;                       // used when the driver's C is FromArgs
};

namespace detail {

template <Cells C, class Grid, class CellFn>
[[gnu::always_inline]] inline bool walk_cells(
        const Grid& grid, const Probe& p, const WalkArgs& wa, CellFn&& cell_fn,
        typename Grid::StencilMemo* memo = nullptr) {
    if constexpr (C == Cells::Stencil) {
        return grid.for_each_neighbor_cell_span_while_unmoved(
            p.x, p.y, p.z, wa.moved_counts, memo, cell_fn);
    } else if constexpr (C == Cells::WithinRadius) {
        return grid.for_each_cell_span_within_fast_unmoved(
            p.x, p.y, p.z, wa.radius, wa.moved_counts, cell_fn);
    } else if constexpr (C == Cells::StencilWithinRadius) {
        return grid.for_each_neighbor_cell_span_while_within(
            p.x, p.y, p.z, wa.radius, cell_fn);
    } else {
        return wa.cells == Cells::StencilWithinRadius
                   ? grid.for_each_neighbor_cell_span_while_within(
                         p.x, p.y, p.z, wa.radius, cell_fn)
                   : grid.for_each_cell_span_within_fast_unmoved(
                         p.x, p.y, p.z, wa.radius, wa.moved_counts, cell_fn);
    }
}

/// The grid's unmoved neighbours of probe p within the prefilter: calls
/// fn(p, j, span, m) for each, skipping moved partners (their packed
/// positions are stale). False once fn returns Visit::Stop.
template <Cells C, class Grid, class Fn>
[[gnu::always_inline]] inline bool probe_static(
        const Grid& grid, const Probe& p, const WalkArgs& wa, Fn& fn,
        typename Grid::StencilMemo* memo = nullptr) {
    static_assert(Grid::CELL_CAPACITY % 8 == 0,
                  "span_mask8 loads whole 8-slot blocks of a cell span");
    const std::uint8_t* const is_moved = wa.is_moved;
    const float lim2 = wa.lim2;
    return walk_cells<C>(
        grid, p, wa,
        [&](const int* __restrict__ cids, const float* __restrict__ cx,
            const float* __restrict__ cy, const float* __restrict__ cz,
            int count) {
            const CellSpan s{cids, cx, cy, cz, count};
            return for_each_span_hit(s, p, lim2, [&](int m) {
                const int j = cids[m];
                if (is_moved[static_cast<std::size_t>(j)]) return Visit::Continue;
                return fn(p, j, s, m);
            });
        },
        memo);
}

/// The pair search's compress table: for an 8-bit hit mask, the lanes of
/// its set bits in increasing order, one byte each.
struct HitLanes {
    std::uint64_t lanes[256];
    constexpr HitLanes() : lanes() {
        for (unsigned b = 0; b < 256; ++b) {
            std::uint64_t v = 0;
            int n = 0;
            for (unsigned k = 0; k < 8; ++k)
                if (b & (1u << k)) v |= static_cast<std::uint64_t>(k) << (8 * n++);
            lanes[b] = v;
        }
    }
};
inline constexpr HitLanes kHitLanes{};

/// probe_static for the Stencil walk, in two phases. The cell walk only
/// collects the slots the prefilter keeps (cell index << 8 | slot), with
/// no branch per slot or per 8-slot block; the second phase then calls fn
/// on them in the same order. One loop over a probe's candidates replaces
/// a short, data-dependent loop per block, whose exits were the walk's
/// largest source of branch mispredicts. Same pairs, same order, same
/// arguments as probe_static, so the result is bit-identical.
template <class Grid, class Fn>
[[gnu::always_inline]] inline bool probe_static_collect(
        const Grid& grid, const Probe& p, const WalkArgs& wa, Fn& fn,
        typename Grid::StencilMemo* memo) {
    static_assert(Grid::CELL_CAPACITY % 8 == 0 && Grid::CELL_CAPACITY >= 16 &&
                      Grid::CELL_CAPACITY < 256,
                  "slots are packed in 8 bits and read in 8-slot blocks");
    constexpr int kCells = 32;
    constexpr int kHits = 27 * Grid::CELL_CAPACITY;
    CellSpan tab[kCells];
    int hits[kHits + 8];
    int n_cells = 0;
    int n_hits = 0;
    const std::uint8_t* const is_moved = wa.is_moved;
    const float lim2 = wa.lim2;
    const auto drain = [&]() -> bool {
        for (int k = 0; k < n_hits; ++k) {
            const int v = hits[k];
            const CellSpan& s = tab[v >> 8];
            const int m = v & 0xFF;
            const int j = s.ids[m];
            if (is_moved[static_cast<std::size_t>(j)]) continue;
            if (fn(p, j, s, m) == Visit::Stop) return false;
        }
        n_cells = 0;
        n_hits = 0;
        return true;
    };
    const auto collect = [&](unsigned bits, int m0) {
#if defined(__AVX2__)
        const __m256i idx = _mm256_add_epi32(
            _mm256_set1_epi32(((n_cells - 1) << 8) + m0),
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
        const __m256i perm = _mm256_cvtepu8_epi32(_mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(&kHitLanes.lanes[bits])));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(hits + n_hits),
                            _mm256_permutevar8x32_epi32(idx, perm));
        n_hits += __builtin_popcount(bits);
#else
        while (bits) {
            hits[n_hits++] = ((n_cells - 1) << 8) + m0 + __builtin_ctz(bits);
            bits &= bits - 1u;
        }
#endif
    };
    // A stencil whose candidates overflow the buffers (never the Mu grid's
    // 27 cells) is walked again from the first cell not yet collected, so
    // fn keeps a single call site and is inlined as in probe_static.
    int skip = 0;
    for (;;) {
        int seen = 0;
        bool full = false;
        (void)walk_cells<Cells::Stencil>(
            grid, p, wa,
            [&](const int* __restrict__ cids, const float* __restrict__ cx,
                const float* __restrict__ cy, const float* __restrict__ cz,
                int count) {
                if (seen++ < skip) return true;
                if (n_cells == kCells || n_hits > kHits - Grid::CELL_CAPACITY) {
                    full = true;
                    return false;
                }
                tab[n_cells++] = CellSpan{cids, cx, cy, cz, count};
                // Two blocks whatever the count: a loop whose trip count
                // follows the cell's occupancy mispredicts around 8 atoms.
                collect(span_mask8(p.x, p.y, p.z, cids, cx, cy, cz, 0, count,
                                   p.i, lim2), 0);
                collect(span_mask8(p.x, p.y, p.z, cids, cx, cy, cz, 8, count,
                                   p.i, lim2), 8);
                for (int m0 = 16; m0 < count; m0 += 8)
                    collect(span_mask8(p.x, p.y, p.z, cids, cx, cy, cz, m0,
                                       count, p.i, lim2), m0);
                return true;
            },
            memo);
        if (!drain()) return false;
        if (!full) return true;
        skip = seen - 1;
    }
}

}  // namespace detail

/// For each moved site i (moved[0..n), in `order`): fn(p, j, span, m) ->
/// Visit for every unmoved site j the grid holds near i's trial position
/// (prefilter r2 <= wa.lim2). Returns false as soon as fn returns
/// Visit::Stop, true when every probe was walked.
template <Cells C, class Grid, class Coords, class Fn>
[[gnu::always_inline]] inline bool moved_vs_static(
        const Grid& grid, const Coords& cnew, const int* moved, int n,
        Order order, const WalkArgs& wa, Fn&& fn) {
    typename Grid::StencilMemo memo;
    for (int k = 0; k < n; ++k) {
        const int i = moved[order == Order::Forward ? k : n - 1 - k];
        const Probe p{i, cnew.x(i), cnew.y(i), cnew.z(i)};
        if constexpr (C == Cells::Stencil) {
            if (!detail::probe_static_collect(grid, p, wa, fn, &memo)) return false;
        } else {
            if (!detail::probe_static<C>(grid, p, wa, fn, &memo)) return false;
        }
    }
    return true;
}

/// Clash-first search: probes the sites of `hot` that the move carries
/// (most recent first; ids out of range for wa.n_sites are skipped), then
/// moved[n-1], ..., moved[0] (pass n = 0 to test the hot sites only).
/// fn(p, j, span, m) -> Visit as in moved_vs_static. Returns the probe
/// site whose walk fn stopped, or -1. A hot site that also appears in
/// `moved` is probed again there; only the order depends on `hot`, never
/// the answer.
template <Cells C, class Grid, class Coords, int Cap, class Fn>
[[gnu::always_inline]] inline int hot_then_moved(
        const Grid& grid, const Coords& cnew, const HotList<Cap>& hot,
        const int* moved, int n, const WalkArgs& wa, Fn&& fn) {
    const int n_hot = hot.n;
    const int n_scan = n_hot + n;
    for (int s = 0; s < n_scan; ++s) {
        int i;
        if (s < n_hot) {
            i = hot.a[s];
            if (static_cast<std::size_t>(i) >= wa.n_sites ||
                !wa.is_moved[static_cast<std::size_t>(i)])
                continue;
        } else {
            i = moved[n - 1 - (s - n_hot)];
        }
        const Probe p{i, cnew.x(i), cnew.y(i), cnew.z(i)};
        if (!detail::probe_static<C>(grid, p, wa, fn)) return i;
    }
    return -1;
}

/// Every pair of moved sites (moved[a], moved[b]), a < b, in that order,
/// with r2 = cnew.dist2(i, j) <= cutoff2: fn(i, j, r2) -> Visit. Returns
/// false as soon as fn returns Visit::Stop.
template <class Coords, class Fn>
[[gnu::always_inline]] inline bool moved_vs_moved(
        const Coords& cnew, const int* moved, int n, float cutoff2, Fn&& fn) {
    for (int a = 0; a < n; ++a) {
        const int i = moved[a];
        for (int b = a + 1; b < n; ++b) {
            const int j = moved[b];
            const float r2 = cnew.dist2(i, j);
            if (r2 > cutoff2) continue;
            if (fn(i, j, r2) == Visit::Stop) return false;
        }
    }
    return true;
}

}  // namespace mcpu::neighbor
