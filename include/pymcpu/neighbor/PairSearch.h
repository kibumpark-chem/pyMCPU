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
[[gnu::always_inline]] inline bool walk_cells(const Grid& grid, const Probe& p,
                                              const WalkArgs& wa,
                                              CellFn&& cell_fn) {
    if constexpr (C == Cells::Stencil) {
        return grid.for_each_neighbor_cell_span_while_unmoved(
            p.x, p.y, p.z, wa.moved_counts, cell_fn);
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
[[gnu::always_inline]] inline bool probe_static(const Grid& grid, const Probe& p,
                                                const WalkArgs& wa, Fn& fn) {
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
        });
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
    for (int k = 0; k < n; ++k) {
        const int i = moved[order == Order::Forward ? k : n - 1 - k];
        const Probe p{i, cnew.x(i), cnew.y(i), cnew.z(i)};
        if (!detail::probe_static<C>(grid, p, wa, fn)) return false;
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
