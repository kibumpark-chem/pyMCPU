#pragma once
/// Fallback neighbor enumeration when dense grid cannot cover a trial (no PBC).
#include <cstdint>
#include <vector>

#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/CoordView.h"

namespace mcpu {

/// Moved-vs-all scan within cutoff (plain Euclidean). Caller applies topo/energy.
struct NeighborFallback {
    /// For each moved atom i, invoke func(i, j, r2) for all j with r2 <= cutoff_sq.
    /// Pair rules (no double count):
    ///   - moved–fixed: once per (i,j)
    ///   - moved–moved: only when i < j (skipped entirely if is_rigid)
    template <typename Func>
    static void for_each_moved_neighbor(
        const CoordsSoA& coords_i_source,
        const CoordsSoA& coords_j_source,
        const std::vector<int>& moved_indices,
        const std::vector<uint8_t>& is_moved,
        float cutoff_sq,
        bool is_rigid,
        Func&& func
    ) {
        const CoordView ci(coords_i_source);
        const CoordView cj(coords_j_source);
        const int n = cj.n_;
        for (int i : moved_indices) {
            for (int j = 0; j < n; ++j) {
                if (j == i) continue;
                const bool j_moved = is_moved[static_cast<size_t>(j)] != 0;
                if (j_moved) {
                    if (is_rigid) continue;
                    if (i > j) continue;
                }
                const float r2 = ci.dist2(i, cj, j);
                if (r2 <= cutoff_sq)
                    func(i, j, r2);
            }
        }
    }
};

} // namespace mcpu
