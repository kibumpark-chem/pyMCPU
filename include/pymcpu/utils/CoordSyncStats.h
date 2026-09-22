#pragma once
/// Process-wide counters for Eigen Matrix3Xf materialization of SoA coords.
/// Hot MC paths must never call these; benches assert they stay zero.

#include <cstdint>

namespace mcpu {

struct CoordSyncStats {
    std::uint64_t num_coords_eigen_materializations = 0; // SoA → Eigen (read)
    std::uint64_t num_coords_eigen_writes_back = 0;      // Eigen → SoA (write)

    void reset() noexcept {
        num_coords_eigen_materializations = 0;
        num_coords_eigen_writes_back = 0;
    }
};

// Intentionally a single process-wide singleton, NOT per-Context state.
// Context::coordSyncStats()/reset_coord_sync_stats() are thin wrappers around
// this same global, so calling them on one Context instance reads/resets
// counters shared by every other Context in the process (e.g. all walkers
// in a multi-walker/replica run).
inline CoordSyncStats& coord_sync_stats() noexcept {
    static CoordSyncStats s;
    return s;
}

inline void note_coords_eigen_materialization() noexcept {
    ++coord_sync_stats().num_coords_eigen_materializations;
}

inline void note_coords_eigen_write_back() noexcept {
    ++coord_sync_stats().num_coords_eigen_writes_back;
}

} // namespace mcpu
