#pragma once
/// Dense open-boundary cell grid (NO PBC / no wrap / no minimum-image).
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <unordered_set>
#include <vector>
#include <chrono>

#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/utils/CoordsSoA.h"

namespace mcpu {

/// Occupied/valid neighbor-cell list (R≤2 → ≤125 entries).
struct NeighborCellList {
    /// Production runs stencil radius 1 (cell size == cutoff), which needs 27
    /// entries. kCap was 125 -- sized for radius 2 -- so ~78% of every entry was
    /// padding that still occupied cache lines: valid_stencil_ + occupied_stencil_
    /// came to 3.97 MB at sce, competing with the 8.66 MB topo_flag_ table for a
    /// 35.75 MB shared L3. Deriving the cap from the max supported radius cuts
    /// that ~4.5x.
    ///
    /// Raising kMaxStencilRadius to 2 restores the old capacity; the grid now
    /// REFUSES to build a stencil that would not fit rather than silently
    /// truncating it (see build_valid_stencil_), because a dropped neighbour cell
    /// means dropped pairs and a wrong energy.
    static constexpr int kMaxStencilRadius = 1;
    static constexpr int kCap =
        (2 * kMaxStencilRadius + 1) * (2 * kMaxStencilRadius + 1) *
        (2 * kMaxStencilRadius + 1);
    std::int16_t count = 0;
    // FIX: was std::int16_t (max 32767) -- silently wrapped for any grid
    // exceeding 32767 total cells, well within NeighborConfig::max_cells_total
    // (2,000,000, NeighborConfig.h). A wrapped (negative) value sign-extended
    // to size_t produced a wild out-of-bounds index into occupied_stencil_
    // (confirmed via valgrind: "Invalid read ... not stack'd, malloc'd or
    // freed" in build_occupied_stencil(), OpenCellGrid.h). int32_t comfortably
    // covers the full configured cell-count range.
    std::int32_t cells[kCap] = {};
};

/// ``MCPU_OCCUPIED_STENCIL`` modes (default 0 = off).
enum class OccupiedStencilMode : int {
    Off = 0,           ///< legacy offset stencil
    QueryOnly = 1,     ///< build once; no incremental maintain (diag)
    MaintainOnly = 2,  ///< maintain lists; query still uses offsets (diag)
    Full = 3,          ///< query occupied + maintain 0↔1
};

struct BoxBounds {
    Eigen::Vector3f lo = Eigen::Vector3f::Zero();
    Eigen::Vector3f hi = Eigen::Vector3f::Ones();
    bool valid = false;
};

inline BoxBounds aabb_of_coords(const Eigen::Matrix3Xf& coords, float margin) {
    BoxBounds b;
    if (coords.cols() == 0) return b;
    b.lo = coords.rowwise().minCoeff();
    b.hi = coords.rowwise().maxCoeff();
    b.lo.array() -= margin;
    b.hi.array() += margin;
    // Avoid zero-thickness axes
    for (int d = 0; d < 3; ++d) {
        if (b.hi[d] <= b.lo[d]) {
            b.hi[d] = b.lo[d] + 1.0f;
        }
    }
    b.valid = true;
    return b;
}

inline BoxBounds aabb_of_coords(const CoordsSoA& coords, float margin) {
    BoxBounds b;
    if (coords.n <= 0) return b;
    float lx = coords.x[0], ly = coords.y[0], lz = coords.z[0];
    float hx = lx, hy = ly, hz = lz;
    for (int i = 1; i < coords.n; ++i) {
        const float xi = coords.x[static_cast<size_t>(i)];
        const float yi = coords.y[static_cast<size_t>(i)];
        const float zi = coords.z[static_cast<size_t>(i)];
        lx = std::min(lx, xi); ly = std::min(ly, yi); lz = std::min(lz, zi);
        hx = std::max(hx, xi); hy = std::max(hy, yi); hz = std::max(hz, zi);
    }
    b.lo = Eigen::Vector3f(lx, ly, lz);
    b.hi = Eigen::Vector3f(hx, hy, hz);
    b.lo.array() -= margin;
    b.hi.array() += margin;
    for (int d = 0; d < 3; ++d) {
        if (b.hi[d] <= b.lo[d]) {
            b.hi[d] = b.lo[d] + 1.0f;
        }
    }
    b.valid = true;
    return b;
}

inline bool point_in_bounds(const Eigen::Vector3f& p, const BoxBounds& b) {
    return b.valid
        && p.x() >= b.lo.x() && p.x() < b.hi.x()
        && p.y() >= b.lo.y() && p.y() < b.hi.y()
        && p.z() >= b.lo.z() && p.z() < b.hi.z();
}

inline bool point_in_bounds(float px, float py, float pz, const BoxBounds& b) {
    return b.valid
        && px >= b.lo.x() && px < b.hi.x()
        && py >= b.lo.y() && py < b.hi.y()
        && pz >= b.lo.z() && pz < b.hi.z();
}

/// Returns false if required cell count exceeds caps.
inline bool compute_grid_shape(const BoxBounds& b, float cell,
                               const NeighborConfig& cfg,
                               int& nx, int& ny, int& nz) {
    if (!b.valid || cell <= 0.f) return false;
    auto dim = [&](float lo, float hi) -> int {
        return std::max(1, static_cast<int>(std::ceil((hi - lo) / cell)));
    };
    nx = dim(b.lo.x(), b.hi.x());
    ny = dim(b.lo.y(), b.hi.y());
    nz = dim(b.lo.z(), b.hi.z());
    if (cfg.max_nx > 0 && nx > cfg.max_nx) return false;
    if (cfg.max_ny > 0 && ny > cfg.max_ny) return false;
    if (cfg.max_nz > 0 && nz > cfg.max_nz) return false;
    const std::uint64_t total =
        static_cast<std::uint64_t>(nx) *
        static_cast<std::uint64_t>(ny) *
        static_cast<std::uint64_t>(nz);
    if (total > cfg.max_cells_total) return false;
    return true;
}

/**
 * Dense linked-cell grid with open boundaries.
 * Cell index from floor((x-lo)/cell); neighbor stencil is clipped (no wrap).
 *
 * Dual storage: legacy head_/next_/prev_ linked lists (always updated) plus
 * optional contiguous per-cell atom arrays for sequential neighbor walks.
 * Actin max occupancy ≈ 22 → CELL_CAPACITY=48.
 * Scale>1 (larger cells) measured max_occ=31 (s=1.2) / 46 (s=1.5) but
 * wall-regressed; CAP=48 gives headroom vs overflow at production scale=1.0.
 */
template <int Cap>
class BasicOpenCellGrid {
public:
    /// Fixed atoms per cell for contiguous mode. Overflow disables contiguous.
    static constexpr int CELL_CAPACITY = Cap;

    BasicOpenCellGrid() { init_contiguous_default_(); }

    explicit BasicOpenCellGrid(float cell_size_in, int num_atoms_hint = 0)
        : cell_size_(cell_size_in), inv_cell_(cell_size_in > 0.f ? 1.f / cell_size_in : 0.f) {
        if (num_atoms_hint > 0) ensure_atom_capacity(num_atoms_hint);
        init_contiguous_default_();
    }

    /// Occupancy count for cell ``c``. O(1).
    [[nodiscard]] int cell_atom_count(int c) const noexcept {
        if (c < 0 || c >= static_cast<int>(cell_count_.size())) return 0;
        return cell_count_[static_cast<size_t>(c)];
    }

    /// Cell index for arbitrary coordinates (alias of cell_index). O(1).
    /// Returns -1 if outside grid bounds or not configured.
    [[nodiscard]] int cell_of(float x, float y, float z) const noexcept {
        return cell_index(x, y, z);
    }

    /// Decode linear cell id to (ix,iy,iz). O(1). Assumes valid in-grid id.
    void decode_cell(int c, int& ix, int& iy, int& iz) const noexcept {
        iz = c % nz_;
        const int t = c / nz_;
        iy = t % ny_;
        ix = t / ny_;
    }

    /**
     * Visit in-grid neighbor cells of linear cell ``c``.
     * Mode Full/QueryOnly: occupied neighbors only. Else: full stencil. O(stencil).
     */
    template <typename Func>
    void for_each_neighbor_cell_of(int c, Func&& func) const {
        if (!configured_ || c < 0 ||
            c >= static_cast<int>(head_.size()))
            return;
        const bool use_occ =
            (occ_mode_ == OccupiedStencilMode::Full ||
             occ_mode_ == OccupiedStencilMode::QueryOnly) &&
            static_cast<size_t>(c) < occupied_stencil_.size();
        if (use_occ) {
            const auto& occ = occupied_stencil_[static_cast<size_t>(c)];
            for (int k = 0; k < occ.count; ++k)
                func(static_cast<int>(occ.cells[k]));
            return;
        }
        if (neighbor_offsets_.empty()) return;
        int ix0 = 0, iy0 = 0, iz0 = 0;
        decode_cell(c, ix0, iy0, iz0);
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_)
                continue;
            func((ix * ny_ + iy) * nz_ + iz);
        }
    }

    OccupiedStencilMode occupied_stencil_mode() const noexcept {
        return occ_mode_;
    }
    /// Force mode (e.g. Off on HB/scratch grids). O(1) or O(n_cells) if clearing.
    void set_occupied_stencil_mode(OccupiedStencilMode m) {
        occ_mode_ = m;
        if (m == OccupiedStencilMode::Off) {
            valid_stencil_.clear();
            occupied_stencil_.clear();
        }
    }
    std::uint64_t occupied_maint_ns() const noexcept { return occ_maint_ns_; }
    std::uint64_t occupied_maint_calls() const noexcept {
        return occ_maint_calls_;
    }
    void reset_occupied_maint_stats() const noexcept {
        occ_maint_ns_ = 0;
        occ_maint_calls_ = 0;
    }

    /// Enable occupied stencil on this grid (Mu denselist only). O(N_CELLS×stencil).
    void enable_occupied_stencil(OccupiedStencilMode mode) {
        if (mode == OccupiedStencilMode::Off) {
            set_occupied_stencil_mode(OccupiedStencilMode::Off);
            return;
        }
        occ_mode_ = mode;
        build_valid_stencil_();
        build_occupied_stencil();
        reset_occupied_maint_stats();
    }

    /// Read ``MCPU_OCCUPIED_STENCIL`` (default Full=3; 0=off). O(1).
    static OccupiedStencilMode occupied_mode_from_env() {
        static const int kMode = [] {
            const char* e = std::getenv("MCPU_OCCUPIED_STENCIL");
            // Default ON (Full). Set =0 to disable.
            if (!e || !e[0]) return 3;
            const int v = std::atoi(e);
            return (v >= 0 && v <= 3) ? v : 3;
        }();
        return static_cast<OccupiedStencilMode>(kMode);
    }

    /// Rebuild occupied_stencil_ from cell_count_. O(N_CELLS × stencil).
    void build_occupied_stencil() {
        if (occ_mode_ == OccupiedStencilMode::Off) return;
        // NOTE: this count-only staleness check is theoretically coarser than
        // checking actual grid shape (nx_/ny_/nz_), but configure() always
        // unconditionally clears valid_stencil_ (see configure(), below), so
        // the "same total count, different shape" scenario this could in
        // principle miss never actually arises via any reachable call path
        // (verified).
        if (valid_stencil_.size() != head_.size()) build_valid_stencil_();
        const size_t n_cells = head_.size();
        occupied_stencil_.assign(n_cells, NeighborCellList{});
        for (size_t c = 0; c < n_cells; ++c) {
            if (cell_count_[c] == 0) continue;
            const auto& vs = valid_stencil_[c];
            for (int k = 0; k < vs.count; ++k) {
                const int nc = vs.cells[k];
                // Defensive bounds check (matches the same guard already used
                // in stencil_cell_became_occupied_/_empty_ below) -- belt and
                // suspenders alongside the int32_t widening above.
                if (nc < 0 || static_cast<size_t>(nc) >= occupied_stencil_.size())
                    continue;
                auto& s = occupied_stencil_[static_cast<size_t>(nc)];
                if (s.count >= NeighborCellList::kCap) continue;
                s.cells[s.count++] = static_cast<std::int32_t>(c);
            }
        }
    }

#ifndef NDEBUG
    /// Verify occupied vs valid∩occupied. O(N_CELLS × stencil). Debug only.
    void verify_occupied_stencil() const {
        if (occ_mode_ == OccupiedStencilMode::Off) return;
        if (valid_stencil_.size() != head_.size() ||
            occupied_stencil_.size() != head_.size())
            return;
        for (size_t c = 0; c < head_.size(); ++c) {
            std::unordered_set<int> expected;
            const auto& vs = valid_stencil_[c];
            for (int k = 0; k < vs.count; ++k) {
                const int nc = static_cast<int>(vs.cells[k]);
                if (cell_count_[static_cast<size_t>(nc)] > 0)
                    expected.insert(nc);
            }
            std::unordered_set<int> actual;
            const auto& occ = occupied_stencil_[c];
            for (int k = 0; k < occ.count; ++k)
                actual.insert(static_cast<int>(occ.cells[k]));
            if (expected != actual) {
                std::fprintf(stderr,
                    "OCCUPIED_STENCIL_MISMATCH cell=%zu exp=%zu actual=%d\n",
                    c, expected.size(), static_cast<int>(occ.count));
            }
        }
    }
#endif

    float cell_size() const noexcept { return cell_size_; }
    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nz() const noexcept { return nz_; }
    std::uint64_t num_cells() const noexcept {
        return static_cast<std::uint64_t>(nx_) * ny_ * nz_;
    }
    const BoxBounds& bounds() const noexcept { return bounds_; }
    bool configured() const noexcept { return configured_; }
    bool use_contiguous() const noexcept { return use_contiguous_; }
    void set_use_contiguous(bool on) noexcept { use_contiguous_ = on; }
    int peak_cell_occupancy() const noexcept { return peak_cell_occupancy_; }

    /// True if denselist geometry matches ``b``/``cell`` (skip reconfigure). O(1).
    bool matches_geometry(const BoxBounds& b, float cell,
                          const NeighborConfig& cfg) const {
        if (!configured_ || !b.valid) return false;
        if (cell_size_ != cell) return false;
        int nx = 0, ny = 0, nz = 0;
        if (!compute_grid_shape(b, cell, cfg, nx, ny, nz)) return false;
        return nx == nx_ && ny == ny_ && nz == nz_ &&
               bounds_.lo.x() == b.lo.x() && bounds_.lo.y() == b.lo.y() &&
               bounds_.lo.z() == b.lo.z();
    }

    /// Diagnostic: count empty vs nonempty stencil cells at (x,y,z). O(stencil).
    /// Temporary for walk characterization; not used in production denselist.
    struct StencilOccupancy {
        std::size_t empty = 0;
        std::size_t nonempty = 0;
        std::size_t atoms = 0;   ///< sum of cell_count over nonempty in-stencil cells
        std::size_t oob = 0;     ///< stencil offsets clipped by grid boundary
    };
    StencilOccupancy probe_stencil_occupancy(float x, float y, float z) const {
        StencilOccupancy s;
        if (!configured_ || neighbor_offsets_.empty()) return s;
        const int ix0 =
            static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 =
            static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 =
            static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_) {
                ++s.oob;
                continue;
            }
            const int c = (ix * ny_ + iy) * nz_ + iz;
            int count = 0;
            if (use_contiguous_) {
                count = cell_count_[static_cast<size_t>(c)];
            } else {
                for (int a = head_[static_cast<size_t>(c)]; a != -1;
                     a = next_[static_cast<size_t>(a)])
                    ++count;
            }
            if (count == 0) {
                ++s.empty;
            } else {
                ++s.nonempty;
                s.atoms += static_cast<std::size_t>(count);
            }
        }
        return s;
    }

    /// Contiguous atom-id span for cell ``c``. O(1). Requires use_contiguous().
    [[nodiscard]] std::pair<const int*, int> cell_atoms_span(int c) const noexcept {
        if (!use_contiguous_ || c < 0 ||
            c >= static_cast<int>(cell_count_.size())) {
            return {nullptr, 0};
        }
        return {cell_atoms_.data() + static_cast<size_t>(c) * CELL_CAPACITY,
                cell_count_[static_cast<size_t>(c)]};
    }

    /// Packed x/y/z spans parallel to cell_atoms_span. O(1).
    [[nodiscard]] const float* cell_x_span(int c) const noexcept {
        return cell_x_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
    }
    [[nodiscard]] const float* cell_y_span(int c) const noexcept {
        return cell_y_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
    }
    [[nodiscard]] const float* cell_z_span(int c) const noexcept {
        return cell_z_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
    }

    /**
     * Visit each in-stencil neighbor cell as contiguous id+coord spans.
     * Empty cells skipped. O(stencil × avg_occ).
     * cell_fn(atoms, cx, cy, cz, count).
     */
    template <typename CellFunc>
    void for_each_neighbor_cell_span(float x, float y, float z,
                                     CellFunc&& cell_fn,
                                     std::uint64_t* cell_visits = nullptr) const {
        if (!configured_ || neighbor_offsets_.empty() || !use_contiguous_) return;
        const int ix0 =
            static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 =
            static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 =
            static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));
        if (cell_visits) {
            *cell_visits +=
                static_cast<std::uint64_t>(neighbor_offsets_.size());
        }
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_)
                continue;
            const int c = (ix * ny_ + iy) * nz_ + iz;
            const int count = cell_count_[static_cast<size_t>(c)];
            if (count == 0) continue;
            const size_t base = static_cast<size_t>(c) * CELL_CAPACITY;
            cell_fn(cell_atoms_.data() + base, cell_x_.data() + base,
                    cell_y_.data() + base, cell_z_.data() + base, count);
        }
    }

    /**
     * Like for_each_neighbor_cell_span_while, but visits ONLY the neighbour
     * cells whose NEAREST POINT lies within `radius` of (x,y,z).
     *
     * Why: a hard-core overlap needs r < ~2.8 A, but the cells are sized for the
     * ~5.1 A Mu cutoff. Asking the overlap question over the full 27-cell
     * stencil sweeps a box ~5.8x larger in volume than the question needs.
     * Culling by point-to-box distance leaves ~1.3 cells per atom instead of 27
     * (measured on actin) for three comparisons per axis.
     *
     * NOTE: the equivalent cull at the CONTACT radius was tried and reverted as
     * slower -- there it removed only ~25% of cell visits, and a ~25%-taken
     * branch in the inner loop mispredicted. At the clash radius it removes
     * ~95%, so the branch resolves the same way almost every time.
     */
    template <typename CellFunc>
    bool for_each_neighbor_cell_span_while_within(float x, float y, float z,
                                                  float radius,
                                                  CellFunc&& cell_fn) const {
        if (!configured_ || neighbor_offsets_.empty() || !use_contiguous_)
            return true;
        const float r2max = radius * radius;
        const int ix0 = static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 = static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 = static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ || iz >= nz_)
                continue;
            float d2 = 0.f;
            const float lx = bounds_.lo.x() + static_cast<float>(ix) * cell_size_;
            if (x < lx) { const float t = lx - x; d2 += t * t; }
            else if (x > lx + cell_size_) { const float t = x - lx - cell_size_; d2 += t * t; }
            const float ly = bounds_.lo.y() + static_cast<float>(iy) * cell_size_;
            if (y < ly) { const float t = ly - y; d2 += t * t; }
            else if (y > ly + cell_size_) { const float t = y - ly - cell_size_; d2 += t * t; }
            const float lz = bounds_.lo.z() + static_cast<float>(iz) * cell_size_;
            if (z < lz) { const float t = lz - z; d2 += t * t; }
            else if (z > lz + cell_size_) { const float t = z - lz - cell_size_; d2 += t * t; }
            if (d2 > r2max) continue;
            const int c = (ix * ny_ + iy) * nz_ + iz;
            const int count = cell_count_[static_cast<size_t>(c)];
            if (count == 0) continue;
            const size_t base = static_cast<size_t>(c) * CELL_CAPACITY;
            if (!cell_fn(cell_atoms_.data() + base, cell_x_.data() + base,
                         cell_y_.data() + base, cell_z_.data() + base, count))
                return false;
        }
        return true;
    }

    /**
     * Visit only the cells that can hold an atom within `radius`, WITHOUT
     * enumerating the 27-cell stencil first.
     *
     * The previous attempt (for_each_neighbor_cell_span_while_within) walked all
     * 27 stencil offsets and culled each by point-to-box distance. Measured: it
     * LOST, because paying ~10 operations per offset to reject 25 of 27 offsets
     * costs more than the distance arithmetic it saves. The cost of a cell walk
     * is the walk, not the distances.
     *
     * Here the surviving offsets are derived directly from where inside its cell
     * the query point sits: along each axis the neighbour at -1 is needed only
     * when the point is within `radius` of the low face, and +1 only when it is
     * within `radius` of the high face. With radius 2.83 A in ~5.1 A cells that
     * is 1 or 2 cells per axis, so 1-8 cells total (~3.8 on average) with no
     * per-offset rejection test at all.
     */
    template <typename CellFunc>
    bool for_each_cell_span_within_fast(float x, float y, float z, float radius,
                                        CellFunc&& cell_fn) const {
        if (!configured_ || !use_contiguous_) return true;
        const float fx = (x - bounds_.lo.x()) * inv_cell_;
        const float fy = (y - bounds_.lo.y()) * inv_cell_;
        const float fz = (z - bounds_.lo.z()) * inv_cell_;
        const int ix0 = static_cast<int>(std::floor(fx));
        const int iy0 = static_cast<int>(std::floor(fy));
        const int iz0 = static_cast<int>(std::floor(fz));
        // offset in Angstrom from this cell's low face, per axis
        const float ox = (fx - static_cast<float>(ix0)) * cell_size_;
        const float oy = (fy - static_cast<float>(iy0)) * cell_size_;
        const float oz = (fz - static_cast<float>(iz0)) * cell_size_;
        // FIXED: was int[3][2]. When the query radius exceeds HALF a cell, an
        // atom can be within `radius` of BOTH faces along an axis, so all three
        // of {i0-1, i0, i0+1} are needed -- three entries, not two. That is the
        // case here: radius ~2.83 A against ~5.1 A cells, which happens for
        // 2.24 < offset < 2.83, i.e. 11.6% of positions per axis and ~31% of
        // atoms on at least one axis. The old size wrote the third entry into
        // the next axis's slot (x, y) or past the array entirely (z), so those
        // atoms were tested against a WRONG cell. It never changed a result only
        // because the full contact walk that follows repeats the clash test, so
        // a miss was caught there; a false positive would have been silent.
        int axs[3][3];
        int nax[3];
        const float hi = cell_size_ - radius;
        auto fill = [&](int k, int i0, float off) {
            int n = 0;
            if (off < radius) axs[k][n++] = i0 - 1;
            axs[k][n++] = i0;
            if (off > hi) axs[k][n++] = i0 + 1;
            nax[k] = n;
        };
        fill(0, ix0, ox);
        fill(1, iy0, oy);
        fill(2, iz0, oz);
        for (int a = 0; a < nax[0]; ++a) {
            const int ix = axs[0][a];
            if (ix < 0 || ix >= nx_) continue;
            for (int b = 0; b < nax[1]; ++b) {
                const int iy = axs[1][b];
                if (iy < 0 || iy >= ny_) continue;
                for (int cc = 0; cc < nax[2]; ++cc) {
                    const int iz = axs[2][cc];
                    if (iz < 0 || iz >= nz_) continue;
                    const int c = (ix * ny_ + iy) * nz_ + iz;
                    const int count = cell_count_[static_cast<size_t>(c)];
                    if (count == 0) continue;
                    const size_t base = static_cast<size_t>(c) * CELL_CAPACITY;
                    if (!cell_fn(cell_atoms_.data() + base, cell_x_.data() + base,
                                 cell_y_.data() + base, cell_z_.data() + base,
                                 count))
                        return false;
                }
            }
        }
        return true;
    }

    /**
     * for_each_cell_span_within_fast for a query that skips every atom a move
     * displaced: `moved_per_cell[c]` counts the atoms listed in cell c that
     * the pending move displaces. A cell whose atoms all moved (count ==
     * moved_per_cell[c], which also covers an empty cell) holds nothing such
     * a query keeps, so it is not visited. The cells still visited come in
     * the unfiltered order, and their spans are the same.
     */
    template <typename CellFunc>
    bool for_each_cell_span_within_fast_unmoved(float x, float y, float z,
                                                float radius,
                                                const std::uint8_t* moved_per_cell,
                                                CellFunc&& cell_fn) const {
        static_assert(CELL_CAPACITY <= 255,
                      "moved_per_cell holds per-cell counts in a uint8");
        if (!configured_ || !use_contiguous_) return true;
        const float fx = (x - bounds_.lo.x()) * inv_cell_;
        const float fy = (y - bounds_.lo.y()) * inv_cell_;
        const float fz = (z - bounds_.lo.z()) * inv_cell_;
        const int ix0 = static_cast<int>(std::floor(fx));
        const int iy0 = static_cast<int>(std::floor(fy));
        const int iz0 = static_cast<int>(std::floor(fz));
        const float ox = (fx - static_cast<float>(ix0)) * cell_size_;
        const float oy = (fy - static_cast<float>(iy0)) * cell_size_;
        const float oz = (fz - static_cast<float>(iz0)) * cell_size_;
        int axs[3][3];
        int nax[3];
        const float hi = cell_size_ - radius;
        auto fill = [&](int k, int i0, float off) {
            int n = 0;
            if (off < radius) axs[k][n++] = i0 - 1;
            axs[k][n++] = i0;
            if (off > hi) axs[k][n++] = i0 + 1;
            nax[k] = n;
        };
        fill(0, ix0, ox);
        fill(1, iy0, oy);
        fill(2, iz0, oz);
        // Collect the cells to visit first, without branching on the skip
        // test (data-dependent, so a frequent mispredict), then visit them
        // in the same order.
        int live[27];
        int n_live = 0;
        for (int a = 0; a < nax[0]; ++a) {
            const int ix = axs[0][a];
            if (ix < 0 || ix >= nx_) continue;
            for (int b = 0; b < nax[1]; ++b) {
                const int iy = axs[1][b];
                if (iy < 0 || iy >= ny_) continue;
                for (int cc = 0; cc < nax[2]; ++cc) {
                    const int iz = axs[2][cc];
                    if (iz < 0 || iz >= nz_) continue;
                    const int c = (ix * ny_ + iy) * nz_ + iz;
                    live[n_live] = c;
                    n_live += (cell_count_[static_cast<size_t>(c)] !=
                               static_cast<int>(moved_per_cell[static_cast<size_t>(c)]));
                }
            }
        }
        for (int k = 0; k < n_live; ++k) {
            const size_t c = static_cast<size_t>(live[k]);
            const size_t base = c * CELL_CAPACITY;
            if (!cell_fn(cell_atoms_.data() + base, cell_x_.data() + base,
                         cell_y_.data() + base, cell_z_.data() + base,
                         cell_count_[c]))
                return false;
        }
        return true;
    }

    /// for_each_neighbor_cell_span_while minus the cells whose atoms all
    /// moved; see for_each_cell_span_within_fast_unmoved.
    template <typename CellFunc>
    bool for_each_neighbor_cell_span_while_unmoved(float x, float y, float z,
                                                   const std::uint8_t* moved_per_cell,
                                                   CellFunc&& cell_fn) const {
        if (!configured_ || neighbor_offsets_.empty() || !use_contiguous_)
            return true;
        const int ix0 =
            static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 =
            static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 =
            static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));
        constexpr std::size_t kMaxLive = 128;
        if (neighbor_offsets_.size() <= kMaxLive) {
            // Branch-free collection, then the visits in the same order; see
            // for_each_cell_span_within_fast_unmoved.
            int live[kMaxLive];
            int n_live = 0;
            for (const CellOffset& o : neighbor_offsets_) {
                const int ix = ix0 + o.dx;
                const int iy = iy0 + o.dy;
                const int iz = iz0 + o.dz;
                if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                    iz >= nz_)
                    continue;
                const int c = (ix * ny_ + iy) * nz_ + iz;
                live[n_live] = c;
                n_live += (cell_count_[static_cast<size_t>(c)] !=
                           static_cast<int>(moved_per_cell[static_cast<size_t>(c)]));
            }
            for (int k = 0; k < n_live; ++k) {
                const size_t c = static_cast<size_t>(live[k]);
                const size_t base = c * CELL_CAPACITY;
                if (!cell_fn(cell_atoms_.data() + base, cell_x_.data() + base,
                             cell_y_.data() + base, cell_z_.data() + base,
                             cell_count_[c]))
                    return false;
            }
            return true;
        }
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_)
                continue;
            const int c = (ix * ny_ + iy) * nz_ + iz;
            const int count = cell_count_[static_cast<size_t>(c)];
            if (count == static_cast<int>(moved_per_cell[static_cast<size_t>(c)]))
                continue;
            const size_t base = static_cast<size_t>(c) * CELL_CAPACITY;
            if (!cell_fn(cell_atoms_.data() + base, cell_x_.data() + base,
                         cell_y_.data() + base, cell_z_.data() + base, count))
                return false;
        }
        return true;
    }

    /// Like for_each_neighbor_cell_span but stops if cell_fn returns false. O(stencil×occ).
    template <typename CellFunc>
    bool for_each_neighbor_cell_span_while(float x, float y, float z,
                                           CellFunc&& cell_fn,
                                           std::uint64_t* cell_visits = nullptr) const {
        if (!configured_ || neighbor_offsets_.empty() || !use_contiguous_)
            return true;
        const int ix0 =
            static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 =
            static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 =
            static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));
        if (cell_visits) {
            *cell_visits +=
                static_cast<std::uint64_t>(neighbor_offsets_.size());
        }
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_)
                continue;
            const int c = (ix * ny_ + iy) * nz_ + iz;
            const int count = cell_count_[static_cast<size_t>(c)];
            if (count == 0) continue;
            const size_t base = static_cast<size_t>(c) * CELL_CAPACITY;
            if (!cell_fn(cell_atoms_.data() + base, cell_x_.data() + base,
                         cell_y_.data() + base, cell_z_.data() + base, count))
                return false;
        }
        return true;
    }

    void ensure_atom_capacity(int n) {
        if (static_cast<int>(next_.size()) < n) {
            const size_t old = next_.size();
            next_.resize(static_cast<size_t>(n), -1);
            prev_.resize(static_cast<size_t>(n), -1);
            atom_cell_.resize(static_cast<size_t>(n), -1);
            for (size_t i = old; i < next_.size(); ++i) {
                next_[i] = prev_[i] = atom_cell_[i] = -1;
            }
        }
    }

    /// Configure / resize dense cells from bounds.
    /// ``cell`` = denselist bin size; ``query_radius`` = interaction range for stencil
    /// (typically r_cut (+ skin for Verlet)). Returns false if caps exceeded.
    bool configure(const BoxBounds& b, float cell, const NeighborConfig& cfg,
                   float query_radius = -1.f) {
        int nx = 0, ny = 0, nz = 0;
        if (!compute_grid_shape(b, cell, cfg, nx, ny, nz)) {
            configured_ = false;
            return false;
        }
        cell_size_ = cell;
        inv_cell_ = 1.f / cell;
        query_radius_ = (query_radius > 0.f) ? query_radius : cell;
        bounds_ = b;
        // Snap hi to exact multiple so hi is exclusive-friendly
        bounds_.hi.x() = bounds_.lo.x() + static_cast<float>(nx) * cell;
        bounds_.hi.y() = bounds_.lo.y() + static_cast<float>(ny) * cell;
        bounds_.hi.z() = bounds_.lo.z() + static_cast<float>(nz) * cell;
        nx_ = nx; ny_ = ny; nz_ = nz;
        const size_t n_cells = static_cast<size_t>(nx_) * ny_ * nz_;
        head_.assign(n_cells, -1);
        // Every reader of the packed arrays stops at cell_count_ (span_mask8
        // masks the lanes past it), so only the slots the previous binning
        // filled are cleared and the allocation is kept. Writing the whole
        // n_cells * CELL_CAPACITY block on every set_positions was ~88% of
        // its time; this is O(atoms) plus the resize.
        clear_packed_slots_();
        cell_count_.assign(n_cells, 0);
        const size_t pack = n_cells * static_cast<size_t>(CELL_CAPACITY);
        cell_atoms_.resize(pack, 0);
        cell_x_.resize(pack, 0.f);
        cell_y_.resize(pack, 0.f);
        cell_z_.resize(pack, 0.f);
        peak_cell_occupancy_ = 0;
        init_contiguous_default_();
        configured_ = true;
        // Clear atom membership (caller must re-insert)
        std::fill(next_.begin(), next_.end(), -1);
        std::fill(prev_.begin(), prev_.end(), -1);
        std::fill(atom_cell_.begin(), atom_cell_.end(), -1);
        precompute_neighbor_offsets_();
        // Always start Off — NeighborSystem enables on Mu denselist only.
        occ_mode_ = OccupiedStencilMode::Off;
        valid_stencil_.clear();
        occupied_stencil_.clear();
        return true;
    }

    int stencil_radius() const noexcept { return stencil_radius_; }
    [[nodiscard]] bool stencil_overflow() const noexcept { return stencil_overflow_; }
    float query_radius() const noexcept { return query_radius_; }
    std::size_t neighbor_offsets_count() const noexcept {
        return neighbor_offsets_.size();
    }

    /// Rebuild neighbor stencil only (O(R³)). Does not clear atom membership.
    /// Used to temporarily widen the Mu grid for Verlet list builds (r_cut+skin)
    /// without re-binning denselist atoms.
    void set_query_radius(float query_radius) {
        if (!configured_ || query_radius <= 0.f) return;
        query_radius_ = query_radius;
        precompute_neighbor_offsets_();
    }

    void clear_cells_keep_shape() {
        std::fill(head_.begin(), head_.end(), -1);
        std::fill(next_.begin(), next_.end(), -1);
        std::fill(prev_.begin(), prev_.end(), -1);
        std::fill(atom_cell_.begin(), atom_cell_.end(), -1);
        std::fill(cell_count_.begin(), cell_count_.end(), 0);
        peak_cell_occupancy_ = 0;
        init_contiguous_default_();
        if (occ_mode_ != OccupiedStencilMode::Off)
            occupied_stencil_.assign(head_.size(), NeighborCellList{});
    }

    /// Rebuild contiguous id+coord arrays from linked lists. O(N_atoms).
    /// Requires coords for packing; leaves use_contiguous_=false on overflow.
    ///
    /// Every cell is rebuilt from the linked list regardless of whether an
    /// earlier cell (in iteration order) overflowed: an early return here
    /// used to leave every not-yet-visited cell's cell_count_/cell_atoms_
    /// stale relative to the (always-authoritative) linked list. Since
    /// use_contiguous_ is unconditionally re-armed to true at the top of
    /// this function on every call, a later call that doesn't happen to hit
    /// the same overflow could re-enable contiguous mode while some cells
    /// still held that stale data -- producing exactly the
    /// "atom not found in cell" desync seen in production (p18.8.4).
    void build_contiguous_from_linked(const CoordsSoA& coords) {
        if (cell_count_.size() != head_.size()) {
            cell_count_.assign(head_.size(), 0);
            const size_t pack = head_.size() * static_cast<size_t>(CELL_CAPACITY);
            cell_atoms_.assign(pack, 0);
            cell_x_.assign(pack, 0.f);
            cell_y_.assign(pack, 0.f);
            cell_z_.assign(pack, 0.f);
        }
        std::fill(cell_count_.begin(), cell_count_.end(), 0);
        peak_cell_occupancy_ = 0;
        use_contiguous_ = true;
        bool warned = false;
        for (size_t c = 0; c < head_.size(); ++c) {
            for (int a = head_[c]; a != -1; a = next_[static_cast<size_t>(a)]) {
                if (cell_count_[c] >= CELL_CAPACITY) {
                    use_contiguous_ = false;
                    if (!warned) {
                        warned = true;
                        std::fprintf(stderr,
                            "WARN: cell %zu overflow (count>=%d CELL_CAPACITY). "
                            "Falling back to linked-list mode.\n",
                            c, CELL_CAPACITY);
                    }
                    break;  // stop packing THIS cell only; still rebuild the rest
                }
                const int k = cell_count_[c]++;
                const size_t base = c * static_cast<size_t>(CELL_CAPACITY) +
                                    static_cast<size_t>(k);
                cell_atoms_[base] = a;
                cell_x_[base] = coords.x[static_cast<size_t>(a)];
                cell_y_[base] = coords.y[static_cast<size_t>(a)];
                cell_z_[base] = coords.z[static_cast<size_t>(a)];
            }
            peak_cell_occupancy_ =
                std::max(peak_cell_occupancy_, cell_count_[c]);
        }
        if (occ_mode_ != OccupiedStencilMode::Off) build_occupied_stencil();
    }

    inline int cell_index(float px, float py, float pz) const {
        if (!configured_) return -1;
        const int ix = static_cast<int>(std::floor((px - bounds_.lo.x()) * inv_cell_));
        const int iy = static_cast<int>(std::floor((py - bounds_.lo.y()) * inv_cell_));
        const int iz = static_cast<int>(std::floor((pz - bounds_.lo.z()) * inv_cell_));
        if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ || iz >= nz_)
            return -1;
        return (ix * ny_ + iy) * nz_ + iz;
    }

    inline int cell_index(const Eigen::Vector3f& pos) const {
        return cell_index(pos.x(), pos.y(), pos.z());
    }

    /// Linear cell id containing atom, or -1 if not inserted / out of bounds.
    inline int atom_cell(int atom_id) const noexcept {
        if (atom_id < 0 || atom_id >= static_cast<int>(atom_cell_.size())) return -1;
        return atom_cell_[static_cast<size_t>(atom_id)];
    }

    /// Walk atoms currently in cell ``c`` (read-only). O(cell_count).
    template <typename Func>
    void for_each_in_cell(int c, Func&& func) const {
        if (!configured_ || c < 0 || c >= static_cast<int>(head_.size())) return;
        // CHANGED: contiguous cell storage
        if (use_contiguous_) {
            const int* atoms =
                cell_atoms_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
            const int count = cell_count_[static_cast<size_t>(c)];
            for (int k = 0; k < count; ++k) func(atoms[k]);
            return;
        }
        for (int a = head_[static_cast<size_t>(c)]; a != -1; a = next_[static_cast<size_t>(a)]) {
            func(a);
        }
    }

    /**
     * Visit every in-grid neighbor cell for query (x,y,z) using the precomputed
     * open-boundary stencil (radius = ceil(query_radius/cell_size)).
     */
    template <typename Func>
    void for_each_neighbor_cell(float x, float y, float z, Func&& func) const {
        if (!configured_ || neighbor_offsets_.empty()) return;
        const int ix0 = static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 = static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 = static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ || iz >= nz_)
                continue;
            func((ix * ny_ + iy) * nz_ + iz);
        }
    }

    /**
     * Append unique linear cell ids from the Mu/HB stencil around (x,y,z).
     * Dedup via stamp array (no hash set).
     */
    void collect_neighbor_cells(float x, float y, float z,
                                std::uint32_t stamp,
                                std::vector<std::uint32_t>& cell_stamp,
                                std::vector<int>& unique_cells) const {
        if (!configured_ || stamp == 0) return;
        if (static_cast<std::uint64_t>(cell_stamp.size()) < num_cells()) {
            cell_stamp.assign(static_cast<size_t>(num_cells()), 0u);
        }
        for_each_neighbor_cell(x, y, z, [&](int c) {
            const size_t ck = static_cast<size_t>(c);
            if (cell_stamp[ck] == stamp) return;
            cell_stamp[ck] = stamp;
            unique_cells.push_back(c);
        });
    }

    void insert(int atom_id, float px, float py, float pz) {
        assert(atom_id >= 0);
        ensure_atom_capacity(atom_id + 1);
        if (atom_cell_[static_cast<size_t>(atom_id)] != -1)
            remove(atom_id);
        const int c = cell_index(px, py, pz);
        if (c < 0) return;
        atom_cell_[static_cast<size_t>(atom_id)] = c;
        const int old = head_[static_cast<size_t>(c)];
        head_[static_cast<size_t>(c)] = atom_id;
        next_[static_cast<size_t>(atom_id)] = old;
        prev_[static_cast<size_t>(atom_id)] = -1;
        if (old != -1) prev_[static_cast<size_t>(old)] = atom_id;
        // CHANGED: contiguous cell storage — push-front to match linked order
        if (use_contiguous_) contiguous_add_front_(atom_id, c, px, py, pz);
    }

    void insert(int atom_id, const Eigen::Vector3f& pos) {
        insert(atom_id, pos.x(), pos.y(), pos.z());
    }

    void insert(int atom_id, const CoordsSoA& coords) {
        insert(atom_id, coords.x[static_cast<size_t>(atom_id)],
               coords.y[static_cast<size_t>(atom_id)],
               coords.z[static_cast<size_t>(atom_id)]);
    }

    void remove(int atom_id) {
        if (atom_id < 0 || atom_id >= static_cast<int>(atom_cell_.size())) return;
        const int c = atom_cell_[static_cast<size_t>(atom_id)];
        if (c < 0) return;
        const int p = prev_[static_cast<size_t>(atom_id)];
        const int n = next_[static_cast<size_t>(atom_id)];
        if (p != -1) next_[static_cast<size_t>(p)] = n;
        else head_[static_cast<size_t>(c)] = n;
        if (n != -1) prev_[static_cast<size_t>(n)] = p;
        atom_cell_[static_cast<size_t>(atom_id)] = -1;
        next_[static_cast<size_t>(atom_id)] = -1;
        prev_[static_cast<size_t>(atom_id)] = -1;
        // CHANGED: contiguous cell storage
        if (use_contiguous_) contiguous_remove_(atom_id, c);
    }

    void update_position(int atom_id, float px, float py, float pz) {
        if (atom_id < 0 || atom_id >= static_cast<int>(atom_cell_.size())) return;
        if (atom_cell_[static_cast<size_t>(atom_id)] < 0) {
            insert(atom_id, px, py, pz);
            return;
        }
        const int oc = atom_cell_[static_cast<size_t>(atom_id)];
        const int nc = cell_index(px, py, pz);
        // Same cell: keep membership, refresh packed coords. O(occ).
        if (nc == oc) {
            if (use_contiguous_) update_packed_coords(atom_id, oc, px, py, pz);
            return;
        }
        remove(atom_id);
        if (nc >= 0) insert(atom_id, px, py, pz);
    }

    /// Update packed cell coords for atom after an in-cell position change. O(occ).
    void update_packed_coords(int atom, int cell, float px, float py, float pz) {
        const size_t base = static_cast<size_t>(cell) * CELL_CAPACITY;
        const int count = cell_count_[static_cast<size_t>(cell)];
        for (int k = 0; k < count; ++k) {
            if (cell_atoms_[base + static_cast<size_t>(k)] == atom) {
                cell_x_[base + static_cast<size_t>(k)] = px;
                cell_y_[base + static_cast<size_t>(k)] = py;
                cell_z_[base + static_cast<size_t>(k)] = pz;
                return;
            }
        }
        std::fprintf(stderr,
            "ERROR: update_packed_coords: atom %d not found in cell %d\n", atom,
            cell);
    }

    /// Update packed coords from SoA for atom in its current cell. O(occ).
    void update_packed_coords(int atom, const CoordsSoA& coords) {
        if (atom < 0 || atom >= static_cast<int>(atom_cell_.size())) return;
        const int cell = atom_cell_[static_cast<size_t>(atom)];
        if (cell < 0) return;
        update_packed_coords(atom, cell, coords.x[static_cast<size_t>(atom)],
                             coords.y[static_cast<size_t>(atom)],
                             coords.z[static_cast<size_t>(atom)]);
    }

    void update_position(int atom_id, const Eigen::Vector3f& new_pos) {
        update_position(atom_id, new_pos.x(), new_pos.y(), new_pos.z());
    }

    void update_position(int atom_id, const CoordsSoA& coords) {
        update_position(atom_id, coords.x[static_cast<size_t>(atom_id)],
                        coords.y[static_cast<size_t>(atom_id)],
                        coords.z[static_cast<size_t>(atom_id)]);
    }

    template <typename Func>
    void for_each_neighbor(float x, float y, float z, Func&& func,
                           std::uint64_t* cell_visits = nullptr,
                           float /*r_cut2*/ = -1.f) const {
        if (!configured_ || neighbor_offsets_.empty()) return;
        const int ix0 =
            static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 =
            static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 =
            static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));

        const std::uint64_t stencil =
            static_cast<std::uint64_t>(neighbor_offsets_.size());
        if (cell_visits) *cell_visits += stencil;

        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_)
                continue;
            const int c = (ix * ny_ + iy) * nz_ + iz;
            // CHANGED: contiguous cell storage
            if (use_contiguous_) {
                const int* atoms =
                    cell_atoms_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
                const int count = cell_count_[static_cast<size_t>(c)];
                for (int k = 0; k < count; ++k) func(atoms[k]);
            } else {
                for (int a = head_[static_cast<size_t>(c)]; a != -1;
                     a = next_[static_cast<size_t>(a)]) {
                    func(a);
                }
            }
        }
    }

    /// for_each_neighbor(x, y, z) minus the cells that for_each_neighbor(px,
    /// py, pz) visits. A caller that has just walked (px, py, pz) and dedupes
    /// what it sees gets nothing new from the cells the two stencils share, so
    /// it can skip them. The cells it does visit come in the same order as in
    /// the full walk.
    template <typename Func>
    void for_each_neighbor_not_near(float x, float y, float z,
                                    float px, float py, float pz,
                                    Func&& func,
                                    std::uint64_t* cell_visits = nullptr) const {
        if (!configured_ || neighbor_offsets_.empty()) return;
        const int ix0 =
            static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 =
            static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 =
            static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));
        const int jx0 =
            static_cast<int>(std::floor((px - bounds_.lo.x()) * inv_cell_));
        const int jy0 =
            static_cast<int>(std::floor((py - bounds_.lo.y()) * inv_cell_));
        const int jz0 =
            static_cast<int>(std::floor((pz - bounds_.lo.z()) * inv_cell_));
        if (ix0 == jx0 && iy0 == jy0 && iz0 == jz0) return;
        const int R = stencil_radius_;
        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_)
                continue;
            if (std::abs(ix - jx0) <= R && std::abs(iy - jy0) <= R &&
                std::abs(iz - jz0) <= R)
                continue;
            if (cell_visits) ++*cell_visits;
            const int c = (ix * ny_ + iy) * nz_ + iz;
            if (use_contiguous_) {
                const int* atoms =
                    cell_atoms_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
                const int count = cell_count_[static_cast<size_t>(c)];
                for (int k = 0; k < count; ++k) func(atoms[k]);
            } else {
                for (int a = head_[static_cast<size_t>(c)]; a != -1;
                     a = next_[static_cast<size_t>(a)]) {
                    func(a);
                }
            }
        }
    }

    template <typename Func>
    void for_each_neighbor(const Eigen::Vector3f& pos, Func&& func,
                           std::uint64_t* cell_visits = nullptr,
                           float r_cut2 = -1.f) const {
        for_each_neighbor(pos.x(), pos.y(), pos.z(), std::forward<Func>(func),
                          cell_visits, r_cut2);
    }

    template <typename Func>
    bool for_each_neighbor_while(float x, float y, float z, Func&& func,
                                 std::uint64_t* cell_visits = nullptr,
                                 float /*r_cut2*/ = -1.f) const {
        if (!configured_ || neighbor_offsets_.empty()) return true;
        const int ix0 =
            static_cast<int>(std::floor((x - bounds_.lo.x()) * inv_cell_));
        const int iy0 =
            static_cast<int>(std::floor((y - bounds_.lo.y()) * inv_cell_));
        const int iz0 =
            static_cast<int>(std::floor((z - bounds_.lo.z()) * inv_cell_));

        const std::uint64_t stencil =
            static_cast<std::uint64_t>(neighbor_offsets_.size());
        if (cell_visits) *cell_visits += stencil;

        for (const CellOffset& o : neighbor_offsets_) {
            const int ix = ix0 + o.dx;
            const int iy = iy0 + o.dy;
            const int iz = iz0 + o.dz;
            if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                iz >= nz_)
                continue;
            const int c = (ix * ny_ + iy) * nz_ + iz;
            // CHANGED: contiguous cell storage
            if (use_contiguous_) {
                const int* atoms =
                    cell_atoms_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
                const int count = cell_count_[static_cast<size_t>(c)];
                for (int k = 0; k < count; ++k) {
                    if (!func(atoms[k])) return false;
                }
            } else {
                for (int a = head_[static_cast<size_t>(c)]; a != -1; ) {
                    const int cur = a;
                    a = next_[static_cast<size_t>(a)];
                    if (!func(cur)) return false;
                }
            }
        }
        return true;
    }

    template <typename Func>
    bool for_each_neighbor_while(const Eigen::Vector3f& pos, Func&& func,
                                 std::uint64_t* cell_visits = nullptr,
                                 float r_cut2 = -1.f) const {
        return for_each_neighbor_while(pos.x(), pos.y(), pos.z(),
                                       std::forward<Func>(func), cell_visits,
                                       r_cut2);
    }

#if !defined(NDEBUG)
    /// Verify contiguous and linked-list cell memberships match. O(N_atoms).
    void verify_sync() const {
        if (!configured_ || !use_contiguous_) return;
        for (size_t c = 0; c < head_.size(); ++c) {
            std::unordered_set<int> ll_atoms;
            for (int a = head_[c]; a != -1; a = next_[static_cast<size_t>(a)]) {
                ll_atoms.insert(a);
            }
            std::unordered_set<int> ca_atoms;
            const int count = cell_count_[c];
            for (int k = 0; k < count; ++k) {
                ca_atoms.insert(
                    cell_atoms_[c * static_cast<size_t>(CELL_CAPACITY) +
                                static_cast<size_t>(k)]);
            }
            if (ll_atoms != ca_atoms) {
                std::fprintf(stderr,
                    "SYNC ERROR cell=%zu: ll_size=%zu ca_size=%d\n", c,
                    ll_atoms.size(), count);
            }
        }
    }

    /// Verify packed cell coords match CoordsSoA for all occupied cells. O(N).
    void verify_packed_coords(const CoordsSoA& coords) const {
        if (!configured_ || !use_contiguous_) return;
        for (size_t c = 0; c < cell_count_.size(); ++c) {
            const size_t base = c * static_cast<size_t>(CELL_CAPACITY);
            for (int k = 0; k < cell_count_[c]; ++k) {
                const int j = cell_atoms_[base + static_cast<size_t>(k)];
                const float dx = std::abs(
                    cell_x_[base + static_cast<size_t>(k)] -
                    coords.x[static_cast<size_t>(j)]);
                const float dy = std::abs(
                    cell_y_[base + static_cast<size_t>(k)] -
                    coords.y[static_cast<size_t>(j)]);
                const float dz = std::abs(
                    cell_z_[base + static_cast<size_t>(k)] -
                    coords.z[static_cast<size_t>(j)]);
                if (dx > 1e-4f || dy > 1e-4f || dz > 1e-4f) {
                    std::fprintf(stderr,
                        "COORD_SYNC_ERROR cell=%zu k=%d atom=%d "
                        "dx=%.6f dy=%.6f dz=%.6f\n",
                        c, k, j, dx, dy, dz);
                }
            }
        }
    }
#endif

private:
    struct CellOffset {
        int dx = 0;
        int dy = 0;
        int dz = 0;
    };

    /// Zero the packed slots [0, cell_count_[c]) of every cell. O(occupied).
    void clear_packed_slots_() {
        const size_t n = std::min(cell_count_.size(),
                                  cell_atoms_.size() / static_cast<size_t>(CELL_CAPACITY));
        for (size_t c = 0; c < n; ++c) {
            const int k = std::min(cell_count_[c], static_cast<int>(CELL_CAPACITY));
            if (k <= 0) continue;
            const size_t base = c * static_cast<size_t>(CELL_CAPACITY);
            std::fill_n(cell_atoms_.begin() + static_cast<std::ptrdiff_t>(base), k, 0);
            std::fill_n(cell_x_.begin() + static_cast<std::ptrdiff_t>(base), k, 0.f);
            std::fill_n(cell_y_.begin() + static_cast<std::ptrdiff_t>(base), k, 0.f);
            std::fill_n(cell_z_.begin() + static_cast<std::ptrdiff_t>(base), k, 0.f);
        }
    }

    void init_contiguous_default_() {
        static const bool kEnvOff = [] {
            const char* e = std::getenv("MCPU_USE_CONTIGUOUS_CELLS");
            return e && e[0] == '0';
        }();
        use_contiguous_ = !kEnvOff;
    }

    /// Insert atom at front of contiguous cell (matches linked push-front). O(occ).
    void contiguous_add_front_(int atom, int cell, float px, float py, float pz) {
        int& count = cell_count_[static_cast<size_t>(cell)];
        const bool was_empty = (count == 0);
        if (count >= CELL_CAPACITY) {
            use_contiguous_ = false;
            std::fprintf(stderr,
                "WARN: contiguous_add overflow cell=%d count=%d. "
                "Switching to linked-list fallback.\n",
                cell, count);
            return;
        }
        const size_t base = static_cast<size_t>(cell) * CELL_CAPACITY;
        int* id_base = cell_atoms_.data() + base;
        float* x_base = cell_x_.data() + base;
        float* y_base = cell_y_.data() + base;
        float* z_base = cell_z_.data() + base;
        for (int k = count; k > 0; --k) {
            id_base[k] = id_base[k - 1];
            x_base[k] = x_base[k - 1];
            y_base[k] = y_base[k - 1];
            z_base[k] = z_base[k - 1];
        }
        id_base[0] = atom;
        x_base[0] = px;
        y_base[0] = py;
        z_base[0] = pz;
        ++count;
        peak_cell_occupancy_ = std::max(peak_cell_occupancy_, count);
        // Only on empty→occupied (Hypothesis A: not every add).
        if (was_empty &&
            (occ_mode_ == OccupiedStencilMode::Full ||
             occ_mode_ == OccupiedStencilMode::MaintainOnly))
            stencil_cell_became_occupied_(cell);
    }

    /// Remove atom from contiguous cell, preserving relative order. O(occ).
    void contiguous_remove_(int atom, int cell) {
        const size_t base = static_cast<size_t>(cell) * CELL_CAPACITY;
        int* id_base = cell_atoms_.data() + base;
        float* x_base = cell_x_.data() + base;
        float* y_base = cell_y_.data() + base;
        float* z_base = cell_z_.data() + base;
        int& count = cell_count_[static_cast<size_t>(cell)];
        for (int k = 0; k < count; ++k) {
            if (id_base[k] == atom) {
                for (int j = k; j < count - 1; ++j) {
                    id_base[j] = id_base[j + 1];
                    x_base[j] = x_base[j + 1];
                    y_base[j] = y_base[j + 1];
                    z_base[j] = z_base[j + 1];
                }
                --count;
                // Only on occupied→empty.
                if (count == 0 &&
                    (occ_mode_ == OccupiedStencilMode::Full ||
                     occ_mode_ == OccupiedStencilMode::MaintainOnly))
                    stencil_cell_became_empty_(cell);
                return;
            }
        }
        std::fprintf(stderr,
            "ERROR: contiguous_remove: atom %d not found in cell %d\n", atom,
            cell);
    }

    /// Build per-cell in-bounds stencil. O(N_CELLS × stencil).
    void build_valid_stencil_() {
        const size_t n_cells = head_.size();
        stencil_overflow_ = false;
        if (neighbor_offsets_.size() > static_cast<size_t>(NeighborCellList::kCap)) {
            // Cell size smaller than the query radius pushed the stencil past
            // radius kMaxStencilRadius. Refuse rather than truncate.
            stencil_overflow_ = true;
            valid_stencil_.clear();
            occupied_stencil_.clear();
            return;
        }
        valid_stencil_.assign(n_cells, NeighborCellList{});
        for (size_t c = 0; c < n_cells; ++c) {
            int ix0 = 0, iy0 = 0, iz0 = 0;
            decode_cell(static_cast<int>(c), ix0, iy0, iz0);
            auto& vs = valid_stencil_[c];
            for (const CellOffset& o : neighbor_offsets_) {
                const int ix = ix0 + o.dx;
                const int iy = iy0 + o.dy;
                const int iz = iz0 + o.dz;
                if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ ||
                    iz >= nz_)
                    continue;
                if (vs.count >= NeighborCellList::kCap) {
                    // Truncating here would silently drop neighbour cells, and
                    // therefore pairs, producing a quietly wrong energy. Bail out
                    // and let the caller fall back to an exact enumeration.
                    stencil_overflow_ = true;
                    return;
                }
                vs.cells[vs.count++] =
                    static_cast<std::int32_t>((ix * ny_ + iy) * nz_ + iz);
            }
        }
    }

    /// Notify valid neighbors that cell is now occupied. O(stencil).
    void stencil_cell_became_occupied_(int cell) {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        if (static_cast<size_t>(cell) >= valid_stencil_.size()) return;
        const auto& vs = valid_stencil_[static_cast<size_t>(cell)];
        for (int k = 0; k < vs.count; ++k) {
            const int nc = static_cast<int>(vs.cells[k]);
            if (static_cast<size_t>(nc) >= occupied_stencil_.size()) continue;
            auto& s = occupied_stencil_[static_cast<size_t>(nc)];
            bool found = false;
            for (int i = 0; i < s.count; ++i) {
                if (s.cells[i] == static_cast<std::int32_t>(cell)) {
                    found = true;
                    break;
                }
            }
            if (!found && s.count < NeighborCellList::kCap)
                s.cells[s.count++] = static_cast<std::int32_t>(cell);
        }
        occ_maint_ns_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                t0)
                .count());
        ++occ_maint_calls_;
    }

    /// Notify valid neighbors that cell is now empty. O(stencil).
    void stencil_cell_became_empty_(int cell) {
        using Clock = std::chrono::steady_clock;
        const auto t0 = Clock::now();
        if (static_cast<size_t>(cell) >= valid_stencil_.size()) return;
        const auto& vs = valid_stencil_[static_cast<size_t>(cell)];
        for (int k = 0; k < vs.count; ++k) {
            const int nc = static_cast<int>(vs.cells[k]);
            if (static_cast<size_t>(nc) >= occupied_stencil_.size()) continue;
            auto& s = occupied_stencil_[static_cast<size_t>(nc)];
            for (int i = 0; i < s.count; ++i) {
                if (s.cells[i] == static_cast<std::int32_t>(cell)) {
                    s.cells[i] = s.cells[--s.count];
                    break;
                }
            }
        }
        occ_maint_ns_ += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() -
                                                                t0)
                .count());
        ++occ_maint_calls_;
    }

    void precompute_neighbor_offsets_() {
        neighbor_offsets_.clear();
        const float cs = cell_size_ > 0.f ? cell_size_ : 1.f;
        const float qr = query_radius_ > 0.f ? query_radius_ : cs;
        int R = static_cast<int>(std::ceil(static_cast<double>(qr) / cs));
        if (R < 1) R = 1;
        stencil_radius_ = R;
        const int extent = 2 * R + 1;
        neighbor_offsets_.reserve(static_cast<size_t>(extent) * extent * extent);
        for (int dx = -R; dx <= R; ++dx) {
            for (int dy = -R; dy <= R; ++dy) {
                for (int dz = -R; dz <= R; ++dz) {
                    CellOffset o;
                    o.dx = dx;
                    o.dy = dy;
                    o.dz = dz;
                    neighbor_offsets_.push_back(o);
                }
            }
        }
    }

    float cell_size_ = 1.f;
    float inv_cell_ = 1.f;
    float query_radius_ = 1.f;
    int stencil_radius_ = 1;
    /// Set when the neighbour stencil does not fit NeighborCellList::kCap. The
    /// stencil is then left empty and callers must not rely on it.
    bool stencil_overflow_ = false;
    BoxBounds bounds_{};
    int nx_ = 0, ny_ = 0, nz_ = 0;
    bool configured_ = false;
    std::vector<int> head_;
    std::vector<int> next_;
    std::vector<int> prev_;
    std::vector<int> atom_cell_;
    std::vector<CellOffset> neighbor_offsets_;

    // Contiguous per-cell storage (Option A). Linked list always kept in sync.
    bool use_contiguous_ = true;
    std::vector<int> cell_count_;   ///< size n_cells
    std::vector<int> cell_atoms_;   ///< size n_cells * CELL_CAPACITY
    // Packed per-cell atom coordinates — parallel to cell_atoms_.
    // cell_x_[c * CELL_CAPACITY + k] is the x coord of the k-th atom in cell c.
    std::vector<float> cell_x_;   ///< packed x; size n_cells_ * CELL_CAPACITY
    std::vector<float> cell_y_;   ///< packed y; size n_cells_ * CELL_CAPACITY
    std::vector<float> cell_z_;   ///< packed z; size n_cells_ * CELL_CAPACITY
    int peak_cell_occupancy_ = 0;

    OccupiedStencilMode occ_mode_ = OccupiedStencilMode::Off;
    std::vector<NeighborCellList> valid_stencil_;
    std::vector<NeighborCellList> occupied_stencil_;
    mutable std::uint64_t occ_maint_ns_ = 0;
    mutable std::uint64_t occ_maint_calls_ = 0;
};

/// The grid every term uses today: 48 slots per cell.
using OpenCellGrid = BasicOpenCellGrid<48>;

} // namespace mcpu
