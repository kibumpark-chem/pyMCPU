#pragma once
/// Dense open-boundary cell grid (NO PBC / no wrap / no minimum-image).
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/utils/CoordsSoA.h"

namespace mcpu {

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
 * Dense cell grid with open boundaries.
 * Cell index from floor((x-lo)/cell); neighbor stencil is clipped (no wrap).
 *
 * Each cell keeps its atoms in one fixed block of Cap slots (ids plus packed
 * x/y/z), newest first, so a walk over a cell is one sequential read. With
 * cells as wide as the cutoff the hard core keeps occupancy near 20 (peaks
 * of 19-24 on actin and PGK1) against Cap=48. An insert into a full cell
 * leaves the atom out and sets overflowed(); the owner then stops using the
 * grid until a rebuild fits.
 */
template <int Cap>
class BasicOpenCellGrid {
public:
    /// Fixed atoms per cell. A full cell sets overflowed().
    static constexpr int CELL_CAPACITY = Cap;

    BasicOpenCellGrid() = default;

    explicit BasicOpenCellGrid(float cell_size_in, int num_atoms_hint = 0)
        : cell_size_(cell_size_in), inv_cell_(cell_size_in > 0.f ? 1.f / cell_size_in : 0.f) {
        if (num_atoms_hint > 0) ensure_atom_capacity(num_atoms_hint);
    }

    /// Occupancy count for cell ``c``. O(1).
    [[nodiscard]] int cell_atom_count(int c) const noexcept {
        if (c < 0 || c >= static_cast<int>(cell_count_.size())) return 0;
        return cell_count_[static_cast<size_t>(c)];
    }

    float cell_size() const noexcept { return cell_size_; }
    int nx() const noexcept { return nx_; }
    int ny() const noexcept { return ny_; }
    int nz() const noexcept { return nz_; }
    std::uint64_t num_cells() const noexcept {
        return static_cast<std::uint64_t>(nx_) * ny_ * nz_;
    }
    const BoxBounds& bounds() const noexcept { return bounds_; }
    bool configured() const noexcept { return configured_; }
    int peak_cell_occupancy() const noexcept { return peak_cell_occupancy_; }
    /// An insert found its cell full since the last configure/clear. The
    /// atom was left out (atom_cell == -1), so the grid is incomplete and its
    /// owner must stop using it until a rebuild fits.
    bool overflowed() const noexcept { return overflowed_; }

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
            count = cell_count_[static_cast<size_t>(c)];
            if (count == 0) {
                ++s.empty;
            } else {
                ++s.nonempty;
                s.atoms += static_cast<std::size_t>(count);
            }
        }
        return s;
    }

    /// Atom-id span for cell ``c``. O(1).
    [[nodiscard]] std::pair<const int*, int> cell_atoms_span(int c) const noexcept {
        if (c < 0 ||
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
     * Visit only the cells that can hold an atom within `radius` and an atom
     * the pending move did not displace, WITHOUT enumerating the 27-cell
     * stencil first.
     *
     * Why: a hard-core overlap needs r < ~2.8 A, but the cells are sized for the
     * ~5.1 A Mu cutoff, so the full 27-cell stencil sweeps a box ~5.8x larger
     * in volume than the question needs. Walking the 27 offsets and culling
     * each by point-to-box distance measured slower than this: ~10 operations
     * per offset to reject 25 of 27 costs more than the distance arithmetic it
     * saves. The cost of a cell walk is the walk, not the distances.
     *
     * Here the surviving offsets are derived directly from where inside its cell
     * the query point sits: along each axis the neighbour at -1 is needed only
     * when the point is within `radius` of the low face, and +1 only when it is
     * within `radius` of the high face. With radius 2.83 A in ~5.1 A cells that
     * is 1 or 2 cells per axis, so 1-8 cells total (~3.8 on average) with no
     * per-offset rejection test at all.
     *
     * `moved_per_cell[c]` counts the atoms listed in cell c that the pending
     * move displaces. A cell whose atoms all moved (count == moved_per_cell[c],
     * which also covers an empty cell) holds nothing such a query keeps, so it
     * is not visited.
     */
    template <typename CellFunc>
    bool for_each_cell_span_within_fast_unmoved(float x, float y, float z,
                                                float radius,
                                                const std::uint8_t* moved_per_cell,
                                                CellFunc&& cell_fn) const {
        static_assert(CELL_CAPACITY <= 255,
                      "moved_per_cell holds per-cell counts in a uint8");
        if (!configured_) return true;
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

    /// The live stencil cells of the last probe a walk collected, keyed by
    /// that probe's home cell. Consecutive moved atoms of a chain mostly
    /// share a home cell, and their stencils are then the same cells: the
    /// next probe reuses the list instead of testing all 27 offsets again.
    /// Valid only while the grid and the moved-cell counts stay as they were
    /// (one moved_vs_static walk); a fresh memo matches no cell.
    struct StencilMemo {
        int ix = std::numeric_limits<int>::min();
        int iy = 0;
        int iz = 0;
        int n_live = 0;
        int live[128];
    };

    /// Same as the overload below, with no memo.
    template <typename CellFunc>
    bool for_each_neighbor_cell_span_while_unmoved(float x, float y, float z,
                                                   const std::uint8_t* moved_per_cell,
                                                   CellFunc&& cell_fn) const {
        return for_each_neighbor_cell_span_while_unmoved(
            x, y, z, moved_per_cell, nullptr, std::forward<CellFunc>(cell_fn));
    }

    /// Visit the stencil cells around (x, y, z) that hold an atom the
    /// pending move did not displace (see
    /// for_each_cell_span_within_fast_unmoved), as contiguous id and
    /// coordinate spans, until cell_fn returns false. With a memo, a
    /// probe whose home cell is the memo's reuses its list of live cells
    /// (see StencilMemo); nullptr collects the list afresh.
    template <typename CellFunc>
    bool for_each_neighbor_cell_span_while_unmoved(float x, float y, float z,
                                                   const std::uint8_t* moved_per_cell,
                                                   StencilMemo* memo,
                                                   CellFunc&& cell_fn) const {
        if (!configured_ || neighbor_offsets_.empty())
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
            int local_live[kMaxLive];
            int* const live = memo ? memo->live : local_live;
            int n_live = 0;
            if (memo && memo->ix == ix0 && memo->iy == iy0 && memo->iz == iz0) {
                n_live = memo->n_live;
                goto visit;
            }
            if (const int R = stencil_radius_;
                ix0 >= R && iy0 >= R && iz0 >= R && ix0 < nx_ - R &&
                iy0 < ny_ - R && iz0 < nz_ - R) {
                // Every stencil cell is inside the grid: step through the
                // offsets as linear cell ids, with no bounds test per cell.
                const int c0 = (ix0 * ny_ + iy0) * nz_ + iz0;
                const int* const lin = neighbor_lin_.data();
                const int n_off = static_cast<int>(neighbor_lin_.size());
#if defined(__AVX2__)
                if (R == 1 && n_off == 27 && iz0 + 2 < nz_) {
                    // The 27 cells are nine runs of three along z. Compare
                    // each run four cells at a time (the fourth, iz0 + 2, is
                    // still in the grid and is masked off), then list the
                    // live cells in offset order, as the loop below does.
                    std::uint32_t mask = 0;
                    for (int row = 0; row < 9; ++row) {
                        const int rb = c0 + lin[3 * row];
                        const __m128i cnt = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(cell_count_.data() + rb));
                        std::uint32_t mv4;
                        std::memcpy(&mv4, moved_per_cell + rb, sizeof(mv4));
                        const __m128i mv = _mm_cvtepu8_epi32(
                            _mm_cvtsi32_si128(static_cast<int>(mv4)));
                        const std::uint32_t eq = static_cast<std::uint32_t>(
                            _mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(cnt, mv))));
                        mask |= ((~eq) & 7u) << (3 * row);
                    }
                    while (mask) {
                        const int k = __builtin_ctz(mask);
                        mask &= mask - 1;
                        live[n_live++] = c0 + lin[k];
                    }
                } else
#endif
                for (int k = 0; k < n_off; ++k) {
                    const int c = c0 + lin[k];
                    live[n_live] = c;
                    n_live += (cell_count_[static_cast<size_t>(c)] !=
                               static_cast<int>(moved_per_cell[static_cast<size_t>(c)]));
                }
            } else {
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
            }
            if (memo) {
                memo->ix = ix0;
                memo->iy = iy0;
                memo->iz = iz0;
                memo->n_live = n_live;
            }
        visit:
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

    void ensure_atom_capacity(int n) {
        if (static_cast<int>(atom_cell_.size()) < n)
            atom_cell_.resize(static_cast<size_t>(n), -1);
    }

    /// Configure / resize dense cells from bounds.
    /// ``cell`` = denselist bin size; ``query_radius`` = interaction range for stencil
    /// (typically r_cut). Returns false if caps exceeded.
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
        overflowed_ = false;
        configured_ = true;
        // Clear atom membership (caller must re-insert)
        std::fill(atom_cell_.begin(), atom_cell_.end(), -1);
        precompute_neighbor_offsets_();
        return true;
    }

    int stencil_radius() const noexcept { return stencil_radius_; }
    float query_radius() const noexcept { return query_radius_; }
    std::size_t neighbor_offsets_count() const noexcept {
        return neighbor_offsets_.size();
    }

    void clear_cells_keep_shape() {
        std::fill(atom_cell_.begin(), atom_cell_.end(), -1);
        std::fill(cell_count_.begin(), cell_count_.end(), 0);
        peak_cell_occupancy_ = 0;
        overflowed_ = false;
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
        if (!configured_ || c < 0 || c >= static_cast<int>(cell_count_.size())) return;
        const int* atoms =
            cell_atoms_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
        const int count = cell_count_[static_cast<size_t>(c)];
        for (int k = 0; k < count; ++k) func(atoms[k]);
    }

    void insert(int atom_id, float px, float py, float pz) {
        assert(atom_id >= 0);
        ensure_atom_capacity(atom_id + 1);
        if (atom_cell_[static_cast<size_t>(atom_id)] != -1)
            remove(atom_id);
        const int c = cell_index(px, py, pz);
        if (c < 0) return;
        atom_cell_[static_cast<size_t>(atom_id)] = c;
        contiguous_add_front_(atom_id, c, px, py, pz);
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
        atom_cell_[static_cast<size_t>(atom_id)] = -1;
        contiguous_remove_(atom_id, c);
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
            update_packed_coords(atom_id, oc, px, py, pz);
            return;
        }
        remove(atom_id);
        if (nc >= 0) insert(atom_id, px, py, pz);
    }

    /// Update packed cell coords for atom after an in-cell position change. O(occ).
    void update_packed_coords(int atom, int cell, float px, float py, float pz) {
        const size_t base = static_cast<size_t>(cell) * CELL_CAPACITY;
        const int count = cell_count_[static_cast<size_t>(cell)];
        const int k = find_slot_(cell_atoms_.data() + base, count, atom);
        if (k < count) {
            cell_x_[base + static_cast<size_t>(k)] = px;
            cell_y_[base + static_cast<size_t>(k)] = py;
            cell_z_[base + static_cast<size_t>(k)] = pz;
            return;
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
            const int* atoms =
                cell_atoms_.data() + static_cast<size_t>(c) * CELL_CAPACITY;
            const int count = cell_count_[static_cast<size_t>(c)];
            for (int k = 0; k < count; ++k) func(atoms[k]);
        }
    }

    template <typename Func>
    void for_each_neighbor(const Eigen::Vector3f& pos, Func&& func,
                           std::uint64_t* cell_visits = nullptr,
                           float r_cut2 = -1.f) const {
        for_each_neighbor(pos.x(), pos.y(), pos.z(), std::forward<Func>(func),
                          cell_visits, r_cut2);
    }

#if !defined(NDEBUG)
    /// Verify every atom's atom_cell matches the cell that holds it. O(N).
    void verify_sync() const {
        if (!configured_) return;
        for (size_t c = 0; c < cell_count_.size(); ++c) {
            for (int k = 0; k < cell_count_[c]; ++k) {
                const int a = cell_atoms_[c * static_cast<size_t>(CELL_CAPACITY) +
                                          static_cast<size_t>(k)];
                if (atom_cell(a) != static_cast<int>(c)) {
                    std::fprintf(stderr, "SYNC ERROR cell=%zu atom=%d atom_cell=%d\n",
                                 c, a, atom_cell(a));
                }
            }
        }
    }

    /// Verify packed cell coords match CoordsSoA for all occupied cells. O(N).
    void verify_packed_coords(const CoordsSoA& coords) const {
        if (!configured_) return;
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

    /// Insert atom at the front of its cell block (newest first). O(occ).
    void contiguous_add_front_(int atom, int cell, float px, float py, float pz) {
        int& count = cell_count_[static_cast<size_t>(cell)];
        if (count >= CELL_CAPACITY) {
            // Leave the atom out and flag the grid; NeighborSystem retires
            // it until a rebuild fits (see overflowed()).
            overflowed_ = true;
            atom_cell_[static_cast<size_t>(atom)] = -1;
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
    }

    /// Remove atom from contiguous cell, preserving relative order. O(occ).
    /// First slot of a cell's packed id list that holds atom, or count when
    /// none does. Eight ids per compare: a cell's slots are CELL_CAPACITY
    /// wide, so a load of eight never leaves the cell, and lanes at or past
    /// count are masked off.
    static int find_slot_(const int* ids, int count, int atom) noexcept {
        int k = 0;
#if defined(__AVX2__)
        if constexpr (CELL_CAPACITY % 8 == 0) {
            const __m256i key = _mm256_set1_epi32(atom);
            for (; k < count; k += 8) {
                const __m256i w =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids + k));
                unsigned bits = static_cast<unsigned>(_mm256_movemask_ps(
                    _mm256_castsi256_ps(_mm256_cmpeq_epi32(w, key))));
                const int left = count - k;
                if (left < 8) bits &= (1u << left) - 1u;
                if (bits) return k + __builtin_ctz(bits);
            }
            return count;
        }
#endif
        for (; k < count; ++k)
            if (ids[k] == atom) return k;
        return count;
    }

    void contiguous_remove_(int atom, int cell) {
        const size_t base = static_cast<size_t>(cell) * CELL_CAPACITY;
        int* id_base = cell_atoms_.data() + base;
        float* x_base = cell_x_.data() + base;
        float* y_base = cell_y_.data() + base;
        float* z_base = cell_z_.data() + base;
        int& count = cell_count_[static_cast<size_t>(cell)];
        for (int k = find_slot_(id_base, count, atom); k < count; ++k) {
            if (id_base[k] == atom) {
                for (int j = k; j < count - 1; ++j) {
                    id_base[j] = id_base[j + 1];
                    x_base[j] = x_base[j + 1];
                    y_base[j] = y_base[j + 1];
                    z_base[j] = z_base[j + 1];
                }
                --count;
                return;
            }
        }
        std::fprintf(stderr,
            "ERROR: contiguous_remove: atom %d not found in cell %d\n", atom,
            cell);
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
        // The same offsets as linear cell ids, valid for a home cell at
        // least R cells from every face (see
        // for_each_neighbor_cell_span_while_unmoved).
        neighbor_lin_.clear();
        neighbor_lin_.reserve(neighbor_offsets_.size());
        for (const CellOffset& o : neighbor_offsets_)
            neighbor_lin_.push_back((o.dx * ny_ + o.dy) * nz_ + o.dz);
    }

    float cell_size_ = 1.f;
    float inv_cell_ = 1.f;
    float query_radius_ = 1.f;
    int stencil_radius_ = 1;
    BoxBounds bounds_{};
    int nx_ = 0, ny_ = 0, nz_ = 0;
    bool configured_ = false;
    std::vector<int> atom_cell_;
    std::vector<CellOffset> neighbor_offsets_;
    std::vector<int> neighbor_lin_;  ///< neighbor_offsets_ as linear cell ids

    // Per-cell storage: CELL_CAPACITY slots per cell.
    std::vector<int> cell_count_;   ///< size n_cells
    std::vector<int> cell_atoms_;   ///< size n_cells * CELL_CAPACITY
    // Packed per-cell atom coordinates — parallel to cell_atoms_.
    // cell_x_[c * CELL_CAPACITY + k] is the x coord of the k-th atom in cell c.
    std::vector<float> cell_x_;   ///< packed x; size n_cells_ * CELL_CAPACITY
    std::vector<float> cell_y_;   ///< packed y; size n_cells_ * CELL_CAPACITY
    std::vector<float> cell_z_;   ///< packed z; size n_cells_ * CELL_CAPACITY
    int peak_cell_occupancy_ = 0;
    bool overflowed_ = false;
};

/// The grid every term uses today: 48 slots per cell.
using OpenCellGrid = BasicOpenCellGrid<48>;

} // namespace mcpu
