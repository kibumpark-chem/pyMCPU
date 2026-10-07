#pragma once
#include <Eigen/Dense>
#include <cassert>
#include <cstdint>
#include <vector>

#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/neighbor/OpenCellGrid.h"

namespace mcpu {

/// Neighbor-list facade. Production backend is dense OpenCellGrid (no PBC).
class CellListMC {
private:
    float cutoff_;
    OpenCellGrid grid_;

public:
    CellListMC(float cutoff, int num_atoms)
        : cutoff_(cutoff), grid_(cutoff, num_atoms) {}

    float cutoff() const noexcept { return cutoff_; }
    /// Update denselist query cutoff (Å) before configure(). O(1).
    void set_cutoff(float cutoff) noexcept {
        if (cutoff > 0.f) cutoff_ = cutoff;
    }
    OpenCellGrid& grid() noexcept { return grid_; }
    const OpenCellGrid& grid() const noexcept { return grid_; }

    /// Configure dense cells from AABB bounds.
    /// Default: cell_size = query = cutoff (HB / baseline Mu).
    /// ``cell_size_override`` &gt; 0 sets the cell edge apart from the query
    /// radius (the stencil then spans ceil(query / cell) cells).
    bool configure(const BoxBounds& bounds, std::uint64_t max_cells,
                   float cell_size_override = -1.f) {
        const float query = cutoff_;
        const float cell =
            (cell_size_override > 0.f) ? cell_size_override : query;
        return grid_.configure(bounds, cell, max_cells, query);
    }

    void ensure_atom_capacity(int n) { grid_.ensure_atom_capacity(n); }

    void insert(int atom_id, const CoordsSoA& coords) {
        grid_.insert(atom_id, coords);
    }

    void insert(int atom_id, float x, float y, float z) {
        grid_.insert(atom_id, x, y, z);
    }

    void insert(int atom_id, const Eigen::Vector3f& pos) {
        grid_.insert(atom_id, pos);
    }

    void remove(int atom_id) { grid_.remove(atom_id); }

    inline void update_position(int atom_id, const CoordsSoA& coords) {
        grid_.update_position(atom_id, coords);
    }

    inline void update_position(int atom_id, const Eigen::Vector3f& new_pos) {
        grid_.update_position(atom_id, new_pos);
    }

    /// Clear membership; keeps shape if already configured.
    inline void reset(int num_atoms) {
        grid_.ensure_atom_capacity(num_atoms);
        if (grid_.configured())
            grid_.clear_cells_keep_shape();
    }

    template <typename Func>
    inline void for_each_neighbor(const Eigen::Vector3f& pos, Func&& func,
                                  std::uint64_t* cell_visits = nullptr,
                                  float r_cut2 = -1.f) const {
        grid_.for_each_neighbor(pos, std::forward<Func>(func), cell_visits,
                                r_cut2);
    }

    template <typename Func>
    inline void for_each_neighbor(float x, float y, float z, Func&& func,
                                  std::uint64_t* cell_visits = nullptr,
                                  float r_cut2 = -1.f) const {
        grid_.for_each_neighbor(x, y, z, std::forward<Func>(func), cell_visits,
                                r_cut2);
    }


};
} // namespace mcpu
