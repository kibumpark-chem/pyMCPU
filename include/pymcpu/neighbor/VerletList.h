#pragma once
/// Optional Verlet/skin pair lists for KIC/SC (no PBC).
/// Pivot accepts track displacement (default); legacy force-invalidate is opt-in.
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pymcpu/CellList.h"
#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/CoordView.h"

namespace mcpu {

class VerletList {
public:
    bool dirty = true;
    /// Why the list was marked dirty (last writer wins; used at next rebuild).
    enum class DirtyCause : std::uint8_t {
        None = 0,
        PivotAccept = 1,
        DispExceeded = 2,
        AutoExpand = 3,
        DirtyFlag = 4,  // generic / init / unspecified
    };
    DirtyCause dirty_cause = DirtyCause::DirtyFlag;
    float skin = 0.f;
    float r_cut = 0.f;
    float r_list_sq = 0.f;

    std::vector<float> disp_acc;          // accepted displacement since rebuild
    std::vector<Eigen::Vector3f> ref_pos; // positions at last rebuild
    // CSR (undirected: each edge stored in both adjacency lists)
    std::vector<int> offsets; // size n+1
    std::vector<int> neighbors;

    /// Scratch for CSR rebuild — capacity preserved across rebuilds.
    std::vector<std::uint32_t> rebuild_degree;
    std::vector<std::uint32_t> rebuild_cursor;
    /// Flat unique-edge list from a single cell enumeration (j > i).
    std::vector<int> rebuild_edge_i;
    std::vector<int> rebuild_edge_j;

    /// Realloc growth counters (lifetime of this VerletList).
    std::uint64_t num_neigh_reallocs = 0;
    std::uint64_t num_offsets_reallocs = 0;

    void clear() {
        dirty = true;
        dirty_cause = DirtyCause::DirtyFlag;
        disp_acc.clear();
        ref_pos.clear();
        offsets.clear();
        neighbors.clear();
        // Keep scratch capacity; zero logical size only.
        rebuild_degree.clear();
        rebuild_cursor.clear();
        rebuild_edge_i.clear();
        rebuild_edge_j.clear();
    }

    void invalidate(DirtyCause cause = DirtyCause::DirtyFlag) {
        dirty = true;
        dirty_cause = cause;
    }

    bool global_valid() const {
        if (dirty || skin <= 0.f) return false;
        const float lim = 0.5f * skin;
        for (float d : disp_acc) {
            if (d > lim) return false;
        }
        return true;
    }

    /// Trial may use Verlet iff global valid and for every moved i: disp_acc[i]+disp_trial[i] <= skin/2.
    bool trial_usable(const std::vector<int>& moved,
                      const CoordsSoA& accepted,
                      const CoordsSoA& trial) const {
        if (!global_valid()) return false;
        const float lim = 0.5f * skin;
        for (int i : moved) {
            if (i < 0 || i >= static_cast<int>(disp_acc.size())) return false;
            const size_t k = static_cast<size_t>(i);
            const float dx = trial.x[k] - accepted.x[k];
            const float dy = trial.y[k] - accepted.y[k];
            const float dz = trial.z[k] - accepted.z[k];
            const float dt = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (disp_acc[k] + dt > lim) return false;
        }
        return true;
    }

    void accumulate_accept(const std::vector<int>& moved,
                           const CoordsSoA& old_c,
                           const CoordsSoA& new_c) {
        for (int i : moved) {
            if (i < 0 || i >= static_cast<int>(disp_acc.size())) continue;
            const size_t k = static_cast<size_t>(i);
            const float dx = new_c.x[k] - old_c.x[k];
            const float dy = new_c.y[k] - old_c.y[k];
            const float dz = new_c.z[k] - old_c.z[k];
            disp_acc[k] += std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        if (!global_valid()) {
            dirty = true;
            dirty_cause = DirtyCause::DispExceeded;
        }
    }

    template <typename Func>
    void for_each_neighbor_of(int i, Func&& func) const {
        if (dirty || i < 0 || i + 1 >= static_cast<int>(offsets.size())) return;
        const int a = offsets[static_cast<size_t>(i)];
        const int b = offsets[static_cast<size_t>(i) + 1];
        for (int k = a; k < b; ++k) func(neighbors[static_cast<size_t>(k)]);
    }
};

/// Build undirected Verlet CSR with a single cell enumeration + CSR fill.
///
/// Hot-path strategy (avoids per-rebuild ``vector<vector<int>>``):
/// 1) One mu-grid pass collecting unique edges (j>i, dist2<=r_list2) into flat
///    reusable ``rebuild_edge_i/j`` buffers while counting degrees.
/// 2) Prefix-sum offsets, resize/reuse ``neighbors``, fill both directions.
///
/// Only atoms in [0, atom_end) are indexed (Mu occupancy: BB+O+SC). Arrays are
/// sized to coords.n so arbitrary moved indices remain addressable (empty lists).
inline void rebuild_verlet_undirected(VerletList& vl,
                                      const CellListMC& cells,
                                      const CoordsSoA& coords,
                                      float r_cut, float skin,
                                      int atom_end = -1) {
    vl.r_cut = r_cut;
    vl.skin = skin;
    const float r_list_sq = (r_cut + skin) * (r_cut + skin);
    vl.r_list_sq = r_list_sq;
    const int n_all = coords.n;
    const int n_mu = (atom_end < 0) ? n_all : std::min(atom_end, n_all);
    const size_t n_all_u = static_cast<size_t>(n_all);

    vl.disp_acc.assign(n_all_u, 0.f);
    vl.ref_pos.resize(n_all_u);
    const CoordView cv(coords);
    for (int i = 0; i < n_all; ++i) {
        const size_t k = static_cast<size_t>(i);
        vl.ref_pos[k] = Eigen::Vector3f(cv.x(i), cv.y(i), cv.z(i));
    }

    // ---- Scratch: reuse capacity ----
    if (vl.rebuild_degree.size() != n_all_u) {
        vl.rebuild_degree.assign(n_all_u, 0u);
    } else {
        std::fill(vl.rebuild_degree.begin(), vl.rebuild_degree.end(), 0u);
    }
    if (vl.rebuild_cursor.size() != n_all_u) {
        vl.rebuild_cursor.resize(n_all_u);
    }
    vl.rebuild_edge_i.clear();
    vl.rebuild_edge_j.clear();
    // Heuristic reserve from prior edge count (or ~N*degree/4 undirected).
    if (vl.rebuild_edge_i.capacity() < n_all_u * 16u) {
        vl.rebuild_edge_i.reserve(n_all_u * 32u);
        vl.rebuild_edge_j.reserve(n_all_u * 32u);
    }

    // ---- Single cell enumeration: unique edges + degrees ----
    for (int i = 0; i < n_mu; ++i) {
        const float px = cv.x(i), py = cv.y(i), pz = cv.z(i);
        cells.for_each_neighbor(px, py, pz, [&](int j) {
            if (j <= i || j >= n_mu) return;
            if (cv.dist2(i, j) <= r_list_sq) {
                vl.rebuild_edge_i.push_back(i);
                vl.rebuild_edge_j.push_back(j);
                ++vl.rebuild_degree[static_cast<size_t>(i)];
                ++vl.rebuild_degree[static_cast<size_t>(j)];
            }
        });
    }

    // ---- Prefix sum → offsets ----
    const size_t old_o_cap = vl.offsets.capacity();
    vl.offsets.resize(n_all_u + 1);
    if (vl.offsets.capacity() > old_o_cap) ++vl.num_offsets_reallocs;

    vl.offsets[0] = 0;
    for (int i = 0; i < n_all; ++i) {
        const size_t k = static_cast<size_t>(i);
        vl.offsets[k + 1] =
            vl.offsets[k] + static_cast<int>(vl.rebuild_degree[k]);
        vl.rebuild_cursor[k] = static_cast<std::uint32_t>(vl.offsets[k]);
    }
    const int total_directed = vl.offsets[n_all_u];

    // ---- Grow neighbors buffer (never shrink) ----
    const size_t old_n_cap = vl.neighbors.capacity();
    const size_t need = static_cast<size_t>(total_directed);
    if (vl.neighbors.capacity() < need) {
        const size_t reserve_n = need + need / 20u;
        vl.neighbors.reserve(reserve_n > need ? reserve_n : need);
    }
    if (vl.neighbors.capacity() > old_n_cap) ++vl.num_neigh_reallocs;
    vl.neighbors.resize(need);


    // ---- Fill undirected CSR from flat edge list (no second grid scan) ----
    const size_t n_edges = vl.rebuild_edge_i.size();
    for (size_t e = 0; e < n_edges; ++e) {
        const int i = vl.rebuild_edge_i[e];
        const int j = vl.rebuild_edge_j[e];
        const std::uint32_t wi = vl.rebuild_cursor[static_cast<size_t>(i)]++;
        const std::uint32_t wj = vl.rebuild_cursor[static_cast<size_t>(j)]++;
        vl.neighbors[static_cast<size_t>(wi)] = j;
        vl.neighbors[static_cast<size_t>(wj)] = i;
    }

#if !defined(NDEBUG)
    for (int i = 0; i < n_all; ++i) {
        const size_t k = static_cast<size_t>(i);
        if (static_cast<int>(vl.rebuild_cursor[k]) != vl.offsets[k + 1]) {
            throw std::runtime_error(
                "rebuild_verlet_undirected: cursor/offset mismatch after fill");
        }
    }
#endif

    vl.dirty = false;
    vl.dirty_cause = VerletList::DirtyCause::None;
}

} // namespace mcpu
