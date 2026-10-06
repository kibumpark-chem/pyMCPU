#include "pymcpu/AtomReorder.h"

#include "pymcpu/neighbor/NeighborSystem.h"
#include "pymcpu/neighbor/OpenCellGrid.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mcpu {

namespace {

struct ResKey {
    int cell_key = 0;
    int res_id = 0;
    bool operator<(const ResKey& o) const {
        if (cell_key != o.cell_key) return cell_key < o.cell_key;
        return res_id < o.res_id;
    }
};

inline int clamp_cell(int v, int n) {
    if (n <= 1) return 0;
    if (v < 0) return 0;
    if (v >= n) return n - 1;
    return v;
}

/// Collect SC atom external ids for residue r (legacy segment layout).
std::vector<int> collect_sc_ext(const System& sys, int r) {
    const auto& blocks = sys.getBlockIndices();
    const BlockIndices& bl = blocks[static_cast<size_t>(r)];
    std::vector<int> sc;
    if (bl.sc_start < 0) return sc;
    int count = bl.sc_count;
    if (count <= 0) {
        const int sc_end = sys.sc_segment_end();
        int next = sc_end;
        const int n_res = sys.getNumResidues();
        for (int rr = r + 1; rr < n_res; ++rr) {
            if (blocks[static_cast<size_t>(rr)].sc_start >= 0) {
                next = blocks[static_cast<size_t>(rr)].sc_start;
                break;
            }
        }
        count = std::max(0, next - bl.sc_start);
    }
    sc.reserve(static_cast<size_t>(count));
    for (int k = 0; k < count; ++k) sc.push_back(bl.sc_start + k);
    return sc;
}

/// Deterministic DFS sidechain order from CA among SC atoms (legacy ext ids).
/// Bonds: distance < 1.85 Å between CA↔SC and SC↔SC (open boundary).
std::vector<int> sidechain_dfs_from_ca(
    const CoordsSoA& coords,
    int ca_ext,
    std::vector<int> sc_ext)
{
    if (sc_ext.empty()) return sc_ext;
    std::sort(sc_ext.begin(), sc_ext.end());  // stable neighbor order by ext id

    constexpr float kBondCut = 1.85f;
    constexpr float kBondCut2 = kBondCut * kBondCut;
    const int n_sc = static_cast<int>(sc_ext.size());

    auto dist2 = [&](int a, int b) {
        const float dx = coords.x[static_cast<size_t>(a)] - coords.x[static_cast<size_t>(b)];
        const float dy = coords.y[static_cast<size_t>(a)] - coords.y[static_cast<size_t>(b)];
        const float dz = coords.z[static_cast<size_t>(a)] - coords.z[static_cast<size_t>(b)];
        return dx * dx + dy * dy + dz * dz;
    };

    // Adjacency among SC nodes (indexed 0..n_sc-1); also which touch CA.
    std::vector<std::vector<int>> adj(static_cast<size_t>(n_sc));
    std::vector<int> roots;
    for (int i = 0; i < n_sc; ++i) {
        if (dist2(ca_ext, sc_ext[static_cast<size_t>(i)]) <= kBondCut2) {
            roots.push_back(i);
        }
        for (int j = i + 1; j < n_sc; ++j) {
            if (dist2(sc_ext[static_cast<size_t>(i)],
                      sc_ext[static_cast<size_t>(j)]) <= kBondCut2) {
                adj[static_cast<size_t>(i)].push_back(j);
                adj[static_cast<size_t>(j)].push_back(i);
            }
        }
    }
    for (auto& v : adj) std::sort(v.begin(), v.end());
    std::sort(roots.begin(), roots.end());

    std::vector<int> order;
    order.reserve(static_cast<size_t>(n_sc));
    std::vector<char> seen(static_cast<size_t>(n_sc), 0);

    auto dfs = [&](auto&& self, int u) -> void {
        if (seen[static_cast<size_t>(u)]) return;
        seen[static_cast<size_t>(u)] = 1;
        order.push_back(sc_ext[static_cast<size_t>(u)]);
        for (int v : adj[static_cast<size_t>(u)]) self(self, v);
    };

    // Prefer CA-bonded roots; fall back to lowest ext id if geometry odd.
    if (roots.empty()) roots.push_back(0);
    for (int r : roots) dfs(dfs, r);
    for (int i = 0; i < n_sc; ++i) {
        if (!seen[static_cast<size_t>(i)]) dfs(dfs, i);
    }
    return order;
}

}  // namespace

// ---------------------------------------------------------------------------
// Residue-contiguous tree-order init-only permutation
//
// Within each residue (external ids → emitted in this order):
//   [N, CA] + sidechain_DFS(from CA, excluding CA–N / CA–C) + [C, O] (+ H)
// Residues sorted by Mu cell of CA, tie-break residue id.
// Result: all atoms of a residue occupy a contiguous internal range.
// ---------------------------------------------------------------------------
AtomPermutation compute_init_only_atom_permutation(
    const System& sys,
    const CoordsSoA& coords,
    float mu_cell_size_A,
    std::vector<BlockIndices>* out_blocks)
{
    const int n = sys.getNumAtoms();
    const int n_res = sys.getNumResidues();
    const auto& blocks = sys.getBlockIndices();

    const float cell =
        (mu_cell_size_A > 0.f)
            ? mu_cell_size_A
            : NeighborSystem::kMuCutoffFallbackA;
    BoxBounds b = aabb_of_coords(coords, /*margin=*/0.f);
    const float lx = std::max(b.hi[0] - b.lo[0], cell);
    const float ly = std::max(b.hi[1] - b.lo[1], cell);
    const float lz = std::max(b.hi[2] - b.lo[2], cell);
    int nx = std::max(1, static_cast<int>(std::ceil(lx / cell)));
    int ny = std::max(1, static_cast<int>(std::ceil(ly / cell)));
    int nz = std::max(1, static_cast<int>(std::ceil(lz / cell)));

    std::vector<ResKey> keys(static_cast<size_t>(n_res));
    for (int r = 0; r < n_res; ++r) {
        const int ca = blocks[static_cast<size_t>(r)].ca_atom();
        const size_t k = static_cast<size_t>(ca);
        const float x = coords.x[k], y = coords.y[k], z = coords.z[k];
        const int ix = clamp_cell(static_cast<int>(std::floor((x - b.lo[0]) / cell)), nx);
        const int iy = clamp_cell(static_cast<int>(std::floor((y - b.lo[1]) / cell)), ny);
        const int iz = clamp_cell(static_cast<int>(std::floor((z - b.lo[2]) / cell)), nz);
        keys[static_cast<size_t>(r)].cell_key = ix + nx * (iy + ny * iz);
        keys[static_cast<size_t>(r)].res_id = r;
    }
    std::sort(keys.begin(), keys.end());

    AtomPermutation perm;
    perm.int_to_ext.reserve(static_cast<size_t>(n));
    std::vector<BlockIndices> new_blocks(static_cast<size_t>(n_res));

    for (const auto& key : keys) {
        const int r = key.res_id;
        const BlockIndices& old = blocks[static_cast<size_t>(r)];
        const int n_ext = old.bb_start;
        const int ca_ext = old.ca_atom();
        const int c_ext = old.c_atom();
        const int o_ext = old.o_start;
        const int h_ext = old.h_start;

        std::vector<int> sc_order =
            sidechain_dfs_from_ca(coords, ca_ext, collect_sc_ext(sys, r));

        // Emit tree order and record new BlockIndices (internal ids = emit cursor).
        BlockIndices nb;
        nb.amide_donor = old.amide_donor;
        nb.sc_count = static_cast<int>(sc_order.size());
        nb.res_begin = static_cast<int>(perm.int_to_ext.size());

        nb.bb_start = static_cast<int>(perm.int_to_ext.size());
        perm.int_to_ext.push_back(n_ext);
        perm.int_to_ext.push_back(ca_ext);

        if (!sc_order.empty()) {
            nb.sc_start = static_cast<int>(perm.int_to_ext.size());
            for (int e : sc_order) perm.int_to_ext.push_back(e);
        } else {
            nb.sc_start = -1;
        }

        nb.c_start = static_cast<int>(perm.int_to_ext.size());
        perm.int_to_ext.push_back(c_ext);

        if (o_ext >= 0) {
            nb.o_start = static_cast<int>(perm.int_to_ext.size());
            perm.int_to_ext.push_back(o_ext);
        } else {
            nb.o_start = -1;
        }

        if (h_ext >= 0) {
            nb.h_start = static_cast<int>(perm.int_to_ext.size());
            perm.int_to_ext.push_back(h_ext);
        } else {
            nb.h_start = -1;
        }

        nb.res_end = static_cast<int>(perm.int_to_ext.size());
        new_blocks[static_cast<size_t>(r)] = nb;
    }

    if (static_cast<int>(perm.int_to_ext.size()) != n) {
        throw std::runtime_error(
            "compute_init_only_atom_permutation: atom count mismatch after tree emit");
    }
    perm.finalize_from_int_to_ext();
    perm.validate_inverses();
    if (out_blocks) *out_blocks = std::move(new_blocks);
    return perm;
}

void remap_system_topology(System& sys, const AtomPermutation& perm) {
    sys.apply_atom_permutation(perm);
}

void permute_coords_soa(CoordsSoA& coords, const AtomPermutation& perm) {
    const int n = perm.n_atoms();
    if (coords.n != n) {
        throw std::runtime_error("permute_coords_soa: size mismatch");
    }
    if (perm.is_identity()) return;
    std::vector<float> nx(static_cast<size_t>(n));
    std::vector<float> ny(static_cast<size_t>(n));
    std::vector<float> nz(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const int e = perm.int_to_ext[static_cast<size_t>(i)];
        nx[static_cast<size_t>(i)] = coords.x[static_cast<size_t>(e)];
        ny[static_cast<size_t>(i)] = coords.y[static_cast<size_t>(e)];
        nz[static_cast<size_t>(i)] = coords.z[static_cast<size_t>(e)];
    }
    coords.x.swap(nx);
    coords.y.swap(ny);
    coords.z.swap(nz);
}

Eigen::Matrix3Xf scatter_internal_to_external(
    const CoordsSoA& coords_internal,
    const AtomPermutation& perm)
{
    const int n = perm.n_atoms();
    Eigen::Matrix3Xf out(3, n);
    if (perm.is_identity()) {
        return coords_internal.as_eigen();
    }
    for (int i = 0; i < n; ++i) {
        const int e = perm.int_to_ext[static_cast<size_t>(i)];
        out(0, e) = coords_internal.x[static_cast<size_t>(i)];
        out(1, e) = coords_internal.y[static_cast<size_t>(i)];
        out(2, e) = coords_internal.z[static_cast<size_t>(i)];
    }
    return out;
}

void gather_external_to_internal(
    const Eigen::Matrix3Xf& coords_external,
    const AtomPermutation& perm,
    CoordsSoA& coords_internal)
{
    const int n = perm.n_atoms();
    if (coords_external.cols() != n) {
        throw std::runtime_error("gather_external_to_internal: size mismatch");
    }
    coords_internal.resize(n);
    if (perm.is_identity()) {
        coords_internal.load_from_eigen(coords_external);
        return;
    }
    for (int e = 0; e < n; ++e) {
        const int i = perm.ext_to_int[static_cast<size_t>(e)];
        coords_internal.x[static_cast<size_t>(i)] = coords_external(0, e);
        coords_internal.y[static_cast<size_t>(i)] = coords_external(1, e);
        coords_internal.z[static_cast<size_t>(i)] = coords_external(2, e);
    }
}

}  // namespace mcpu
