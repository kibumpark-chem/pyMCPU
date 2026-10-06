#pragma once
#include <algorithm>
#include <cassert>
#include <vector>
#include <unordered_set>
#include <limits>
#include <cstdint>
#include <cstring>
#include <utility>

namespace mcpu {

struct ProposalPatch {
    bool is_valid = true;
    // Jacobian determinant holder for concerted rotation moves
    float log_jacobian_weight = 0.0f;

    // Cartesian bounds (for contact / grid)
    int first_affected_residue = std::numeric_limits<int>::max();
    int last_affected_residue  = std::numeric_limits<int>::min();

    // Track moved atoms (rigid/deform)
    std::vector<uint8_t> moving_atoms;
    /// Dense list of atom indices with moving_atoms[i]==1.
    /// Filled when the move is applied; stable for the proposal lifetime.
    std::vector<int> moved_indices;
    /// The half-open index ranges mark_moved_range marked, in marking order.
    /// They describe moved_indices only while every moved atom came from a
    /// range (moved_as_ranges()); a pivot marks four or so ranges of hundreds
    /// of atoms, so the per-step bookkeeping (clearing the masks, copying
    /// coordinates, the bounds test) runs over ranges instead of atoms.
    std::vector<std::pair<int, int>> moved_ranges;
    /// Number of moved_indices entries mark_moved_range appended.
    size_t ranged_count = 0;
    /// Every moved atom gets the same rotation, applied in double to the
    /// accepted float coordinates and rounded to float once
    /// (CoordsSoA::rotate_atoms). Mu relies on that to bound how far the
    /// distances such a move carries can change (MuPotential::carry_bound_A).
    /// Only the pivot sets it.
    bool is_rigid = false;

    // Internal distortions (for torsion)
    std::unordered_set<int> distorted_bb_residues;
    std::unordered_set<int> distorted_sc_residues;

    // Tracking moved atoms
    std::vector<uint8_t> sc_atom_moved;
    std::vector<uint8_t> bb_atom_moved;
    std::vector<uint8_t> o_atom_moved;
    std::vector<uint8_t> h_atom_moved;

    ProposalPatch() = default;
    explicit ProposalPatch(int num_atoms) { ensure_capacity(num_atoms); }

    /// Grow masks to ``num_atoms`` (zero-filled). No-op if already sized.
    void ensure_capacity(int num_atoms) {
        const size_t n = static_cast<size_t>(num_atoms);
        if (moving_atoms.size() == n) return;
        sc_atom_moved.assign(n, 0);
        bb_atom_moved.assign(n, 0);
        o_atom_moved.assign(n, 0);
        h_atom_moved.assign(n, 0);
        moving_atoms.assign(n, 0);
        distorted_bb_residues.reserve(16);
        distorted_sc_residues.reserve(16);
        moved_indices.reserve(64);
        moved_indices.clear();
        moved_ranges.reserve(8);
        clear_moved_ranges();
        distorted_bb_residues.clear();
        distorted_sc_residues.clear();
        reset_scalars();
    }

    /// Clear masks sparsely via ``moved_indices`` (O(n_moved), not O(N)).
    /// Call once at the start of each MC step before applying a move.
    /// A move that marked only ranges clears each range with memset.
    void reset_for_step() {
        if (moved_as_ranges()) {
            for (const auto& rg : moved_ranges) {
                const size_t lo = static_cast<size_t>(rg.first);
                const size_t len = static_cast<size_t>(rg.second - rg.first);
                std::memset(moving_atoms.data() + lo, 0, len);
                std::memset(bb_atom_moved.data() + lo, 0, len);
                std::memset(sc_atom_moved.data() + lo, 0, len);
                std::memset(o_atom_moved.data() + lo, 0, len);
                std::memset(h_atom_moved.data() + lo, 0, len);
            }
        } else {
            for (int i : moved_indices) {
                const size_t u = static_cast<size_t>(i);
                if (u >= moving_atoms.size()) continue;
                moving_atoms[u] = 0;
                bb_atom_moved[u] = 0;
                sc_atom_moved[u] = 0;
                o_atom_moved[u] = 0;
                h_atom_moved[u] = 0;
            }
        }
        moved_indices.clear();
        clear_moved_ranges();
        distorted_bb_residues.clear();
        distorted_sc_residues.clear();
        reset_scalars();
    }

    /// Mark atom i as moved (mask + dense index list). Idempotent on the mask;
    /// callers must not double-push the same index into moved_indices.
    ///
    /// The no-duplicates contract is load-bearing, not tidiness: Mu's delta
    /// counts the moved atoms each grid cell lists and skips a cell whose
    /// count equals its occupancy, so a duplicated index makes a cell that
    /// still holds an unmoved atom look fully moved and its pairs are dropped
    /// (and potentials that walk moved_indices count a duplicate twice). The
    /// engine's moves mark disjoint atom sets by construction, so the release
    /// build does not pay for a check here; debug builds assert it, and the
    /// Python bindings (the only way outside code builds a patch) validate.
    void mark_moved(int i) {
        assert(moving_atoms[static_cast<size_t>(i)] == 0 &&
               "mark_moved: atom already marked; moved_indices must not hold duplicates");
        moving_atoms[static_cast<size_t>(i)] = 1;
        moved_indices.push_back(i);
    }

    /// Mark atoms [lo, hi) as moved and set the same atoms in ``kind_flags``
    /// (one of the bb/sc/o/h masks). Equivalent to setting kind_flags[i] and
    /// calling mark_moved(i) for each i in order, with the same no-duplicates
    /// contract, but it fills the masks and moved_indices in bulk and records
    /// the range for the range-wise bookkeeping above.
    void mark_moved_range(int lo, int hi, std::vector<uint8_t>& kind_flags) {
        if (hi <= lo) return;
        const size_t ulo = static_cast<size_t>(lo);
        const size_t len = static_cast<size_t>(hi - lo);
#ifndef NDEBUG
        for (size_t u = ulo; u < ulo + len; ++u)
            assert(moving_atoms[u] == 0 &&
                   "mark_moved_range: atom already marked; moved_indices must not hold duplicates");
#endif
        std::memset(moving_atoms.data() + ulo, 1, len);
        std::memset(kind_flags.data() + ulo, 1, len);
        const size_t base = moved_indices.size();
        moved_indices.resize(base + len);
        int* out = moved_indices.data() + base;
        for (size_t k = 0; k < len; ++k) out[k] = lo + static_cast<int>(k);
        // Two ranges that touch are kept as one, so a caller that rotates a
        // span piecewise still hands the bookkeeping a single range.
        if (ranged_count == base && !moved_ranges.empty() && moved_ranges.back().second == lo)
            moved_ranges.back().second = hi;
        else
            moved_ranges.emplace_back(lo, hi);
        ranged_count += len;
    }

    /// True when every entry of moved_indices came from mark_moved_range, so
    /// moved_ranges lists exactly the moved atoms (and is non-empty).
    bool moved_as_ranges() const noexcept {
        return ranged_count != 0 && ranged_count == moved_indices.size();
    }

    /// Forget the recorded ranges; the bookkeeping then walks moved_indices.
    /// Anything that rewrites moved_indices other than through mark_moved and
    /// mark_moved_range must call this.
    void clear_moved_ranges() noexcept {
        moved_ranges.clear();
        ranged_count = 0;
    }

    void add_distorted_bb_residue(int r) { distorted_bb_residues.insert(r); }
    void add_distorted_sc_residue(int r) { distorted_sc_residues.insert(r); }

private:
    void reset_scalars() {
        is_valid = true;
        log_jacobian_weight = 0.0f;
        first_affected_residue = std::numeric_limits<int>::max();
        last_affected_residue = std::numeric_limits<int>::min();
        is_rigid = false;
    }
};

} // namespace mcpu
