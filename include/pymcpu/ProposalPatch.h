#pragma once
#include <algorithm>
#include <cassert>
#include <vector>
#include <unordered_set>
#include <limits>
#include <cstdint>

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
        distorted_bb_residues.clear();
        distorted_sc_residues.clear();
        reset_scalars();
    }

    /// Clear masks sparsely via ``moved_indices`` (O(n_moved), not O(N)).
    /// Call once at the start of each MC step before applying a move.
    void reset_for_step() {
        for (int i : moved_indices) {
            const size_t u = static_cast<size_t>(i);
            if (u >= moving_atoms.size()) continue;
            moving_atoms[u] = 0;
            bb_atom_moved[u] = 0;
            sc_atom_moved[u] = 0;
            o_atom_moved[u] = 0;
            h_atom_moved[u] = 0;
        }
        moved_indices.clear();
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
