#pragma once

#include "pymcpu/AtomPermutation.h"
#include "pymcpu/System.h"
#include "pymcpu/utils/CoordsSoA.h"

#include <utility>
#include <vector>

namespace mcpu {

class Context;

/// Compute init-only locality permutation from accepted SoA coords.
///
/// Residue-contiguous tree order:
/// - Residues sorted by Mu cell of CA (open AABB, cell = effective Mu cell),
///   tie-break residue id
/// - Within residue: [N, CA] + sidechain_DFS(from CA) + [C, O] (+ optional amide H)
/// - All atoms of each residue occupy a contiguous internal range [res_begin, res_end)
///
/// If out_blocks != nullptr, fills BlockIndices in the new internal layout.
/// ``mu_cell_size_A`` &lt;= 0 uses the 6 Å fallback Mu cell.
AtomPermutation compute_init_only_atom_permutation(
    const System& sys,
    const CoordsSoA& coords,
    float mu_cell_size_A,
    std::vector<BlockIndices>* out_blocks = nullptr);

/// Permute SoA columns: out[i] = in[perm.int_to_ext[i]].
void permute_coords_soa(CoordsSoA& coords, const AtomPermutation& perm);

/// Scatter internal SoA → external Eigen (3×N).
Eigen::Matrix3Xf scatter_internal_to_external(
    const CoordsSoA& coords_internal,
    const AtomPermutation& perm);

/// Gather external Eigen (3×N) → internal SoA.
void gather_external_to_internal(
    const Eigen::Matrix3Xf& coords_external,
    const AtomPermutation& perm,
    CoordsSoA& coords_internal);

}  // namespace mcpu
