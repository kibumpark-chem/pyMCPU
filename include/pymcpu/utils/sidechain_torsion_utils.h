#pragma once
#include "pymcpu/System.h"
#include "pymcpu/State.h"
#include "pymcpu/utils/geometry_utils.h"
#include "pymcpu/utils/numbers_compat.h"

namespace mcpu {

/// Recomputes residue r's 4 chi angles from CURRENT coordinates in `st`,
/// using System's real per-residue atom-name-resolved chi topology
/// (System::getChiAtomIndices()) instead of a positional sc_start+k
/// assumption -- the latter gets the wrong atom for any branched sidechain
/// (e.g. ILE's chi2 is CA-CB-CG1-CD1, not CA-CB-CG1-CG2, the 3rd stored
/// sidechain atom). Writes -PI_F sentinels for chi indices >= that
/// residue's torsion count. Single source of truth shared by
/// Context::computeTorsions(), Integrator.cpp's recompute_sidechain_torsion(),
/// and Context::commit_accepted_move()'s debug-only consistency check, so
/// the three call sites can never again drift out of sync with each other.
inline void compute_sidechain_chi_angles(State& st, const System& system, int r) {
    auto& chi = st.sidechain_torsions[static_cast<size_t>(r)].chi_angles;
    chi = {-mcpu::PI_F, -mcpu::PI_F, -mcpu::PI_F, -mcpu::PI_F};

    const auto& chi_table = system.getChiAtomIndices();
    // Hand-built/synthetic Systems used by some lower-level tests (see
    // tests/physics/helpers/minimal_system_builders.py) set
    // ntorsions_per_residue directly without ever calling
    // setChiAtomIndices() -- real MCPUForceField-built Systems always
    // populate both together with matching sizes. Leave sentinels rather
    // than reading out of bounds when the table is missing/undersized.
    if (r < 0 || static_cast<size_t>(r) >= chi_table.size()) return;

    const int n_chi = system.getTorsionsPerResidue()[static_cast<size_t>(r)];
    const auto& chi_idx = chi_table[static_cast<size_t>(r)];
    for (int k = 0; k < n_chi && k < 4; ++k) {
        const auto& a = chi_idx[static_cast<size_t>(k)];
        if (a[0] < 0) continue;  // no atoms resolved for this chi slot
        chi[static_cast<size_t>(k)] = GeometryUtils::calculate_dihedral(
            st.atom_pos(a[0]), st.atom_pos(a[1]), st.atom_pos(a[2]), st.atom_pos(a[3]));
    }
}

} // namespace mcpu
