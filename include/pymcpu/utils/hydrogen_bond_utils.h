#pragma once

#include <Eigen/Dense>
#include "pymcpu/utils/geometry_utils.h"
#include "pymcpu/utils/virtual_amide_h.h"
#include "pymcpu/State.h"
#include "pymcpu/System.h"
#include "pymcpu/utils/numbers_compat.h"

using mcpu::State;
using mcpu::BlockIndices;

struct HBondDonors {
    int residue_index;
    char secondary_structure = 'C';
    Eigen::Vector3f H;
    Eigen::Vector3f N;
    Eigen::Vector3f CA;
    Eigen::Vector3f prev_C;
    Eigen::Vector3f prev_CA;
    Eigen::Vector3f prev_N;
    Eigen::Vector3f C;
    Eigen::Vector3f next_CA;
};

struct HBondAcceptors {
    int residue_index;
    char secondary_structure = 'C';
    Eigen::Vector3f O;
    Eigen::Vector3f C;
    Eigen::Vector3f CA;
    Eigen::Vector3f N;
    Eigen::Vector3f next_N;
    Eigen::Vector3f next_CA;
    Eigen::Vector3f next_C;
    Eigen::Vector3f prev_CA;
};

constexpr float HBOND_CUTOFF_SQUARED = 2.5f * 2.5f;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX = 5.8f * 5.8f;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX_ALL = 5.5f * 5.5f;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET = 6.0f * 6.0f;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET_ALL = 5.4f * 5.4f;
constexpr float HBOND_BIN_SIZE = 20.0f * (mcpu::PI_F / 180.0f);
// legacy hbonds.h: `ang_CACA *= rad2deg; if (ang_CACA < 90) ...` -- 90 degrees in radians.
constexpr float HBOND_CACA_HELIX_SHEET_THRESHOLD = mcpu::PI_F / 2.0f;

namespace HydrogenBondUtils {
    inline bool is_close_enough_for_hbond(const Eigen::Vector3f& H, const Eigen::Vector3f& O) {
        return (H - O).squaredNorm() < HBOND_CUTOFF_SQUARED;
    }

    inline bool is_close_enough_for_hbond(const float* H, const float* O) {
        const float dx = H[0] - O[0];
        const float dy = H[1] - O[1];
        const float dz = H[2] - O[2];
        return (dx * dx + dy * dy + dz * dz) < HBOND_CUTOFF_SQUARED;
    }

    inline HBondDonors construct_donor(const State& state, const mcpu::System& sys, int residue_index) {
        const auto& block_indices = sys.getBlockIndices();
        const BlockIndices& b = block_indices[static_cast<size_t>(residue_index)];
        const BlockIndices& bp = block_indices[static_cast<size_t>(residue_index - 1)];
        const BlockIndices& bn = block_indices[static_cast<size_t>(residue_index + 1)];
        const Eigen::Vector3f N = state.atom_pos(b.bb_start);
        const Eigen::Vector3f CA = state.atom_pos(b.ca_atom());
        const Eigen::Vector3f C = state.atom_pos(b.c_atom());
        const Eigen::Vector3f prev_C = state.atom_pos(bp.c_atom());
        const Eigen::Vector3f prev_CA = state.atom_pos(bp.ca_atom());
        const Eigen::Vector3f prev_N = state.atom_pos(bp.bb_start);
        const Eigen::Vector3f next_CA = state.atom_pos(bn.ca_atom());
        Eigen::Vector3f H;
        if (b.has_explicit_h()) {
            H = state.atom_pos(b.h_start);
        } else {
            float hx, hy, hz;
            HydrogenBondUtils::compute_virtual_amide_H(
                N.x(), N.y(), N.z(),
                CA.x(), CA.y(), CA.z(),
                prev_C.x(), prev_C.y(), prev_C.z(),
                hx, hy, hz);
            H = Eigen::Vector3f(hx, hy, hz);
        }
        return {
            residue_index,
            sys.secondary_structure(residue_index),
            H,
            N,
            CA,
            prev_C,
            prev_CA,
            prev_N,
            C,
            next_CA
        };
    }

    inline HBondAcceptors construct_acceptor(const State& state, const mcpu::System& sys, int residue_index) {
        const auto& block_indices = sys.getBlockIndices();
        const BlockIndices& b = block_indices[static_cast<size_t>(residue_index)];
        const BlockIndices& bp = block_indices[static_cast<size_t>(residue_index - 1)];
        const BlockIndices& bn = block_indices[static_cast<size_t>(residue_index + 1)];
        return {
            residue_index,
            sys.secondary_structure(residue_index),
            state.atom_pos(b.o_start),
            state.atom_pos(b.c_atom()),
            state.atom_pos(b.ca_atom()),
            state.atom_pos(b.bb_start),
            state.atom_pos(bn.bb_start),
            state.atom_pos(bn.ca_atom()),
            state.atom_pos(bn.c_atom()),
            state.atom_pos(bp.ca_atom())
        };
    }

    /// legacy hbonds.h ~409-423: CA-CA distance-based orientation prefilter, plus (for
    /// res_idx_diff>4 only -- legacy hbonds.h ~456-459) the helix-secondary-structure
    /// long-range rejection. This 'H' check was previously applied unconditionally to
    /// every res_idx_diff, which is wrong (dormant until secondary structure became
    /// real: with SS hardcoded to 'C' the branch never fired) -- verified this
    /// misplacement would destroy ~80% of helical (res_idx_diff==4) H-bonds if SS were
    /// ever enabled without this fix.
    inline bool passes_ca_geometry_gate(const HBondDonors& donor, const HBondAcceptors& acceptor) {
        float ca_distance1 = (donor.prev_CA - acceptor.next_CA).squaredNorm();
        float ca_distance2 = (donor.CA - acceptor.next_CA).squaredNorm();
        float ca_distance3 = (donor.prev_CA - acceptor.CA).squaredNorm();
        float ca_distance4 = (donor.CA - acceptor.CA).squaredNorm();
        float min1 = std::min(ca_distance1, ca_distance3);
        float min2 = std::min(ca_distance2, ca_distance4);
        float min3 = std::min(min1, min2);
        int res_idx_diff = std::abs(donor.residue_index - acceptor.residue_index);
        if (res_idx_diff == 4) {
            if ( ( min1 > HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX) || ( min2 > HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX) ) {
                return false;
            } else if ( min3 > HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX_ALL ) {
                return false;
            }
        } else if (res_idx_diff > 4) {
            if ( ( min1 > HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET) || ( min2 > HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET) ) {
                return false;
            } else if ( min3 > HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET_ALL ) {
                return false;
            }
            if (donor.secondary_structure == 'H' || acceptor.secondary_structure == 'H') {
                return false;
            }
        }
        return true;
    }

    /// legacy hbonds.h ~395-432: donor phi/psi and acceptor phi/psi, in degrees,
    /// shifted +180 (legacy convention, matching TripletPotential's `phi + pi`).
    /// All eight atoms are already loaded into HBondDonors/HBondAcceptors.
    inline void donor_acceptor_rama_angles(const HBondDonors& d, const HBondAcceptors& a,
                                            float& Dphi, float& Dpsi, float& Aphi, float& Apsi) {
        constexpr float RAD2DEG = 180.0f / mcpu::PI_F;
        Dphi = GeometryUtils::calculate_dihedral(d.prev_C, d.N, d.CA, d.C) * RAD2DEG + 180.0f;
        Dpsi = GeometryUtils::calculate_dihedral(d.prev_N, d.prev_CA, d.prev_C, d.N) * RAD2DEG + 180.0f;
        Aphi = GeometryUtils::calculate_dihedral(a.C, a.next_N, a.next_CA, a.next_C) * RAD2DEG + 180.0f;
        Apsi = GeometryUtils::calculate_dihedral(a.N, a.CA, a.C, a.next_N) * RAD2DEG + 180.0f;
    }

    /// legacy hbonds.h ~433-460: hard Ramachandran-quadrant rejections (not a soft
    /// penalty -- an excluded pair contributes exactly 0, same as NO_HBOND).
    inline bool passes_ramachandran_gate(const HBondDonors& donor, const HBondAcceptors& acceptor) {
        int res_idx_diff = std::abs(donor.residue_index - acceptor.residue_index);
        float Dphi, Dpsi, Aphi, Apsi;
        donor_acceptor_rama_angles(donor, acceptor, Dphi, Dpsi, Aphi, Apsi);
        if (res_idx_diff == 4) {
            // legacy checks 'E' and 'L' as two separate if-blocks with an identical
            // body; combined here since the effect is exactly the same.
            if (donor.secondary_structure == 'E' || donor.secondary_structure == 'L' ||
                acceptor.secondary_structure == 'E' || acceptor.secondary_structure == 'L') {
                if (Dphi < 180.0f && Dpsi < 180.0f) return false;
                if (Aphi < 180.0f && Apsi < 180.0f) return false;
            }
        } else if (res_idx_diff > 4) {
            if (Dphi > 150.0f) return false;
            if (Dpsi > 30.0f && Dpsi < 210.0f) return false;
            if (Aphi > 150.0f) return false;
            if (Apsi > 30.0f && Apsi < 210.0f) return false;
            if (donor.secondary_structure == 'L' || acceptor.secondary_structure == 'L') return false;
        }
        return true;
    }

    inline bool is_hydrogen_bond(const HBondDonors& donor, const HBondAcceptors& acceptor) {
        return passes_ca_geometry_gate(donor, acceptor) && passes_ramachandran_gate(donor, acceptor);
    }

    inline void hydrogen_bond_indices(HBondDonors& donor, HBondAcceptors& acceptor, std::array<int, 7>& indices) {
        
        // Helix Sheet
        int res_idx_diff = std::abs(donor.residue_index - acceptor.residue_index);
        // legacy hbonds.h: ang_CACA = Angle(donor7-donor5, acceptor6-acceptor8), i.e. the
        // angle between each side's OWN chain axis (next_CA - prev_CA) -- not a cross-chain
        // comparison. calculate_a_CACA(CA1a,CA2a,CA1b,CA2b) computes angle(CA1a-CA1b,CA2a-CA2b),
        // so donor's/acceptor's next_CA must be the first two args and prev_CA the last two.
        float angle_caca = GeometryUtils::calculate_a_CACA(donor.next_CA, acceptor.next_CA,
                                                     donor.prev_CA, acceptor.prev_CA);
        if (res_idx_diff == 4) {
            indices[0] = 0;
        } else {
            if (angle_caca < HBOND_CACA_HELIX_SHEET_THRESHOLD) {
                indices[0] = 1;
            } else {
                indices[0] = 2;
            }
        }
        // Angles donor/acceptor residues
        float bisector_angle = GeometryUtils::calculate_a_bCA(donor.N, donor.CA, donor.C,
                                                          acceptor.N, acceptor.CA, acceptor.C);
        float interplanar_angle = GeometryUtils::calculate_a_PCA(donor.N, donor.CA, donor.C,
                                                              acceptor.N, acceptor.CA, acceptor.C);
        indices[1] = int(interplanar_angle / HBOND_BIN_SIZE);
        indices[2] = int(bisector_angle / HBOND_BIN_SIZE);

        // Angles donor/acceptor neighboring residues
        float bisector_angle_prev = GeometryUtils::calculate_a_bCA(donor.prev_N, donor.prev_CA, donor.prev_C,
                                                               acceptor.next_N, acceptor.next_CA, acceptor.next_C);
        float interplanar_angle_prev = GeometryUtils::calculate_a_PCA(donor.prev_N, donor.prev_CA, donor.prev_C,
                                                             acceptor.next_N, acceptor.next_CA, acceptor.next_C);
        indices[3] = int(interplanar_angle_prev / HBOND_BIN_SIZE);
        indices[4] = int(bisector_angle_prev / HBOND_BIN_SIZE);

        // Angles donor H/ acceptor O centered atoms
        float bisector_angle_H_O = GeometryUtils::calculate_a_bCA(donor.prev_C, donor.N, donor.CA,
                                                          acceptor.CA, acceptor.C, acceptor.next_N);
        float interplanar_angle_H_O = GeometryUtils::calculate_a_PCA(donor.prev_C, donor.N, donor.CA,
                                                              acceptor.CA, acceptor.C, acceptor.next_N);
        indices[5] = int(interplanar_angle_H_O / HBOND_BIN_SIZE); // 30 degree bins
        indices[6] = int(bisector_angle_H_O / HBOND_BIN_SIZE); // 30 degree bins
    }
}