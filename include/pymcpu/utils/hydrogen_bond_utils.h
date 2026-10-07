#pragma once

#include <algorithm>
#include <array>
#include <Eigen/Dense>
#include "pymcpu/utils/geometry_utils.h"
#include "pymcpu/utils/virtual_amide_h.h"
#include "pymcpu/State.h"
#include "pymcpu/System.h"
#include "pymcpu/utils/numbers_compat.h"
#include "pymcpu/utils/pair_r2.h"

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

constexpr float HBOND_CUTOFF = 2.5f;
constexpr float HBOND_CUTOFF_SQUARED = HBOND_CUTOFF * HBOND_CUTOFF;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX = 5.8f * 5.8f;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX_ALL = 5.5f * 5.5f;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET = 6.0f * 6.0f;
constexpr float HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET_ALL = 5.4f * 5.4f;
constexpr float HBOND_BIN_SIZE = 20.0f * (mcpu::PI_F / 180.0f);
// legacy hbonds.h: `ang_CACA *= rad2deg; if (ang_CACA < 90) ...` -- 90 degrees in radians.
constexpr float HBOND_CACA_HELIX_SHEET_THRESHOLD = mcpu::PI_F / 2.0f;

namespace HydrogenBondUtils {
    inline bool is_close_enough_for_hbond(const Eigen::Vector3f& H, const Eigen::Vector3f& O) {
        return mcpu::pair_r2(H - O) < HBOND_CUTOFF_SQUARED;
    }

    inline bool is_close_enough_for_hbond(const float* H, const float* O) {
        const float dx = H[0] - O[0];
        const float dy = H[1] - O[1];
        const float dz = H[2] - O[2];
        return mcpu::pair_r2(dx, dy, dz) < HBOND_CUTOFF_SQUARED;
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
        float ca_distance1 = mcpu::pair_r2(donor.prev_CA - acceptor.next_CA);
        float ca_distance2 = mcpu::pair_r2(donor.CA - acceptor.next_CA);
        float ca_distance3 = mcpu::pair_r2(donor.prev_CA - acceptor.CA);
        float ca_distance4 = mcpu::pair_r2(donor.CA - acceptor.CA);
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

    /// legacy hbonds.h ~433-460: hard Ramachandran-quadrant rejections (not a soft
    /// penalty -- an excluded pair contributes exactly 0, same as NO_HBOND).
    inline bool passes_ramachandran_gate(const HBondDonors& donor, const HBondAcceptors& acceptor) {
        // legacy hbonds.h ~395-432: donor phi/psi and acceptor phi/psi, in
        // degrees, shifted +180 (legacy convention, matching TripletPotential's
        // `phi + pi`). Each angle is computed only when a test reads it, in the
        // order the tests run: most pairs leave after one or two atan2 calls
        // instead of four.
        constexpr float RAD2DEG = 180.0f / mcpu::PI_F;
        const HBondDonors& d = donor;
        const HBondAcceptors& a = acceptor;
        auto Dphi = [&] { return GeometryUtils::calculate_dihedral(d.prev_C, d.N, d.CA, d.C) * RAD2DEG + 180.0f; };
        auto Dpsi = [&] { return GeometryUtils::calculate_dihedral(d.prev_N, d.prev_CA, d.prev_C, d.N) * RAD2DEG + 180.0f; };
        auto Aphi = [&] { return GeometryUtils::calculate_dihedral(a.C, a.next_N, a.next_CA, a.next_C) * RAD2DEG + 180.0f; };
        auto Apsi = [&] { return GeometryUtils::calculate_dihedral(a.N, a.CA, a.C, a.next_N) * RAD2DEG + 180.0f; };
        int res_idx_diff = std::abs(donor.residue_index - acceptor.residue_index);
        if (res_idx_diff == 4) {
            // legacy checks 'E' and 'L' as two separate if-blocks with an identical
            // body; combined here since the effect is exactly the same.
            if (donor.secondary_structure == 'E' || donor.secondary_structure == 'L' ||
                acceptor.secondary_structure == 'E' || acceptor.secondary_structure == 'L') {
                if (Dphi() < 180.0f && Dpsi() < 180.0f) return false;
                if (Aphi() < 180.0f && Apsi() < 180.0f) return false;
            }
        } else if (res_idx_diff > 4) {
            if (donor.secondary_structure == 'L' || acceptor.secondary_structure == 'L') return false;
            if (Dphi() > 150.0f) return false;
            const float dpsi = Dpsi();
            if (dpsi > 30.0f && dpsi < 210.0f) return false;
            if (Aphi() > 150.0f) return false;
            const float apsi = Apsi();
            if (apsi > 30.0f && apsi < 210.0f) return false;
        }
        return true;
    }

    /// The angle bins' edges, cos(20 deg * k) for k = 1..9, and the band
    /// around each edge in which angle_bin_from_cos bins by acos instead.
    inline constexpr float kBinEdgeCos[9] = {
        0.93969262f, 0.76604444f, 0.5f, 0.17364818f, -0.17364818f,
        -0.5f, -0.76604444f, -0.93969262f, -1.0f};
    inline constexpr float kBinEdgeBand = 1e-4f;
    /// The band around 0 in which hydrogen_bond_indices decides the CA-CA
    /// orientation by acos instead of by the sign of its cosine.
    inline constexpr float kCacaSignBand = 1e-4f;

    /// int(angle / HBOND_BIN_SIZE) for angle = acos(c), without the acos when c is
    /// clear of every bin edge. Away from an edge the bin follows from comparing c
    /// with cos(k * 20 deg); within 1e-4 of one (where acos rounding could decide
    /// the bin) it falls back to the acos, so the result is always identical to
    /// int(std::acos(c) / HBOND_BIN_SIZE).
    inline int angle_bin_from_cos(float c) {
        int bin = 0;
        bool near_edge = false;
        for (float e : kBinEdgeCos) {
            bin += (c < e) ? 1 : 0;
            near_edge |= std::abs(c - e) < kBinEdgeBand;
        }
        if (near_edge) return int(std::acos(c) / HBOND_BIN_SIZE);
        return bin;
    }

    /// With `cosines`, also stores the quantities it decides by: [0] the
    /// CA-CA orientation cos (pairs more than 4 apart), [1..6] the cos that
    /// indices[1..6] bin.
    inline void hydrogen_bond_indices(HBondDonors& donor, HBondAcceptors& acceptor, std::array<int, 7>& indices,
                                      float* cosines = nullptr) {
        float c[7] = {};
        // Helix Sheet
        int res_idx_diff = std::abs(donor.residue_index - acceptor.residue_index);
        // legacy hbonds.h: ang_CACA = Angle(donor7-donor5, acceptor6-acceptor8), i.e. the
        // angle between each side's OWN chain axis (next_CA - prev_CA) -- not a cross-chain
        // comparison. calculate_a_CACA(CA1a,CA2a,CA1b,CA2b) computes angle(CA1a-CA1b,CA2a-CA2b),
        // so donor's/acceptor's next_CA must be the first two args and prev_CA the last two.
        if (res_idx_diff == 4) {
            indices[0] = 0;
        } else {
            // angle_caca < 90 deg, decided on its cosine; acos only when the cosine
            // is too close to 0 for its sign to settle the comparison.
            const float c_caca = GeometryUtils::calculate_a_CACA_cos(donor.next_CA, acceptor.next_CA,
                                                                     donor.prev_CA, acceptor.prev_CA);
            const bool helix_like = std::abs(c_caca) < kCacaSignBand
                ? std::acos(c_caca) < HBOND_CACA_HELIX_SHEET_THRESHOLD
                : c_caca > 0.0f;
            indices[0] = helix_like ? 1 : 2;
            c[0] = c_caca;
        }
        // Angles donor/acceptor residues (20 degree bins, see angle_bin_from_cos)
        c[1] = GeometryUtils::calculate_a_PCA_cos(
            donor.N, donor.CA, donor.C, acceptor.N, acceptor.CA, acceptor.C);
        c[2] = GeometryUtils::calculate_a_bCA_cos(
            donor.N, donor.CA, donor.C, acceptor.N, acceptor.CA, acceptor.C);
        // Angles donor/acceptor neighboring residues
        c[3] = GeometryUtils::calculate_a_PCA_cos(
            donor.prev_N, donor.prev_CA, donor.prev_C, acceptor.next_N, acceptor.next_CA, acceptor.next_C);
        c[4] = GeometryUtils::calculate_a_bCA_cos(
            donor.prev_N, donor.prev_CA, donor.prev_C, acceptor.next_N, acceptor.next_CA, acceptor.next_C);
        // Angles donor H/ acceptor O centered atoms
        c[5] = GeometryUtils::calculate_a_PCA_cos(
            donor.prev_C, donor.N, donor.CA, acceptor.CA, acceptor.C, acceptor.next_N);
        c[6] = GeometryUtils::calculate_a_bCA_cos(
            donor.prev_C, donor.N, donor.CA, acceptor.CA, acceptor.C, acceptor.next_N);
        for (int k = 1; k < 7; ++k) indices[k] = angle_bin_from_cos(c[k]);
        if (cosines) std::copy(c, c + 7, cosines);
    }
}