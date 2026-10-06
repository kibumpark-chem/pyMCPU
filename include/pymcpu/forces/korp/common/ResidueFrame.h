#pragma once
/// KORP's per-residue frame and the six coordinates of a residue pair.
///
/// Transcribed from the reference implementation published by the Chacon lab
/// (github.com/chaconlab/Korp, sbg/src/libenergy/korpe.cpp): `frameCoord`,
/// `frames2ic` and `dihedral3DunitN`. pyMCPU's Python port of the same routines
/// reproduces the reference `korpe` binary to <= 5e-9 relative on four test
/// structures; see pymcpu/forcefields/korp_map.py and
/// tests/physics/forces/test_korp_reference_parity.py.
///
/// Arithmetic is double precision even though the engine stores coordinates as
/// float. The lookup this feeds is a STEP function of the angles, so a value
/// sitting near a bin boundary is decided by the last bit of the angle rather
/// than being smoothed over, and the cost of computing ~50 flops per pair in
/// double is small next to the table gather that follows.

#include <Eigen/Dense>
#include <cmath>

namespace mcpu::forces {

/// One residue's local orthonormal frame. Origin is CA; columns of R are
/// (vx, vy, vz) and det(R) == +1.
struct ResidueFrame {
    Eigen::Vector3d origin;
    Eigen::Vector3d vx, vy, vz;
};

/// The six KORP coordinates of an ordered residue pair, in the form the
/// binning needs.
///
/// The polar angles are carried as their COSINES rather than as angles. The
/// only thing done with theta is to compare it against the ring boundaries of
/// the shell's tessellation, and acos is monotonically decreasing, so
/// comparing cosines against the cosines of those boundaries selects exactly
/// the same ring.
///
/// The azimuths psi_a, psi_b and the dihedral chi are carried as the (y, x)
/// pair whose angle they are: each is pi + atan2(y, x). The only thing done
/// with them is to find the bin, and OrientationalPairMap does that by testing
/// the vector against the directions of the bin edges, so no inverse trig is
/// left on the per-pair path. The reference implementation in
/// pymcpu/forcefields/korp_map.py keeps the angles, and the two agree.
struct PairVectors {
    double d;              ///< CA-CA distance (Angstrom)
    double cos_theta_a;    ///< cos of the polar angle in A's frame, [-1, 1]
    double cos_theta_b;    ///< [-1, 1]
    double psi_a_y, psi_a_x;   ///< psi_a = pi + atan2(psi_a_y, psi_a_x)
    double psi_b_y, psi_b_x;   ///< psi_b = pi + atan2(psi_b_y, psi_b_x)
    double chi_y, chi_x;       ///< chi = pi + atan2(chi_y, chi_x)
};

/// Build a residue's frame from its own N, CA and C.
///
/// NOTE the `vy` cross product uses C-CA, not N-CA. KORP's published Eq. (2)
/// prints N-CA, but `frameCoord` in the released source uses C-CA, and the
/// released energy map was built by that source. The two differ by exactly a
/// negation -- since (N-CA) = lambda*vz - (C-CA), the cross products are
/// opposite -- so the paper's frame is this one rotated 180 degrees about vz.
/// Both are right-handed, so no invariant catches the difference; it shows up
/// only as both psi angles shifting by pi into a different bin. Follow the code.
[[nodiscard]] inline ResidueFrame make_residue_frame(
    const Eigen::Vector3f& n, const Eigen::Vector3f& ca, const Eigen::Vector3f& c) noexcept
{
    const Eigen::Vector3d nd = n.cast<double>();
    const Eigen::Vector3d cad = ca.cast<double>();
    const Eigen::Vector3d cd = c.cast<double>();

    const Eigen::Vector3d r12 = nd - cad;  // N  - CA
    const Eigen::Vector3d r13 = cd - cad;  // C  - CA

    ResidueFrame f;
    f.origin = cad;
    f.vz = (r12 + r13).normalized();
    f.vy = f.vz.cross(r13).normalized();
    f.vx = f.vy.cross(f.vz);
    return f;
}

/// The six coordinates for the ordered pair (a, b).
///
/// `a` must be the residue with the LOWER index: the map is not symmetric under
/// swapping the partners (upstream's own phrasing is that it is not the same to
/// have a proline before an alanine as after).
///
/// The frame axes are orthonormal (make_residue_frame), so the angles come from
/// dot products with them: cos(theta) is vz.r / |r|, and psi, upstream's angle
/// between vx and r projected into the xy-plane, signed by vy, is
/// atan2(vy.r, vx.r) shifted by pi. These agree with the projected-vector form
/// to round-off.
///
/// chi is upstream's `dihedral3DunitN` of (a.vz, r/|r|, b.vz): the dihedral of
/// three consecutive unit vectors with the first cross product negated. The
/// negation is upstream's, marked there as a fix predating v1 of the map, so
/// the released table was trained with it.
[[nodiscard]] inline PairVectors pair_vectors(
    const ResidueFrame& a, const ResidueFrame& b) noexcept
{
    const Eigen::Vector3d rab = b.origin - a.origin;

    PairVectors pv;
    pv.d = rab.norm();
    const double inv_d = pv.d > 0.0 ? 1.0 / pv.d : 0.0;
    const auto clamp1 = [](double c) noexcept {
        return c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
    };
    pv.cos_theta_a = pv.d > 0.0 ? clamp1(a.vz.dot(rab) * inv_d) : 1.0;
    pv.cos_theta_b = pv.d > 0.0 ? clamp1(-b.vz.dot(rab) * inv_d) : 1.0;

    pv.psi_a_y = a.vy.dot(rab);
    pv.psi_a_x = a.vx.dot(rab);
    pv.psi_b_y = -b.vy.dot(rab);
    pv.psi_b_x = -b.vx.dot(rab);

    const Eigen::Vector3d ub = rab * inv_d;
    const Eigen::Vector3d v1 = -a.vz.cross(ub);
    const Eigen::Vector3d v2 = ub.cross(b.vz);
    const Eigen::Vector3d v3 = v1.cross(ub);
    pv.chi_y = v3.dot(v2);
    pv.chi_x = v1.dot(v2);
    return pv;
}

} // namespace mcpu::forces
