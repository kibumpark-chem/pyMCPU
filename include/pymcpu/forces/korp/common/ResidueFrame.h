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

/// The six KORP coordinates of an ordered residue pair.
///
/// The polar angles are carried as their COSINES rather than as angles. The
/// only thing done with theta is to compare it against the ring boundaries of
/// the shell's tessellation, and acos is monotonically decreasing, so
/// comparing cosines against the cosines of those boundaries selects exactly
/// the same ring -- while removing two inverse trig calls from a loop that
/// runs a few thousand times per MC step. The reference implementation in
/// pymcpu/forcefields/korp_map.py keeps the angles, and the two agree.
struct PairCoordinates {
    double d;            ///< CA-CA distance (Angstrom)
    double cos_theta_a;  ///< cos of the polar angle in A's frame, [-1, 1]
    double psi_a;        ///< [0, 2pi]
    double cos_theta_b;  ///< [-1, 1]
    double psi_b;        ///< [0, 2pi]
    double chi;          ///< [0, 2pi]
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

/// Cosine of the angle between two vectors, clamped against round-off.
[[nodiscard]] inline double korp_cos_angle(
    const Eigen::Vector3d& a, const Eigen::Vector3d& b) noexcept
{
    const double denom = a.norm() * b.norm();
    if (denom <= 0.0) return 1.0;
    const double c = a.dot(b) / denom;
    if (c > 1.0) return 1.0;
    if (c < -1.0) return -1.0;
    return c;
}

/// atan2 without libm, for the psi and chi bins.
///
/// glibc's correctly rounded atan2 and acos were about a third of all cycles
/// in a KORP run. This is the Cephes double-precision atan (argument reduced to
/// [0, 0.66] with the pi/4 shift above that, then a 4/5 rational in x^2),
/// extended to all four quadrants. Its error on [0, 1] is about 1e-16, so a bin
/// can change only for an angle within a few ulp of a bin edge, where the old
/// acos form (ill-conditioned near 0 and pi) was already less precise.
[[nodiscard]] inline double korp_atan2(double y, double x) noexcept
{
    const double ay = std::fabs(y);
    const double ax = std::fabs(x);
    const double hi = ay > ax ? ay : ax;
    if (!(hi > 0.0)) return 0.0;   // atan2(0, 0) = 0, as libm gives for +0
    const double lo = ay > ax ? ax : ay;
    const double t = lo / hi;      // [0, 1]
    const bool big = t > 0.66;
    const double r = big ? (t - 1.0) / (t + 1.0) : t;
    const double z = r * r;
    const double p = (((-8.750608600031904122785E-1 * z
                        - 1.615753718733365076637E1) * z
                       - 7.500855792314704667340E1) * z
                      - 1.228866684490136173410E2) * z
                     - 6.485021904942025371773E1;
    const double q = ((((z + 2.485846490142306297962E1) * z
                        + 1.650270098316988542046E2) * z
                       + 4.328810604912902668951E2) * z
                      + 4.853903996359136964868E2) * z
                     + 1.945506571482613964425E2;
    double a = r + r * z * p / q;
    if (big) a += 0.25 * M_PI;
    if (ay > ax) a = 0.5 * M_PI - a;
    if (x < 0.0) a = M_PI - a;
    return y < 0.0 ? -a : a;      // -0 counts as +0, as the old psi sign test did
}

/// `dihedral3DunitN`: the dihedral of three consecutive unit vectors, with the
/// first cross product negated. The negation is upstream's, marked there as a
/// fix predating v1 of the map, so the released table was trained with it.
[[nodiscard]] inline double korp_dihedral_negated(
    const Eigen::Vector3d& ua, const Eigen::Vector3d& ub,
    const Eigen::Vector3d& uc) noexcept
{
    const Eigen::Vector3d v1 = -ua.cross(ub);
    const Eigen::Vector3d v2 = ub.cross(uc);
    const Eigen::Vector3d v3 = v1.cross(ub);
    return korp_atan2(v3.dot(v2), v1.dot(v2));
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
[[nodiscard]] inline PairCoordinates pair_coordinates(
    const ResidueFrame& a, const ResidueFrame& b) noexcept
{
    const Eigen::Vector3d rab = b.origin - a.origin;

    PairCoordinates pc;
    pc.d = rab.norm();
    const double inv_d = pc.d > 0.0 ? 1.0 / pc.d : 0.0;
    const auto clamp1 = [](double c) noexcept {
        return c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
    };
    pc.cos_theta_a = pc.d > 0.0 ? clamp1(a.vz.dot(rab) * inv_d) : 1.0;
    pc.cos_theta_b = pc.d > 0.0 ? clamp1(-b.vz.dot(rab) * inv_d) : 1.0;

    pc.psi_a = M_PI + korp_atan2(a.vy.dot(rab), a.vx.dot(rab));
    pc.psi_b = M_PI + korp_atan2(-b.vy.dot(rab), -b.vx.dot(rab));

    pc.chi = M_PI + korp_dihedral_negated(a.vz, rab * inv_d, b.vz);
    return pc;
}

} // namespace mcpu::forces
