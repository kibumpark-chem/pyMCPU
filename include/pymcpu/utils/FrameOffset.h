#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace mcpu {

// ── The engine frame ──
// Coordinates are float32, and every move rounds each coordinate it changes
// at its absolute value, so the rounding noise grows with distance from the
// origin: one float step is 3.8e-6 A at 50 A but 2.4e-4 A at 4000 A. Far out,
// bond lengths random-walk, KIC's reverse check fails, and the pairs a rigid
// pivot carries reach the hard-core margin sooner. A Context therefore
// runs a structure placed far from the origin in an engine frame shifted
// next to it: engine = user - frame_offset. Every exit adds the offset back.

/// Below this largest |coordinate| (A) nothing is shifted.
inline constexpr double kFrameShiftMinA = 64.0;
/// Engine-frame largest |coordinate| (A) from which a Context notes (once per
/// process) that its coordinates are still far out: a structure several
/// hundred A across, an axis that straddles the origin and reaches far, or an
/// explicit frame offset.
inline constexpr double kFarFrameNoteA = 256.0;

/// The frame offset for user coordinates u (3 x N, A). Zero when every
/// |u| < kFrameShiftMinA, or for empty or non-finite input. Otherwise each
/// axis whose coordinates all have one sign is shifted by a whole number of A
/// close to its midpoint but at most twice its smallest |u|; an axis that
/// straddles the origin is not shifted.
///
/// For float32 input the shift is exact. The offset s on an axis is an
/// integer with the sign of every x on it, and |s| <= 2|x|, so |x - s| <= |x|.
/// Then ulp(x - s) <= ulp(x), which divides x; s, an integer (|x| < 2^24),
/// is a multiple of it too, so x - s is a float32. Every pairwise difference,
/// and every energy term computed from differences only (Mu, torsions, KORP,
/// the CA guard, the Q bias), is therefore unchanged bit for bit. Points built
/// from absolute positions (virtual amide H, aromatic ring centres) round more
/// finely in the engine frame and can differ at a cutoff or bin edge.
inline Eigen::Vector3d choose_frame_offset(const Eigen::Matrix3Xd& u) {
    Eigen::Vector3d s = Eigen::Vector3d::Zero();
    if (u.cols() == 0 || !u.allFinite()) return s;
    if (u.cwiseAbs().maxCoeff() < kFrameShiftMinA) return s;
    for (int d = 0; d < 3; ++d) {
        const double lo = u.row(d).minCoeff();
        const double hi = u.row(d).maxCoeff();
        if (lo > 0.0) {
            s[d] = std::min(std::floor(0.5 * (lo + hi) + 0.5), std::floor(2.0 * lo));
        } else if (hi < 0.0) {
            s[d] = std::max(std::ceil(0.5 * (lo + hi) - 0.5), std::ceil(2.0 * hi));
        }
        s[d] += 0.0;  // -0.0 (ceil of a small negative) reads as 0
    }
    return s;
}

}  // namespace mcpu
