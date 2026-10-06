#pragma once
/// The one rounding of a squared pair distance that every cutoff test uses.

#include <cmath>

namespace mcpu {

/// Squared length of (dx, dy, dz), rounded the way the vector pair search
/// (span_hits8 in SpanMask.h) rounds it: fma(dz, dz, fma(dx, dx, dy * dy))
/// with FMA and the default contraction, the unfused sum otherwise
/// (MCPU_FP_CONTRACT off/on, or no FMA). Written out because the compiler is
/// free to contract dx * dx + dy * dy + dz * dz either way at each call site,
/// and a pair on a cutoff then lands on different sides of it in different
/// code paths (an incremental delta and the full energy). Every scalar test
/// of a pair distance against a cutoff goes through here.
[[nodiscard, gnu::always_inline]] inline float pair_r2(
        float dx, float dy, float dz) noexcept {
#if defined(__FMA__) && !defined(MCPU_FP_CONTRACT_OFF)
    return std::fma(dz, dz, std::fma(dx, dx, dy * dy));
#else
    return dx * dx + dy * dy + dz * dz;
#endif
}

/// pair_r2 of a 3-vector (an Eigen::Vector3f or a difference of two).
template <class Vec3>
[[nodiscard, gnu::always_inline]] inline float pair_r2(const Vec3& d) noexcept {
    return pair_r2(d[0], d[1], d[2]);
}

}  // namespace mcpu
