#pragma once

// Portable π constants. Prefer <numbers> when the standard library provides it;
// otherwise use numeric fallbacks so older toolchains (e.g. GCC 8) still build.

#if defined(__has_include)
#  if __has_include(<numbers>) && (__cplusplus >= 202002L)
#    include <numbers>
#    define MCPU_HAS_STD_NUMBERS 1
#  endif
#endif

#if defined(MCPU_HAS_STD_NUMBERS)
namespace mcpu {
inline constexpr double PI = std::numbers::pi;
inline constexpr float PI_F = std::numbers::pi_v<float>;
}  // namespace mcpu
#else
namespace mcpu {
inline constexpr double PI = 3.14159265358979323846;
inline constexpr float PI_F = 3.14159265358979323846f;
}  // namespace mcpu
#endif
