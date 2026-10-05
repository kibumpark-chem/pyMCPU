#pragma once

#include <chrono>
#include <cstdint>

#if defined(__x86_64__) || defined(__i386__)
#include <x86intrin.h>
#define MCPU_PROFILER_RDTSC 1
#endif

namespace mcpu {

#ifdef MCPU_PROFILER_RDTSC
/// Nanoseconds per TSC tick, measured once per process against steady_clock
/// over 2 ms. The TSC runs at a constant rate on every x86 this targets
/// (constant_tsc), so one calibration holds for the process lifetime.
inline double profiler_ns_per_tick() noexcept {
    static const double ns_per_tick = [] {
        using Clock = std::chrono::steady_clock;
        const auto c0 = Clock::now();
        const std::uint64_t t0 = __rdtsc();
        while (Clock::now() - c0 < std::chrono::milliseconds(2)) {}
        const auto c1 = Clock::now();
        const std::uint64_t t1 = __rdtsc();
        const double ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(c1 - c0).count());
        return t1 > t0 ? ns / static_cast<double>(t1 - t0) : 1.0;
    }();
    return ns_per_tick;
}

/// Lightweight timer for hot-path instrumentation (no perf needed). Reads the
/// TSC (~25 cycles) instead of steady_clock through the vDSO (~20 ns): the
/// step loop runs ~10 of these per MC step, ~1-2% of a 30 us actin step.
/// The constructor calibrates before it reads the start, so no timer ever
/// spans the calibration. A reading behind the start (a migration across
/// unsynchronised sockets) adds nothing.
struct ScopedTimer {
    std::uint64_t start;
    uint64_t* dest_ns;

    explicit ScopedTimer(uint64_t* accum_ns) noexcept
        : start((static_cast<void>(profiler_ns_per_tick()), __rdtsc())),
          dest_ns(accum_ns) {}

    ~ScopedTimer() {
        if (dest_ns) {
            const std::uint64_t now = __rdtsc();
            if (now > start) {
                *dest_ns += static_cast<uint64_t>(
                    static_cast<double>(now - start) * profiler_ns_per_tick());
            }
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};
#else
/// Lightweight steady_clock timer for hot-path instrumentation (no perf needed).
struct ScopedTimer {
    using Clock = std::chrono::steady_clock;

    Clock::time_point start;
    uint64_t* dest_ns;

    explicit ScopedTimer(uint64_t* accum_ns) noexcept
        : start(Clock::now()), dest_ns(accum_ns) {}

    ~ScopedTimer() {
        if (dest_ns) {
            const auto elapsed = Clock::now() - start;
            *dest_ns += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count());
        }
    }

    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};
#endif

} // namespace mcpu

#undef MCPU_PROFILER_RDTSC
