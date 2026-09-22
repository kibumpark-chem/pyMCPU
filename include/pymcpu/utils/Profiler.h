#pragma once

#include <chrono>
#include <cstdint>

namespace mcpu {

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

} // namespace mcpu
