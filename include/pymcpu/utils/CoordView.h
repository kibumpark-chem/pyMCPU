#pragma once
/// Fast coordinate access for Mu / neighbor hot loops (SoA scalar loads, no Eigen temporaries).

#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/pair_r2.h"
#include <cmath>
#include <cstddef>

namespace mcpu {

struct CoordView {
    const float* x_ = nullptr;
    const float* y_ = nullptr;
    const float* z_ = nullptr;
    int n_ = 0;

    CoordView() = default;

    explicit CoordView(const CoordsSoA& c) noexcept
        : x_(c.x.data())
        , y_(c.y.data())
        , z_(c.z.data())
        , n_(c.n)
    {}

    CoordView(const float* x, const float* y, const float* z, int num_atoms) noexcept
        : x_(x)
        , y_(y)
        , z_(z)
        , n_(num_atoms)
    {}

    [[nodiscard]] inline float x(int i) const noexcept { return x_[static_cast<size_t>(i)]; }
    [[nodiscard]] inline float y(int i) const noexcept { return y_[static_cast<size_t>(i)]; }
    [[nodiscard]] inline float z(int i) const noexcept { return z_[static_cast<size_t>(i)]; }

    inline void load_xyz(int i, float out[3]) const noexcept {
        const size_t k = static_cast<size_t>(i);
        out[0] = x_[k];
        out[1] = y_[k];
        out[2] = z_[k];
    }

    [[nodiscard]] static inline float dist2(const float* a, const float* b) noexcept {
        const float dx = a[0] - b[0];
        const float dy = a[1] - b[1];
        const float dz = a[2] - b[2];
        return pair_r2(dx, dy, dz);
    }

    [[nodiscard]] inline float dist2(int i, int j) const noexcept {
        const float dx = x_[static_cast<size_t>(i)] - x_[static_cast<size_t>(j)];
        const float dy = y_[static_cast<size_t>(i)] - y_[static_cast<size_t>(j)];
        const float dz = z_[static_cast<size_t>(i)] - z_[static_cast<size_t>(j)];
        return pair_r2(dx, dy, dz);
    }

    [[nodiscard]] inline float dist2(int i, const CoordView& other, int j) const noexcept {
        const float dx = x_[static_cast<size_t>(i)] - other.x_[static_cast<size_t>(j)];
        const float dy = y_[static_cast<size_t>(i)] - other.y_[static_cast<size_t>(j)];
        const float dz = z_[static_cast<size_t>(i)] - other.z_[static_cast<size_t>(j)];
        return pair_r2(dx, dy, dz);
    }
    [[nodiscard]] static inline float dist2(const float* a, float bx, float by, float bz) noexcept {
        const float dx = a[0] - bx;
        const float dy = a[1] - by;
        const float dz = a[2] - bz;
        return pair_r2(dx, dy, dz);
    }
};

} // namespace mcpu
