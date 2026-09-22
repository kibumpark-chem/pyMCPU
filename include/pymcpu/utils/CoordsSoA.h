#pragma once
/// Structure-of-arrays coordinate storage (x[], y[], z[]) for cache-friendly hot loops.

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace mcpu {

struct CoordsSoA {
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;
    int n = 0;

    CoordsSoA() = default;

    explicit CoordsSoA(int num_atoms) { resize(num_atoms); }

    void resize(int num_atoms) {
        n = num_atoms;
        const size_t sz = static_cast<size_t>(num_atoms);
        x.resize(sz, 0.f);
        y.resize(sz, 0.f);
        z.resize(sz, 0.f);
    }

    void zero() {
        std::fill(x.begin(), x.end(), 0.f);
        std::fill(y.begin(), y.end(), 0.f);
        std::fill(z.begin(), z.end(), 0.f);
    }

    void set_xyz(int i, float xi, float yi, float zi) noexcept {
        x[static_cast<size_t>(i)] = xi;
        y[static_cast<size_t>(i)] = yi;
        z[static_cast<size_t>(i)] = zi;
    }

    void set_atom(int i, const Eigen::Vector3f& p) noexcept {
        set_xyz(i, p.x(), p.y(), p.z());
    }

    [[nodiscard]] Eigen::Vector3f atom(int i) const {
        return {x[static_cast<size_t>(i)], y[static_cast<size_t>(i)], z[static_cast<size_t>(i)]};
    }

    void copy_atom_from(const CoordsSoA& src, int dst, int src_idx) {
        x[static_cast<size_t>(dst)] = src.x[static_cast<size_t>(src_idx)];
        y[static_cast<size_t>(dst)] = src.y[static_cast<size_t>(src_idx)];
        z[static_cast<size_t>(dst)] = src.z[static_cast<size_t>(src_idx)];
    }

    void copy_all_from(const CoordsSoA& src) {
        n = src.n;
        x = src.x;
        y = src.y;
        z = src.z;
    }

    void load_from_eigen(const Eigen::Matrix3Xf& m) {
        resize(static_cast<int>(m.cols()));
        for (int i = 0; i < n; ++i) {
            x[static_cast<size_t>(i)] = m(0, i);
            y[static_cast<size_t>(i)] = m(1, i);
            z[static_cast<size_t>(i)] = m(2, i);
        }
    }

    [[nodiscard]] Eigen::Matrix3Xf as_eigen() const {
        Eigen::Matrix3Xf m(3, n);
        for (int i = 0; i < n; ++i) {
            m(0, i) = x[static_cast<size_t>(i)];
            m(1, i) = y[static_cast<size_t>(i)];
            m(2, i) = z[static_cast<size_t>(i)];
        }
        return m;
    }

    void recenter() {
        if (n <= 0) return;
        float cx = 0.f, cy = 0.f, cz = 0.f;
        const float inv = 1.f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) {
            cx += x[static_cast<size_t>(i)];
            cy += y[static_cast<size_t>(i)];
            cz += z[static_cast<size_t>(i)];
        }
        cx *= inv;
        cy *= inv;
        cz *= inv;
        for (int i = 0; i < n; ++i) {
            x[static_cast<size_t>(i)] -= cx;
            y[static_cast<size_t>(i)] -= cy;
            z[static_cast<size_t>(i)] -= cz;
        }
    }

    void rotate_atoms(int start, int end, const Eigen::Matrix3f& R, const Eigen::Vector3f& pivot) {
        for (int i = start; i < end; ++i) {
            Eigen::Vector3f p = atom(i) - pivot;
            p = R * p + pivot;
            set_atom(i, p);
        }
    }

    [[nodiscard]] float max_displacement_sq(const CoordsSoA& other,
                                            const std::vector<int>& indices) const {
        float d2max = 0.f;
        for (int i : indices) {
            const float dx = x[static_cast<size_t>(i)] - other.x[static_cast<size_t>(i)];
            const float dy = y[static_cast<size_t>(i)] - other.y[static_cast<size_t>(i)];
            const float dz = z[static_cast<size_t>(i)] - other.z[static_cast<size_t>(i)];
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > d2max) d2max = d2;
        }
        return d2max;
    }
};

} // namespace mcpu
