#pragma once
/// Structure-of-arrays coordinate storage (x[], y[], z[]) for cache-friendly hot loops.

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>
#include "pymcpu/utils/pair_r2.h"
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

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

    void set_atom(int i, const Eigen::Vector3f& p) noexcept {
        x[static_cast<size_t>(i)] = p.x();
        y[static_cast<size_t>(i)] = p.y();
        z[static_cast<size_t>(i)] = p.z();
    }

    [[nodiscard]] Eigen::Vector3f atom(int i) const {
        return {x[static_cast<size_t>(i)], y[static_cast<size_t>(i)], z[static_cast<size_t>(i)]};
    }

    void copy_atom_from(const CoordsSoA& src, int dst, int src_idx) {
        x[static_cast<size_t>(dst)] = src.x[static_cast<size_t>(src_idx)];
        y[static_cast<size_t>(dst)] = src.y[static_cast<size_t>(src_idx)];
        z[static_cast<size_t>(dst)] = src.z[static_cast<size_t>(src_idx)];
    }

    /// Copy atoms [lo, hi) from src at the same indices.
    void copy_range_from(const CoordsSoA& src, int lo, int hi) {
        if (hi <= lo) return;
        const size_t ulo = static_cast<size_t>(lo);
        const size_t bytes = static_cast<size_t>(hi - lo) * sizeof(float);
        std::memcpy(x.data() + ulo, src.x.data() + ulo, bytes);
        std::memcpy(y.data() + ulo, src.y.data() + ulo, bytes);
        std::memcpy(z.data() + ulo, src.z.data() + ulo, bytes);
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

    /// Rotate atoms [start, end) by R about pivot.
    ///
    /// The arithmetic is done in double and each coordinate is rounded to float
    /// once, when it is stored. In float32 it rounded twice, once relative to the
    /// pivot and again in lab coordinates, and that shrinks every distance the
    /// rotation should keep: about -1e-8 A per move, steadily. Over 5M
    /// pivot-only chignolin steps CA-C bonds shrank by 3e-3 A. Rounded once,
    /// what remains is unbiased float noise (+6e-6 A on average there), and
    /// a distance the rotation keeps moves by at most sqrt(3) float steps of
    /// the larger coordinate, which Context::rigid_carry_bound_A relies on.
    ///
    /// With FMA (MCPU_ARCH=v3) four atoms go through each AVX pass. Each
    /// output is spelt out as the exact chain of fused multiply-adds GCC
    /// emitted for the Eigen expression below on this target:
    ///   x' = fma(R02, dz, fma(R01, dy, R00*dx)) + px  (y' likewise, row 1)
    ///   z' = fma(R20, dx, fma(R22, dz, R21*dy)) + pz
    /// so the coordinates are bit for bit those of the scalar loop, and no
    /// longer depend on how the compiler happens to contract it.
    void rotate_atoms(int start, int end, const Eigen::Matrix3d& R, const Eigen::Vector3d& pivot) {
#if defined(__AVX2__) && defined(__FMA__)
        float* const px_ = x.data();
        float* const py_ = y.data();
        float* const pz_ = z.data();
        const __m256d r00 = _mm256_set1_pd(R(0, 0)), r01 = _mm256_set1_pd(R(0, 1)),
                      r02 = _mm256_set1_pd(R(0, 2));
        const __m256d r10 = _mm256_set1_pd(R(1, 0)), r11 = _mm256_set1_pd(R(1, 1)),
                      r12 = _mm256_set1_pd(R(1, 2));
        const __m256d r20 = _mm256_set1_pd(R(2, 0)), r21 = _mm256_set1_pd(R(2, 1)),
                      r22 = _mm256_set1_pd(R(2, 2));
        const __m256d cx = _mm256_set1_pd(pivot.x()), cy = _mm256_set1_pd(pivot.y()),
                      cz = _mm256_set1_pd(pivot.z());
        int i = start;
        for (; i + 4 <= end; i += 4) {
            const __m256d dx = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(px_ + i)), cx);
            const __m256d dy = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(py_ + i)), cy);
            const __m256d dz = _mm256_sub_pd(_mm256_cvtps_pd(_mm_loadu_ps(pz_ + i)), cz);
            const __m256d ox = _mm256_add_pd(
                _mm256_fmadd_pd(r02, dz, _mm256_fmadd_pd(r01, dy, _mm256_mul_pd(r00, dx))), cx);
            const __m256d oy = _mm256_add_pd(
                _mm256_fmadd_pd(r12, dz, _mm256_fmadd_pd(r11, dy, _mm256_mul_pd(r10, dx))), cy);
            const __m256d oz = _mm256_add_pd(
                _mm256_fmadd_pd(r20, dx, _mm256_fmadd_pd(r22, dz, _mm256_mul_pd(r21, dy))), cz);
            _mm_storeu_ps(px_ + i, _mm256_cvtpd_ps(ox));
            _mm_storeu_ps(py_ + i, _mm256_cvtpd_ps(oy));
            _mm_storeu_ps(pz_ + i, _mm256_cvtpd_ps(oz));
        }
        for (; i < end; ++i) {
            const size_t k = static_cast<size_t>(i);
            const double dx = static_cast<double>(x[k]) - pivot.x();
            const double dy = static_cast<double>(y[k]) - pivot.y();
            const double dz = static_cast<double>(z[k]) - pivot.z();
            x[k] = static_cast<float>(std::fma(R(0, 2), dz, std::fma(R(0, 1), dy, R(0, 0) * dx)) + pivot.x());
            y[k] = static_cast<float>(std::fma(R(1, 2), dz, std::fma(R(1, 1), dy, R(1, 0) * dx)) + pivot.y());
            z[k] = static_cast<float>(std::fma(R(2, 0), dx, std::fma(R(2, 2), dz, R(2, 1) * dy)) + pivot.z());
        }
#else
        for (int i = start; i < end; ++i) {
            const Eigen::Vector3d p = R * (atom(i).cast<double>() - pivot) + pivot;
            set_atom(i, p.cast<float>());
        }
#endif
    }
};

} // namespace mcpu
