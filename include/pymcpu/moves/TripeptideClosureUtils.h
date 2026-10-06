#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>

namespace TripeptideLoopClosure {

    // ---------------------------------------------------------
    // Polynomial products for the closure polynomial
    // ---------------------------------------------------------
    // Every degree is a template argument: the closure polynomial is always built from the
    // same chain of products (get_poly_coeff), so the degrees are known when compiling and
    // each product unrolls into straight-line code with no zero-filled 17-entry scratch
    // vectors. Each output coefficient sums its terms in the order of the loops below (outer
    // factor index first), which is the order the run-time-degree versions used, so the
    // coefficients are bit-identical to theirs.

    // out[0..P1+P2] = u1 * u2 (coefficient k of u is u[k]).
    template <int P1, int P2>
    inline void poly_mul1(const double* u1, const double* u2, double* out) {
#pragma GCC unroll 32
        for (int k = 0; k <= P1 + P2; ++k) out[k] = 0.0;
#pragma GCC unroll 32
        for (int i1 = 0; i1 <= P1; ++i1)
#pragma GCC unroll 32
            for (int i2 = 0; i2 <= P2; ++i2)
                out[i1 + i2] += u1[i1] * u2[i2];
    }

    // out[0..max(P1, P2)] = u1 - u2.
    template <int P1, int P2>
    inline void poly_sub1(const double* u1, const double* u2, double* out) {
        constexpr int P = (P1 > P2) ? P1 : P2;
#pragma GCC unroll 32
        for (int k = 0; k <= P; ++k)
            out[k] = ((k <= P1) ? u1[k] : 0.0) - ((k <= P2) ? u2[k] : 0.0);
    }

    // out = u1 * u2 - u3 * u4.
    template <int P1, int P2, int P3, int P4>
    inline void poly_mul_sub1(const double* u1, const double* u2,
                              const double* u3, const double* u4, double* out) {
        double d1[P1 + P2 + 1], d2[P3 + P4 + 1];
        poly_mul1<P1, P2>(u1, u2, d1);
        poly_mul1<P3, P4>(u3, u4, d2);
        poly_sub1<P1 + P2, P3 + P4>(d1, d2, out);
    }

    // Two variables: u(j, i) is the coefficient of x^j y^i; a polynomial of degree (R, C)
    // has R in x and C in y.
    using Poly2 = Eigen::Matrix<double, 5, 5>;

    template <int R1, int C1, int R2, int C2>
    inline void poly_mul2(const Poly2& u1, const Poly2& u2, Poly2& out) {
#pragma GCC unroll 8
        for (int i = 0; i <= C1 + C2; ++i)
#pragma GCC unroll 8
            for (int j = 0; j <= R1 + R2; ++j) out(j, i) = 0.0;
#pragma GCC unroll 8
        for (int i1 = 0; i1 <= C1; ++i1)
#pragma GCC unroll 8
            for (int j1 = 0; j1 <= R1; ++j1) {
                const double u1ij = u1(j1, i1);
#pragma GCC unroll 8
                for (int i2 = 0; i2 <= C2; ++i2)
#pragma GCC unroll 8
                    for (int j2 = 0; j2 <= R2; ++j2)
                        out(j1 + j2, i1 + i2) += u1ij * u2(j2, i2);
            }
    }

    // out = u1 * u2 - u3 * u4, degrees (R1, C1) ... (R4, C4); only the result's degree
    // block of out is written.
    template <int R1, int C1, int R2, int C2, int R3, int C3, int R4, int C4>
    inline void poly_mul_sub2(const Poly2& u1, const Poly2& u2,
                              const Poly2& u3, const Poly2& u4, Poly2& out) {
        constexpr int RA = R1 + R2, CA = C1 + C2, RB = R3 + R4, CB = C3 + C4;
        constexpr int R = (RA > RB) ? RA : RB, C = (CA > CB) ? CA : CB;
        Poly2 d1, d2;
        poly_mul2<R1, C1, R2, C2>(u1, u2, d1);
        poly_mul2<R3, C3, R4, C4>(u3, u4, d2);
#pragma GCC unroll 8
        for (int i = 0; i <= C; ++i)
#pragma GCC unroll 8
            for (int j = 0; j <= R; ++j)
                out(j, i) = ((i <= CA && j <= RA) ? d1(j, i) : 0.0)
                          - ((i <= CB && j <= RB) ? d2(j, i) : 0.0);
    }

    // ========================================================================
    // High-Performance Geometry Helpers
    // ========================================================================
    inline double calc_dih_ang(const Eigen::Vector3d& r1, const Eigen::Vector3d& r2, const Eigen::Vector3d& r3) {
        Eigen::Vector3d p = r1.cross(r2);
        Eigen::Vector3d q = r2.cross(r3);
        Eigen::Vector3d s = r3.cross(r1);
        double arg = p.dot(q) / std::sqrt(p.squaredNorm() * q.squaredNorm());
        arg = std::clamp(arg, -1.0, 1.0);
        double angle = std::acos(arg);
        return (s.dot(r2) >= 0.0) ? angle : -angle;
    }

    inline double calc_bnd_ang(const Eigen::Vector3d& r1, const Eigen::Vector3d& r2) {
        double arg = r1.dot(r2); 
        arg = std::clamp(arg, -1.0, 1.0);
        return std::acos(arg);
    }

} // namespace TripeptideLoopClosure