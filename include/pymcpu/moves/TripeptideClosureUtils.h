#pragma once
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>

namespace TripeptideLoopClosure {

    // ---------------------------------------------------------
    // 1-Dimensional Polynomial Functions
    // ---------------------------------------------------------
    template <typename D1, typename D2, typename DOut>
    inline void poly_mul1(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                          int p1, int p2, const Eigen::MatrixBase<DOut>& out_const, int& p3) {
        Eigen::MatrixBase<DOut>& out = const_cast<Eigen::MatrixBase<DOut>&>(out_const);
        p3 = p1 + p2;
        out.derived().setZero();
        for (int i1 = 0; i1 <= p1; ++i1) {
            for (int i2 = 0; i2 <= p2; ++i2) {
                out.derived()(i1 + i2) += u1(i1) * u2(i2);
            }
        }
    }

    template <typename D1, typename D2, typename DOut>
    inline void poly_sub1(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                          int p1, int p2, const Eigen::MatrixBase<DOut>& out_const, int& p3) {
        Eigen::MatrixBase<DOut>& out = const_cast<Eigen::MatrixBase<DOut>&>(out_const);
        p3 = std::max(p1, p2);
        out.derived().setZero();
        for (int i = 0; i <= p3; ++i) {
            double val1 = (i <= p1) ? u1(i) : 0.0;
            double val2 = (i <= p2) ? u2(i) : 0.0;
            out.derived()(i) = val1 - val2;
        }
    }

    template <typename D1, typename D2, typename D3, typename D4, typename DOut>
    inline void poly_mul_sub1(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                              const Eigen::MatrixBase<D3>& u3, const Eigen::MatrixBase<D4>& u4,
                              int p1, int p2, int p3, int p4, const Eigen::MatrixBase<DOut>& out, int& p5) {
        Eigen::Matrix<double, 17, 1> d1 = Eigen::Matrix<double, 17, 1>::Zero();
        Eigen::Matrix<double, 17, 1> d2 = Eigen::Matrix<double, 17, 1>::Zero();
        int pd1, pd2;
        poly_mul1(u1, u2, p1, p2, d1, pd1);
        poly_mul1(u3, u4, p3, p4, d2, pd2);
        poly_sub1(d1, d2, pd1, pd2, out, p5);
    }

    // Overloads for fixed-size vectors (used in initialize)
    template <typename D1, typename D2, typename DOut>
    inline void poly_mul1(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                          const Eigen::MatrixBase<DOut>& out_const) {
        Eigen::MatrixBase<DOut>& out = const_cast<Eigen::MatrixBase<DOut>&>(out_const);
        out.derived().setZero();
        for (int i1 = 0; i1 < u1.size(); ++i1) {
            for (int i2 = 0; i2 < u2.size(); ++i2) {
                out.derived()(i1 + i2) += u1(i1) * u2(i2);
            }
        }
    }

    template <typename D1, typename D2, typename DOut>
    inline void poly_sub1(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                          const Eigen::MatrixBase<DOut>& out_const) {
        Eigen::MatrixBase<DOut>& out = const_cast<Eigen::MatrixBase<DOut>&>(out_const);
        out.derived().setZero();
        out.derived().head(u1.size()) += u1;
        out.derived().head(u2.size()) -= u2;
    }

    // ---------------------------------------------------------
    // 2-Dimensional Polynomial Functions
    // ---------------------------------------------------------
    template <typename D1, typename D2, typename DOut>
    inline void poly_mul2(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                          const std::array<int, 2>& p1, const std::array<int, 2>& p2, 
                          const Eigen::MatrixBase<DOut>& out_const, std::array<int, 2>& p3) {
        Eigen::MatrixBase<DOut>& out = const_cast<Eigen::MatrixBase<DOut>&>(out_const);
        p3[0] = p1[0] + p2[0];
        p3[1] = p1[1] + p2[1];
        out.derived().setZero();
        for (int i1 = 0; i1 <= p1[1]; ++i1) {
            for (int j1 = 0; j1 <= p1[0]; ++j1) {
                double u1ij = u1(j1, i1);
                for (int i2 = 0; i2 <= p2[1]; ++i2) {
                    for (int j2 = 0; j2 <= p2[0]; ++j2) {
                        out.derived()(j1 + j2, i1 + i2) += u1ij * u2(j2, i2);
                    }
                }
            }
        }
    }

    template <typename D1, typename D2, typename DOut>
    inline void poly_sub2(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                          const std::array<int, 2>& p1, const std::array<int, 2>& p2, 
                          const Eigen::MatrixBase<DOut>& out_const, std::array<int, 2>& p3) {
        Eigen::MatrixBase<DOut>& out = const_cast<Eigen::MatrixBase<DOut>&>(out_const);
        p3[0] = std::max(p1[0], p2[0]);
        p3[1] = std::max(p1[1], p2[1]);
        out.derived().setZero();
        for (int i = 0; i <= p3[1]; ++i) {
            for (int j = 0; j <= p3[0]; ++j) {
                double val1 = (i <= p1[1] && j <= p1[0]) ? u1(j, i) : 0.0;
                double val2 = (i <= p2[1] && j <= p2[0]) ? u2(j, i) : 0.0;
                out.derived()(j, i) = val1 - val2;
            }
        }
    }

    template <typename D1, typename D2, typename D3, typename D4, typename DOut>
    inline void poly_mul_sub2(const Eigen::MatrixBase<D1>& u1, const Eigen::MatrixBase<D2>& u2, 
                              const Eigen::MatrixBase<D3>& u3, const Eigen::MatrixBase<D4>& u4,
                              const std::array<int, 2>& p1, const std::array<int, 2>& p2, 
                              const std::array<int, 2>& p3, const std::array<int, 2>& p4, 
                              const Eigen::MatrixBase<DOut>& out, std::array<int, 2>& p5) {
        Eigen::Matrix<double, 5, 5> d1 = Eigen::Matrix<double, 5, 5>::Zero();
        Eigen::Matrix<double, 5, 5> d2 = Eigen::Matrix<double, 5, 5>::Zero();
        std::array<int, 2> pd1, pd2;
        poly_mul2(u1, u2, p1, p2, d1, pd1);
        poly_mul2(u3, u4, p3, p4, d2, pd2);
        poly_sub2(d1, d2, pd1, pd2, out, p5);
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