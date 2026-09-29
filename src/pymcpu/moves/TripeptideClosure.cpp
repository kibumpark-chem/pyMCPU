#include <cmath>
#include <algorithm>
#include <codecvt>
#include "pymcpu/utils/numbers_compat.h"
#include <iostream>
#include <iomanip>
#include <limits>

#include "pymcpu/moves/TripeptideClosure.h"
#include "pymcpu/moves/TripeptideClosureUtils.h"
#include "pymcpu/moves/SturmSolver.h"

using namespace TripeptideLoopClosure;

namespace {

// Helpers for the pole-free back-substitution (TripeptideSolver::back_substitute).
//
// Each closure equation is biquadratic in two half-angle tangents t = tan(tau/2). Multiplied by
// cos^2(tau_a/2) cos^2(tau_b/2) it becomes v(tau_a)^T T v(tau_b) with v(tau) = (1, cos tau, sin tau),
// which has no pole at tau = pi. With one angle known it is a line  k + a cos + b sin = 0  in the other.

// The two unit vectors (c, s) on the line w[0] + w[1] c + w[2] s = 0; at a tangency both are the touching
// point. No transcendental function; the only divisor is w[1]^2 + w[2]^2 > 0.
inline void circle_line(const double w[3], double c[2], double s[2]) {
    const double k = w[0], a = w[1], b = w[2];
    const double rho2 = a * a + b * b;
    const double h = rho2 - k * k;
    const double r = (h > 0.0) ? std::sqrt(h) : 0.0;
    c[0] = (-k * a - b * r) / rho2;  s[0] = (-k * b + a * r) / rho2;
    c[1] = (-k * a + b * r) / rho2;  s[1] = (-k * b - a * r) / rho2;
}

// Relative residual of the line w at the unit vector (c, s).
inline double line_residual(const double w[3], double c, double s) {
    const double num = std::abs(w[0] + w[1] * c + w[2] * s);
    const double den = std::abs(w[0]) + std::abs(w[1] * c) + std::abs(w[2] * s);
    return den > 0.0 ? num / den : 0.0;
}

// Do the lines w0 and w1 meet ON the unit circle? Division-free form of |Cramer intersection|^2 == 1,
// relative to the size of its terms.
inline double pair_residual(const double w0[3], const double w1[3]) {
    const double x = w0[0] * w1[2] - w1[0] * w0[2];
    const double y = w0[1] * w1[0] - w1[1] * w0[0];
    const double d = w0[1] * w1[2] - w1[1] * w0[2];
    const double nx = std::abs(w0[0] * w1[2]) + std::abs(w1[0] * w0[2]);
    const double ny = std::abs(w0[1] * w1[0]) + std::abs(w1[1] * w0[0]);
    const double nd = std::abs(w0[1] * w1[2]) + std::abs(w1[1] * w0[2]);
    const double den = nx * nx + ny * ny + nd * nd;
    return den > 0.0 ? std::abs(x * x + y * y - d * d) / den : 0.0;
}

}  // namespace

int TripeptideSolver::solv_3pep_poly(const Vec3& r_n1, const Vec3& r_a1, 
                                     const Vec3& r_a3, const Vec3& r_c3, 
                                     std::vector<Solution>& solutions) {
    int n_soln = 0;
    solutions.clear();
    n_rejected_ = 0;

    get_input_angles(n_soln, r_n1, r_a1, r_a3, r_c3);
    if (n_soln == 0) {
        return 0; // No geometric solution possible
    }

    Eigen::Matrix<double, 17, 1> poly_coeff = Eigen::Matrix<double, 17, 1>::Zero();
    get_poly_coeff(poly_coeff);

    std::vector<double> roots;
    solve_roots(poly_coeff, roots);
    
    if (roots.empty()) {
        return 0; 
    }

    coord_from_poly_roots(roots, r_n1, r_a1, r_a3, r_c3, solutions);

    return solutions.size();
}

std::vector<Solution> TripeptideSolver::solve(
    const Vec3& r_n1, const Vec3& r_a1, 
    const Vec3& r_a3, const Vec3& r_c3) 
{
    std::vector<Solution> solutions;
    solv_3pep_poly(r_n1, r_a1, r_a3, r_c3, solutions);
    return solutions;
}

void TripeptideSolver::initialize(
    const std::array<double, 6>& b_len,
    const std::array<double, 7>& b_ang,
    const std::array<double, 2>& t_ang)
{
    len0   = b_len;
    b_ang0 = b_ang;
    t_ang0 = t_ang;

    using Vector5d = Eigen::Matrix<double, 5, 1>;

    Eigen::Vector3d rr_c1(0.0, 0.0, 0.0);
    Eigen::Vector3d axis(1.0, 0.0, 0.0);

    for (int i = 0; i < 2; i++) {
        // 1. Map bond lengths and angles (Fortran 1-based indexing -> C++ 0-based)
        double len_ca_c = len0[3 * i];     // len0(3*i+1)
        double len_c_n  = len0[3 * i + 1]; // len0(3*i+2)
        double len_n_ca = len0[3 * i + 2]; // len0(3*i+3)
        
        double ang_ca_c_n = b_ang0[3 * i + 1]; // b_ang0(3*i+2)
        double ang_c_n_ca = b_ang0[3 * i + 2]; // b_ang0(3*i+3)

        // 2. Place first Alpha Carbon (rr_a1) and next Nitrogen (rr_n2) flat on XY plane
        Eigen::Vector3d rr_a1(std::cos(ang_ca_c_n) * len_ca_c, std::sin(ang_ca_c_n) * len_ca_c, 0.0);
        Eigen::Vector3d rr_n2(len_c_n, 0.0, 0.0);
        Eigen::Vector3d rr_c1a1 = rr_a1 - rr_c1;

        // 3. Place reference Alpha Carbon (rr_n2a2_ref) flat on XY plane
        Eigen::Vector3d rr_n2a2_ref(-std::cos(ang_c_n_ca) * len_n_ca, std::sin(ang_c_n_ca) * len_n_ca, 0.0);

        // 4. Rotate rr_n2a2_ref around X-axis by the peptide torsion (t_ang0)
        // (This replaces the Fortran quaternion/rotation_matrix calls)
        Eigen::AngleAxisd Us_rot(t_ang0[i], axis);
        Eigen::Vector3d rr_a2 = Us_rot * rr_n2a2_ref + rr_n2;

        // 5. Calculate C_alpha distances
        Eigen::Vector3d rr_a1a2 = rr_a2 - rr_a1;
        double len1 = rr_a1a2.norm();
        len_aa[i + 1] = len1;

        // 6. Generate Unit Vectors
        Eigen::Vector3d bb_c1a1 = rr_c1a1 / len_ca_c;
        Eigen::Vector3d bb_a1a2 = rr_a1a2 / len1;
        Eigen::Vector3d bb_a2n2 = (rr_n2 - rr_a2) / len_n_ca;

        // 7. Calculate final KIC internal variables using your new 3D math functions
        xi[i + 1] = calc_bnd_ang(-bb_a1a2, bb_a2n2);
        eta[i]    = calc_bnd_ang(bb_a1a2, -bb_c1a1);
        delta[i + 1] = mcpu::PI - calc_dih_ang(bb_c1a1, bb_a1a2, bb_a2n2);
    }

    double a_min = b_ang[3] - (xi[1] + eta[1]);
    double a_max = std::min(b_ang[3] + (xi[1] + eta[1]), mcpu::PI);
    aa13_min_sqr = len_aa[1]*len_aa[1]  + len_aa[2]*len_aa[2] - 2.0*len_aa[1]*len_aa[2]*std::cos(a_min);
    aa13_max_sqr = len_aa[1]*len_aa[1]  + len_aa[2]*len_aa[2] - 2.0*len_aa[1]*len_aa[2]*std::cos(a_max);

    is_initialized_ = true;
}

void TripeptideSolver::get_input_angles(
    int& n_soln,
    const Eigen::Vector3d& r_n1,
    const Eigen::Vector3d& r_a1,
    const Eigen::Vector3d& r_a3,
    const Eigen::Vector3d& r_c3)
{
    n_soln = max_soln;

    r_a1a3 = r_a3 - r_a1;
    double dr_sqr = r_a1a3.squaredNorm(); 
    len_aa[0] = std::sqrt(dr_sqr);        

    if (dr_sqr < aa13_min_sqr || dr_sqr > aa13_max_sqr) {
        n_soln = 0;
        return;
    }

    r_a1n1 = r_n1 - r_a1;
    len_na[0] = r_a1n1.norm();
    len_na[1] = len0[2];  
    len_na[2] = len0[5];  

    r_a3c3 = r_c3 - r_a3;
    len_ac[0] = len0[0];  
    len_ac[1] = len0[3];  
    len_ac[2] = r_a3c3.norm();

    b_a1n1 = r_a1n1.normalized();
    b_a3c3 = r_a3c3.normalized();
    b_a1a3 = r_a1a3.normalized();

    delta[3] = calc_dih_ang(-b_a1n1, b_a1a3, b_a3c3);
    delta[0] = delta[3]; 

    xi[0] = calc_bnd_ang(-b_a1a3, b_a1n1);
    eta[2] = calc_bnd_ang(b_a1a3, b_a3c3);

    for (int i = 0; i < 3; ++i) {
        cos_delta[i+1] = std::cos(delta[i+1]);
        sin_delta[i+1] = std::sin(delta[i+1]);
        
        cos_xi[i] = std::cos(xi[i]);
        sin_xi[i] = std::sin(xi[i]);
        
        cos_eta[i] = std::cos(eta[i]);
        sin_eta[i] = std::sin(eta[i]);
    }
    cos_delta[0] = cos_delta[3]; 
    sin_delta[0] = sin_delta[3];

    theta[0] = b_ang0[0]; 
    theta[1] = b_ang0[3]; 
    theta[2] = b_ang0[6]; 
    for (int i = 0; i < 3; ++i) {
        cos_theta[i] = std::cos(theta[i]);
    }

    cos_alpha[0] = std::clamp(-(std::pow(len_aa[0], 2) + std::pow(len_aa[1], 2) - std::pow(len_aa[2], 2)) / (2.0 * len_aa[0] * len_aa[1]), -1.0, 1.0);
    alpha[0] = std::acos(cos_alpha[0]);
    sin_alpha[0] = std::sin(alpha[0]);

    cos_alpha[1] = std::clamp((std::pow(len_aa[1], 2) + std::pow(len_aa[2], 2) - std::pow(len_aa[0], 2)) / (2.0 * len_aa[1] * len_aa[2]), -1.0, 1.0);
    alpha[1] = std::acos(cos_alpha[1]);
    sin_alpha[1] = std::sin(alpha[1]);

    alpha[2] = mcpu::PI - alpha[0] + alpha[1];
    cos_alpha[2] = std::cos(alpha[2]);
    sin_alpha[2] = std::sin(alpha[2]);

    for (int i = 0; i < 3; ++i) {
        test_two_cone_existence_soln(theta[i], xi[i], eta[i], alpha[i], n_soln);
        if (n_soln == 0) {
            return;
        }
    }
}

void TripeptideSolver::test_two_cone_existence_soln(
    double tt, double kx, double et, double ap,
    int& n_soln)
{
    n_soln = max_soln;

    double at = ap - tt;
    double ex = kx + et;

    if (std::abs(at) > ex) {
        n_soln = 0;
    }
}

void TripeptideSolver::get_poly_coeff(Eigen::Matrix<double, 17, 1>& poly_coeff)
{
    using Matrix5d = Eigen::Matrix<double, 5, 5>;
    using Vector17d = Eigen::Matrix<double, 17, 1>;

    Matrix5d u11, u12, u13, u31, u32, u33;
    u11.setZero(); u12.setZero(); u13.setZero();
    u31.setZero(); u32.setZero(); u33.setZero();

    Eigen::Vector3d B0, B1, B2, B3, B4, B5, B6, B7, B8;

    for (int i = 0; i < 3; ++i) {
        double A0 = cos_alpha[i] * cos_xi[i] * cos_eta[i] - cos_theta[i];
        double A1 = -sin_alpha[i] * cos_xi[i] * sin_eta[i];
        double A2 = sin_alpha[i] * sin_xi[i] * cos_eta[i];
        double A3 = sin_xi[i] * sin_eta[i];
        double A4 = A3 * cos_alpha[i];

        int j = i; 
        
        double A21 = A2 * cos_delta[j];
        double A22 = A2 * sin_delta[j];
        double A31 = A3 * cos_delta[j];
        double A32 = A3 * sin_delta[j];
        double A41 = A4 * cos_delta[j];
        double A42 = A4 * sin_delta[j];

        B0[i] = A0 + A22 + A31;
        B1[i] = 2.0 * (A1 + A42);
        B2[i] = 2.0 * (A32 - A21);
        B3[i] = -4.0 * A41;
        B4[i] = A0 + A22 - A31;
        B5[i] = A0 - A22 - A31;
        B6[i] = -2.0 * (A21 + A32);
        B7[i] = 2.0 * (A1 - A42);
        B8[i] = A0 - A22 + A31;
    }

    C0.col(0) << B0[0], B2[0], B5[0];
    C1.col(0) << B1[0], B3[0], B7[0];
    C2.col(0) << B4[0], B6[0], B8[0];

    for (int i = 1; i < 3; ++i) {
        C0.col(i) << B0[i], B1[i], B4[i];
        C1.col(i) << B2[i], B3[i], B6[i];
        C2.col(i) << B5[i], B7[i], B8[i];
    }

    for (int i = 0; i < 3; ++i) {
        u11(i, 0) = C0(i, 0); 
        u12(i, 0) = C1(i, 0);
        u13(i, 0) = C2(i, 0);
        
        u31(0, i) = C0(i, 1); 
        u32(0, i) = C1(i, 1);
        u33(0, i) = C2(i, 1);
    }

    std::array<int, 2> p1 = {2, 0};
    std::array<int, 2> p3 = {0, 2};
    
    Matrix5d um1, um2, um3, um4, um5, um6, q_tmp;
    std::array<int, 2> p_um1, p_um2, p_um3, p_um4, p_um5, p_um6, p_Q;

    poly_mul_sub2(u32, u32, u31, u33, p3, p3, p3, p3, um1, p_um1);
    poly_mul_sub2(u12, u32, u11, u33, p1, p3, p1, p3, um2, p_um2);
    poly_mul_sub2(u12, u33, u13, u32, p1, p3, p1, p3, um3, p_um3);
    poly_mul_sub2(u11, u33, u31, u13, p1, p3, p3, p1, um4, p_um4);
    poly_mul_sub2(u13, um1, u33, um2, p1, p_um1, p3, p_um2, um5, p_um5);
    poly_mul_sub2(u13, um4, u12, um3, p1, p_um4, p1, p_um3, um6, p_um6);
    poly_mul_sub2(u11, um5, u31, um6, p1, p_um5, p3, p_um6, q_tmp, p_Q);
    
    Q.block<5,5>(0,0) = q_tmp;

    R.setZero();
    R.block<3, 1>(0, 0) = C0.col(2); 
    R.block<3, 1>(0, 1) = C1.col(2);
    R.block<3, 1>(0, 2) = C2.col(2);

    int p2 = 2;
    int p4 = 4;
    int p_f1, p_f2, p_f3, p_f4, p_f5, p_f6, p_f7, p_f8, p_f9, p_f10;
    int p_f11, p_f12, p_f13, p_f14, p_f15, p_f16, p_f17, p_f18, p_f19, p_f20;
    int p_f21, p_f22, p_f23, p_f24, p_f25, p_f26, p_final;

    Vector17d f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16;
    Vector17d f17, f18, f19, f20, f21, f22, f23, f24, f25, f26;

    poly_mul_sub1(R.col(1), R.col(1), R.col(0), R.col(2), p2, p2, p2, p2, f1, p_f1);
    poly_mul1(R.col(1), R.col(2), p2, p2, f2, p_f2);
    poly_mul_sub1(R.col(1), f1, R.col(0), f2, p2, p_f1, p2, p_f2, f3, p_f3);
    poly_mul1(R.col(2), f1, p2, p_f1, f4, p_f4);
    poly_mul_sub1(R.col(1), f3, R.col(0), f4, p2, p_f3, p2, p_f4, f5, p_f5);

    poly_mul_sub1(Q.col(1), R.col(1), Q.col(0), R.col(2), p4, p2, p4, p2, f6, p_f6);
    poly_mul_sub1(Q.col(2), f1, R.col(2), f6, p4, p_f1, p2, p_f6, f7, p_f7);
    poly_mul_sub1(Q.col(3), f3, R.col(2), f7, p4, p_f3, p2, p_f7, f8, p_f8);
    poly_mul_sub1(Q.col(4), f5, R.col(2), f8, p4, p_f5, p2, p_f8, f9, p_f9);

    poly_mul_sub1(Q.col(3), R.col(1), Q.col(4), R.col(0), p4, p2, p4, p2, f10, p_f10);
    poly_mul_sub1(Q.col(2), f1, R.col(0), f10, p4, p_f1, p2, p_f10, f11, p_f11);
    poly_mul_sub1(Q.col(1), f3, R.col(0), f11, p4, p_f3, p2, p_f11, f12, p_f12);

    poly_mul_sub1(Q.col(2), R.col(1), Q.col(1), R.col(2), p4, p2, p4, p2, f13, p_f13);
    poly_mul_sub1(Q.col(3), f1, R.col(2), f13, p4, p_f1, p2, p_f13, f14, p_f14);
    poly_mul_sub1(Q.col(3), R.col(1), Q.col(2), R.col(2), p4, p2, p4, p2, f15, p_f15);
    poly_mul_sub1(Q.col(4), f1, R.col(2), f15, p4, p_f1, p2, p_f15, f16, p_f16);
    poly_mul_sub1(Q.col(1), f14, Q.col(0), f16, p4, p_f14, p4, p_f16, f17, p_f17);

    poly_mul_sub1(Q.col(2), R.col(2), Q.col(3), R.col(1), p4, p2, p4, p2, f18, p_f18);
    poly_mul_sub1(Q.col(1), R.col(2), Q.col(3), R.col(0), p4, p2, p4, p2, f19, p_f19);
    poly_mul_sub1(Q.col(3), f19, Q.col(2), f18, p4, p_f19, p4, p_f18, f20, p_f20);
    
    poly_mul_sub1(Q.col(1), R.col(1), Q.col(2), R.col(0), p4, p2, p4, p2, f21, p_f21);
    poly_mul1(Q.col(4), f21, p4, p_f21, f22, p_f22);
    poly_sub1(f20, f22, p_f20, p_f22, f23, p_f23);
    
    poly_mul1(R.col(0), f23, p2, p_f23, f24, p_f24);
    poly_sub1(f17, f24, p_f17, p_f24, f25, p_f25);
    
    poly_mul_sub1(Q.col(4), f12, R.col(2), f25, p4, p_f12, p2, p_f25, f26, p_f26);
    poly_mul_sub1(Q.col(0), f9, R.col(0), f26, p4, p_f9, p2, p_f26, poly_coeff, p_final);

    if (poly_coeff[16] < 0.0) {
        poly_coeff = -poly_coeff; 
    }
}

void TripeptideSolver::solve_roots(const Eigen::Matrix<double, 17, 1>& poly_coeff, std::vector<double>& roots) {
    roots.clear();
    
    // Fast, thread-safe execution of the native C++ Sturm sequence
    SturmSolver sturm;
    sturm.solve(poly_coeff, roots);
}

// double TripeptideSolver::calc_t2(double t0) const {
//     double t0_2 = t0 * t0;
//     double t0_3 = t0_2 * t0;
//     double t0_4 = t0_3 * t0;
    
//     double num = Q(0,4) + Q(1,4)*t0 + Q(2,4)*t0_2 + Q(3,4)*t0_3 + Q(4,4)*t0_4;
//     double den = Q(0,3) + Q(1,3)*t0 + Q(2,3)*t0_2 + Q(3,3)*t0_3 + Q(4,3)*t0_4;
//     return -num / den;
// }

// double TripeptideSolver::calc_t1(double t0, double t2) const {
//     double t0_2 = t0 * t0;
//     double t2_2 = t2 * t2;

//     double U11 = C0(0, 1) + C0(1, 1) * t0 + C0(2, 1) * t0_2;
//     double U12 = C1(0, 1) + C1(1, 1) * t0 + C1(2, 1) * t0_2;
//     double U13 = C2(0, 1) + C2(1, 1) * t0 + C2(2, 1) * t0_2;

//     double U31 = C0(0, 2) + C0(1, 2) * t2 + C0(2, 2) * t2_2;
//     double U32 = C1(0, 2) + C1(1, 2) * t2 + C1(2, 2) * t2_2;
//     double U33 = C2(0, 2) + C2(1, 2) * t2 + C2(2, 2) * t2_2;

//     return (U31 * U13 - U11 * U33) / (U12 * U33 - U13 * U32);
// }

// Replaces calc_t2 / calc_t1 in coord_from_poly_roots. Those return the half-tangents t2, t1 as ratios whose
// numerator and denominator both fall to rounding level when tau2 or tau1 is near pi or two roots nearly
// coincide. Here every equation is used in the (1, cos, sin) basis, where no variable has a pole.
//
// T_i = M^T E_i M, with E_i(j, k) = C_k(j, i) the coefficient of
//   eq0: t3^j t1^k,   eq1: t2^j t1^k,   eq2: t3^j t2^k     (t3 = the polynomial root, half_tan[2])
// and M mapping (1, cos, sin) to the half-angle monomials (c^2, s c, s^2): c^2 = (1 + cos)/2, s c = sin/2,
// s^2 = (1 - cos)/2.
void TripeptideSolver::build_trig_coeff()
{
    // M = {{1/2, 1/2, 0}, {0, 0, 1/2}, {1/2, -1/2, 0}}; the product M^T E M written out (27 flops per equation).
    const Eigen::Matrix3d* C[3] = {&C0, &C1, &C2};
    for (int i = 0; i < 3; ++i) {
        double F[3][3];                                   // F = E M
        for (int j = 0; j < 3; ++j) {
            const double e0 = (*C[0])(j, i), e1 = (*C[1])(j, i), e2 = (*C[2])(j, i);   // E(j, k) = C_k(j, i)
            F[j][0] = 0.5 * (e0 + e2);
            F[j][1] = 0.5 * (e0 - e2);
            F[j][2] = 0.5 * e1;
        }
        for (int q = 0; q < 3; ++q) {                     // T = M^T F
            trig_coeff_[i][0][q] = 0.5 * (F[0][q] + F[2][q]);
            trig_coeff_[i][1][q] = 0.5 * (F[0][q] - F[2][q]);
            trig_coeff_[i][2][q] = 0.5 * F[1][q];
        }
    }
}

// Given (cos tau3, sin tau3) of a root, return (cos, sin) of tau1 and tau2.
//   tau2: eq2(tau3, .) is a line in (cos tau2, sin tau2); of its two circle points keep the one for which
//         eq0(tau3, .) and eq1(tau2, .) meet on the circle (pair_residual; this is the quartic condition
//         calc_t2 reduced, written without a division).
//   tau1: of the two circle points of eq0(tau3, .) keep the one that satisfies eq1(tau2, .).
void TripeptideSolver::back_substitute(double c3, double s3,
                                       double& c1, double& s1,
                                       double& c2, double& s2) const
{
    const auto& T0 = trig_coeff_[0];
    const auto& T1 = trig_coeff_[1];
    const auto& T2 = trig_coeff_[2];
    double w0[3], w2[3];
    for (int q = 0; q < 3; ++q) {
        w0[q] = T0[0][q] + T0[1][q] * c3 + T0[2][q] * s3;
        w2[q] = T2[0][q] + T2[1][q] * c3 + T2[2][q] * s3;
    }
    double cc[2], ss[2], w1[3], w1_best[3];
    circle_line(w2, cc, ss);
    double best = std::numeric_limits<double>::infinity();
    for (int m = 0; m < 2; ++m) {
        for (int q = 0; q < 3; ++q)
            w1[q] = T1[0][q] + T1[1][q] * cc[m] + T1[2][q] * ss[m];
        const double r = pair_residual(w0, w1);
        if (m == 0 || r < best) {
            best = r;
            c2 = cc[m]; s2 = ss[m];
            std::copy(w1, w1 + 3, w1_best);
        }
    }
    circle_line(w0, cc, ss);
    const int m = (line_residual(w1_best, cc[0], ss[0]) <= line_residual(w1_best, cc[1], ss[1])) ? 0 : 1;
    c1 = cc[m]; s1 = ss[m];
}

// A returned closure must reproduce its three N-CA-C targets (theta, set in get_input_angles).
bool TripeptideSolver::closes(const Solution& sol) const
{
    for (int i = 0; i < 3; ++i) {
        const Vec3 u = sol.r_n[i] - sol.r_a[i];
        const Vec3 v = sol.r_c[i] - sol.r_a[i];
        const double cosang = std::clamp(u.dot(v) / std::sqrt(u.squaredNorm() * v.squaredNorm()), -1.0, 1.0);
        if (!(std::abs(std::acos(cosang) - theta[i]) <= kClosureTol)) return false;
    }
    return true;
}

double TripeptideSolver::calc_t1(double t0, double t2) const {
    double t0_2 = t0 * t0;
    double t2_2 = t2 * t2;
    
    // Notice the indices are flipped to (row, power) to match the below code's [row][col]
    double U11 = C0(0, 0) + C0(1, 0) * t0 + C0(2, 0) * t0_2;
    double U12 = C1(0, 0) + C1(1, 0) * t0 + C1(2, 0) * t0_2;
    double U13 = C2(0, 0) + C2(1, 0) * t0 + C2(2, 0) * t0_2;

    double U31 = C0(0, 1) + C0(1, 1) * t2 + C0(2, 1) * t2_2;
    double U32 = C1(0, 1) + C1(1, 1) * t2 + C1(2, 1) * t2_2;
    double U33 = C2(0, 1) + C2(1, 1) * t2 + C2(2, 1) * t2_2;

    return (U31 * U13 - U11 * U33) / (U12 * U33 - U13 * U32);
}

double TripeptideSolver::calc_t2(double t0) const {
    double t0_2 = t0 * t0;
    double t0_3 = t0_2 * t0;
    double t0_4 = t0_3 * t0;

    // Evaluating polynomials A0 through A4 using rows of Q
    double A0 = Q(0,0) + Q(1,0)*t0 + Q(2,0)*t0_2 + Q(3,0)*t0_3 + Q(4,0)*t0_4;
    double A1 = Q(0,1) + Q(1,1)*t0 + Q(2,1)*t0_2 + Q(3,1)*t0_3 + Q(4,1)*t0_4;
    double A2 = Q(0,2) + Q(1,2)*t0 + Q(2,2)*t0_2 + Q(3,2)*t0_3 + Q(4,2)*t0_4;
    double A3 = Q(0,3) + Q(1,3)*t0 + Q(2,3)*t0_2 + Q(3,3)*t0_3 + Q(4,3)*t0_4;
    double A4 = Q(0,4) + Q(1,4)*t0 + Q(2,4)*t0_2 + Q(3,4)*t0_3 + Q(4,4)*t0_4;

    // Evaluating polynomials B0 through B2 using rows of R
    // Note: Your TripeptideSolver MUST have the R matrix defined for this to work.
    double B0 = R(0,0) + R(1,0)*t0 + R(2,0)*t0_2;
    double B1 = R(0,1) + R(1,1)*t0 + R(2,1)*t0_2;
    double B2 = R(0,2) + R(1,2)*t0 + R(2,2)*t0_2;

    double B2_2 = B2 * B2;
    double B2_3 = B2_2 * B2;

    double K0 = A2 * B2 - A4 * B0;
    double K1 = A3 * B2 - A4 * B1;
    double K2 = A1 * B2_2 - K1 * B0;
    double K3 = K0 * B2 - K1 * B1;
    
    return (K3 * B0 - A0 * B2_3) / (K2 * B2 - K3 * B1);
}

void TripeptideSolver::coord_from_poly_roots(const std::vector<double>& roots, 
                                             const Vec3& r_n1, const Vec3& r_a1, 
                                             const Vec3& r_a3, const Vec3& r_c3,
                                             std::vector<Solution>& solutions) {
    if (roots.empty()) return;

    Eigen::Vector3d ex = b_a1a3;
    Eigen::Vector3d ez = r_a1n1.cross(ex).normalized();
    Eigen::Vector3d ey = ez.cross(ex);

    Eigen::Vector3d b_a1a2 = -cos_alpha[0] * ex + sin_alpha[0] * ey;
    Eigen::Vector3d b_a3a2 =  cos_alpha[2] * ex + sin_alpha[2] * ey;

    std::array<Eigen::Vector3d, 3> p_s, s1, s2, p_t, t1, t2;
    std::array<Eigen::Vector3d, 3> p_s_c, s1_s, s2_s, p_t_c, t1_s, t2_s;

    p_s[0] = -ex; s1[0] = ez; s2[0] = ey; 
    p_t[0] = b_a1a2; t1[0] = ez; t2[0] = sin_alpha[0] * ex + cos_alpha[0] * ey; 

    p_s[1] = -b_a1a2; s1[1] = -ez; s2[1] = t2[0]; 
    p_t[1] = -b_a3a2; t1[1] = -ez; t2[1] = sin_alpha[2] * ex - cos_alpha[2] * ey; 

    p_s[2] = b_a3a2; s2[2] = t2[1]; s1[2] = ez;  
    p_t[2] = ex; t1[2] = ez; t2[2] = -ey; 

    for (int i = 0; i < 3; ++i) {
        p_s_c[i] = p_s[i] * cos_xi[i]; s1_s[i] = s1[i] * sin_xi[i]; s2_s[i] = s2[i] * sin_xi[i];
        p_t_c[i] = p_t[i] * cos_eta[i]; t1_s[i] = t1[i] * sin_eta[i]; t2_s[i] = t2[i] * sin_eta[i];
    }

    Eigen::Vector3d r_tmp = (r_a1n1 / len_na[0] - p_s_c[0]) / sin_xi[0];
    double angle = std::acos(std::clamp(s1[0].dot(r_tmp), -1.0, 1.0)); 
    double sig1_init = std::copysign(angle, r_tmp.dot(s2[0]));

    std::array<Eigen::Vector3d, 3> r_a, r_n, r_c;
    r_a[0] = r_a1; r_a[1] = r_a1 + len_aa[1] * b_a1a2; r_a[2] = r_a3;
    Eigen::Vector3d r0 = r_a1;

    std::array<double, 4> cos_tau, sin_tau; 
    std::array<double, 3> cos_sig, sin_sig;

    build_trig_coeff();
    for (size_t i_soln = 0; i_soln < roots.size(); ++i_soln) {
        // tau3 from the root t3 = tan(tau3/2); tau1, tau2 from the pole-free back-substitution
        // (was: half_tan[1] = calc_t2(t3); half_tan[0] = calc_t1(t3, half_tan[1]); then t -> cos, sin).
        const double t3 = roots[i_soln];
        const double d3 = 1.0 + t3 * t3;
        cos_tau[3] = (1.0 - t3 * t3) / d3;
        sin_tau[3] = 2.0 * t3 / d3;
        back_substitute(cos_tau[3], sin_tau[3], cos_tau[1], sin_tau[1], cos_tau[2], sin_tau[2]);

        cos_tau[0] = cos_tau[3]; sin_tau[0] = sin_tau[3];

        for (int i = 0; i < 3; ++i) {
            cos_sig[i] = cos_delta[i] * cos_tau[i] + sin_delta[i] * sin_tau[i];
            sin_sig[i] = sin_delta[i] * cos_tau[i] - cos_delta[i] * sin_tau[i];
        }

        for (int i = 0; i < 3; ++i) {
            Eigen::Vector3d r_s = p_s_c[i] + cos_sig[i] * s1_s[i] + sin_sig[i] * s2_s[i];
            Eigen::Vector3d r_t = p_t_c[i] + cos_tau[i+1] * t1_s[i] + sin_tau[i+1] * t2_s[i]; 
            r_n[i] = r_s * len_na[i] + r_a[i];
            r_c[i] = r_t * len_ac[i] + r_a[i];
        }
        double sig1 = std::atan2(sin_sig[0], cos_sig[0]);
        Eigen::AngleAxisd Us_rotation(-(sig1 - sig1_init), -ex);
        Eigen::Matrix3d Us = Us_rotation.toRotationMatrix();

 
        Solution sol;
        sol.r_n[0] = r_n1; sol.r_a[0] = r_a1; sol.r_c[0] = Us * (r_c[0] - r0) + r0;
        sol.r_n[1] = Us * (r_n[1] - r0) + r0; sol.r_a[1] = Us * (r_a[1] - r0) + r0; sol.r_c[1] = Us * (r_c[1] - r0) + r0;
        sol.r_n[2] = Us * (r_n[2] - r0) + r0; sol.r_a[2] = r_a3; sol.r_c[2] = r_c3;

        // Safety net: drop a closure that misses an N-CA-C target by more than kClosureTol. Both solves of a
        // KIC move (Integrator.cpp:1300, 1342) come through here, so the two solution counts in the
        // acceptance ratio are filtered alike. The caller can add last_rejected() to kic_geometry_invalid_.
        if (!closes(sol)) {
            ++n_rejected_;
            continue;
        }
        solutions.push_back(sol);
    }
}

// double TripeptideSolver::calculate_jacobian(const Solution& sol) const {
//     // We are building the 6x6 Spatial Manipulator Jacobian
//     // J = [ u_i x (r_end - p_i) ]
//     //     [         u_i         ]
//     Eigen::Matrix<double, 6, 6> J_matrix;

//     // 1. Define the 6 rotation axes (u_i) for phi and psi
//     std::array<Eigen::Vector3d, 6> u;
//     u[0] = (sol.r_a[0] - sol.r_n[0]).normalized(); // phi 1
//     u[1] = (sol.r_c[0] - sol.r_a[0]).normalized(); // psi 1
//     u[2] = (sol.r_a[1] - sol.r_n[1]).normalized(); // phi 2
//     u[3] = (sol.r_c[1] - sol.r_a[1]).normalized(); // psi 2
//     u[4] = (sol.r_a[2] - sol.r_n[2]).normalized(); // phi 3
//     u[5] = (sol.r_c[2] - sol.r_a[2]).normalized(); // psi 3

//     // 2. Define the pivot points (p_i) for each axis
//     std::array<Eigen::Vector3d, 6> p;
//     p[0] = sol.r_a[0];
//     p[1] = sol.r_c[0];
//     p[2] = sol.r_a[1];
//     p[3] = sol.r_c[1];
//     p[4] = sol.r_a[2];
//     p[5] = sol.r_c[2];

//     // 3. Define the end-effector (The final anchor point, CA3)
//     Eigen::Vector3d r_end = sol.r_a[2];

//     // 4. Populate the 6x6 matrix
//     for (int i = 0; i < 6; ++i) {
//         Eigen::Vector3d r_cross = u[i].cross(r_end - p[i]);
        
//         // Top 3 rows: Linear velocity component
//         J_matrix.block<3, 1>(0, i) = r_cross;
        
//         // Bottom 3 rows: Angular velocity component
//         J_matrix.block<3, 1>(3, i) = u[i];
//     }

//     // Calculate determinant
//     double det = J_matrix.determinant();

//     // Guard against true physical singularities
//     if (std::abs(det) < 1e-10) {
//         return -1.0; 
//     }

//     // 5. Return the INVERSE Jacobian to match thermodynamic requirements
//     return 1.0 / std::abs(det); 
// }

// KIC FIX (F6): orientation-free Jacobian. The previous body (legacy jac_local.h:114-130) used
// the lab x/y (or x/z) components of the CA3->C3 bond; the phi driver rotates that bond, so its J
// was J_true / |u_z| and a phi-driver move's weight depended on how the molecule sat in the lab
// (checked to 1e-12 on 1,240 closures). This is 1/|det| of the 6x6 matrix of Pluecker twists
// (u_i, p_i x u_i) of the six window torsion axes (phi1, psi1, phi2, psi2, phi3, psi3): invariant
// under any rigid motion, equal to J_true. Moments are taken about CA1 to keep them small.
double TripeptideSolver::calculate_jacobian(const Solution& sol) const
{
    using Mat66 = Eigen::Matrix<double, 6, 6>;
    const Eigen::Vector3d o = sol.r_a[0];
    Mat66 m;
    for (int i = 0; i < 3; ++i) {
        const Eigen::Vector3d u_phi = (sol.r_a[i] - sol.r_n[i]).normalized();  // N_i -> CA_i
        const Eigen::Vector3d u_psi = (sol.r_c[i] - sol.r_a[i]).normalized();  // CA_i -> C_i
        m.col(2 * i).head<3>() = u_phi;
        m.col(2 * i).tail<3>() = (sol.r_a[i] - o).cross(u_phi);
        m.col(2 * i + 1).head<3>() = u_psi;
        m.col(2 * i + 1).tail<3>() = (sol.r_c[i] - o).cross(u_psi);
    }
    const double det = m.determinant();

    if (!(std::abs(det) >= 1.0e-10)) {   // near-singular (or non-finite): caller rejects
        return -1.0;
    }

    return 1.0 / std::abs(det);
}