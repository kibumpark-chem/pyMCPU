#include <cmath>
#include <algorithm>
#include <codecvt>
#include "pymcpu/utils/numbers_compat.h"
#include <iostream>
#include <iomanip>

#include "pymcpu/moves/TripeptideClosure.h"
#include "pymcpu/moves/TripeptideClosureUtils.h"
#include "pymcpu/moves/SturmSolver.h"

using namespace TripeptideLoopClosure;

int TripeptideSolver::solv_3pep_poly(const Vec3& r_n1, const Vec3& r_a1, 
                                     const Vec3& r_a3, const Vec3& r_c3, 
                                     std::vector<Solution>& solutions) {
    int n_soln = 0;
    solutions.clear();

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
    std::array<double, 3> cos_sig, sin_sig, half_tan;

    for (size_t i_soln = 0; i_soln < roots.size(); ++i_soln) {
        half_tan[2] = roots[i_soln];
        half_tan[1] = calc_t2(half_tan[2]);
        half_tan[0] = calc_t1(half_tan[2], half_tan[1]);

        for (int i = 1; i < 4; ++i) {
            double ht = half_tan[i-1];
            double tmp = 1.0 + ht * ht;
            cos_tau[i] = (1.0 - ht * ht) / tmp;
            sin_tau[i] = 2.0 * ht / tmp;
        }
        
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

double TripeptideSolver::calculate_jacobian(const Solution& sol) const
{
    using Vec3 = Eigen::Vector3d;
    using Mat66 = Eigen::Matrix<double, 6, 6>;

    // 1. Build axes and pivots exactly like legacy code
    std::array<Vec3, 6> axis;
    std::array<Vec3, 6> pivot;

    // Residue 1
    axis[0]  = (sol.r_a[0] - sol.r_n[0]);  // phi1: N1->CA1
    axis[0].normalize();
    pivot[0] = sol.r_a[0];                 // pivot at CA1

    axis[1]  = (sol.r_c[0] - sol.r_a[0]);  // psi1: CA1->C1
    axis[1].normalize();
    pivot[1] = sol.r_c[0];                 // pivot at C1

    // Residue 2
    axis[2]  = (sol.r_a[1] - sol.r_n[1]);  // phi2
    axis[2].normalize();
    pivot[2] = sol.r_a[1];

    axis[3]  = (sol.r_c[1] - sol.r_a[1]);  // psi2
    axis[3].normalize();
    pivot[3] = sol.r_c[1];

    // Residue 3
    axis[4]  = (sol.r_a[2] - sol.r_n[2]);  // phi3
    axis[4].normalize();
    pivot[4] = sol.r_a[2];

    axis[5]  = (sol.r_c[2] - sol.r_a[2]);  // psi3
    axis[5].normalize();
    pivot[5] = sol.r_c[2];

    // 2. End point (CA3) and final bond direction (CA3 -> C3)
    Vec3 r_ca3  = sol.r_a[2];
    Vec3 r_cac3 = (sol.r_c[2] - sol.r_a[2]).normalized();

    // 3. Choose which components (c1, c2) to use for last two rows, as in legacy
    int c1, c2;
    if (std::abs(r_cac3.z()) < 1.0e-10) {
        // x and z components: Fortran (4,6) -> zero-based (3,5)
        c1 = 3;
        c2 = 5;
    } else {
        // x and y components: Fortran (4,5) -> zero-based (3,4)
        c1 = 3;
        c2 = 4;
    }

    // 4. Build the 6x6 j matrix like the legacy routine
    Mat66 j = Mat66::Zero();

    // j(1..4, 1..3): axis x (r_ca3 - pivot)
    for (int n = 0; n < 4; ++n) {
        Vec3 v = axis[n].cross(r_ca3 - pivot[n]);
        j(n, 0) = v.x();
        j(n, 1) = v.y();
        j(n, 2) = v.z();
    }

    // j(1..5, 4..6): axis x r_cac3
    for (int n = 0; n < 5; ++n) {
        Vec3 v = axis[n].cross(r_cac3);
        j(n, 3) = v.x();
        j(n, 4) = v.y();
        j(n, 5) = v.z();
    }

    // 5. Helper det3() with the same orientation as legacy
    auto det3 = [](const Vec3& j1, const Vec3& j2, const Vec3& j3) -> double {
        // Replicates your C det3 mapping:
        // j11 = j1[0]; j12 = j2[0]; j13 = j3[0];
        // j21 = j1[1]; j22 = j2[1]; j23 = j3[1];
        // j31 = j1[2]; j32 = j2[2]; j33 = j3[2];
        double j11 = j1[0], j12 = j2[0], j13 = j3[0];
        double j21 = j1[1], j22 = j2[1], j23 = j3[1];
        double j31 = j1[2], j32 = j2[2], j33 = j3[2];

        return j11 * (j22*j33 - j23*j32)
             - j12 * (j21*j33 - j23*j31)
             + j13 * (j21*j32 - j22*j31);
    };

    // 6. Build the 3-vectors used in the 3x3 determinants
    Vec3 va_1, va_2, va_3;
    Vec3 vb_1, vb_2, vb_3;
    Vec3 vc_1, vc_2, vc_3;
    Vec3 vd_1, vd_2, vd_3;

    for (int i = 0; i < 3; ++i) {
        va_1[i] = j(1, i);
        va_2[i] = j(2, i);
        va_3[i] = j(3, i);

        vb_1[i] = j(0, i);
        vb_2[i] = j(2, i);
        vb_3[i] = j(3, i);

        vc_1[i] = j(0, i);
        vc_2[i] = j(1, i);
        vc_3[i] = j(3, i);

        vd_1[i] = j(0, i);
        vd_2[i] = j(1, i);
        vd_3[i] = j(2, i);
    }

    // 7. Determinant using the exact cofactor expression
    double det =
        -(j(4, c2)*j(0, c1) - j(4, c1)*j(0, c2)) * det3(va_1, va_2, va_3)
        +(j(4, c2)*j(1, c1) - j(4, c1)*j(1, c2)) * det3(vb_1, vb_2, vb_3)
        -(j(4, c2)*j(2, c1) - j(4, c1)*j(2, c2)) * det3(vc_1, vc_2, vc_3)
        +(j(4, c2)*j(3, c1) - j(4, c1)*j(3, c2)) * det3(vd_1, vd_2, vd_3);

    if (std::abs(det) < 1.0e-10) {
        // If you want to fully match the original, you’d also implement
        // the "random rotation and retry" logic here.
        return -1.0;
    }

    return 1.0 / std::abs(det);
}