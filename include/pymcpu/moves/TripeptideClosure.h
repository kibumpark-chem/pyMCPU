#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/Polynomials>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <string>

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;

constexpr int MAX_SOLN = 16; 

// Fixed-size, so a closure costs no heap allocation (a KIC step builds up to 2 x 16).
struct Solution {
    std::array<Vec3, 3> r_n;
    std::array<Vec3, 3> r_a;
    std::array<Vec3, 3> r_c;
};

// Removed the namespace wrapper so pybind11 can see this directly
class TripeptideSolver {
private:
    static constexpr double pi = 3.14159265358979323846;
    static constexpr double deg2rad = pi / 180.0;
    static constexpr double rad2deg = 180.0 / pi;
    static constexpr int max_soln = 16;

    int print_level = 0;
    bool is_initialized_ = false;

    // Module state variables
    std::array<double, 6> len0;
    std::array<double, 7> b_ang0;
    std::array<double, 2> t_ang0;
    double aa13_min_sqr, aa13_max_sqr;
    
    double delta[4], xi[3], eta[3], alpha[3], theta[3];
    double cos_alpha[3], sin_alpha[3], cos_theta[3], sin_theta[3];
    double cos_delta[4], sin_delta[4];
    double cos_xi[3], cos_eta[3], sin_xi[3], sin_eta[3];
    
    Vec3 r_a1a3, r_a1n1, r_a3c3;
    Vec3 b_a1a3, b_a1n1, b_a3c3;
    double len_na[3], len_ac[3], len_aa[3];

    // Stack-allocated matrix state for high performance
    Eigen::Matrix<double, 17, 3> R;
    Eigen::Matrix3d C0;
    Eigen::Matrix3d C1;
    Eigen::Matrix3d C2;

    // Closure equations in the (1, cos tau, sin tau) basis (build_trig_coeff), and the closure check.
    double trig_coeff_[3][3][3];
    int n_rejected_ = 0;
    static constexpr double kClosureTol = 1.0e-6;   // rad, on each of the three N-CA-C angles

public:
    void initialize(const std::array<double, 6>& b_len,
                                 const std::array<double, 7>& b_ang,
                                 const std::array<double, 2>& t_ang);

    std::vector<Solution> solve(const Vec3& r_n1, const Vec3& r_a1, 
                                const Vec3& r_a3, const Vec3& r_c3);

    int solv_3pep_poly(const Vec3& r_n1, const Vec3& r_a1, 
                       const Vec3& r_a3, const Vec3& r_c3, 
                       std::vector<Solution>& solutions);

    double calculate_jacobian(const Solution& sol) const;

    // Closures dropped by the N-CA-C check in the last solve.
    int last_rejected() const { return n_rejected_; }

    // For DEBUGGING: Expose internal state for testing
    std::vector<double> get_xi() const {
        return {xi[0], xi[1], xi[2]};
    }
    
    std::vector<double> get_eta() const {
        return {eta[0], eta[1], eta[2]};
    }

    std::vector<double> get_delta() const {
        return {delta[0], delta[1], delta[2], delta[3]};
    }

    std::vector<double> get_polynomial_coefficients() {
        // 1. Create a blank Eigen matrix just like your solve function does
        Eigen::Matrix<double, 17, 1> p_coeff = Eigen::Matrix<double, 17, 1>::Zero();
        
        // 2. Call your existing internal function to fill it
        // (solve() already set up the geometry it reads.)
        this->get_poly_coeff(p_coeff);
        
        // 3. Convert to a standard C++ vector for Python
        std::vector<double> coeffs(17);
        for(int i=0; i<17; ++i) {
            coeffs[i] = p_coeff[i]; 
        }
        return coeffs;
    }
private:
    // Helper signatures
    void get_input_angles(int& n_soln, const Vec3& r_n1, const Vec3& r_a1, 
                          const Vec3& r_a3, const Vec3& r_c3);
    
    void test_two_cone_existence_soln(double tt, double kx, double et, double ap,
                                      int& n_soln);

    void get_poly_coeff(Eigen::Matrix<double, 17, 1>& poly_coeff);
    void solve_roots(const Eigen::Matrix<double, 17, 1>& poly_coeff, std::vector<double>& roots);
    
    void coord_from_poly_roots(const std::vector<double>& roots, 
                               const Vec3& r_n1, const Vec3& r_a1, 
                               const Vec3& r_a3, const Vec3& r_c3,
                               std::vector<Solution>& solutions);

    void build_trig_coeff();
    void back_substitute(double c3, double s3, double& c1, double& s1, double& c2, double& s2) const;
    bool closes(const Solution& sol) const;
};