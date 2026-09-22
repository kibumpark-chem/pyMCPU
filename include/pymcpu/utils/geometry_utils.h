#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <algorithm> // for std::clamp

namespace GeometryUtils {
    inline float calculate_angle(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
        float dot_product = v1.normalized().dot(v2.normalized());
        return std::acos(std::clamp(dot_product, -1.0f, 1.0f));
    }

    inline float calculate_bond_angle(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2, const Eigen::Vector3f& v3) {
        Eigen::Vector3f vec1 = v1 - v2;
        Eigen::Vector3f vec2 = v3 - v2;
        return calculate_angle(vec1, vec2);
    }

    // Computes the dihedral angle defined by four points: p1-p2-p3-p4
    inline float calculate_dihedral(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, 
                                    const Eigen::Vector3f& p3, const Eigen::Vector3f& p4) {
        // Calculate the three vectors between the four points
        Eigen::Vector3f b1 = p2 - p1;
        Eigen::Vector3f b2 = p3 - p2;
        Eigen::Vector3f b3 = p4 - p3;

        // Calculate the normals to the two planes
        Eigen::Vector3f n1 = b1.cross(b2);
        Eigen::Vector3f n2 = b2.cross(b3);

        // Compute the x and y coordinates for atan2
        // x represents the cosine of the angle, scaled by |n1||n2|
        float x = n1.dot(n2);

        // y represents the sine of the angle.
        // Mathematically, y = (n1 cross n2) dot (b2 / |b2|).
        // Using the vector triple product identity, this simplifies to:
        // y = (b1 dot n2) * |b2|, saving us an entire cross product operation!
        float y = b1.dot(n2) * b2.norm(); 

        // 4. atan2 inherently handles signs and correctly maps the angle
        float angle = std::atan2(y, x);

        return angle; 
    }

    inline Eigen::Vector3f bisector(const Eigen::Vector3f& v1, const Eigen::Vector3f& v2) {
        return (v1.normalized() + v2.normalized()).normalized();
    }

    inline Eigen::Vector3f plane_normal(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3) {
        // p2 is the central point, so we compute vectors from p2 to p1 and p3
        Eigen::Vector3f v1 = p1 - p2;
        Eigen::Vector3f v2 = p3 - p2;
        return v1.cross(v2).normalized();
    }

    // Computes the a_PCA (Interplanar) angle between two residues
    inline float calculate_a_PCA(const Eigen::Vector3f& N1, const Eigen::Vector3f& CA1, const Eigen::Vector3f& O1,
                                 const Eigen::Vector3f& N2, const Eigen::Vector3f& CA2, const Eigen::Vector3f& O2) {
        Eigen::Vector3f plane1 = plane_normal(N1, CA1, O1);
        Eigen::Vector3f plane2 = plane_normal(N2, CA2, O2);

        return calculate_angle(plane1, plane2);
    }

    // Computes the a_bCA (Bisector) angle between two residues
    inline float calculate_a_bCA(const Eigen::Vector3f& N1, const Eigen::Vector3f& CA1, const Eigen::Vector3f& O1,
                                 const Eigen::Vector3f& N2, const Eigen::Vector3f& CA2, const Eigen::Vector3f& O2) {
        Eigen::Vector3f v1 = N1 - CA1;
        Eigen::Vector3f v2 = O1 - CA1;
        Eigen::Vector3f v3 = N2 - CA2;
        Eigen::Vector3f v4 = O2 - CA2;

        Eigen::Vector3f bisect1 = bisector(v1, v2);
        Eigen::Vector3f bisect2 = bisector(v3, v4);

        return calculate_angle(bisect1, bisect2);
    }

    inline float calculate_a_CACA(const Eigen::Vector3f& CA1a, const Eigen::Vector3f& CA2a,
                                  const Eigen::Vector3f& CA1b, const Eigen::Vector3f& CA2b) {
        Eigen::Vector3f v1 = CA1a - CA1b;
        Eigen::Vector3f v2 = CA2a - CA2b;
        return calculate_angle(v1, v2);
    }
} // namespace GeometryUtils