#pragma once

#include <cmath>

/// Legacy MCPU CheckHBond() amide H placement (`hbonds.h`):
///   H_dir = (CA - N) + (C_prev - N)
///   H_dir = normalize(H_dir); H_dir = -H_dir
///   H = N + H_dir   // unit length → N–H = 1.0 Å
///
/// Inputs and outputs are Angstroms. Scalar, no allocation.
namespace HydrogenBondUtils {

inline void compute_virtual_amide_H(
    float nx, float ny, float nz,
    float cax, float cay, float caz,
    float cpx, float cpy, float cpz,
    float& hx, float& hy, float& hz) noexcept
{
    float vx = (cax - nx) + (cpx - nx);
    float vy = (cay - ny) + (cpy - ny);
    float vz = (caz - nz) + (cpz - nz);
    const float inv = 1.0f / std::sqrt(vx * vx + vy * vy + vz * vz);
    vx *= inv;
    vy *= inv;
    vz *= inv;
    hx = nx - vx;
    hy = ny - vy;
    hz = nz - vz;
}

inline void compute_virtual_amide_H(
    const float* N,
    const float* CA,
    const float* Cprev,
    float* H_out) noexcept
{
    compute_virtual_amide_H(
        N[0], N[1], N[2],
        CA[0], CA[1], CA[2],
        Cprev[0], Cprev[1], Cprev[2],
        H_out[0], H_out[1], H_out[2]);
}

}  // namespace HydrogenBondUtils
