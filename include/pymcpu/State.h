#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>
#include "pymcpu/utils/numbers_compat.h"
#include <cstdint>
#include <stdexcept>
#include <string>

#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/CoordSyncStats.h"
#include "pymcpu/utils/CoordView.h"

namespace mcpu {

struct BackboneTorsionAngles {
    float phi;
    float psi;
    float pCA; // Interplanar angle
    float bCA; // Bisector angle

    BackboneTorsionAngles() : phi(0.0f), psi(0.0f), pCA(0.0f), bCA(0.0f) {}
    BackboneTorsionAngles(float p, float s, float pc, float bc) : phi(p), psi(s), pCA(pc), bCA(bc) {}
};

struct SidechainTorsionAngles {
    std::array<float, 4> chi_angles = {-mcpu::PI_F, -mcpu::PI_F, -mcpu::PI_F, -mcpu::PI_F};
};

class State {
public:
    /// Storage-of-record: contiguous x[], y[], z[] (SoA layout).
    CoordsSoA coords_soa;

    std::vector<BackboneTorsionAngles> backbone_torsions;
    std::vector<SidechainTorsionAngles> sidechain_torsions;


    /// Live Mu contact list: for each atom, who it is CURRENTLY in contact with
    /// and what that contact is worth. The Mu delta reads its "energy before the
    /// move" straight off this instead of re-walking the old neighbourhood.
    /// It also holds, with energy 0, every contact pair just outside its cutoff
    /// (within MuPotential::kContactBandA), so that a rigid pivot can re-decide
    /// each listed pair it carries; mu_list_drift bounds how far the unlisted
    /// ones can have moved since the list was measured (see MuPotential).
    ///
    /// This lives on State, not on MuPotential, and that placement is load-bearing:
    /// partition_replicas() hands ONE System (hence one MuPotential) to every
    /// replica a rank owns, so a list owned by the potential would be shared between
    /// replicas that have completely different coordinates. It is per-State, like
    /// q_pair_cache, so each replica has its own.
    ///
    /// Only the ACCEPTED state carries one; proposal buffers leave it empty
    /// (copy_dynamic_from does not copy it). Built from the coordinates on the
    /// first move that needs it, kept up to date by each accepted move, and
    /// rewritten whenever Context::calculate_total_energy(-1) resets the
    /// running energy (MuPotential::resyncEnergy). Discarded by anything that
    /// replaces the coordinates wholesale, by a move that cannot use it, and
    /// by such a reset under a residue energy mask.
    struct MuContactEntry {
        std::int32_t j;
        float energy;
    };
    mutable std::vector<std::vector<MuContactEntry>> mu_contact_list;
    mutable bool mu_contact_list_ready = false;
    /// Upper bound (A) on how far any pair's distance can have changed through
    /// accepted rigid carries since the list was last measured from coordinates.
    mutable float mu_list_drift = 0.f;

    /// List the pair (i, j), worth `e` (0 for a near miss). O(1) amortized.
    void mu_contact_add(int i, int j, float e) const {
        mu_contact_list[static_cast<size_t>(i)].push_back(
            MuContactEntry{static_cast<std::int32_t>(j), e});
        mu_contact_list[static_cast<size_t>(j)].push_back(
            MuContactEntry{static_cast<std::int32_t>(i), e});
    }

    /// Unlist the pair (i, j). O(degree).
    void mu_contact_remove(int i, int j) const {
        auto drop = [&](int a, int b) {
            auto& v = mu_contact_list[static_cast<size_t>(a)];
            for (size_t k = 0; k < v.size(); ++k) {
                if (v[k].j == b) {
                    v[k] = v.back();
                    v.pop_back();
                    return;
                }
            }
        };
        drop(i, j);
        drop(j, i);
    }

    /// Throw the list away; the Mu potential rebuilds it on next use. O(N).
    void mu_contact_invalidate() const {
        mu_contact_list.clear();
        mu_contact_list_ready = false;
        mu_list_drift = 0.f;
    }

    [[nodiscard]] bool has_mu_contact_list() const noexcept {
        return mu_contact_list_ready;
    }
    /// Hard-Q native pair cache (accepted state). Empty on proposal buffers.
    std::vector<uint8_t> q_pair_cache;

    float getEnergy() const noexcept { return current_energy; }

    explicit State(int num_atoms, int num_residues)
        : coords_soa(num_atoms)
        , backbone_torsions(num_residues)
        , sidechain_torsions(num_residues)
        , current_energy(0.0f)
    {}

    State(const State&) = default;
    State& operator=(const State&) = default;
    State(State&&) = default;
    State& operator=(State&&) = default;

    /// Hot-path read view (no allocation).
    [[nodiscard]] CoordView coord_view() const noexcept { return CoordView(coords_soa); }

    [[nodiscard]] Eigen::Vector3f atom_pos(int i) const { return coords_soa.atom(i); }
    void set_atom_pos(int i, const Eigen::Vector3f& p) { coords_soa.set_atom(i, p); }
    void copy_atom_from(const State& src, int dst, int src_idx) {
        coords_soa.copy_atom_from(src.coords_soa, dst, src_idx);
    }

    void set_coords_from_eigen(const Eigen::Matrix3Xf& m) {
        if (m.cols() != coords_soa.n) {
            throw std::invalid_argument(
                "State.coords: got coordinates for " + std::to_string(m.cols()) +
                " atoms, but this state has " + std::to_string(coords_soa.n));
        }
        note_coords_eigen_write_back();
        coords_soa.load_from_eigen(m);
        mu_contact_invalidate();
    }
    [[nodiscard]] Eigen::Matrix3Xf coords_as_eigen() const {
        note_coords_eigen_materialization();
        return coords_soa.as_eigen();
    }

    void rotate_atoms(int start, int end, const Eigen::Matrix3d& R, const Eigen::Vector3d& pivot) {
        coords_soa.rotate_atoms(start, end, R, pivot);
    }

    /// Copy coords, torsions, and energy only — not the Mu contact list or
    /// q_pair_cache.
    void copy_dynamic_from(const State& src) {
        coords_soa.copy_all_from(src.coords_soa);
        backbone_torsions = src.backbone_torsions;
        sidechain_torsions = src.sidechain_torsions;
        current_energy = src.current_energy;
    }

private:
    float current_energy = 0.0f;
    friend class Context;
    friend class MCIntegrator;
};

} // namespace mcpu
