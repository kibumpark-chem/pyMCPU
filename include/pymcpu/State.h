#pragma once
#include <Eigen/Dense>
#include <array>
#include <vector>
#include "pymcpu/utils/numbers_compat.h"
#include <cstdint>
#include <stdexcept>
#include <string>

#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/CoordView.h"
#include "pymcpu/neighbor/PairLedger.h"
#include "pymcpu/forces/mcpu/common/HBondStateCache.h"

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
    /// Pairs (j, energy) per atom; ready() once built.
    mutable neighbor::PairLedger<float> mu_contacts;
    /// The list was filled by a full-energy resync while not ready. The next
    /// move that can use a list adopts it instead of rebuilding the same list
    /// (MuPotential::calculateEnergyChange); until then it reads as not ready,
    /// so every other path behaves as if it did not exist.
    mutable bool mu_contact_list_prebuilt = false;
    /// Upper bound (A) on how far any pair's distance can have changed through
    /// accepted rigid carries since the list was last measured from coordinates.
    mutable double mu_list_drift = 0.0;
    /// System::energy_mask_epoch() the list was built under. Masked pairs are
    /// never listed, so a list from another mask is stale.
    mutable std::uint64_t mu_list_mask_epoch = 0;

    /// Throw the list away; the Mu potential rebuilds it on next use. O(N).
    void mu_contact_invalidate() const {
        mu_contacts.invalidate();
        mu_contact_list_prebuilt = false;
        mu_list_drift = 0.0;
    }

    /// Hard-Q native pair cache (accepted state). Empty on proposal buffers.
    std::vector<uint8_t> q_pair_cache;

    /// Nonzero H-bond pair energies of this state (see HBondStateCache.h).
    /// Dropped on copy, and by anything that replaces the coordinates.
    mutable forces::HBondStateCache hbond_cache;

    /// Drop every cache that describes the current coordinates. Call this
    /// when coordinates change outside an accepted move.
    void invalidate_coordinate_caches() const {
        mu_contact_invalidate();
        hbond_cache.invalidate();
    }

    double getEnergy() const noexcept { return current_energy; }

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
        coords_soa.load_from_eigen(m);
        invalidate_coordinate_caches();
    }
    [[nodiscard]] Eigen::Matrix3Xf coords_as_eigen() const {
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
        hbond_cache.invalidate();
    }

private:
    double current_energy = 0.0;
    friend class Context;
    friend class MCIntegrator;
};

} // namespace mcpu
