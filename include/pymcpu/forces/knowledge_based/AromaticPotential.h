#pragma once
#include "pymcpu/Potential.h"
#include <Eigen/Dense>
#include <vector>
#include <array>
#include <cassert>
#include <cstddef>

namespace mcpu::forces {
    // Aromatic ring-ring orientation potential (PHE / TRP), faithful to the
    // original MCPU `aromaticenergy()`. Each ring is described by three atoms:
    //   index 0 = CG, index 1/2 = the two outer ring atoms
    //   (CE1/CE2 for PHE, CZ2/CZ3 for TRP).
    // The energy of a ring pair depends only on the angle between the two ring
    // planes, binned into 10-degree bins over [0, 90).
    class AromaticPotential : public Potential {
    private:
        std::vector<std::array<int, 3>> aromatic_atom_indices;
        std::vector<float> params;

        // Plane angle in [0, 90) degrees, 10-degree bins -> 9 bins.
        // Mirrors `short aromatic_E[9]` in the reference implementation.
        static constexpr int NUM_ANGLE_BINS = 9;
        static constexpr float ANGLE_BIN_SIZE_DEG = 10.0f;
        static constexpr float MAX_ANGLE_DEG = 89.9f;
        // Ring-center distance cutoff: AROMATIC_DISTANCE = 7.0 A, compared squared.
        static constexpr float DISTANCE_CUTOFF_SQ = 49.0f;

        struct RingGeometry {
            Eigen::Vector3f center;
            Eigen::Vector3f normal; // un-normalized; magnitude checked before use
        };

        RingGeometry computeRingGeometry(
            const State& state,
            const std::array<int, 3>& ring
        ) const;

        float get(int bin) const {
            assert(bin >= 0 && static_cast<std::size_t>(bin) < params.size() &&
                   "AromaticPotential::get() index out of bounds");
            return params[bin];
        }

    public:
        explicit AromaticPotential(
            std::vector<std::array<int, 3>> aromatic_atom_indices,
            std::vector<float> loaded_params
        );

        void permute_atom_indices(const AtomPermutation& perm) override;

        float calculateEnergy(
            const Context& context,
            const State& state
        ) const override;

        EnergyChangeResult calculateEnergyChange(
            const Context& context,
            const State& old_state,
            const State& proposed_state,
            const ProposalPatch& patch
        ) const override;
    };
}
