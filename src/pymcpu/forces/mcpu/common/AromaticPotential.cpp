#include "pymcpu/forces/mcpu/common/AromaticPotential.h"
#include "pymcpu/Context.h"
#include "pymcpu/State.h"
#include <algorithm>
#include <cmath>
#include "pymcpu/utils/numbers_compat.h"

namespace mcpu::forces {
    AromaticPotential::AromaticPotential(
        std::vector<std::array<int, 3>> aromatic_atom_indices,
        std::vector<float> loaded_params
    ) : aromatic_atom_indices(std::move(aromatic_atom_indices)),
        params(std::move(loaded_params)) {
    }

    void AromaticPotential::permute_atom_indices(const AtomPermutation& perm) {
        if (perm.is_identity()) return;
        for (auto& ring : aromatic_atom_indices) {
            for (int& idx : ring) {
                if (idx >= 0) idx = perm.to_internal(idx);
            }
        }
    }

    AromaticPotential::RingGeometry AromaticPotential::computeRingGeometry(
        const State& state,
        const std::array<int, 3>& ring
    ) const {
        Eigen::Vector3f atom1_pos = state.atom_pos(ring[0]);
        Eigen::Vector3f atom2_pos = state.atom_pos(ring[1]);
        Eigen::Vector3f atom3_pos = state.atom_pos(ring[2]);

        RingGeometry geom;
        geom.center = (atom1_pos + atom2_pos + atom3_pos) / 3.0f;
        // Plane normal = (CG->outer1) x (CG->outer2)
        geom.normal = (atom2_pos - atom1_pos).cross(atom3_pos - atom1_pos);
        return geom;
    }

    float AromaticPotential::calculateEnergy(
        const Context& context,
        const State& state
    ) const {
        constexpr float RAD2DEG = 180.0f / mcpu::PI_F;
        constexpr float EPS = 1e-6f;

        float total_energy = 0.0f;
        const std::size_t n = aromatic_atom_indices.size();
        const auto& sys = context.getSystem();
        const auto& a2r = sys.atom_to_residue;

        for (std::size_t i = 0; i < n; ++i) {
            const int res_i = a2r[static_cast<std::size_t>(aromatic_atom_indices[i][0])];
            if (sys.is_residue_energy_ignored(res_i)) continue;

            RingGeometry ring_i = computeRingGeometry(state, aromatic_atom_indices[i]);
            float norm_i = ring_i.normal.norm();
            if (norm_i < EPS) continue; // degenerate / collinear ring guard

            for (std::size_t j = i + 1; j < n; ++j) {
                const int res_j = a2r[static_cast<std::size_t>(aromatic_atom_indices[j][0])];
                if (sys.is_residue_energy_ignored(res_j)) continue;

                RingGeometry ring_j = computeRingGeometry(state, aromatic_atom_indices[j]);

                float distance_sq = (ring_i.center - ring_j.center).squaredNorm();
                if (distance_sq >= DISTANCE_CUTOFF_SQ) continue;

                float norm_j = ring_j.normal.norm();
                if (norm_j < EPS) continue;

                // Angle between ring planes, clamped to a valid acos domain.
                float cos_angle = ring_i.normal.dot(ring_j.normal) / (norm_i * norm_j);
                cos_angle = std::clamp(cos_angle, -1.0f, 1.0f);
                float angle = std::acos(cos_angle) * RAD2DEG;

                // Fold to [0, 90) and bin (matches reference aromatic_plane / aromaticenergy).
                if (angle > 90.0f) angle = 180.0f - angle;
                if (angle > MAX_ANGLE_DEG) angle = MAX_ANGLE_DEG;

                int bin = static_cast<int>(angle / ANGLE_BIN_SIZE_DEG);
                if (bin < 0) bin = 0;
                if (bin >= NUM_ANGLE_BINS) bin = NUM_ANGLE_BINS - 1;

                total_energy += get(bin);
            }
        }
        return total_energy / 1000.0f;
    }

    EnergyChangeResult AromaticPotential::calculateEnergyChange(
        const Context& context,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& /*patch*/
    ) const {
        const float de =
            calculateEnergy(context, proposed_state) - calculateEnergy(context, old_state);
        return EnergyChangeResult::finite(de);
    }
}
