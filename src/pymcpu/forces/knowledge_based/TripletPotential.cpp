#include "pymcpu/forces/knowledge_based/TripletPotential.h"
#include "pymcpu/Context.h"
#include "pymcpu/State.h"
#include "pymcpu/utils/numbers_compat.h"

namespace mcpu::forces {

    TripletPotential::TripletPotential(std::vector<float> loaded_params) 
        : params(std::move(loaded_params)) {
        if (params.size() % STRIDE4 != 0) {
            throw std::invalid_argument(
                "TripletPotential: params.size() must be divisible by STRIDE4");
        }
        n_pos = static_cast<int>(params.size() / STRIDE4);
    }

    // ---------------------------------------------------------
    // THE DELTA ENERGY CALCULATOR
    // ---------------------------------------------------------
    EnergyChangeResult TripletPotential::calculateEnergyChange(
        const Context& context, 
        const State& old_state, 
        const State& proposed_state, 
        const ProposalPatch& patch
    ) const {
        float delta_energy = 0.0f;
        const float pi = mcpu::PI_F;
        
        const System& sys = context.getSystem();

        // Loop over the continuous affected block
        for (int r : patch.distorted_bb_residues) {
            
            // Total E evaluates torsion indices 1 through n_pos inclusive.
            if (r < 1 || r > n_pos) continue;
            if (sys.is_residue_energy_ignored(r)) continue;

            // Old State evaluation
            const auto& old_bb = old_state.backbone_torsions[r];
            float e_old = get(r - 1, 
                              get_bin_30(old_bb.pCA), 
                              get_bin_30(old_bb.bCA), 
                              get_bin_60(old_bb.phi + pi), 
                              get_bin_60(old_bb.psi + pi));

            // New State evaluation
            const auto& new_bb = proposed_state.backbone_torsions[r];
            float e_new = get(r - 1, 
                              get_bin_30(new_bb.pCA), 
                              get_bin_30(new_bb.bCA), 
                              get_bin_60(new_bb.phi + pi), 
                              get_bin_60(new_bb.psi + pi));

            delta_energy += (e_new - e_old);
        }

        // Must scale down by 1000 to match Total E
        return EnergyChangeResult::finite(delta_energy / 1000.0f);
    }

    // ---------------------------------------------------------
    // THE TOTAL ENERGY CALCULATOR (Ground Truth)
    // ---------------------------------------------------------
    float TripletPotential::calculateEnergy(const Context& context, const State& state) const {
        float total_energy = 0.0f;
        const float pi = mcpu::PI_F;
        const System& sys = context.getSystem();

        for (int r = 0; r < n_pos; ++r) {
            if (sys.is_residue_energy_ignored(r + 1)) continue;
            const auto& bb = state.backbone_torsions[r + 1];
            total_energy += get(r, 
                                get_bin_30(bb.pCA), 
                                get_bin_30(bb.bCA), 
                                get_bin_60(bb.phi + pi), 
                                get_bin_60(bb.psi + pi));
        }
        
        return total_energy / 1000.0f;
    }

} // namespace mcpu::forces