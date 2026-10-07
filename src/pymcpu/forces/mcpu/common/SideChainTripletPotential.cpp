#include "pymcpu/forces/mcpu/common/SideChainTripletPotential.h"
#include "pymcpu/Context.h"
#include "pymcpu/State.h"
#include <iostream>

namespace {
    /// True when this residue has no chi dihedrals (GLY, ALA). Both the delta and
    /// the full-energy loop gate on this, so they cannot disagree.
    inline bool residue_has_no_chi(const mcpu::System& sys, int residue) {
        const auto& ntor = sys.getTorsionsPerResidue();
        if (residue < 0 || residue >= static_cast<int>(ntor.size())) return false;
        return ntor[static_cast<size_t>(residue)] == 0;
    }
}  // namespace

namespace mcpu::forces {

    SidechainTripletPotential::SidechainTripletPotential(std::vector<float> loaded_params)
        : params(std::move(loaded_params))
    {
        if (params.size() % STRIDE1 != 0) {
            throw std::invalid_argument(
                "SidechainTripletPotential: params.size() must be divisible by STRIDE1 (20736)");
        }
        n_pos = static_cast<int>(params.size() / STRIDE4);
    }

    // ---------------------------------------------------------
    // THE DELTA ENERGY CALCULATOR
    // ---------------------------------------------------------
    EnergyChangeResult SidechainTripletPotential::calculateEnergyChange(
        const Context& context, 
        const State& old_state, 
        const State& proposed_state, 
        const ProposalPatch& patch
    ) const {
        double delta_energy = 0.0;
        
        const System& sys = context.getSystem();

        // Loop over the affected residue block
        for (int r : patch.distorted_sc_residues) {
            
            // Total E evaluates sidechain residue indices 1 through n_pos inclusive.
            if (r < 1 || r > n_pos) continue;
            if (sys.is_residue_energy_ignored(r)) continue;
            // A residue with no chi angles contributes nothing -- see the note in
            // calculateEnergy(). Must match that loop exactly or the incremental
            // and from-scratch energies drift apart.
            if (residue_has_no_chi(sys, r)) continue;

            const auto& old_chi = old_state.sidechain_torsions[r].chi_angles;
            float e_old = get(r - 1, 
                              get_bin(old_chi[0]), 
                              get_bin(old_chi[1]), 
                              get_bin(old_chi[2]), 
                              get_bin(old_chi[3]));

            const auto& new_chi = proposed_state.sidechain_torsions[r].chi_angles;
            float e_new = get(r - 1, 
                              get_bin(new_chi[0]), 
                              get_bin(new_chi[1]), 
                              get_bin(new_chi[2]), 
                              get_bin(new_chi[3]));

            delta_energy += static_cast<double>(e_new) - static_cast<double>(e_old);
        }

        // Must scale down by 1000.0 to match Total E
        return EnergyChangeResult::finite(delta_energy / 1000.0f);
    }

    // ---------------------------------------------------------
    // THE TOTAL ENERGY CALCULATOR (Ground Truth)
    // ---------------------------------------------------------
    double SidechainTripletPotential::calculateEnergy(const Context& context, const State& state) const {
        double total_energy = 0.0;
        const System& sys = context.getSystem();
        for (int r = 0; r < n_pos; ++r) {
            if (sys.is_residue_energy_ignored(r + 1)) continue;
            // FIXED: skip residues with no chi angles at all (GLY, ALA), as
            // legacy sctenergy() does:
            //     if (native_residue[i+1].ntorsions == 0) continue;
            //
            // Without this the residue still reads its table cell at chi bins
            // (0,0,0,0) -- and that cell holds +1000, the table's saturated
            // "never observed" sentinel, for 3390 of the 8000 residue-type
            // triplets. After the /1000 below and the 2.50 SCT weight that is
            // +2.5 per zero-chi residue, with the WRONG SIGN. On actin (58 such
            // residues) it came to +145, which was essentially the entire
            // sidechain-torsion disagreement with legacy (+140.3 measured).
            //
            // Note the chi DEFAULT is fine: -pi maps to bin 0, matching legacy's
            // i_ang[j] = 0 for unused slots, so residues with 1-3 real chi
            // angles already agreed.
            if (residue_has_no_chi(sys, r + 1)) continue;
            const auto& chi = state.sidechain_torsions[r + 1].chi_angles;
            total_energy += get(r, 
                                get_bin(chi[0]), 
                                get_bin(chi[1]), 
                                get_bin(chi[2]), 
                                get_bin(chi[3]));
        }

        return total_energy / 1000.0f;
    }

} // namespace mcpu::forces