#pragma once
#include <vector>
#include <cassert>
#include "pymcpu/Potential.h"
#include "pymcpu/ProposalPatch.h"

namespace mcpu {
    class Context;
    class State;
    class System;
}

namespace mcpu::forces {

    class HBondPotential : public Potential {
    private:
        std::vector<float> params;
        // legacy hbonds.h seq_hb[3][20][20]: multiplicative sequence-dependent
        // scaling of the table lookup, indexed by (helix_sheet, donor aa, acceptor aa).
        std::vector<float> seq_dep_params;

        static constexpr int HBOND_DIM = 9;
        static constexpr int STRIDE1 = HBOND_DIM;
        static constexpr int STRIDE2 = HBOND_DIM * HBOND_DIM;
        static constexpr int STRIDE3 = HBOND_DIM * HBOND_DIM * HBOND_DIM;
        static constexpr int STRIDE4 = HBOND_DIM * HBOND_DIM * HBOND_DIM * HBOND_DIM;
        static constexpr int STRIDE5 = HBOND_DIM * HBOND_DIM * HBOND_DIM * HBOND_DIM * HBOND_DIM;
        static constexpr int STRIDE6 = HBOND_DIM * HBOND_DIM * HBOND_DIM * HBOND_DIM * HBOND_DIM * HBOND_DIM;
        static constexpr int N_AMINO = 20;
        // legacy hbonds.h: `e += beta_favor * seq_hb[...] * hbond_E[...]` for res_dif>4.
        static constexpr float BETA_FAVOR = 3.0f;

        inline float get(int i, int a, int b, int c, int d, int e, int f) const {
            size_t index = (static_cast<size_t>(i) * STRIDE6) +
                           (a * STRIDE5) + (b * STRIDE4) +
                           (c * STRIDE3) + (d * STRIDE2) +
                           (e * STRIDE1) + f;
            assert(index < params.size() && "HBondPotential::get() index out of bounds");
            return params[index];
        }

        inline float seq_dep_factor(int helix_sheet, int aa_don, int aa_acc) const {
            size_t index = (static_cast<size_t>(helix_sheet) * N_AMINO * N_AMINO)
                          + (static_cast<size_t>(aa_don) * N_AMINO)
                          + static_cast<size_t>(aa_acc);
            assert(index < seq_dep_params.size() && "HBondPotential::seq_dep_factor() index out of bounds");
            return seq_dep_params[index];
        }

        // The core physics helper, now taking the System instead of Topology
        float evaluate_directional(
            int r_don,
            int r_acc,
            const State& state,
            const System& sys
        ) const;

    public:
        HBondPotential(std::vector<float> loaded_params, std::vector<float> seq_dep_params);

        /// Stores no atom ids: it reads State torsions and System blocks,
        /// which are remapped for it.
        void permute_atom_indices(const AtomPermutation&) override {}

        EnergyChangeResult calculateEnergyChange(
            const Context& context, 
            const State& old_state, 
            const State& proposed_state, 
            const ProposalPatch& patch
        ) const override;

        float calculateEnergy(
            const Context& context, 
            const State& state
        ) const override;
    };

} // namespace mcpu::forces