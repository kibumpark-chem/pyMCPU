#pragma once
#include <vector>
#include "pymcpu/Potential.h"
#include "pymcpu/ProposalPatch.h"
#include <cassert>
#include "pymcpu/utils/numbers_compat.h"
#include <cmath>

namespace mcpu {
    class Context;
    struct State;
}

namespace mcpu::forces {

    class SidechainTripletPotential : public Potential {
    private:
        std::vector<float> params;
        int n_pos = 0;
        
        static constexpr int SC_DIM = 12;
        static constexpr int STRIDE1 = SC_DIM;
        static constexpr int STRIDE2 = SC_DIM * SC_DIM;
        static constexpr int STRIDE3 = SC_DIM * SC_DIM * SC_DIM;
        static constexpr int STRIDE4 = SC_DIM * SC_DIM * SC_DIM * SC_DIM;
        static constexpr float SC_BIN_SIZE = 30.0f * (mcpu::PI_F / 180.0f);
        static constexpr float SC_BIN_SIZE_HALF = 15.0f * (mcpu::PI_F / 180.0f);

        float get(int r, int a, int b, int c, int d) const {
            size_t index = (static_cast<size_t>(r) * STRIDE4) + 
                           (a * STRIDE3) + (b * STRIDE2) + (c * STRIDE1) + d;
            assert(index < params.size() && "SidechainTripletPotential::get() index out of bounds");
            return params[index];
        }

        int get_bin(float angle) const {
            // Normalize angle to [0, 2pi) before binning to handle
            // negative torsion angles from the [-pi, pi] convention
            angle = std::fmod(angle + mcpu::PI_F + SC_BIN_SIZE_HALF, 2.0f * mcpu::PI_F);
            int bin = static_cast<int>(angle / SC_BIN_SIZE) % 12;
            if (bin < 0) bin += SC_DIM;
            return bin;
        }

    public:
        explicit SidechainTripletPotential(std::vector<float> loaded_params);

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