#pragma once
#include <vector>
#include <cassert>
#include "pymcpu/Potential.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/utils/numbers_compat.h"
#include <stdexcept>
#include <string>

namespace mcpu {
    class Context;
    struct State;
}

namespace mcpu::forces {
    static constexpr float kPi = mcpu::PI_F;

    class TripletPotential : public Potential {
    private:
        std::vector<float> params;
        int n_pos = 0;
        
        static constexpr int BB_DIM = 6;
        static constexpr int STRIDE1 = BB_DIM;
        static constexpr int STRIDE2 = BB_DIM * BB_DIM;
        static constexpr int STRIDE3 = BB_DIM * BB_DIM * BB_DIM;
        static constexpr int STRIDE4 = BB_DIM * BB_DIM * BB_DIM * BB_DIM;
        static constexpr float BB_BIN_SIZE = 60.0f * (mcpu::PI_F / 180.0f);
        static constexpr float BB_BIN_SIZE_30 = 30.0f * (mcpu::PI_F / 180.0f);

        float get(int r, int a, int b, int c, int d) const {
            size_t index = (static_cast<size_t>(r) * STRIDE4) + 
                           (a * STRIDE3) + (b * STRIDE2) + (c * STRIDE1) + d;
            // FIXED: this was assert-only, and CMakeLists defines NDEBUG for both
            // Release and RelWithDebInfo, so the only bounds guard on a raw
            // vector index was compiled out of every shipped build. A real check
            // is affordable here -- get() is per-residue, not the Mu pair loop.
            if (index >= params.size()) {
                throw std::out_of_range(
                    "TripletPotential::get(): index " + std::to_string(index) +
                    " >= params.size() " + std::to_string(params.size()) +
                    " (r=" + std::to_string(r) + " a=" + std::to_string(a) +
                    " b=" + std::to_string(b) + " c=" + std::to_string(c) +
                    " d=" + std::to_string(d) + ")");
            }
            return params[index];
        }

        /// phi/psi bins. periodic: phi+pi, psi+pi span [0, 2pi], so a value
        /// landing exactly on 2pi must wrap to bin 0. `% BB_DIM` does that and
        /// matches the table's 6-wide dimension -- correct as written.
        int get_bin_60(float angle) const {
            int bin = static_cast<int>(angle / BB_BIN_SIZE) % BB_DIM;
            if (bin < 0) bin += BB_DIM;
            return bin;
        }

        /// pCA/bCA bins. NOT periodic: both come from
        /// acos(clamp(dot,-1,1)), so they span the CLOSED interval [0, pi] and
        /// angle/(pi/6) attains exactly 6.0 at pi (antiparallel planes or
        /// bisectors, which is geometrically reachable).
        ///
        /// FIXED: this used `% 12` against a table whose corresponding
        /// dimension is BB_DIM == 6. 6 % 12 == 6, so an angle of exactly pi
        /// produced an in-range-looking bin that overflowed the per-residue
        /// 6^4 == 1296 block by up to 1511 elements -- a silent out-of-bounds
        /// read, since get()'s only guard was an assert removed by NDEBUG.
        /// Clamping (not wrapping) is the right treatment for a non-periodic
        /// angle: pi is the top edge of the last bin, not the bottom of the
        /// first. This also makes a NaN angle safe -- static_cast<int>(NaN) is
        /// UB and typically yields INT_MIN, which the `< 0` branch maps to 0.
        int get_bin_30(float angle) const {
            int bin = static_cast<int>(angle / BB_BIN_SIZE_30);
            if (bin < 0) return 0;
            if (bin >= BB_DIM) return BB_DIM - 1;
            return bin;
        }

    public:
        explicit TripletPotential(std::vector<float> loaded_params);

        /// Stores no atom ids: it reads State torsions and System blocks,
        /// which are remapped for it.
        void permute_atom_indices(const AtomPermutation&) override {}

        EnergyChangeResult calculateEnergyChange(
            const Context& context, 
            const State& old_state, 
            const State& proposed_state, 
            const ProposalPatch& patch
        ) const override;

        double calculateEnergy(
            const Context& context, 
            const State& state
        ) const override;
    };

} // namespace mcpu::forces