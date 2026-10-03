#pragma once

#include <cstdint>
#include <vector>

#include "pymcpu/Potential.h"

namespace mcpu {
class Context;
struct QBiasWorkspace;
}

namespace mcpu::forces {

/// Harmonic umbrella bias on the hard native-contacts **count** N.
///
/// N = number of formed native CA pairs, where a pair is formed when its
/// current CA–CA distance is below ``q_cutoff``.
///
/// Bias energy:  0.5 * k_bias * (N - N0)^2
///
/// Fraction Q = N / n_pairs is a convenience observable only (Python CV);
/// the C++ bias and Context target store **N0**, not Q*.
///
/// Per-replica ``k_bias`` and ``N0`` live on Context. The native pair list and
/// cutoff are static and shared across replicas on the same System.
class QBiasPotential : public Potential {
public:
    explicit QBiasPotential(
        std::vector<int> ca_atom_i,
        std::vector<int> ca_atom_j,
        float q_cutoff
    );

    int numPairs() const noexcept { return n_pairs_; }

    void initializePairCache(const State& state, std::vector<uint8_t>& cache) const;

    float calculateEnergy(const Context& context, const State& state) const override;
    EnergyChangeResult calculateEnergyChange(
        const Context& context,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& patch
    ) const override;

    /// Pair endpoints are build-order atom ids; init_only maps them to storage
    /// order. Pair numbers, and so State::q_pair_cache's layout, are unchanged.
    void permute_atom_indices(const AtomPermutation& perm) override;

private:
    std::vector<int> pairs_i_;
    std::vector<int> pairs_j_;
    float q_cutoff_sq_ = 0.0f;
    int n_pairs_ = 0;
    /// pair indices touching each atom (for incremental delta evaluation)
    std::vector<std::vector<int>> atom_to_pairs_;

    void rebuild_atom_to_pairs();
    bool pairFormed(const State& state, int pair_idx) const noexcept;
    static float biasEnergy(float n, float k_bias, float n_target) noexcept;
};

}  // namespace mcpu::forces
