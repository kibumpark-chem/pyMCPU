#include "pymcpu/forces/bias/QBiasPotential.h"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "pymcpu/Context.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/State.h"
#include "pymcpu/utils/pair_r2.h"

namespace mcpu::forces {

QBiasPotential::QBiasPotential(
    std::vector<int> ca_atom_i,
    std::vector<int> ca_atom_j,
    float q_cutoff
)
    : pairs_i_(std::move(ca_atom_i)),
      pairs_j_(std::move(ca_atom_j)),
      q_cutoff_sq_(q_cutoff * q_cutoff) {
    if (pairs_i_.size() != pairs_j_.size()) {
        throw std::invalid_argument("QBiasPotential: pair index vectors differ in length");
    }
    n_pairs_ = static_cast<int>(pairs_i_.size());
    if (n_pairs_ == 0) {
        throw std::invalid_argument("QBiasPotential: native pair list is empty");
    }

    for (int p = 0; p < n_pairs_; ++p) {
        if (pairs_i_[p] < 0 || pairs_j_[p] < 0) {
            throw std::invalid_argument("QBiasPotential: negative atom index in pair list");
        }
    }
    rebuild_pair_atoms();
}

void QBiasPotential::rebuild_pair_atoms() {
    int max_atom = 0;
    for (int p = 0; p < n_pairs_; ++p) {
        max_atom = std::max({max_atom, pairs_i_[p], pairs_j_[p]});
    }
    std::vector<std::vector<int>> pairs_of_atom(static_cast<size_t>(max_atom) + 1u);
    for (int p = 0; p < n_pairs_; ++p) {
        pairs_of_atom[static_cast<size_t>(pairs_i_[p])].push_back(p);
        pairs_of_atom[static_cast<size_t>(pairs_j_[p])].push_back(p);
    }
    pair_atoms_.clear();
    for (size_t a = 0; a < pairs_of_atom.size(); ++a) {
        if (!pairs_of_atom[a].empty()) {
            pair_atoms_.push_back({static_cast<int>(a), std::move(pairs_of_atom[a])});
        }
    }
}

void QBiasPotential::permute_atom_indices(const AtomPermutation& perm) {
    if (perm.is_identity()) return;
    for (int p = 0; p < n_pairs_; ++p) {
        if (pairs_i_[p] >= perm.n_atoms() || pairs_j_[p] >= perm.n_atoms()) {
            throw std::invalid_argument(
                "QBiasPotential: pair atom index " +
                std::to_string(std::max(pairs_i_[p], pairs_j_[p])) +
                " is outside the system's " + std::to_string(perm.n_atoms()) + " atoms");
        }
    }
    // Map into copies, so nothing is half-remapped if a lookup fails.
    std::vector<int> new_i(pairs_i_.size());
    std::vector<int> new_j(pairs_j_.size());
    for (size_t p = 0; p < pairs_i_.size(); ++p) {
        new_i[p] = perm.to_internal(pairs_i_[p]);
        new_j[p] = perm.to_internal(pairs_j_[p]);
    }
    pairs_i_.swap(new_i);
    pairs_j_.swap(new_j);
    rebuild_pair_atoms();
}

bool QBiasPotential::pairFormed(const State& state, int pair_idx) const noexcept {
    const Eigen::Vector3f diff =
        state.atom_pos(pairs_i_[pair_idx]) - state.atom_pos(pairs_j_[pair_idx]);
    return pair_r2(diff) < q_cutoff_sq_;
}

float QBiasPotential::biasEnergy(float n, float k_bias, float n_target) noexcept {
    if (k_bias <= 0.0f) {
        return 0.0f;
    }
    const float delta = n - n_target;
    return 0.5f * k_bias * delta * delta;
}

void QBiasPotential::initializePairCache(
    const State& state,
    std::vector<uint8_t>& cache
) const {
    cache.resize(static_cast<size_t>(n_pairs_));
    for (int p = 0; p < n_pairs_; ++p) {
        cache[static_cast<size_t>(p)] = static_cast<uint8_t>(pairFormed(state, p));
    }
}

double QBiasPotential::calculateEnergy(const Context& context, const State& state) const {
    const float k_bias = context.getQBiasK();
    if (k_bias <= 0.0f) {
        return 0.0f;
    }

    int n_formed = 0;
    if (state.q_pair_cache.size() == static_cast<size_t>(n_pairs_)) {
        for (uint8_t formed : state.q_pair_cache) {
            n_formed += static_cast<int>(formed);
        }
    } else {
        for (int p = 0; p < n_pairs_; ++p) {
            n_formed += static_cast<int>(pairFormed(state, p));
        }
    }

    const float n = static_cast<float>(n_formed);
    return biasEnergy(n, k_bias, context.getQBiasTarget());
}

EnergyChangeResult QBiasPotential::calculateEnergyChange(
    const Context& context,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch
) const {
    const float k_bias = context.getQBiasK();
    if (k_bias <= 0.0f) {
        return EnergyChangeResult::finite(0.0f);
    }

    const float n_target = context.getQBiasTarget();
    auto& workspace = const_cast<QBiasWorkspace&>(context.getQBiasWorkspace());

    auto atom_moved = [&](int atom_idx) -> bool {
        return patch.bb_atom_moved[static_cast<size_t>(atom_idx)] != 0
            || patch.sc_atom_moved[static_cast<size_t>(atom_idx)] != 0
            || patch.h_atom_moved[static_cast<size_t>(atom_idx)] != 0
            || patch.o_atom_moved[static_cast<size_t>(atom_idx)] != 0;
    };

    auto& pairs_to_check = workspace.pairs_to_check;
    pairs_to_check.clear();
    for (const PairAtom& pa : pair_atoms_) {
        if (atom_moved(pa.atom)) {
            pairs_to_check.insert(pairs_to_check.end(), pa.pairs.begin(), pa.pairs.end());
        }
    }
    if (pairs_to_check.empty()) {
        return EnergyChangeResult::finite(0.0f);
    }

    std::sort(pairs_to_check.begin(), pairs_to_check.end());
    pairs_to_check.erase(
        std::unique(pairs_to_check.begin(), pairs_to_check.end()),
        pairs_to_check.end()
    );

    const auto& cache = old_state.q_pair_cache;
    if (cache.size() != static_cast<size_t>(n_pairs_)) {
        return EnergyChangeResult::finite(
            calculateEnergy(context, proposed_state)
            - calculateEnergy(context, old_state));
    }

    int n_formed_old = 0;
    for (uint8_t formed : cache) {
        n_formed_old += static_cast<int>(formed);
    }

    int n_formed_new = n_formed_old;
    for (int pair_idx : pairs_to_check) {
        const bool old_formed = cache[static_cast<size_t>(pair_idx)] != 0;
        const bool new_formed = pairFormed(proposed_state, pair_idx);
        if (old_formed == new_formed) {
            continue;
        }
        n_formed_new += new_formed ? 1 : -1;
        workspace.pending_updates.push_back({pair_idx, new_formed});
    }

    const float n_old = static_cast<float>(n_formed_old);
    const float n_new = static_cast<float>(n_formed_new);
    return EnergyChangeResult::finite(
        biasEnergy(n_new, k_bias, n_target) - biasEnergy(n_old, k_bias, n_target));
}

}  // namespace mcpu::forces
