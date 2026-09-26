#include "pymcpu/forces/korp/common/CalphaExcludedVolumePotential.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "pymcpu/Context.h"
#include "pymcpu/System.h"

namespace mcpu::forces {

CalphaExcludedVolumePotential::CalphaExcludedVolumePotential(
    std::vector<int> ca_atom,
    std::vector<int> seq_number,
    std::vector<std::uint8_t> chain_id,
    int min_separation,
    float min_distance)
    : ca_atom_(std::move(ca_atom)),
      seq_number_(std::move(seq_number)),
      chain_id_(std::move(chain_id)),
      min_separation_(min_separation),
      min_distance_(min_distance),
      min_distance_sq_(min_distance * min_distance)
{
    const std::size_t n = ca_atom_.size();
    const auto need = [&](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(
            std::string("CalphaExcludedVolumePotential: ") + what);
    };
    need(n > 0, "no residues");
    need(seq_number_.size() == n, "seq_number length != residue count");
    need(chain_id_.size() == n, "chain_id length != residue count");
    need(min_separation_ >= 1, "min_separation must be at least 1");
    need(min_distance_ > 0.f, "min_distance must be positive");
    for (int idx : ca_atom_) need(idx >= 0, "a residue has no CA atom index");

    moved_.assign(n, 0);
    rebuild_atom_lookup();
}

void CalphaExcludedVolumePotential::rebuild_atom_lookup() {
    int max_atom = -1;
    for (int idx : ca_atom_) {
        if (idx > max_atom) max_atom = idx;
    }
    residue_of_ca_.assign(static_cast<std::size_t>(max_atom) + 1, -1);
    const int n = num_residues();
    for (int r = 0; r < n; ++r) {
        residue_of_ca_[static_cast<std::size_t>(ca_atom_[static_cast<std::size_t>(r)])] = r;
    }
}

void CalphaExcludedVolumePotential::permute_atom_indices(const AtomPermutation& perm) {
    if (perm.is_identity()) return;
    for (int& idx : ca_atom_) {
        if (idx >= 0) idx = perm.to_internal(idx);
    }
    rebuild_atom_lookup();
}

bool CalphaExcludedVolumePotential::pair_is_checked(int a, int b) const noexcept {
    const std::size_t u = static_cast<std::size_t>(a);
    const std::size_t v = static_cast<std::size_t>(b);
    // Two chains can always clash with each other, however their residues
    // happen to be numbered.
    if (chain_id_[u] != chain_id_[v]) return true;
    return std::abs(seq_number_[v] - seq_number_[u]) >= min_separation_;
}

float CalphaExcludedVolumePotential::calculateEnergy(
    const Context& /*context*/, const State& state) const
{
    const int n = num_residues();
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector3f pi = state.atom_pos(ca_atom_[static_cast<std::size_t>(i)]);
        for (int j = i + 1; j < n; ++j) {
            if (!pair_is_checked(i, j)) continue;
            const Eigen::Vector3f pj = state.atom_pos(ca_atom_[static_cast<std::size_t>(j)]);
            if ((pj - pi).squaredNorm() < min_distance_sq_) {
                return kClashPenalty;  // one overlap is enough; stop looking
            }
        }
    }
    return 0.0f;
}

EnergyChangeResult CalphaExcludedVolumePotential::calculateEnergyChange(
    const Context& /*context*/,
    const State& /*old_state*/,
    const State& proposed_state,
    const ProposalPatch& patch) const
{
    const int n = num_residues();
    moved_.assign(static_cast<std::size_t>(n), 0);
    moved_residues_.clear();

    const std::size_t lookup_size = residue_of_ca_.size();
    for (int atom : patch.moved_indices) {
        const std::size_t a = static_cast<std::size_t>(atom);
        if (atom < 0 || a >= lookup_size) continue;
        const int r = residue_of_ca_[a];
        if (r < 0) continue;
        if (!moved_[static_cast<std::size_t>(r)]) {
            moved_[static_cast<std::size_t>(r)] = 1;
            moved_residues_.push_back(r);
        }
    }

    // No CA moved, so no CA-CA distance changed. A sidechain move lands here.
    if (moved_residues_.empty()) return EnergyChangeResult::finite(0.f);

    // A rigid move preserves every distance among the atoms it carries, so a
    // pair with both partners moved cannot newly clash. Same argument the Mu
    // term uses for its own moved-moved elision.
    const bool skip_moved_moved = patch.is_rigid;

    const auto clashes = [&](int a, int b) {
        const Eigen::Vector3f pa =
            proposed_state.atom_pos(ca_atom_[static_cast<std::size_t>(a)]);
        const Eigen::Vector3f pb =
            proposed_state.atom_pos(ca_atom_[static_cast<std::size_t>(b)]);
        return (pb - pa).squaredNorm() < min_distance_sq_;
    };

    for (std::size_t ia = 0; ia < moved_residues_.size(); ++ia) {
        const int a = moved_residues_[ia];
        for (int j = 0; j < n; ++j) {
            if (j == a) continue;
            if (moved_[static_cast<std::size_t>(j)]) {
                if (skip_moved_moved) continue;
                if (j < a) continue;  // unordered pair, visit once
            }
            if (!pair_is_checked(a, j)) continue;
            if (clashes(a, j)) {
                return EnergyChangeResult::rejected(
                    kClashPenalty, RejectReason::StericClash);
            }
        }
    }

    // The guard holds no energy in an accepted state, so a proposal that does
    // not clash changes nothing.
    return EnergyChangeResult::finite(0.f);
}

} // namespace mcpu::forces
