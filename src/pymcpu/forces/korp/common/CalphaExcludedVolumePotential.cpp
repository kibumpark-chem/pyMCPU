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
      min_distance_sq_(min_distance * min_distance),
      min_distance_state_sq_((min_distance - kStateClashBufferA) *
                             (min_distance - kStateClashBufferA))
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
    need(min_distance_ > kStateClashBufferA, "min_distance must exceed 0.001 A");
    for (int idx : ca_atom_) need(idx >= 0, "a residue has no CA atom index");

    moved_.assign(n, 0);
    chain_int_.assign(chain_id_.begin(), chain_id_.end());
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
            if ((pj - pi).squaredNorm() < min_distance_state_sq_) {
                return kClashPenalty;  // one overlap is enough; stop looking
            }
        }
    }
    return 0.0f;
}

bool CalphaExcludedVolumePotential::clashesAtMoveCutoff(
    const Context& context,
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
    if (moved_residues_.empty()) return false;

    // A rigid move keeps the distance of a pair it carries, so such a pair
    // cannot start to overlap. Rounding can move it a few 1e-6 A, which
    // calculateEnergy allows for (kStateClashBufferA).
    const bool skip_carried =
        patch.is_rigid && context.neighborConfig().skip_rigid_mm;
    const auto clashes = [&](int a, int b) {
        const Eigen::Vector3f pa =
            proposed_state.atom_pos(ca_atom_[static_cast<std::size_t>(a)]);
        const Eigen::Vector3f pb =
            proposed_state.atom_pos(ca_atom_[static_cast<std::size_t>(b)]);
        return (pb - pa).squaredNorm() < min_distance_sq_;
    };

    // Prefilter: for each moved residue, one branch-free pass over every
    // residue asks whether ANY checked partner is under a slightly loosened
    // cutoff. Only then does the exact loop below run for that residue, with
    // the moved/moved rules and the exact distance test. The loosened cutoff
    // means the prefilter cannot miss a pair the exact test would catch, so
    // the yes/no answer is unchanged.
    const std::size_t un = static_cast<std::size_t>(n);
    cx_.resize(un); cy_.resize(un); cz_.resize(un);
    for (std::size_t j = 0; j < un; ++j) {
        const Eigen::Vector3f p = proposed_state.atom_pos(ca_atom_[j]);
        cx_[j] = p.x(); cy_[j] = p.y(); cz_[j] = p.z();
    }
    const float filter_sq = min_distance_sq_ * (1.0f + 1e-4f);
    const int min_sep = min_separation_;

    for (std::size_t ia = 0; ia < moved_residues_.size(); ++ia) {
        const int a = moved_residues_[ia];
        const std::size_t ua = static_cast<std::size_t>(a);
        const float ax = cx_[ua], ay = cy_[ua], az = cz_[ua];
        const int sa = seq_number_[ua];
        const int ka = chain_int_[ua];
        int hit = 0;
        for (std::size_t j = 0; j < un; ++j) {
            const float dx = cx_[j] - ax;
            const float dy = cy_[j] - ay;
            const float dz = cz_[j] - az;
            const int sep = seq_number_[j] - sa;
            const int checked = (chain_int_[j] != ka) | (sep >= min_sep) | (-sep >= min_sep);
            hit |= (dx * dx + dy * dy + dz * dz < filter_sq) & checked;
        }
        if (!hit) continue;

        for (int j = 0; j < n; ++j) {
            if (j == a) continue;
            if (moved_[static_cast<std::size_t>(j)]) {
                if (skip_carried) continue;
                if (j < a) continue;  // unordered pair, visit once
            }
            if (!pair_is_checked(a, j)) continue;
            if (clashes(a, j)) return true;
        }
    }
    return false;
}

EnergyChangeResult CalphaExcludedVolumePotential::calculateEnergyChange(
    const Context& context,
    const State& /*old_state*/,
    const State& proposed_state,
    const ProposalPatch& patch) const
{
    // The guard holds no energy in an accepted state, so a proposal that does
    // not clash changes nothing.
    if (clashesAtMoveCutoff(context, proposed_state, patch)) {
        return EnergyChangeResult::rejected(kClashPenalty, RejectReason::StericClash);
    }
    return EnergyChangeResult::finite(0.f);
}

} // namespace mcpu::forces
