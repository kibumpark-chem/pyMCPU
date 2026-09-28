#include "pymcpu/forces/korp/common/OrientationalPairPotential.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "pymcpu/Context.h"
#include "pymcpu/System.h"

namespace mcpu::forces {

namespace {
constexpr std::uint8_t kBitN = 1;
constexpr std::uint8_t kBitCA = 2;
constexpr std::uint8_t kBitC = 4;
constexpr std::uint8_t kBitAll = kBitN | kBitCA | kBitC;
} // namespace

OrientationalPairPotential::OrientationalPairPotential(
    std::shared_ptr<const OrientationalPairMap> map,
    std::vector<int> n_atom,
    std::vector<int> ca_atom,
    std::vector<int> c_atom,
    std::vector<std::uint8_t> korp_type,
    std::vector<int> seq_number,
    std::vector<std::uint8_t> chain_id)
    : map_(std::move(map)),
      n_atom_(std::move(n_atom)), ca_atom_(std::move(ca_atom)), c_atom_(std::move(c_atom)),
      korp_type_(std::move(korp_type)),
      seq_number_(std::move(seq_number)), chain_id_(std::move(chain_id))
{
    if (!map_) throw std::invalid_argument("OrientationalPairPotential: null map");
    const std::size_t n = ca_atom_.size();
    const auto need = [&](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(
            std::string("OrientationalPairPotential: ") + what);
    };
    need(n > 0, "no residues");
    need(n_atom_.size() == n && c_atom_.size() == n, "N/CA/C arrays differ in length");
    need(korp_type_.size() == n, "korp_type length != residue count");
    need(seq_number_.size() == n, "seq_number length != residue count");
    need(chain_id_.size() == n, "chain_id length != residue count");
    for (std::size_t r = 0; r < n; ++r) {
        need(n_atom_[r] >= 0 && ca_atom_[r] >= 0 && c_atom_[r] >= 0,
             "a residue is missing an N, CA or C atom index; KORP builds its "
             "frame from all three and cannot score a residue without them");
        need(korp_type_[r] < OrientationalPairMap::kNumTypes,
             "korp_type out of range (expected 0..19)");
    }

    frames_old_.resize(n);
    frames_new_.resize(n);
    cls_.assign(n, FrameClass::Fixed);
    bits_.assign(n, 0);
    rebuild_atom_lookup();
}

void OrientationalPairPotential::rebuild_atom_lookup() {
    int max_atom = -1;
    for (const auto* vec : {&n_atom_, &ca_atom_, &c_atom_}) {
        for (int idx : *vec) {
            if (idx > max_atom) max_atom = idx;
        }
    }
    const std::size_t size = static_cast<std::size_t>(max_atom) + 1;
    frame_residue_of_atom_.assign(size, -1);
    frame_bit_of_atom_.assign(size, 0);

    const int n = num_residues();
    for (int r = 0; r < n; ++r) {
        const std::size_t u = static_cast<std::size_t>(r);
        const int atoms[3] = {n_atom_[u], ca_atom_[u], c_atom_[u]};
        const std::uint8_t bit[3] = {kBitN, kBitCA, kBitC};
        for (int k = 0; k < 3; ++k) {
            const std::size_t a = static_cast<std::size_t>(atoms[k]);
            // Glycine aside, a backbone atom belongs to exactly one residue, so
            // a collision here means the caller handed us overlapping indices.
            if (frame_residue_of_atom_[a] >= 0) {
                throw std::invalid_argument(
                    "OrientationalPairPotential: atom " + std::to_string(atoms[k]) +
                    " is claimed as a frame atom by more than one residue");
            }
            frame_residue_of_atom_[a] = r;
            frame_bit_of_atom_[a] = bit[k];
        }
    }
}

void OrientationalPairPotential::permute_atom_indices(const AtomPermutation& perm) {
    if (perm.is_identity()) return;
    for (auto* vec : {&n_atom_, &ca_atom_, &c_atom_}) {
        for (int& idx : *vec) {
            if (idx >= 0) idx = perm.to_internal(idx);
        }
    }
    rebuild_atom_lookup();
}

ResidueFrame OrientationalPairPotential::frame_of(const State& state, int residue) const {
    const std::size_t r = static_cast<std::size_t>(residue);
    return make_residue_frame(state.atom_pos(n_atom_[r]),
                              state.atom_pos(ca_atom_[r]),
                              state.atom_pos(c_atom_[r]));
}

void OrientationalPairPotential::build_frames(
    const State& state, std::vector<ResidueFrame>& out) const
{
    const int n = num_residues();
    out.resize(static_cast<std::size_t>(n));
    for (int r = 0; r < n; ++r) {
        out[static_cast<std::size_t>(r)] = frame_of(state, r);
    }
}

double OrientationalPairPotential::pair_energy(
    const std::vector<ResidueFrame>& frames, int lo, int hi) const noexcept
{
    const std::size_t a = static_cast<std::size_t>(lo);
    const std::size_t b = static_cast<std::size_t>(hi);

    const Eigen::Vector3d rab = frames[b].origin - frames[a].origin;
    const double d2 = rab.squaredNorm();
    if (d2 >= static_cast<double>(map_->cutoff_sq())) return 0.0;

    // Sequence separation from PDB numbering, as upstream does; a pair spanning
    // two chains is unconditionally non-bonding.
    const int separation = (chain_id_[a] != chain_id_[b])
        ? 0
        : std::abs(seq_number_[b] - seq_number_[a]);
    const int slice = map_->slice_for_separation(separation);
    if (slice < 0) return 0.0;   // too close in sequence to be scored at all

    const PairCoordinates pc = pair_coordinates(frames[a], frames[b]);
    if (pc.d <= static_cast<double>(map_->min_r())) return 0.0;

    const PairBins bins = map_->bins(pc);
    return static_cast<double>(map_->slice_weight(slice))
         * static_cast<double>(map_->lookup(slice, korp_type_[a], korp_type_[b], bins));
}

float OrientationalPairPotential::calculateEnergy(
    const Context& /*context*/, const State& state) const
{
    build_frames(state, frames_old_);
    const int n = num_residues();
    // Accumulated in double, matching upstream, because a few thousand
    // table entries of order 1 summed in float loses digits that the
    // reference-parity test would then have to tolerate.
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            total += pair_energy(frames_old_, i, j);
        }
    }
    return static_cast<float>(total);
}

bool OrientationalPairPotential::classify(const ProposalPatch& patch) const
{
    const int n = num_residues();
    bits_.assign(static_cast<std::size_t>(n), 0);
    cls_.assign(static_cast<std::size_t>(n), FrameClass::Fixed);
    changed_.clear();

    // Which of each residue's three frame atoms moved. Keyed on membership in
    // the moved set, NOT on displacement: a C-terminal phi pivot at residue r
    // moves C(r) but leaves N(r) and CA(r) behind, so residue r's frame is
    // reshaped even though the move as a whole is rigid. Classifying by
    // displacement would call that residue rigid and wrongly skip its pairs --
    // and CA and C sit ON the rotation axis of an N-terminal pivot, so they do
    // not move at all while still being carried by the rotation.
    const std::size_t lookup_size = frame_residue_of_atom_.size();
    for (int atom : patch.moved_indices) {
        const std::size_t a = static_cast<std::size_t>(atom);
        if (atom < 0 || a >= lookup_size) continue;  // O atom, or not ours
        const int r = frame_residue_of_atom_[a];
        if (r < 0) continue;
        bits_[static_cast<std::size_t>(r)] |= frame_bit_of_atom_[a];
    }

    for (int r = 0; r < n; ++r) {
        const std::size_t u = static_cast<std::size_t>(r);
        if (bits_[u] == 0) continue;
        cls_[u] = (bits_[u] == kBitAll && patch.is_rigid && rigid_skip_enabled_)
            ? FrameClass::RigidMoved
            : FrameClass::Distorted;
        changed_.push_back(r);
    }
    return !changed_.empty();
}

EnergyChangeResult OrientationalPairPotential::calculateEnergyChange(
    const Context& /*context*/,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch) const
{
    // A sidechain-only move touches no N, CA or C, so KORP's energy cannot have
    // changed and there is nothing to compute. This is exact, not an
    // approximation -- the potential reads no other atom.
    if (!classify(patch)) return EnergyChangeResult::finite(0.f);

    build_frames(old_state, frames_old_);
    build_frames(proposed_state, frames_new_);

    const int n = num_residues();
    double delta = 0.0;

    const auto accumulate = [&](int a, int b) {
        const int lo = a < b ? a : b;
        const int hi = a < b ? b : a;
        delta += pair_energy(frames_new_, lo, hi) - pair_energy(frames_old_, lo, hi);
    };

    const std::size_t n_changed = changed_.size();
    for (std::size_t ia = 0; ia < n_changed; ++ia) {
        const int a = changed_[ia];

        // changed x fixed
        for (int j = 0; j < n; ++j) {
            if (cls_[static_cast<std::size_t>(j)] != FrameClass::Fixed) continue;
            accumulate(a, j);
        }

        // changed x changed, each unordered pair once
        for (std::size_t ib = ia + 1; ib < n_changed; ++ib) {
            const int b = changed_[ib];
            // Both partners carried by the SAME rigid motion: in real
            // arithmetic every one of the six coordinates is invariant, because
            // both frames transform together and a proper rotation commutes with
            // the cross products the frame is built from. In float32 it is NOT
            // exact -- a pair within rounding of a bin edge can change bin -- so
            // this branch is reachable only when rigid_skip_enabled_ is set
            // explicitly (default off; see the header).
            if (cls_[static_cast<std::size_t>(a)] == FrameClass::RigidMoved &&
                cls_[static_cast<std::size_t>(b)] == FrameClass::RigidMoved) {
                continue;
            }
            accumulate(a, b);
        }
    }

    return EnergyChangeResult::finite(static_cast<float>(delta));
}

} // namespace mcpu::forces
