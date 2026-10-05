#include "pymcpu/forces/korp/common/OrientationalPairPotential.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "pymcpu/Context.h"
#include "pymcpu/State.h"
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
            // A backbone atom belongs to exactly one residue, so a collision
            // here means the caller handed us overlapping indices.
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

double OrientationalPairPotential::fill_cache(const State& state) const
{
    const int n = num_residues();
    KorpStateCache& cache = state.korp_cache;
    cache.reset(n, this);
    build_frames(state, cache.frames);
    // Same frames, pairs and order as calculateEnergy, so the total is
    // bit-identical to it.
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double e = pair_energy(cache.frames, i, j);
            total += e;
            if (e != 0.0) cache.set(i, j, static_cast<float>(e));
        }
    }
    return total;
}

float OrientationalPairPotential::resyncEnergy(
    const Context& /*context*/, const State& state) const
{
    pending_.valid = false;
    return static_cast<float>(fill_cache(state));
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
        cls_[u] = neighbor::moved_site_class(patch.is_rigid, rigid_skip_enabled_,
                                             /*whole=*/bits_[u] == kBitAll);
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
    KorpStateCache& cache = old_state.korp_cache;
    pending_.valid = true;
    pending_.old_state = &old_state;
    pending_.proposed_state = &proposed_state;
    pending_.num_moved = patch.moved_indices.size();
    pending_.generation = cache.generation();
    pending_.pairs.clear();

    // A sidechain-only move touches no N, CA or C, so KORP's energy cannot have
    // changed and there is nothing to compute. This is exact, not an
    // approximation -- the potential reads no other atom.
    if (!classify(patch)) return EnergyChangeResult::finite(0.f);

    const int n = num_residues();
    if (!cache.ready_for(this, n)) {
        fill_cache(old_state);
        pending_.generation = cache.generation();
    }

    // New frames: the accepted ones, with only the changed residues rebuilt.
    frames_new_ = cache.frames;
    for (int r : changed_) {
        frames_new_[static_cast<std::size_t>(r)] = frame_of(proposed_state, r);
    }
    const std::size_t un = static_cast<std::size_t>(n);
    ox_.resize(un); oy_.resize(un); oz_.resize(un);
    for (std::size_t r = 0; r < un; ++r) {
        ox_[r] = frames_new_[r].origin.x();
        oy_[r] = frames_new_[r].origin.y();
        oz_[r] = frames_new_[r].origin.z();
    }

    // Prefilter on CA-CA distance with a little slack; pair_energy repeats the
    // exact cutoff test, so a pair at the edge is scored exactly as before.
    const double cut2 = static_cast<double>(map_->cutoff_sq()) * (1.0 + 1e-9);

    // Each pair with at least one changed residue is visited once: changed x
    // fixed from the changed side, changed x changed from the lower index.
    // The old energy comes from the cache, and only pairs whose energy
    // changed enter the sum and the pending update. Both sides are the float
    // values the cache holds, so the running total stays the sum of the cache.
    double delta = 0.0;
    for (int a : changed_) {
        const std::size_t ua = static_cast<std::size_t>(a);
        const float* old_row = cache.row(a);
        // Rigid exists only with the rigid skip on; see the header.
        const bool a_rigid = cls_[ua] == FrameClass::Rigid;
        const double ax = ox_[ua], ay = oy_[ua], az = oz_[ua];
        for (int j = 0; j < n; ++j) {
            const std::size_t uj = static_cast<std::size_t>(j);
            const FrameClass cj = cls_[uj];
            if (cj != FrameClass::Fixed) {
                if (j <= a) continue;
                if (a_rigid && cj == FrameClass::Rigid) continue;
            }
            const double dx = ox_[uj] - ax;
            const double dy = oy_[uj] - ay;
            const double dz = oz_[uj] - az;
            float e_new = 0.f;
            if (dx * dx + dy * dy + dz * dz < cut2) {
                e_new = static_cast<float>(a < j ? pair_energy(frames_new_, a, j)
                                                 : pair_energy(frames_new_, j, a));
            }
            const float e_old = old_row[j];
            if (e_new != e_old) {
                delta += static_cast<double>(e_new) - static_cast<double>(e_old);
                pending_.pairs.push_back(PendingPair{a, j, e_new});
            }
        }
    }

    return EnergyChangeResult::finite(static_cast<float>(delta));
}

void OrientationalPairPotential::commitAcceptedMove(
    const Context& /*context*/,
    const State& state,
    const State& proposed_state,
    const ProposalPatch& patch) const
{
    KorpStateCache& cache = state.korp_cache;
    const bool mine = pending_.valid && isEnabled() &&
                      pending_.old_state == &state &&
                      pending_.proposed_state == &proposed_state &&
                      pending_.num_moved == patch.moved_indices.size() &&
                      pending_.generation == cache.generation();
    pending_.valid = false;
    if (!cache.ready_for(this, num_residues())) return;
    if (!mine) {
        // The record is for some other move (or none): the cache no longer
        // matches the coordinates and is rebuilt when next needed.
        cache.invalidate();
        return;
    }
    for (int r : changed_) {
        const std::size_t u = static_cast<std::size_t>(r);
        cache.frames[u] = frames_new_[u];
    }
    for (const PendingPair& p : pending_.pairs) cache.set(p.i, p.j, p.energy);
    cache.note_commit();
}

} // namespace mcpu::forces
