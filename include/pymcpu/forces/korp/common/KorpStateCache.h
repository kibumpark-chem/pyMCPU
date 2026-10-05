#pragma once
/// KORP bookkeeping for one accepted state: the residue frames and the energy
/// of every residue pair, as OrientationalPairPotential last scored them.
///
/// Lives on State (State::korp_cache) because it describes that state's
/// coordinates. OrientationalPairPotential builds it the first time a move
/// needs it, folds each accepted move into it through
/// Potential::commitAcceptedMove, and rebuilds it in resyncEnergy. With it, a
/// move rebuilds frames only for the residues whose N, CA or C moved, and the
/// old side of every pair delta is a lookup instead of a second evaluation.
///
/// The pair store is dense (n x n floats, symmetric) for now. Its interface --
/// reset, ready_for, invalidate, get, set -- is the subset of the planned
/// PairLedger<float> that KORP needs, so the ledger can replace it without
/// touching the potential's delta loop.
///
/// A copy starts EMPTY. A State is copied for proposals and verifier scratch,
/// and those copies then have their coordinates changed without anyone
/// telling the cache. An empty cache is rebuilt from the coordinates when it
/// is next needed, so dropping it on copy can cost time but never accuracy.
/// A move carries the cache along and leaves the source empty.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "pymcpu/forces/korp/common/ResidueFrame.h"

namespace mcpu::forces {

class KorpStateCache {
public:
    KorpStateCache() = default;
    KorpStateCache(const KorpStateCache&) noexcept {}
    KorpStateCache& operator=(const KorpStateCache& other) noexcept {
        if (this != &other) invalidate();
        return *this;
    }
    KorpStateCache(KorpStateCache&& other) noexcept { take(other); }
    KorpStateCache& operator=(KorpStateCache&& other) noexcept {
        if (this != &other) take(other);
        return *this;
    }

    /// Size for n residues, zero every pair, and mark the cache as built by
    /// `owner`. The caller then fills frames and pair energies.
    void reset(int n, const void* owner) {
        n_ = n;
        owner_ = owner;
        frames.resize(static_cast<std::size_t>(n));
        energy_.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.f);
        ready_ = true;
        ++generation_;
    }

    /// True when the cache was built by this potential for n residues. A
    /// State moved between contexts with different KORP terms fails this and
    /// is rebuilt.
    [[nodiscard]] bool ready_for(const void* owner, int n) const noexcept {
        return ready_ && owner_ == owner && n_ == n;
    }

    void invalidate() noexcept {
        ready_ = false;
        owner_ = nullptr;
        ++generation_;
    }

    /// Bumped by every reset, invalidate and committed move, so a pending
    /// update can tell whether the cache it was computed against is still
    /// the one in place.
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    void note_commit() noexcept { ++generation_; }

    [[nodiscard]] const float* row(int i) const noexcept {
        return energy_.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(n_);
    }
    [[nodiscard]] float get(int i, int j) const noexcept { return row(i)[j]; }
    void set(int i, int j, float e) noexcept {
        const std::size_t n = static_cast<std::size_t>(n_);
        energy_[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)] = e;
        energy_[static_cast<std::size_t>(j) * n + static_cast<std::size_t>(i)] = e;
    }

    /// Frame of every residue in the cached state.
    std::vector<ResidueFrame> frames;

private:
    void take(KorpStateCache& other) noexcept {
        frames = std::move(other.frames);
        energy_ = std::move(other.energy_);
        n_ = other.n_;
        owner_ = other.owner_;
        ready_ = other.ready_;
        ++generation_;
        other.frames.clear();
        other.energy_.clear();
        other.invalidate();
    }

    std::vector<float> energy_;
    int n_ = 0;
    const void* owner_ = nullptr;
    bool ready_ = false;
    std::uint64_t generation_ = 0;
};

} // namespace mcpu::forces
