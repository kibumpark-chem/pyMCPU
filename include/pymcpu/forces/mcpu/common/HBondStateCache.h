#pragma once
/// H-bond bookkeeping for one accepted state: every (donor, acceptor) residue
/// pair whose H and O are within NeighborSystem::kHBondListA, with its energy
/// (often 0) as HBondPotential last scored it.
///
/// Lives on State (State::hbond_cache) because it describes that state's
/// coordinates. HBondPotential builds it the first time a move needs it and
/// folds each accepted move into it through Potential::commitAcceptedMove.
/// With it, the old side of the H-bond delta is a lookup instead of a second
/// evaluation of every candidate pair.
///
/// A pair is directional, so it is listed twice: in the donor's row with the
/// acceptor as partner, and in the acceptor's row with the donor as partner.
/// A residue rarely has more than three. Listing the pairs just outside the
/// 2.5 A H-bond cutoff lets a rigid move re-decide each pair it carries;
/// `drift` bounds how far the unlisted ones can have moved since they were
/// last measured, and each entry's `fresh_until` is the drift up to which
/// its energy is still exact (see HBondPotential::calculateEnergyChange).
///
/// Like KorpStateCache, a copy starts EMPTY (a copied State has its
/// coordinates changed without telling the cache), and a move carries the
/// cache along and leaves the source empty.
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mcpu::forces {

class HBondStateCache {
public:
    struct Entry {
        std::int32_t partner;
        float energy;
        /// A rigid move may carry the pair at `energy` while the ledger's
        /// drift stays at or below this: the drift when the pair was scored,
        /// plus its slack (HBondPotential::evaluate_with_slack).
        float fresh_until;
    };

    HBondStateCache() = default;
    HBondStateCache(const HBondStateCache&) noexcept {}
    HBondStateCache& operator=(const HBondStateCache& other) noexcept {
        if (this != &other) invalidate();
        return *this;
    }
    HBondStateCache(HBondStateCache&& other) noexcept { take(other); }
    HBondStateCache& operator=(HBondStateCache&& other) noexcept {
        if (this != &other) take(other);
        return *this;
    }

    /// Empty rows for n residues, marked as built by `owner` under the
    /// residue energy mask `mask_epoch`. The caller then adds the pairs.
    void reset(int n, const void* owner, std::uint64_t mask_epoch) {
        n_ = n;
        owner_ = owner;
        mask_epoch_ = mask_epoch;
        don_.resize(static_cast<std::size_t>(n));
        acc_.resize(static_cast<std::size_t>(n));
        for (auto& r : don_) r.clear();
        for (auto& r : acc_) r.clear();
        ready_ = true;
        drift = 0.f;
        ++generation_;
    }
    /// True when built by this potential, for n residues, under this mask.
    [[nodiscard]] bool ready_for(const void* owner, int n, std::uint64_t mask_epoch) const noexcept {
        return ready_ && owner_ == owner && n_ == n && mask_epoch_ == mask_epoch;
    }
    void invalidate() noexcept {
        ready_ = false;
        owner_ = nullptr;
        ++generation_;
    }
    /// Bumped by every reset, invalidate and committed move, so a pending
    /// update can tell whether it was computed against the cache in place.
    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_; }
    void note_commit() noexcept { ++generation_; }

    [[nodiscard]] const std::vector<Entry>& as_donor(int d) const noexcept {
        return don_[static_cast<std::size_t>(d)];
    }
    [[nodiscard]] const std::vector<Entry>& as_acceptor(int a) const noexcept {
        return acc_[static_cast<std::size_t>(a)];
    }
    /// Energy of (d, a), or 0 when it is not listed. O(row length).
    [[nodiscard]] float get(int d, int a) const noexcept {
        for (const Entry& e : don_[static_cast<std::size_t>(d)]) {
            if (e.partner == a) return e.energy;
        }
        return 0.f;
    }
    void add(int d, int a, float e, float fresh_until) {
        don_[static_cast<std::size_t>(d)].push_back(Entry{a, e, fresh_until});
        acc_[static_cast<std::size_t>(a)].push_back(Entry{d, e, fresh_until});
    }
    /// Unlist every pair with r as donor or acceptor, except those `keep`
    /// accepts. keep(entry) sees each of r's entries once, with the other
    /// residue as partner, and must give both copies of a pair one answer.
    template <class Keep>
    void unlist_except(int r, Keep keep) {
        filter_(don_[static_cast<std::size_t>(r)], acc_, r, keep);
        filter_(acc_[static_cast<std::size_t>(r)], don_, r, keep);
    }

private:
    template <class Keep>
    static void filter_(std::vector<Entry>& row, std::vector<std::vector<Entry>>& mirror, int r, Keep& keep) {
        std::size_t kept = 0;
        for (std::size_t k = 0; k < row.size(); ++k) {
            if (keep(row[k])) {
                row[kept++] = row[k];
            } else {
                drop_(mirror[static_cast<std::size_t>(row[k].partner)], r);
            }
        }
        row.resize(kept);
    }
    static void drop_(std::vector<Entry>& row, int partner) noexcept {
        for (std::size_t k = 0; k < row.size(); ++k) {
            if (row[k].partner == partner) {
                row[k] = row.back();
                row.pop_back();
                return;
            }
        }
    }
    void take(HBondStateCache& other) noexcept {
        don_ = std::move(other.don_);
        acc_ = std::move(other.acc_);
        n_ = other.n_;
        owner_ = other.owner_;
        mask_epoch_ = other.mask_epoch_;
        ready_ = other.ready_;
        drift = other.drift;
        ++generation_;
        other.invalidate();
    }

    std::vector<std::vector<Entry>> don_, acc_;
    int n_ = 0;
    const void* owner_ = nullptr;
    std::uint64_t mask_epoch_ = 0;
    bool ready_ = false;
    std::uint64_t generation_ = 0;

public:
    /// Sum of the carry bounds (Context::rigid_carry_bound_A, times
    /// HBondPotential's factor for the virtual H) of the rigid moves folded in
    /// since the ledger was built from the coordinates.
    float drift = 0.f;
};

}  // namespace mcpu::forces
