#pragma once
/// Pair values of the accepted state, kept per site, and the changes a
/// pending move stages against them. Header-only.
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpu::neighbor {

/// For each site, the partners it is listed with and the payload of each pair.
/// A pair (i, j) appears twice: as j in i's row and as i in j's row, with the
/// same payload. Rows keep insertion order, and removal swaps the last entry
/// into the hole, so the order of a row depends only on the sequence of
/// add/remove calls. Built and read by a term (Mu's contact list is the first
/// user); the term decides what a payload means.
template <class P>
class PairLedger {
public:
    struct Entry {
        std::int32_t j;
        P payload;
    };

    /// n empty rows, freshly allocated; not ready. O(n).
    void reset(int n) {
        rows_.assign(static_cast<std::size_t>(n), {});
        ready_ = false;
    }
    /// n empty rows, keeping each row's capacity; readiness unchanged. O(n).
    void clear_rows(int n) {
        rows_.resize(static_cast<std::size_t>(n));
        for (auto& r : rows_) r.clear();
    }
    /// Drop every row and mark not ready. O(n).
    void invalidate() noexcept {
        rows_.clear();
        ready_ = false;
    }
    void set_ready(bool on) noexcept { ready_ = on; }
    [[nodiscard]] bool ready() const noexcept { return ready_; }

    /// List the pair (i, j). O(1) amortized.
    void add(int i, int j, const P& p) {
        rows_[static_cast<std::size_t>(i)].push_back(Entry{static_cast<std::int32_t>(j), p});
        rows_[static_cast<std::size_t>(j)].push_back(Entry{static_cast<std::int32_t>(i), p});
    }
    /// Unlist the pair (i, j). O(degree).
    void remove(int i, int j) {
        drop_(i, j);
        drop_(j, i);
    }
    [[nodiscard]] const std::vector<Entry>& partners(int i) const {
        return rows_[static_cast<std::size_t>(i)];
    }

private:
    void drop_(int a, int b) {
        auto& v = rows_[static_cast<std::size_t>(a)];
        for (std::size_t k = 0; k < v.size(); ++k) {
            if (v[k].j == b) {
                v[k] = v.back();
                v.pop_back();
                return;
            }
        }
    }

    std::vector<std::vector<Entry>> rows_;
    bool ready_ = false;
};

/// Ledger changes a pending move stages: drops first, then adds, applied only
/// if the move is accepted. Lives in per-Context scratch so replicas that share
/// one term object cannot tread on each other.
template <class P>
struct PendingPairs {
    struct Pair {
        std::int32_t i;
        std::int32_t j;
        P payload;
    };
    std::vector<Pair> drop;
    std::vector<Pair> add;

    void clear() noexcept {
        drop.clear();
        add.clear();
    }
    void commit_into(PairLedger<P>& ledger) const {
        for (const auto& p : drop) ledger.remove(p.i, p.j);
        for (const auto& p : add) ledger.add(p.i, p.j, p.payload);
    }
};

}  // namespace mcpu::neighbor
