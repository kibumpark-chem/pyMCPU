#pragma once
/// Pair values of the accepted state, kept per site, and the changes a
/// pending move stages against them. Header-only.
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

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
        const std::size_t k = find_(v, b);
        if (k < v.size()) {
            v[k] = v.back();
            v.pop_back();
        }
    }

    /// Index of the first entry listing partner b, or v.size(). Four 8-byte
    /// entries per compare when an entry is two 32-bit words; the first match
    /// is the one the scalar scan returns, so the row order after a drop is
    /// unchanged.
    static std::size_t find_(const std::vector<Entry>& v, int b) {
        const std::size_t n = v.size();
        std::size_t k = 0;
#if defined(__AVX2__)
        if constexpr (sizeof(Entry) == 8 && offsetof(Entry, j) == 0) {
            const __m256i key = _mm256_set1_epi32(b);
            const char* base = reinterpret_cast<const char*>(v.data());
            for (; k + 4 <= n; k += 4) {
                const __m256i w = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(base + k * sizeof(Entry)));
                const unsigned bits = static_cast<unsigned>(_mm256_movemask_ps(
                    _mm256_castsi256_ps(_mm256_cmpeq_epi32(w, key)))) & 0x55u;
                if (bits) return k + (static_cast<unsigned>(__builtin_ctz(bits)) >> 1);
            }
        }
#endif
        for (; k < n; ++k)
            if (v[k].j == b) return k;
        return n;
    }

    std::vector<std::vector<Entry>> rows_;
    bool ready_ = false;
};

/// A grow-only list of trivially copyable records for a hot loop. push_back
/// is a capacity test and a store, always inlined; the rare growth is a
/// std::vector call (resize), so the hot loop calls nothing in the layer.
/// std::vector::push_back here was left out of line by LTO (1.5-2.1% of
/// actin/PGK1 cycles in its own frame). clear() keeps the storage, so a
/// list reused every move stops growing after a few moves. Unlike
/// std::vector, push_back reads v after growing, so v must not refer to an
/// element of the same buffer.
template <class T>
class PushBuffer {
public:
    [[gnu::always_inline]] inline void push_back(const T& v) {
        if (__builtin_expect(n_ == store_.size(), 0))
            store_.resize(store_.empty() ? 64 : 2 * store_.size());
        store_[n_++] = v;
    }
    void clear() noexcept { n_ = 0; }
    std::size_t size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }
    const T* begin() const noexcept { return store_.data(); }
    const T* end() const noexcept { return store_.data() + n_; }
    const T& operator[](std::size_t k) const noexcept { return store_[k]; }

private:
    std::vector<T> store_;
    std::size_t n_ = 0;
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
    PushBuffer<Pair> drop;
    PushBuffer<Pair> add;

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
