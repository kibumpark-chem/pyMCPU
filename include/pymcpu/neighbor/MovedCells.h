#pragma once
/// Per-move scratch for the pair walks: how many of the atoms each grid cell
/// lists the pending move displaces, and the short list of atoms that
/// overlapped in recent rejected moves. Header-only and always inlined.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mcpu::neighbor {

/// One count per cell of a grid: how many of the atoms the cell lists the
/// pending move displaces. All zeros between moves; MovedCellScope fills
/// it for one move and zeroes it again. A walk skips a cell whose count
/// equals the number of atoms it lists (every one of them moved, so the
/// grid's packed positions in it are all stale).
class MovedCellCounts {
public:
    /// Grows to at least n_cells counts, all zero. Never shrinks.
    [[gnu::always_inline]] inline void ensure(std::size_t n_cells) {
        if (counts_.size() < n_cells) counts_.assign(n_cells, 0);
    }
    [[gnu::always_inline]] inline std::uint8_t* data() noexcept {
        return counts_.data();
    }
    std::size_t size() const noexcept { return counts_.size(); }

private:
    template <class Grid> friend class MovedCellScope;
    std::vector<std::uint8_t> counts_;
    int depth_ = 0;  // open scopes; checked in debug builds only
};

/// Fills a MovedCellCounts for one move and zeroes it again when the scope
/// ends, on every return path. O(n_moved) each way. `moved` must list each
/// moved atom once: a duplicate makes a cell that still holds an unmoved
/// atom look fully moved (see ProposalPatch::mark_moved). Two scopes on the
/// same counts at once would double count, which debug builds assert.
template <class Grid>
class MovedCellScope {
public:
    [[gnu::always_inline]] inline MovedCellScope(MovedCellCounts& c,
                                                 const Grid& grid,
                                                 const int* moved, int n)
        : c_(c), grid_(grid), moved_(moved), n_(n) {
        c_.ensure(static_cast<std::size_t>(grid_.num_cells()));
#ifndef NDEBUG
        assert(c_.depth_ == 0 && "nested MovedCellScope on the same counts");
        ++c_.depth_;
#endif
        std::uint8_t* const counts = c_.counts_.data();
        for (int k = 0; k < n_; ++k) {
            const int cell = grid_.atom_cell(moved_[k]);
            if (cell >= 0) ++counts[static_cast<std::size_t>(cell)];
        }
    }
    [[gnu::always_inline]] inline ~MovedCellScope() {
        std::uint8_t* const counts = c_.counts_.data();
        for (int k = 0; k < n_; ++k) {
            const int cell = grid_.atom_cell(moved_[k]);
            if (cell >= 0) counts[static_cast<std::size_t>(cell)] = 0;
        }
#ifndef NDEBUG
        --c_.depth_;
#endif
    }
    MovedCellScope(const MovedCellScope&) = delete;
    MovedCellScope& operator=(const MovedCellScope&) = delete;

    const std::uint8_t* counts() const noexcept { return c_.counts_.data(); }

private:
    MovedCellCounts& c_;
    const Grid& grid_;
    const int* moved_;
    int n_;
};

/// Up to Cap site ids, most recently noted first. A walk that tests these
/// first finds a recurring overlap on its first probe; only the ORDER of
/// such a walk depends on the list, never its answer.
template <int Cap>
struct HotList {
    static constexpr int kCap = Cap;
    int a[Cap] = {};
    int n = 0;

    /// Move `site` to the front, dropping the oldest entry when full.
    /// O(Cap).
    [[gnu::always_inline]] inline void note(int site) noexcept {
        int k = 0;
        while (k < n && a[k] != site) ++k;
        if (k == n) {
            if (n < Cap) ++n;
            k = n - 1;
        }
        for (; k > 0; --k) a[k] = a[k - 1];
        a[0] = site;
    }
};

}  // namespace mcpu::neighbor
