#pragma once
/// Per-replica scratch for pair searches. Header-only.
#include <array>
#include <cstdint>

#include "pymcpu/neighbor/MovedCells.h"

namespace mcpu::neighbor {

/// Index of a grid in NeighborSystem's registry.
using GridId = std::uint8_t;
inline constexpr GridId kMuGrid = 0;
inline constexpr GridId kHBondOGrid = 1;
inline constexpr GridId kHBondHGrid = 2;
inline constexpr int kMaxGrids = 8;

/// Scratch the pair walks use while scoring a move. One per Context (mutable,
/// never shared between replicas); nothing in it carries meaning between
/// moves except the order hints in clash_hot.
struct PairScratch {
    /// Moved atoms per cell of each registered grid, indexed by GridId. A
    /// MovedCellScope fills one at the start of a delta and zeroes it before
    /// the delta returns, so each is all zeros between moves.
    std::array<MovedCellCounts, kMaxGrids> moved;

    /// Atoms that overlapped a fixed atom in recent rejected moves, most
    /// recent first. Mu's clash-first pass tests the ones a move carries
    /// before any other atom: overlaps recur on a few dozen atoms, so this
    /// finds most of them on the first atom it tests. Only the ORDER of the
    /// pass depends on it, never its answer, and an entry that is out of
    /// range for the system is skipped.
    static constexpr int kClashHotCap = 64;
    HotList<kClashHotCap> clash_hot;
};

}  // namespace mcpu::neighbor
