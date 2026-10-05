#pragma once
/// What a proposed move touches: its moved sites and how each one moved.
/// Header-only.
#include <cstddef>
#include <cstdint>

#include "pymcpu/ProposalPatch.h"

namespace mcpu::neighbor {

/// How a proposed move affects a site (an atom, or a residue frame).
/// The one definition every term uses:
///   Fixed  the site did not move;
///   Rigid  it moved under a rigid move, together with every other Rigid
///          site, so distances among Rigid sites are unchanged (up to
///          rounding) and a term may skip those pairs;
///   Flex   anything else: its distances to other moved sites changed.
enum class SiteClass : std::uint8_t { Fixed = 0, Rigid, Flex };

/// Class of a site that moved. `whole` is false when only part of a
/// multi-atom site (a residue frame) moved, which always counts as Flex.
[[nodiscard]] inline SiteClass moved_site_class(bool move_is_rigid, bool rigid_skip_enabled,
                                                bool whole = true) noexcept {
    return (whole && move_is_rigid && rigid_skip_enabled) ? SiteClass::Rigid : SiteClass::Flex;
}

/// The moved atoms of one proposal and their class, read straight from the
/// patch (moved_indices, moving_atoms): O(1) to build, no scan of all atoms.
struct MoveFootprint {
    const int* moved = nullptr;   ///< patch.moved_indices
    std::size_t n_moved = 0;
    const std::uint8_t* is_moved = nullptr;
    SiteClass moved_class = SiteClass::Flex;

    [[nodiscard]] static MoveFootprint of(const ProposalPatch& patch,
                                          bool rigid_skip_enabled) noexcept {
        MoveFootprint f;
        f.moved = patch.moved_indices.data();
        f.n_moved = patch.moved_indices.size();
        f.is_moved = patch.moving_atoms.data();
        f.moved_class = moved_site_class(patch.is_rigid, rigid_skip_enabled);
        return f;
    }
    [[nodiscard]] SiteClass class_of(int site) const noexcept {
        return is_moved[static_cast<std::size_t>(site)] ? moved_class : SiteClass::Fixed;
    }
    /// Moved sites keep their mutual distances, so moved-moved pairs can be skipped.
    [[nodiscard]] bool moved_rigid() const noexcept { return moved_class == SiteClass::Rigid; }
    [[nodiscard]] bool any_flex() const noexcept {
        return moved_class == SiteClass::Flex && n_moved != 0;
    }
};

}  // namespace mcpu::neighbor
