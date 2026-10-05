#pragma once
/// CA-CA excluded volume: the steric floor KORP does not supply.
///
/// KORP is a scoring potential. It ranks structures well but carries no
/// hard-core repulsion, so on its own it will happily let a chain pass through
/// itself during Monte Carlo -- the table simply has no entry that says a 1 A
/// CA-CA contact is impossible, because no such contact appears in the PDB it
/// was fitted to.
///
/// This term is a pure FILTER, not an energy: it contributes exactly zero to
/// every accepted state and returns the engine's clash sentinel otherwise. That
/// is deliberate. A soft repulsion would be tunable, but it would also add an
/// unvalidated energy scale on top of a fitted potential and shift the
/// ensemble; a filter cannot, because it only deletes configurations that KORP
/// was never fitted to score.
///
/// A move is rejected if it puts a CA pair under min_distance. A whole state
/// is judged against a floor kStateClashBufferA (0.001 A) lower, which allows
/// for a pair a rigid pivot carried a few 1e-6 A under min_distance by
/// rounding.
///
/// The coarse-graining matches KORP's: one sphere per residue at CA. Note the
/// limit that comes with that -- 2 A spheres spaced 3.8 A apart along the
/// backbone leave gaps, so this reliably prevents collapse but does not
/// rigorously prevent one strand threading through another the way an
/// all-backbone-atom guard would. With finite MC step sizes that is the usual
/// coarse-grained trade; if threading shows up in practice the upgrade is a
/// 4x4 N/CA/C/O radius table over the same neighbour structure.

#include <cstdint>
#include <vector>

#include "pymcpu/Potential.h"

namespace mcpu::forces {

class CalphaExcludedVolumePotential : public Potential {
public:
    /// Matches the engine-wide clash sentinel (see MuPotential and
    /// PhysicsVerifier): anything at or above half of it reads as a clash.
    static constexpr float kClashPenalty = 99999.0f;

    /// @param ca_atom         per residue: engine atom index of CA
    /// @param seq_number      per residue: PDB residue number
    /// @param chain_id        per residue: chain tag
    /// @param min_separation  pairs closer than this in sequence are exempt.
    ///                        Consecutive CAs sit at ~3.8 A, below any useful
    ///                        floor, so the bonded ones have to be excused.
    ///                        Pairs on different chains are never exempt.
    /// @param min_distance    hard floor in Angstrom. The default is measured,
    ///                        not assumed: across 1CEO, 1DOS, T0860D1, actin
    ///                        and chignolin the closest CA-CA contact at three
    ///                        or more apart in sequence is 3.53 A, and those
    ///                        are real packing contacts rather than numbering
    ///                        artefacts -- actin has no numbering gaps at all
    ///                        and still reaches 3.88 A at separation 9. A floor
    ///                        of 4.0 A therefore rejects native structures. 3.2
    ///                        leaves room under the observed minimum for
    ///                        thermal fluctuation while still being far above
    ///                        the ~2 A where two backbones would have to
    ///                        interpenetrate.
    CalphaExcludedVolumePotential(
        std::vector<int> ca_atom,
        std::vector<int> seq_number,
        std::vector<std::uint8_t> chain_id,
        int min_separation = 3,
        float min_distance = 3.2f);

    [[nodiscard]] int num_residues() const noexcept {
        return static_cast<int>(ca_atom_.size());
    }
    [[nodiscard]] float min_distance() const noexcept { return min_distance_; }
    [[nodiscard]] int min_separation() const noexcept { return min_separation_; }

    void permute_atom_indices(const AtomPermutation& perm) override;

    float calculateEnergy(const Context& context, const State& state) const override;

    EnergyChangeResult calculateEnergyChange(
        const Context& context,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& patch) const override;

    bool canHardReject() const noexcept override { return true; }

    /// The delta's test: a moved CA under min_distance of another, except a
    /// pair a rigid move carries (with skip_rigid_mm on).
    bool clashesAtMoveCutoff(
        const Context& context,
        const State& proposed_state,
        const ProposalPatch& patch) const override;

    RejectReason rejectionForEnergy(float energy) const noexcept override {
        return energy >= kClashPenalty * 0.5f
            ? RejectReason::StericClash
            : RejectReason::None;
    }

private:
    [[nodiscard]] bool pair_is_checked(int a, int b) const noexcept;

    std::vector<int> ca_atom_;
    std::vector<int> seq_number_;
    std::vector<std::uint8_t> chain_id_;
    int min_separation_;
    float min_distance_, min_distance_sq_;
    /// calculateEnergy's floor: min_distance less kStateClashBufferA, squared.
    float min_distance_state_sq_;

    /// -1 where the atom is not some residue's CA.
    std::vector<int> residue_of_ca_;
    mutable std::vector<std::uint8_t> moved_;
    mutable std::vector<int> moved_residues_;
    /// Proposed CA coordinates gathered into x/y/z arrays, and chain ids
    /// widened to int, for the branch-free prefilter in clashesAtMoveCutoff.
    mutable std::vector<float> cx_, cy_, cz_;
    std::vector<int> chain_int_;

    void rebuild_atom_lookup();
};

} // namespace mcpu::forces
