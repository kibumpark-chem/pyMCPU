#pragma once
/// KORP's 6D orientation-dependent residue-pair energy.
///
/// One frame per residue, built from that residue's own N, CA and C; the pair
/// coordinate is CA-CA. Nothing here reads a sidechain atom, which is what
/// makes KORP usable as a backbone-only force field.
///
/// The lookup is nearest-bin with no interpolation, so the energy is a step
/// function of geometry. That is fine for Metropolis -- no derivative is ever
/// taken -- and it makes the incremental delta exact by construction rather
/// than an approximation that has to be bounded.

#include <cstdint>
#include <memory>
#include <vector>

#include "pymcpu/Potential.h"
#include "pymcpu/forces/korp/common/OrientationalPairMap.h"
#include "pymcpu/forces/korp/common/ResidueFrame.h"
#include "pymcpu/neighbor/Footprint.h"

namespace mcpu::forces {

class OrientationalPairPotential : public Potential {
public:
    /// @param map          shared, read-only energy map
    /// @param n_atom       per residue: engine atom index of N
    /// @param ca_atom      per residue: engine atom index of CA
    /// @param c_atom       per residue: engine atom index of C
    /// @param korp_type    per residue: 0..19 in KORP's one-letter-alphabetical
    ///                     order, which is NOT pyMCPU's AMINO_INDEX order
    /// @param seq_number   per residue: the PDB residue number. KORP derives
    ///                     sequence separation from these and not from array
    ///                     position, so renumbered or gapped input genuinely
    ///                     scores differently -- upstream does the same.
    /// @param chain_id     per residue: chain tag; pairs spanning two chains
    ///                     are always non-bonding.
    OrientationalPairPotential(
        std::shared_ptr<const OrientationalPairMap> map,
        std::vector<int> n_atom,
        std::vector<int> ca_atom,
        std::vector<int> c_atom,
        std::vector<std::uint8_t> korp_type,
        std::vector<int> seq_number,
        std::vector<std::uint8_t> chain_id);

    [[nodiscard]] int num_residues() const noexcept {
        return static_cast<int>(ca_atom_.size());
    }
    [[nodiscard]] float cutoff_angstrom() const noexcept { return map_->cutoff(); }

    /// Frame for one residue in a given state. Exposed so tests can compare the
    /// engine's geometry against the reference implementation bin for bin,
    /// rather than only through a summed energy where errors can cancel.
    [[nodiscard]] ResidueFrame frame_of(const State& state, int residue) const;

    /// Enable the moved-moved elision described in calculateEnergyChange.
    ///
    /// OFF BY DEFAULT, because for this potential it is not exact. The
    /// argument for it -- two residues carried by one rigid motion keep all six
    /// pair coordinates -- holds in real arithmetic only. The pivot is applied
    /// in float32, and the table is nearest-bin with no interpolation, so a
    /// co-moving pair sitting within rounding of a bin edge can change bin with
    /// nothing entering delta_E. Metropolis then accepts on a wrong delta_E.
    ///
    /// Measured on CLN025, T = 8, pivot-heavy MC: single accepted moves scored
    /// dE 0.052 / 1.221 / 0.000 against a true 3.801 / 2.186 / 5.764; the
    /// running total drifted 17.4 from a full recompute within 1e5 steps with
    /// the elision on, and 2.4e-4 with it off. Events are rare (~1e-4 per
    /// accepted move) but individually large, and Metropolis preferentially
    /// accepts the ones whose hidden cost is positive, so the error has a sign.
    /// A few hundred steps of MCPU_VERIFY_PHYSICS cannot see it.
    ///
    /// Kept only to measure what the elision would buy; the cost of leaving it
    /// off was < 6 % of wall time on NuG2.
    void set_rigid_skip_enabled(bool on) noexcept { rigid_skip_enabled_ = on; }
    [[nodiscard]] bool rigid_skip_enabled() const noexcept { return rigid_skip_enabled_; }

    void permute_atom_indices(const AtomPermutation& perm) override;

    float calculateEnergy(const Context& context, const State& state) const override;

    /// Full energy, and rebuilds state.korp_cache from the same pass, so a
    /// running total reset here starts from a cache with no accumulated drift.
    float resyncEnergy(const Context& context, const State& state) const override;

    /// Folds the last calculateEnergyChange into the accepted state's cache:
    /// new frames for the changed residues, new energies for the pairs whose
    /// energy changed. Drops the cache if that record is for another move.
    void commitAcceptedMove(const Context& context, const State& state,
                            const State& proposed_state,
                            const ProposalPatch& patch) const override;

    EnergyChangeResult calculateEnergyChange(
        const Context& context,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& patch) const override;

private:
    /// How a residue's frame is affected by the proposed move.
    /// The shared site classes: Fixed (none of N/CA/C moved), Rigid (all
    /// three moved, under a rigid move), Flex (anything else: the frame
    /// changed shape).
    using FrameClass = neighbor::SiteClass;

    /// Energy of the ordered pair (lo, hi), lo < hi, or 0 if it does not count.
    /// The table entry and slice weight of the pair (lo < hi): false when the
    /// pair scores zero (beyond the cutoff, too close in sequence, or under
    /// min_r). pair_energy is weight * entry, in double.
    bool pair_entry(const std::vector<ResidueFrame>& frames, int lo, int hi,
                    std::size_t& index, float& weight) const noexcept;
    [[nodiscard]] double pair_energy(
        const std::vector<ResidueFrame>& frames, int lo, int hi) const noexcept;

    void build_frames(const State& state, std::vector<ResidueFrame>& out) const;

    /// Build state.korp_cache from scratch and return the total energy, summed
    /// in double exactly as calculateEnergy does.
    double fill_cache(const State& state) const;

    /// atom index -> (residue, which frame-atom bit), so classifying a move is
    /// linear in the moved-atom count instead of scanning every residue for
    /// every moved atom. Rebuilt whenever the atom indices change.
    void rebuild_atom_lookup();

    /// Classify every residue and collect the non-Fixed ones. Returns false
    /// when nothing that matters moved, in which case the delta is exactly 0.
    bool classify(const ProposalPatch& patch) const;

    std::shared_ptr<const OrientationalPairMap> map_;
    std::vector<int> n_atom_, ca_atom_, c_atom_;
    std::vector<std::uint8_t> korp_type_;
    std::vector<int> seq_number_;
    std::vector<std::uint8_t> chain_id_;

    /// -1 where the atom is not one of some residue's N/CA/C.
    std::vector<int> frame_residue_of_atom_;
    std::vector<std::uint8_t> frame_bit_of_atom_;

    bool rigid_skip_enabled_ = false;   // see set_rigid_skip_enabled: not exact here

    /// Per-call scratch. Single-threaded within one calculate* call and fully
    /// rewritten at the top of it, which is the same arrangement MuPotential
    /// uses for its per-call mask state. The one exception is that
    /// commitAcceptedMove reads changed_ and frames_new_ with pending_, which
    /// records the state and proposal they were computed for, so two replicas
    /// sharing this object cannot desynchronise through it.
    mutable std::vector<ResidueFrame> frames_old_, frames_new_;
    mutable std::vector<FrameClass> cls_;
    mutable std::vector<std::uint8_t> bits_;
    mutable std::vector<int> changed_, touched_;

    /// Origins of frames_new_ as separate x/y/z arrays, for the distance
    /// prefilter in calculateEnergyChange.
    mutable std::vector<double> ox_, oy_, oz_;
    // Per changed residue: the partners to visit (index, or ~index when
    // beyond the prefilter cutoff), and each one's table entry and weight.
    mutable std::vector<int> cand_;
    mutable std::vector<std::size_t> entry_;
    mutable std::vector<float> weight_;

    /// What the last calculateEnergyChange would change in the accepted
    /// state's cache. Unlike the scratch above it survives until the next
    /// commitAcceptedMove, which applies it only if it was computed against
    /// that same state, proposal and cache generation.
    struct PendingPair { std::int32_t i, j; float energy; };
    struct Pending {
        bool valid = false;
        const State* old_state = nullptr;
        const State* proposed_state = nullptr;
        std::uint64_t generation = 0;
        std::size_t num_moved = 0;
        std::vector<PendingPair> pairs;
    };
    mutable Pending pending_;
};

} // namespace mcpu::forces
