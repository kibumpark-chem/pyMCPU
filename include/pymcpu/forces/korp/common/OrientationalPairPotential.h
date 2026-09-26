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

    /// Turn off the moved-moved elision described in calculateEnergyChange.
    ///
    /// Only useful for testing: with it off the delta enumerates every changed
    /// pair, which is slower but makes no assumption about rigid motion. The
    /// two paths must agree, and a test that compares them is the direct guard
    /// against misclassifying a residue as rigidly moved when its frame was
    /// actually reshaped -- a bug that would otherwise show up only as a slow
    /// drift in the running energy.
    void set_rigid_skip_enabled(bool on) noexcept { rigid_skip_enabled_ = on; }
    [[nodiscard]] bool rigid_skip_enabled() const noexcept { return rigid_skip_enabled_; }

    void permute_atom_indices(const AtomPermutation& perm) override;

    float calculateEnergy(const Context& context, const State& state) const override;

    EnergyChangeResult calculateEnergyChange(
        const Context& context,
        const State& old_state,
        const State& proposed_state,
        const ProposalPatch& patch) const override;

private:
    /// How a residue's frame is affected by the proposed move.
    enum class FrameClass : std::uint8_t {
        Fixed = 0,     ///< none of N/CA/C moved
        RigidMoved,    ///< all three moved, under a rigid move
        Distorted,     ///< anything else: the frame changed shape
    };

    /// Energy of the ordered pair (lo, hi), lo < hi, or 0 if it does not count.
    [[nodiscard]] double pair_energy(
        const std::vector<ResidueFrame>& frames, int lo, int hi) const noexcept;

    void build_frames(const State& state, std::vector<ResidueFrame>& out) const;

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

    bool rigid_skip_enabled_ = true;

    /// Per-call scratch. Single-threaded within one calculate* call and fully
    /// rewritten at the top of it, which is the same arrangement MuPotential
    /// uses for its per-call mask state. Nothing here survives a call, so two
    /// replicas sharing this object cannot desynchronise through it.
    mutable std::vector<ResidueFrame> frames_old_, frames_new_;
    mutable std::vector<FrameClass> cls_;
    mutable std::vector<std::uint8_t> bits_;
    mutable std::vector<int> changed_, touched_;
};

} // namespace mcpu::forces
