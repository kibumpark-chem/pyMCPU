#pragma once
#include <array>
#include <vector>
#include <memory>
#include <cstdint>
#include <string>
#include "pymcpu/EnergyWeights.h"
#include "pymcpu/Potential.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/AtomPermutation.h"
#include "pymcpu/moves/RotamerLibrary.h"
#include "pymcpu/moves/RamaMixtureLibrary.h"

namespace mcpu {

class Context;
class State;

/// How residue energy masks interact with Mu steric clash / contact terms.
enum class EnergyMaskMode : std::uint8_t {
    IgnoreAll = 0,  ///< skip clash + contact for masked residues
    ClashOnly = 1,  ///< keep hard-clash detection; zero contact energy
};

// ---------------------------------------------------------------
// Per-residue atom block descriptor.
// sc_start == -1  →  no sidechain atoms (GLY)
// h_start  == -1  →  PRO         (no amide hydrogen)
// o_start  == -1  →  reserved for future use
// bb_start is ALWAYS valid for every residue.
// ---------------------------------------------------------------
struct BlockIndices {
    int bb_start = -1;          // N (CA is always bb_start+1)
    int c_start  = -1;          // carbonyl C; -1 → legacy bb_start+2
    int sc_start = -1;
    int o_start  = -1;
    int h_start  = -1;          // explicit amide H atom index, or -1 if none/virtual
    bool amide_donor = false;   // non-PRO donor residue (r>=1); H may be virtual
    int sc_count = 0;           // contiguous SC atoms from sc_start (0 if none)
    int res_begin = -1;         // first atom of residue block (-1 → unknown / legacy)
    int res_end   = -1;         // one-past-last atom of residue block

    // -- Semantic helpers: hide raw -1 checks at all call sites --
    [[nodiscard]] bool has_explicit_h() const noexcept { return h_start >= 0; }

    [[nodiscard]] int n_atom() const noexcept { return bb_start; }
    [[nodiscard]] int ca_atom() const noexcept { return bb_start + 1; }
    [[nodiscard]] int c_atom() const noexcept {
        return c_start >= 0 ? c_start : (bb_start + 2);
    }
    [[nodiscard]] bool has_residue_span() const noexcept {
        return res_begin >= 0 && res_end > res_begin;
    }
};

// ---------------------------------------------------------------
// Precomputed O(1) downstream-lookup results.
// Built ONCE by the Python builder after all block_indices are set.
// first_sc_of_residue[r] = sc_start of the first residue >= r with SC atoms,
//                    OR sc_segment_end if no SC exists from r onwards.
// Zero-overhead at runtime — no scan loops needed in the hot path.
// ---------------------------------------------------------------
struct DownstreamCache {
    std::vector<int> first_sc_of_residue;
    std::vector<int> first_o_of_residue;
    std::vector<int> first_h_of_residue;
};

// ---------------------------------------------------------------
// The loop-closure move's bond lengths, bond angles and omegas, measured ONCE
// from the start structure. Re-measured from the current coordinates on every
// proposal, a bad closure would become the next move's target and N-CA-C
// would drift without bound.
// Internal coordinates only (no atom indices), so an atom permutation, a
// replica exchange or a checkpoint restore -- which replace positions, never
// the System -- cannot change them. Double precision on the float32 start
// coordinates. Per residue k: |N-CA|, |CA-C|, N-CA-C. Per peptide bond
// k -> k+1 (size n_res - 1): |C(k)-N(k+1)|, CA(k)-C(k)-N(k+1),
// C(k)-N(k+1)-CA(k+1) and omega = CA(k)-C(k)-N(k+1)-CA(k+1).
// ---------------------------------------------------------------
struct KicReference {
    std::vector<double> len_na, len_ac, ang_nac;
    std::vector<double> len_cn, ang_acn, ang_cna, omega;
    [[nodiscard]] bool empty() const noexcept { return len_na.empty(); }
};

// ---------------------------------------------------------------
// Atom memory layout:
//
// Legacy (default / reorder off):
//   [BB N,CA,C ...][O ...][SC ...][H ...]
//
// Residue-contiguous (init_only reorder):
//   For each residue (spatial CA-cell order):
//     [N, CA] + sidechain_DFS(from CA) + [C, O] (+ optional amide H)
//   All atoms of a residue occupy [res_begin, res_end).
// ---------------------------------------------------------------


// TODO: Simplify sc_segment_end if possible
class System {
private:
    int num_atoms;
    int num_residues;
    std::vector<std::shared_ptr<Potential>> potentials;
    bool reads_neighbor_grids_ = false;  ///< some potential reads the neighbour grids

    // Variables set by Python builders
    int total_bb_atoms = 0;
    int total_o_atoms  = 0;
    int total_sc_atoms = 0;
    int total_h_atoms  = 0;
    bool virtual_amide_h_ = true;  // default: HBond-only virtual amide H (legacy CheckHBond)
    bool residue_contiguous_layout_ = false;
    AtomPermutation applied_perm_ = AtomPermutation::identity(0);
    bool atoms_reordered_ = false;
    /// Remap every potential's atom ids and record perm as applied.
    void permute_potentials(const AtomPermutation& perm);
    std::vector<BlockIndices>  block_indices;
    std::vector<int>           ntorsions_per_residue;
    // chi_atom_indices_[r][k] = {i1,i2,i3,i4}: current/internal atom indices
    // for residue r's k-th chi dihedral (k < ntorsions_per_residue[r]);
    // {-1,-1,-1,-1} for k >= that residue's torsion count. Resolved from
    // real per-residue atom-NAME topology at build time (see
    // MCPUForceField._initialize_attributes / standard_amino_acids.json's
    // "chi_atoms"), not from positional sc_start+k offsets -- those get the
    // wrong atom for any branched sidechain (e.g. ILE's chi2 needs CD1, not
    // the 3rd stored sidechain atom CG2).
    std::vector<std::array<std::array<int, 4>, 4>> chi_atom_indices_;
    // chi_moved_ranges_[r][k] = {lo, hi}: half-open [lo, hi) engine atom index
    // range of the atoms distal to (rotated by) residue r's k-th chi bond
    // (k < ntorsions_per_residue[r]); {-1,-1} for k >= that residue's torsion
    // count. Resolved from standard_amino_acids.json's "chi_moved_atoms"
    // (transcribed from legacy's amino_torsion.data section 3) at build time,
    // with contiguity asserted Python-side -- NOT a positional
    // sc_start+k formula (e.g. ILE's chi2 moves only CD1, a single atom that
    // sits after CG2 in storage order, not "everything from CG2 onward").
    // Used by the rotamer-library sidechain move's per-chi cascading rotation
    // (MCIntegrator::apply_rotamer_at); the existing continuous sidechain
    // move (chi1-only) does not need this table.
    std::vector<std::array<std::array<int, 2>, 4>> chi_moved_ranges_;
    // Per-amino-acid-type discrete rotamer table (bbind02.May.lib), used only
    // by the rotamer-library sidechain move. Default-constructed empty and
    // simply unused unless that move mode is selected (see Integrator's
    // SidechainMoveMode) -- same pattern as chi_atom_indices_ being unused by
    // moves that don't need it.
    RotamerLibrary rotamer_library_;
    // Per-residue-category (phi, psi) mixture, used only by the
    // knowledge-based backbone pivot move. Default-constructed empty and
    // simply unused unless pivot_rama_probability > 0 (see Integrator) --
    // same pattern as rotamer_library_ above.
    RamaMixtureLibrary rama_mixture_library_;
    std::vector<uint8_t>       is_proline_;  // size n_res; 1 if PRO
    // size n_res; 0-19 per legacy pdb_util.h GetAminoNumber() alphabetical order
    // (ALA=0 ... VAL=19), used to index HBondPotential's sequence-dependent scaling table.
    std::vector<uint8_t>       amino_index_;
    // size n_res; legacy hbonds.h secstr[] state ('H'/'E'/'L'/'C'). Empty ⇒ every
    // residue reads as 'C' (today's byte-identical default; DSSP-derived SS is opt-in).
    std::string                secondary_structure_;
    DownstreamCache            downstream;

    /// Per-residue mask (1 = ignored). Empty / inactive when has_energy_mask_ is false.
    std::vector<uint8_t> energy_ignored_mask_;
    EnergyMaskMode energy_mask_mode_ = EnergyMaskMode::IgnoreAll;
    bool has_energy_mask_ = false;
    /// Changes whenever the mask is set or cleared, and is unique across every
    /// System in the process, so a contact list built under one mask is never
    /// reused under another (State::mu_list_mask_epoch).
    std::uint64_t energy_mask_epoch_ = 0;

    KicReference kic_reference_;  // start-structure closure targets (setKicReference)

public:
    System(int atoms, int residues);

    std::vector<int> atom_to_residue;

    // Controlled setters for the Python builder
    void setAtomCounts(int bb, int o, int sc, int h) noexcept {
        total_bb_atoms = bb;
        total_o_atoms  = o;
        total_sc_atoms = sc;
        total_h_atoms  = h;
    }
    void setVirtualAmideH(bool on) noexcept { virtual_amide_h_ = on; }
    [[nodiscard]] bool virtualAmideH() const noexcept { return virtual_amide_h_; }
    [[nodiscard]] bool residueContiguousLayout() const noexcept {
        return residue_contiguous_layout_;
    }
    void setBlockIndices(std::vector<BlockIndices> indices) {
        block_indices = std::move(indices);
    }
    void setTorsionsPerResidue(std::vector<int> torsions) {
        ntorsions_per_residue = std::move(torsions);
    }
    void setChiAtomIndices(std::vector<std::array<std::array<int, 4>, 4>> idx) {
        chi_atom_indices_ = std::move(idx);
    }
    void setChiMovedAtomRanges(std::vector<std::array<std::array<int, 2>, 4>> ranges) {
        chi_moved_ranges_ = std::move(ranges);
    }
    void setRotamerLibrary(RotamerLibrary lib) {
        rotamer_library_ = std::move(lib);
    }
    [[nodiscard]] const RotamerLibrary& getRotamerLibrary() const noexcept {
        return rotamer_library_;
    }
    void setRamaMixtureLibrary(RamaMixtureLibrary lib) {
        rama_mixture_library_ = std::move(lib);
    }
    [[nodiscard]] const RamaMixtureLibrary& getRamaMixtureLibrary() const noexcept {
        return rama_mixture_library_;
    }
    void setDownstreamCache(DownstreamCache cache) {
        downstream = std::move(cache);
    }
    void setIsProline(std::vector<uint8_t> flags) {
        is_proline_ = std::move(flags);
    }
    [[nodiscard]] bool is_proline(int res_id) const noexcept {
        if (res_id < 0 || res_id >= static_cast<int>(is_proline_.size())) return false;
        return is_proline_[static_cast<size_t>(res_id)] != 0;
    }

    /// Store the KIC closure targets from the START coordinates (3 x num_atoms,
    /// Angstrom, build order -- the coordinates the replicas are positioned with).
    /// Needs block indices; refused after a Context has reordered the atoms.
    void setKicReference(const Eigen::Matrix3Xf& start_coords);
    [[nodiscard]] bool hasKicReference() const noexcept { return !kic_reference_.empty(); }
    [[nodiscard]] const KicReference& kicReference() const noexcept { return kic_reference_; }

    void setAminoIndex(std::vector<uint8_t> indices) {
        amino_index_ = std::move(indices);
    }
    [[nodiscard]] int amino_index(int res_id) const noexcept {
        if (res_id < 0 || res_id >= static_cast<int>(amino_index_.size())) return 0;
        return static_cast<int>(amino_index_[static_cast<size_t>(res_id)]);
    }

    void setSecondaryStructure(std::string ss) {
        secondary_structure_ = std::move(ss);
    }
    [[nodiscard]] char secondary_structure(int res_id) const noexcept {
        if (secondary_structure_.empty()) return 'C';
        if (res_id < 0 || res_id >= static_cast<int>(secondary_structure_.size())) return 'C';
        return secondary_structure_[static_cast<size_t>(res_id)];
    }

    /// Mark residues whose energy terms are suppressed (linker masking).
    void set_energy_ignored_residues(const std::vector<int>& residues,
                                     EnergyMaskMode mode);
    void clear_energy_ignored_residues() noexcept;
    [[nodiscard]] bool is_residue_energy_ignored(int res) const noexcept;
    [[nodiscard]] bool has_energy_mask() const noexcept { return has_energy_mask_; }
    [[nodiscard]] std::uint64_t energy_mask_epoch() const noexcept {
        return energy_mask_epoch_;
    }
    [[nodiscard]] EnergyMaskMode energy_mask_mode() const noexcept {
        return energy_mask_mode_;
    }
    [[nodiscard]] const std::vector<uint8_t>& energy_ignored_mask() const noexcept {
        return energy_ignored_mask_;
    }

    /// Remap BlockIndices / atom_to_residue / DownstreamCache under an atom permutation.
    /// External ids in tables are rewritten to internal ids (int_to_ext direction).
    void apply_atom_permutation(const AtomPermutation& perm);

    /// Install residue-contiguous tree-order BlockIndices + remap atom_to_residue.
    void apply_residue_contiguous_blocks(std::vector<BlockIndices> blocks,
                                         const AtomPermutation& perm);

    /// True once an atom permutation has been applied to this System. Both
    /// functions above also remap every potential's atom ids, and refuse a
    /// second permutation; addPotential maps a potential added afterwards,
    /// since callers always build potentials from build-order ids.
    [[nodiscard]] bool atoms_reordered() const noexcept { return atoms_reordered_; }
    [[nodiscard]] const AtomPermutation& applied_atom_permutation() const noexcept {
        return applied_perm_;
    }

    // Read-only access
    int getTotalBBAtoms() const noexcept { return total_bb_atoms; }
    int getTotalOAtoms()  const noexcept { return total_o_atoms;  }
    int getTotalSCAtoms() const noexcept { return total_sc_atoms; }
    int getTotalHAtoms()  const noexcept { return total_h_atoms;  }
    const std::vector<BlockIndices>& getBlockIndices() const noexcept { return block_indices; }
    const std::vector<int>& getTorsionsPerResidue()   const noexcept { return ntorsions_per_residue; }
    const std::vector<std::array<std::array<int, 4>, 4>>& getChiAtomIndices() const noexcept {
        return chi_atom_indices_;
    }
    const std::vector<std::array<std::array<int, 2>, 4>>& getChiMovedAtomRanges() const noexcept {
        return chi_moved_ranges_;
    }
    const DownstreamCache&  getDownstreamCache()      const noexcept { return downstream; }

    /// True if atom is an explicit amide H (excluded from Mu occupancy).
    [[nodiscard]] bool is_amide_h_atom(int atom_id) const noexcept;

    // -- Computed segment boundaries (call after builder sets totals) --
    [[nodiscard]] int sc_segment_end() const noexcept {
        return total_bb_atoms + total_o_atoms + total_sc_atoms;
    }
    [[nodiscard]] int h_segment_start() const noexcept {
        return sc_segment_end();
    }

    int  addPotential(std::shared_ptr<Potential> potential);
    int  getNumAtoms()    const noexcept;
    int  getNumResidues() const noexcept;

    const std::vector<std::shared_ptr<Potential>>& getPotentials() const;
    /// Whether any potential, enabled or not, reads the neighbour grids
    /// (Potential::readsNeighborGrids). O(1).
    bool readsNeighborGrids() const noexcept { return reads_neighbor_grids_; }

    /// (group, name) for every energy group that has a potential, sorted by
    /// group id. Disabled potentials are included. A group whose potentials
    /// are all unnamed is reported as "group_<n>". Throws std::invalid_argument if
    /// one group carries two different names or one name is used by two
    /// groups -- the same check addPotential runs.
    std::vector<std::pair<int, std::string>> energyTerms() const;

    /// Weighted total: Σ_g weight[g] * E_raw[g] (see EnergyWeights).
    /// When target_group >= 0, returns weight[g] * E_raw[g] for that group only.
    double getTotalEnergy(
        const Context& ctx,
        const State&   state,
        int            target_group = -1) const;

    /// Unweighted (raw) total / per-group energy from Potential::calculateEnergy.
    double getTotalEnergyRaw(
        const Context& ctx,
        const State&   state,
        int            target_group = -1) const;

    double getDeltaEnergy(
        const Context&      ctx,
        const State&        old_state,
        const State&        proposed_state,
        const ProposalPatch& patch) const;

    /// Weighted incremental ΔE with hard-rejection reason (Integrator hot path).
    EnergyChangeResult evaluateDeltaEnergy(
        const Context&       ctx,
        const State&         old_state,
        const State&         proposed_state,
        const ProposalPatch& patch) const;

    /// Weighted total energy with optional hard-rejection reason (baseline).
    /// With resync, each term is evaluated through Potential::resyncEnergy.
    TotalEnergyResult evaluateTotalEnergy(
        const Context& ctx,
        const State&   state,
        int            target_group = -1,
        bool           resync = false) const;

    EnergyBreakdown energyBreakdown(const Context& ctx, const State& state) const;
};

} // namespace mcpu