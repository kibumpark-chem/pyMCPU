#include "pymcpu/Integrator.h"
#include "pymcpu/moves/TripeptideClosure.h"
#include "pymcpu/moves/RotamerLibrary.h"
#include "pymcpu/moves/RamaMixtureLibrary.h"
#include "pymcpu/System.h"
#include "pymcpu/Context.h"
#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/Profiler.h"
#include "pymcpu/testing/PhysicsVerifier.h"
#include "pymcpu/utils/geometry_utils.h"
#include "pymcpu/utils/sidechain_torsion_utils.h"
#include "pymcpu/reporters/XTCReporter.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
/// Full recompute after every accepted move, to locate the move that introduces a
/// hard-core violation (MCPU_CLASH_TRACE=1). Diagnostic only -- O(N^2) per accept.
bool mcpu_clash_trace_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("MCPU_CLASH_TRACE");
        return e && e[0] == '1';
    }();
    return on;
}
}  // namespace

using namespace mcpu;

namespace {

void print_step_stats_summary(const StepStats& stats) {
    if (stats.n_steps == 0) {
        std::cerr << "[step_stats] n_steps=0\n";
        return;
    }
    const double inv = 1.0 / static_cast<double>(stats.n_steps);
    const double total = static_cast<double>(stats.step_total_ns);
    const auto pct = [&](std::uint64_t ns) -> double {
        return total > 0.0 ? (100.0 * static_cast<double>(ns) / total) : 0.0;
    };
    const double avg_moved = static_cast<double>(stats.moved_atoms_sum) * inv;
    std::cerr << "[step_stats] n_steps=" << stats.n_steps
              << " valid_moves=" << stats.n_valid_moves
              << " accepts=" << stats.n_accepts
              << " avg_moved_atoms=" << avg_moved
              << " copy_dynamic_ns=" << stats.copy_dynamic_ns
              << " (" << pct(stats.copy_dynamic_ns) << "%)"
              << " delta_energy_ns=" << stats.delta_energy_ns
              << " (" << pct(stats.delta_energy_ns) << "%)"
              << " commit_ns=" << stats.commit_ns
              << " (" << pct(stats.commit_ns) << "%)"
              << " step_total_ns=" << stats.step_total_ns
              << '\n';
}

// Recompute residue r's cached backbone/sidechain torsion angles from its
// CURRENT coordinates in `st`, mirroring Context::computeTorsions()'s
// per-residue formula exactly.
//
// Every move-application function (Pivot/KIC/Sidechain) MUST call the
// matching one of these for every residue it lists in ProposalPatch's
// distorted_bb_residues/distorted_sc_residues, on the *proposal* state,
// before energy evaluation. TripletPotential/SidechainTripletPotential's
// calculateEnergyChange() reads these cached angles directly rather than
// recomputing from coordinates; with the default pooled/sparse proposal
// reuse the `proposal` object is never otherwise fully resynced between
// steps, so a residue whose angle isn't refreshed here keeps
// old_state.<angle> == proposed_state.<angle> (both frozen at whatever the
// last full resync produced) — making that residue's contribution to the
// Metropolis delta silently zero for this move, regardless of what was
// actually proposed. (Confirmed via a same-engine setPositions
// self-consistency check: energy changes after enough accepted moves even
// though setPositions is a geometric no-op — see the pymcpu-westpa repo.)
void recompute_backbone_torsion(State& st, const System& system, int r) {
    const auto& blocks = system.getBlockIndices();
    const int n_res = system.getNumResidues();
    if (r < 1 || r >= n_res - 1) return;
    const int n = blocks[static_cast<size_t>(r)].bb_start;
    const int ca = blocks[static_cast<size_t>(r)].ca_atom();
    const int c = blocks[static_cast<size_t>(r)].c_atom();
    const int n_prev = blocks[static_cast<size_t>(r - 1)].bb_start;
    const int ca_prev = blocks[static_cast<size_t>(r - 1)].ca_atom();
    const int o_prev = blocks[static_cast<size_t>(r - 1)].o_start;
    const int n_next = blocks[static_cast<size_t>(r + 1)].bb_start;
    const int ca_next = blocks[static_cast<size_t>(r + 1)].ca_atom();
    const int o_next = blocks[static_cast<size_t>(r + 1)].o_start;

    const float phi = GeometryUtils::calculate_dihedral(
        st.atom_pos(blocks[static_cast<size_t>(r - 1)].c_atom()), st.atom_pos(n), st.atom_pos(ca), st.atom_pos(c));
    const float psi = GeometryUtils::calculate_dihedral(
        st.atom_pos(n), st.atom_pos(ca), st.atom_pos(c), st.atom_pos(n_next));
    const float pca = GeometryUtils::calculate_a_PCA(
        st.atom_pos(n_prev), st.atom_pos(ca_prev), st.atom_pos(o_prev),
        st.atom_pos(n_next), st.atom_pos(ca_next), st.atom_pos(o_next));
    const float bca = GeometryUtils::calculate_a_bCA(
        st.atom_pos(n_prev), st.atom_pos(ca_prev), st.atom_pos(o_prev),
        st.atom_pos(n_next), st.atom_pos(ca_next), st.atom_pos(o_next));
    st.backbone_torsions[static_cast<size_t>(r)] = BackboneTorsionAngles(phi, psi, pca, bca);
}

void recompute_sidechain_torsion(State& st, const System& system, int r) {
    const auto& br = system.getBlockIndices()[static_cast<size_t>(r)];
    if (br.sc_start < 0) return;

    // Real per-residue atom-name-resolved chi topology (see
    // sidechain_torsion_utils.h), not a positional sc_start+k assumption --
    // shared with Context::computeTorsions() so the two can never drift
    // out of sync with each other again.
    compute_sidechain_chi_angles(st, system, r);
}

// Applies a per-chi cascading rotation (proximal -> distal) to residue r's
// chi angles, given a target angular delta (radians) for each of its first
// `ntorsions` chi indices. Each chi is rotated about its own bond axis (the
// middle bond of chi_atom_indices_[r][k]), touching only the atoms distal to
// that bond (chi_moved_ranges_[r][k]). Shared by the continuous
// (independent-per-chi-Gaussian) and rotamer-library sidechain moves, which
// differ only in how `delta` is computed -- see their respective callers.
//
// Marks the UNION of all touched ranges in `patch` exactly once, after every
// rotation is applied: per-chi ranges are nested (e.g. a 4-chi residue's
// chi4 range is a subset of its chi1 range), so marking per-chi inside the
// loop would push the same atom index into patch.moved_indices once per
// range it belongs to, violating ProposalPatch::mark_moved's documented "no
// double-push" contract and corrupting every downstream per-atom energy pass
// (confirmed via PhysicsVerifier: duplicated indices broke the Mu-potential
// incremental-vs-full delta-energy consistency check).
//
// Returns false (no-op, caller must leave patch.is_valid false) if `system`
// lacks per-residue chi topology tables for r -- hand-built/synthetic
// Systems used by some lower-level tests set ntorsions_per_residue directly
// without ever calling setChiAtomIndices()/setChiMovedAtomRanges() (see
// sidechain_torsion_utils.h's compute_sidechain_chi_angles for the same
// defensive pattern); real MCPUForceField-built Systems always populate all
// three tables together with matching sizes.
bool apply_chi_cascade(State& proposal, const System& system, int r, int ntorsions,
                       const std::array<float, 4>& delta, ProposalPatch& patch) {
    const auto& chi_idx_all = system.getChiAtomIndices();
    const auto& moved_ranges_all = system.getChiMovedAtomRanges();
    if (static_cast<size_t>(r) >= chi_idx_all.size() ||
        static_cast<size_t>(r) >= moved_ranges_all.size()) {
        return false;
    }
    const auto& chi_idx_table = chi_idx_all[static_cast<size_t>(r)];
    const auto& moved_ranges = moved_ranges_all[static_cast<size_t>(r)];

    int overall_lo = -1, overall_hi = -1;
    for (int k = 0; k < ntorsions; ++k) {
        const size_t ks = static_cast<size_t>(k);
        const int lo = moved_ranges[ks][0];
        const int hi = moved_ranges[ks][1];
        if (lo < 0 || hi <= lo) continue;  // defensive; shouldn't happen for k < ntorsions

        // Each rotation axis is measured from CURRENT coordinates, since a
        // more-proximal chi's rotation may have already moved a more-distal
        // chi's axis atoms; this is safe and order-independent (rotating
        // about chi_k's bond changes only chi_k's own value and leaves every
        // other chi_j, j != k, exactly unchanged -- a dihedral straddling or
        // entirely inside a rigidly-rotated subtree is invariant under that
        // rotation).
        const auto& atoms4 = chi_idx_table[ks];  // {i1, i2, i3, i4}; axis = i2->i3 bond
        const Eigen::Vector3f pos_i2 = proposal.atom_pos(atoms4[1]);
        const Eigen::Vector3f pos_i3 = proposal.atom_pos(atoms4[2]);
        const Eigen::Vector3d pivot = pos_i2.cast<double>();
        const Eigen::Vector3d axis = (pos_i3.cast<double>() - pivot).normalized();
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(static_cast<double>(delta[ks]), axis).toRotationMatrix();
        proposal.rotate_atoms(lo, hi, R, pivot);

        if (overall_lo < 0 || lo < overall_lo) overall_lo = lo;
        if (hi > overall_hi) overall_hi = hi;
    }
    if (overall_lo < 0) return false;  // nothing valid to rotate (defensive)

    for (int i = overall_lo; i < overall_hi; ++i) {
        patch.sc_atom_moved[static_cast<size_t>(i)] = 2;
        patch.mark_moved(i);
    }
    return true;
}

// ---------------------------------------------------------------
// Shared helpers for the knowledge-based backbone (rama-mixture) pivot
// move (MCIntegrator::apply_rama_pivot_at, below).
// ---------------------------------------------------------------

// Resolves which RamaMixtureLibrary category residue r's (phi,psi) should
// be drawn from. Today this is just the residue's own amino-acid identity,
// but is deliberately factored into its own function rather than inlined
// at the call site: this is the extensibility seam for future refinements
// -- e.g. a "pre-proline" category (residues immediately before a PRO have
// a distinct, more restricted Ramachandran distribution due to steric
// clash with the pyrrolidine ring -- a different, well-established effect
// from proline's own excluded phi, see apply_rama_pivot_move). When that
// data exists, this function grows a check like
// `if (r+1 < N && system.is_proline(r+1)) return kPreProCategory;` before
// falling back to amino_index(r); nothing else in apply_rama_pivot_at, the
// dispatch code, or the bindings needs to change.
int resolve_rama_category(const System& system, int r) {
    return system.amino_index(r);
}

// The DownstreamCache-based C-term ("!scattered") atom ranges that a
// single phi- or psi-pivot at residue r would rotate -- extracted from
// apply_pivot_at's !rotate_n_term/!scattered branch (this function's own
// direction is always the C-term one; see apply_rama_pivot_at's docs on
// why the rama-mixture move never uses the N-term direction) so both the
// existing single-dihedral pivot and this joint (phi,psi) move share
// exactly one source of truth for "what atoms does this bond rotation
// touch." Returns empty if the atom layout is scattered (non-identity
// permutation) -- the rama-mixture move requires AtomReorderMode::Off,
// the only mode any production caller uses today.
std::vector<std::pair<int, int>> compute_pivot_c_term_ranges(
    const System& system, const Context& context, int r, bool is_phi) {
    if (!context.atom_permutation_is_identity()) return {};

    const auto& blocks = system.getBlockIndices();
    const int bb_start_contig = is_phi ? blocks[static_cast<size_t>(r)].c_atom()
                                        : blocks[static_cast<size_t>(r + 1)].bb_start;
    const int start_res_sc = is_phi ? r : r + 1;
    const int o_res = r;
    const int h_res = r + 1;

    const int sc_start = system.getDownstreamCache().first_sc_of_residue[
        static_cast<size_t>(start_res_sc)];
    const int o_start = system.getDownstreamCache().first_o_of_residue[
        static_cast<size_t>(o_res)];
    const int h_start = system.getDownstreamCache().first_h_of_residue[
        static_cast<size_t>(h_res)];

    const int bb_end = system.getTotalBBAtoms();
    const int o_end  = bb_end + system.getTotalOAtoms();
    const int sc_end = o_end  + system.getTotalSCAtoms();
    const int h_end  = sc_end + system.getTotalHAtoms();

    std::vector<std::pair<int, int>> ranges;
    ranges.emplace_back(bb_start_contig, bb_end);
    ranges.emplace_back(o_start, o_end);
    ranges.emplace_back(sc_start, sc_end);
    if (system.getTotalHAtoms() > 0) {
        ranges.emplace_back(h_start, h_end);
    }
    return ranges;
}

// Rotates each valid [start,end) range in `ranges` by `R` about `pivot`,
// WITHOUT touching `patch` -- callers must mark the union of all rotated
// ranges themselves via mark_ranges, exactly once.
void rotate_ranges(State& proposal, const std::vector<std::pair<int, int>>& ranges,
                   const Eigen::Matrix3d& R, const Eigen::Vector3d& pivot) {
    for (const auto& span : ranges) {
        if (span.first < span.second) {
            proposal.rotate_atoms(span.first, span.second, R, pivot);
        }
    }
}

// Marks every atom in the UNION of `ranges` into `patch` exactly once.
// Sorts + merges overlapping/adjacent ranges first (mirrors
// apply_pivot_at's own rotate_residue_spans merge) so callers can pass
// phi's and psi's range lists directly without pre-computing the union
// themselves and without any risk of a double-push into
// patch.moved_indices -- the exact bug class apply_chi_cascade's own
// "union... exactly once" comment above warns about (confirmed there via
// PhysicsVerifier: duplicated indices broke the Mu-potential incremental-
// vs-full delta-energy consistency check). Classifies each atom into
// bb/o/sc/h exactly as apply_pivot_at's mark_atom lambda does, since
// (unlike apply_chi_cascade's single-category sidechain range) this
// move's touched union spans all four categories.
void mark_ranges(ProposalPatch& patch, const System& system,
                 std::vector<std::pair<int, int>> ranges) {
    std::sort(ranges.begin(), ranges.end());
    size_t i = 0;
    while (i < ranges.size()) {
        int a = ranges[i].first;
        int b = ranges[i].second;
        size_t j = i + 1;
        while (j < ranges.size() && ranges[j].first <= b) {
            b = std::max(b, ranges[j].second);
            ++j;
        }
        const auto& blocks = system.getBlockIndices();
        for (int idx = a; idx < b; ++idx) {
            const auto& blk = blocks[static_cast<size_t>(
                system.atom_to_residue[static_cast<size_t>(idx)])];
            if (idx == blk.o_start) {
                patch.o_atom_moved[static_cast<size_t>(idx)] = 1;
            } else if (idx == blk.h_start) {
                patch.h_atom_moved[static_cast<size_t>(idx)] = 1;
            } else if (blk.sc_start >= 0 && idx >= blk.sc_start &&
                       idx < blk.sc_start + blk.sc_count) {
                patch.sc_atom_moved[static_cast<size_t>(idx)] = 1;
            } else {
                patch.bb_atom_moved[static_cast<size_t>(idx)] = 1;
            }
            patch.mark_moved(idx);
        }
        i = j;
    }
}

} // namespace

MCIntegrator::MCIntegrator(float temperature, float step_size_rad,
                           float sidechain_step_size_rad)
    : temperature(temperature),
      step_size_rad(step_size_rad),
      // Resolve the "negative means same as backbone" sentinel once, here, so
      // no downstream code ever sees it.
      sidechain_step_size_rad_(sidechain_step_size_rad < 0.0f ? step_size_rad
                                                             : sidechain_step_size_rad),
      pivot_residue_dist(1, 1),   // re-seeded in run() after system is known
      sc_residue_dist(0, 1),
      angle_dist(0.0f, step_size_rad),  // Fix Round2-#5: now wired to parameter
      sc_angle_dist_(0.0f, sidechain_step_size_rad_)
{}

std::string MCIntegrator::get_rng_state() {
    // Discard any pending normal_distribution spare (see header doc) so the
    // checkpoint boundary is cache-free -- this engine and any engine restored
    // from the returned string are then guaranteed to agree on all future draws
    // given the same rng state, regardless of how many angle_dist(rng) calls
    // happened to precede this checkpoint. unit_normal_dist_ (rotamer-library
    // move's per-chi noise) and sc_angle_dist_ (continuous sidechain chi) have
    // the identical caching hazard and need the same treatment.
    angle_dist.reset();
    sc_angle_dist_.reset();
    unit_normal_dist_.reset();
    std::ostringstream oss;
    oss << rng;
    return oss.str();
}

void MCIntegrator::set_rng_state(const std::string& state) {
    std::istringstream iss(state);
    iss >> rng;
    if (iss.fail()) {
        throw std::runtime_error("Integrator::set_rng_state: failed to parse RNG state");
    }
    // Restored engines start with no distributions constructed yet, so this is a
    // no-op in that case -- kept for symmetry with get_rng_state and to protect
    // against restoring into a previously-used (already-run) Integrator instance.
    angle_dist.reset();
    sc_angle_dist_.reset();
    unit_normal_dist_.reset();
}

void MCIntegrator::set_sidechain_move_mode(const std::string& mode) {
    if (mode == "continuous") {
        sidechain_move_mode_ = SidechainMoveMode::Continuous;
    } else if (mode == "rotamer_library") {
        sidechain_move_mode_ = SidechainMoveMode::RotamerLibrary;
    } else {
        throw std::invalid_argument(
            "sidechain_move_mode must be 'continuous' or 'rotamer_library', got: " + mode);
    }
}

// One table for save and restore, so a counter cannot be saved but not
// restored (or the reverse).
const MCIntegrator::MoveCounterTable& MCIntegrator::move_counter_table() {
    static const MoveCounterTable table = {
        {"bb_attempted", &MCIntegrator::bb_attempted_},
        {"bb_accepted", &MCIntegrator::bb_accepted_},
        {"sc_attempted", &MCIntegrator::sc_attempted_},
        {"sc_accepted", &MCIntegrator::sc_accepted_},
        {"kic_attempted", &MCIntegrator::kic_attempted_},
        {"kic_accepted", &MCIntegrator::kic_accepted_},
        {"rotamer_attempted", &MCIntegrator::rotamer_attempted_},
        {"rotamer_accepted", &MCIntegrator::rotamer_accepted_},
        {"rama_pivot_attempted", &MCIntegrator::rama_pivot_attempted_},
        {"rama_pivot_accepted", &MCIntegrator::rama_pivot_accepted_},
        {"kic_presolve_zero", &MCIntegrator::kic_presolve_zero_},
        {"kic_jacobian_invalid", &MCIntegrator::kic_jacobian_invalid_},
        {"kic_geometry_invalid", &MCIntegrator::kic_geometry_invalid_},
        {"kic_reverse_missing", &MCIntegrator::kic_reverse_missing_},
        {"kic_proline_skipped", &MCIntegrator::kic_proline_skipped_},
        {"steric_rejected", &MCIntegrator::steric_rejected_},
        {"fixed_rejected", &MCIntegrator::fixed_rejected_},
        {"num_pivot_resample_pro_phi", &MCIntegrator::num_pivot_resample_pro_phi_},
        {"num_sc_resample_pro", &MCIntegrator::num_sc_resample_pro_},
    };
    return table;
}

std::vector<std::pair<std::string, long long>> MCIntegrator::get_move_counters() const {
    std::vector<std::pair<std::string, long long>> out;
    for (const auto& [name, member] : move_counter_table()) out.emplace_back(name, this->*member);
    return out;
}

void MCIntegrator::set_move_counters(
    const std::vector<std::pair<std::string, long long>>& counters) {
    // Resolve every name before touching anything, so a bad name changes nothing.
    std::vector<std::pair<long long MCIntegrator::*, long long>> resolved;
    for (const auto& [name, value] : counters) {
        long long MCIntegrator::* member = nullptr;
        for (const auto& [known, m] : move_counter_table()) {
            if (name == known) member = m;
        }
        if (!member) {
            throw std::invalid_argument("unknown move counter: '" + name + "'");
        }
        resolved.emplace_back(member, value);
    }
    for (const auto& entry : move_counter_table()) this->*(entry.second) = 0;
    for (const auto& [member, value] : resolved) this->*member = value;
}

void MCIntegrator::set_move_weights(float pivot, float kic, float sidechain) {
    if (!(pivot >= 0.0f && kic >= 0.0f && sidechain >= 0.0f)) {
        throw std::invalid_argument("move weights must be non-negative");
    }
    const float sum = pivot + kic + sidechain;
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        throw std::invalid_argument("move weights must have a positive, finite sum");
    }
    move_w_pivot_ = pivot / sum;
    move_w_kic_   = kic / sum;
    move_w_sc_    = sidechain / sum;
}

void MCIntegrator::check_move_weights_are_usable(const Context& context) const {
    if (move_w_sc_ <= 0.0f) return;
    const std::vector<int>& ntors = context.getSystem().getTorsionsPerResidue();
    const bool any_chi = std::any_of(ntors.begin(), ntors.end(),
                                     [](int n) { return n > 0; });
    if (any_chi) return;
    // Every sidechain proposal would return early with patch.is_valid false,
    // so this share of the step budget is spent producing nothing. That is
    // exactly what a backbone-only force field walks into, and it is silent:
    // the run completes, just with (1 - pivot - kic) of its steps discarded.
    throw std::invalid_argument(
        "MCIntegrator: the sidechain move weight is positive but no residue in "
        "this system has a chi angle, so every sidechain proposal would be a "
        "silent no-op. Call set_move_weights(pivot, kic, 0.0).");
}

std::string MCIntegrator::sidechain_move_mode() const {
    return sidechain_move_mode_ == SidechainMoveMode::RotamerLibrary
        ? "rotamer_library" : "continuous";
}

void MCIntegrator::set_use_pooled_proposal(bool on) {
#if MCPU_USE_POOLED_PROPOSAL
    if (use_pooled_proposal_ == on) return;
    use_pooled_proposal_ = on;
    // Recreate the proposal buffer, and so fully resync it, on the next run.
    pooled_num_atoms_ = -1;
    pooled_num_residues_ = -1;
    proposal_.reset();
#else
    (void)on;
    use_pooled_proposal_ = false;
#endif
}

void MCIntegrator::ensure_proposal_buffers(const Context& context) {
    const int num_atoms = context.getSystem().getNumAtoms();
    const int num_residues = context.getSystem().getNumResidues();
    if (proposal_ && pooled_num_atoms_ == num_atoms &&
        pooled_num_residues_ == num_residues) {
        return;
    }
    proposal_ = std::make_unique<State>(num_atoms, num_residues);
    patch_.ensure_capacity(num_atoms);
    pooled_num_atoms_ = num_atoms;
    pooled_num_residues_ = num_residues;
    proposal_synced_ = false; // CHANGED: sparse — new buffer needs a full sync
}

void MCIntegrator::restore_proposal_from_accepted(State& proposal, const State& accepted,
                                                  const ProposalPatch& patch) {
    if (patch.moved_as_ranges()) {
        for (const auto& rg : patch.moved_ranges)
            proposal.coords_soa.copy_range_from(accepted.coords_soa, rg.first, rg.second);
    } else {
        for (int i : patch.moved_indices) {
            proposal.copy_atom_from(accepted, i, i);
        }
    }
    for (int r : patch.distorted_bb_residues) {
        proposal.backbone_torsions[static_cast<size_t>(r)] =
            accepted.backbone_torsions[static_cast<size_t>(r)];
    }
    for (int r : patch.distorted_sc_residues) {
        proposal.sidechain_torsions[static_cast<size_t>(r)] =
            accepted.sidechain_torsions[static_cast<size_t>(r)];
    }
    proposal.current_energy = accepted.getEnergy();
}

void MCIntegrator::apply_pivot_move(Context& context, State& proposal, ProposalPatch& patch) {
    const System& system = context.getSystem();
    patch.is_valid = false;
    int num_residues = system.getNumResidues();

    constexpr int kMaxPivotResample = 64;
    bool is_phi = false;
    int r = 1;
    for (int attempt = 0; attempt < kMaxPivotResample; ++attempt) {
        r = pivot_residue_dist(rng);
        is_phi = (coin_flip(rng) < 0.5f);
        if (is_phi && system.is_proline(r)) {
            ++num_pivot_resample_pro_phi_;
            continue;
        }
        if (hasFixedResidues()) {
            if (isResidueFixed(r)) {
                ++fixed_rejected_;
                continue;
            }
            // Determine which segment moves and check for fixed residues.
            // For pivot at r: C-term rotates [r+1, n_res), N-term rotates [1, r).
            // Choose the shorter segment; if it contains fixed, try the other.
            const bool c_segment_ok = !segmentContainsFixed(r + 1, num_residues);
            const bool n_segment_ok = !segmentContainsFixed(1, r);
            if (!c_segment_ok && !n_segment_ok) {
                ++fixed_rejected_;
                continue;
            }
            // Force direction to the segment without fixed residues.
            // If both are ok, randomly pick (already fine — apply_pivot_at handles it).
        }
        break;
    }
    if (is_phi && system.is_proline(r)) {
        return;
    }
    if (hasFixedResidues() && isResidueFixed(r)) {
        return;
    }
    apply_pivot_at(context, proposal, patch, r, is_phi);
}

void MCIntegrator::apply_pivot_at(Context& context, State& proposal, ProposalPatch& patch,
                                  int r, bool is_phi) {
    const System& system = context.getSystem();
    patch.is_valid = false;
    int num_residues = system.getNumResidues();
    const auto& blocks = system.getBlockIndices();
    const bool residue_contig = system.residueContiguousLayout();
    const bool scattered = !context.atom_permutation_is_identity();

    // ── Direction selection: N-term vs C-term pivot ─────────────────
    // C-term pivot rotates residues [r+1, num_residues).
    // N-term pivot rotates residues [1, r) (residue 0 N-terminus is anchored).
    // Default: choose the shorter segment; if fixed residues constrain, pick the safe side.
    bool rotate_n_term = (r < num_residues / 2);
    if (hasFixedResidues()) {
        const bool c_ok = !segmentContainsFixed(r + 1, num_residues);
        const bool n_ok = !segmentContainsFixed(1, r);
        if (!c_ok && !n_ok) {
            ++fixed_rejected_;
            return;
        }
        if (!c_ok) rotate_n_term = true;
        else if (!n_ok) rotate_n_term = false;
        // else both ok — keep heuristic choice
    }

    int idx_N  = blocks[static_cast<size_t>(r)].bb_start;
    int idx_CA = blocks[static_cast<size_t>(r)].ca_atom();
    int idx_C  = blocks[static_cast<size_t>(r)].c_atom();

    Eigen::Vector3f pos_A, pos_B;
    int start_res_sc, start_res_o, start_res_h;
    bool rotate_pivot_C = false;
    int bb_start_contig = -1;

    if (is_phi) {
        pos_A = proposal.atom_pos(idx_N);
        pos_B = proposal.atom_pos(idx_CA);
        rotate_pivot_C = true;
        bb_start_contig = idx_C;
        start_res_sc = r;
        start_res_o  = r;
        start_res_h  = r + 1;
    } else {
        pos_A = proposal.atom_pos(idx_CA);
        pos_B = proposal.atom_pos(idx_C);
        rotate_pivot_C = false;
        bb_start_contig = blocks[static_cast<size_t>(r + 1)].bb_start;
        start_res_sc = r + 1;
        start_res_o  = r;
        start_res_h  = r + 1;
    }

    float delta_theta = angle_dist(rng);
    if (rotate_n_term) {
        delta_theta = -delta_theta;
    }
    const Eigen::Vector3d pivot_A = pos_A.cast<double>();
    const Eigen::Vector3d axis = (pos_B.cast<double>() - pivot_A).normalized();
    const Eigen::Matrix3d R =
        Eigen::AngleAxisd(static_cast<double>(delta_theta), axis).toRotationMatrix();

    auto rotate_range = [&](int start, int end, std::vector<uint8_t>& moved_flags) {
        if (start < 0 || end <= start) return;
        proposal.rotate_atoms(start, end, R, pivot_A);
        patch.mark_moved_range(start, end, moved_flags);
    };

    auto mark_atom = [&](int i) {
        const auto& b = blocks[static_cast<size_t>(system.atom_to_residue[static_cast<size_t>(i)])];
        if (i == b.o_start) {
            patch.o_atom_moved[static_cast<size_t>(i)] = 1;
        } else if (i == b.h_start) {
            patch.h_atom_moved[static_cast<size_t>(i)] = 1;
        } else if (b.sc_start >= 0 && i >= b.sc_start && i < b.sc_start + b.sc_count) {
            patch.sc_atom_moved[static_cast<size_t>(i)] = 1;
        } else {
            patch.bb_atom_moved[static_cast<size_t>(i)] = 1;
        }
        patch.mark_moved(i);
    };

    auto rotate_and_mark = [&](int start, int end) {
        if (start < 0 || end <= start) return;
        proposal.rotate_atoms(start, end, R, pivot_A);
        for (int i = start; i < end; ++i) mark_atom(i);
    };

    auto rotate_residue_spans = [&](int res_lo, int res_hi_exclusive) {
        std::vector<std::pair<int, int>> spans;
        spans.reserve(static_cast<size_t>(std::max(0, res_hi_exclusive - res_lo)));
        for (int res = res_lo; res < res_hi_exclusive; ++res) {
            const auto& b = blocks[static_cast<size_t>(res)];
            if (b.has_residue_span()) {
                spans.emplace_back(b.res_begin, b.res_end);
            }
        }
        std::sort(spans.begin(), spans.end());
        for (size_t i = 0; i < spans.size(); ) {
            int a = spans[i].first;
            int b = spans[i].second;
            size_t j = i + 1;
            while (j < spans.size() && spans[j].first == b) {
                b = spans[j].second;
                ++j;
            }
            rotate_and_mark(a, b);
            i = j;
        }
    };

    patch.is_valid = true;

    if (!rotate_n_term) {
        // ── C-term pivot (original behaviour) ───────────────────────
        patch.first_affected_residue = r;
        patch.last_affected_residue = num_residues - 1;

        if (!scattered) {
            const int sc_start = system.getDownstreamCache().first_sc_of_residue[
                static_cast<size_t>(start_res_sc)];
            const int o_start  = system.getDownstreamCache().first_o_of_residue[
                static_cast<size_t>(start_res_o)];
            const int h_start  = system.getDownstreamCache().first_h_of_residue[
                static_cast<size_t>(start_res_h)];
            const int bb_end = system.getTotalBBAtoms();
            const int o_end  = bb_end + system.getTotalOAtoms();
            const int sc_end = o_end  + system.getTotalSCAtoms();
            const int h_end  = sc_end + system.getTotalHAtoms();
            rotate_range(bb_start_contig, bb_end, patch.bb_atom_moved);
            rotate_range(o_start, o_end, patch.o_atom_moved);
            rotate_range(sc_start, sc_end, patch.sc_atom_moved);
            if (system.getTotalHAtoms() > 0) {
                rotate_range(h_start, h_end, patch.h_atom_moved);
            }
        } else if (residue_contig) {
            if (is_phi) {
                // KIC FIX (F7): was [CA+1, res_end), which also swung H(r); H(r) is bonded to
                // N(r) and stays with the fixed N side. Rotate SC(r), C(r) and O(r) only.
                const auto& br = blocks[static_cast<size_t>(r)];
                if (br.sc_start >= 0 && br.sc_count > 0)
                    rotate_and_mark(br.sc_start, br.sc_start + br.sc_count);
                rotate_and_mark(br.c_atom(), br.c_atom() + 1);
                if (br.o_start >= 0) rotate_and_mark(br.o_start, br.o_start + 1);
            } else {
                const auto& br = blocks[static_cast<size_t>(r)];
                if (br.o_start >= 0) rotate_and_mark(br.o_start, br.o_start + 1);
            }
            rotate_residue_spans(r + 1, num_residues);
        } else {
            if (rotate_pivot_C) {
                rotate_range(idx_C, idx_C + 1, patch.bb_atom_moved);
            }
            for (int res = r + 1; res < num_residues; ++res) {
                const auto& b = blocks[static_cast<size_t>(res)];
                rotate_range(b.bb_start, b.bb_start + 2, patch.bb_atom_moved);
                rotate_range(b.c_atom(), b.c_atom() + 1, patch.bb_atom_moved);
            }
            for (int res = start_res_o; res < num_residues; ++res) {
                const int o = blocks[static_cast<size_t>(res)].o_start;
                if (o >= 0) rotate_range(o, o + 1, patch.o_atom_moved);
            }
            for (int res = start_res_sc; res < num_residues; ++res) {
                const auto& b = blocks[static_cast<size_t>(res)];
                if (b.sc_start >= 0 && b.sc_count > 0) {
                    rotate_range(b.sc_start, b.sc_start + b.sc_count, patch.sc_atom_moved);
                }
            }
            if (system.getTotalHAtoms() > 0) {
                for (int res = start_res_h; res < num_residues; ++res) {
                    const int h = blocks[static_cast<size_t>(res)].h_start;
                    if (h >= 0) rotate_range(h, h + 1, patch.h_atom_moved);
                }
            }
        }
    } else {
        // ── N-term pivot: rotate residues [0, r) ────────────────────
        patch.first_affected_residue = 0;
        patch.last_affected_residue = r;

        if (residue_contig || scattered) {
            // Per-residue rotate for reordered layouts.
            // KIC FIX (F7): psi used to rotate O(r) (bonded to C(r), which is on the axis, so it
            // belongs to the fixed side) and to leave N(r) and SC(r) behind (both on the moving
            // side). H(r) is bonded to N(r), so it moves for phi and psi alike.
            if (!residue_contig) {
                throw std::runtime_error(
                    "apply_pivot_at: N-terminal pivot on a permuted layout without residue "
                    "spans is not supported");
            }
            rotate_residue_spans(0, r);
            const auto& br = blocks[static_cast<size_t>(r)];
            if (!is_phi) {
                rotate_and_mark(br.bb_start, br.bb_start + 1);
                if (br.sc_start >= 0 && br.sc_count > 0)
                    rotate_and_mark(br.sc_start, br.sc_start + br.sc_count);
            }
            if (br.h_start >= 0) rotate_and_mark(br.h_start, br.h_start + 1);
        } else {
            // Legacy contiguous BB|O|SC|H layout: rotate prefix segments.
            const int bb_end_nterm = is_phi ? idx_N : bb_start_contig;
            rotate_range(blocks[0].bb_start, bb_end_nterm, patch.bb_atom_moved);

            // KIC FIX (F7): was [is_phi ? r : r + 1], which swung O(r) out of the peptide plane
            // on psi. O(r) is bonded to C(r): on the axis for psi, on the fixed side for phi.
            const int o_end_nterm = system.getDownstreamCache().first_o_of_residue[
                static_cast<size_t>(r)];
            const int o_seg_start = system.getDownstreamCache().first_o_of_residue[0];
            rotate_range(o_seg_start, o_end_nterm, patch.o_atom_moved);

            const int sc_end_nterm = system.getDownstreamCache().first_sc_of_residue[
                static_cast<size_t>(is_phi ? r : r + 1)];
            const int sc_seg_start = system.getDownstreamCache().first_sc_of_residue[0];
            rotate_range(sc_seg_start, sc_end_nterm, patch.sc_atom_moved);

            if (system.getTotalHAtoms() > 0) {
                // KIC FIX (F7, explicit-H layout only): was [is_phi ? r : r + 1], which left H(r)
                // behind on phi. H(r) is bonded to N(r), so it moves with the N side for both.
                const int h_end_nterm = system.getDownstreamCache().first_h_of_residue[
                    static_cast<size_t>(r + 1)];
                const int h_seg_start = system.getDownstreamCache().first_h_of_residue[0];
                rotate_range(h_seg_start, h_end_nterm, patch.h_atom_moved);
            }
        }
    }
    patch.is_rigid = true;

    if (r > 0) patch.add_distorted_bb_residue(r - 1);
    patch.add_distorted_bb_residue(r);
    if (r < num_residues - 1) patch.add_distorted_bb_residue(r + 1);

    // Refresh the cached backbone torsions for the residues just marked
    // distorted — see recompute_backbone_torsion()'s docstring above.
    if (r > 0) recompute_backbone_torsion(proposal, system, r - 1);
    recompute_backbone_torsion(proposal, system, r);
    if (r < num_residues - 1) recompute_backbone_torsion(proposal, system, r + 1);
}

// ---------------------------------------------------------------
// THE KNOWLEDGE-BASED BACKBONE (RAMA-MIXTURE) PIVOT MOVE
//
// Jointly resamples (phi, psi) at one residue from a fitted per-residue-
// category wrapped-bivariate-normal mixture (RamaMixtureLibrary; see that
// class's docs), rather than perturbing a single dihedral by a small
// symmetric random step. This is a much bigger, statistically-informed
// jump than the continuous pivot move above.
//
// This is an independence sampler exactly like the rotamer-library
// sidechain move (apply_rotamer_at): the proposal density q(phi',psi')
// does not depend on the current (phi,psi), so q(x'|x) = q(x') and
// q(x|x') = q(x), and the correct Metropolis-Hastings proposal-ratio term
// is log q(x_old) - log q(x_new) -- exactly what ProposalPatch::
// log_jacobian_weight already exists for.
//
// Realizing the target (phi', psi') requires TWO sequential rigid
// rotations (about the N(r)-Ca(r) bond for phi, then about the
// Ca(r)-C(r) bond for psi), because a single-dihedral pivot only ever
// rotates one bond at a time. Each individual rotation leaves the OTHER
// dihedral exactly invariant: rotating about N(r)-Ca(r) (phi) moves only
// atoms strictly downstream of Ca(r) (its sidechain, C(r), O(r), and every
// atom of residues r+1..N-1) as a single rigid body about an axis passing
// through N(r) and Ca(r) -- both of which are themselves ALSO psi's first
// two defining atoms (psi = N(r)-Ca(r)-C(r)-N(r+1)) and are on the
// rotation axis, hence unmoved; the other two psi-defining atoms, C(r)
// and N(r+1), are rotated together as a rigid pair, which cannot change
// the dihedral they define relative to the two fixed axis atoms. The
// symmetric argument holds for phi under the psi rotation (whose touched
// range starts at C(r), never touching phi's own defining atoms
// C(r-1)/N(r)/Ca(r)). So applying phi then psi in sequence, reading each
// rotation's axis fresh from the just-updated coordinates, realizes
// exactly the target (phi', psi') pair, order-independent.
//
// Because psi's touched atom range is a strict subset of phi's (verified
// directly against compute_pivot_c_term_ranges: every one of psi's four
// ranges -- bb/o/sc/h -- is contained in phi's corresponding range for the
// !scattered fast path), applying the two rotations as two separate calls
// into the SAME patch (the way apply_pivot_at is called for a single
// dihedral) would double-push every shared atom index into
// patch.moved_indices. This is exactly the bug class apply_chi_cascade's
// "union... exactly once" comment above already warns about (confirmed
// there via PhysicsVerifier: duplicated indices broke the Mu-potential
// incremental-vs-full delta-energy consistency check) -- so this function
// applies both rotations to raw coordinates first (via rotate_ranges,
// which never touches patch), then marks the UNION of touched ranges
// exactly once (via mark_ranges).
//
// patch.is_rigid is set to FALSE, not true: residue r's sidechain
// undergoes only the phi rotation, while O(r) and residues r+1..N-1
// undergo phi-then-psi -- two different net rotations -- so the whole
// touched set is NOT a single rigid body (pairwise distances between
// those two subsets DO change). This is the same fix already applied for
// apply_rotamer_at's/apply_chi_cascade's multi-chi cascade, for the same
// reason: MuPotential's skip_rigid_mm fast path would otherwise silently
// drop real energy changes between them.
// ---------------------------------------------------------------
void MCIntegrator::apply_rama_pivot_move(Context& context, State& proposal, ProposalPatch& patch) {
    const System& system = context.getSystem();
    patch.is_valid = false;
    const int num_residues = system.getNumResidues();

    constexpr int kMaxPivotResample = 64;
    int r = 1;
    for (int attempt = 0; attempt < kMaxPivotResample; ++attempt) {
        r = pivot_residue_dist(rng);
        if (system.is_proline(r)) {
            ++num_pivot_resample_pro_phi_;
            continue;
        }
        // C-term direction only (see apply_rama_pivot_at's docs on why the
        // rama-mixture move never uses the N-term direction): reject if any
        // residue in the rotated downstream segment [r+1, num_residues) --
        // or r itself -- is fixed. No N-term fallback in v1.
        if (hasFixedResidues() &&
            (isResidueFixed(r) || segmentContainsFixed(r + 1, num_residues))) {
            ++fixed_rejected_;
            continue;
        }
        break;
    }
    if (system.is_proline(r)) return;
    if (hasFixedResidues() &&
        (isResidueFixed(r) || segmentContainsFixed(r + 1, num_residues))) {
        return;
    }
    apply_rama_pivot_at(context, proposal, patch, r);
}

// Realizes an EXPLICIT (phi', psi') target at residue r -- shared by
// apply_rama_pivot_at (which draws the target from the mixture via RNG)
// and debug_force_rama_pivot_to (a test-only hook that skips the RNG draw
// entirely, forcing a hand-picked transition). Factored out so a direct
// numerical detailed-balance check can force the engine through an exact,
// independently-computable state pair and inspect
// last_delta_energy()/log_jacobian_weight against a ground truth computed
// outside the engine, without duplicating steps 4-9's geometry/patch logic
// (whose correctness -- the union-marking/is_rigid handling -- is exactly
// what this whole move exists to get right; a second copy of it would be a
// second place to get it wrong).
bool MCIntegrator::apply_rama_pivot_to_target(Context& context, State& proposal,
                                              ProposalPatch& patch, int r, int category,
                                              const std::array<float, 2>& new_phi_psi) {
    const System& system = context.getSystem();
    patch.is_valid = false;
    const int num_residues = system.getNumResidues();
    const RamaMixtureLibrary& lib = system.getRamaMixtureLibrary();

    // Refresh residue r's cached (phi, psi) from CURRENT geometry before
    // reading it -- same staleness guard apply_rotamer_at applies to
    // old_chi: a passenger rotation by an earlier move can leave cached
    // values off by ~1e-5 rad from true current coordinates, invisible
    // everywhere else but not safe to feed into an MH log-density.
    recompute_backbone_torsion(proposal, system, r);
    const std::array<float, 2> old_phi_psi{
        proposal.backbone_torsions[static_cast<size_t>(r)].phi,
        proposal.backbone_torsions[static_cast<size_t>(r)].psi};

    const float delta_phi = wrap_angle_to_pi(new_phi_psi[0] - old_phi_psi[0]);
    const float delta_psi = wrap_angle_to_pi(new_phi_psi[1] - old_phi_psi[1]);

    // MH correction for this independence-sampler proposal -- computed
    // from the not-yet-mutated old_phi_psi and the target new_phi_psi,
    // BEFORE any coordinates are touched (mirrors apply_rotamer_at's
    // ordering exactly).
    patch.log_jacobian_weight =
        lib.log_mixture_density(category, old_phi_psi) -
        lib.log_mixture_density(category, new_phi_psi);

    // Phi rotation: pivot at N(r), axis N(r)->Ca(r).
    const auto& blocks = system.getBlockIndices();
    const int idx_N = blocks[static_cast<size_t>(r)].bb_start;
    const int idx_CA = blocks[static_cast<size_t>(r)].ca_atom();
    const int idx_C = blocks[static_cast<size_t>(r)].c_atom();
    const Eigen::Vector3f pos_N = proposal.atom_pos(idx_N);
    const Eigen::Vector3f pos_CA = proposal.atom_pos(idx_CA);
    const Eigen::Vector3d pivot_N = pos_N.cast<double>();
    const Eigen::Matrix3d R_phi =
        Eigen::AngleAxisd(static_cast<double>(delta_phi),
                          (pos_CA.cast<double>() - pivot_N).normalized())
            .toRotationMatrix();

    auto phi_ranges = compute_pivot_c_term_ranges(system, context, r, /*is_phi=*/true);
    if (phi_ranges.empty()) return false;  // scattered atom layout: unsupported in v1
    rotate_ranges(proposal, phi_ranges, R_phi, pivot_N);

    // Re-read Ca(r)/C(r) fresh from the now-phi-rotated coordinates
    // (Ca(r) is unchanged by the phi rotation; C(r) just moved) to build
    // the psi axis -- see this function's header comment on why this
    // ordering realizes the target (phi',psi') pair correctly.
    const Eigen::Vector3f pos_CA2 = proposal.atom_pos(idx_CA);
    const Eigen::Vector3f pos_C2 = proposal.atom_pos(idx_C);
    const Eigen::Vector3d pivot_CA = pos_CA2.cast<double>();
    const Eigen::Matrix3d R_psi =
        Eigen::AngleAxisd(static_cast<double>(delta_psi),
                          (pos_C2.cast<double>() - pivot_CA).normalized())
            .toRotationMatrix();

    auto psi_ranges = compute_pivot_c_term_ranges(system, context, r, /*is_phi=*/false);
    rotate_ranges(proposal, psi_ranges, R_psi, pivot_CA);

    // Mark the UNION of both rotations' touched ranges exactly once (see
    // this function's header comment).
    std::vector<std::pair<int, int>> all_ranges = phi_ranges;
    all_ranges.insert(all_ranges.end(), psi_ranges.begin(), psi_ranges.end());
    mark_ranges(patch, system, std::move(all_ranges));

    patch.is_valid = true;
    patch.first_affected_residue = r;
    patch.last_affected_residue = num_residues - 1;
    // NOT rigid -- see this function's header comment.
    patch.is_rigid = false;

    if (r > 0) patch.add_distorted_bb_residue(r - 1);
    patch.add_distorted_bb_residue(r);
    if (r < num_residues - 1) patch.add_distorted_bb_residue(r + 1);
    if (r > 0) recompute_backbone_torsion(proposal, system, r - 1);
    recompute_backbone_torsion(proposal, system, r);
    if (r < num_residues - 1) recompute_backbone_torsion(proposal, system, r + 1);
    return true;
}

void MCIntegrator::apply_rama_pivot_at(Context& context, State& proposal, ProposalPatch& patch,
                                       int r) {
    const System& system = context.getSystem();
    patch.is_valid = false;
    const int num_residues = system.getNumResidues();

    // Fixed-residue check lives HERE (not just in apply_rama_pivot_move's
    // outer resample loop), mirroring apply_pivot_at's own internal check
    // -- this is the single authoritative check exercised by BOTH
    // apply_rama_pivot_move's production dispatch and
    // debug_force_rama_pivot's direct call, so they can't drift apart the
    // way a duplicated check in each caller could. C-term direction only
    // (v1 scope limit): reject if r itself or anything in the rotated
    // downstream segment [r+1, num_residues) is fixed.
    if (hasFixedResidues() &&
        (isResidueFixed(r) || segmentContainsFixed(r + 1, num_residues))) {
        ++fixed_rejected_;
        return;
    }

    const int category = resolve_rama_category(system, r);
    const RamaMixtureLibrary& lib = system.getRamaMixtureLibrary();
    if (lib.num_rows(category) <= 0) return;  // no mixture registered for this category

    // Refresh residue r's cached (phi, psi) before the row/target draw --
    // apply_rama_pivot_to_target refreshes it again internally (needed for
    // its own callers that skip this step), which is harmless (idempotent
    // recompute from unchanged coordinates), but reading it here first is
    // what apply_rotamer_at's identical staleness-guard pattern does too.
    recompute_backbone_torsion(proposal, system, r);

    // Component draw (identical pattern to apply_rotamer_at's row draw).
    const int row_idx = lib.sample_row(category, coin_flip(rng));
    const RamaComponent& comp = lib.row(category, row_idx);

    // Bivariate-normal target via the component's precomputed Cholesky
    // factor (two unit_normal_dist_ draws) -- replicates
    // rng.multivariate_normal(mean, cov) -> wrap_to_pi(...) from the
    // Python reference (wrapped_normal_mixture.py) exactly, so the density
    // used for the MH correction matches what was actually sampled.
    const float z1 = unit_normal_dist_(rng);
    const float z2 = unit_normal_dist_(rng);
    const std::array<float, 2> new_phi_psi{
        wrap_angle_to_pi(comp.mean[0] + comp.chol_l11 * z1),
        wrap_angle_to_pi(comp.mean[1] + comp.chol_l21 * z1 + comp.chol_l22 * z2)};

    apply_rama_pivot_to_target(context, proposal, patch, r, category, new_phi_psi);
}

void MCIntegrator::dispatch_pivot_move(Context& context, State& proposal, ProposalPatch& patch,
                                       bool& used_rama_pivot) {
    bool use_rama;
    if (pivot_rama_probability_ <= 0.0f) {
        use_rama = false;
    } else if (pivot_rama_probability_ >= 1.0f) {
        use_rama = true;
    } else {
        // Exactly one extra coin_flip draw, only when the probability is
        // strictly between 0 and 1 -- see set_pivot_rama_probability's
        // docs on why the p=0.0/1.0 extremes must not consume this draw
        // (bit-identical legacy behavior / checkpoint compatibility).
        use_rama = coin_flip(rng) < pivot_rama_probability_;
    }
    used_rama_pivot = use_rama;
    if (use_rama) {
        apply_rama_pivot_move(context, proposal, patch);
    } else {
        apply_pivot_move(context, proposal, patch);
    }
}

// ---------------------------------------------------------
// THE CONTINUOUS SIDECHAIN MOVE (see apply_sidechain_at). The
// rotamer-library alternative lives below (apply_rotamer_move/_at) and is
// selected via sidechain_move_mode_ (default: RotamerLibrary).
// ---------------------------------------------------------
void MCIntegrator::apply_sidechain_move(Context& context, State& proposal, ProposalPatch& patch) {
    const System& system = context.getSystem();
    patch.is_valid = false;

    constexpr int kMaxScResample = 64;
    int r = 0;
    for (int attempt = 0; attempt < kMaxScResample; ++attempt) {
        r = sc_residue_dist(rng);
        if (system.is_proline(r)) {
            ++num_sc_resample_pro_;
            continue;
        }
        if (hasFixedResidues() && isResidueFixed(r)) {
            ++fixed_rejected_;
            continue;
        }
        break;
    }
    if (system.is_proline(r)) return;
    if (hasFixedResidues() && isResidueFixed(r)) return;
    apply_sidechain_at(context, proposal, patch, r);
}

// Continuous sidechain move: an independent, zero-mean Gaussian perturbation
// per chi angle (chi1..chi4, whichever the residue has), applied as a
// per-chi cascading rotation (apply_chi_cascade) -- mirrors legacy MCPU's
// non-rotamer SIDECHAIN_NOISE fallback (MakeSidechainMove(),
// USE_ROTAMERS=0: `delta_angle[i] = GaussianNum() * SIDECHAIN_NOISE` per
// chi, independently). Each chi's proposal is symmetric (a centered
// Gaussian: +delta and -delta are equally likely), so no Metropolis-
// Hastings correction is needed -- patch.log_jacobian_weight stays at its
// default 0, exactly as before this move perturbed more than chi1.
void MCIntegrator::apply_sidechain_at(Context& context, State& proposal, ProposalPatch& patch, int r) {
    const System& system = context.getSystem();
    patch.is_valid = false;

    const int ntorsions = system.getTorsionsPerResidue()[static_cast<size_t>(r)];
    if (ntorsions <= 0) return;  // Gly/Ala: no chi angles, nothing to propose

    patch.add_distorted_sc_residue(r);

    // One sc_angle_dist_(rng) draw per chi (1-4, always in chi-index order).
    // Uses the SIDECHAIN amplitude, which defaults to the backbone one but can
    // be set independently -- legacy has two separate knobs (MC_STEP_SIZE 2 deg
    // vs SIDECHAIN_NOISE 10 deg) and a single shared amplitude cannot match
    // both. See MCIntegrator's constructor docs.
    std::array<float, 4> delta{0.f, 0.f, 0.f, 0.f};
    for (int k = 0; k < ntorsions; ++k) {
        delta[static_cast<size_t>(k)] = sc_angle_dist_(rng);
    }

    if (!apply_chi_cascade(proposal, system, r, ntorsions, delta, patch)) return;

    patch.is_valid = true;
    patch.first_affected_residue = r;
    patch.last_affected_residue = r; // Only this residue is affected!
    // Deliberately false -- see apply_rotamer_at's identical note on why a
    // multi-chi cascade must not be marked rigid (MuPotential's
    // skip_rigid_mm would otherwise silently drop real moved-vs-moved
    // energy changes).
    patch.is_rigid = false;

    // Refresh the cached chi angles for the residue just rotated — see
    // recompute_sidechain_torsion()'s docstring above.
    recompute_sidechain_torsion(proposal, system, r);
}

// ---------------------------------------------------------------
// THE ROTAMER-LIBRARY SIDECHAIN MOVE
//
// Mirrors legacy MCPU's MakeSidechainMove() (dbfold MCPU/src_mpi_umbrella/
// move.h): pick a rotamer "row" from a per-residue-type Dunbrack backbone-
// independent library (bbind02.May.lib), weighted by the row's stored
// probability, then propose a target chi vector as one Gaussian draw per
// chi around that row's mean/SD, applied as a per-chi cascading rotation
// (each chi rotated only about its own bond, proximal -> distal).
//
// UNLIKE legacy, this move is a corrected independence sampler: legacy
// applies the plain exp(-dE/T) Metropolis criterion with no correction for
// this proposal's asymmetry (a real, uncorrected detailed-balance gap in
// the reference implementation — confirmed by reading its acceptance code,
// which *does* apply such a correction for its Yang/loop-closure backbone
// move but not for this one). Here, the proposal density is a K-component
// Gaussian mixture over the residue's rotamer rows,
//   q(x) = sum_k w_k * prod_i Normal(x_i; mean_{k,i}, sigma_{k,i})
// independent of the current chi state x (RotamerLibrary::log_mixture_
// density). Since q(x'|x) = q(x') and q(x|x') = q(x) (independence
// sampler), the correct Metropolis-Hastings proposal-ratio term is
// log q(x_old) - log q(x_new), which is exactly what ProposalPatch::
// log_jacobian_weight already exists for (the same field KIC populates
// with its own Jacobian/solution-count correction, Integrator.cpp's
// apply_concerted_rotation_move) — no changes to the shared Metropolis
// acceptance code (see `run()`) are needed.
// ---------------------------------------------------------------
void MCIntegrator::apply_rotamer_move(Context& context, State& proposal, ProposalPatch& patch) {
    const System& system = context.getSystem();
    patch.is_valid = false;

    constexpr int kMaxScResample = 64;
    int r = 0;
    for (int attempt = 0; attempt < kMaxScResample; ++attempt) {
        r = sc_residue_dist(rng);
        if (system.is_proline(r)) {
            ++num_sc_resample_pro_;
            continue;
        }
        if (hasFixedResidues() && isResidueFixed(r)) {
            ++fixed_rejected_;
            continue;
        }
        break;
    }
    if (system.is_proline(r)) return;
    if (hasFixedResidues() && isResidueFixed(r)) return;
    apply_rotamer_at(context, proposal, patch, r);
}

void MCIntegrator::apply_rotamer_at(Context& context, State& proposal, ProposalPatch& patch, int r) {
    const System& system = context.getSystem();
    patch.is_valid = false;

    const int ntorsions = system.getTorsionsPerResidue()[static_cast<size_t>(r)];
    if (ntorsions <= 0) return;  // Gly/Ala: no chi angles, nothing to propose

    const int amino_idx = system.amino_index(r);
    const RotamerLibrary& lib = system.getRotamerLibrary();
    if (lib.num_rows(amino_idx) <= 0) return;  // no rotamer table for this residue type

    patch.add_distorted_sc_residue(r);

    // Refresh residue r's cached chi vector from CURRENT geometry before
    // reading it. The cache is normally kept in sync by whichever move most
    // recently targeted r directly, but a pivot/KIC move can carry r's
    // atoms along as a rigid-rotation "passenger" without recomputing its
    // chi (correctly so: chi is exactly invariant under any common rigid
    // rotation, in EXACT arithmetic) -- yet the coordinates themselves
    // still pick up fresh float32 rounding on every such rotation, so a
    // long-untouched residue's cached chi can drift from what the CURRENT
    // coordinates actually give by ~1e-5 rad. That's invisible everywhere
    // else (SideChainTripletPotential's incremental-vs-full energy checks
    // read this same cache on both sides, so a stale-but-self-consistent
    // value never shows up as a mismatch there) -- but this move is the
    // first to feed old_chi into a Metropolis-Hastings decision
    // (log_jacobian_weight below), where even a tiny discrepancy can flip
    // an accept/reject right at the threshold and desync two runs that
    // started from bit-identical coordinates (confirmed via a WE-engine
    // teardown/rebuild round-trip test). Recomputing here removes any
    // dependence on other moves' cache-freshness assumptions.
    recompute_sidechain_torsion(proposal, system, r);
    const std::array<float, 4>& old_chi =
        proposal.sidechain_torsions[static_cast<size_t>(r)].chi_angles;

    // 1. Row draw: exactly one coin_flip(rng) draw, cumulative-weight walk
    // (mirrors legacy's USE_ROT_PROB cumulative-probability sampling).
    const float u = coin_flip(rng);
    const int row_idx = lib.sample_row(amino_idx, u);
    const RotamerComponent& row = lib.row(amino_idx, row_idx);

    // 2. Per-chi target sampling: exactly one unit_normal_dist_ draw per chi
    // (1-4 draws, always in chi-index order), converted to a signed wrapped
    // delta from the CURRENT (pre-move) chi value.
    std::array<float, 4> delta{0.f, 0.f, 0.f, 0.f};
    std::array<float, 4> new_chi = old_chi;
    for (int k = 0; k < ntorsions; ++k) {
        const size_t ks = static_cast<size_t>(k);
        const float target_k = row.mean[ks] + row.sigma[ks] * unit_normal_dist_(rng);
        delta[ks] = wrap_angle_to_pi(target_k - old_chi[ks]);
        new_chi[ks] = wrap_angle_to_pi(old_chi[ks] + delta[ks]);
    }

    // 3. Metropolis-Hastings correction for this independence-sampler
    // proposal (see this function's header comment) -- computed from the
    // not-yet-mutated old_chi and the just-sampled new_chi, before any
    // coordinates are touched.
    const float log_q_old = lib.log_mixture_density(amino_idx, ntorsions, old_chi);
    const float log_q_new = lib.log_mixture_density(amino_idx, ntorsions, new_chi);
    patch.log_jacobian_weight = log_q_old - log_q_new;

    // 4. Apply the per-chi cascade, proximal -> distal (chi1, chi2, ...).
    if (!apply_chi_cascade(proposal, system, r, ntorsions, delta, patch)) return;

    // 5. Record the Patch
    patch.is_valid = true;
    patch.first_affected_residue = r;
    patch.last_affected_residue = r; // Only this residue is affected!

    // Deliberately false, NOT a missed optimization: unlike a single rigid
    // rotation, this cascading multi-chi move generally changes moved-vs-
    // moved intra-residue distances (atoms in an inner chi's shell move
    // relative to atoms only in an outer chi's shell). MuPotential's
    // skip_rigid_mm (default ON, NeighborConfig::skip_rigid_mm) would
    // silently drop that energy contribution if is_rigid were ever true here.
    patch.is_rigid = false;

    // Refresh the cached chi angles for the residue just rotated — see
    // recompute_sidechain_torsion()'s docstring above. Recomputes from the
    // now-rotated coordinates, which should equal new_chi up to
    // floating-point rounding.
    recompute_sidechain_torsion(proposal, system, r);
}

// Carries a residue's dependent atoms (sidechain, O, H) rigidly with its backbone frame
// (bisector of p1-center-p3), from old_coords to new_coords.
//
// Done in double and rounded to float once per coordinate, like CoordsSoA::rotate_atoms. In
// float32 the frame maths was biased, not just noisy: over 3M default-mix chignolin steps
// the bonds it carries in KIC's residues drifted linearly (sidechain bonds +8e-4 A on average,
// CA-CB -2.2e-4 A), while bonds in residues KIC never moves stayed under 1e-4 A.
void transfer_dependent_atoms(
    const CoordsSoA& old_coords, CoordsSoA& new_coords,
    int p1_idx, int center_idx, int p3_idx,
    int dep_start, int dep_len)
{
    if (dep_len <= 0 || dep_start < 0) return;
    const int n = old_coords.n;
    if (p1_idx < 0 || p1_idx >= n || center_idx < 0 || center_idx >= n ||
        p3_idx < 0 || p3_idx >= n || dep_start + dep_len > n) {
        return;
    }

    // One division per unit vector, and none for the third axis: nrm is
    // perpendicular to bisector, so their cross product is already a unit.
    auto unit = [](const Eigen::Vector3d& v) { return Eigen::Vector3d(v * (1.0 / v.norm())); };
    auto build_frame = [&unit](const Eigen::Vector3d& p1, const Eigen::Vector3d& center,
                               const Eigen::Vector3d& p3) {
        const Eigen::Vector3d v1 = unit(p1 - center);
        const Eigen::Vector3d v2 = unit(p3 - center);
        const Eigen::Vector3d bisector = unit(v1 + v2);
        const Eigen::Vector3d nrm = unit(v1.cross(v2));
        Eigen::Matrix3d frame;
        frame.col(0) = bisector;
        frame.col(1) = nrm.cross(bisector);
        frame.col(2) = nrm;
        return frame;
    };

    const Eigen::Vector3d center_old = old_coords.atom(center_idx).cast<double>();
    const Eigen::Vector3d center_new = new_coords.atom(center_idx).cast<double>();
    const Eigen::Matrix3d frame_old = build_frame(
        old_coords.atom(p1_idx).cast<double>(), center_old, old_coords.atom(p3_idx).cast<double>());
    const Eigen::Matrix3d frame_new = build_frame(
        new_coords.atom(p1_idx).cast<double>(), center_new, new_coords.atom(p3_idx).cast<double>());
    const Eigen::Matrix3d M = frame_new * frame_old.transpose();

    for (int i = 0; i < dep_len; ++i) {
        const int atom_idx = dep_start + i;
        const Eigen::Vector3d p = center_new + M * (old_coords.atom(atom_idx).cast<double>() - center_old);
        new_coords.set_atom(atom_idx, p.cast<float>());
    }
}

void MCIntegrator::apply_concerted_rotation_move(Context& context, State& proposal, ProposalPatch& patch) {
    const System& system = context.getSystem();
    patch.is_valid = false;
    int num_residues = system.getNumResidues();
    const auto& blocks = system.getBlockIndices();

    int r = pivot_residue_dist(rng);
    bool is_phi = (coin_flip(rng) < 0.5f);

    // Driver rotation needs an anchor residue beyond the tripeptide:
    //   phi driver: anchor at r+3 (need r+3 < num_residues)
    //   psi driver: anchor at r-1 (need r >= 2)
    if (is_phi) {
        if (r < 1 || r > num_residues - 4) return;
    } else {
        if (r < 2 || r > num_residues - 3) return;
    }

    // KIC FIX (F8): never change a proline's phi (its ring would stay closed but N would go
    // non-planar). The closure changes phi of r, r+1, r+2; the phi driver also changes phi(r+3)
    // by moving C(r+2), while the psi driver changes psi(r-1) only. Legacy loop.h:77-101 refuses
    // the same residues. Depends on the sequence alone, so detailed balance is kept.
    {
        const int pro_hi = is_phi ? r + 3 : r + 2;
        for (int k = r; k <= pro_hi; ++k) {
            if (system.is_proline(k)) {
                ++kic_proline_skipped_;
                return;
            }
        }
    }

    // KIC affects residues r, r+1, r+2.  Driver anchor extends one more.
    if (hasFixedResidues()) {
        int lo = is_phi ? r : (r - 1);
        int hi = is_phi ? (r + 4) : (r + 3);
        if (segmentContainsFixed(lo, hi)) {
            ++fixed_rejected_;
            return;
        }
    }

    // Use semantic helpers: after residue-contiguous tree order, C ≠ bb_start+2.
    int n1 = blocks[static_cast<size_t>(r)].bb_start;
    int a1 = blocks[static_cast<size_t>(r)].ca_atom();
    int c1 = blocks[static_cast<size_t>(r)].c_atom();
    int n2 = blocks[static_cast<size_t>(r + 1)].bb_start;
    int a2 = blocks[static_cast<size_t>(r + 1)].ca_atom();
    int c2 = blocks[static_cast<size_t>(r + 1)].c_atom();
    int n3 = blocks[static_cast<size_t>(r + 2)].bb_start;
    int a3 = blocks[static_cast<size_t>(r + 2)].ca_atom();
    int c3 = blocks[static_cast<size_t>(r + 2)].c_atom();

    Eigen::Vector3d r_n1 = proposal.atom_pos(n1).cast<double>();
    Eigen::Vector3d r_a1 = proposal.atom_pos(a1).cast<double>();
    Eigen::Vector3d r_a3 = proposal.atom_pos(a3).cast<double>();
    Eigen::Vector3d r_c3 = proposal.atom_pos(c3).cast<double>();

    // 3. Initialize solver using explicit, safe indices
    // KIC FIX (F3): the 6 lengths, 7 angles and 2 omegas come from the START structure
    // (System::setKicReference), not from the current coordinates. Re-measuring them made
    // every accepted closure error the next move's target, so N-CA-C random-walked.
    if (!system.hasKicReference()) {
        throw std::runtime_error(
            "KIC move: System has no start-structure closure targets; call "
            "System.set_kic_reference(start_coords) when building it "
            "(MCPUForceField.create_system does)");
    }
    const KicReference& ref = system.kicReference();
    const size_t R0 = static_cast<size_t>(r), R1 = R0 + 1, R2 = R0 + 2;
    TripeptideSolver solver;
    std::array<double, 6> b_len = {
        ref.len_ac[R0], ref.len_cn[R0], ref.len_na[R1],
        ref.len_ac[R1], ref.len_cn[R1], ref.len_na[R2]
    };
    std::array<double, 7> b_ang = {
        ref.ang_nac[R0], ref.ang_acn[R0], ref.ang_cna[R0], ref.ang_nac[R1],
        ref.ang_acn[R1], ref.ang_cna[R1], ref.ang_nac[R2]
    };
    std::array<double, 2> t_ang = { ref.omega[R0], ref.omega[R1] };

    solver.initialize(b_len, b_ang, t_ang);

    // 3b. Solve PRE-rotation to count solutions for detailed balance.
    //     The current state must itself be reachable by the solver (n_soln >= 1)
    //     for the reverse-move probability to be well-defined.  If the solver
    //     finds 0 solutions the acceptance ratio n_new/n_old is undefined, so
    //     we must reject rather than clamp to 1.
    // Closure buffers reused across steps, so a KIC step makes no heap allocation here.
    static thread_local std::vector<Solution> pre_solutions;
    static thread_local std::vector<Solution> new_solutions;
    solver.solv_3pep_poly(r_n1, r_a1, r_a3, r_c3, pre_solutions);
    // KIC FIX (F2): the solver drops closures that miss an N-CA-C target by > 1e-6 rad, in
    // this solve and the post-move one alike, so both counts below are filtered the same way.
    kic_geometry_invalid_ += solver.last_rejected();
    int n_soln_before = static_cast<int>(pre_solutions.size());
    if (n_soln_before == 0) {
        ++kic_presolve_zero_;
        return;
    }

    // KIC FIX (F5): the move is reversible only if the current window is itself one of the
    // surviving pre-move solutions (the reverse move would have to pick it). Refuse otherwise.
    {
        constexpr double kReverseTolA = 1.0e-3;
        const Eigen::Vector3d cur_c1 = proposal.atom_pos(c1).cast<double>();
        const Eigen::Vector3d cur_n2 = proposal.atom_pos(n2).cast<double>();
        const Eigen::Vector3d cur_a2 = proposal.atom_pos(a2).cast<double>();
        const Eigen::Vector3d cur_c2 = proposal.atom_pos(c2).cast<double>();
        const Eigen::Vector3d cur_n3 = proposal.atom_pos(n3).cast<double>();
        bool reversible = false;
        for (const Solution& s : pre_solutions) {
            const double dev = std::max({
                (s.r_c[0] - cur_c1).cwiseAbs().maxCoeff(),
                (s.r_n[1] - cur_n2).cwiseAbs().maxCoeff(),
                (s.r_a[1] - cur_a2).cwiseAbs().maxCoeff(),
                (s.r_c[1] - cur_c2).cwiseAbs().maxCoeff(),
                (s.r_n[2] - cur_n3).cwiseAbs().maxCoeff()});
            if (dev <= kReverseTolA) { reversible = true; break; }
        }
        if (!reversible) {
            ++kic_reverse_missing_;
            return;
        }
    }

    // 3c. Apply driver rotation to perturb one set of KIC endpoints.
    //     phi: rotate CA(r+2), C(r+2) around axis CA(r+3)→N(r+3)
    //     psi: rotate N(r),  CA(r)  around axis CA(r-1)→C(r-1)
    float dih_ch = angle_dist(rng);

    Eigen::Vector3d driver_axis_origin, driver_center;
    if (is_phi) {
        driver_axis_origin = proposal.atom_pos(
            blocks[static_cast<size_t>(r + 3)].ca_atom()).cast<double>();
        driver_center = proposal.atom_pos(
            blocks[static_cast<size_t>(r + 3)].bb_start).cast<double>();
    } else {
        driver_axis_origin = proposal.atom_pos(
            blocks[static_cast<size_t>(r - 1)].ca_atom()).cast<double>();
        driver_center = proposal.atom_pos(
            blocks[static_cast<size_t>(r - 1)].c_atom()).cast<double>();
    }

    Eigen::Vector3d driver_axis = (driver_center - driver_axis_origin).normalized();
    Eigen::AngleAxisd driver_rot(static_cast<double>(dih_ch), driver_axis);
    Eigen::Matrix3d driver_R = driver_rot.toRotationMatrix();

    auto apply_driver = [&](const Eigen::Vector3d& pt) -> Eigen::Vector3d {
        return driver_center + driver_R * (pt - driver_center);
    };

    // KIC FIX (F4): round the driver-moved anchors to float BEFORE the solve, so the solver
    // sees exactly the coordinates stored below. The next move's pre-move solve of this window
    // is then bit-identical to this move's reverse problem, and the reverse check (F5) passes.
    if (is_phi) {
        r_a3 = apply_driver(r_a3).cast<float>().cast<double>();
        r_c3 = apply_driver(r_c3).cast<float>().cast<double>();
    } else {
        r_n1 = apply_driver(r_n1).cast<float>().cast<double>();
        r_a1 = apply_driver(r_a1).cast<float>().cast<double>();
    }

    // 4. Run the KIC Solver for the POST-rotation endpoints
    solver.solv_3pep_poly(r_n1, r_a1, r_a3, r_c3, new_solutions);
    kic_geometry_invalid_ += solver.last_rejected();  // KIC FIX (F2)
    int n_new = static_cast<int>(new_solutions.size());
    if (n_new == 0) return;

    // 5. Pick one solution uniformly at random
    std::uniform_int_distribution<int> root_dist(0, n_new - 1);
    const Solution& chosen_soln = new_solutions[static_cast<size_t>(root_dist(rng))];

    // 6. Calculate Jacobian of the NEW state
    double J_new = solver.calculate_jacobian(chosen_soln);

    // 7. Calculate Jacobian of the OLD (current) state
    Solution old_state_soln;
    for (int i = 0; i < 3; ++i) {
        int res_idx = r + i;
        const auto& b = blocks[static_cast<size_t>(res_idx)];
        int n_idx  = b.bb_start;
        int ca_idx = b.ca_atom();
        int c_idx  = b.c_atom();

        old_state_soln.r_n[i] = context.state.atom_pos(n_idx).cast<double>();
        old_state_soln.r_a[i] = context.state.atom_pos(ca_idx).cast<double>();
        old_state_soln.r_c[i] = context.state.atom_pos(c_idx).cast<double>();
    }
    double J_old = solver.calculate_jacobian(old_state_soln);

    // Guard: both Jacobians must be finite and positive.  calculate_jacobian()
    // returns -1.0 for near-singular determinants; without this guard the
    // subsequent log() would produce NaN, causing a silent implicit rejection.
    // An explicit reject is cleaner and lets us track the failure rate.
    if (!std::isfinite(J_new) || !std::isfinite(J_old) ||
        J_new <= 0.0 || J_old <= 0.0) {
        ++kic_jacobian_invalid_;
        return;
    }

    // 8. Statistical weight: Jacobian ratio × solution-count ratio (detailed balance)
    patch.log_jacobian_weight = static_cast<float>(
        std::log(J_new / J_old) + std::log(static_cast<double>(n_new) / n_soln_before));

    // 9. Apply the chosen_soln coordinates to the proposal state
    //    Internal backbone atoms from the solver:
    int c1_idx = blocks[static_cast<size_t>(r)].c_atom();
    proposal.set_atom_pos(c1_idx, chosen_soln.r_c[0].cast<float>());

    int n2_idx = blocks[static_cast<size_t>(r + 1)].bb_start;
    int a2_idx = blocks[static_cast<size_t>(r + 1)].ca_atom();
    int c2_idx = blocks[static_cast<size_t>(r + 1)].c_atom();
    proposal.set_atom_pos(n2_idx, chosen_soln.r_n[1].cast<float>());
    proposal.set_atom_pos(a2_idx, chosen_soln.r_a[1].cast<float>());
    proposal.set_atom_pos(c2_idx, chosen_soln.r_c[1].cast<float>());

    int n3_idx = blocks[static_cast<size_t>(r + 2)].bb_start;
    proposal.set_atom_pos(n3_idx, chosen_soln.r_n[2].cast<float>());

    //    Driver-rotated endpoint atoms (these don't change without driver rotation):
    if (is_phi) {
        int a3_idx = blocks[static_cast<size_t>(r + 2)].ca_atom();
        int c3_idx = blocks[static_cast<size_t>(r + 2)].c_atom();
        proposal.set_atom_pos(a3_idx, r_a3.cast<float>());
        proposal.set_atom_pos(c3_idx, r_c3.cast<float>());
    } else {
        int n1_idx = blocks[static_cast<size_t>(r)].bb_start;
        int a1_idx = blocks[static_cast<size_t>(r)].ca_atom();
        proposal.set_atom_pos(n1_idx, r_n1.cast<float>());
        proposal.set_atom_pos(a1_idx, r_a1.cast<float>());
    }

    // 10. Execute rigid updates for dependent atoms (per-residue; storage may be reordered)
    {
        const int n_atoms = context.state.coords_soa.n;
        for (int i = 0; i < 3; ++i) {
            int res_idx = r + i;
            const auto& b = blocks[static_cast<size_t>(res_idx)];
            int n_idx  = b.bb_start;
            int ca_idx = b.ca_atom();
            int c_idx  = b.c_atom();

            int sc_start = b.sc_start;
            int sc_len = b.sc_count;
            if (sc_start >= 0 && sc_len <= 0) {
                int sc_end = system.sc_segment_end();
                for (int k = res_idx + 1; k < num_residues; ++k) {
                    if (blocks[static_cast<size_t>(k)].sc_start >= 0) {
                        sc_end = blocks[static_cast<size_t>(k)].sc_start;
                        break;
                    }
                }
                sc_len = sc_end - sc_start;
            }
            // Guard: clamp sc_len so sc_start + sc_len does not exceed num_atoms
            if (sc_start >= 0 && sc_start + sc_len > n_atoms) {
                sc_len = n_atoms - sc_start;
            }
            int o_start = b.o_start;
            int o_len = (o_start >= 0) ? 1 : 0;
            int h_start = b.h_start;
            int h_len = (h_start >= 0) ? 1 : 0;

            // Sidechain Transform
            if (sc_start >= 0 && sc_len > 0 &&
                n_idx >= 0 && n_idx < n_atoms &&
                ca_idx >= 0 && ca_idx < n_atoms &&
                c_idx >= 0 && c_idx < n_atoms) {
                transfer_dependent_atoms(context.state.coords_soa, proposal.coords_soa, n_idx, ca_idx, c_idx, sc_start, sc_len);
            }

            // Hydrogen Transform (skipped when amide H are virtual / total_h_atoms==0)
            // KIC FIX (F7, explicit H only): with the phi driver, H(r)'s frame C(r-1), N(r), CA(r)
            // does not move, so skip it (it used to be rewritten at rounding level, unmarked).
            if (system.getTotalHAtoms() > 0 && h_len > 0 && !(is_phi && i == 0)) {
                int prev_c_idx = blocks[static_cast<size_t>(res_idx - 1)].c_atom();
                if (prev_c_idx >= 0 && prev_c_idx < n_atoms &&
                    n_idx >= 0 && n_idx < n_atoms &&
                    ca_idx >= 0 && ca_idx < n_atoms &&
                    h_start >= 0 && h_start < n_atoms) {
                    transfer_dependent_atoms(context.state.coords_soa, proposal.coords_soa, prev_c_idx, n_idx, ca_idx, h_start, h_len);
                }
            }

            // Oxygen Transform
            // FIX: was blocks[res_idx+1] read unconditionally -- when the
            // psi-driver's own bounds allow res_idx to reach num_residues-1
            // (the last residue), res_idx+1 == num_residues indexes one
            // element past the end of blocks. Confirmed live via valgrind
            // ("Invalid read of size 4 ... 0 bytes after a block of size
            // 9,612 alloc'd"). The out-of-range read was already guarded
            // before use (next_n_idx>=0 && <n_atoms), so in practice it just
            // silently skipped the transform -- but the read itself was
            // undefined behavior. Guard the read itself instead.
            if (res_idx + 1 < num_residues) {
                int next_n_idx = blocks[static_cast<size_t>(res_idx + 1)].bb_start;
                if (o_start >= 0 && o_len > 0 &&
                    ca_idx >= 0 && ca_idx < n_atoms &&
                    c_idx >= 0 && c_idx < n_atoms &&
                    next_n_idx >= 0 && next_n_idx < n_atoms &&
                    o_start < n_atoms) {
                    transfer_dependent_atoms(context.state.coords_soa, proposal.coords_soa, ca_idx, c_idx, next_n_idx, o_start, o_len);
                }
            }
        }
    }

    // 10b. Psi driver: O(r-1) must also rotate with the driver
    if (!is_phi) {
        int o_prev = blocks[static_cast<size_t>(r - 1)].o_start;
        if (o_prev >= 0) {
            Eigen::Vector3d o_pos = context.state.atom_pos(o_prev).cast<double>();
            o_pos = apply_driver(o_pos);
            proposal.set_atom_pos(o_prev, o_pos.cast<float>());
        }
    }
    // KIC FIX (F7, explicit H only): the phi driver swings C(r+2) about N(r+3)-CA(r+3), and
    // H(r+3) (bonded to N(r+3), in the C(r+2)-N(r+3)-CA(r+3) plane) must swing with it.
    const int h_next = (is_phi && system.getTotalHAtoms() > 0)
                           ? blocks[static_cast<size_t>(r + 3)].h_start : -1;
    if (h_next >= 0) {
        Eigen::Vector3d h_pos = context.state.atom_pos(h_next).cast<double>();
        proposal.set_atom_pos(h_next, apply_driver(h_pos).cast<float>());
    }

    // 11. Record moved atoms for energy delta evaluation
    patch.is_valid = true;
    patch.first_affected_residue = is_phi ? r : (r - 1);
    patch.last_affected_residue = r + 2;

    patch.bb_atom_moved[static_cast<size_t>(c1_idx)] = 1; patch.mark_moved(c1_idx);
    patch.bb_atom_moved[static_cast<size_t>(n2_idx)] = 1; patch.mark_moved(n2_idx);
    patch.bb_atom_moved[static_cast<size_t>(a2_idx)] = 1; patch.mark_moved(a2_idx);
    patch.bb_atom_moved[static_cast<size_t>(c2_idx)] = 1; patch.mark_moved(c2_idx);
    patch.bb_atom_moved[static_cast<size_t>(n3_idx)] = 1; patch.mark_moved(n3_idx);

    // Mark driver-rotated endpoint atoms
    if (is_phi) {
        int a3_idx = blocks[static_cast<size_t>(r + 2)].ca_atom();
        int c3_idx = blocks[static_cast<size_t>(r + 2)].c_atom();
        patch.bb_atom_moved[static_cast<size_t>(a3_idx)] = 1; patch.mark_moved(a3_idx);
        patch.bb_atom_moved[static_cast<size_t>(c3_idx)] = 1; patch.mark_moved(c3_idx);
    } else {
        int n1_idx = blocks[static_cast<size_t>(r)].bb_start;
        int a1_idx = blocks[static_cast<size_t>(r)].ca_atom();
        patch.bb_atom_moved[static_cast<size_t>(n1_idx)] = 1; patch.mark_moved(n1_idx);
        patch.bb_atom_moved[static_cast<size_t>(a1_idx)] = 1; patch.mark_moved(a1_idx);
    }

    for (int res = r; res <= r + 2; ++res) {
        const auto& b = blocks[static_cast<size_t>(res)];
        if (b.sc_start >= 0) {
            int sc_len = b.sc_count;
            if (sc_len <= 0) {
                int sc_end = system.sc_segment_end();
                for (int k = res + 1; k < num_residues; ++k) {
                    if (blocks[static_cast<size_t>(k)].sc_start >= 0) {
                        sc_end = blocks[static_cast<size_t>(k)].sc_start;
                        break;
                    }
                }
                sc_len = sc_end - b.sc_start;
            }
            for (int i = b.sc_start; i < b.sc_start + sc_len; ++i) {
                patch.sc_atom_moved[static_cast<size_t>(i)] = 1;
                patch.mark_moved(i);
            }
        }
        if (b.o_start >= 0) {
            patch.o_atom_moved[static_cast<size_t>(b.o_start)] = 1;
            patch.mark_moved(b.o_start);
        }
    }
    // Mark O(r-1) for psi driver
    if (!is_phi) {
        int o_prev = blocks[static_cast<size_t>(r - 1)].o_start;
        if (o_prev >= 0) {
            patch.o_atom_moved[static_cast<size_t>(o_prev)] = 1;
            patch.mark_moved(o_prev);
        }
    }
    if (system.getTotalHAtoms() > 0) {
        // H atoms for KIC residues r+1, r+2 (and r for psi, since N(r) moved)
        int h_lo = is_phi ? (r + 1) : r;
        for (int res = h_lo; res <= r + 2; ++res) {
            const int h = blocks[static_cast<size_t>(res)].h_start;
            if (h >= 0) {
                patch.h_atom_moved[static_cast<size_t>(h)] = 1;
                patch.mark_moved(h);
            }
        }
        if (h_next >= 0) {  // KIC FIX (F7): H(r+3), moved by the phi driver above
            patch.h_atom_moved[static_cast<size_t>(h_next)] = 1;
            patch.mark_moved(h_next);
        }
    }
    patch.is_rigid = false;

    // KIC FIX (F9): residue k's cached (phi, psi, pCA, bCA) reads C(k-1), N(k-1), CA(k-1), O(k-1),
    // N(k), CA(k), C(k), N(k+1), CA(k+1), O(k+1) (recompute_backbone_torsion). Moved atoms:
    //   phi driver: C,O of r; N,CA,C,O of r+1 and r+2        -> k = r-1 .. r+3
    //   psi driver: O of r-1; N,CA,C,O of r and r+1; N,O of r+2 -> k = r-2 .. r+3
    // It used to refresh only r..r+3 (phi) and r-1..r+2 (psi): r-1 (phi) and r-2, r+3 (psi)
    // kept stale pCA/bCA, and the error was billed to a later move.
    const int bb_lo = is_phi ? r - 1 : r - 2;   // >= 0: phi needs r >= 1, psi r >= 2
    const int bb_hi = std::min(r + 3, num_residues - 1);
    for (int k = bb_lo; k <= bb_hi; ++k) patch.add_distorted_bb_residue(k);

    // Refresh the cached backbone torsions for the residues just marked
    // distorted — see recompute_backbone_torsion()'s docstring above.
    for (int k = bb_lo; k <= bb_hi; ++k) recompute_backbone_torsion(proposal, system, k);
}

void MCIntegrator::verify_physics_consistency(Context& context, int num_steps, float atol)
{
    const int N = context.getSystem().getNumResidues();
    if (N < 3) {
        throw std::invalid_argument(
            "MCIntegrator::verify_physics_consistency: needs at least 3 "
            "residues; the pivot residue range is [1, n_residues-2].");
    }
    check_move_weights_are_usable(context);
    context.require_current_atom_order();
    pivot_residue_dist = std::uniform_int_distribution<int>(1, N - 2);
    sc_residue_dist    = std::uniform_int_distribution<int>(0, N - 1);
    ensure_proposal_buffers(context);

    std::uniform_real_distribution<float> move_type_dist(0.0f, 1.0f);
    State& proposal = *proposal_;
    ProposalPatch& move_patch = patch_;

    for (int step = 0; step < num_steps; ++step) {
        proposal.copy_dynamic_from(context.getState());
        move_patch.reset_for_step();

        const float move_roll = move_type_dist(rng);
        const int move_slot = select_move_slot(move_roll);
        if (move_slot == 0) {
            bool used_rama_pivot = false;
            dispatch_pivot_move(context, proposal, move_patch, used_rama_pivot);
        } else if (move_slot == 1) {
            apply_concerted_rotation_move(context, proposal, move_patch);
        } else if (sidechain_move_mode_ == SidechainMoveMode::RotamerLibrary) {
            apply_rotamer_move(context, proposal, move_patch);
        } else {
            apply_sidechain_move(context, proposal, move_patch);
        }

        if (!move_patch.is_valid) {
            context.getWorkspace().clear();
            context.getQBiasWorkspace().clear();
            context.getHBondWorkspace().clear();
            continue;
        }

        auto& mu_ws = context.getWorkspace();
        mu_ws.use_trial_fallback = !context.trial_in_bounds(proposal, move_patch);
        if (move_slot == 0) mu_ws.move_kind = MoveKind::Pivot;
        else if (move_slot == 1) mu_ws.move_kind = MoveKind::KIC;
        else mu_ws.move_kind = MoveKind::Sidechain;

        mu_ws.neighbor_mode = resolve_neighbor_mode(
            mu_ws.move_kind, move_patch.is_rigid, context.neighborConfig(),
            static_cast<int>(move_patch.moved_indices.size()));
        if (mu_ws.neighbor_mode == NeighborMode::VerletPreferred &&
            context.neighborConfig().skin > 0.f) {
            if (context.verletContact().dirty && context.denseGridsActive()) {
                context.maybe_rebuild_verlet();
            }
            if (!context.verletContact().trial_usable(
                    move_patch.moved_indices, context.getState().coords_soa, proposal.coords_soa)) {
                mu_ws.neighbor_mode = NeighborMode::CellOnly;
            }
        }

        const auto checks = PhysicsVerifier::verify_all_potential_deltas(
            context, context.getState(), proposal, move_patch, atol);

        for (const auto& check : checks) {
            if (!check.passed) {
                std::ostringstream oss;
                oss << "Physics verification failed at step " << step
                    << " energy_group=" << check.energy_group
                    << " delta_inc=" << check.delta_incremental
                    << " delta_direct=" << check.delta_direct
                    << " (" << check.message << ")";
                throw std::runtime_error(oss.str());
            }
        }

        context.getWorkspace().clear();
        context.getQBiasWorkspace().clear();
        context.getHBondWorkspace().clear();
    }
}


void MCIntegrator::run(Context& context, int num_steps, int step_offset)
{
    const int N = context.getSystem().getNumResidues();
    // uniform_int_distribution(1, N-2) is undefined for N < 3 (a > b), and in
    // a release build it returns garbage rather than complaining.
    if (N < 3) {
        throw std::invalid_argument(
            "MCIntegrator::run: needs at least 3 residues; the pivot residue "
            "range is [1, n_residues-2].");
    }
    check_move_weights_are_usable(context);
    context.require_current_atom_order();
    pivot_residue_dist = std::uniform_int_distribution<int>(1, N - 2);
    sc_residue_dist    = std::uniform_int_distribution<int>(0, N - 1);
    ensure_proposal_buffers(context);

    std::uniform_real_distribution<float> move_type_dist(0.0f, 1.0f);
    const float beta = 1 / temperature;

    State& proposal = *proposal_;
    ProposalPatch& move_patch = patch_;
    last_accept_bits_.assign(static_cast<size_t>(num_steps), 0);
    reset_step_stats();
    step_stats_.energy_terms = context.getSystem().energyTerms();
    // Before any setup that needs undoing and before any move: a reporter
    // that rejects this run (e.g. a CSV header mismatch on resume) stops it
    // here, cleanly.
    for (auto& reporter : reporters_) {
        reporter->begin_run(context, *this);
    }
    proposal_synced_ = false; // CHANGED: sparse — resync at start of every run()
    // CHANGED: was unconditional. These are 2 steady_clock::now() calls per
    // enabled potential per move (5 potentials -> ~195 ns/move at 19.55 ns/call on this
    // machine's tsc clocksource) and they ran in EVERY production run, including
    // every published timing. MCPU_ENERGY_TIMING=0 turns them off so pyMCPU can be
    // measured against the uninstrumented legacy baseline on equal terms.
    {
        static const bool ft = [] {
            const char* e = std::getenv("MCPU_ENERGY_TIMING");
            return !(e && e[0] == '0');
        }();
        context.set_energy_delta_timing(ft);
    }
    context.reset_energy_delta_ns();

    context.neighborStats().reset();
    if (context.neighborConfig().skin > 0.f && context.denseGridsActive()) {
        context.maybe_rebuild_verlet();
    }

    // OpenMM-style: emit the initial frame at global step 0 before any moves
    // (attempt/accept counters still zero). Subsequent frames use completed
    // step counts (step_offset + local_index + 1).
    auto fire_reporters = [&](int global_step) {
        for (auto& reporter : reporters_) {
            if (auto xtc = std::dynamic_pointer_cast<XtcReporter>(reporter)) {
                xtc->report(global_step, context, *this);
            } else {
                reporter->report(global_step, context, *this);
            }
        }
    };
    if (step_offset == 0 && num_steps >= 0 && !reporters_.empty()) {
        fire_reporters(0);
    }

    for (int step = 0; step < num_steps; ++step) {
        ScopedTimer step_timer(&step_stats_.step_total_ns);
        {
            ScopedTimer copy_timer(&step_stats_.copy_dynamic_ns);
            // CHANGED: sparse — gate full O(N) sync to run-start / desync only.
            const bool need_full_sync =
                !use_sparse_proposal_ || !use_pooled_proposal_ || !proposal_synced_;
            if (need_full_sync) {
                if (use_pooled_proposal_) {
                    proposal.copy_dynamic_from(context.state);
                } else {
                    // Vanilla cost model: a whole State copy, contact list included.
                    proposal = context.state;
                }
                proposal_synced_ = true;
            }
        }
        {
            if (use_pooled_proposal_) {
                move_patch.reset_for_step();
            } else {
                // Vanilla: allocate/zero all length-N masks each step.
                move_patch = ProposalPatch(context.getSystem().getNumAtoms());
            }
        }

        bool tried_pivot = false;
        bool tried_kic   = false;
        bool tried_sc    = false;
        bool tried_rama_pivot = false;

        // One draw per step, unconditionally, whatever the weights are --
        // that invariance is what keeps the default RNG stream byte-identical.
        float move_roll = move_type_dist(rng);
        const int move_slot = select_move_slot(move_roll);

        {
            // FIXED: gen_pivot_ns / gen_kic_ns / gen_sc_ns were declared in
            // StepStats and exposed through the bindings but NEVER WRITTEN, so
            // move generation reported as zero in every build and sat inside the
            // unattributed residual. That residual is 32% of the step at
            // chignolin and 16% at sce -- the single largest unmeasured
            // component, and the one that sets the small-system floor. Three
            // ScopedTimers on a path taken once per step cost ~2 clock reads.
            if (move_slot == 0) {
                tried_pivot = true;
                bb_attempted_++;
                ScopedTimer gen_timer(&step_stats_.gen_pivot_ns);
                dispatch_pivot_move(context, proposal, move_patch, tried_rama_pivot);
                if (tried_rama_pivot) rama_pivot_attempted_++;
            } else if (move_slot == 1) {
                tried_kic = true;
                kic_attempted_++;
                ScopedTimer gen_timer(&step_stats_.gen_kic_ns);
                apply_concerted_rotation_move(context, proposal, move_patch);
            } else {
                tried_sc = true;
                sc_attempted_++;
                ScopedTimer gen_timer(&step_stats_.gen_sc_ns);
                if (sidechain_move_mode_ == SidechainMoveMode::RotamerLibrary) {
                    rotamer_attempted_++;
                    apply_rotamer_move(context, proposal, move_patch);
                } else {
                    apply_sidechain_move(context, proposal, move_patch);
                }
            }
        }

        // DEBUG: set MCPU_DEBUG_MOVES=1 to log per-step move validity (parity bisect).
        static const bool kDebugMoves = [] {
            const char* e = std::getenv("MCPU_DEBUG_MOVES");
            return e && e[0] == '1';
        }();
        if (kDebugMoves) {
            float dmax_dbg = 0.f;
            if (!move_patch.moved_indices.empty()) {
                dmax_dbg = Context::max_moved_displacement(
                    context.state, proposal, move_patch);
            }
            const char* kind = tried_pivot ? "Pivot" : (tried_kic ? "KIC" : "SC");
            std::fprintf(stderr,
                "[move] step=%d kind=%s valid=%d n_moved=%zu dmax=%.4f skin=%.2f "
                "rigid=%d roll=%.6f\n",
                step_offset + step, kind, move_patch.is_valid ? 1 : 0,
                move_patch.moved_indices.size(), dmax_dbg,
                context.neighborConfig().skin,
                move_patch.is_rigid ? 1 : 0, move_roll);
        }

        if (move_patch.is_valid) {
            const float dmax = Context::max_moved_displacement(context.state, proposal, move_patch);
            const float hard = context.neighborConfig().max_atom_displacement_hard;
            if (dmax > hard) {
                ++context.neighborStats().num_reject_hard_disp;
                move_patch.is_valid = false;
            }
        }

        MoveKind move_kind = MoveKind::Other;
        if (tried_pivot) move_kind = MoveKind::Pivot;
        else if (tried_kic) move_kind = MoveKind::KIC;
        else if (tried_sc) move_kind = MoveKind::Sidechain;

        if (move_patch.is_valid) {
            auto& mu_ws = context.mu_workspace_;
            mu_ws.move_kind = move_kind;
            mu_ws.neighbor_mode = resolve_neighbor_mode(
                move_kind, move_patch.is_rigid, context.neighborConfig(),
                static_cast<int>(move_patch.moved_indices.size()));
            mu_ws.use_trial_fallback = false;

            const bool in_box = context.trial_in_bounds(proposal, move_patch);
            if (!in_box) {
                mu_ws.use_trial_fallback = true;
                ++context.neighborStats().num_trial_fallback;
            }

            if (move_patch.is_valid &&
                mu_ws.neighbor_mode == NeighborMode::CellOnly &&
                move_kind == MoveKind::Pivot) {
                ++context.neighborStats().num_delta_cell_pivot;
            }
        }

        if (move_patch.is_valid &&
            context.mu_workspace_.neighbor_mode != NeighborMode::CellOnly &&
            context.neighborConfig().skin > 0.f &&
            context.neighborConfig().mu_verlet_enabled &&
            context.verletContact().dirty && context.denseGridsActive()) {
            context.maybe_rebuild_verlet();
        }

        if (move_patch.is_valid &&
            context.mu_workspace_.neighbor_mode != NeighborMode::CellOnly &&
            context.neighborConfig().skin > 0.f &&
            context.neighborConfig().mu_verlet_enabled) {
            if (!context.verletContact().trial_usable(
                    move_patch.moved_indices, context.state.coords_soa, proposal.coords_soa)) {
                context.mu_workspace_.neighbor_mode = NeighborMode::CellOnly;
                ++context.neighborStats().num_verlet_fallback_cell();
            }
        }

        uint8_t accepted_bit = 0;
        if (move_patch.is_valid) {
            step_stats_.n_valid_moves += 1;
            step_stats_.moved_atoms_sum += move_patch.moved_indices.size();
            // Snapshot Mu counters before ΔE for per-kind attribution.
            const std::uint64_t mu_ns0 =
                context.energy_delta_timing()
                    ? (context.energy_delta_ns_slot(1)
                           ? *context.energy_delta_ns_slot(1)
                           : 0ull)
                    : 0ull;
            const std::uint64_t mu_calls0 = context.neighborStats().mu_eval_pair_calls;
            const std::uint64_t mu_r2_0 = context.neighborStats().mu_num_pair_distance_checks;
            const std::uint64_t mu_rcut0 = context.neighborStats().mu_num_pairs_within_rcut;
            const std::uint64_t mu_nz0 = context.neighborStats().mu_eval_pair_nonzero;
            EnergyChangeResult energy_change;
            {
                ScopedTimer delta_timer(&step_stats_.delta_energy_ns);
                energy_change = context.getSystem().evaluateDeltaEnergy(
                    context, context.state, proposal, move_patch);
            }
            // Accumulate Mu-by-kind (Pivot=0, KIC=1, SC=2).
            if (static_cast<int>(move_kind) >= 0 && static_cast<int>(move_kind) <= 2) {
                auto& mk = step_stats_.mu_by_kind[static_cast<size_t>(move_kind)];
                const std::uint64_t mu_ns1 =
                    context.energy_delta_timing()
                        ? (context.energy_delta_ns_slot(1)
                               ? *context.energy_delta_ns_slot(1)
                               : 0ull)
                        : 0ull;
                mk.ns += mu_ns1 - mu_ns0;
                mk.eval_pair_calls +=
                    context.neighborStats().mu_eval_pair_calls - mu_calls0;
                mk.pair_distance_checks +=
                    context.neighborStats().mu_num_pair_distance_checks - mu_r2_0;
                mk.pairs_within_rcut +=
                    context.neighborStats().mu_num_pairs_within_rcut - mu_rcut0;
                mk.eval_pair_nonzero +=
                    context.neighborStats().mu_eval_pair_nonzero - mu_nz0;
                mk.moved_atoms_sum += move_patch.moved_indices.size();
                mk.n_steps += 1;
            }
            const float delta_E = energy_change.delta_energy;
            bool accept = false;
            if (kDebugMoves) {
                std::fprintf(stderr,
                    "[delta] step=%d kind=%s dE=%.12g reject=%d verlet_mode=%d "
                    "clash=%d\n",
                    step_offset + step,
                    tried_pivot ? "Pivot" : (tried_kic ? "KIC" : "SC"),
                    delta_E,
                    static_cast<int>(energy_change.reject_reason),
                    static_cast<int>(context.mu_workspace_.neighbor_mode),
                    energy_change.reject_reason == RejectReason::StericClash ? 1 : 0);
            }
            if (energy_change.reject_reason == RejectReason::StericClash) {
                ++steric_rejected_;
                // Preserve the historical RNG stream: the finite-sentinel
                // Metropolis path consumed one acceptance draw for clashes.
                (void)coin_flip(rng);
                if (kDebugMoves) std::fprintf(stderr, "[rng] step=%d clash_consume\n", step_offset + step);
            } else {
                float total_beta_E = (delta_E * beta) - move_patch.log_jacobian_weight;
                // FIXED: float32 Mu pair sums leave |ΔE|~1e-8 with opposite signs
                // across CellOnly vs Verlet. Strict `> 0` then spuriously consumes a
                // Metropolis coin_flip on one path only → RNG desync → cascading
                // accept-bit divergence (parity_verlet_vs_cellonly). Treat tiny
                // positive beta*ΔE as zero; threshold ≪ any physical contact.
                constexpr float kMetropolisZeroEps = 1e-5f;
                const bool need_flip = total_beta_E > kMetropolisZeroEps;
                accept = (!need_flip || coin_flip(rng) < std::exp(-total_beta_E));
                if (kDebugMoves) {
                    std::fprintf(stderr,
                        "[rng] step=%d need_flip=%d total_beta_E=%.12g accept=%d\n",
                        step_offset + step, need_flip ? 1 : 0, total_beta_E,
                        accept ? 1 : 0);
                }
            }
            if (accept) {
                accepted_bit = 1;
                step_stats_.n_accepts += 1;
                proposal.current_energy = context.state.getEnergy() + delta_E;
                {
                    ScopedTimer commit_timer(&step_stats_.commit_ns);
                    context.commit_accepted_move(proposal, move_patch, move_kind);
                }
                if (tried_pivot) {
                    bb_accepted_++;
                    if (tried_rama_pivot) rama_pivot_accepted_++;
                }
                if (tried_kic)   kic_accepted_++;
                if (tried_sc) {
                    sc_accepted_++;
                    if (sidechain_move_mode_ == SidechainMoveMode::RotamerLibrary) {
                        rotamer_accepted_++;
                    }
                }
            } else if (use_sparse_proposal_ || !use_pooled_proposal_) {
                // CHANGED: sparse — O(n_moved) restore so next step can skip full copy.
                // Vanilla (!pooled) keeps historical reject-restore dead-work.
                restore_proposal_from_accepted(proposal, context.state, move_patch);
            }
            // ADDED: retain move context for failure/crash NPZ snapshots.
            last_move_kind_str_ =
                tried_pivot ? "Pivot" : (tried_kic ? "KIC" : (tried_sc ? "Sidechain" : "Other"));
            last_is_rigid_ = move_patch.is_rigid;
            last_moved_indices_ = move_patch.moved_indices;
            last_delta_e_ = accept ? delta_E : 0.f;
            
            // DIAGNOSTIC (MCPU_CLASH_TRACE=1): catch a hard-core violation at the move that
            // INTRODUCES it, not at the next full recompute thousands of steps later. The
            // delta path rejects any proposal it sees as clashing, so a clash in an accepted
            // state means the pair was never enumerated -- and by then we no longer know
            // which move did it, whether that move was rigid, or whether both atoms were in
            // its moved set. Those three facts separate "the cell stencil never reached the
            // pair" from "it was enumerated but masked out", which is the whole question.
            //
            // Runs a full O(N^2) recompute after EVERY accepted move -- ruinously slow, so
            // single-replica reproduction only, never production. Pair with
            // MCPU_CLASH_REPORT=1, which prints the offending (i, j) from inside the
            // recompute; this follows it with the move context.
            if (accept && mcpu_clash_trace_enabled()) {
                context.calculate_total_energy(-1);
                if (context.has_steric_clash()) {
                    std::fprintf(stderr,
                        "[clash-trace] step=%d kind=%s is_rigid=%d n_moved=%zu dE=%.6f\n",
                        step_offset + step, last_move_kind_str_.c_str(),
                        last_is_rigid_ ? 1 : 0, last_moved_indices_.size(),
                        static_cast<double>(delta_E));
                    std::fprintf(stderr, "[clash-trace] moved_set=");
                    for (size_t q = 0; q < last_moved_indices_.size() && q < 80; ++q)
                        std::fprintf(stderr, "%d ", last_moved_indices_[q]);
                    std::fprintf(stderr, "%s\n", last_moved_indices_.size() > 80 ? "..." : "");
                    std::fflush(stderr);
                }
            }
        } else if ((use_sparse_proposal_ || !use_pooled_proposal_) &&
                   !move_patch.moved_indices.empty()) {
            // CHANGED: sparse — invalid move may have already mutated proposal.
            restore_proposal_from_accepted(proposal, context.state, move_patch);
            last_move_kind_str_ =
                tried_pivot ? "Pivot" : (tried_kic ? "KIC" : (tried_sc ? "Sidechain" : "Other"));
            last_is_rigid_ = move_patch.is_rigid;
            last_moved_indices_ = move_patch.moved_indices;
            last_delta_e_ = 0.f;
        }
        last_accept_bits_[static_cast<size_t>(step)] = accepted_bit;

        auto& mu_ws = context.mu_workspace_;
        mu_ws.clear();
        context.q_bias_workspace_.clear();
        context.getHBondWorkspace().clear();

        {
            // Completed MC steps so far in this trajectory (1-based within the
            // combined offset+local indexing). Report when this lands on the
            // reporter interval — not at local index 0 after the first move.
            const int completed_step = step_offset + step + 1;
            for (auto& reporter : reporters_) {
                const int interval = reporter->get_interval();
                if (interval <= 0 || completed_step % interval != 0) {
                    continue;
                }
                // pybind11 can fail to dispatch XtcReporter::report through
                // std::shared_ptr<Reporter>; call the concrete type directly.
                if (auto xtc = std::dynamic_pointer_cast<XtcReporter>(reporter)) {
                    xtc->report(completed_step, context, *this);
                } else {
                    reporter->report(completed_step, context, *this);
                }
            }
        }

        ++context.neighborStats().num_steps_executed;
        ++step_stats_.n_steps;
    }

    if (step_stats_verbose_) {
        print_step_stats_summary(step_stats_);
    }

    context.copy_energy_delta_ns_into(step_stats_.energy_delta_ns);
    step_stats_.mu_eval_pair_calls = context.neighborStats().mu_eval_pair_calls;
    step_stats_.verlet_used = context.neighborStats().num_verlet_used();
    step_stats_.verlet_fallback_cell = context.neighborStats().num_verlet_fallback_cell();
    step_stats_.verlet_rebuilds = context.neighborStats().num_verlet_rebuilds;
    step_stats_.verlet_partial_rebuilds =
        context.neighborStats().num_verlet_partial_rebuilds;
    step_stats_.verlet_partial_affected_sum =
        context.neighborStats().num_verlet_partial_affected_sum;
    {
        const auto& ns = context.neighborStats();
        auto& b = step_stats_.pivot_mu_breakdown;
        b.cell_pairs = ns.pivot_mu_cell_pairs;
        b.cell_pairs_empty = ns.pivot_mu_cell_pairs_empty;
        b.n_groups = ns.pivot_mu_n_groups;
        b.group_atoms = ns.pivot_mu_group_atoms;
        b.cell_pair_evals = ns.pivot_mu_cell_pair_evals;

        auto& cpb = step_stats_.cell_pair_breakdown;
        cpb.group_build_ns = ns.cp_group_build_ns;
        cpb.new_walk_ns = ns.cp_new_walk_ns;
        cpb.new_r2_ns = ns.cp_new_r2_ns;
        cpb.new_eval_ns = ns.cp_new_eval_ns;
        cpb.old_walk_ns = ns.cp_old_walk_ns;
        cpb.old_r2_ns = ns.cp_old_r2_ns;
        cpb.old_eval_ns = ns.cp_old_eval_ns;
        cpb.clash_aborts = ns.cp_clash_aborts;
        cpb.full_evals = ns.cp_full_evals;
        cpb.new_r2_checks = ns.cp_new_r2_checks;
        cpb.old_r2_checks = ns.cp_old_r2_checks;
        cpb.new_eval_calls = ns.cp_new_eval_calls;
        cpb.old_eval_calls = ns.cp_old_eval_calls;
        cpb.n_steps = static_cast<std::size_t>(ns.cp_n_steps);
        cpb.new_n_groups = ns.cp_new_n_groups;
        cpb.new_group_atoms = ns.cp_new_group_atoms;
        cpb.cell_pairs = ns.pivot_mu_cell_pairs;
        cpb.cell_pairs_empty = ns.pivot_mu_cell_pairs_empty;
    }
    context.set_energy_delta_timing(false);

    // Occupied-stencil maintenance diagnostic (opt-in).
    // CHANGED: gate behind MCPU_VERBOSE — was unconditional every Integrator::run.
    {
        static const bool kVerbose = [] {
            const char* e = std::getenv("MCPU_VERBOSE");
            return e && e[0] && e[0] != '0';
        }();
        if (kVerbose) {
            const auto& g = context.neighbors().muGrid().grid();
            if (g.occupied_stencil_mode() != OccupiedStencilMode::Off) {
                const auto n = step_stats_.n_steps > 0 ? step_stats_.n_steps : 1;
                std::fprintf(
                    stderr,
                    "occ_stencil mode=%d maint_ns/step=%.0f maint_calls/step=%.2f "
                    "commit_ns/step=%.0f mu_energy_ns/step=%.0f "
                    "delta_energy_ns/step=%.0f step_total_ns/step=%.0f\n",
                    static_cast<int>(g.occupied_stencil_mode()),
                    static_cast<double>(g.occupied_maint_ns()) /
                        static_cast<double>(n),
                    static_cast<double>(g.occupied_maint_calls()) /
                        static_cast<double>(n),
                    static_cast<double>(step_stats_.commit_ns) /
                        static_cast<double>(n),
                    static_cast<double>(step_stats_.energy_delta_ns[1]) /
                        static_cast<double>(n),
                    static_cast<double>(step_stats_.delta_energy_ns) /
                        static_cast<double>(n),
                    static_cast<double>(step_stats_.step_total_ns) /
                        static_cast<double>(n));
            }
        }
    }
}

bool MCIntegrator::debug_force_pivot(Context& context, int residue, bool is_phi) {
    const System& system = context.getSystem();
    if (residue < 1 || residue > system.getNumResidues() - 2) return false;
    if (is_phi && system.is_proline(residue)) {
        ++num_pivot_resample_pro_phi_;
        return false;
    }
    if (hasFixedResidues() && isResidueFixed(residue)) {
        ++fixed_rejected_;
        return false;
    }
    ensure_proposal_buffers(context);
    State& proposal = *proposal_;
    ProposalPatch& move_patch = patch_;
    proposal.copy_dynamic_from(context.getState());
    move_patch.reset_for_step();
    apply_pivot_at(context, proposal, move_patch, residue, is_phi);
    return record_forced_proposal(context, "Pivot");
}

bool MCIntegrator::record_forced_proposal(Context& context, const char* kind) {
    if (!patch_.is_valid) return false;
    // The energy change is computed the way run() computes it
    // (System::evaluateDeltaEnergy against the just-built proposal), but
    // there is no accept/reject and no commit: the forced move is only
    // inspected.
    const EnergyChangeResult energy_change =
        context.getSystem().evaluateDeltaEnergy(context, context.getState(), *proposal_, patch_);
    // The energy terms queue the changes this move would commit (contact
    // flips, native-pair flips) in the workspaces. run() clears them after
    // every step; a forced move is never committed, so clear them here, or
    // the next accepted step of run() would commit them too.
    context.getWorkspace().clear();
    context.getQBiasWorkspace().clear();
    context.getHBondWorkspace().clear();
    last_delta_e_ = energy_change.delta_energy;
    last_log_jacobian_weight_ = patch_.log_jacobian_weight;
    last_move_kind_str_ = kind;
    last_is_rigid_ = patch_.is_rigid;
    last_moved_indices_ = patch_.moved_indices;
    return true;
}

bool MCIntegrator::debug_force_sc(Context& context, int residue) {
    const System& system = context.getSystem();
    if (residue < 0 || residue >= system.getNumResidues()) return false;
    if (system.is_proline(residue)) {
        ++num_sc_resample_pro_;
        return false;
    }
    if (hasFixedResidues() && isResidueFixed(residue)) {
        ++fixed_rejected_;
        return false;
    }
    ensure_proposal_buffers(context);
    State& proposal = *proposal_;
    ProposalPatch& move_patch = patch_;
    proposal.copy_dynamic_from(context.getState());
    move_patch.reset_for_step();
    apply_sidechain_at(context, proposal, move_patch, residue);
    return record_forced_proposal(context, "Sidechain");
}

bool MCIntegrator::debug_force_rotamer(Context& context, int residue) {
    const System& system = context.getSystem();
    if (residue < 0 || residue >= system.getNumResidues()) return false;
    if (system.is_proline(residue)) {
        ++num_sc_resample_pro_;
        return false;
    }
    if (hasFixedResidues() && isResidueFixed(residue)) {
        ++fixed_rejected_;
        return false;
    }
    ensure_proposal_buffers(context);
    State& proposal = *proposal_;
    ProposalPatch& move_patch = patch_;
    proposal.copy_dynamic_from(context.getState());
    move_patch.reset_for_step();
    apply_rotamer_at(context, proposal, move_patch, residue);
    return record_forced_proposal(context, "Sidechain");
}

bool MCIntegrator::debug_force_rama_pivot(Context& context, int residue) {
    const System& system = context.getSystem();
    // Same residue-range contract as debug_force_pivot (C-term direction
    // needs at least one downstream residue; apply_rama_pivot_at also
    // reads residue-1/residue+1 for torsion-cache refresh, matching
    // apply_pivot_at's [1, N-2] contract).
    if (residue < 1 || residue > system.getNumResidues() - 2) return false;
    if (system.is_proline(residue)) {
        ++num_pivot_resample_pro_phi_;
        return false;
    }
    ensure_proposal_buffers(context);
    State& proposal = *proposal_;
    ProposalPatch& move_patch = patch_;
    proposal.copy_dynamic_from(context.getState());
    move_patch.reset_for_step();
    apply_rama_pivot_at(context, proposal, move_patch, residue);
    return record_forced_proposal(context, "Pivot");
}

bool MCIntegrator::debug_force_rama_pivot_to(Context& context, int residue, float phi, float psi) {
    const System& system = context.getSystem();
    if (residue < 1 || residue > system.getNumResidues() - 2) return false;
    if (hasFixedResidues() &&
        (isResidueFixed(residue) || segmentContainsFixed(residue + 1, system.getNumResidues()))) {
        ++fixed_rejected_;
        return false;
    }
    const int category = resolve_rama_category(system, residue);
    if (system.getRamaMixtureLibrary().num_rows(category) <= 0) return false;

    ensure_proposal_buffers(context);
    State& proposal = *proposal_;
    ProposalPatch& move_patch = patch_;
    proposal.copy_dynamic_from(context.getState());
    move_patch.reset_for_step();
    const std::array<float, 2> target{wrap_angle_to_pi(phi), wrap_angle_to_pi(psi)};
    apply_rama_pivot_to_target(context, proposal, move_patch, residue, category, target);
    return record_forced_proposal(context, "Pivot");
}
