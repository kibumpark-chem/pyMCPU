#pragma once
#include <random>
#include <vector>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <string>
#include <array>
#include <utility>
#include "ProposalPatch.h"
#include "pymcpu/State.h"
#include "pymcpu/reporters/Reporter.h"

#ifndef MCPU_USE_POOLED_PROPOSAL
#define MCPU_USE_POOLED_PROPOSAL 1
#endif

namespace mcpu {
    class System;
    class Context;

    inline constexpr float kB = 0.001987f;

/// Which sidechain-proposal algorithm the "Sidechain" move slot uses.
/// Continuous (default) preserves today's behavior exactly; RotamerLibrary
/// selects the discrete Dunbrack-rotamer-table-based proposal (see
/// MCIntegrator::apply_rotamer_move). Both live inside the existing
/// pivot/KIC/sidechain move-mix -- this does not add a 4th move kind.
enum class SidechainMoveMode : std::uint8_t {
    Continuous = 0,
    RotamerLibrary = 1,
};

/// Snapshot of proposal-lifecycle knobs (bench / smoke logging).
struct ProposalLifecycleInfo {
    bool pooled_proposal_compiled_in = (MCPU_USE_POOLED_PROPOSAL != 0);
    bool use_pooled_proposal = true;
    bool proposal_dynamic_only = true;
    bool reject_restore_enabled = false;
};

/// Per-move-kind Mu ΔE accumulators (Pivot=0, KIC=1, Sidechain=2).
struct MuKindStats {
    std::uint64_t ns = 0;
    std::uint64_t eval_pair_calls = 0;
    std::uint64_t pair_distance_checks = 0;
    std::uint64_t pairs_within_rcut = 0;
    std::uint64_t eval_pair_nonzero = 0;
    std::size_t n_steps = 0;
    std::size_t moved_atoms_sum = 0;
};

/// Pivot denselist Mu phase split (diagnostic; MCPU_PIVOT_MU_BREAKDOWN=1).
struct PivotMuBreakdown {
    std::uint64_t cell_walk_ns = 0;
    std::uint64_t r2_filter_ns = 0;
    std::uint64_t eval_pair_ns = 0;
    std::uint64_t overhead_ns = 0;
    std::uint64_t candidates = 0;
    std::uint64_t in_cutoff = 0;
    std::size_t n_pivot_steps = 0;
    std::uint64_t walk_empty_cells = 0;
    std::uint64_t walk_nonempty_cells = 0;
    std::uint64_t walk_atom_visits = 0;
    std::uint64_t walk_probe_ns = 0;
    std::uint64_t walk_oob_cells = 0;
    /// Cell-pair inversion counters (copied from NeighborStats).
    std::uint64_t cell_pairs = 0;
    std::uint64_t cell_pairs_empty = 0;
    std::uint64_t n_groups = 0;
    std::uint64_t group_atoms = 0;
    std::uint64_t cell_pair_evals = 0;
};

/// Cell-pair denselist phase split (MCPU_CELL_PAIR_BREAKDOWN=1).
struct CellPairBreakdown {
    std::uint64_t group_build_ns = 0;
    std::uint64_t new_walk_ns = 0;
    std::uint64_t new_r2_ns = 0;
    std::uint64_t new_eval_ns = 0;
    std::uint64_t old_walk_ns = 0;
    std::uint64_t old_r2_ns = 0;
    std::uint64_t old_eval_ns = 0;
    std::uint64_t mmguard_ns = 0;
    std::uint64_t movedbits_ns = 0;
    std::uint64_t skipmask_ns = 0;
    std::uint64_t clash_aborts = 0;
    std::uint64_t full_evals = 0;
    std::uint64_t new_r2_checks = 0;
    std::uint64_t old_r2_checks = 0;
    std::uint64_t new_eval_calls = 0;
    std::uint64_t old_eval_calls = 0;
    std::size_t n_steps = 0;
    std::uint64_t new_n_groups = 0;
    std::uint64_t new_group_atoms = 0;
    std::uint64_t cell_pairs = 0;
    std::uint64_t cell_pairs_empty = 0;
};

/// Per-step hot-path accumulators (Integrator::run instrumentation).
struct StepStats {
    std::uint64_t copy_dynamic_ns = 0;
    std::uint64_t commit_ns = 0;
    std::uint64_t delta_energy_ns = 0;
    std::uint64_t step_total_ns = 0;
    /// DIAG: per-move-kind proposal-generation and whole-step time. There was
    /// no timer around move generation at all, so its cost sat in the
    /// unattributed residual (28% of wall at chignolin, 12% at actin).
    std::uint64_t gen_pivot_ns = 0, gen_kic_ns = 0, gen_sc_ns = 0;
    /// DIAG: move economics binned by [kind][secondary structure][moved-size].
    /// kind 0=pivot 1=KIC 2=sidechain; ss 0=H 1=E 2=C (at the first moved
    /// residue); size bucket 0..5 = moved atoms <16,<64,<256,<512,<1024,>=1024.
    /// Answers: does acceptance depend on where in the chain / what secondary
    /// structure a move lands, and what does each bin cost? Move selection is
    /// currently UNIFORM over residues, so any structure here is unexploited.
    static constexpr int kBinKinds = 3, kBinSS = 3, kBinSize = 6;
    static constexpr int kNBins = kBinKinds * kBinSS * kBinSize;
    std::uint64_t bin_attempts[kNBins] = {};
    std::uint64_t bin_accepts[kNBins]  = {};
    std::uint64_t bin_ns[kNBins]       = {};
    std::uint64_t tot_pivot_ns = 0, tot_kic_ns = 0, tot_sc_ns = 0;
    /// Per energy-group ΔE time (index = Potential::getEnergyGroup(); 0 unused).
    std::uint64_t energy_delta_ns[8] = {};
    /// (group, name) of the System's energy terms, copied at the start of
    /// run() so step_stats() can label energy_delta_ns without a System.
    std::vector<std::pair<int, std::string>> energy_terms;
    std::size_t moved_atoms_sum = 0;
    std::size_t n_steps = 0;
    std::size_t n_valid_moves = 0;
    std::size_t n_accepts = 0;
    std::uint64_t mu_eval_pair_calls = 0;
    /// Index: 0=Pivot, 1=KIC, 2=Sidechain.
    MuKindStats mu_by_kind[3] = {};
    /// Neighbor Verlet amortization (copied from NeighborStats at end of run).
    std::uint64_t verlet_used = 0;
    std::uint64_t verlet_fallback_cell = 0;
    std::uint64_t verlet_rebuilds = 0;           ///< full CSR rebuilds
    std::uint64_t verlet_partial_rebuilds = 0;   ///< Strategy B partial accepts
    std::uint64_t verlet_partial_affected_sum = 0;
    /// Copied from NeighborStats when pivot Mu breakdown diagnostic is active.
    PivotMuBreakdown pivot_mu_breakdown;
    CellPairBreakdown cell_pair_breakdown;
};

// Monte Carlo integrator using pivot + sidechain compound moves.
class MCIntegrator {
public:
    // @param temperature   Simulation temperature in Kelvin
    // @param step_size_rad Gaussian std-dev for BACKBONE torsion perturbation
    //                      (radians) -- used by the pivot move AND by the KIC
    //                      driver angle, matching legacy, whose single
    //                      MC_STEP_SIZE covers every backbone move
    //                      (MakeMove(STEP_SIZE, ...) in move.h).
    // @param sidechain_step_size_rad
    //                      Gaussian std-dev for the CONTINUOUS sidechain chi
    //                      perturbation (radians). Negative (the default) means
    //                      "same as step_size_rad", reproducing exactly the
    //                      single-amplitude behavior this class had before this
    //                      knob existed. Set it separately to match legacy,
    //                      which has two INDEPENDENT amplitudes: MC_STEP_SIZE
    //                      for the backbone (2 deg in the reference configs)
    //                      and SIDECHAIN_NOISE for chi (10 deg) -- 5x larger,
    //                      because rotating a chi displaces only a few
    //                      sidechain atoms whereas a backbone torsion swings a
    //                      whole chain segment. Without this knob a benchmark
    //                      against legacy could match one amplitude or the
    //                      other but never both.
    //                      Ignored by the rotamer-library sidechain mode, which
    //                      takes its per-chi widths from the library rows (see
    //                      apply_rotamer_at).
    explicit MCIntegrator(float temperature,
                          float step_size_rad = 0.02f,
                          float sidechain_step_size_rad = -1.0f);

    /// Effective continuous-sidechain chi amplitude in radians (never negative:
    /// resolves the "same as backbone" sentinel to the actual value in use).
    [[nodiscard]] float sidechain_step_size_rad() const noexcept {
        return sidechain_step_size_rad_;
    }

    /// Backbone amplitude in radians (pivot + KIC driver).
    [[nodiscard]] float backbone_step_size_rad() const noexcept { return step_size_rad; }

    /// Re-set the continuous-sidechain chi amplitude. Negative restores
    /// "same as backbone". Resets sc_angle_dist_'s cached spare for the same
    /// reason set_seed does (see get_rng_state).
    void set_sidechain_step_size_rad(float sigma_rad) {
        sidechain_step_size_rad_ = (sigma_rad < 0.0f) ? step_size_rad : sigma_rad;
        sc_angle_dist_.param(
            std::normal_distribution<float>::param_type(0.0f, sidechain_step_size_rad_));
        sc_angle_dist_.reset();
    }

    void run(Context& context, int num_steps, int step_offset = 0);

    void verify_physics_consistency(Context& context, int num_steps, float atol = 1e-3f);


    long long get_bb_attempted() const { return bb_attempted_; }
    long long get_bb_accepted()  const { return bb_accepted_; }

    long long get_sc_attempted() const { return sc_attempted_; }
    long long get_sc_accepted()  const { return sc_accepted_; }

    /// Rotamer-library submode counters -- additive to (not a replacement
    /// for) get_sc_attempted()/get_sc_accepted(), which already count every
    /// Sidechain-slot attempt regardless of submode.
    long long get_rotamer_attempted() const noexcept { return rotamer_attempted_; }
    long long get_rotamer_accepted()  const noexcept { return rotamer_accepted_; }

    long long get_kic_attempted() const { return kic_attempted_; }
    long long get_kic_accepted()  const { return kic_accepted_; }

    /// Knowledge-based backbone (rama-mixture) pivot submode counters --
    /// additive to (not a replacement for) get_bb_attempted()/
    /// get_bb_accepted(), which already count every Pivot-slot attempt
    /// regardless of submode.
    long long get_rama_pivot_attempted() const noexcept { return rama_pivot_attempted_; }
    long long get_rama_pivot_accepted()  const noexcept { return rama_pivot_accepted_; }

    /// Accept/attempt counts for one move kind; see move_counts().
    struct MoveCount {
        const char* name;
        long long accepted;
        long long attempted;
        /// The kind can be proposed with the current slot weights and
        /// sidechain mode, or has been proposed already.
        bool in_use;
    };

    /// Counts per move kind, each move counted once, in the fixed order
    /// pivot, rama_pivot, kic, sidechain, rotamer. "pivot" and "sidechain"
    /// are the continuous moves; the slot getters above are sums of these
    /// (bb = pivot + rama_pivot, sc = sidechain + rotamer).
    ///
    /// in_use deliberately ignores the rama-pivot probability: a pivot slot
    /// in use reports both pivot kinds, so replicas whose schedules give
    /// different probabilities still report the same kinds.
    [[nodiscard]] std::array<MoveCount, 5> move_counts() const noexcept {
        const bool pivot_slot = move_w_pivot_ > 0.0f;
        const bool sc_slot = move_w_sc_ > 0.0f;
        const bool rotamer_mode = sidechain_move_mode_ == SidechainMoveMode::RotamerLibrary;
        const long long pivot_att = bb_attempted_ - rama_pivot_attempted_;
        const long long sc_att = sc_attempted_ - rotamer_attempted_;
        return {{
            {"pivot", bb_accepted_ - rama_pivot_accepted_, pivot_att,
             pivot_slot || pivot_att > 0},
            {"rama_pivot", rama_pivot_accepted_, rama_pivot_attempted_,
             pivot_slot || rama_pivot_attempted_ > 0},
            {"kic", kic_accepted_, kic_attempted_,
             move_w_kic_ > 0.0f || kic_attempted_ > 0},
            {"sidechain", sc_accepted_ - rotamer_accepted_, sc_att,
             (sc_slot && !rotamer_mode) || sc_att > 0},
            {"rotamer", rotamer_accepted_, rotamer_attempted_,
             (sc_slot && rotamer_mode) || rotamer_attempted_ > 0},
        }};
    }

    long long num_pivot_resample_pro_phi() const noexcept { return num_pivot_resample_pro_phi_; }
    long long num_sc_resample_pro() const noexcept { return num_sc_resample_pro_; }
    long long get_kic_presolve_zero() const noexcept { return kic_presolve_zero_; }
    long long get_kic_jacobian_invalid() const noexcept { return kic_jacobian_invalid_; }
    /// KIC FIX: closures dropped by the solver's 1e-6 rad N-CA-C check (pre- and post-move solves).
    long long get_kic_geometry_invalid() const noexcept { return kic_geometry_invalid_; }
    /// KIC FIX: moves refused because the current window is not among its own pre-move solutions.
    long long get_kic_reverse_missing() const noexcept { return kic_reverse_missing_; }
    /// KIC FIX: moves skipped because they would change a proline's phi.
    long long get_kic_proline_skipped() const noexcept { return kic_proline_skipped_; }
    long long get_steric_rejected() const noexcept { return steric_rejected_; }

    /// Test helpers: force a pivot/SC choice. Returns whether a move was proposed.
    /// Proline φ / proline SC increments the resample counters and returns false.
    bool debug_force_pivot(Context& context, int residue, bool is_phi);
    bool debug_force_sc(Context& context, int residue);
    /// Forces the rotamer-library sidechain move at `residue` regardless of
    /// the current sidechain_move_mode(). Proline / no-chi residues increment
    /// the resample counter and return false, matching debug_force_sc.
    bool debug_force_rotamer(Context& context, int residue);
    /// Forces the knowledge-based backbone pivot move at `residue`
    /// regardless of pivot_rama_probability(). Proline / no-mixture-
    /// registered residues increment the resample counter and return
    /// false, matching debug_force_pivot's proline-skip contract.
    bool debug_force_rama_pivot(Context& context, int residue);
    /// Test-only: forces the knowledge-based backbone pivot move at
    /// `residue` to an EXPLICIT (phi, psi) target (radians) instead of
    /// drawing one from the mixture -- for direct numerical
    /// detailed-balance checks that need to force the engine through a
    /// hand-picked, independently-computable state transition. Same
    /// fixed-residue/no-mixture-registered rejection contract as
    /// debug_force_rama_pivot (proline is NOT specially handled here --
    /// callers choose their own residue).
    bool debug_force_rama_pivot_to(Context& context, int residue, float phi, float psi);

    /// Selects which algorithm the "Sidechain" move slot uses:
    /// "rotamer_library" (the default) or "continuous".
    void set_sidechain_move_mode(const std::string& mode);
    [[nodiscard]] std::string sidechain_move_mode() const;

    /// Fraction of Pivot-slot attempts that use the knowledge-based
    /// (phi,psi)-resampling proposal (see apply_rama_pivot_at) instead of
    /// the continuous single-dihedral pivot. Default 0.0, i.e. OPT-IN.
    ///
    /// Deliberately opt-in rather than following sidechain_move_mode_'s
    /// precedent of defaulting to the newer proposal. Measured acceptance
    /// of this move is 0.003-0.013 on folded and expanded states alike,
    /// because apply_rama_pivot_at rotates the entire C-terminal segment
    /// and an independence draw from the (phi,psi) mixture lands far
    /// enough from the current torsions that the resulting lever-arm
    /// displacement is rejected by NeighborConfig::
    /// max_atom_displacement_hard ~79-91% of the time. A nonzero default
    /// therefore spends p * 0.25 of every run's step budget at a few-per-
    /// thousand accept rate, silently. Enable per-config once the move
    /// rotates the shorter segment (as apply_pivot_at does) or is paired
    /// with a closure.
    ///
    /// p=0.0 recovers the exact continuous-only behavior, including
    /// RNG-draw count (no extra coin_flip is consumed at that extreme --
    /// see dispatch_pivot_move); p=1.0 always uses the knowledge-based
    /// move.
    void set_pivot_rama_probability(float p) {
        if (!(p >= 0.0f && p <= 1.0f)) {
            throw std::invalid_argument("pivot_rama_probability must be in [0, 1]");
        }
        pivot_rama_probability_ = p;
    }
    [[nodiscard]] float pivot_rama_probability() const noexcept { return pivot_rama_probability_; }

    /// Relative probabilities of the three move slots (Pivot, KIC, Sidechain).
    ///
    /// Normalized internally, so (1,1,0) and (0.5,0.5,0) mean the same thing.
    /// All three must be non-negative with a positive, finite sum. The default
    /// (0.25, 0.25, 0.50) is the mix this integrator has always used, and a
    /// caller who never touches this gets a bit-identical RNG stream.
    ///
    /// RNG contract: exactly ONE move_type_dist(rng) draw is consumed per step
    /// regardless of the weights. The draw was already unconditional, and
    /// keeping it that way is what preserves the default stream. Unlike
    /// set_pivot_rama_probability, no extreme needs a special case here: a zero
    /// weight makes its comparison unreachable by construction, because a roll
    /// drawn from [0,1) is never < 0 and always < 1.
    ///
    /// Setting sidechain to 0 is REQUIRED for a force field whose residues have
    /// no chi angles -- a backbone-only one, say. Otherwise every sidechain
    /// proposal is a guaranteed no-op (apply_sidechain_at / apply_rotamer_at
    /// return early on ntorsions <= 0, leaving patch.is_valid false), so that
    /// share of the step budget is silently discarded. run() rejects that
    /// combination rather than letting it cost half a run.
    void set_move_weights(float pivot, float kic, float sidechain);

    /// (pivot, kic, sidechain), normalized to sum 1. Default (0.25,0.25,0.50).
    [[nodiscard]] std::array<float, 3> move_weights() const noexcept {
        return {move_w_pivot_, move_w_kic_, move_w_sc_};
    }

    /// Sets pivot_rama_probability() from a piecewise-linear schedule in
    /// this Integrator's own (fixed, construction-time) `temperature`
    /// (pyMCPU's reduced-temperature units): p=p_min for temperature <=
    /// t_low, p=p_max for temperature >= t_high, linear in between.
    /// Evaluated once, immediately -- temperature never changes on a live
    /// Integrator (no set_temperature exists; replica exchange swaps
    /// coordinates between fixed-temperature replicas, never a replica's
    /// own temperature), so no per-step recomputation is needed.
    void set_pivot_rama_schedule(float t_low, float t_high, float p_min, float p_max) {
        if (!(t_low < t_high)) {
            throw std::invalid_argument("set_pivot_rama_schedule: t_low must be < t_high");
        }
        if (!(p_min >= 0.0f && p_min <= 1.0f && p_max >= 0.0f && p_max <= 1.0f)) {
            throw std::invalid_argument("set_pivot_rama_schedule: p_min/p_max must be in [0, 1]");
        }
        float p;
        if (temperature <= t_low) {
            p = p_min;
        } else if (temperature >= t_high) {
            p = p_max;
        } else {
            const float frac = (temperature - t_low) / (t_high - t_low);
            p = p_min + frac * (p_max - p_min);
        }
        pivot_rama_probability_ = p;
    }

    void add_reporter(std::shared_ptr<Reporter> reporter) {
        reporters_.push_back(reporter);
    }

    void clear_reporters() { reporters_.clear(); }

    [[nodiscard]] std::size_t num_reporters() const noexcept { return reporters_.size(); }

    /// Re-seed the MC RNG (reproducible sweeps / benches). Does not change physics.
    /// Also clears angle_dist's/unit_normal_dist_'s internal caches (see
    /// get_rng_state) so the first post-reseed draw can't be contaminated by
    /// a spare left over from a prior seed.
    void set_seed(unsigned int seed) {
        rng.seed(seed);
        angle_dist.reset();
        sc_angle_dist_.reset();
        unit_normal_dist_.reset();
    }

    /// Serialize ``std::mt19937`` state for checkpoint / resume.
    ///
    /// NOTE: ``std::normal_distribution`` (used by ``angle_dist``) commonly caches a
    /// second "spare" Gaussian internally (e.g. libstdc++'s Marsaglia-polar
    /// implementation returns two values per two ``rng`` draws, keeping the second
    /// for the following call). That cache is NOT part of ``rng``'s serialized
    /// state, so it can't be captured/restored -- checkpointing only ``rng`` would
    /// silently lose a pending spare on odd-parity call counts and desync the
    /// restored stream from the original one. To keep "checkpoint" well-defined,
    /// this call discards any pending spare via ``angle_dist.reset()`` before
    /// returning, so the boundary is always cache-free on both ends.
    [[nodiscard]] std::string get_rng_state();

    /// Restore ``std::mt19937`` state previously returned by ``get_rng_state``.
    /// Also resets ``angle_dist`` (see get_rng_state) so the restored engine starts
    /// from the same cache-free boundary as the one that produced the checkpoint.
    void set_rng_state(const std::string& state);

    /// Accept/reject bit for step i of the last ``run`` (1=accept, 0=reject/invalid).
    /// Size == num_steps of last run. For determinism regression tests.
    const std::vector<uint8_t>& last_accept_bits() const noexcept { return last_accept_bits_; }

    /// Toggle pooled DynamicOnly proposal path vs vanilla Full-copy cost model.
    /// No-op (stays false) when compiled with ``MCPU_USE_POOLED_PROPOSAL=0``.
    void set_use_pooled_proposal(bool on);

    bool use_pooled_proposal() const noexcept { return use_pooled_proposal_; }
    bool reject_restore_enabled() const noexcept { return !use_pooled_proposal_; }
    bool proposal_is_dynamic_only() const noexcept { return use_pooled_proposal_; }
    static bool pooled_proposal_compiled_in() noexcept {
        return MCPU_USE_POOLED_PROPOSAL != 0;
    }

    ProposalLifecycleInfo proposal_lifecycle_info() const noexcept {
        ProposalLifecycleInfo info;
        info.pooled_proposal_compiled_in = pooled_proposal_compiled_in();
        info.use_pooled_proposal = use_pooled_proposal_;
        info.proposal_dynamic_only = use_pooled_proposal_;
        info.reject_restore_enabled = !use_pooled_proposal_;
        return info;
    }

    /// Cumulative timing from the most recent ``run`` (reset at each ``run`` start).
    [[nodiscard]] const StepStats& step_stats() const noexcept { return step_stats_; }

    void reset_step_stats() noexcept { step_stats_ = {}; }

    /// When true, ``run`` prints a one-line ``step_stats`` summary to stderr at exit.
    void set_step_stats_verbose(bool on) noexcept { step_stats_verbose_ = on; }

    [[nodiscard]] bool step_stats_verbose() const noexcept { return step_stats_verbose_; }

    /// Skip per-step ``copy_dynamic_from``; sync once then O(n_moved) restore on reject.
    void set_use_sparse_proposal(bool on) noexcept {
        use_sparse_proposal_ = on;
        proposal_synced_ = false; // CHANGED: sparse — force resync after toggle
    }

    [[nodiscard]] bool use_sparse_proposal() const noexcept { return use_sparse_proposal_; }

    /// Last proposed move context (updated each MC step in ``run``).
    [[nodiscard]] const std::string& last_move_kind() const noexcept {
        return last_move_kind_str_;
    }
    [[nodiscard]] bool last_move_is_rigid() const noexcept { return last_is_rigid_; }
    [[nodiscard]] const std::vector<int>& last_moved_indices() const noexcept {
        return last_moved_indices_;
    }
    [[nodiscard]] float last_delta_energy() const noexcept { return last_delta_e_; }
    /// The Metropolis-Hastings correction term from the most recent
    /// debug_force_rama_pivot_to call (0 for a move with a symmetric
    /// proposal, or if no such call has happened yet). NOT updated by
    /// run() -- for that path, log_jacobian_weight is consumed directly
    /// inside the acceptance formula (see total_beta_E in run()) and never
    /// needs to be separately retained. This getter exists specifically to
    /// support direct numerical detailed-balance tests that need to
    /// inspect the correction term the engine actually used for a
    /// hand-forced transition, independent of accept/reject.
    [[nodiscard]] float last_log_jacobian_weight() const noexcept {
        return last_log_jacobian_weight_;
    }

    // ── Fixed residues ──────────────────────────────────────────────
    /// Mark residues that must never be moved by any MC proposal.
    /// @param residue_indices  0-based engine residue indices.
    /// @param n_residues       total number of residues in the system (for validation).
    void setFixedResidues(const std::vector<int>& residue_indices, int n_residues) {
        fixed_residue_.assign(static_cast<size_t>(n_residues), 0);
        for (int r : residue_indices) {
            if (r < 0 || r >= n_residues) {
                throw std::out_of_range(
                    "setFixedResidues: residue index " + std::to_string(r) +
                    " out of range [0, " + std::to_string(n_residues) + ")");
            }
            fixed_residue_[static_cast<size_t>(r)] = 1;
        }
        rebuildFixedPrefix_();
    }

    void clearFixedResidues() noexcept {
        fixed_residue_.clear();
        fixed_prefix_.clear();
    }

    const std::vector<uint8_t>& fixedResidueMask() const noexcept { return fixed_residue_; }

    std::vector<int> getFixedResidues() const {
        std::vector<int> out;
        for (size_t i = 0; i < fixed_residue_.size(); ++i) {
            if (fixed_residue_[i]) out.push_back(static_cast<int>(i));
        }
        return out;
    }

    bool hasFixedResidues() const noexcept { return !fixed_residue_.empty(); }

    /// O(1) query: does the half-open residue range [lo, hi) contain any fixed residue?
    bool segmentContainsFixed(int lo, int hi) const noexcept {
        if (fixed_prefix_.empty() || lo >= hi) return false;
        return fixed_prefix_[static_cast<size_t>(hi)] -
               fixed_prefix_[static_cast<size_t>(lo)] > 0;
    }

    /// O(1) single-residue check.
    bool isResidueFixed(int r) const noexcept {
        if (fixed_residue_.empty()) return false;
        return fixed_residue_[static_cast<size_t>(r)] != 0;
    }

    long long get_fixed_rejected() const noexcept { return fixed_rejected_; }

private:
    float temperature;
    float step_size_rad;
    /// Resolved continuous-sidechain chi amplitude (the "-1 = same as backbone"
    /// sentinel is never stored here; the constructor resolves it).
    float sidechain_step_size_rad_;

    std::vector<std::shared_ptr<Reporter>> reporters_;
    long long bb_attempted_ = 0;
    long long bb_accepted_ = 0;
    long long sc_attempted_ = 0;
    long long sc_accepted_ = 0;
    long long kic_attempted_ = 0;
    long long kic_accepted_ = 0;
    long long num_pivot_resample_pro_phi_ = 0;
    long long num_sc_resample_pro_ = 0;
    long long kic_presolve_zero_ = 0;
    long long kic_jacobian_invalid_ = 0;
    long long kic_geometry_invalid_ = 0;
    long long kic_reverse_missing_ = 0;   // KIC FIX: reverse check refusals
    long long kic_proline_skipped_ = 0;   // KIC FIX: proline-phi skips
    long long steric_rejected_ = 0;
    long long rotamer_attempted_ = 0;
    long long rotamer_accepted_ = 0;
    long long rama_pivot_attempted_ = 0;
    long long rama_pivot_accepted_ = 0;

    SidechainMoveMode sidechain_move_mode_ = SidechainMoveMode::RotamerLibrary;
    /// See set_pivot_rama_probability's docs. Default 0.0 (opt-in).
    float pivot_rama_probability_ = 0.0f;

    /// Move-slot weights, always normalized to sum 1 by set_move_weights.
    /// The defaults are the literals the mix was hard-coded to, so an
    /// untouched integrator reproduces the previous stream exactly.
    float move_w_pivot_ = 0.25f;
    float move_w_kic_   = 0.25f;
    float move_w_sc_    = 0.50f;

    /// Slot for a [0,1) roll: 0 = Pivot, 1 = KIC, 2 = Sidechain.
    ///
    /// THE single definition of the mix. run() and verify_physics_consistency
    /// both go through here; they previously carried the split as duplicated
    /// literals and could drift apart without anything noticing.
    [[nodiscard]] int select_move_slot(float roll) const noexcept {
        if (roll < move_w_pivot_) return 0;
        if (roll < move_w_pivot_ + move_w_kic_) return 1;
        return 2;
    }

    /// Throws when the weights would spend steps on moves this system cannot
    /// make. Called once per run() / verify_physics_consistency(), O(n_res).
    void check_move_weights_are_usable(const Context& context) const;

    std::mt19937 rng{ std::random_device{}() };
    std::uniform_int_distribution<int> pivot_residue_dist;
    std::uniform_int_distribution<int> sc_residue_dist;
    std::uniform_real_distribution<float> coin_flip{ 0.0f, 1.0f };
    std::normal_distribution<float> angle_dist;
    /// Continuous-sidechain chi amplitude. A dedicated distribution rather than
    /// angle_dist with a transient param_type, for the same reason
    /// unit_normal_dist_ is (see its note below). Constructed with the resolved
    /// (non-sentinel) sigma; kept in sync by set_sidechain_step_size_rad.
    ///
    /// Note this DOES change the continuous-sidechain RNG stream relative to
    /// the pre-knob code even when the two amplitudes are equal, because a
    /// separate distribution keeps its own cached spare Gaussian. That is
    /// invisible in the default rotamer_library mode (which never draws from
    /// either of these for chi), and continuous-mode trajectories were going to
    /// change the moment anyone set a different amplitude anyway -- which is
    /// the entire point of the knob. No test pins a fingerprint VALUE; the
    /// bit-identical tests are all same-seed self-comparisons.
    std::normal_distribution<float> sc_angle_dist_;
    /// Standard-normal draws for the rotamer-library move's per-chi noise.
    /// Deliberately a dedicated member rather than reusing angle_dist with a
    /// transient custom param_type: whether libstdc++'s cached "spare"
    /// Gaussian is safe to share across calls with differing (mean, sigma)
    /// params is implementation-defined, and this codebase otherwise avoids
    /// relying on unspecified RNG behavior (see get_rng_state's docs on
    /// build-toolchain-exact reproducibility). Needs the identical .reset()
    /// treatment as angle_dist -- see get_rng_state/set_rng_state/set_seed.
    std::normal_distribution<float> unit_normal_dist_{0.0f, 1.0f};

    /// Proposal buffer: DynamicOnly when pooled; Full when emulating vanilla.
    std::unique_ptr<State> proposal_;
    ProposalPatch patch_;
    int pooled_num_atoms_ = -1;
    int pooled_num_residues_ = -1;
    bool proposal_buffer_is_full_ = false;
    bool use_pooled_proposal_ = (MCPU_USE_POOLED_PROPOSAL != 0);
    /// When true (default): skip per-step full proposal sync; restore on reject. O(n_moved).
    bool use_sparse_proposal_ = true;
    /// False until first ``copy_dynamic_from`` in the current ``run`` (or after AutoRecenter).
    bool proposal_synced_ = false;
    std::vector<uint8_t> last_accept_bits_;
    StepStats step_stats_;
    bool step_stats_verbose_ = false;

    /// ADDED: last-step move context for crash/failure snapshots.
    std::string last_move_kind_str_ = "unknown";
    bool last_is_rigid_ = false;
    std::vector<int> last_moved_indices_;
    float last_delta_e_ = 0.f;
    /// Retained MH correction of the most recent forced-transition probe;
    /// see last_log_jacobian_weight(). run() consumes the patch's weight
    /// inline and never needs it kept.
    float last_log_jacobian_weight_ = 0.f;

    // ── Fixed-residue storage ───────────────────────────────────────
    std::vector<uint8_t> fixed_residue_;   // size n_res; 1 = fixed
    std::vector<int>     fixed_prefix_;    // size n_res+1; prefix sum for O(1) segment queries
    long long fixed_rejected_ = 0;

    void rebuildFixedPrefix_() {
        const size_t n = fixed_residue_.size();
        fixed_prefix_.resize(n + 1);
        fixed_prefix_[0] = 0;
        for (size_t i = 0; i < n; ++i) {
            fixed_prefix_[i + 1] = fixed_prefix_[i] + static_cast<int>(fixed_residue_[i]);
        }
    }

    void ensure_proposal_buffers(const Context& context);

    void apply_pivot_move(Context& context, State& proposal, ProposalPatch& patch);
    /// Apply pivot at fixed residue / φ|ψ (used by production + debug_force_pivot).
    void apply_pivot_at(Context& context, State& proposal, ProposalPatch& patch,
                        int r, bool is_phi);
    void apply_sidechain_move(Context& context, State& proposal, ProposalPatch& patch);
    void apply_sidechain_at(Context& context, State& proposal, ProposalPatch& patch, int r);
    /// Discrete rotamer-library sidechain proposal (see RotamerLibrary and
    /// this function's definition for the full detailed-balance argument).
    /// Same residue-selection contract as apply_sidechain_move (proline/
    /// fixed-residue skip via the shared sc_residue_dist/resample loop).
    void apply_rotamer_move(Context& context, State& proposal, ProposalPatch& patch);
    void apply_rotamer_at(Context& context, State& proposal, ProposalPatch& patch, int r);
    void apply_concerted_rotation_move(Context& context, State& proposal, ProposalPatch& patch);
    /// Knowledge-based backbone pivot proposal: jointly resamples (phi,psi)
    /// at one residue from RamaMixtureLibrary (see that class and this
    /// function's definition for the full detailed-balance argument). Same
    /// residue-selection contract as apply_pivot_move (proline/fixed-
    /// residue skip via the shared pivot_residue_dist/resample loop), but
    /// restricted to the C-term (downstream) direction only -- see
    /// apply_rama_pivot_at's definition for why.
    void apply_rama_pivot_move(Context& context, State& proposal, ProposalPatch& patch);
    void apply_rama_pivot_at(Context& context, State& proposal, ProposalPatch& patch, int r);
    /// Shared by apply_rama_pivot_at (draws its own target via RNG) and
    /// debug_force_rama_pivot_to (forces an explicit target) -- see this
    /// function's definition for why the geometry/patch logic must not be
    /// duplicated between the two callers. Returns whether the move was
    /// successfully applied (false only for the scattered-layout v1
    /// limit); `patch.is_valid` mirrors the return value.
    bool apply_rama_pivot_to_target(Context& context, State& proposal, ProposalPatch& patch,
                                    int r, int category, const std::array<float, 2>& new_phi_psi);
    /// Picks continuous vs. knowledge-based pivot proposal per
    /// pivot_rama_probability_ and applies it; sets used_rama_pivot to
    /// which one was used. Shared by run() and verify_physics_consistency
    /// so their dispatch logic can never drift apart.
    void dispatch_pivot_move(Context& context, State& proposal, ProposalPatch& patch,
                             bool& used_rama_pivot);

    /// Emulate pre-pooling reject restore (moved coords/torsions only).
    static void restore_proposal_from_accepted(State& proposal, const State& accepted,
                                               const ProposalPatch& patch);
};

} // namespace mcpu
