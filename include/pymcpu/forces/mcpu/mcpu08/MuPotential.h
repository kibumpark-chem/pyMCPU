#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <Eigen/Dense>
#include <cstdint>
#include <cstddef>
#include "pymcpu/Potential.h"
#include "pymcpu/System.h"
#include "pymcpu/neighbor/NeighborConfig.h"


namespace mcpu {
    class Context;
    class CellListMC;
    struct MuWorkspace;
}

namespace mcpu::forces::mcpu08 {

    class MuPotential : public mcpu::Potential {
    private:
        Eigen::MatrixXf contact_energies;
        Eigen::MatrixXf contact_dist_sq;
        Eigen::MatrixXf hard_core_sq;

        std::vector<int> atom_types;
        std::vector<int> atom_to_residue;

        std::vector<uint8_t> topo_contact_mask_;
        std::vector<uint8_t> topo_clash_mask_;
        int num_atoms_cached_ = 0;
        /// Accumulated across calculateEnergyChange_*; flushed into NeighborStats.
        mutable std::uint64_t eval_pair_calls_local_ = 0;
        mutable std::uint64_t eval_pair_nonzero_local_ = 0;

        /// Per-call mask state (set at start of calculate* methods, single-threaded).
        mutable const uint8_t* energy_mask_ptr_ = nullptr;
        mutable EnergyMaskMode energy_mask_mode_cached_ = EnergyMaskMode::IgnoreAll;

        inline void setup_mask_cache(const System& sys) const {
            if (sys.has_energy_mask()) {
                energy_mask_ptr_ = sys.energy_ignored_mask().data();
                energy_mask_mode_cached_ = sys.energy_mask_mode();
            } else {
                // Reset the mode too: the full energy reads it on its own to
                // drop clashes under ClashOnly, so a mode left over from a
                // cleared mask would keep hiding every clash.
                energy_mask_ptr_ = nullptr;
                energy_mask_mode_cached_ = EnergyMaskMode::IgnoreAll;
            }
        }

        // ── Compact pair table ─────────────────────────────────────────────

        /// Type×type energy parameters, 16 B/entry (deliberately a power of two).
        /// Indexed [type_i * n_types_ + type_j]. At N_TYPES=84 that is 7056
        /// entries = 110 KB -- L2-resident, NOT L1 (32 KB L1d).
        ///
        /// Keep sizeof == 16: it divides the 64 B line, so no entry straddles
        /// two lines, and the by-value copy in eval_pair is a single movups
        /// (20 B would split 25% of entries).
        struct TypePairParams {
            float hard_r2    = 0.f;
            float contact_r2 = 0.f;
            float energy     = 0.f;
            // The move cutoff, squared: a move that puts the pair closer is
            // rejected. See hard_tol_r2_from() and ClashCutoff.
            float hard_tol_r2 = 0.f;
        };
        std::vector<TypePairParams> type_params_;  ///< size n_types_ * n_types_

        /// r² beyond which no pair can overlap: the largest hard-core radius
        /// over all type pairs, plus 0.002 A of slack. overlaps_at_move_cutoff()
        /// rejects anything farther without touching the pair tables, and
        /// clash_query_radius() sizes the clash-first stencil from it.
        /// Set in apply_mu_denselist_cutoff(); +inf default = never skip (safe).
        float clash_prefilter_r2_ = std::numeric_limits<float>::infinity();
        int n_types_ = 0;

        /// Per-pair enable flags, index i*N+j: bit0 = the pair can clash,
        /// bit1 = it can make a contact. Built in cache_necessary_data from
        /// the Python topology masks (mu_builder) and the native structure.
        std::vector<uint8_t> topo_flag_;

        /// topo_flag_ in a form that fits in cache, read by topo_flag().
        /// topo_flag_ is N*N bytes (8.5 MB on actin) and its lookups were
        /// the top L2-miss line of a Mu move (28% of all L2 misses on actin
        /// and LDH-A). Pairs within kTopoBand residues read a banded table
        /// (one row per atom, holding the atoms of residues r-kTopoBand to
        /// r+kTopoBand); farther pairs read one byte per atom-type pair, and
        /// the few far pairs that differ from their type pair's byte (native
        /// clash exemptions) are listed per atom. Built from the final
        /// topo_flag_ and checked against every entry of it, so a lookup
        /// returns exactly topo_flag_'s byte; when the check fails or the
        /// list grows past a few entries per atom, topo_flag() reads
        /// topo_flag_ itself. See build_compact_topo().
        static constexpr int kTopoBand = 4;  // kSkipLocalContactRange
        bool tf_ready_ = false;
        std::vector<int32_t> tf_res_;    ///< residue per atom
        std::vector<int32_t> tf_rank_;   ///< position in residue order
        std::vector<int32_t> tf_row_;    ///< band row start - first rank of the row
        std::vector<uint8_t> tf_cls_;    ///< atom type + 1 (0 = untyped)
        std::vector<uint8_t> tf_band_;
        std::vector<uint8_t> tf_far_;    ///< (n_cls x n_cls) far-pair byte
        int tf_ncls_ = 0;
        std::vector<uint8_t> tf_exc_atom_;    ///< 1: atom has a far exception
        std::vector<int32_t> tf_exc_start_;   ///< CSR rows over atoms, N + 1
        std::vector<int32_t> tf_exc_col_;
        std::vector<uint8_t> tf_exc_flag_;
        void build_compact_topo();
        [[gnu::noinline]] uint8_t topo_far_exception(int i, int j,
                                                     uint8_t f) const noexcept {
            const int32_t* c = tf_exc_col_.data();
            for (int32_t k = tf_exc_start_[static_cast<size_t>(i)],
                         e = tf_exc_start_[static_cast<size_t>(i) + 1];
                 k < e; ++k) {
                if (c[k] == j) return tf_exc_flag_[static_cast<size_t>(k)];
            }
            return f;
        }
        /// topo_flag_[i*N + j], from the compact form when it is built.
        [[nodiscard, gnu::always_inline]] inline uint8_t topo_flag(
            int i, int j) const noexcept {
            const size_t ui = static_cast<size_t>(i);
            const size_t uj = static_cast<size_t>(j);
            if (!tf_ready_) {
                return topo_flag_[ui * static_cast<size_t>(num_atoms_cached_) + uj];
            }
            const int d = tf_res_[uj] - tf_res_[ui];
            const bool in_band =
                static_cast<unsigned>(d + kTopoBand) <= 2u * kTopoBand;
            // Both bytes are read and one is selected with masks, so whether
            // a pair is near in sequence costs no branch (a conditional here
            // compiled to one, 3.5% of a Mu move's mispredicts). A far pair's
            // band index is masked to 0, which is always in range.
            const int32_t band_mask = -static_cast<int32_t>(in_band);
            const size_t bidx = static_cast<size_t>(
                static_cast<uint32_t>(tf_row_[ui] + tf_rank_[uj]) &
                static_cast<uint32_t>(band_mask));
            const uint32_t fb = tf_band_[bidx];
            uint32_t ff = tf_far_[static_cast<size_t>(tf_cls_[ui]) *
                                      static_cast<size_t>(tf_ncls_) +
                                  tf_cls_[uj]];
            if (__builtin_expect((tf_exc_atom_[ui] & tf_exc_atom_[uj]) != 0, 0) &&
                !in_band) {
                ff = topo_far_exception(i, j, static_cast<uint8_t>(ff));
            }
            return static_cast<uint8_t>(
                ff ^ ((fb ^ ff) & static_cast<uint32_t>(band_mask)));
        }
        /// Denselist / r² prefilter cutoff² (= mu_exact_cutoff_²).
        float contact_cutoff_sq_ = 36.f;
        /// Mu query cutoff (Å): the largest contact cutoff widened by the
        /// near-miss band (contact_band_w_r2_), or the largest hard-core
        /// cutoff if that is larger, ×1.0001. The neighbour grid's cells are
        /// at least this wide, so a one-cell stencil sees every listed pair.
        float mu_exact_cutoff_ = 6.f;

        /// The contact list's near-miss band, in Å: a contact pair less than
        /// this far outside its cutoff is listed (with energy 0), so a rigid
        /// pivot that carries it re-decides it. kContactBandSlackA covers the
        /// float error of measuring a distance.
        static constexpr float kContactBandA = 0.05f;
        static constexpr float kContactBandSlackA = 1e-4f;
        /// The band in r² terms, one value for all type pairs:
        /// 2*cmax*band + band² for the largest contact cutoff cmax, so
        /// contact_r2 + w >= (contact_r + band)² for every type pair.
        float contact_band_w_r2_ = 0.f;

        /// Matches mu_builder skip_local_contact_range default.
        static constexpr int kSkipLocalContactRange = 4;
        static_assert(kTopoBand == kSkipLocalContactRange,
                      "the band holds the pairs whose contacts are switched off by sequence");

        /// Rebuild type_params_ from contact matrices (after atom permute). O(N²).
        void rebuild_type_params_from_matrices();
        /// Stores atom pair (i, j)'s parameters as its type pair's entry.
        /// type_params_ keeps one entry per unordered type pair and is filled
        /// in atom order, so every pair of the same two types must agree; a
        /// pair that disagrees with an earlier one (two radii for one type,
        /// or an asymmetric or NaN energy) throws, instead of the last pair
        /// in atom order silently winning.
        void store_type_pair_params(std::vector<uint8_t>& filled, int i, int j,
                                    const TypePairParams& tp);

        /// The move cutoff (squared) for one type pair: the hard-core radius
        /// rounded to 0.001 A, less 0.0015 A. In 3-decimal terms,
        ///     round(r*1000) < round(hard_r*1000) - 1
        /// which is r < (round(hard_r*1000) - 1.5) / 1000, so PDB-precision
        /// noise cannot decide a clash. Squared at setup, so the hot loop
        /// compares r2 only.
        [[nodiscard]] static inline float hard_tol_r2_from(float hard_r) noexcept {
            if (hard_r <= 0.f) return 0.f;
            const float t = (std::round(hard_r * 1000.f) - 1.5f) / 1000.f;
            return t > 0.f ? t * t : 0.f;
        }

        [[nodiscard]] static inline bool is_hard_clash(float r2,
                                                      float hard_tol_r2) noexcept {
            return r2 < hard_tol_r2;
        }

        /// Which hard-core cutoff a pair evaluation applies.
        ///
        /// Move: the cutoff a move is tested against (hard_tol_r2), on the new
        ///   side of every delta path, so no move can put a pair under it.
        /// State: kStateClashBufferA looser, for judging a state that exists
        ///   (calculateEnergy). An accepted state can hold a pair that a rigid
        ///   pivot carried a few 1e-6 A under the move cutoff by rounding (see
        ///   kStateClashBufferA); that is no clash.
        /// None: no clash test. The old side of every delta path takes back
        ///   what the running energy holds for a pair, and the running energy
        ///   never holds a clash, since a move that makes one is rejected: it
        ///   holds the pair's contact energy, wherever rounding has carried it
        ///   since. rebuild_contact_list lists a pair the same way, and a
        ///   carried pair is re-decided for its contact only.
        enum class ClashCutoff : uint8_t { Move, State, None };

        /// The state cutoff (squared) for a pair whose move cutoff is
        /// hard_tol_r2. Only called for a pair already under the move cutoff.
        [[nodiscard]] static inline float state_clash_r2_from(float hard_tol_r2) noexcept {
            const float t = std::sqrt(hard_tol_r2) - kStateClashBufferA;
            return t > 0.f ? t * t : 0.f;
        }

        template <ClashCutoff C>
        [[nodiscard]] static inline bool clashes_at(float r2, float hard_tol_r2) noexcept {
            if constexpr (C == ClashCutoff::None) return false;
            if (!is_hard_clash(r2, hard_tol_r2)) return false;
            if constexpr (C == ClashCutoff::Move) return true;
            return r2 < state_clash_r2_from(hard_tol_r2);
        }

        /// Largest hard-core radius over all type pairs, plus float slack.
        /// Beyond this distance no pair can overlap, so it is the query radius
        /// of the clash-first neighbour search.
        [[nodiscard]] inline float clash_query_radius() const noexcept {
            const float r2 = clash_prefilter_r2_;
            return (r2 > 0.f && std::isfinite(r2)) ? std::sqrt(r2) : 0.f;
        }

        /// True when the residue energy mask switches pair (i, j) off entirely
        /// (IgnoreAll with either residue masked), as eval_pair does: such a
        /// pair can neither clash nor make a contact. ClashOnly keeps clashes,
        /// so it never switches a pair off here.
        [[nodiscard]] inline bool mask_ignores_pair(int i, int j) const noexcept {
            if (!energy_mask_ptr_ || energy_mask_mode_cached_ != EnergyMaskMode::IgnoreAll) {
                return false;
            }
            const auto ri = atom_to_residue[static_cast<size_t>(i)];
            const auto rj = atom_to_residue[static_cast<size_t>(j)];
            return (energy_mask_ptr_[static_cast<size_t>(ri)] |
                    energy_mask_ptr_[static_cast<size_t>(rj)]) != 0;
        }

        /// Whether pair (i, j) at distance² r2 overlaps under the move cutoff:
        /// the clash half of eval_pair, without the contact energy. Used by the
        /// clash-first pass, which tests moved atoms against fixed ones.
        [[nodiscard]] inline bool overlaps_at_move_cutoff(
            int i, int j, float r2) const noexcept
        {
            // Cheap scalar reject before ANY memory traffic: no pair overlaps
            // beyond this radius. Avoids a random byte load into the N²-byte
            // topo_flag_ table plus a type_params_ load per pair.
            if (!(r2 < clash_prefilter_r2_)) return false;
            if (mask_ignores_pair(i, j)) return false;
            if (!(topo_flag(i, j) & 1u)) return false;
            const size_t NT = static_cast<size_t>(n_types_);
            const int ti = atom_types[static_cast<size_t>(i)];
            const int tj = atom_types[static_cast<size_t>(j)];
            float hard_tol_r2 = 0.f;
            if (ti >= 0 && tj >= 0 && NT > 0 &&
                !type_params_.empty()) {
                hard_tol_r2 = type_params_[static_cast<size_t>(ti) * NT +
                                           static_cast<size_t>(tj)]
                                  .hard_tol_r2;
            }
            return is_hard_clash(r2, hard_tol_r2);
        }

        /// Hot pair energy for known r2. Returns contact energy or 0; sets
        /// *clash_out on a hard-core overlap under cutoff C, and *near_out for
        /// a contact pair outside its cutoff by less than the near-miss band
        /// (see kContactBandA). One topo_flag() byte decides which halves
        /// apply; type_params_ supplies the radii and energy.
        template <ClashCutoff C = ClashCutoff::Move>
        [[gnu::always_inline]] inline float eval_pair(
            int i, int j, float r2, bool* clash_out, bool* near_out = nullptr
        ) const {
            ++eval_pair_calls_local_;
            const uint8_t flag = topo_flag(i, j);
            if (flag == 0) {
                return 0.0f;
            }

            const size_t NT = static_cast<size_t>(n_types_);
            const int ti = atom_types[static_cast<size_t>(i)];
            const int tj = atom_types[static_cast<size_t>(j)];
            TypePairParams g{};
            if (ti >= 0 && tj >= 0 && NT > 0) {
                g = type_params_[static_cast<size_t>(ti) * NT +
                                 static_cast<size_t>(tj)];
            }

            if (energy_mask_ptr_) {
                const auto ri = atom_to_residue[static_cast<size_t>(i)];
                const auto rj = atom_to_residue[static_cast<size_t>(j)];
                if (energy_mask_ptr_[static_cast<size_t>(ri)] |
                    energy_mask_ptr_[static_cast<size_t>(rj)]) {
                    if (energy_mask_mode_cached_ == EnergyMaskMode::ClashOnly) {
                        if ((flag & 1u) && clashes_at<C>(r2, g.hard_tol_r2)) {
                            if (clash_out) *clash_out = true;
                        }
                        return 0.0f;
                    }
                    return 0.0f;
                }
            }

            if ((flag & 1u) && clashes_at<C>(r2, g.hard_tol_r2)) {
                if (clash_out) *clash_out = true;
                return 0.0f;
            }
            if ((flag & 2u) && r2 <= g.contact_r2) {
                ++eval_pair_nonzero_local_;
                return g.energy;
            }
            if (near_out && (flag & 2u) && g.energy != 0.0f &&
                r2 < g.contact_r2 + contact_band_w_r2_) {
                *near_out = true;
            }
            return 0.0f;
        }

        /// Contact energy of a pair already on the contact list at distance²
        /// r2. Being listed proves it contact-eligible and unmasked, so only
        /// the type cutoff is left to test.
        [[nodiscard]] inline float listed_contact_energy(
            int i, int j, float r2) const noexcept
        {
            const size_t NT = static_cast<size_t>(n_types_);
            const TypePairParams& g =
                type_params_[static_cast<size_t>(atom_types[static_cast<size_t>(i)]) * NT +
                             static_cast<size_t>(atom_types[static_cast<size_t>(j)])];
            return r2 <= g.contact_r2 ? g.energy : 0.0f;
        }

        /// calculateEnergyChange_fast's moved-vs-all scan, for a trial out of
        /// the neighbour grid (or with none).
        double delta_moved_vs_all(const Context& context, const State& old_state,
                                 const State& new_state, const ProposalPatch& patch,
                                 const std::vector<int>& moved_indices,
                                 bool list_exact) const;

        /// Contact energy change of every pair a rigid move carries (both
        /// atoms in moved), re-decided from the new coordinates without a
        /// clash test (ClashCutoff::None). O(n_moved^2); for moves without
        /// an exact contact list.
        double carried_pairs_delta(const State& old_state, const State& new_state,
                                  const std::vector<int>& moved) const;

        /// The delta without the contact list. list_exact: the old state's
        /// list is exact for this move (ready, and within its drift budget
        /// after it), so a rigid move re-decides only the carried pairs it
        /// lists.
        double calculateEnergyChange_fast(
            const Context& context,
            const State& old_state,
            const State& new_state,
            const ProposalPatch& patch,
            bool list_exact = false
        ) const;

        // ── Live contact list (default; MCPU_CONTACT_LIST=0 turns it off) ──
        // A running list, per atom, of the atoms it is CURRENTLY in contact
        // with, together with that contact's energy. It answers one question
        // cheaply: "what is this atom's contact energy right now?"
        //
        // It also lists, with energy 0, every contact pair less than
        // kContactBandA outside its cutoff. A rigid pivot does not re-measure
        // the pairs it carries -- that is what makes it cheap -- but its
        // rounding moves them by up to sqrt(3) float steps of the largest
        // coordinate (Context::rigid_carry_bound_A). So it re-decides each listed pair it
        // carries, and an unlisted one cannot cross: State::mu_list_drift sums
        // the bound over accepted carries, and the list is rebuilt from the
        // coordinates before the sum reaches the band.
        //
        // Without the list, the delta answers that by re-walking the atom's old
        // neighbourhood and re-measuring every distance -- roughly 3500
        // distance checks per actin pivot to rediscover about 50 contacts it
        // already knew about last step. Reading them off a list instead is
        // ~60x less work for that half of the move.
        //
        // A pair bitmap (legacy MCPU's data[][]) cannot do this: you can ask
        // a bitmap "is THIS pair in contact" but not "list this atom's
        // contacts" without scanning a whole row. That distinction is why the
        // list wins on a large protein where the bitmap loses.
        //
        // THE LIST ITSELF LIVES ON State, NOT HERE. One System (hence one
        // MuPotential) is shared by every replica a rank owns, so potential-owned
        // storage would be shared across replicas with different coordinates.
        // This class is stateless with respect to it: it reads
        // state.mu_contact_list and stages changes in the per-Context
        // MuWorkspace, which Context::commit_accepted_move applies.

        /// Count of moves that could not use the list (diagnostic only).
        mutable std::uint64_t clist_fallbacks_ = 0;
        /// Count of rebuild_contact_list calls (diagnostic only).
        mutable std::uint64_t contact_list_rebuilds_ = 0;


        /// Rebuild a state's list from its coordinates. O(N^2).
        void rebuild_contact_list(const Context& context, const State& state) const;

        /// calculateEnergy and resyncEnergy: the O(N^2) full energy; with
        /// resync, also rewrites the state's live contact list.
        double full_energy(const Context& context, const State& state, bool resync) const;

        /// Delta using the live contact list (default; MCPU_CONTACT_LIST=0 turns
        /// it off). carry_bound is Context::rigid_carry_bound_A for a rigid move, else 0.
        double calculateEnergyChange_clist(
            const Context& context,
            const State& old_state,
            const State& new_state,
            const ProposalPatch& patch,
            float carry_bound
        ) const;

        /// First moved atom whose new position overlaps an unmoved atom held
        /// by the accepted Mu grid, or -1 if none does: the clash-first pass.
        /// Tests the clash_hot atoms the move carries first, then, unless
        /// `hot_only`, every other moved atom, last first. moved_per_cell
        /// counts, per cell, the listed atoms the move displaces. Only the
        /// order of the search depends on clash_hot, never its answer.
        inline int first_grid_overlap(const Context& context,
                                      const State& new_state,
                                      const ProposalPatch& patch,
                                      const std::uint8_t* moved_per_cell,
                                      bool hot_only) const;

    public:
        explicit MuPotential(
            Eigen::MatrixXf  contact_energies,
            Eigen::MatrixXf  contact_dist_sq,
            Eigen::MatrixXf  hard_core_sq,
            std::vector<int> atom_types,
            std::vector<int> atom_to_residue
        );

        void cache_necessary_data(
            const std::vector<int8_t>& topo_contact_mask,
            const std::vector<int8_t>& topo_clash_mask,
            const Eigen::Matrix3Xf& coords
        );

        void permute_atom_indices(const AtomPermutation& perm) override;

        /// Times a state's contact list was rebuilt from its coordinates.
        [[nodiscard]] std::uint64_t contact_list_rebuilds() const noexcept {
            return contact_list_rebuilds_;
        }
        /// Moves that had to fall back off the live-list path (diagnostic).
        [[nodiscard]] std::uint64_t clist_fallbacks() const noexcept {
            return clist_fallbacks_;
        }


        double calculateEnergy(const Context& context, const State& state) const override;
        /// Also rewrites the state's live contact list from the same pass.
        double resyncEnergy(const Context& context, const State& state) const override;
        bool canHardReject() const noexcept override { return true; }
        RejectReason rejectionForEnergy(double energy) const noexcept override {
            return energy >= 99999.0f * 0.5f
                ? RejectReason::StericClash
                : RejectReason::None;
        }
        /// O(n_moved x N) scan of the pairs the move re-evaluates, at the
        /// move cutoff. For PhysicsVerifier, not the hot path.
        bool clashesAtMoveCutoff(
            const Context& context,
            const State& proposed_state,
            const ProposalPatch& patch
        ) const override;
        EnergyChangeResult calculateEnergyChange(
            const Context& context,
            const State& old_state,
            const State& new_state,
            const ProposalPatch& patch
        ) const override;

        /// The Mu query cutoff (Å); see mu_exact_cutoff_. O(1).
        [[nodiscard]] float mu_exact_cutoff() const noexcept {
            return mu_exact_cutoff_;
        }
        /// Recompute mu_exact_cutoff_ from type_params_ (or the matrices).
        void apply_mu_denselist_cutoff();

    };

} // namespace mcpu::forces::mcpu08
