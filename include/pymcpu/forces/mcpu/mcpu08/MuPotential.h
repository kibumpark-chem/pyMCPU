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

#ifndef MCPU_FAST_MU_DELTA
#define MCPU_FAST_MU_DELTA 1
#endif

namespace mcpu {
    class Context;
    class CellListMC;
    struct MuWorkspace;
}

namespace mcpu::forces::mcpu08 {

    struct ContactData {
        bool check_contact=true;
        bool check_clash=true;
        float energy=0.0f;
        float contact_dist_sq=0.0f;
        float hard_core_sq=0.0f;
    };

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
                energy_mask_ptr_ = nullptr;
            }
        }

#if MCPU_FAST_MU_DELTA
        /// Per-pair contact energy (neighbor-grid contact-cache bookkeeping).

        // ── Compact pair table (opt-in; default off until validated) ──────────

        /// Type×type energy parameters, 16 B/entry (deliberately a power of two).
        /// Indexed [type_i * n_types_ + type_j]. At N_TYPES=84 that is 7056
        /// entries = 110 KB -- L2-resident, NOT L1 (32 KB L1d).
        ///
        /// FIXED: keep sizeof == 16. A precomputed `hard_r` field briefly took
        /// this to 20 B, which is not a divisor of the 64 B line, so 25% of
        /// entries straddled two lines (at 16 B: none do) and the by-value copy
        /// in eval_pair_layered_v2 stopped being a single movups. `hard_r` was
        /// write-only anyway -- it existed solely to feed hard_tol_r2_from() on
        /// the following line at setup, so it is now a local there instead.
        struct TypePairParams {
            float hard_r2    = 0.f;
            float contact_r2 = 0.f;
            float energy     = 0.f;
            // FIXED: ONE tolerant hard-core threshold, squared, shared by BOTH
            // the full recompute and the rigid-MM delta guard.
            //
            // These two paths previously used DIFFERENT thresholds: the
            // recompute an exact `r2 < hard_r2`, the guard a 3-decimal rounded
            // `round(r*1e3) < round(hard_r*1e3) - 1`. That left a 0.001 A window
            // where the delta path accepted a move and the recompute then
            // reported StericClash on the ACCEPTED state. Since skip_rigid_mm
            // elides moved-moved evaluation entirely, only the permissive guard
            // covered intra-segment pairs, so ~1e-6 A of float32 per-rotation
            // drift sufficed to cross it -- 38-41% of top-rung REMD cycles on
            // p19.14.2/p19.14.3 carried a sentinel energy as a result.
            // Sharing one number makes the paths agree by construction.
            float hard_tol_r2 = 0.f;
        };
        std::vector<TypePairParams> type_params_;  ///< size n_types_ * n_types_

        /// Conservative r² bound above which ``rigid_mm_clash_3decimal`` provably
        /// cannot return true, so the rigid-MM clash guard can be skipped without
        /// changing any outcome. Derivation: the predicate is
        ///   round(sqrt(r2)*1000) < round(hard_r*1000)
        /// which requires sqrt(r2) < hard_r + 0.001. Using the GLOBAL maximum
        /// hard_r over all type pairs makes the bound valid for every pair.
        /// A further +0.001 Å is added for float rounding slack.
        /// Set in apply_mu_denselist_cutoff(); +inf default = never skip (safe).
        float mm_guard_prefilter_r2_ = std::numeric_limits<float>::infinity();
        int n_types_ = 0;

        /// ADDED: MM clash margin (Å² added to hard_r2) when skip_rigid_mm.
        /// 0 = off (default). Env ``MCPU_MM_CLASH_MARGIN`` (e.g. 0.01).
        float mm_clash_margin_ = 0.f;
        /// ADDED: double-r² MM boundary clash check when skip_rigid_mm.
        /// Env ``MCPU_MM_DOUBLE_BOUNDARY=1``.
        bool mm_double_boundary_ = false;
        /// Half-width (Å²) around hard_r2 for double-boundary MM checks.
        float mm_double_boundary_sq_ = 0.1f;

        // ── Three-layer eval (opt-in; default off until validated) ───────────
        // Layer 1: O(N) topology metadata — on-the-fly clash/contact enable.
        // Layer 2: type_params_ (always built) — hard_r2 / contact_r2 / energy.
        // Layer 3: SparseClashExceptions — native-dist structure clash disables.
        // Needs atom_role + is_sc, not just is_bb.

        /// Atom-name roles for Layer 1 (mirrors mu_builder.py name checks).
        enum class MuAtomRole : uint8_t {
            Other = 0,
            H = 1,
            N = 2,
            CA = 3,
            C = 4,
            O = 5,       ///< O / OCT / OXT
            CB = 6,
            CD = 7,
            SG = 8,
            Gx = 9       ///< name.startswith('G') — CG*, OG*, …
        };
        /// Residue class for Layer 1 PRO / CYS specials.
        enum class MuResClass : uint8_t {
            Other = 0,
            PRO = 1,
            CYS = 2
        };

        std::vector<int32_t> res_index_;   ///< residue index per atom; size N
        std::vector<uint8_t> is_sc_;       ///< 1 if sidechain atom, else 0
        std::vector<uint8_t> atom_role_;   ///< MuAtomRole per atom
        std::vector<uint8_t> res_class_;   ///< MuResClass per atom
        bool layer1_meta_ready_ = false;

        /// Sparse CSR of structure-clash-disabled atom pairs (Layer 3).
        /// Only pairs where native_dist² < hard_r2 AND topology would keep clash on.
        struct SparseClashExceptions {
            std::vector<int32_t> row_start;  ///< size N+1
            std::vector<int32_t> col;        ///< sorted neighbor indices

            [[nodiscard]] bool contains(int i, int j) const noexcept {
                const auto* b = col.data() + row_start[static_cast<size_t>(i)];
                const auto* e = col.data() + row_start[static_cast<size_t>(i) + 1];
                const int n = static_cast<int>(e - b);
                if (n == 0) return false;
                if (n <= 8) {
                    for (const auto* p = b; p != e; ++p)
                        if (*p == j) return true;
                    return false;
                }
                return std::binary_search(b, e, static_cast<int32_t>(j));
            }

            [[nodiscard]] bool has_any() const noexcept { return !col.empty(); }
        };
        SparseClashExceptions clash_exceptions_;

        /// Precomputed per-pair topology+structure enable flags (layered v2).
        /// Bits: bit0=clash_ok, bit1=contact_ok. Index i*N+j.
        /// Built in cache_necessary_data (includes Rules 0–8 + structure).
        /// Env ``MCPU_TOPO_FLAGS=0`` forces Layer 1 branch decode (v1).
        std::vector<uint8_t> topo_flag_;
        bool use_topo_flags_ = true;
        /// Denselist / r² prefilter cutoff² (= mu_exact_cutoff_²).
        float contact_cutoff_sq_ = 36.f;
        /// Exact Mu query cutoff (Å) from max(type_params_ contact/hard r²)×1.0001.
        /// Default ON; override with ``MCPU_MU_CUTOFF_OVERRIDE``.
        float mu_exact_cutoff_ = 6.f;

        /// Matches mu_builder skip_local_contact_range default.
        static constexpr int kSkipLocalContactRange = 4;

        /// Fills clash/contact enable from Layer 1 (mirrors mu_builder Rules 0–6).
        [[gnu::always_inline]] inline void topology_pair_flags(
            int i, int j, bool& clash_on, bool& contact_on) const noexcept
        {
            clash_on = false;
            contact_on = false;
            const auto ri = static_cast<MuAtomRole>(atom_role_[static_cast<size_t>(i)]);
            const auto rj = static_cast<MuAtomRole>(atom_role_[static_cast<size_t>(j)]);
            // Rule 0: mute H for all pairs
            if (ri == MuAtomRole::H || rj == MuAtomRole::H) {
                return;
            }
            const int sep = std::abs(
                static_cast<int>(res_index_[static_cast<size_t>(i)]) -
                static_cast<int>(res_index_[static_cast<size_t>(j)]));
            const bool sc_i = is_sc_[static_cast<size_t>(i)] != 0;
            const bool sc_j = is_sc_[static_cast<size_t>(j)] != 0;
            const auto rc_i =
                static_cast<MuResClass>(res_class_[static_cast<size_t>(i)]);
            const auto rc_j =
                static_cast<MuResClass>(res_class_[static_cast<size_t>(j)]);

            if (sep == 0) {
                // Rule 1 — contact always off
                if (sc_i == sc_j) return;  // BB–BB or SC–SC: clash off
                // BB–SC
                const int b = sc_i ? j : i;
                const int s = sc_i ? i : j;
                const auto rb = static_cast<MuAtomRole>(
                    atom_role_[static_cast<size_t>(b)]);
                const auto rs = static_cast<MuAtomRole>(
                    atom_role_[static_cast<size_t>(s)]);
                const auto rc = static_cast<MuResClass>(
                    res_class_[static_cast<size_t>(i)]);  // same residue
                if ((rb == MuAtomRole::C || rb == MuAtomRole::N ||
                     rb == MuAtomRole::CA) &&
                    rs == MuAtomRole::CB) {
                    return;  // clash off
                }
                if (rc == MuResClass::PRO) return;
                if (rb == MuAtomRole::CA && rs == MuAtomRole::Gx) return;
                clash_on = true;
                return;
            }
            if (sep == 1) {
                // Rule 2 — contact always off
                const bool i_first =
                    res_index_[static_cast<size_t>(i)] <
                    res_index_[static_cast<size_t>(j)];
                const int first = i_first ? i : j;
                const int second = i_first ? j : i;
                const auto rf = static_cast<MuAtomRole>(
                    atom_role_[static_cast<size_t>(first)]);
                const auto rs = static_cast<MuAtomRole>(
                    atom_role_[static_cast<size_t>(second)]);
                const auto rc_second = static_cast<MuResClass>(
                    res_class_[static_cast<size_t>(second)]);
                const bool sc_f = is_sc_[static_cast<size_t>(first)] != 0;
                const bool sc_s = is_sc_[static_cast<size_t>(second)] != 0;
                const bool is_pro_cd =
                    (rs == MuAtomRole::CD && rc_second == MuResClass::PRO);
                if (is_pro_cd &&
                    (rf == MuAtomRole::C || rf == MuAtomRole::CA)) {
                    return;  // clash off
                }
                if (rf == MuAtomRole::N ||
                    (rs != MuAtomRole::CA && rs != MuAtomRole::N) || sc_f ||
                    sc_s) {
                    clash_on = true;
                    return;
                }
                return;  // peptide-local: clash off
            }
            if (sep <= kSkipLocalContactRange) {
                // Rule 3: clash on, contact off.
                //
                // FIXED: was `sep < kSkipLocalContactRange`, which let residue
                // pairs at EXACTLY the skip range (default 4 -- the canonical
                // alpha-helix i,i+4 spacing) fall through to the long-range
                // branch and switch contacts ON. Legacy MCPU's CheckCorrelation
                // (init.h) gates on `> SKIP_LOCAL_CONTACT_RANGE`, i.e. contacts
                // only from sep >= 5, and mu_builder.py:194 was already fixed to
                // match (`res_diff <= skip_local_contact_range`). This Layer-1
                // decode was the last place still disagreeing.
                //
                // Not reachable in a default run -- use_topo_flags_ is on, so the
                // Python-built topo_flag_ table is what production reads -- but it
                // IS live under MCPU_TOPO_FLAGS=0, which is why that knob used to
                // change the trajectory instead of just the code path.
                clash_on = true;
                return;
            }
            // Rule 4 / 6 — distant
            if (rc_i == MuResClass::CYS && rc_j == MuResClass::CYS &&
                ri == MuAtomRole::SG && rj == MuAtomRole::SG) {
                // Rule 6: clash off, contact on
                contact_on = true;
                return;
            }
            clash_on = true;
            contact_on = sc_i || sc_j;  // BB–BB: contact off
        }

        /// Build Layer 3 CSR from final clash masks + Layer 1 rules. O(N²).
        void build_clash_exceptions();
        /// Rebuild type_params_ from contact matrices (after atom permute). O(N²).
        void rebuild_type_params_from_matrices();

        /// Shared hard-core test for both the incremental ΔE and the full
        /// O(N²) recalculation, so the two can never disagree.
        ///
        /// NOTE: this previously added a `kHardClashEpsSq = 1e-8f` slack term
        /// "for float noise near hard_r2". That term was provably a no-op and
        /// has been removed: real hard_r2 values are 4.6–8.0 Å², where the
        /// float32 ULP is 4.77e-7, so `r2 + 1e-8f == r2` exactly for every
        /// type pair in the parameter set. Dropping it is bit-identical and
        /// removes a comment that claimed a tolerance the code did not have.
        /// If a genuine tolerance is ever wanted it must be scaled to hard_r2
        /// (e.g. a relative epsilon), not an absolute 1e-8.
        /// Tolerant hard-core threshold (squared) for one type pair.
        ///
        ///     round(r*1000) < round(hard_r*1000) - 1
        /// is equivalent to
        ///     r < (round(hard_r*1000) - 1.5) / 1000
        /// so the whole 3-decimal predicate collapses to a single r2 compare
        /// once this bound is squared at setup -- no sqrt and no round in the
        /// hot loop, making the delta guard CHEAPER than the version it
        /// replaces while removing the threshold mismatch.
        [[nodiscard]] static inline float hard_tol_r2_from(float hard_r) noexcept {
            if (hard_r <= 0.f) return 0.f;
            const float t = (std::round(hard_r * 1000.f) - 1.5f) / 1000.f;
            return t > 0.f ? t * t : 0.f;
        }

        [[nodiscard]] static inline bool is_hard_clash(float r2,
                                                      float hard_tol_r2) noexcept {
            return r2 < hard_tol_r2;
        }

        /// FIXED: 3-decimal distance rounding for rigid MM clash guard.
        /// PDB coords have 0.001 Å precision — differences below that must not
        /// decide clash. ``hard_r`` is precomputed at setup (free in hot path).
        /// Rigid-MM clash test. Now IDENTICAL to is_hard_clash by construction:
        /// both take the same precomputed tolerant threshold, so the delta path
        /// and the full recompute can no longer disagree about any pair. The
        /// sqrt + round this used to perform per MM pair are gone -- the whole
        /// predicate was folded into hard_tol_r2_from() at setup.
        [[nodiscard]] static inline bool rigid_mm_clash_3decimal(
            float r2_new, float hard_tol_r2) noexcept
        {
            return is_hard_clash(r2_new, hard_tol_r2);
        }

        /// Largest hard-core radius over all type pairs, plus float slack.
        /// Beyond this distance NO pair can be a hard-core overlap, so it is the
        /// correct query radius for a clash-only neighbour search. Derived from
        /// mm_guard_prefilter_r2_, which apply_mu_denselist_cutoff() already sets
        /// from the global max hard_r.
        [[nodiscard]] inline float clash_query_radius() const noexcept {
            const float r2 = mm_guard_prefilter_r2_;
            return (r2 > 0.f && std::isfinite(r2)) ? std::sqrt(r2) : 0.f;
        }

        /// Clash-check one MM pair under skip_rigid_mm using topo clash bit +
        /// 3-decimal hard_r comparison (not soft eval_pair energy).
        [[nodiscard]] inline bool rigid_mm_pair_clashes(
            int i, int j, float r2_new) const noexcept
        {
            // Cheap scalar reject before ANY memory traffic. Provably conservative:
            // beyond this radius rigid_mm_clash_3decimal cannot fire, so skipping
            // here cannot change a decision. Avoids a random byte load into the
            // N²-byte topo_flag_ table plus a type_params_ load per MM pair.
            if (!(r2_new < mm_guard_prefilter_r2_)) return false;
            const size_t N = static_cast<size_t>(num_atoms_cached_);
            if (!topo_flag_.empty()) {
                const uint8_t flag =
                    topo_flag_[static_cast<size_t>(i) * N +
                               static_cast<size_t>(j)];
                if (!(flag & 1u)) return false;
            }
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
            return rigid_mm_clash_3decimal(r2_new, hard_tol_r2);
        }

        /// Layered v2: single-byte topo_flag_ + type_params_. O(1).
        [[gnu::always_inline]] inline float eval_pair_layered_v2(
            int i, int j, float r2, bool* clash_out) const noexcept
        {
            const size_t N = static_cast<size_t>(num_atoms_cached_);
            const uint8_t flag =
                topo_flag_[static_cast<size_t>(i) * N + static_cast<size_t>(j)];
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
                        if ((flag & 1u) && is_hard_clash(r2, g.hard_tol_r2)) {
                            if (clash_out) *clash_out = true;
                        }
                        return 0.0f;
                    }
                    return 0.0f;
                }
            }

            if ((flag & 1u) && is_hard_clash(r2, g.hard_tol_r2)) {
                if (clash_out) *clash_out = true;
                return 0.0f;
            }
            if ((flag & 2u) && r2 <= g.contact_r2) {
                ++eval_pair_nonzero_local_;
                return g.energy;
            }
            return 0.0f;
        }

        /// Layered v1: on-the-fly topology decode + CSR exceptions. O(1) amortized.
        [[gnu::always_inline]] inline float eval_pair_layered_v1(
            int i, int j, float r2, bool* clash_out) const noexcept
        {
            bool clash_on = false, contact_on = false;
            topology_pair_flags(i, j, clash_on, contact_on);
            if (!clash_on && !contact_on) {
                return 0.0f;
            }

            if (clash_on && clash_exceptions_.has_any()) {
                if (clash_exceptions_.contains(i, j)) clash_on = false;
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
                        if (clash_on && is_hard_clash(r2, g.hard_tol_r2)) {
                            if (clash_out) *clash_out = true;
                        }
                        return 0.0f;
                    }
                    return 0.0f;
                }
            }

            if (clash_on && is_hard_clash(r2, g.hard_tol_r2)) {
                if (clash_out) *clash_out = true;
                return 0.0f;
            }
            if (contact_on && g.energy != 0.0f && r2 <= g.contact_r2) {
                ++eval_pair_nonzero_local_;
                return g.energy;
            }
            return 0.0f;
        }

        /// Dispatch layered v1 (branches) or v2 (topo_flag_).
        [[gnu::always_inline]] inline float eval_pair_layered(
            int i, int j, float r2, bool* clash_out) const noexcept
        {
            if (use_topo_flags_ && !topo_flag_.empty())
                return eval_pair_layered_v2(i, j, r2, clash_out);
            return eval_pair_layered_v1(i, j, r2, clash_out);
        }

#endif

#if !MCPU_FAST_MU_DELTA
        std::vector<ContactData> contact_cache;
#endif


        /// Hot pair energy for known r2. Uses flat tables (FAST) / ContactData (legacy).
        /// Returns contact energy or 0; sets *clash_out on hard-core violation.
        [[gnu::always_inline]] inline float eval_pair(
            int i, int j, float r2, bool* clash_out
        ) const {
            ++eval_pair_calls_local_;
#if MCPU_FAST_MU_DELTA
            return eval_pair_layered(i, j, r2, clash_out);
#else
            // Residue energy masking (legacy path)
            const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(num_atoms_cached_)
                             + static_cast<size_t>(j);
            if (energy_mask_ptr_) {
                const auto ri = atom_to_residue[static_cast<size_t>(i)];
                const auto rj = atom_to_residue[static_cast<size_t>(j)];
                if (energy_mask_ptr_[static_cast<size_t>(ri)] |
                    energy_mask_ptr_[static_cast<size_t>(rj)]) {
                    if (energy_mask_mode_cached_ == EnergyMaskMode::ClashOnly) {
                        const bool check_clash = topo_clash_mask_[idx] != 0;
                        if (check_clash &&
                            is_hard_clash(r2, hard_core_sq(i, j))) {
                            if (clash_out) *clash_out = true;
                        }
                        return 0.0f;
                    }
                    return 0.0f;  // IgnoreAll
                }
            }

            const bool check_clash = topo_clash_mask_[idx] != 0;
            const bool check_contact = topo_contact_mask_[idx] != 0;
            if (check_clash && is_hard_clash(r2, hard_core_sq(i, j))) {
                if (clash_out) *clash_out = true;
                return 0.0f;
            }
            if (check_contact && r2 <= contact_dist_sq(i, j)) {
                ++eval_pair_nonzero_local_;
                return contact_energies(i, j);
            }
#endif
            return 0.0f;
        }

        float calculateEnergyChange_legacy(
            const Context& context,
            const State& old_state,
            const State& new_state,
            const ProposalPatch& patch
        ) const;

        float calculateEnergyChange_fast(
            const Context& context,
            const State& old_state,
            const State& new_state,
            const ProposalPatch& patch
        ) const;

        // ── Live contact list (MCPU_CONTACT_LIST=1) ─────────────────────
        // A running list, per atom, of the atoms it is CURRENTLY in contact
        // with, together with that contact's energy. It answers one question
        // cheaply: "what is this atom's contact energy right now?"
        //
        // The default path answers that by re-walking the atom's old
        // neighbourhood and re-measuring every distance -- roughly 3500
        // distance checks per actin pivot to rediscover about 50 contacts it
        // already knew about last step. Reading them off a list instead is
        // ~60x less work for that half of the move.
        //
        // A bitmap (State::is_contact_cache, which is what legacy MCPU's
        // data[][] is) cannot do this: you can ask a bitmap "is THIS pair in
        // contact" but not "list this atom's contacts" without scanning a whole
        // row. That distinction is why the list wins on a large protein where
        // the bitmap loses.
        //
        // THE LIST ITSELF LIVES ON State, NOT HERE. One System (hence one
        // MuPotential) is shared by every replica a rank owns, so potential-owned
        // storage would be shared across replicas with different coordinates.
        // This class is stateless with respect to it: it reads
        // state.mu_contact_list and stages changes in the per-Context
        // MuWorkspace, which Context::commit_accepted_move applies.

        /// Count of moves that could not use the list (diagnostic only).
        mutable std::uint64_t clist_fallbacks_ = 0;

        /// Rebuild a state's list from its coordinates. O(N x neighbours).
        void rebuild_contact_list(const Context& context, const State& state) const;

        /// Delta using the live contact list. Enabled with MCPU_CONTACT_LIST=1.
        float calculateEnergyChange_clist(
            const Context& context,
            const State& old_state,
            const State& new_state,
            const ProposalPatch& patch
        ) const;

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

        /// Install Layer 1 per-atom topology metadata (from Python builder).
        /// Must be called before cache_necessary_data for layered eval.
        void set_topology_atom_meta(
            std::vector<int32_t> res_index,
            std::vector<uint8_t> is_sidechain,
            std::vector<uint8_t> atom_role,
            std::vector<uint8_t> res_class
        );

        void permute_atom_indices(const AtomPermutation& perm) override;

        /// Moves that had to fall back off the live-list path (diagnostic).
        [[nodiscard]] std::uint64_t clist_fallbacks() const noexcept {
            return clist_fallbacks_;
        }


        float calculateEnergy(const Context& context, const State& state) const override;
        float calculateEnergy(
            const Context& context, const State& state,
            bool update_cache = false
        ) const;
        float calculateEnergyBrute(
            const Context& context, const State& state
        ) const;
        bool canHardReject() const noexcept override { return true; }
        RejectReason rejectionForEnergy(float energy) const noexcept override {
            return energy >= 99999.0f * 0.5f
                ? RejectReason::StericClash
                : RejectReason::None;
        }
        EnergyChangeResult calculateEnergyChange(
            const Context& context,
            const State& old_state,
            const State& new_state,
            const ProposalPatch& patch
        ) const override;

#if MCPU_FAST_MU_DELTA
        /// Diagnostic sizes (KB). O(1).
        [[nodiscard]] double type_params_size_kb() const noexcept {
            return static_cast<double>(type_params_.size() * sizeof(TypePairParams))
                / 1024.0;
        }

        /// ADDED: double r² comparison path
        void set_mm_clash_margin(float margin_r2) noexcept {
            mm_clash_margin_ = margin_r2;
        }
        [[nodiscard]] float mm_clash_margin() const noexcept {
            return mm_clash_margin_;
        }
        /// ADDED: double MM boundary clash check
        void set_mm_double_boundary(bool on) noexcept { mm_double_boundary_ = on; }
        [[nodiscard]] bool mm_double_boundary() const noexcept {
            return mm_double_boundary_;
        }

        /// Use precomputed topo_flag_ (v2) vs on-the-fly Layer 1 decode (v1).
        void set_use_topo_flags(bool on) noexcept { use_topo_flags_ = on; }
        [[nodiscard]] bool use_topo_flags() const noexcept {
            return use_topo_flags_;
        }
        /// Exact denselist cutoff (Å) from max(type_params_ contact/hard). O(1).
        [[nodiscard]] float mu_exact_cutoff() const noexcept {
            return mu_exact_cutoff_;
        }
        /// Active denselist r² prefilter (= mu_exact_cutoff_²). O(1).
        [[nodiscard]] float mu_cutoff_sq() const noexcept {
            return contact_cutoff_sq_;
        }
        [[nodiscard]] float contact_cutoff_sq() const noexcept {
            return contact_cutoff_sq_;
        }
        /// Recompute mu_exact_cutoff_ from type_params_ (or matrices) + env override.
        void apply_mu_denselist_cutoff();
        /// Debug: compare layered v1 vs v2 for representative r². O(N²).
        void verify_layered_eval_consistency() const;
        /// Microbench: eval_pair only over fixed pairs. Reports ns/call to stderr.
        void bench_eval_pair_only(int n_iter = 1000000) const;
        /// Layer 3 exception count (unique undirected pairs).
        [[nodiscard]] std::size_t clash_exception_count() const noexcept {
            return clash_exceptions_.col.size() / 2;
        }
        [[nodiscard]] double topo_flag_size_mb() const noexcept {
            return static_cast<double>(topo_flag_.size()) / (1024.0 * 1024.0);
        }
#endif

    };

} // namespace mcpu::forces::mcpu08
