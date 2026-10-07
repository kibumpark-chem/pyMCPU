#pragma once
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
/// NeighborSystem: single lifecycle owner for Mu + HBond spatial indices (NO PBC).
///
/// Desync is prevented by construction: potentials never insert/remove/update grids.
/// Only rebuild_from_accepted_state() and commit_accepted_move() mutate indices.
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "pymcpu/CellList.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/System.h"
#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/neighbor/OpenCellGrid.h"
#include "pymcpu/neighbor/PairScratch.h"
#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/CoordView.h"
#include "pymcpu/utils/virtual_amide_h.h"

namespace mcpu {

class NeighborSystem {
public:
    static constexpr float kMuCutoffFallbackA = 6.0f;
    static constexpr float kHBondCutoffA = 2.5f;
    /// The H-bond ledger lists every donor-acceptor pair whose H and O are
    /// closer than this (HBondStateCache), so the H-bond cells are this big.
    static constexpr float kHBondListA = 2.55f;

    /// Active Mu denselist cutoff (Å): from MuPotential (exact) or fallback 6.0.
    [[nodiscard]] float mu_cutoff_A() const noexcept {
        return (cfg_.mu_denselist_cutoff_A > 0.f) ? cfg_.mu_denselist_cutoff_A
                                                    : kMuCutoffFallbackA;
    }

    NeighborSystem() = default;

    void init(const System& sys, NeighborConfig cfg = {}) {
        cfg_ = cfg;
        n_atoms_ = sys.getNumAtoms();
        n_bb_ = sys.getTotalBBAtoms();
        n_o_ = sys.getTotalOAtoms();
        n_sc_ = sys.getTotalSCAtoms();
        n_h_ = sys.getTotalHAtoms();
        residue_contiguous_ = sys.residueContiguousLayout();
        // Legacy segment bounds (valid when !residue_contiguous_).
        h_begin_ = n_bb_ + n_o_ + n_sc_;
        o_begin_ = n_bb_;
        o_end_ = n_bb_ + n_o_;
        virtual_amide_h_ = sys.virtualAmideH() && n_h_ == 0;
        hb_h_ids_are_residues_ = virtual_amide_h_;

        donor_bb_starts_.assign(static_cast<size_t>(sys.getNumResidues()), -1);
        donor_c_starts_.assign(static_cast<size_t>(sys.getNumResidues()), -1);
        amide_donor_.assign(static_cast<size_t>(sys.getNumResidues()), 0);
        in_mu_.assign(static_cast<size_t>(n_atoms_), 0);
        is_o_.assign(static_cast<size_t>(n_atoms_), 0);
        is_h_.assign(static_cast<size_t>(n_atoms_), 0);
        o_atom_ids_.clear();
        h_dep_.clear();
        h_atom_ids_.clear();

        const auto& blocks = sys.getBlockIndices();
        for (int r = 0; r < sys.getNumResidues(); ++r) {
            const auto& b = blocks[static_cast<size_t>(r)];
            donor_bb_starts_[static_cast<size_t>(r)] = b.bb_start;
            donor_c_starts_[static_cast<size_t>(r)] = b.c_atom();
            amide_donor_[static_cast<size_t>(r)] = b.amide_donor ? 1 : 0;

            auto mark_mu = [&](int i) {
                if (i >= 0 && i < n_atoms_) in_mu_[static_cast<size_t>(i)] = 1;
            };
            mark_mu(b.bb_start);
            mark_mu(b.ca_atom());
            mark_mu(b.c_atom());
            if (b.o_start >= 0 && b.o_start < n_atoms_) {
                is_o_[static_cast<size_t>(b.o_start)] = 1;
                o_atom_ids_.push_back(b.o_start);
                mark_mu(b.o_start);
            }
            if (b.sc_start >= 0 && b.sc_count > 0) {
                for (int k = 0; k < b.sc_count; ++k) mark_mu(b.sc_start + k);
            }
            if (b.h_start >= 0 && b.h_start < n_atoms_) {
                is_h_[static_cast<size_t>(b.h_start)] = 1;
                h_atom_ids_.push_back(b.h_start);
            }
        }
        // Mu full energy evaluates every non-amide-H atom. Mirror that exact
        // occupancy here so terminal OXT/OCT and other valid extra atoms cannot
        // be absent from incremental cell-list candidates.
        for (int i = 0; i < n_atoms_; ++i) {
            in_mu_[static_cast<size_t>(i)] =
                sys.is_amide_h_atom(i) ? 0 : 1;
        }
        // Legacy layout fallback if blocks lacked SC counts: Mu = [0, h_begin).
        if (!residue_contiguous_ && o_atom_ids_.empty() && n_o_ > 0) {
            for (int i = 0; i < n_atoms_; ++i) {
                in_mu_[static_cast<size_t>(i)] = (i < h_begin_) ? 1 : 0;
                is_o_[static_cast<size_t>(i)] = (i >= o_begin_ && i < o_end_) ? 1 : 0;
                is_h_[static_cast<size_t>(i)] = (i >= h_begin_) ? 1 : 0;
            }
            o_atom_ids_.clear();
            for (int i = o_begin_; i < o_end_; ++i) o_atom_ids_.push_back(i);
            h_atom_ids_.clear();
            for (int i = h_begin_; i < n_atoms_; ++i) h_atom_ids_.push_back(i);
        }
        in_mu_base_ = in_mu_;
        sys_ = &sys;
        apply_energy_mask_();

        mu_grid_ = std::make_unique<CellListMC>(kMuCutoffFallbackA, n_atoms_);
        hb_o_grid_ = std::make_unique<CellListMC>(kHBondListA, n_atoms_);
        // Virtual-H grid uses residue ids as keys (capacity ≥ n_res); explicit H uses atom ids.
        const int hb_h_cap = virtual_amide_h_
            ? std::max(n_atoms_, sys.getNumResidues())
            : n_atoms_;
        hb_h_grid_ = std::make_unique<CellListMC>(kHBondListA, hb_h_cap);
        mu_grid_->ensure_atom_capacity(n_atoms_);
        hb_o_grid_->ensure_atom_capacity(n_atoms_);
        hb_h_grid_->ensure_atom_capacity(hb_h_cap);
        // The three built-in grids head the registry, at the fixed ids in
        // PairScratch.h; register_subset_grid appends after them.
        grids_.clear();
        grids_.push_back(RegisteredGrid{"mu", &NeighborSystem::mu_grid_, &NeighborSystem::in_mu_, -1});
        grids_.push_back(RegisteredGrid{"hb_o", &NeighborSystem::hb_o_grid_, &NeighborSystem::is_o_, -1});
        grids_.push_back(RegisteredGrid{"hb_h", &NeighborSystem::hb_h_grid_, &NeighborSystem::is_h_, -1});
        // Subset grids registered before a re-init (Context re-inits after
        // an atom reorder) keep their ids: re-size them for the current
        // atom count and put them back in the registry in the same order.
        for (size_t k = 0; k < subset_grids_.size(); ++k) {
            seed_subset_grid(subset_grids_[k]);
            grids_.push_back(RegisteredGrid{subset_grids_[k].spec.name, nullptr, nullptr,
                                            static_cast<int>(k)});
        }
        dense_active_ = true;
        hb_fallback_ = false;
    }

    /// A grid over a subset of atoms, for a term that needs its own cell size.
    struct GridSpec {
        const char* name;
        std::vector<int> members;   ///< atom ids the grid indexes
        float cell_A;               ///< cell edge (A)
        float max_cutoff_A;         ///< largest query radius the term uses (A)
    };

    /// Add a grid over spec.members. It is rebuilt with the others from the
    /// accepted state and kept current by commit_accepted_move, which touches
    /// it only when a move displaces one of its members. Returns its id
    /// (an index into PairScratch::moved). O(N) once; call before the first
    /// rebuild_from_accepted_state.
    neighbor::GridId register_subset_grid(const GridSpec& spec) {
        if (grids_.size() >= static_cast<size_t>(neighbor::kMaxGrids)) {
            throw std::length_error("register_subset_grid: grid registry is full");
        }
        SubsetGrid g;
        g.spec = spec;
        seed_subset_grid(g);
        subset_grids_.push_back(std::move(g));
        grids_.push_back(RegisteredGrid{spec.name, nullptr, nullptr,
                                        static_cast<int>(subset_grids_.size()) - 1});
        return static_cast<neighbor::GridId>(grids_.size() - 1);
    }
    /// Rename the members of every registered subset grid after an atom
    /// reorder: old id i becomes old_to_new[i]. Call before the re-init
    /// that follows the reorder. O(total members).
    void remap_subset_members(const std::vector<int>& old_to_new) {
        for (SubsetGrid& g : subset_grids_) {
            for (int& i : g.spec.members) {
                if (i >= 0 && i < static_cast<int>(old_to_new.size())) {
                    i = old_to_new[static_cast<size_t>(i)];
                }
            }
        }
    }
    /// Number of registered grids, built-ins included. O(1).
    int num_grids() const noexcept { return static_cast<int>(grids_.size()); }
    /// A registered subset grid, or nullptr if `id` is a built-in or unknown
    /// or the grid could not be built for the current bounds. O(1).
    const CellListMC* subset_grid(neighbor::GridId id) const noexcept {
        if (id >= grids_.size() || grids_[id].subset < 0) return nullptr;
        const SubsetGrid& g = subset_grids_[static_cast<size_t>(grids_[id].subset)];
        return g.active ? g.grid.get() : nullptr;
    }

    NeighborConfig& config() noexcept { return cfg_; }
    const NeighborConfig& config() const noexcept { return cfg_; }
    NeighborStats& stats() const noexcept { return stats_; }
    const BoxBounds& bounds() const noexcept { return bounds_; }
    /// The Mu grid can answer queries: it is built, and its membership
    /// follows the System's current energy mask (see sync_energy_mask).
    bool denseActive() const noexcept { return dense_active_ && mask_current(); }
    /// True when the Mu grid membership was taken from the System's current
    /// energy mask.
    bool mask_current() const noexcept {
        return sys_ != nullptr && sys_->energy_mask_epoch() == mask_epoch_;
    }

    /// Bring the Mu grid membership up to the System's energy mask after
    /// set/clear_energy_ignored_residues. Atoms that leave are removed in
    /// place, which keeps the order of every other atom in its cell; atoms
    /// that join go in at ``coords`` (the accepted state). O(N) when the
    /// mask changed, O(1) otherwise.
    void sync_energy_mask(const CoordsSoA& coords) {
        if (mask_current()) return;
        const std::vector<uint8_t> before = in_mu_;
        apply_energy_mask_();
        if (!dense_active_ || !mu_grid_) return;
        for (int i = 0; i < n_atoms_; ++i) {
            const size_t k = static_cast<size_t>(i);
            if (in_mu_[k] == before[k]) continue;
            if (in_mu_[k]) mu_grid_->insert(i, coords);
            else mu_grid_->remove(i);
        }
        if (note_overflow_(*mu_grid_, "Mu", stats_.mu_grid_overflows))
            dense_active_ = false;
    }
    bool hbondUsesFallback() const noexcept { return hb_fallback_; }
    /// The O and H grids, for walks on the pair-search layer; null before
    /// setup. Their ids are O atoms, and H atoms or (virtual amide H) donor
    /// residues; see hbondHIdIsResidue().
    const CellListMC* hbond_o_cells() const noexcept { return hb_o_grid_.get(); }
    const CellListMC* hbond_h_cells() const noexcept { return hb_h_grid_.get(); }
    int hBegin() const noexcept { return h_begin_; }
    bool virtualAmideH() const noexcept { return virtual_amide_h_; }
    bool hbondHIdIsResidue() const noexcept { return hb_h_ids_are_residues_; }

    /// One-shot audit label for HBond candidate source (current NeighborSystem).
    const char* hbond_backend_name() const noexcept {
        return hb_fallback_ ? "bruteforce_OH_cap_fallback" : "opencell_typed_OH_grids";
    }
    float hbond_cutoff_A() const noexcept { return kHBondCutoffA; }
    float hbond_cell_size_A() const noexcept {
        return hb_fallback_ ? 0.f : kHBondListA;
    }
    const char* mu_backend_name() const noexcept {
        return dense_active_ ? "opencell_mu_BBO_SC" : "mu_dense_cap_fallback";
    }
    float mu_cell_size_A() const noexcept {
        if (!dense_active_ || !mu_grid_) return 0.f;
        return mu_grid_->grid().cell_size();
    }

    /// Print backend signature once per process (stderr). Safe to call often.
    void maybe_print_neighbor_audit(const char* where) const {
        static bool printed = false;
        if (printed) return;
        printed = true;
        std::fprintf(stderr,
            "[neighbor-audit] where=%s "
            "Mu candidates=%s cell=%.3f cutoff=%.3f | "
            "HBond candidates=%s cell=%.3f cutoff=%.3f fallback=%d "
            "lifecycle=NeighborSystem_single\n",
            where ? where : "?",
            mu_backend_name(), mu_cell_size_A(), mu_cutoff_A(),
            hbond_backend_name(), hbond_cell_size_A(), kHBondCutoffA,
            hb_fallback_ ? 1 : 0);
    }

    /// Read-only Mu index (BB+O+SC).
    const CellListMC& muGrid() const { return *mu_grid_; }

    bool trial_in_bounds(const CoordsSoA& trial_coords,
                         const ProposalPatch& patch) const {
        if (!bounds_.valid) return true;
        if (patch.moved_as_ranges()) {
            // A pivot marks its atoms as a few index ranges (one per atom
            // kind), so test each range with the branch-free pass below; the
            // ranges are exactly the moved atoms, so the answer is the same.
            const float* x = trial_coords.x.data();
            const float* y = trial_coords.y.data();
            const float* z = trial_coords.z.data();
            const float lx = bounds_.lo.x(), ly = bounds_.lo.y(), lz = bounds_.lo.z();
            const float hx = bounds_.hi.x(), hy = bounds_.hi.y(), hz = bounds_.hi.z();
            int ok = 1;
            for (const auto& rg : patch.moved_ranges) {
                for (size_t k = static_cast<size_t>(rg.first);
                     k < static_cast<size_t>(rg.second); ++k) {
                    ok &= static_cast<int>(x[k] >= lx) & static_cast<int>(x[k] < hx)
                        & static_cast<int>(y[k] >= ly) & static_cast<int>(y[k] < hy)
                        & static_cast<int>(z[k] >= lz) & static_cast<int>(z[k] < hz);
                }
            }
            return ok != 0;
        }
        if (!patch.moved_indices.empty()) {
            // moved_indices holds no duplicates, so when its span max - min + 1
            // equals its size it is exactly the range [min, max] (every pivot
            // move): test that range with one branch-free, vectorisable pass
            // instead of a gather and six branches per atom. Same predicate
            // (a NaN coordinate still fails), so the same answer.
            int imin = patch.moved_indices.front(), imax = imin;
            for (int i : patch.moved_indices) {   // branch-free min/max reduction
                imin = std::min(imin, i);
                imax = std::max(imax, i);
            }
            const size_t lo = static_cast<size_t>(imin);
            const size_t hi = static_cast<size_t>(imax);
            if (imin >= 0 && hi - lo + 1 == patch.moved_indices.size()) {
                const float* x = trial_coords.x.data();
                const float* y = trial_coords.y.data();
                const float* z = trial_coords.z.data();
                const float lx = bounds_.lo.x(), ly = bounds_.lo.y(), lz = bounds_.lo.z();
                const float hx = bounds_.hi.x(), hy = bounds_.hi.y(), hz = bounds_.hi.z();
                int ok = 1;
                for (size_t k = lo; k <= hi; ++k) {
                    ok &= static_cast<int>(x[k] >= lx) & static_cast<int>(x[k] < hx)
                        & static_cast<int>(y[k] >= ly) & static_cast<int>(y[k] < hy)
                        & static_cast<int>(z[k] >= lz) & static_cast<int>(z[k] < hz);
                }
                return ok != 0;
            }
            for (int i : patch.moved_indices) {
                const size_t k = static_cast<size_t>(i);
                if (!point_in_bounds(trial_coords.x[k], trial_coords.y[k],
                                     trial_coords.z[k], bounds_))
                    return false;
            }
            return true;
        }
        for (size_t i = 0; i < patch.moving_atoms.size(); ++i) {
            if (patch.moving_atoms[i]) {
                const size_t k = i;
                if (!point_in_bounds(trial_coords.x[k], trial_coords.y[k],
                                     trial_coords.z[k], bounds_))
                    return false;
            }
        }
        return true;
    }

    static float max_moved_displacement(const CoordsSoA& accepted,
                                        const CoordsSoA& trial,
                                        const ProposalPatch& patch) {
        float d2max = 0.f;
        auto consider = [&](int i) {
            const size_t k = static_cast<size_t>(i);
            const float dx = trial.x[k] - accepted.x[k];
            const float dy = trial.y[k] - accepted.y[k];
            const float dz = trial.z[k] - accepted.z[k];
            const float d2 = pair_r2(dx, dy, dz);
            if (d2 > d2max) d2max = d2;
        };
        if (patch.moved_as_ranges()) {
            // Same atoms as moved_indices, read contiguously. The scalar loop
            // is bound by the latency of its running max; eight lanes each
            // keep their own max and are folded at the end. Each lane applies
            // the scalar rule (d2 > max ? d2 : max, which vmaxps is) to the
            // same d2 (fma(dz, dz, fma(dy, dy, dx*dx)), as GCC contracts the
            // scalar expression), and the max of non-NaN values does not
            // depend on order, so the result is the same bit for bit.
            for (const auto& rg : patch.moved_ranges) {
                int i = rg.first;
#if defined(__AVX2__) && defined(__FMA__)
                __m256 vmax = _mm256_setzero_ps();
                for (; i + 8 <= rg.second; i += 8) {
                    const size_t k = static_cast<size_t>(i);
                    const __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(trial.x.data() + k),
                                                    _mm256_loadu_ps(accepted.x.data() + k));
                    const __m256 dy = _mm256_sub_ps(_mm256_loadu_ps(trial.y.data() + k),
                                                    _mm256_loadu_ps(accepted.y.data() + k));
                    const __m256 dz = _mm256_sub_ps(_mm256_loadu_ps(trial.z.data() + k),
                                                    _mm256_loadu_ps(accepted.z.data() + k));
                    const __m256 d2 = _mm256_fmadd_ps(dz, dz,
                                                      _mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dy, dy)));
                    vmax = _mm256_max_ps(d2, vmax);
                }
                alignas(32) float lanes[8];
                _mm256_store_ps(lanes, vmax);
                for (float d2 : lanes)
                    if (d2 > d2max) d2max = d2;
#endif
                for (; i < rg.second; ++i) consider(i);
            }
        } else if (!patch.moved_indices.empty()) {
            for (int i : patch.moved_indices) consider(i);
        } else {
            for (size_t i = 0; i < patch.moving_atoms.size(); ++i) {
                if (patch.moving_atoms[i]) consider(static_cast<int>(i));
            }
        }
        return std::sqrt(d2max);
    }

    /// Full rebuild from accepted coords. Only public mutator besides commit.
    bool rebuild_from_accepted_state(const CoordsSoA& coords) {
        ++stats_.num_aabb_rebuild_accept;
        if (!mask_current()) apply_energy_mask_();
        retry_rebuild_ = false;
        // AABB margin gives AutoExpand headroom around the Mu cutoff.
        // Denselist cell + query stay at mu_cutoff (legacy 6 Å or exact contact).
        // Cell == cutoff: a one-cell stencil (27 cells) finds every pair,
        // and the hard core bounds what a cell of that size can hold.
        const float r_mu = mu_cutoff_A();
        const float mu_cell = r_mu;
        const float margin = cfg_.effective_margin(r_mu);
        BoxBounds b = aabb_of_coords(coords, margin);
        bounds_ = b;

        // --- Mu grid ---
        int nx = 0, ny = 0, nz = 0;
        bool mu_ok = compute_grid_shape(b, mu_cell, cfg_, nx, ny, nz);
        dense_active_ = false;
        hb_fallback_ = false;

        if (mu_ok) {
            mu_grid_->set_cutoff(r_mu); // CHANGED: denselist query matches cell cut
            if (mu_grid_->configure(b, cfg_, mu_cell)) {
                mu_grid_->reset(n_atoms_);
                for (int i = 0; i < n_atoms_; ++i) {
                    if (in_mu_[static_cast<size_t>(i)]) mu_grid_->insert(i, coords);
                }
                // Bulk insert, then the occupied stencil (Mu grid only: the
                // H-bond grids walk the full stencil, measured wall-neutral).
                mu_grid_->grid().enable_occupied_stencil();
                dense_active_ = !note_overflow_(*mu_grid_, "Mu",
                                                stats_.mu_grid_overflows);
                stats_.neighbor_offsets_count =
                    static_cast<std::uint64_t>(
                        mu_grid_->grid().neighbor_offsets_count());
                // ADDED: one-shot occupancy dump (MCPU_GRID_OCCUPANCY=1)
                // CHANGED: also requires MCPU_VERBOSE so INFO never prints by default.
                {
                    static bool printed = false;
                    static const bool kVerbose = [] {
                        const char* e = std::getenv("MCPU_VERBOSE");
                        return e && e[0] && e[0] != '0';
                    }();
                    const char* e = std::getenv("MCPU_GRID_OCCUPANCY");
                    if (kVerbose && e && e[0] == '1' && !printed) {
                        printed = true;
                        const auto& g = mu_grid_->grid();
                        int n_occ = 0, sum = 0, mx = 0;
                        const int nc = static_cast<int>(g.num_cells());
                        for (int c = 0; c < nc; ++c) {
                            const int n = g.cell_atom_count(c);
                            if (n <= 0) continue;
                            ++n_occ;
                            sum += n;
                            if (n > mx) mx = n;
                        }
                        const double avg =
                            n_occ > 0 ? static_cast<double>(sum) / n_occ : 0.0;
                        std::fprintf(stderr, // CHANGED: gated behind MCPU_VERBOSE
                            "INFO: Mu grid occupancy cell=%.3f Å dims=%dx%dx%d "
                            "n_cells=%llu occupied=%d avg_occ=%.2f max_occ=%d "
                            "peak=%d CELL_CAPACITY=%d stencil_R=%d "
                            "offsets=%zu query=%.3f\n",
                            g.cell_size(), g.nx(), g.ny(), g.nz(),
                            static_cast<unsigned long long>(g.num_cells()), n_occ,
                            avg, mx, g.peak_cell_occupancy(),
                            OpenCellGrid::CELL_CAPACITY,
                            g.stencil_radius(), g.neighbor_offsets_count(),
                            r_mu);
                        if (mx > OpenCellGrid::CELL_CAPACITY * 4 / 5) {
                            std::fprintf(stderr,
                                "WARN: max occupancy %d within 20%% of "
                                "CELL_CAPACITY=%d\n",
                                mx, OpenCellGrid::CELL_CAPACITY);
                        }
                    }
                }
            } else {
                ++stats_.num_dense_cap_fallback;
                stats_.neighbor_offsets_count = 0;
            }
        } else {
            ++stats_.num_dense_cap_fallback;
            stats_.neighbor_offsets_count = 0;
        }

        // --- HBond O / H grids (same lo/hi, smaller cell) ---
        const float hb_cell = kHBondListA;
        bool hb_ok = compute_grid_shape(b, hb_cell, cfg_, nx, ny, nz);
        if (hb_ok &&
            hb_o_grid_->configure(b, cfg_) &&
            hb_h_grid_->configure(b, cfg_)) {
            // Occupied stencil stays Off for HB (empty≈89% but wall-neutral;
            // maintenance cost not worth enabling).
            hb_o_grid_->reset(n_atoms_);
            hb_h_grid_->reset(virtual_amide_h_
                ? static_cast<int>(amide_donor_.size())
                : n_atoms_);
            for (int o : o_atom_ids_) hb_o_grid_->insert(o, coords);
            if (virtual_amide_h_) {
                insert_virtual_amide_h_(coords);
            } else {
                for (int h : h_atom_ids_) hb_h_grid_->insert(h, coords);
            }
            // One-shot HB stencil emptiness (O-grid probes at acceptor sites).
            // CHANGED: gated behind MCPU_VERBOSE — suppress in production.
            {
                static bool printed = false;
                static const bool kVerbose = [] {
                    const char* e = std::getenv("MCPU_VERBOSE");
                    return e && e[0] && e[0] != '0';
                }();
                if (kVerbose && !printed && !o_atom_ids_.empty()) {
                    printed = true;
                    std::size_t empty = 0, nonempty = 0;
                    const int nprobe = std::min(64, static_cast<int>(o_atom_ids_.size()));
                    for (int i = 0; i < nprobe; ++i) {
                        const int o = o_atom_ids_[static_cast<size_t>(i)];
                        auto s = hb_o_grid_->grid().probe_stencil_occupancy(
                            coords.x[static_cast<size_t>(o)],
                            coords.y[static_cast<size_t>(o)],
                            coords.z[static_cast<size_t>(o)]);
                        empty += s.empty;
                        nonempty += s.nonempty;
                    }
                    const double tot = static_cast<double>(empty + nonempty);
                    std::fprintf(stderr, // CHANGED: gated behind MCPU_VERBOSE
                        "INFO: HB O-grid stencil empty_frac=%.3f "
                        "(empty=%zu nonempty=%zu probes=%d) "
                        "cell=%.3f cutoff=%.3f\n",
                        tot > 0.0 ? empty / tot : 0.0, empty, nonempty, nprobe,
                        hb_cell,
                        kHBondCutoffA);
                }
            }
            hb_fallback_ = hbond_overflowed_();
        } else {
            hb_fallback_ = true;
            ++stats_.num_dense_cap_fallback;
        }

        // --- Registered subset grids ---
        for (SubsetGrid& g : subset_grids_) {
            g.active = compute_grid_shape(b, g.spec.cell_A, cfg_, nx, ny, nz) &&
                       g.grid->configure(b, cfg_, g.spec.cell_A);
            if (!g.active) continue;
            g.grid->reset(n_atoms_);
            for (int i : g.spec.members) {
                if (i >= 0 && i < n_atoms_) g.grid->insert(i, coords);
            }
            if (g.grid->grid().overflowed()) {
                g.active = false;
                retry_rebuild_ = true;
            }
        }

        return dense_active_;
    }

    /// Incremental update after Metropolis accept. Call exactly once per accept.
    void commit_accepted_move(const ProposalPatch& patch,
                              const CoordsSoA& coords_new) {
        sync_energy_mask(coords_new);
        bool left_bounds = false;
        if (bounds_.valid) {
            if (!patch.moved_indices.empty()) {
                for (int i : patch.moved_indices) {
                    const size_t k = static_cast<size_t>(i);
                    if (!point_in_bounds(coords_new.x[k], coords_new.y[k],
                                         coords_new.z[k], bounds_)) {
                        left_bounds = true;
                        break;
                    }
                }
            }
        }

        // A grid retired by an overflow is rebuilt on every accept until
        // the atoms spread out enough for it to fit.
        if (left_bounds || retry_rebuild_) {
            rebuild_from_accepted_state(coords_new);
            return;
        }

        if (!dense_active_ && !hb_fallback_) {
            rebuild_from_accepted_state(coords_new);
            return;
        }

        // Each grid that takes position updates walks the moved atoms in
        // ascending index order (the order slots are vacated and refilled in
        // decides each cell's slot order, and so every later walk's order) and
        // updates its members. Built from moved_indices: O(n_moved) per grid,
        // no scan over all atoms.
        const MovedOrder order = moved_ascending_(patch.moved_indices);
        for (size_t id = 0; id < grids_.size(); ++id) {
            CellListMC* grid = nullptr;
            const std::uint8_t* member = nullptr;
            if (!grid_takes_updates_(id, grid, member)) continue;
            order.for_each([&](int i) {
                if (member[static_cast<size_t>(i)]) grid->update_position(i, coords_new);
            });
        }

        // Virtual amide H is a derived site: donor r's H follows its own N
        // and CA and residue r-1's C. Move only the donors with a moved
        // parent, in the order the moved atoms are walked.
        if (virtual_amide_h_ && !hb_fallback_) {
            if (h_dep_.size() != static_cast<size_t>(n_atoms_)) build_h_dep_();
            int last = -1;
            order.for_each([&](int i) {
                const int r = h_dep_[static_cast<size_t>(i)];
                if (r < 0 || r == last) return;
                last = r;
                float hx, hy, hz;
                if (!virtual_amide_h_xyz_(coords_new, static_cast<size_t>(r), hx, hy, hz)) return;
                hb_h_grid_->update_position(r, Eigen::Vector3f(hx, hy, hz));
            });
        }

        // A cell that filled up during the update leaves its grid
        // incomplete: retire it (Mu takes the all-pairs delta, H-bonds the
        // brute-force search) until the next accept rebuilds it.
        if (dense_active_ &&
            note_overflow_(*mu_grid_, "Mu", stats_.mu_grid_overflows))
            dense_active_ = false;
        if (!hb_fallback_ && hbond_overflowed_()) hb_fallback_ = true;
        for (SubsetGrid& g : subset_grids_) {
            if (g.active && g.grid->grid().overflowed()) {
                g.active = false;
                retry_rebuild_ = true;
            }
        }

#if !defined(NDEBUG)
        if (dense_active_ && mu_grid_) {
            mu_grid_->grid().verify_sync();
            mu_grid_->grid().verify_packed_coords(coords_new);
            mu_grid_->grid().verify_occupied_stencil();
        }
        if (!hb_fallback_ && hb_o_grid_) {
            hb_o_grid_->grid().verify_sync();
            hb_o_grid_->grid().verify_packed_coords(coords_new);
        }
        if (!hb_fallback_ && hb_h_grid_) {
            hb_h_grid_->grid().verify_sync();
            // Virtual amide H may use synthetic coords — skip packed check then.
            if (!virtual_amide_h_)
                hb_h_grid_->grid().verify_packed_coords(coords_new);
        }
#endif
    }

    // ---- Candidate enumeration (read-only; indices = accepted state) ----

    template <typename Func>
    void for_each_hbond_acceptor_candidate(const Eigen::Vector3f& pos, Func&& func) const {
        if (hb_fallback_ || !hb_o_grid_) return;
        hb_o_grid_->for_each_neighbor(pos, [&](int j) {
            ++stats_.hbond_num_candidates_iterated;
            func(j);
        }, &stats_.neighbor_num_cell_visits);
    }

    template <typename Func>
    void for_each_hbond_acceptor_candidate(float x, float y, float z, Func&& func) const {
        if (hb_fallback_ || !hb_o_grid_) return;
        hb_o_grid_->for_each_neighbor(x, y, z, [&](int j) {
            ++stats_.hbond_num_candidates_iterated;
            func(j);
        }, &stats_.neighbor_num_cell_visits);
    }

    /// for_each_hbond_acceptor_candidate(x, y, z) without the cells that the
    /// same query from (px, py, pz) visits; see
    /// OpenCellGrid::for_each_neighbor_not_near.
    template <typename Func>
    void for_each_hbond_acceptor_candidate_not_near(float x, float y, float z,
                                                    float px, float py, float pz,
                                                    Func&& func) const {
        if (hb_fallback_ || !hb_o_grid_) return;
        hb_o_grid_->for_each_neighbor_not_near(x, y, z, px, py, pz, [&](int j) {
            ++stats_.hbond_num_candidates_iterated;
            func(j);
        }, &stats_.neighbor_num_cell_visits);
    }

    /// The H-grid counterpart of for_each_hbond_acceptor_candidate_not_near.
    template <typename Func>
    void for_each_hbond_h_candidate_not_near(float x, float y, float z,
                                             float px, float py, float pz,
                                             Func&& func) const {
        if (hb_fallback_ || !hb_h_grid_) return;
        hb_h_grid_->for_each_neighbor_not_near(x, y, z, px, py, pz, [&](int j) {
            ++stats_.hbond_num_candidates_iterated;
            func(j);
        }, &stats_.neighbor_num_cell_visits);
    }

    template <typename Func>
    void for_each_hbond_h_candidate(const Eigen::Vector3f& pos, Func&& func) const {
        if (hb_fallback_ || !hb_h_grid_) return;
        hb_h_grid_->for_each_neighbor(pos, [&](int j) {
            ++stats_.hbond_num_candidates_iterated;
            func(j);
        }, &stats_.neighbor_num_cell_visits);
    }

    template <typename Func>
    void for_each_hbond_h_candidate(float x, float y, float z, Func&& func) const {
        if (hb_fallback_ || !hb_h_grid_) return;
        hb_h_grid_->for_each_neighbor(x, y, z, [&](int j) {
            ++stats_.hbond_num_candidates_iterated;
            func(j);
        }, &stats_.neighbor_num_cell_visits);
    }

    /// Brute O/H scan when HBond dense grid is capped (physics unchanged).
    template <typename Func>
    void for_each_hbond_acceptor_bruteforce(const CoordsSoA& coords,
                                            float px, float py, float pz,
                                            float cutoff_sq,
                                            Func&& func) const {
        for (int i : o_atom_ids_) {
            const size_t k = static_cast<size_t>(i);
            const float dx = coords.x[k] - px;
            const float dy = coords.y[k] - py;
            const float dz = coords.z[k] - pz;
            if (pair_r2(dx, dy, dz) <= cutoff_sq) func(i);
        }
    }

    template <typename Func>
    void for_each_hbond_h_bruteforce(const CoordsSoA& coords,
                                     float px, float py, float pz,
                                     float cutoff_sq,
                                     Func&& func) const {
        for (int i : h_atom_ids_) {
            const size_t k = static_cast<size_t>(i);
            const float dx = coords.x[k] - px;
            const float dy = coords.y[k] - py;
            const float dz = coords.z[k] - pz;
            if (pair_r2(dx, dy, dz) <= cutoff_sq) func(i);
        }
    }

    /// Virtual-amide mode: brute-force donor residues by computing H on the fly.
    /// Callback receives residue index (r_don).
    template <typename Func>
    void for_each_hbond_donor_bruteforce(const CoordsSoA& coords,
                                         const System& /*sys*/,
                                         float px, float py, float pz,
                                         float cutoff_sq,
                                         Func&& func) const {
        for (size_t r = 0; r < amide_donor_.size(); ++r) {
            if (!amide_donor_[r] || r == 0) continue;
            const int bb = donor_bb_starts_[r];
            const int prev_c = donor_c_starts_[r - 1];
            if (bb < 0 || prev_c < 0) continue;
            float hx, hy, hz;
            HydrogenBondUtils::compute_virtual_amide_H(
                coords.x[static_cast<size_t>(bb)],
                coords.y[static_cast<size_t>(bb)],
                coords.z[static_cast<size_t>(bb)],
                coords.x[static_cast<size_t>(bb + 1)],
                coords.y[static_cast<size_t>(bb + 1)],
                coords.z[static_cast<size_t>(bb + 1)],
                coords.x[static_cast<size_t>(prev_c)],
                coords.y[static_cast<size_t>(prev_c)],
                coords.z[static_cast<size_t>(prev_c)],
                hx, hy, hz);
            const float dx = hx - px, dy = hy - py, dz = hz - pz;
            if (pair_r2(dx, dy, dz) <= cutoff_sq) {
                func(static_cast<int>(r));
            }
        }
    }

    /// Virtual amide H for residue r (requires r>=1 and prev donor_bb).
    bool virtual_amide_h_xyz_(const CoordsSoA& coords, size_t r,
                              float& hx, float& hy, float& hz) const {
        if (r == 0 || r >= amide_donor_.size() || !amide_donor_[r]) return false;
        const int bb = donor_bb_starts_[r];
        const int prev_c = donor_c_starts_[r - 1];
        if (bb < 0 || prev_c < 0) return false;
        HydrogenBondUtils::compute_virtual_amide_H(
            coords.x[static_cast<size_t>(bb)],
            coords.y[static_cast<size_t>(bb)],
            coords.z[static_cast<size_t>(bb)],
            coords.x[static_cast<size_t>(bb + 1)],
            coords.y[static_cast<size_t>(bb + 1)],
            coords.z[static_cast<size_t>(bb + 1)],
            coords.x[static_cast<size_t>(prev_c)],
            coords.y[static_cast<size_t>(prev_c)],
            coords.z[static_cast<size_t>(prev_c)],
            hx, hy, hz);
        return true;
    }

    /// Compare HBond grid candidates to brute-force within kHBondCutoffA.
    /// Returns number of query centers with mismatched neighbor sets (0 = OK).
    std::uint64_t count_hbond_candidate_mismatches(const CoordsSoA& coords) const {
        if (hb_fallback_) return 0;
        const float cut2 = kHBondCutoffA * kHBondCutoffA;
        std::uint64_t mismatches = 0;

        auto collect_sorted = [](std::vector<int>& ids) {
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        };

        if (virtual_amide_h_) {
            for (size_t r = 0; r < amide_donor_.size(); ++r) {
                float hx, hy, hz;
                if (!virtual_amide_h_xyz_(coords, r, hx, hy, hz)) continue;
                std::vector<int> grid_ids, brute_ids;
                for_each_hbond_acceptor_candidate(hx, hy, hz, [&](int j) {
                    const size_t jk = static_cast<size_t>(j);
                    const float dx = coords.x[jk] - hx;
                    const float dy = coords.y[jk] - hy;
                    const float dz = coords.z[jk] - hz;
                    if (pair_r2(dx, dy, dz) <= cut2) grid_ids.push_back(j);
                });
                for_each_hbond_acceptor_bruteforce(coords, hx, hy, hz, cut2, [&](int j) {
                    brute_ids.push_back(j);
                });
                collect_sorted(grid_ids);
                collect_sorted(brute_ids);
                if (grid_ids != brute_ids) ++mismatches;
            }
            for (int o : o_atom_ids_) {
                const size_t ok = static_cast<size_t>(o);
                const float ox = coords.x[ok], oy = coords.y[ok], oz = coords.z[ok];
                std::vector<int> grid_ids, brute_ids;
                for_each_hbond_h_candidate(ox, oy, oz, [&](int j) {
                    if (j < 0 || static_cast<size_t>(j) >= amide_donor_.size()) return;
                    float hx, hy, hz;
                    if (!virtual_amide_h_xyz_(coords, static_cast<size_t>(j), hx, hy, hz)) return;
                    const float dx = hx - ox, dy = hy - oy, dz = hz - oz;
                    if (pair_r2(dx, dy, dz) <= cut2) grid_ids.push_back(j);
                });
                // Brute: residue ids within cutoff of virtual H
                for (size_t r = 0; r < amide_donor_.size(); ++r) {
                    float hx, hy, hz;
                    if (!virtual_amide_h_xyz_(coords, r, hx, hy, hz)) continue;
                    const float dx = hx - ox, dy = hy - oy, dz = hz - oz;
                    if (pair_r2(dx, dy, dz) <= cut2) {
                        brute_ids.push_back(static_cast<int>(r));
                    }
                }
                collect_sorted(grid_ids);
                collect_sorted(brute_ids);
                if (grid_ids != brute_ids) ++mismatches;
            }
            return mismatches;
        }

        for (int h : h_atom_ids_) {
            const size_t hk = static_cast<size_t>(h);
            const float hx = coords.x[hk], hy = coords.y[hk], hz = coords.z[hk];
            std::vector<int> grid_ids, brute_ids;
            for_each_hbond_acceptor_candidate(hx, hy, hz, [&](int j) {
                const size_t jk = static_cast<size_t>(j);
                const float dx = coords.x[jk] - hx;
                const float dy = coords.y[jk] - hy;
                const float dz = coords.z[jk] - hz;
                if (pair_r2(dx, dy, dz) <= cut2) grid_ids.push_back(j);
            });
            for_each_hbond_acceptor_bruteforce(coords, hx, hy, hz, cut2, [&](int j) {
                brute_ids.push_back(j);
            });
            collect_sorted(grid_ids);
            collect_sorted(brute_ids);
            if (grid_ids != brute_ids) ++mismatches;
        }
        for (int o : o_atom_ids_) {
            const size_t ok = static_cast<size_t>(o);
            const float ox = coords.x[ok], oy = coords.y[ok], oz = coords.z[ok];
            std::vector<int> grid_ids, brute_ids;
            for_each_hbond_h_candidate(ox, oy, oz, [&](int j) {
                const size_t jk = static_cast<size_t>(j);
                const float dx = coords.x[jk] - ox;
                const float dy = coords.y[jk] - oy;
                const float dz = coords.z[jk] - oz;
                if (pair_r2(dx, dy, dz) <= cut2) grid_ids.push_back(j);
            });
            for_each_hbond_h_bruteforce(coords, ox, oy, oz, cut2, [&](int j) {
                brute_ids.push_back(j);
            });
            collect_sorted(grid_ids);
            collect_sorted(brute_ids);
            if (grid_ids != brute_ids) ++mismatches;
        }
        return mismatches;
    }

private:
    /// h_dep_[i] = the donor whose virtual H atom i places (its N, its CA,
    /// or the previous residue's C), else -1. Each atom places at most one.
    void build_h_dep_() {
        h_dep_.assign(static_cast<size_t>(n_atoms_), -1);
        auto set = [&](int atom, int r) {
            if (atom >= 0 && atom < n_atoms_) h_dep_[static_cast<size_t>(atom)] = r;
        };
        for (size_t r = 1; r < amide_donor_.size(); ++r) {
            if (!amide_donor_[r]) continue;
            const int bb = donor_bb_starts_[r];
            const int prev_c = donor_c_starts_[r - 1];
            if (bb < 0 || prev_c < 0) continue;
            set(bb, static_cast<int>(r));
            set(bb + 1, static_cast<int>(r));
            set(prev_c, static_cast<int>(r));
        }
    }

    void insert_virtual_amide_h_(const CoordsSoA& coords) {
        for (size_t r = 0; r < amide_donor_.size(); ++r) {
            float hx, hy, hz;
            if (!virtual_amide_h_xyz_(coords, r, hx, hy, hz)) continue;
            hb_h_grid_->insert(static_cast<int>(r), hx, hy, hz);
        }
    }

    /// Iterates a moved list in ascending order without sorting it when it
    /// is already ascending or descending (the usual cases).
    struct MovedOrder {
        const int* p;
        size_t n;
        int dir;  // +1 ascending, -1 descending
        template <class F>
        void for_each(F&& f) const {
            if (dir > 0) {
                for (size_t k = 0; k < n; ++k) f(p[k]);
            } else {
                for (size_t k = n; k-- > 0;) f(p[k]);
            }
        }
    };
    MovedOrder moved_ascending_(const std::vector<int>& moved) {
        const size_t n = moved.size();
        if (std::is_sorted(moved.begin(), moved.end())) return {moved.data(), n, +1};
        if (std::is_sorted(moved.rbegin(), moved.rend())) return {moved.data(), n, -1};
        commit_order_.assign(moved.begin(), moved.end());
        std::sort(commit_order_.begin(), commit_order_.end());
        return {commit_order_.data(), n, +1};
    }

    /// Whether registered grid `id` takes position updates now, and if so
    /// which grid and member mask. Built-ins follow the current mode flags.
    bool grid_takes_updates_(size_t id, CellListMC*& grid, const std::uint8_t*& member) {
        const RegisteredGrid& r = grids_[id];
        if (r.subset >= 0) {
            SubsetGrid& g = subset_grids_[static_cast<size_t>(r.subset)];
            if (!g.active) return false;
            grid = g.grid.get();
            member = g.member.data();
            return true;
        }
        switch (static_cast<neighbor::GridId>(id)) {
            case neighbor::kMuGrid:     if (!dense_active_) return false; break;
            case neighbor::kHBondOGrid: if (hb_fallback_) return false; break;
            case neighbor::kHBondHGrid: if (hb_fallback_ || virtual_amide_h_) return false; break;
            default: return false;
        }
        grid = (this->*r.grid).get();
        member = (this->*r.member).data();
        return true;
    }

    /// One entry per registered grid, indexed by GridId. Built-ins point at
    /// their fields (member pointers, so a moved NeighborSystem stays valid);
    /// subset grids index subset_grids_.
    struct RegisteredGrid {
        const char* name;
        std::unique_ptr<CellListMC> NeighborSystem::* grid;
        std::vector<uint8_t> NeighborSystem::* member;
        int subset;
    };
    struct SubsetGrid {
        GridSpec spec;
        std::vector<uint8_t> member;
        std::unique_ptr<CellListMC> grid;
        bool active = false;
    };
    /// (Re)build g's membership mask and empty grid for n_atoms_. O(N).
    void seed_subset_grid(SubsetGrid& g) const {
        g.member.assign(static_cast<size_t>(n_atoms_), 0);
        for (int i : g.spec.members) {
            if (i >= 0 && i < n_atoms_) g.member[static_cast<size_t>(i)] = 1;
        }
        g.grid = std::make_unique<CellListMC>(g.spec.max_cutoff_A, n_atoms_);
        g.grid->ensure_atom_capacity(n_atoms_);
        g.active = false;
    }
    /// in_mu_ = in_mu_base_ less the atoms of residues an ignore_all energy
    /// mask switches off. Every pair with such an atom scores 0 and cannot
    /// clash (MuPotential::mask_ignores_pair), so leaving them out of the
    /// grid drops only zero terms and keeps the order of the rest: deltas
    /// are the same to the bit. It also keeps the hard-core bound on cell
    /// occupancy, which masked atoms, free to overlap, do not obey.
    void apply_energy_mask_() {
        mask_epoch_ = sys_->energy_mask_epoch();
        in_mu_ = in_mu_base_;
        if (!sys_->has_energy_mask() ||
            sys_->energy_mask_mode() != EnergyMaskMode::IgnoreAll)
            return;
        const std::vector<uint8_t>& mask = sys_->energy_ignored_mask();
        const std::vector<int>& res = sys_->atom_to_residue;
        for (int i = 0; i < n_atoms_ && static_cast<size_t>(i) < res.size(); ++i) {
            const int r = res[static_cast<size_t>(i)];
            if (r >= 0 && static_cast<size_t>(r) < mask.size() &&
                mask[static_cast<size_t>(r)])
                in_mu_[static_cast<size_t>(i)] = 0;
        }
    }

    /// True (and counted, with one MCPU_VERBOSE warning per process) when
    /// ``g`` had to leave an atom out because its cell was full. The caller
    /// then retires the grid: the exact fallbacks for an inactive grid take
    /// over, and the next accepted move rebuilds it. CELL_CAPACITY stays a
    /// compile-time constant so the hot walks keep their fixed strides; the
    /// hard core keeps real occupancy near half of it (peaks of 20-24 on
    /// actin and PGK1), so this is a guard, not a path runs live on.
    bool note_overflow_(const CellListMC& g, const char* name,
                        std::uint64_t& counter) {
        if (!g.grid().overflowed()) return false;
        ++counter;
        retry_rebuild_ = true;
        static bool warned = false;
        static const bool kVerbose = [] {
            const char* e = std::getenv("MCPU_VERBOSE");
            return e && e[0] && e[0] != '0';
        }();
        if (kVerbose && !warned) {
            warned = true;
            std::fprintf(stderr,
                "WARN: a %s grid cell is over its capacity of %d atoms; the "
                "grid is off (exact fallback) until an accepted move "
                "rebuilds it.\n",
                name, OpenCellGrid::CELL_CAPACITY);
        }
        return true;
    }
    /// Either H-bond grid overflowed (both are checked and counted).
    bool hbond_overflowed_() {
        const bool o = note_overflow_(*hb_o_grid_, "H-bond O",
                                      stats_.hbond_grid_overflows);
        const bool h = note_overflow_(*hb_h_grid_, "H-bond H",
                                      stats_.hbond_grid_overflows);
        return o || h;
    }

    std::vector<RegisteredGrid> grids_;
    std::vector<SubsetGrid> subset_grids_;
    std::vector<int> commit_order_;

    NeighborConfig cfg_{};
    mutable NeighborStats stats_{};
    BoxBounds bounds_{};
    bool dense_active_ = false;
    bool hb_fallback_ = false;
    bool retry_rebuild_ = false;  ///< a grid overflowed: rebuild on the next accept
    bool virtual_amide_h_ = false;
    bool hb_h_ids_are_residues_ = false;

    int n_atoms_ = 0;
    int n_bb_ = 0, n_o_ = 0, n_sc_ = 0, n_h_ = 0;
    int h_begin_ = 0, o_begin_ = 0, o_end_ = 0;
    bool residue_contiguous_ = false;

    std::vector<int> donor_bb_starts_;
    std::vector<int> donor_c_starts_;
    std::vector<uint8_t> amide_donor_;
    std::vector<int> h_dep_;
    std::vector<uint8_t> in_mu_;
    std::vector<uint8_t> in_mu_base_;  ///< every non-amide-H atom
    const System* sys_ = nullptr;
    std::uint64_t mask_epoch_ = 0;
    std::vector<uint8_t> is_o_;
    std::vector<uint8_t> is_h_;
    std::vector<int> o_atom_ids_;
    std::vector<int> h_atom_ids_;

    std::unique_ptr<CellListMC> mu_grid_;
    std::unique_ptr<CellListMC> hb_o_grid_;
    std::unique_ptr<CellListMC> hb_h_grid_;
};

} // namespace mcpu
