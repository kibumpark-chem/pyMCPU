#pragma once
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
#include <utility>
#include <vector>

#include "pymcpu/CellList.h"
#include "pymcpu/ProposalPatch.h"
#include "pymcpu/System.h"
#include "pymcpu/neighbor/NeighborConfig.h"
#include "pymcpu/neighbor/OpenCellGrid.h"
#include "pymcpu/neighbor/VerletList.h"
#include "pymcpu/utils/CoordsSoA.h"
#include "pymcpu/utils/CoordView.h"
#include "pymcpu/utils/virtual_amide_h.h"

namespace mcpu {

class NeighborSystem {
public:
    static constexpr float kMuCutoffFallbackA = 6.0f;
    static constexpr float kHBondCutoffA = 2.5f;

    /// Active Mu denselist cutoff (Å): from MuPotential (exact) or fallback 6.0.
    [[nodiscard]] float mu_cutoff_A() const noexcept {
        return (cfg_.mu_denselist_cutoff_A > 0.f) ? cfg_.mu_denselist_cutoff_A
                                                    : kMuCutoffFallbackA;
    }

    NeighborSystem() = default;

    void init(const System& sys, NeighborConfig cfg = {}) {
        apply_neighbor_env_overrides(cfg);
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
        // be absent from incremental cell/Verlet candidates.
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

        mu_grid_ = std::make_unique<CellListMC>(kMuCutoffFallbackA, n_atoms_);
        hb_o_grid_ = std::make_unique<CellListMC>(kHBondCutoffA, n_atoms_);
        // Virtual-H grid uses residue ids as keys (capacity ≥ n_res); explicit H uses atom ids.
        const int hb_h_cap = virtual_amide_h_
            ? std::max(n_atoms_, sys.getNumResidues())
            : n_atoms_;
        hb_h_grid_ = std::make_unique<CellListMC>(kHBondCutoffA, hb_h_cap);
        mu_grid_->ensure_atom_capacity(n_atoms_);
        hb_o_grid_->ensure_atom_capacity(n_atoms_);
        hb_h_grid_->ensure_atom_capacity(hb_h_cap);
        dense_active_ = true;
        hb_fallback_ = false;
    }

    NeighborConfig& config() noexcept { return cfg_; }
    const NeighborConfig& config() const noexcept { return cfg_; }
    NeighborStats& stats() const noexcept { return stats_; }
    const BoxBounds& bounds() const noexcept { return bounds_; }
    bool denseActive() const noexcept { return dense_active_; }
    bool hbondUsesFallback() const noexcept { return hb_fallback_; }
    int hBegin() const noexcept { return h_begin_; }
    bool virtualAmideH() const noexcept { return virtual_amide_h_; }
    bool hbondHIdIsResidue() const noexcept { return hb_h_ids_are_residues_; }

    /// One-shot audit label for HBond candidate source (current NeighborSystem).
    const char* hbond_backend_name() const noexcept {
        return hb_fallback_ ? "bruteforce_OH_cap_fallback" : "opencell_typed_OH_grids";
    }
    float hbond_cutoff_A() const noexcept { return kHBondCutoffA; }
    float hbond_cell_size_A() const noexcept {
        return hb_fallback_ ? 0.f : kHBondCutoffA; // skin_hb = 0
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

    VerletList& muVerlet() noexcept { return mu_verlet_; }
    const VerletList& muVerlet() const noexcept { return mu_verlet_; }

    /// Read-only Mu index (BB+O+SC). For moved_new_grid bounds / Verlet rebuild.
    const CellListMC& muGrid() const { return *mu_grid_; }

    bool trial_in_bounds(const CoordsSoA& trial_coords,
                         const ProposalPatch& patch) const {
        if (!bounds_.valid) return true;
        if (!patch.moved_indices.empty()) {
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
            const float d2 = dx * dx + dy * dy + dz * dz;
            if (d2 > d2max) d2max = d2;
        };
        if (!patch.moved_indices.empty()) {
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
        // AABB margin tracks r_cut+skin (Verlet validity / AutoExpand headroom).
        // Denselist cell + query stay at mu_cutoff (legacy 6 Å or exact contact).
        const float r_mu = mu_cutoff_A();
        const float mu_query_list = r_mu + cfg_.skin;
        const float mu_cell = effective_mu_cell_size_A(r_mu, /*skin=*/0.f, cfg_);
        const float margin = cfg_.effective_margin(mu_query_list);
        BoxBounds b = aabb_of_coords(coords, margin);
        bounds_ = b;

        // --- Mu grid ---
        int nx = 0, ny = 0, nz = 0;
        NeighborConfig cfg_mu = cfg_;
        bool mu_ok = compute_grid_shape(b, mu_cell, cfg_mu, nx, ny, nz);
        dense_active_ = false;
        hb_fallback_ = false;

        if (mu_ok) {
            mu_grid_->set_cutoff(r_mu); // CHANGED: denselist query matches cell cut
            if (mu_grid_->configure(b, cfg_mu, /*skin=*/0.f, mu_cell)) {
                mu_grid_->reset(n_atoms_);
                for (int i = 0; i < n_atoms_; ++i) {
                    if (in_mu_[static_cast<size_t>(i)]) mu_grid_->insert(i, coords);
                }
                // Bulk insert then enable Mu-only occupied stencil (default Full).
                // HB/scratch grids stay Off (never call enable_occupied_stencil).
                mu_grid_->grid().enable_occupied_stencil(
                    OpenCellGrid::occupied_mode_from_env());
                dense_active_ = true;
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
                            "peak=%d contiguous=%d CELL_CAPACITY=%d stencil_R=%d "
                            "offsets=%zu query=%.3f\n",
                            g.cell_size(), g.nx(), g.ny(), g.nz(),
                            static_cast<unsigned long long>(g.num_cells()), n_occ,
                            avg, mx, g.peak_cell_occupancy(),
                            g.use_contiguous() ? 1 : 0, OpenCellGrid::CELL_CAPACITY,
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
                mu_verlet_.invalidate();
                stats_.neighbor_offsets_count = 0;
            }
        } else {
            ++stats_.num_dense_cap_fallback;
            mu_verlet_.invalidate();
            stats_.neighbor_offsets_count = 0;
        }

        // --- HBond O / H grids (same lo/hi, smaller cell) ---
        NeighborConfig cfg_hb = cfg_;
        cfg_hb.skin = 0.f;
        const float hb_cell = kHBondCutoffA;
        bool hb_ok = compute_grid_shape(b, hb_cell, cfg_hb, nx, ny, nz);
        if (hb_ok &&
            hb_o_grid_->configure(b, cfg_hb, 0.f) &&
            hb_h_grid_->configure(b, cfg_hb, 0.f)) {
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
                        "contig=%d cell=%.3f cutoff=%.3f\n",
                        tot > 0.0 ? empty / tot : 0.0, empty, nonempty, nprobe,
                        hb_o_grid_->grid().use_contiguous() ? 1 : 0, hb_cell,
                        kHBondCutoffA);
                }
            }
            hb_fallback_ = false;
        } else {
            hb_fallback_ = true;
            ++stats_.num_dense_cap_fallback;
        }

        maybe_rebuild_mu_verlet(coords);
        return dense_active_;
    }

    /// Incremental update after Metropolis accept. Call exactly once per accept.
    void commit_accepted_move(const ProposalPatch& patch,
                              const CoordsSoA& coords_old,
                              const CoordsSoA& coords_new,
                              MoveKind move_kind,
                              bool is_rigid) {
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

        if (left_bounds) {
            rebuild_from_accepted_state(coords_new);
            if (move_kind == MoveKind::Pivot || is_rigid) {
                mu_verlet_.invalidate(VerletList::DirtyCause::AutoExpand);
                ++stats_.num_verlet_invalidate_pivot_accept;
                // Actual CSR rebuild attributed later in maybe_rebuild_mu_verlet.
            } else {
                mu_verlet_.invalidate(VerletList::DirtyCause::AutoExpand);
            }
            return;
        }

        if (!dense_active_ && !hb_fallback_) {
            rebuild_from_accepted_state(coords_new);
            return;
        }

        for (size_t a = 0; a < patch.moving_atoms.size(); ++a) {
            if (!patch.moving_atoms[a]) continue;
            const int i = static_cast<int>(a);
            if (dense_active_ && in_mu_[static_cast<size_t>(i)]) {
                mu_grid_->update_position(i, coords_new);
            }
            if (!hb_fallback_) {
                if (is_o_[static_cast<size_t>(i)]) hb_o_grid_->update_position(i, coords_new);
                if (!virtual_amide_h_ && is_h_[static_cast<size_t>(i)]) {
                    hb_h_grid_->update_position(i, coords_new);
                }
            }
        }

        // Virtual amide H tracks backbone; refresh donor entries when BB moved.
        if (virtual_amide_h_ && !hb_fallback_) {
            bool any_bb = false;
            for (uint8_t v : patch.bb_atom_moved) {
                if (v) { any_bb = true; break; }
            }
            if (any_bb) {
                hb_h_grid_->reset(static_cast<int>(amide_donor_.size()));
                insert_virtual_amide_h_(coords_new);
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

        // Pivot: default is displacement tracking (Context::commit_accepted_move).
        // Legacy invalidate_verlet_on_pivot_accept still force-dirties.
        if ((move_kind == MoveKind::Pivot || is_rigid) &&
            cfg_.invalidate_verlet_on_pivot_accept) {
            invalidate_mu_verlet_pivot();
        }

        // Strategy B': accumulate always; partial rebuild only when skin exceeded.
        // Rebuilding on every accept was ~180 µs/step (full CSR pack) — too costly.
        last_commit_did_partial_verlet_ = false;
        if (cfg_.skin > 0.f && cfg_.mu_verlet_enabled && dense_active_ &&
            !(move_kind == MoveKind::Pivot || is_rigid) &&
            !mu_verlet_.dirty) {
            const int n_moved = static_cast<int>(patch.moved_indices.size());
            if (n_moved > 0 && n_moved <= cfg_.verlet_partial_threshold) {
                mu_verlet_.accumulate_accept(patch.moved_indices, coords_old,
                                             coords_new);
                if (mu_verlet_.dirty &&
                    mu_verlet_.dirty_cause ==
                        VerletList::DirtyCause::DispExceeded) {
                    partial_rebuild_verlet(patch.moved_indices, coords_old,
                                           coords_new);
                    last_commit_did_partial_verlet_ = true;
                }
            }
        }
    }

    /// True if last commit_accepted_move ran a partial Verlet rebuild. O(1).
    [[nodiscard]] bool last_commit_did_partial_verlet() const noexcept {
        return last_commit_did_partial_verlet_;
    }

    /// Partial Verlet rebuild after small-move accept -- currently a full rebuild.
    ///
    /// The selective "only touch affected rows" optimization this function's name
    /// promises was never committed (see rebuild_verlet_undirected below); this
    /// always does a full CSR rebuild instead. Kept as its own function so the
    /// call site (a small-move accept) can still be distinguished from other
    /// rebuild triggers via stats_.num_verlet_partial_rebuilds, but it delivers no
    /// selective-rebuild speedup today. stats_.num_verlet_partial_affected_sum is
    /// intentionally left at 0 by this path (there is no "affected atom set" to
    /// report while it's a full rebuild) rather than incrementing it with a value
    /// that would misrepresent what actually happened.
    void partial_rebuild_verlet(
        const std::vector<int>& /*moved_indices*/,
        const CoordsSoA& /*coords_old_moved*/,
        const CoordsSoA& coords_new)
    {
        const float r_list = mu_cutoff_A() + cfg_.skin;

        OpenCellGrid& g = mu_grid_->grid();
        const float q_dense = g.query_radius();
        const bool widen = r_list > q_dense + 1e-4f;
        if (widen) g.set_query_radius(r_list);

        rebuild_verlet_undirected(
            mu_verlet_, *mu_grid_, coords_new, mu_cutoff_A(), cfg_.skin,
            residue_contiguous_ ? -1 : h_begin_);

        if (widen) g.set_query_radius(q_dense);

        ++stats_.num_verlet_partial_rebuilds;
        const std::uint64_t directed =
            static_cast<std::uint64_t>(mu_verlet_.neighbors.size());
        stats_.verlet_edges_total = directed / 2ull;
    }

    void invalidate_mu_verlet_pivot() {
        mu_verlet_.invalidate(VerletList::DirtyCause::PivotAccept);
        ++stats_.num_verlet_invalidate_pivot_accept;
    }

    bool muVerletEnabled() const noexcept { return cfg_.mu_verlet_enabled; }

    void maybe_rebuild_mu_verlet(const CoordsSoA& coords) {
        if (cfg_.skin <= 0.f || !dense_active_ || !cfg_.mu_verlet_enabled) {
            if (cfg_.skin <= 0.f || !cfg_.mu_verlet_enabled) {
                mu_verlet_.invalidate(VerletList::DirtyCause::DirtyFlag);
            }
            return;
        }
        const auto cause = mu_verlet_.dirty_cause;
        const std::uint64_t realloc_n0 = mu_verlet_.num_neigh_reallocs;
        const std::uint64_t realloc_o0 = mu_verlet_.num_offsets_reallocs;
        // Widen denselist stencil to r_cut+skin for the list build only, then
        // restore r_cut so Pivot CellOnly keeps the tight 27-cell neighborhood.
        OpenCellGrid& g = mu_grid_->grid();
        const float q_dense = g.query_radius();
        const float q_list = mu_cutoff_A() + cfg_.skin;
#if !defined(NDEBUG)
        assert(q_list + 1e-4f >= mu_cutoff_A() + cfg_.skin);
#endif
        const bool widen = q_list > q_dense + 1e-4f;
        if (widen) g.set_query_radius(q_list);
        // r_list = r_mu + skin; only BB+O+SC (Mu occupancy).
        rebuild_verlet_undirected(mu_verlet_, *mu_grid_, coords, mu_cutoff_A(), cfg_.skin,
                                  residue_contiguous_ ? -1 : h_begin_);
        if (widen) g.set_query_radius(q_dense);
        ++stats_.num_verlet_rebuilds;
        stats_.num_verlet_neigh_reallocs +=
            (mu_verlet_.num_neigh_reallocs - realloc_n0);
        stats_.num_verlet_offsets_reallocs +=
            (mu_verlet_.num_offsets_reallocs - realloc_o0);
        switch (cause) {
            case VerletList::DirtyCause::PivotAccept:
                ++stats_.num_verlet_rebuild_due_to_pivot_accept;
                break;
            case VerletList::DirtyCause::DispExceeded:
                ++stats_.num_verlet_rebuild_due_to_disp_acc_exceeded;
                break;
            case VerletList::DirtyCause::AutoExpand:
                ++stats_.num_verlet_rebuild_due_to_autoexpand_accept;
                break;
            case VerletList::DirtyCause::None:
            case VerletList::DirtyCause::DirtyFlag:
            default:
                ++stats_.num_verlet_rebuild_due_to_dirty_flag;
                break;
        }
        // Undirected CSR stores each edge twice.
        const std::uint64_t directed = static_cast<std::uint64_t>(mu_verlet_.neighbors.size());
        stats_.verlet_edges_total = directed / 2ull;
        const int n_mu = residue_contiguous_ ? n_atoms_ : std::max(0, h_begin_);
        const int denom = n_mu > 0 ? n_mu : 1;
        stats_.verlet_avg_degree =
            static_cast<double>(directed) / static_cast<double>(denom);
    }

    // ---- Candidate enumeration (read-only; indices = accepted state) ----

    template <typename Func>
    void for_each_mu_candidate(const Eigen::Vector3f& pos, Func&& func) const {
        for_each_mu_candidate(pos.x(), pos.y(), pos.z(), std::forward<Func>(func));
    }

    template <typename Func>
    void for_each_mu_candidate(float x, float y, float z, Func&& func) const {
        if (!dense_active_) return;
        const float r_cut = mu_cutoff_A();
        const float r2 = r_cut * r_cut;
        mu_grid_->for_each_neighbor(
            x, y, z,
            [&](int j) {
                ++stats_.mu_num_candidates_iterated;
                func(j);
            },
            &stats_.neighbor_num_cell_visits, r2);
    }

    template <typename Func>
    bool for_each_mu_candidate_while(const Eigen::Vector3f& pos, Func&& func) const {
        return for_each_mu_candidate_while(pos.x(), pos.y(), pos.z(),
                                           std::forward<Func>(func));
    }

    template <typename Func>
    bool for_each_mu_candidate_while(float x, float y, float z, Func&& func) const {
        if (!dense_active_) return true;
        const float r_cut = mu_cutoff_A();
        const float r2 = r_cut * r_cut;
        return mu_grid_->for_each_neighbor_while(
            x, y, z,
            [&](int j) {
                ++stats_.mu_num_candidates_iterated;
                return func(j);
            },
            &stats_.neighbor_num_cell_visits, r2);
    }


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
            if (dx * dx + dy * dy + dz * dz <= cutoff_sq) func(i);
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
            if (dx * dx + dy * dy + dz * dz <= cutoff_sq) func(i);
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
            if (dx * dx + dy * dy + dz * dz <= cutoff_sq) {
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
                    if (dx * dx + dy * dy + dz * dz <= cut2) grid_ids.push_back(j);
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
                    if (dx * dx + dy * dy + dz * dz <= cut2) grid_ids.push_back(j);
                });
                // Brute: residue ids within cutoff of virtual H
                for (size_t r = 0; r < amide_donor_.size(); ++r) {
                    float hx, hy, hz;
                    if (!virtual_amide_h_xyz_(coords, r, hx, hy, hz)) continue;
                    const float dx = hx - ox, dy = hy - oy, dz = hz - oz;
                    if (dx * dx + dy * dy + dz * dz <= cut2) {
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
                if (dx * dx + dy * dy + dz * dz <= cut2) grid_ids.push_back(j);
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
                if (dx * dx + dy * dy + dz * dz <= cut2) grid_ids.push_back(j);
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
    void insert_virtual_amide_h_(const CoordsSoA& coords) {
        for (size_t r = 0; r < amide_donor_.size(); ++r) {
            float hx, hy, hz;
            if (!virtual_amide_h_xyz_(coords, r, hx, hy, hz)) continue;
            hb_h_grid_->insert(static_cast<int>(r), hx, hy, hz);
        }
    }

    NeighborConfig cfg_{};
    mutable NeighborStats stats_{};
    BoxBounds bounds_{};
    bool dense_active_ = false;
    bool hb_fallback_ = false;
    bool virtual_amide_h_ = false;
    bool hb_h_ids_are_residues_ = false;

    int n_atoms_ = 0;
    int n_bb_ = 0, n_o_ = 0, n_sc_ = 0, n_h_ = 0;
    int h_begin_ = 0, o_begin_ = 0, o_end_ = 0;
    bool residue_contiguous_ = false;

    std::vector<int> donor_bb_starts_;
    std::vector<int> donor_c_starts_;
    std::vector<uint8_t> amide_donor_;
    std::vector<uint8_t> in_mu_;
    std::vector<uint8_t> is_o_;
    std::vector<uint8_t> is_h_;
    std::vector<int> o_atom_ids_;
    std::vector<int> h_atom_ids_;

    std::unique_ptr<CellListMC> mu_grid_;
    std::unique_ptr<CellListMC> hb_o_grid_;
    std::unique_ptr<CellListMC> hb_h_grid_;
    VerletList mu_verlet_;

    bool last_commit_did_partial_verlet_ = false;
};

} // namespace mcpu
