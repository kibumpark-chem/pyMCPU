#include "pymcpu/forces/mcpu/common/HydrogenBondPotential.h"
#include "pymcpu/Context.h"
#include "pymcpu/State.h"
#include "pymcpu/System.h"
#include "pymcpu/utils/CoordView.h"
#include "pymcpu/utils/hydrogen_bond_utils.h"
#include "pymcpu/utils/virtual_amide_h.h"
#include "pymcpu/forces/mcpu/common/HBondStateCache.h"
#include <array>
#include <cmath>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace mcpu::forces {

namespace {

// A donor/acceptor pair whose H and O are at least HBOND_CUTOFF apart in both
// states scores 0 in both, so its term in the delta is exactly +0.0f and it
// can be dropped without changing the sum. The filters below drop a pair only
// when both distances clear this slightly larger cutoff, which leaves room for
// the last-bit differences between their arithmetic and the evaluator's.
constexpr float kFarCut2 = 2.55f * 2.55f;
static_assert(kFarCut2 > HBOND_CUTOFF_SQUARED, "far filter must be looser than the H-bond cutoff");

/// Load donor amide H (explicit atom or legacy virtual) into hpos[3].
inline bool load_donor_h(
    const State& state,
    const System& sys,
    int r_don,
    float* hpos) noexcept
{
    const auto& blocks = sys.getBlockIndices();
    const BlockIndices& b = blocks[static_cast<size_t>(r_don)];
    if (!b.amide_donor) return false;
    if (b.has_explicit_h()) {
        const CoordView cv(state.coord_view());
        cv.load_xyz(b.h_start, hpos);
        return true;
    }
    const CoordView cv(state.coord_view());
    float n[3], ca[3], cp[3];
    cv.load_xyz(b.bb_start, n);
    cv.load_xyz(b.ca_atom(), ca);
    const int prev_c = sys.getBlockIndices()[static_cast<size_t>(r_don - 1)].c_atom();
    cv.load_xyz(prev_c, cp);
    HydrogenBondUtils::compute_virtual_amide_H(
        n[0], n[1], n[2], ca[0], ca[1], ca[2], cp[0], cp[1], cp[2],
        hpos[0], hpos[1], hpos[2]);
    return true;
}

}  // namespace

HBondPotential::HBondPotential(std::vector<float> loaded_params, std::vector<float> seq_dep_params)
    : params(std::move(loaded_params)), seq_dep_params(std::move(seq_dep_params)) {}

float HBondPotential::evaluate_directional(int r_don, int r_acc, const State& state, const System& sys) const {
    const auto& blocks = sys.getBlockIndices();
    if (!blocks[static_cast<size_t>(r_don)].amide_donor) return 0.0f;
    int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
    if (o_atom == -1) return 0.0f;

    const CoordView cv(state.coord_view());
    float hpos[3], opos[3];
    if (!load_donor_h(state, sys, r_don, hpos)) return 0.0f;
    cv.load_xyz(o_atom, opos);
    if (!HydrogenBondUtils::is_close_enough_for_hbond(hpos, opos)) return 0.0f;

    HBondDonors donor = HydrogenBondUtils::construct_donor(state, sys, r_don);
    HBondAcceptors acceptor = HydrogenBondUtils::construct_acceptor(state, sys, r_acc);

    if (HydrogenBondUtils::is_hydrogen_bond(donor, acceptor)) {
        std::array<int, 7> indices;
        HydrogenBondUtils::hydrogen_bond_indices(donor, acceptor, indices);
        float e = get(indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6]);
        // legacy hbonds.h: `e += [beta_favor *] seq_hb[helix_sheet][aa_don][aa_acc] * hbond_E[...]`.
        // indices[0] (helix_sheet) is 0 only for res_idx_diff==4; beta_favor applies to
        // every res_idx_diff>4 pair regardless of parallel/antiparallel (indices[0] 1 or 2).
        e *= seq_dep_factor(indices[0], sys.amino_index(r_don), sys.amino_index(r_acc));
        if (indices[0] != 0) {
            e *= BETA_FAVOR;
        }
        return e;
    }
    return 0.0f;
}

EnergyChangeResult HBondPotential::calculateEnergyChange(
    const Context& context,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch
) const {
    const auto& ns = context.neighbors();
    const auto& sys = context.getSystem();
    const int num_residues = sys.getNumResidues();
    const auto& blocks = sys.getBlockIndices();
    const float cut2 = NeighborSystem::kHBondCutoffA * NeighborSystem::kHBondCutoffA;
    const bool use_brute = ns.hbondUsesFallback();
    const bool virt = sys.virtualAmideH() && sys.getTotalHAtoms() == 0;

    auto& hb_ws = const_cast<Context&>(context).getHBondWorkspace();
    hb_ws.begin_call(num_residues);

    // The old side of the delta comes from the accepted state's ledger of
    // nonzero pair energies; build it on first use.
    HBondStateCache& cache = old_state.hbond_cache;
    if (!cache.ready_for(this, num_residues, sys.energy_mask_epoch())) {
        build_cache_(context, old_state);
    }
    hb_ws.pending.clear();
    hb_ws.pending_valid = true;
    hb_ws.pending_old_state = &old_state;
    hb_ws.pending_proposed_state = &proposed_state;
    hb_ws.pending_num_moved = patch.moved_indices.size();
    hb_ws.pending_generation = cache.generation();

    const CoordView cold(old_state.coord_view());
    const CoordView cnew(proposed_state.coord_view());

    // A pair's energy can change only if its donor or acceptor residue is
    // affected (holds a moved atom, or is next to one). Every such pair that
    // is within reach in the proposed state is scored once here; its old
    // energy is in the ledger.
    double e_new_sum = 0.0;
    auto evaluate_new = [&](int r_don, int r_acc) {
        if (r_acc <= 0 || r_acc >= num_residues - 1) return;
        if (r_don <= 0 || r_don >= num_residues - 1) return;
        if (std::abs(r_don - r_acc) <= 3) return;
        if (sys.is_residue_energy_ignored(r_don)) return;
        if (sys.is_residue_energy_ignored(r_acc)) return;

        if (!hb_ws.mark_pair(r_don, r_acc)) return;

        const float e = evaluate_directional(r_don, r_acc, proposed_state, sys);
        ns.stats().hbond_num_geom_checks += 1;
        if (e != 0.0f) {
            e_new_sum += static_cast<double>(e);
            hb_ws.pending.push_back({r_don, r_acc, e});
        }
    };

    auto& res_affected = hb_ws.res_affected;
    const bool have_masks =
        !patch.o_atom_moved.empty()
        && static_cast<int>(patch.o_atom_moved.size()) == sys.getNumAtoms();
    if (!patch.moved_indices.empty()) {
        for (int a : patch.moved_indices) {
            if (a < 0 || a >= static_cast<int>(sys.atom_to_residue.size())) continue;
            const int r = sys.atom_to_residue[static_cast<size_t>(a)];
            if (r < 0 || r >= num_residues) continue;
            res_affected[static_cast<size_t>(r)] = 1;
            if (r > 0) res_affected[static_cast<size_t>(r - 1)] = 1;
            if (r + 1 < num_residues) res_affected[static_cast<size_t>(r + 1)] = 1;
        }
    } else if (have_masks) {
        for (int r = 0; r < num_residues; ++r) {
            const int h_atom = blocks[static_cast<size_t>(r)].h_start;
            const int o_atom = blocks[static_cast<size_t>(r)].o_start;
            const int bb = blocks[static_cast<size_t>(r)].bb_start;
            bool hit = false;
            if (h_atom >= 0 && !patch.h_atom_moved.empty()
                && patch.h_atom_moved[static_cast<size_t>(h_atom)]) hit = true;
            if (o_atom >= 0 && patch.o_atom_moved[static_cast<size_t>(o_atom)]) hit = true;
            if (bb >= 0 && !patch.bb_atom_moved.empty()
                && patch.bb_atom_moved[static_cast<size_t>(bb)]) hit = true;
            if (hit) {
                res_affected[static_cast<size_t>(r)] = 1;
                if (r > 0) res_affected[static_cast<size_t>(r - 1)] = 1;
                if (r + 1 < num_residues) res_affected[static_cast<size_t>(r + 1)] = 1;
            }
        }
    } else {
        for (int r = patch.first_affected_residue; r <= patch.last_affected_residue; ++r) {
            if (r >= 0 && r < num_residues) res_affected[static_cast<size_t>(r)] = 1;
        }
    }
    auto& aff_list = hb_ws.aff_list;
    for (int r = 0; r < num_residues; ++r) {
        if (res_affected[static_cast<size_t>(r)]) aff_list.push_back(r);
    }

    // Old side: every listed pair with an affected end, each counted once.
    double e_old_sum = 0.0;
    for (int r : aff_list) {
        for (const auto& en : cache.as_donor(r)) e_old_sum += static_cast<double>(en.energy);
        for (const auto& en : cache.as_acceptor(r)) {
            if (!res_affected[static_cast<size_t>(en.partner)]) {
                e_old_sum += static_cast<double>(en.energy);
            }
        }
    }

    // New side, part 1: affected donor x unaffected acceptor. The O grid holds
    // the accepted positions, which are the proposed ones for an unaffected
    // acceptor, so one walk at the donor's new H finds every such pair.
    auto process_donor = [&](int r_don) {
        float nh[3];
        if (!load_donor_h(proposed_state, sys, r_don, nh)) return;
        auto query_oxygen = [&](int o) {
            const int r_acc = sys.atom_to_residue[static_cast<size_t>(o)];
            if (r_acc < 0 || r_acc >= num_residues) return;
            if (res_affected[static_cast<size_t>(r_acc)]) return;
            const float bx = nh[0] - cold.x(o), by = nh[1] - cold.y(o), bz = nh[2] - cold.z(o);
            if (bx * bx + by * by + bz * bz > kFarCut2) return;
            evaluate_new(r_don, r_acc);
        };
        if (use_brute) {
            ns.for_each_hbond_acceptor_bruteforce(old_state.coords_soa, nh[0], nh[1], nh[2], cut2, query_oxygen);
        } else {
            ns.for_each_hbond_acceptor_candidate(nh[0], nh[1], nh[2], query_oxygen);
        }
    };

    // Part 2: affected acceptor x unaffected donor, from the H grid at the
    // acceptor's new O (an unaffected donor's H has not moved either).
    auto process_acceptor = [&](int r_acc, int o_atom) {
        const float x = cnew.x(o_atom), y = cnew.y(o_atom), z = cnew.z(o_atom);
        auto query_donor = [&](int neighbor_id) {
            const int r_don = virt
                ? neighbor_id
                : sys.atom_to_residue[static_cast<size_t>(neighbor_id)];
            if (r_don < 0 || r_don >= num_residues) return;
            if (res_affected[static_cast<size_t>(r_don)]) return;
            evaluate_new(r_don, r_acc);
        };
        if (use_brute) {
            if (virt) {
                ns.for_each_hbond_donor_bruteforce(old_state.coords_soa, sys, x, y, z, cut2, query_donor);
            } else {
                ns.for_each_hbond_h_bruteforce(old_state.coords_soa, x, y, z, cut2, query_donor);
            }
        } else {
            ns.for_each_hbond_h_candidate(x, y, z, query_donor);
        }
    };

    for (int r : aff_list) {
        if (blocks[static_cast<size_t>(r)].amide_donor) process_donor(r);
        const int o_atom = blocks[static_cast<size_t>(r)].o_start;
        if (o_atom != -1) process_acceptor(r, o_atom);
    }

    // Part 3: affected donor x affected acceptor, both at proposed positions.
    // Pack the acceptors' new O once, drop far pairs eight at a time, and
    // score the rest in (donor, acceptor) order.
    auto& acc_res = hb_ws.acc_res;
    auto& anx = hb_ws.acc_new_x; auto& any = hb_ws.acc_new_y; auto& anz = hb_ws.acc_new_z;
    acc_res.clear();
    anx.clear(); any.clear(); anz.clear();
    for (int r_acc : aff_list) {
        const int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
        if (o_atom < 0) continue;
        acc_res.push_back(r_acc);
        anx.push_back(cnew.x(o_atom)); any.push_back(cnew.y(o_atom)); anz.push_back(cnew.z(o_atom));
    }
    const int n_acc = static_cast<int>(acc_res.size());
    const int n_acc8 = (n_acc + 7) & ~7;
    // Padding sits ~1e18 A away, so it never passes the far filter.
    for (auto* v : {&anx, &any, &anz}) v->resize(static_cast<size_t>(n_acc8), 1e18f);
    for (int r_don : aff_list) {
        if (!blocks[static_cast<size_t>(r_don)].amide_donor) continue;
        float hnew[3];
        if (!load_donor_h(proposed_state, sys, r_don, hnew)) continue;
        auto visit = [&](int k) {
            const float dx = hnew[0] - anx[static_cast<size_t>(k)];
            const float dy = hnew[1] - any[static_cast<size_t>(k)];
            const float dz = hnew[2] - anz[static_cast<size_t>(k)];
            if (dx * dx + dy * dy + dz * dz > cut2) return;
            evaluate_new(r_don, acc_res[static_cast<size_t>(k)]);
        };
#if defined(__AVX2__)
        const __m256 far = _mm256_set1_ps(kFarCut2);
        const __m256 hnx = _mm256_set1_ps(hnew[0]), hny = _mm256_set1_ps(hnew[1]), hnz = _mm256_set1_ps(hnew[2]);
        for (int k0 = 0; k0 < n_acc8; k0 += 8) {
            const __m256 dxn = _mm256_sub_ps(hnx, _mm256_loadu_ps(anx.data() + k0));
            const __m256 dyn = _mm256_sub_ps(hny, _mm256_loadu_ps(any.data() + k0));
            const __m256 dzn = _mm256_sub_ps(hnz, _mm256_loadu_ps(anz.data() + k0));
            const __m256 rn = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dxn, dxn), _mm256_mul_ps(dyn, dyn)), _mm256_mul_ps(dzn, dzn));
            unsigned m = static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(rn, far, _CMP_LE_OQ)));
            while (m) {
                visit(k0 + __builtin_ctz(m));
                m &= m - 1;
            }
        }
#else
        for (int k = 0; k < n_acc; ++k) visit(k);
#endif
    }

    if (hb_ws.ledger_check) check_ledger_(context, old_state, proposed_state);

    return EnergyChangeResult::finite(static_cast<float>(e_new_sum - e_old_sum) / 1000.0f);
}

void HBondPotential::build_cache_(const Context& context, const State& state) const {
    const auto& ns = context.neighbors();
    const auto& sys = context.getSystem();
    const int num_residues = sys.getNumResidues();
    const auto& blocks = sys.getBlockIndices();
    const float cut2 = NeighborSystem::kHBondCutoffA * NeighborSystem::kHBondCutoffA;
    HBondStateCache& cache = state.hbond_cache;
    cache.reset(num_residues, this, sys.energy_mask_epoch());
    const CoordView cv(state.coord_view());
    for (int r_don = 1; r_don < num_residues - 1; ++r_don) {
        if (!blocks[static_cast<size_t>(r_don)].amide_donor) continue;
        if (sys.is_residue_energy_ignored(r_don)) continue;
        float h[3];
        if (!load_donor_h(state, sys, r_don, h)) continue;
        auto visit = [&](int o) {
            const int r_acc = sys.atom_to_residue[static_cast<size_t>(o)];
            if (r_acc <= 0 || r_acc >= num_residues - 1) return;
            if (std::abs(r_don - r_acc) <= 3) return;
            if (sys.is_residue_energy_ignored(r_acc)) return;
            if (blocks[static_cast<size_t>(r_acc)].o_start != o) return;
            const float dx = h[0] - cv.x(o), dy = h[1] - cv.y(o), dz = h[2] - cv.z(o);
            if (dx * dx + dy * dy + dz * dz > kFarCut2) return;
            const float e = evaluate_directional(r_don, r_acc, state, sys);
            if (e != 0.0f) cache.add(r_don, r_acc, e);
        };
        if (ns.hbondUsesFallback()) {
            ns.for_each_hbond_acceptor_bruteforce(state.coords_soa, h[0], h[1], h[2], cut2, visit);
        } else {
            ns.for_each_hbond_acceptor_candidate(h[0], h[1], h[2], visit);
        }
    }
}

void HBondPotential::check_ledger_(const Context& context, const State& old_state,
                                   const State& proposed_state) const {
    auto& hb_ws = const_cast<Context&>(context).getHBondWorkspace();
    const auto& sys = context.getSystem();
    const int n = sys.getNumResidues();
    const auto& blocks = sys.getBlockIndices();
    const HBondStateCache& cache = old_state.hbond_cache;
    ++hb_ws.ledger_checks;
    auto pending_e = [&](int d, int a) {
        for (const auto& p : hb_ws.pending) if (p.d == d && p.a == a) return p.e;
        return 0.0f;
    };
    for (int d = 1; d < n - 1; ++d) {
        if (!blocks[static_cast<size_t>(d)].amide_donor || sys.is_residue_energy_ignored(d)) continue;
        for (int a = 1; a < n - 1; ++a) {
            if (std::abs(d - a) <= 3 || sys.is_residue_energy_ignored(a)) continue;
            if (blocks[static_cast<size_t>(a)].o_start < 0) continue;
            const bool aff = hb_ws.res_affected[static_cast<size_t>(d)]
                || hb_ws.res_affected[static_cast<size_t>(a)];
            const float e_old = evaluate_directional(d, a, old_state, sys);
            // The ledger must hold the accepted state's energy of every pair.
            if (cache.get(d, a) != e_old) { ++hb_ws.ledger_mismatches; continue; }
            if (!aff) continue;
            // Old path: (d, a) enters the delta iff e_new != e_old.
            const float e_new = evaluate_directional(d, a, proposed_state, sys);
            const bool old_path = (e_new - e_old) != 0.0f;
            const bool new_path = (pending_e(d, a) - cache.get(d, a)) != 0.0f;
            if (old_path != new_path || (new_path && pending_e(d, a) != e_new)) {
                ++hb_ws.ledger_mismatches;
            }
        }
    }
}

void HBondPotential::commitAcceptedMove(
    const Context& context,
    const State& state,
    const State& proposed_state,
    const ProposalPatch& patch
) const {
    auto& hb_ws = const_cast<Context&>(context).getHBondWorkspace();
    HBondStateCache& cache = state.hbond_cache;
    const bool mine = hb_ws.pending_valid && isEnabled()
        && hb_ws.pending_old_state == &state
        && hb_ws.pending_proposed_state == &proposed_state
        && hb_ws.pending_num_moved == patch.moved_indices.size()
        && hb_ws.pending_generation == cache.generation();
    hb_ws.pending_valid = false;
    if (!cache.ready_for(this, context.getSystem().getNumResidues(),
                         context.getSystem().energy_mask_epoch())) return;
    if (!mine) {
        // This move was not scored against this ledger; the coordinates it
        // describes are gone.
        cache.invalidate();
        return;
    }
    for (int r : hb_ws.aff_list) {
        cache.clear_donor(r);
        cache.clear_acceptor(r);
    }
    for (const auto& p : hb_ws.pending) cache.add(p.d, p.a, p.e);
    cache.note_commit();
}

float HBondPotential::calculateEnergy(const Context& context, const State& state) const {
    float total_E = 0.0f;
    const auto& sys = context.getSystem();
    const int num_residues = sys.getNumResidues();
    const auto& blocks = sys.getBlockIndices();

    for (int r_acc = 1; r_acc < num_residues - 1; ++r_acc) {
        if (sys.is_residue_energy_ignored(r_acc)) continue;
        const int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
        if (o_atom < 0) continue;
        for (int r_don = 1; r_don < num_residues - 1; ++r_don) {
            if (sys.is_residue_energy_ignored(r_don)) continue;
            if (std::abs(r_don - r_acc) <= 3) continue;
            if (!blocks[static_cast<size_t>(r_don)].amide_donor) continue;
            total_E += evaluate_directional(r_don, r_acc, state, sys);
        }
    }
    return total_E / 1000.0f;
}

} // namespace mcpu::forces
