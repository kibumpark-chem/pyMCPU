#include "pymcpu/forces/mcpu/common/HydrogenBondPotential.h"
#include "pymcpu/Context.h"
#include "pymcpu/State.h"
#include "pymcpu/System.h"
#include "pymcpu/utils/CoordView.h"
#include "pymcpu/utils/hydrogen_bond_utils.h"
#include "pymcpu/utils/virtual_amide_h.h"
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
    float delta_E = 0.0f;

    const auto& ns = context.neighbors();
    const auto& sys = context.getSystem();
    const int num_residues = sys.getNumResidues();
    const auto& blocks = sys.getBlockIndices();
    const float cut2 = NeighborSystem::kHBondCutoffA * NeighborSystem::kHBondCutoffA;
    const bool use_brute = ns.hbondUsesFallback();
    const bool virt = sys.virtualAmideH() && sys.getTotalHAtoms() == 0;

    auto& hb_ws = const_cast<Context&>(context).getHBondWorkspace();
    hb_ws.begin_call(num_residues);

    auto evaluate_hbond_pair = [&](int r_don, int r_acc) {
        if (r_acc <= 0 || r_acc >= num_residues - 1) return;
        if (r_don <= 0 || r_don >= num_residues - 1) return;
        if (std::abs(r_don - r_acc) <= 3) return;
        if (sys.is_residue_energy_ignored(r_don)) return;
        if (sys.is_residue_energy_ignored(r_acc)) return;

        if (!hb_ws.mark_pair(r_don, r_acc)) return;

        const float e_old = evaluate_directional(r_don, r_acc, old_state, sys);
        const float e_new = evaluate_directional(r_don, r_acc, proposed_state, sys);
        delta_E += (e_new - e_old);
        ns.stats().hbond_num_geom_checks += 2;
    };

    const CoordView cold(old_state.coord_view());
    const CoordView cnew(proposed_state.coord_view());

    auto process_moved_donor = [&](int r_don) {
        float oh[3], nh[3];
        if (!load_donor_h(old_state, sys, r_don, oh)) return;
        if (!load_donor_h(proposed_state, sys, r_don, nh)) return;

        auto query_oxygen = [&](int neighbor_atom) {
            const int r_acc = sys.atom_to_residue[static_cast<size_t>(neighbor_atom)];
            if (r_acc >= 0 && r_acc < num_residues) {
                // Skip pairs that are far apart in both states (see kFarCut2).
                const int o = blocks[static_cast<size_t>(r_acc)].o_start;
                if (o < 0) return;
                const float ax = oh[0] - cold.x(o), ay = oh[1] - cold.y(o), az = oh[2] - cold.z(o);
                const float bx = nh[0] - cnew.x(o), by = nh[1] - cnew.y(o), bz = nh[2] - cnew.z(o);
                if (ax * ax + ay * ay + az * az > kFarCut2
                    && bx * bx + by * by + bz * bz > kFarCut2) return;
            }
            evaluate_hbond_pair(r_don, r_acc);
        };
        if (use_brute) {
            ns.for_each_hbond_acceptor_bruteforce(old_state.coords_soa, oh[0], oh[1], oh[2], cut2, query_oxygen);
            ns.for_each_hbond_acceptor_bruteforce(old_state.coords_soa, nh[0], nh[1], nh[2], cut2, query_oxygen);
        } else {
            // The second walk skips the cells the first one covered: every
            // candidate there has been marked or is far in both states.
            ns.for_each_hbond_acceptor_candidate(oh[0], oh[1], oh[2], query_oxygen);
            ns.for_each_hbond_acceptor_candidate_not_near(nh[0], nh[1], nh[2], oh[0], oh[1], oh[2], query_oxygen);
        }
    };

    auto process_moved_acceptor = [&](int r_acc, int o_atom) {
        const float oo0 = cold.x(o_atom), oo1 = cold.y(o_atom), oo2 = cold.z(o_atom);
        const float no0 = cnew.x(o_atom), no1 = cnew.y(o_atom), no2 = cnew.z(o_atom);

        auto query_donor = [&](int neighbor_id) {
            const int r_don = virt
                ? neighbor_id
                : sys.atom_to_residue[static_cast<size_t>(neighbor_id)];
            evaluate_hbond_pair(r_don, r_acc);
        };
        if (use_brute) {
            if (virt) {
                ns.for_each_hbond_donor_bruteforce(old_state.coords_soa, sys, oo0, oo1, oo2, cut2, query_donor);
                ns.for_each_hbond_donor_bruteforce(old_state.coords_soa, sys, no0, no1, no2, cut2, query_donor);
            } else {
                ns.for_each_hbond_h_bruteforce(old_state.coords_soa, oo0, oo1, oo2, cut2, query_donor);
                ns.for_each_hbond_h_bruteforce(old_state.coords_soa, no0, no1, no2, cut2, query_donor);
            }
        } else {
            // As for donors: the first walk marks every candidate it sees.
            ns.for_each_hbond_h_candidate(oo0, oo1, oo2, query_donor);
            ns.for_each_hbond_h_candidate_not_near(no0, no1, no2, oo0, oo1, oo2, query_donor);
        }
    };

    const bool have_masks =
        !patch.o_atom_moved.empty()
        && static_cast<int>(patch.o_atom_moved.size()) == sys.getNumAtoms();

    auto& res_affected = hb_ws.res_affected;
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

    for (int r = 0; r < num_residues; ++r) {
        if (!res_affected[static_cast<size_t>(r)]) continue;
        if (blocks[static_cast<size_t>(r)].amide_donor) process_moved_donor(r);
        const int o_atom = blocks[static_cast<size_t>(r)].o_start;
        if (o_atom != -1) process_moved_acceptor(r, o_atom);
    }

    auto& aff_list = hb_ws.aff_list;
    for (int r = 0; r < num_residues; ++r) {
        if (res_affected[static_cast<size_t>(r)]) aff_list.push_back(r);
    }
    // Affected donor x affected acceptor pairs, which the grid walks above can
    // miss when both ends moved. Pack the acceptors' O coordinates once, drop
    // the pairs that are far in both states eight at a time, and send the rest
    // through the original test in the original (donor, acceptor) order.
    auto& acc_res = hb_ws.acc_res;
    auto& aox = hb_ws.acc_old_x; auto& aoy = hb_ws.acc_old_y; auto& aoz = hb_ws.acc_old_z;
    auto& anx = hb_ws.acc_new_x; auto& any = hb_ws.acc_new_y; auto& anz = hb_ws.acc_new_z;
    acc_res.clear();
    aox.clear(); aoy.clear(); aoz.clear(); anx.clear(); any.clear(); anz.clear();
    for (int r_acc : aff_list) {
        const int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
        if (o_atom < 0) continue;
        acc_res.push_back(r_acc);
        aox.push_back(cold.x(o_atom)); aoy.push_back(cold.y(o_atom)); aoz.push_back(cold.z(o_atom));
        anx.push_back(cnew.x(o_atom)); any.push_back(cnew.y(o_atom)); anz.push_back(cnew.z(o_atom));
    }
    const int n_acc = static_cast<int>(acc_res.size());
    const int n_acc8 = (n_acc + 7) & ~7;
    // Padding sits ~1e18 A away, so it never passes the far filter.
    for (auto* v : {&aox, &aoy, &aoz, &anx, &any, &anz}) v->resize(static_cast<size_t>(n_acc8), 1e18f);
    auto d2 = [](const float* h, float x, float y, float z) {
        const float dx = h[0] - x, dy = h[1] - y, dz = h[2] - z;
        return dx * dx + dy * dy + dz * dz;
    };
    for (int r_don : aff_list) {
        if (!blocks[static_cast<size_t>(r_don)].amide_donor) continue;
        float hold[3], hnew[3];
        if (!load_donor_h(old_state, sys, r_don, hold)) continue;
        if (!load_donor_h(proposed_state, sys, r_don, hnew)) continue;
        auto visit = [&](int k) {
            const int r_acc = acc_res[static_cast<size_t>(k)];
            const int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
            const float ox = cold.x(o_atom), oy = cold.y(o_atom), oz = cold.z(o_atom);
            const float nx = cnew.x(o_atom), ny = cnew.y(o_atom), nz = cnew.z(o_atom);
            const float d_old2 = d2(hold, ox, oy, oz);
            const float d_new2 = d2(hnew, nx, ny, nz);
            if (d_old2 > cut2 && d_new2 > cut2) return;
            evaluate_hbond_pair(r_don, r_acc);
        };
#if defined(__AVX2__)
        const __m256 far = _mm256_set1_ps(kFarCut2);
        const __m256 hox = _mm256_set1_ps(hold[0]), hoy = _mm256_set1_ps(hold[1]), hoz = _mm256_set1_ps(hold[2]);
        const __m256 hnx = _mm256_set1_ps(hnew[0]), hny = _mm256_set1_ps(hnew[1]), hnz = _mm256_set1_ps(hnew[2]);
        for (int k0 = 0; k0 < n_acc8; k0 += 8) {
            const __m256 dxo = _mm256_sub_ps(hox, _mm256_loadu_ps(aox.data() + k0));
            const __m256 dyo = _mm256_sub_ps(hoy, _mm256_loadu_ps(aoy.data() + k0));
            const __m256 dzo = _mm256_sub_ps(hoz, _mm256_loadu_ps(aoz.data() + k0));
            const __m256 dxn = _mm256_sub_ps(hnx, _mm256_loadu_ps(anx.data() + k0));
            const __m256 dyn = _mm256_sub_ps(hny, _mm256_loadu_ps(any.data() + k0));
            const __m256 dzn = _mm256_sub_ps(hnz, _mm256_loadu_ps(anz.data() + k0));
            const __m256 ro = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dxo, dxo), _mm256_mul_ps(dyo, dyo)), _mm256_mul_ps(dzo, dzo));
            const __m256 rn = _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(dxn, dxn), _mm256_mul_ps(dyn, dyn)), _mm256_mul_ps(dzn, dzn));
            unsigned m = static_cast<unsigned>(_mm256_movemask_ps(_mm256_or_ps(
                _mm256_cmp_ps(ro, far, _CMP_LE_OQ), _mm256_cmp_ps(rn, far, _CMP_LE_OQ))));
            while (m) {
                visit(k0 + __builtin_ctz(m));
                m &= m - 1;
            }
        }
#else
        for (int k = 0; k < n_acc; ++k) visit(k);
#endif
    }
    return EnergyChangeResult::finite(delta_E / 1000.0f);
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
