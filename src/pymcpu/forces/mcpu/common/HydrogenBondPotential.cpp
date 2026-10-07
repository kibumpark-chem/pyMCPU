#include "pymcpu/forces/mcpu/common/HydrogenBondPotential.h"
#include "pymcpu/Context.h"
#include "pymcpu/State.h"
#include "pymcpu/System.h"
#include "pymcpu/utils/CoordView.h"
#include "pymcpu/utils/hydrogen_bond_utils.h"
#include "pymcpu/utils/virtual_amide_h.h"
#include "pymcpu/forces/mcpu/common/HBondStateCache.h"
#include "pymcpu/neighbor/PairSearch.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#if defined(__AVX2__)
#include <immintrin.h>
#include "pymcpu/utils/pair_r2.h"
#endif

namespace mcpu::forces {

namespace {

// A donor/acceptor pair whose H and O are at least HBOND_CUTOFF apart in both
// states scores 0 in both, so its term in the delta is exactly +0.0f and it
// can be dropped without changing the sum. The walks drop a pair only when
// it is farther than the ledger's listing distance, 0.05 A past the cutoff,
// and the ledger lists (with energy 0 if need be) every pair they keep.
constexpr float kFarCut2 = NeighborSystem::kHBondListA * NeighborSystem::kHBondListA;
static_assert(kFarCut2 > HBOND_CUTOFF_SQUARED, "far filter must be looser than the H-bond cutoff");

// An unlisted pair was farther than kHBondListA when last measured, so it
// cannot score until it has moved by the band between that and the cutoff.
// The ledger is rebuilt before its drift reaches kListBudgetA; the rest of
// the band covers the rounding of the squared distances themselves.
constexpr float kListBudgetA = 0.04f;
static_assert(kListBudgetA < NeighborSystem::kHBondListA - HBOND_CUTOFF, "budget must fit in the listing band");

// A virtual amide H is N minus the unit vector along (CA - N) + (C_prev - N)
// (|v| ~ 1.36 A), so it moves by up to ~4.9 times the per-atom carry error
// and an H...O distance by ~5.9 times it, i.e. ~3x Context::rigid_carry_bound_A
// (which bounds a heavy-atom pair). Explicit H and O need only 1x.
constexpr float kCarryFactor = 4.0f;

// How much ledger drift a pair's score is certain to survive (its slack).
//
// evaluate_directional picks a table value by discrete tests on the pair's
// geometry: the H...O cutoff, the CA-CA orientation gates, the backbone
// dihedral gates, the CA-CA orientation sign and the six angle bins (each
// edge with the band around it where angle_bin_from_cos bins by acos). Every
// tested quantity is rigid-invariant. A rigid carry moves each atom off the
// exact rigid image of the old coordinates by rounding only, and the ledger
// drift grows by kCarryFactor * Context::rigid_carry_bound_A per carry, a
// bound on a pair of atoms, so the atoms of a carried pair stay within
// drift / (2 kCarryFactor) of a rigid image of the coordinates it was scored
// at. If every atom moves by at most eps, a quantity moves by at most L * eps
// (the bounds below, from |d(w / |w|)| <= 2 |dw| / |w|). The slack is the
// smallest margin / L over the tests the score went through, in drift units
// and halved for the second-order terms the bounds leave out; each margin
// first loses a floor for the rounding of the quantity itself. While every
// one of those tests keeps its outcome, the score takes the same path to the
// same table entry, so a pair carried within its slack keeps its energy bit
// for bit.
constexpr float kDriftPerAtomA = 2.0f * kCarryFactor;
// The floors assume engine-frame coordinates below ~500 A (the engine frame
// keeps them near the chain): kDistFloorA must cover about 1.7 ulp of the
// largest coordinate, and 1e-4 A does up to |x| ~ 512 A.
constexpr float kDistFloorA = 1e-4f;
constexpr float kCosFloor = 1e-5f;
constexpr float kDihedralFloorRad = 2e-5f;

struct Slack {
    float eps = std::numeric_limits<float>::infinity();  // A per atom
    // A margin or bound that is not a finite positive number (degenerate
    // geometry) gives no slack.
    void add(float margin, float lip) {
        const float e = margin / lip;
        eps = e > 0.f ? std::min(eps, e) : 0.f;
    }
    [[nodiscard]] float drift() const { return 0.5f * kDriftPerAtomA * eps; }

    // The H...O cutoff. A virtual H is N minus the unit vector along
    // v = (CA - N) + (C_prev - N), so it moves by up to (1 + 8 / |v|) eps.
    void add_cutoff(const float* h, const float* o, const float* v_or_null) {
        const float r = std::sqrt(mcpu::pair_r2(h[0] - o[0], h[1] - o[1], h[2] - o[2]));
        const float h_lip = v_or_null
            ? 1.f + 8.f / std::sqrt(mcpu::pair_r2(v_or_null[0], v_or_null[1], v_or_null[2])) : 1.f;
        add(std::fabs(r - HBOND_CUTOFF) - kDistFloorA, 1.f + h_lip);
    }

    // The CA-CA orientation gates (passes_ca_geometry_gate).
    void add_ca_gate(const HBondDonors& d, const HBondAcceptors& a) {
        const bool helix = std::abs(d.residue_index - a.residue_index) == 4;
        const float t1 = std::sqrt(helix ? HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX_ALL
                                         : HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET_ALL);
        const float t2 = std::sqrt(helix ? HBOND_ORIENTATION_CUTOFF_SQUARED_HELIX
                                         : HBOND_ORIENTATION_CUTOFF_SQUARED_SHEET);
        for (const Eigen::Vector3f& v : {Eigen::Vector3f(d.prev_CA - a.next_CA), Eigen::Vector3f(d.CA - a.next_CA),
                                         Eigen::Vector3f(d.prev_CA - a.CA), Eigen::Vector3f(d.CA - a.CA)}) {
            const float r = v.norm();
            add(std::min(std::fabs(r - t1), std::fabs(r - t2)) - kDistFloorA, 2.f);
        }
    }

    // The Ramachandran gates (passes_ramachandran_gate), where they test.
    void add_rama_gate(const HBondDonors& d, const HBondAcceptors& a) {
        const int diff = std::abs(d.residue_index - a.residue_index);
        const auto el = [](char c) { return c == 'E' || c == 'L'; };
        const bool helix = diff == 4 && (el(d.secondary_structure) || el(a.secondary_structure));
        const bool sheet = diff > 4 && d.secondary_structure != 'L' && a.secondary_structure != 'L';
        if (!helix && !sheet) return;
        add_dihedral(d.prev_C, d.N, d.CA, d.C, sheet);
        add_dihedral(d.prev_N, d.prev_CA, d.prev_C, d.N, sheet);
        add_dihedral(a.C, a.next_N, a.next_CA, a.next_C, sheet);
        add_dihedral(a.N, a.CA, a.C, a.next_N, sheet);
    }

    // A dihedral, shifted to [0, 360] degrees, is tested against 180 (helix
    // pairs) or 30, 150, 180 and 210 (sheet pairs); 0 and 360 are the atan2
    // wrap. It moves by at most the arcs its two plane normals move, which
    // are at most pi / 2 times their chords.
    void add_dihedral(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2,
                      const Eigen::Vector3f& p3, const Eigen::Vector3f& p4, bool sheet) {
        constexpr float kDeg = mcpu::PI_F / 180.0f;
        const float phi = GeometryUtils::calculate_dihedral(p1, p2, p3, p4) + mcpu::PI_F;
        float m = std::min({phi, std::fabs(phi - 180.f * kDeg), std::fabs(phi - 360.f * kDeg)});
        if (sheet) {
            for (float t : {30.f, 150.f, 210.f}) m = std::min(m, std::fabs(phi - t * kDeg));
        }
        const Eigen::Vector3f b1 = p2 - p1, b2 = p3 - p2, b3 = p4 - p3;
        const float lip = 2.f * mcpu::PI_F * ((b1.norm() + b2.norm()) / b1.cross(b2).norm()
                                              + (b2.norm() + b3.norm()) / b2.cross(b3).norm());
        add(m - kDihedralFloorRad, lip);
    }

    // The CA-CA orientation sign and the six angle bins, from the cosines
    // hydrogen_bond_indices reports. The cos of two unit vectors moves by at
    // most the sum of their moves.
    void add_bins(const HBondDonors& d, const HBondAcceptors& a, const float* c) {
        if (std::abs(d.residue_index - a.residue_index) != 4) {
            const float lip = 4.f / (d.next_CA - d.prev_CA).norm() + 4.f / (a.next_CA - a.prev_CA).norm();
            add(std::min(std::fabs(c[0]), std::fabs(std::fabs(c[0]) - HydrogenBondUtils::kCacaSignBand)) - kCosFloor, lip);
        }
        const auto bins = [&](float c_pca, float c_bca, const Eigen::Vector3f& N1, const Eigen::Vector3f& CA1,
                              const Eigen::Vector3f& C1, const Eigen::Vector3f& N2, const Eigen::Vector3f& CA2,
                              const Eigen::Vector3f& C2) {
            add(bin_margin(c_pca), normal_lip(N1, CA1, C1) + normal_lip(N2, CA2, C2));
            add(bin_margin(c_bca), bisector_lip(N1, CA1, C1) + bisector_lip(N2, CA2, C2));
        };
        bins(c[1], c[2], d.N, d.CA, d.C, a.N, a.CA, a.C);
        bins(c[3], c[4], d.prev_N, d.prev_CA, d.prev_C, a.next_N, a.next_CA, a.next_C);
        bins(c[5], c[6], d.prev_C, d.N, d.CA, a.CA, a.C, a.next_N);
    }

    static float bin_margin(float c) {
        float m = std::numeric_limits<float>::infinity();
        for (float e : HydrogenBondUtils::kBinEdgeCos) {
            const float d = std::fabs(c - e);
            m = std::min({m, d, std::fabs(d - HydrogenBondUtils::kBinEdgeBand)});
        }
        return m - kCosFloor;
    }
    // Move per eps of the unit normal to the plane (p1, p2, p3), p2 central.
    static float normal_lip(const Eigen::Vector3f& p1, const Eigen::Vector3f& p2, const Eigen::Vector3f& p3) {
        const Eigen::Vector3f v1 = p1 - p2, v2 = p3 - p2;
        return 4.f * (v1.norm() + v2.norm()) / v1.cross(v2).norm();
    }
    // Move per eps of the unit bisector of p1 - c and p3 - c.
    static float bisector_lip(const Eigen::Vector3f& p1, const Eigen::Vector3f& c, const Eigen::Vector3f& p3) {
        const Eigen::Vector3f v1 = p1 - c, v2 = p3 - c;
        const float n1 = v1.norm(), n2 = v2.norm();
        return 8.f * (1.f / n1 + 1.f / n2) / (v1 / n1 + v2 / n2).norm();
    }
};

// A listed pair of two rigid sites that the pending move carries at its
// ledger energy, on neither side of the delta: the drift the move would
// leave is still within the pair's slack. Any other listed pair of two rigid
// sites is re-scored.
bool carried(const HBondWorkspace& ws, int r, const HBondStateCache::Entry& en) {
    const auto& ra = ws.res_affected;
    return (ra[static_cast<size_t>(r)] & ra[static_cast<size_t>(en.partner)] & HBondWorkspace::kRigidSite) != 0
        && ws.pending_drift <= en.fresh_until;
}

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
    return evaluate_<false>(r_don, r_acc, state, sys, nullptr);
}

float HBondPotential::evaluate_with_slack(int r_don, int r_acc, const State& state, const System& sys,
                                          float& slack) const {
    return evaluate_<true>(r_don, r_acc, state, sys, &slack);
}

template <bool kSlack>
float HBondPotential::evaluate_(int r_don, int r_acc, const State& state, const System& sys, float* slack) const {
    // Every return reports the slack of the tests passed on the way to it.
    [[maybe_unused]] Slack s;
    const auto done = [&](float e) {
        if constexpr (kSlack) *slack = s.drift();
        return e;
    };
    const auto& blocks = sys.getBlockIndices();
    if (!blocks[static_cast<size_t>(r_don)].amide_donor) return done(0.0f);
    int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
    if (o_atom == -1) return done(0.0f);

    const CoordView cv(state.coord_view());
    float hpos[3], opos[3];
    if (!load_donor_h(state, sys, r_don, hpos)) return done(0.0f);
    cv.load_xyz(o_atom, opos);
    if constexpr (kSlack) {
        const BlockIndices& b = blocks[static_cast<size_t>(r_don)];
        float v[3];
        if (!b.has_explicit_h()) {
            const int n = b.bb_start, ca = b.ca_atom(), cp = blocks[static_cast<size_t>(r_don - 1)].c_atom();
            v[0] = (cv.x(ca) - cv.x(n)) + (cv.x(cp) - cv.x(n));
            v[1] = (cv.y(ca) - cv.y(n)) + (cv.y(cp) - cv.y(n));
            v[2] = (cv.z(ca) - cv.z(n)) + (cv.z(cp) - cv.z(n));
        }
        s.add_cutoff(hpos, opos, b.has_explicit_h() ? nullptr : v);
    }
    if (!HydrogenBondUtils::is_close_enough_for_hbond(hpos, opos)) return done(0.0f);

    HBondDonors donor = HydrogenBondUtils::construct_donor(state, sys, r_don);
    HBondAcceptors acceptor = HydrogenBondUtils::construct_acceptor(state, sys, r_acc);
    if constexpr (kSlack) s.add_ca_gate(donor, acceptor);
    if (!HydrogenBondUtils::passes_ca_geometry_gate(donor, acceptor)) return done(0.0f);
    if constexpr (kSlack) s.add_rama_gate(donor, acceptor);
    if (!HydrogenBondUtils::passes_ramachandran_gate(donor, acceptor)) return done(0.0f);

    std::array<int, 7> indices;
    [[maybe_unused]] float cosines[7];
    HydrogenBondUtils::hydrogen_bond_indices(donor, acceptor, indices, kSlack ? cosines : nullptr);
    if constexpr (kSlack) s.add_bins(donor, acceptor, cosines);
    float e = get(indices[0], indices[1], indices[2], indices[3], indices[4], indices[5], indices[6]);
    // legacy hbonds.h: `e += [beta_favor *] seq_hb[helix_sheet][aa_don][aa_acc] * hbond_E[...]`.
    // indices[0] (helix_sheet) is 0 only for res_idx_diff==4; beta_favor applies to
    // every res_idx_diff>4 pair regardless of parallel/antiparallel (indices[0] 1 or 2).
    e *= seq_dep_factor(indices[0], sys.amino_index(r_don), sys.amino_index(r_acc));
    if (indices[0] != 0) {
        e *= BETA_FAVOR;
    }
    return done(e);
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
    const bool use_brute = ns.hbondUsesFallback();
    const bool virt = sys.virtualAmideH() && sys.getTotalHAtoms() == 0;

    auto& hb_ws = const_cast<Context&>(context).getHBondWorkspace();
    hb_ws.begin_call(num_residues);

    // The old side of the delta comes from the accepted state's ledger of
    // listed pair energies; build it on first use.
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
        e_new_sum += static_cast<double>(e);
        hb_ws.pending.push_back({r_don, r_acc, e, hb_ws.pending_drift});  // re-scored at its next carry
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

    // Rigid sites: under a rigid move, a residue whose donor and acceptor
    // geometry (backbone of r-1, r, r+1 and its amide H) moved as one body
    // keeps its energy with every other such residue up to the rounding of
    // the rotation. Such a pair is carried at its ledger energy while the
    // drift since it was scored is within its slack (evaluate_with_slack), and
    // re-scored otherwise. skip_rigid_mm = false scores them all, as a
    // reference.
    constexpr uint8_t kRigid = HBondWorkspace::kRigidSite;
    constexpr uint8_t kBbMoved = HBondWorkspace::kBackboneMoved;
    const int n_atoms = sys.getNumAtoms();
    if (patch.is_rigid && !patch.moved_indices.empty()
        && context.neighborConfig().skip_rigid_mm
        && static_cast<int>(patch.moving_atoms.size()) == n_atoms) {
        const uint8_t* mv = patch.moving_atoms.data();
        for (int r : aff_list) {
            const BlockIndices& b = blocks[static_cast<size_t>(r)];
            if (b.bb_start < 0) continue;
            if (mv[b.bb_start] && mv[b.ca_atom()] && mv[b.c_atom()]
                && (b.o_start < 0 || mv[b.o_start])
                && (!b.has_explicit_h() || mv[b.h_start])) {
                res_affected[static_cast<size_t>(r)] |= kBbMoved;
            }
        }
        for (int r : aff_list) {
            if (r <= 0 || r >= num_residues - 1) continue;
            if (res_affected[static_cast<size_t>(r - 1)] & res_affected[static_cast<size_t>(r)]
                & res_affected[static_cast<size_t>(r + 1)] & kBbMoved) {
                res_affected[static_cast<size_t>(r)] |= kRigid;
            }
        }
    }
    auto rigid = [&](int r) { return (res_affected[static_cast<size_t>(r)] & kRigid) != 0; };

    // Drift budget. A pair of two rigid sites is carried or re-decided only
    // if listed; the carry moves an unlisted one by at most `carry`. Before the sum
    // since the ledger was measured could reach the band, measure it again
    // (the energies it finds are the ones carried, since every pair that can
    // score was re-decided). A move that alone exceeds the band skips nothing.
    float carry = 0.f;
    if (std::any_of(aff_list.begin(), aff_list.end(), rigid)) {
        carry = kCarryFactor * context.rigid_carry_bound_A(proposed_state, patch);
        if (carry > kListBudgetA) {
            for (int r : aff_list) res_affected[static_cast<size_t>(r)] &= static_cast<uint8_t>(~kRigid);
            carry = 0.f;
        } else if (cache.drift + carry > kListBudgetA) {
            build_cache_(context, old_state);
            hb_ws.pending_generation = cache.generation();
        }
    }
    hb_ws.pending_drift = cache.drift + carry;

    // Old side: every listed pair with an affected end, each counted once,
    // except the carried ones. A listed pair of two rigid sites that is not
    // carried is re-scored here, with a new slack; an unlisted one cannot
    // cross the cutoff within the drift budget.
    double e_old_sum = 0.0;
    for (int r : aff_list) {
        const bool r_rigid = rigid(r);
        for (const auto& en : cache.as_donor(r)) {
            if (carried(hb_ws, r, en)) continue;
            e_old_sum += static_cast<double>(en.energy);
            if (r_rigid && rigid(en.partner)) {
                float slack;
                const float e = evaluate_with_slack(r, en.partner, proposed_state, sys, slack);
                ns.stats().hbond_num_geom_checks += 1;
                e_new_sum += static_cast<double>(e);
                hb_ws.pending.push_back({r, en.partner, e, hb_ws.pending_drift + slack});
            }
        }
        for (const auto& en : cache.as_acceptor(r)) {
            if (!res_affected[static_cast<size_t>(en.partner)]) {
                e_old_sum += static_cast<double>(en.energy);
            }
        }
    }

    // New side, part 1: affected donor x unaffected acceptor. The O grid holds
    // the accepted positions, which are the proposed ones for an unaffected
    // acceptor, so one walk at the donor's new H finds every such pair.
    // Part 2: affected acceptor x unaffected donor, from the H grid at the
    // acceptor's new O (an unaffected donor's H has not moved either).
    const CellListMC* o_cells = ns.hbond_o_cells();
    const CellListMC* h_cells = ns.hbond_h_cells();
    const bool on_layer = !use_brute && o_cells && h_cells;
    if (on_layer) {
        // The shared pair-search walk: cells whose listed sites all belong
        // to affected residues are skipped, and eight slots at a time are
        // dropped beyond the far cutoff before any per-pair work.
        using neighbor::Visit;
        auto& atom_aff = hb_ws.atom_aff;
        if (static_cast<int>(atom_aff.size()) != n_atoms) atom_aff.assign(static_cast<size_t>(n_atoms), 0);
        auto& o_sites = hb_ws.o_sites;
        auto& h_sites = hb_ws.h_sites;
        o_sites.clear();
        h_sites.clear();
        for (int r : aff_list) {
            const BlockIndices& b = blocks[static_cast<size_t>(r)];
            if (b.o_start >= 0) {
                atom_aff[static_cast<size_t>(b.o_start)] = 1;
                o_sites.push_back(b.o_start);
            }
            if (!b.amide_donor) continue;
            if (virt) {
                h_sites.push_back(r);
            } else if (b.has_explicit_h()) {
                atom_aff[static_cast<size_t>(b.h_start)] = 1;
                h_sites.push_back(b.h_start);
            }
        }
        const float lim2 = kFarCut2 * neighbor::kSpanMaskSlack;
        {
            const OpenCellGrid& g = o_cells->grid();
            const neighbor::MovedCellScope<OpenCellGrid> scope(
                hb_ws.o_cells, g, o_sites.data(), static_cast<int>(o_sites.size()));
            const neighbor::WalkArgs wa{atom_aff.data(), atom_aff.size(), scope.counts(),
                                        0.f, lim2};
            OpenCellGrid::StencilMemo memo;
            for (int r_don : aff_list) {
                if (!blocks[static_cast<size_t>(r_don)].amide_donor) continue;
                float nh[3];
                if (!load_donor_h(proposed_state, sys, r_don, nh)) continue;
                auto fn = [&](const neighbor::Probe&, int o, const neighbor::CellSpan&, int) {
                    ++ns.stats().hbond_num_candidates_iterated;
                    const int r_acc = sys.atom_to_residue[static_cast<size_t>(o)];
                    if (r_acc < 0 || r_acc >= num_residues) return Visit::Continue;
                    if (res_affected[static_cast<size_t>(r_acc)]) return Visit::Continue;
                    const float bx = nh[0] - cold.x(o), by = nh[1] - cold.y(o), bz = nh[2] - cold.z(o);
                    if (pair_r2(bx, by, bz) > kFarCut2) return Visit::Continue;
                    evaluate_new(r_don, r_acc);
                    return Visit::Continue;
                };
                neighbor::detail::probe_static<neighbor::Cells::Stencil>(
                    g, neighbor::Probe{-1, nh[0], nh[1], nh[2]}, wa, fn, &memo);
            }
        }
        {
            const OpenCellGrid& g = h_cells->grid();
            const neighbor::MovedCellScope<OpenCellGrid> scope(
                hb_ws.h_cells, g, h_sites.data(), static_cast<int>(h_sites.size()));
            const neighbor::WalkArgs wa{virt ? res_affected.data() : atom_aff.data(),
                                        virt ? res_affected.size() : atom_aff.size(),
                                        scope.counts(), 0.f, lim2};
            OpenCellGrid::StencilMemo memo;
            for (int r_acc : aff_list) {
                const int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
                if (o_atom < 0) continue;
                const float x = cnew.x(o_atom), y = cnew.y(o_atom), z = cnew.z(o_atom);
                auto fn = [&](const neighbor::Probe&, int id, const neighbor::CellSpan&, int) {
                    ++ns.stats().hbond_num_candidates_iterated;
                    const int r_don = virt ? id : sys.atom_to_residue[static_cast<size_t>(id)];
                    if (r_don < 0 || r_don >= num_residues) return Visit::Continue;
                    if (res_affected[static_cast<size_t>(r_don)]) return Visit::Continue;
                    evaluate_new(r_don, r_acc);
                    return Visit::Continue;
                };
                neighbor::detail::probe_static<neighbor::Cells::Stencil>(
                    g, neighbor::Probe{-1, x, y, z}, wa, fn, &memo);
            }
        }
        for (int o : o_sites) atom_aff[static_cast<size_t>(o)] = 0;
        if (!virt) for (int h : h_sites) atom_aff[static_cast<size_t>(h)] = 0;
    } else {
        // The H-bond grids are off (a cell overflowed): brute force.
        auto process_donor = [&](int r_don) {
            float nh[3];
            if (!load_donor_h(proposed_state, sys, r_don, nh)) return;
            auto query_oxygen = [&](int o) {
                const int r_acc = sys.atom_to_residue[static_cast<size_t>(o)];
                if (r_acc < 0 || r_acc >= num_residues) return;
                if (res_affected[static_cast<size_t>(r_acc)]) return;
                const float bx = nh[0] - cold.x(o), by = nh[1] - cold.y(o), bz = nh[2] - cold.z(o);
                if (pair_r2(bx, by, bz) > kFarCut2) return;
                evaluate_new(r_don, r_acc);
            };
            ns.for_each_hbond_acceptor_bruteforce(old_state.coords_soa, nh[0], nh[1], nh[2], kFarCut2, query_oxygen);
        };
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
            if (virt) {
                ns.for_each_hbond_donor_bruteforce(old_state.coords_soa, sys, x, y, z, kFarCut2, query_donor);
            } else {
                ns.for_each_hbond_h_bruteforce(old_state.coords_soa, x, y, z, kFarCut2, query_donor);
            }
        };
        for (int r : aff_list) {
            if (blocks[static_cast<size_t>(r)].amide_donor) process_donor(r);
            const int o_atom = blocks[static_cast<size_t>(r)].o_start;
            if (o_atom != -1) process_acceptor(r, o_atom);
        }
    }

    // Part 3: affected donor x affected acceptor, both at proposed positions.
    // Pack the acceptors' new O once (those that are not rigid sites first,
    // each group padded to a multiple of 8), drop far pairs eight at a time,
    // and score the rest in (donor, acceptor) order. A rigid-site donor scans
    // only the first group.
    auto& acc_res = hb_ws.acc_res;
    auto& anx = hb_ws.acc_new_x; auto& any = hb_ws.acc_new_y; auto& anz = hb_ws.acc_new_z;
    acc_res.clear();
    anx.clear(); any.clear(); anz.clear();
    // Padding sits ~1e18 A away, so it never passes the far filter.
    auto pack = [&](bool want_rigid) {
        for (int r_acc : aff_list) {
            const int o_atom = blocks[static_cast<size_t>(r_acc)].o_start;
            if (o_atom < 0 || rigid(r_acc) != want_rigid) continue;
            acc_res.push_back(r_acc);
            anx.push_back(cnew.x(o_atom)); any.push_back(cnew.y(o_atom)); anz.push_back(cnew.z(o_atom));
        }
        const size_t n8 = (acc_res.size() + 7) & ~static_cast<size_t>(7);
        acc_res.resize(n8, 0);
        for (auto* v : {&anx, &any, &anz}) v->resize(n8, 1e18f);
    };
    pack(false);
    const int n_flex8 = static_cast<int>(acc_res.size());
    pack(true);
    const int n_all8 = static_cast<int>(acc_res.size());
    for (int r_don : aff_list) {
        if (!blocks[static_cast<size_t>(r_don)].amide_donor) continue;
        float hnew[3];
        if (!load_donor_h(proposed_state, sys, r_don, hnew)) continue;
        const int n_scan = rigid(r_don) ? n_flex8 : n_all8;
        auto visit = [&](int k) {
            const float dx = hnew[0] - anx[static_cast<size_t>(k)];
            const float dy = hnew[1] - any[static_cast<size_t>(k)];
            const float dz = hnew[2] - anz[static_cast<size_t>(k)];
            if (pair_r2(dx, dy, dz) > kFarCut2) return;
            evaluate_new(r_don, acc_res[static_cast<size_t>(k)]);
        };
#if defined(__AVX2__)
        const __m256 far = _mm256_set1_ps(kFarCut2);
        const __m256 hnx = _mm256_set1_ps(hnew[0]), hny = _mm256_set1_ps(hnew[1]), hnz = _mm256_set1_ps(hnew[2]);
        for (int k0 = 0; k0 < n_scan; k0 += 8) {
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
        for (int k = 0; k < n_scan; ++k) visit(k);
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
            if (pair_r2(dx, dy, dz) > kFarCut2) return;
            float slack;
            const float e = evaluate_with_slack(r_don, r_acc, state, sys, slack);
            cache.add(r_don, r_acc, e, slack);
        };
        if (ns.hbondUsesFallback()) {
            ns.for_each_hbond_acceptor_bruteforce(state.coords_soa, h[0], h[1], h[2], kFarCut2, visit);
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
    // The energy the move gives (d, a): re-scored, carried, or 0 (dropped).
    auto pending_e = [&](int d, int a) {
        for (const auto& p : hb_ws.pending) if (p.d == d && p.a == a) return p.e;
        for (const auto& en : cache.as_donor(d)) {
            if (en.partner == a) return carried(hb_ws, d, en) ? en.energy : 0.0f;
        }
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
        cache.unlist_except(r, [&](const HBondStateCache::Entry& en) { return carried(hb_ws, r, en); });
    }
    for (const auto& p : hb_ws.pending) cache.add(p.d, p.a, p.e, p.fresh_until);
    cache.drift = hb_ws.pending_drift;
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
