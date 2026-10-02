#include "pymcpu/System.h"
#include "pymcpu/Context.h"
#include "pymcpu/AtomPermutation.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

using namespace mcpu;

// KIC FIX (fixed targets): see KicReference in System.h. Double-precision versions of
// GeometryUtils::calculate_bond_angle / calculate_dihedral (same formulas, same sign
// convention), which the move used to apply in float to the current coordinates.
namespace {
double kic_ref_angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
    const Eigen::Vector3d u = (a - b).normalized();
    const Eigen::Vector3d v = (c - b).normalized();
    return std::acos(std::clamp(u.dot(v), -1.0, 1.0));
}
double kic_ref_dihedral(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                        const Eigen::Vector3d& p3, const Eigen::Vector3d& p4) {
    const Eigen::Vector3d b1 = p2 - p1, b2 = p3 - p2, b3 = p4 - p3;
    const Eigen::Vector3d n1 = b1.cross(b2), n2 = b2.cross(b3);
    return std::atan2(b1.dot(n2) * b2.norm(), n1.dot(n2));
}
}  // namespace

void System::setKicReference(const Eigen::Matrix3Xf& start_coords) {
    if (start_coords.cols() != num_atoms) {
        throw std::invalid_argument(
            "System::setKicReference: expected 3 x " + std::to_string(num_atoms) +
            " start coordinates, got 3 x " + std::to_string(start_coords.cols()));
    }
    if (static_cast<int>(block_indices.size()) != num_residues) {
        throw std::runtime_error("System::setKicReference: set_block_indices must come first");
    }
    if (residue_contiguous_layout_) {
        // The coordinates are in build order; after a reorder the blocks are not.
        throw std::runtime_error(
            "System::setKicReference: must be called before a Context reorders the atoms");
    }
    const auto P = [&](int i) -> Eigen::Vector3d {
        return start_coords.col(i).cast<double>();
    };
    const size_t n = static_cast<size_t>(num_residues);
    const size_t nb = n > 0 ? n - 1 : 0;
    KicReference ref;
    ref.len_na.resize(n); ref.len_ac.resize(n); ref.ang_nac.resize(n);
    ref.len_cn.resize(nb); ref.ang_acn.resize(nb); ref.ang_cna.resize(nb); ref.omega.resize(nb);
    for (size_t k = 0; k < n; ++k) {
        const auto& b = block_indices[k];
        const Eigen::Vector3d N = P(b.bb_start), CA = P(b.ca_atom()), C = P(b.c_atom());
        ref.len_na[k] = (CA - N).norm();
        ref.len_ac[k] = (C - CA).norm();
        ref.ang_nac[k] = kic_ref_angle(N, CA, C);
        if (k + 1 < n) {
            const auto& b2 = block_indices[k + 1];
            const Eigen::Vector3d N2 = P(b2.bb_start), CA2 = P(b2.ca_atom());
            ref.len_cn[k] = (N2 - C).norm();
            ref.ang_acn[k] = kic_ref_angle(CA, C, N2);
            ref.ang_cna[k] = kic_ref_angle(C, N2, CA2);
            ref.omega[k] = kic_ref_dihedral(CA, C, N2, CA2);
        }
    }
    for (const auto* v : {&ref.len_na, &ref.len_ac, &ref.ang_nac, &ref.len_cn,
                          &ref.ang_acn, &ref.ang_cna, &ref.omega}) {
        for (double x : *v) {
            if (!std::isfinite(x)) {
                throw std::runtime_error("System::setKicReference: non-finite start geometry");
            }
        }
    }
    kic_reference_ = std::move(ref);
}

// Constructs an empty System. Potentials and topology data are added
// separately via addPotential() and the Python builder.
System::System(int atoms, int residues) 
    : num_atoms(atoms), 
      num_residues(residues)
{}

void System::set_energy_ignored_residues(const std::vector<int>& residues,
                                         EnergyMaskMode mode) {
    energy_ignored_mask_.assign(static_cast<size_t>(num_residues), 0);
    for (int r : residues) {
        if (r < 0 || r >= num_residues) {
            throw std::out_of_range(
                "System::set_energy_ignored_residues: residue index out of range");
        }
        energy_ignored_mask_[static_cast<size_t>(r)] = 1;
    }
    energy_mask_mode_ = mode;
    has_energy_mask_ = true;
}

void System::clear_energy_ignored_residues() noexcept {
    energy_ignored_mask_.clear();
    energy_mask_mode_ = EnergyMaskMode::IgnoreAll;
    has_energy_mask_ = false;
}

bool System::is_residue_energy_ignored(int res) const noexcept {
    if (!has_energy_mask_ || res < 0 ||
        res >= static_cast<int>(energy_ignored_mask_.size())) {
        return false;
    }
    return energy_ignored_mask_[static_cast<size_t>(res)] != 0;
}

void System::apply_atom_permutation(const AtomPermutation& perm) {
    if (perm.n_atoms() != num_atoms) {
        throw std::runtime_error("System::apply_atom_permutation: atom count mismatch");
    }
    if (perm.is_identity()) return;
    if (atoms_reordered_) {
        throw std::runtime_error("System::apply_atom_permutation: this System's atoms are already reordered");
    }
    perm.validate_inverses();

    auto map_id = [&](int ext) -> int {
        if (ext < 0) return ext;
        if (ext >= num_atoms) return ext;  // past-the-end sentinel
        return perm.to_internal(ext);
    };

    for (auto& b : block_indices) {
        b.bb_start = map_id(b.bb_start);
        b.c_start = map_id(b.c_start);
        b.o_start = map_id(b.o_start);
        b.sc_start = map_id(b.sc_start);
        b.h_start = map_id(b.h_start);
        b.res_begin = map_id(b.res_begin);
        // res_end is one-past-last; map last atom then +1 if in range
        if (b.res_end > 0 && b.res_end <= num_atoms) {
            b.res_end = map_id(b.res_end - 1) + 1;
        } else if (b.res_end > num_atoms) {
            b.res_end = num_atoms;
        }
    }

    // Rebuild DownstreamCache from remapped BlockIndices so first_* are
    // residue-local (multi-residue contiguous spans are invalid after reorder).
    const int n_res = num_residues;
    const int sc_end = sc_segment_end();
    const int h_end = num_atoms;
    downstream.first_sc_of_residue.resize(static_cast<size_t>(n_res));
    downstream.first_o_of_residue.resize(static_cast<size_t>(n_res));
    downstream.first_h_of_residue.resize(static_cast<size_t>(n_res));
    for (int r = 0; r < n_res; ++r) {
        const auto& b = block_indices[static_cast<size_t>(r)];
        downstream.first_o_of_residue[static_cast<size_t>(r)] = b.o_start;
        if (b.sc_start >= 0) {
            downstream.first_sc_of_residue[static_cast<size_t>(r)] = b.sc_start;
        } else {
            int next = sc_end;
            for (int k = r + 1; k < n_res; ++k) {
                if (block_indices[static_cast<size_t>(k)].sc_start >= 0) {
                    next = block_indices[static_cast<size_t>(k)].sc_start;
                    break;
                }
            }
            downstream.first_sc_of_residue[static_cast<size_t>(r)] = next;
        }
        if (b.h_start >= 0) {
            downstream.first_h_of_residue[static_cast<size_t>(r)] = b.h_start;
        } else {
            int next = h_end;
            for (int k = r + 1; k < n_res; ++k) {
                if (block_indices[static_cast<size_t>(k)].h_start >= 0) {
                    next = block_indices[static_cast<size_t>(k)].h_start;
                    break;
                }
            }
            downstream.first_h_of_residue[static_cast<size_t>(r)] = next;
        }
    }

    // atom_to_residue is currently indexed by external id; rewrite to internal.
    std::vector<int> new_a2r(static_cast<size_t>(num_atoms), -1);
    for (int e = 0; e < num_atoms; ++e) {
        const int i = perm.to_internal(e);
        new_a2r[static_cast<size_t>(i)] = atom_to_residue[static_cast<size_t>(e)];
    }
    atom_to_residue.swap(new_a2r);

    // chi_atom_indices_ holds atom ids in the pre-permutation space; remap
    // through the same map_id() used for BlockIndices above (identical -1 /
    // past-the-end sentinel handling).
    for (auto& res_chis : chi_atom_indices_) {
        for (auto& atoms4 : res_chis) {
            for (int& a : atoms4) a = map_id(a);
        }
    }

    // chi_moved_ranges_ holds [lo, hi) atom-id ranges in the pre-permutation
    // space, same treatment as res_end above (map the last INCLUDED atom,
    // then +1 -- hi itself is an exclusive endpoint, not a real atom id, so
    // it can't be passed through map_id() directly).
    for (auto& res_ranges : chi_moved_ranges_) {
        for (auto& range : res_ranges) {
            if (range[0] < 0) continue;  // {-1,-1} sentinel: unused chi slot
            const int lo = map_id(range[0]);
            const int hi = map_id(range[1] - 1) + 1;
            range[0] = lo;
            range[1] = hi;
        }
    }
    permute_potentials(perm);

#if !defined(NDEBUG)
    for (const auto& b : block_indices) {
        if (b.bb_start < 0 || b.bb_start >= num_atoms) {
            throw std::runtime_error("System::apply_atom_permutation: bad bb_start");
        }
        if (b.o_start >= num_atoms) {
            throw std::runtime_error("System::apply_atom_permutation: bad o_start");
        }
        if (b.sc_start >= num_atoms) {
            throw std::runtime_error("System::apply_atom_permutation: bad sc_start");
        }
        if (b.h_start >= num_atoms) {
            throw std::runtime_error("System::apply_atom_permutation: bad h_start");
        }
    }
#endif
}

void System::apply_residue_contiguous_blocks(std::vector<BlockIndices> blocks,
                                             const AtomPermutation& perm) {
    if (static_cast<int>(blocks.size()) != num_residues) {
        throw std::runtime_error("apply_residue_contiguous_blocks: residue count mismatch");
    }
    if (perm.n_atoms() != num_atoms) {
        throw std::runtime_error("apply_residue_contiguous_blocks: atom count mismatch");
    }
    if (atoms_reordered_) {
        throw std::runtime_error("apply_residue_contiguous_blocks: this System's atoms are already reordered");
    }
    block_indices = std::move(blocks);
    residue_contiguous_layout_ = true;

    // chi_atom_indices_ holds atom ids in the pre-permutation space; this
    // function replaces block_indices wholesale (residue correspondence is
    // preserved, only intra-residue atom order changes), so chi_atom_indices_
    // just needs remapping through perm, not replacing.
    auto map_id = [&](int ext) -> int {
        if (ext < 0) return ext;
        if (ext >= num_atoms) return ext;  // past-the-end sentinel
        return perm.to_internal(ext);
    };
    for (auto& res_chis : chi_atom_indices_) {
        for (auto& atoms4 : res_chis) {
            for (int& a : atoms4) a = map_id(a);
        }
    }

    // chi_moved_ranges_: same [lo, hi) remap as apply_atom_permutation above.
    for (auto& res_ranges : chi_moved_ranges_) {
        for (auto& range : res_ranges) {
            if (range[0] < 0) continue;  // {-1,-1} sentinel: unused chi slot
            const int lo = map_id(range[0]);
            const int hi = map_id(range[1] - 1) + 1;
            range[0] = lo;
            range[1] = hi;
        }
    }

    std::vector<int> new_a2r(static_cast<size_t>(num_atoms), -1);
    for (int i = 0; i < num_atoms; ++i) {
        const int e = perm.to_external(i);
        new_a2r[static_cast<size_t>(i)] = atom_to_residue[static_cast<size_t>(e)];
    }
    atom_to_residue.swap(new_a2r);

    // Residue-local DownstreamCache (no multi-residue span assumptions).
    downstream.first_sc_of_residue.resize(static_cast<size_t>(num_residues));
    downstream.first_o_of_residue.resize(static_cast<size_t>(num_residues));
    downstream.first_h_of_residue.resize(static_cast<size_t>(num_residues));
    for (int r = 0; r < num_residues; ++r) {
        const auto& b = block_indices[static_cast<size_t>(r)];
        downstream.first_o_of_residue[static_cast<size_t>(r)] = b.o_start;
        downstream.first_sc_of_residue[static_cast<size_t>(r)] =
            b.sc_start >= 0 ? b.sc_start : num_atoms;
        downstream.first_h_of_residue[static_cast<size_t>(r)] =
            b.h_start >= 0 ? b.h_start : num_atoms;
    }
    permute_potentials(perm);

#if !defined(NDEBUG)
    for (const auto& b : block_indices) {
        if (!b.has_residue_span()) {
            throw std::runtime_error("apply_residue_contiguous_blocks: missing res span");
        }
        if (b.ca_atom() != b.bb_start + 1) {
            throw std::runtime_error("apply_residue_contiguous_blocks: N/CA not adjacent");
        }
    }
#endif
}

void System::permute_potentials(const AtomPermutation& perm) {
    for (auto& potential : potentials) potential->permute_atom_indices(perm);
    applied_perm_ = perm;
    atoms_reordered_ = true;
}

bool System::is_amide_h_atom(int atom_id) const noexcept {
    if (atom_id < 0 || atom_id >= num_atoms) return false;
    if (total_h_atoms <= 0) return false;
    if (!residue_contiguous_layout_) {
        return atom_id >= h_segment_start();
    }
    if (atom_id >= static_cast<int>(atom_to_residue.size())) return false;
    const int r = atom_to_residue[static_cast<size_t>(atom_id)];
    if (r < 0 || r >= num_residues) return false;
    return block_indices[static_cast<size_t>(r)].h_start == atom_id;
}

namespace {
// group -> name for every group in use; unnamed groups map to "".
std::map<int, std::string> collect_energy_terms(
    const std::vector<std::shared_ptr<Potential>>& potentials) {
    std::map<int, std::string> by_group;
    std::map<std::string, int> by_name;
    for (const auto& p : potentials) {
        const int g = p->getEnergyGroup();
        const std::string& name = p->getName();
        auto it = by_group.emplace(g, std::string()).first;
        if (name.empty()) continue;
        if (!it->second.empty() && it->second != name) {
            throw std::invalid_argument(
                "energy group " + std::to_string(g) + " has two names: '" +
                it->second + "' and '" + name + "'");
        }
        auto [nit, inserted] = by_name.emplace(name, g);
        if (!inserted && nit->second != g) {
            throw std::invalid_argument(
                "energy term name '" + name + "' is used by groups " +
                std::to_string(nit->second) + " and " + std::to_string(g));
        }
        it->second = name;
    }
    return by_group;
}
}  // namespace

int System::addPotential(std::shared_ptr<Potential> potential) {
    if (!potential) {
        throw std::invalid_argument("System::addPotential: null potential pointer");
    }
    potentials.push_back(std::move(potential));
    try {
        collect_energy_terms(potentials);
        // Callers build potentials from build-order ids (forcefield.blocks,
        // ordered_atom_list); after a reorder those are not storage ids.
        if (atoms_reordered_) potentials.back()->permute_atom_indices(applied_perm_);
    } catch (...) {
        potentials.pop_back();
        throw;
    }
    return static_cast<int>(potentials.size() - 1);
}

std::vector<std::pair<int, std::string>> System::energyTerms() const {
    std::vector<std::pair<int, std::string>> out;
    for (auto& [g, name] : collect_energy_terms(potentials)) {
        out.emplace_back(g, name.empty() ? "group_" + std::to_string(g) : name);
    }
    return out;
}

int System::getNumAtoms() const noexcept { 
    return num_atoms; 
}

int System::getNumResidues() const noexcept { 
    return num_residues; 
}

const std::vector<std::shared_ptr<Potential>>& System::getPotentials() const {
    return potentials;
}

float System::getTotalEnergyRaw(const Context& ctx, const State& state, int target_group) const {
    float total = 0.0f;
    for (const auto& potential : potentials) {
        if (!potential->isEnabled()) continue;
        if (target_group == -1 || potential->getEnergyGroup() == target_group) {
            total += potential->calculateEnergy(ctx, state);
        }
    }
    return total;
}

float System::getTotalEnergy(const Context& ctx, const State& state, int target_group) const {
    const EnergyWeights& weights = ctx.energyWeights();
    float total = 0.0f;
    for (const auto& potential : potentials) {
        if (!potential->isEnabled()) continue;
        if (target_group == -1 || potential->getEnergyGroup() == target_group) {
            const float raw = potential->calculateEnergy(ctx, state);
            total += weights.weight_for_group(potential->getEnergyGroup()) * raw;
        }
    }
    return total;
}

float System::getDeltaEnergyRaw(
    const Context& ctx,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch) const
{
    float delta = 0.0f;
    for (const auto& potential : potentials) {
        if (!potential->isEnabled()) continue;
        delta += potential->calculateEnergyChange(
                    ctx, old_state, proposed_state, patch)
                     .delta_energy;
    }
    return delta;
}

float System::getDeltaEnergy(
    const Context& ctx,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch) const
{
    const EnergyWeights& weights = ctx.energyWeights();
    float delta = 0.0f;
    for (const auto& potential : potentials) {
        if (!potential->isEnabled()) continue;
        const float w = weights.weight_for_group(potential->getEnergyGroup());
        delta += w * potential->calculateEnergyChange(
                         ctx, old_state, proposed_state, patch)
                         .delta_energy;
    }
    return delta;
}

EnergyChangeResult System::evaluateDeltaEnergy(
    const Context& ctx,
    const State& old_state,
    const State& proposed_state,
    const ProposalPatch& patch) const
{
    const EnergyWeights& weights = ctx.energyWeights();
    EnergyChangeResult out = EnergyChangeResult::finite(0.f);
    for (const auto& potential : potentials) {
        if (!potential->isEnabled()) continue;
        const int g = potential->getEnergyGroup();
        const float w = weights.weight_for_group(g);
        std::uint64_t* slot = ctx.energy_delta_ns_slot(g);
        EnergyChangeResult r;
        if (slot) {
            const auto t0 = std::chrono::steady_clock::now();
            r = potential->calculateEnergyChange(
                ctx, old_state, proposed_state, patch);
            const auto t1 = std::chrono::steady_clock::now();
            *slot += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
                    .count());
        } else {
            r = potential->calculateEnergyChange(
                ctx, old_state, proposed_state, patch);
        }
        out.delta_energy += w * r.delta_energy;
        if (r.reject_reason != RejectReason::None) {
            out.reject_reason = r.reject_reason;
            return out;
        }
    }
    return out;
}

TotalEnergyResult System::evaluateTotalEnergy(
    const Context& ctx,
    const State& state,
    int target_group) const
{
    const EnergyWeights& weights = ctx.energyWeights();
    TotalEnergyResult out;
    for (const auto& potential : potentials) {
        if (!potential->isEnabled()) continue;
        if (target_group != -1 && potential->getEnergyGroup() != target_group) {
            continue;
        }
        const float raw = potential->calculateEnergy(ctx, state);
        const float w = weights.weight_for_group(potential->getEnergyGroup());
        out.energy += w * raw;
        if (potential->canHardReject()) {
            const RejectReason rr = potential->rejectionForEnergy(raw);
            if (rr != RejectReason::None) {
                out.reject_reason = rr;
            }
        }
    }
    return out;
}

EnergyBreakdown System::energyBreakdown(const Context& ctx, const State& state) const {
    EnergyBreakdown out;
    const EnergyWeights& weights = ctx.energyWeights();
    for (const auto& potential : potentials) {
        if (!potential->isEnabled()) continue;
        const int g = potential->getEnergyGroup();
        const float raw = potential->calculateEnergy(ctx, state);
        const float w = weights.weight_for_group(g);
        out.raw_by_group[g] += raw;
        out.weighted_by_group[g] += w * raw;
        out.raw_total += raw;
        out.weighted_total += w * raw;
    }
    return out;
}
