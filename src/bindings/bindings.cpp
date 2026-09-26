#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h> // Necessary for Eigen matrices
#include <pybind11/stl.h>   // Necessary for std::vector
#include <sstream>
#include <iomanip>

#include "pymcpu/BuildConfig.h"  // GENERATED -- see cmake/BuildConfig.h.in

#include "pymcpu/State.h"
#include "pymcpu/System.h"
#include "pymcpu/Context.h"
#include "pymcpu/EnergyWeights.h"
#include "pymcpu/Integrator.h"

// Potentials
#include "pymcpu/forces/mcpu/mcpu08/MuPotential.h"
#include "pymcpu/forces/mcpu/common/TripletPotential.h"
#include "pymcpu/forces/mcpu/common/SideChainTripletPotential.h"
#include "pymcpu/forces/mcpu/common/HydrogenBondPotential.h"
#include "pymcpu/forces/korp/common/OrientationalPairMap.h"
#include "pymcpu/forces/korp/common/OrientationalPairPotential.h"
#include "pymcpu/forces/korp/common/CalphaExcludedVolumePotential.h"
#include "pymcpu/forces/mcpu/common/AromaticPotential.h"
#include "pymcpu/forces/bias/QBiasPotential.h"

// Tripeptide Closure
#include "pymcpu/moves/TripeptideClosure.h"
#include "pymcpu/moves/RotamerLibrary.h"
#include "pymcpu/moves/RamaMixtureLibrary.h"

#include "pymcpu/ProposalPatch.h"
#include "pymcpu/testing/PhysicsVerifier.h"
#include "pymcpu/utils/CoordSyncStats.h"
#include "pymcpu/reporters/Reporter.h"
#include "pymcpu/reporters/EnergyReporter.h"
#include "pymcpu/reporters/SimulationReporter.h"
#include "pymcpu/reporters/XTCReporter.h"

namespace py = pybind11;
using namespace mcpu;
// mcpu08 is the fit tag; see include/pymcpu/forces/README.md.
namespace m08 = mcpu::forces::mcpu08;

// The module name MUST be mcpu_core
PYBIND11_MODULE(mcpu_core, m) {
    // Ensure NumPy C-API is initialized before any Eigen↔ndarray conversions.
    py::module_::import("numpy");

    m.doc() = "PyMCPU: A fast Monte Carlo protein folding engine. "
              "C++ physics core with Python interface.";

    // Reporters (bindings match current reporter headers only)
    py::class_<mcpu::Reporter, std::shared_ptr<mcpu::Reporter>>(m, "Reporter",
        "Base class of the output reporters. Exposed so that\n"
        "Simulation.add_reporter has a type to accept; not intended to be\n"
        "subclassed from Python.");
    py::class_<mcpu::EnergyComponents>(m, "EnergyComponents")
        .def(py::init<>())
        .def_readwrite("total", &mcpu::EnergyComponents::total)
        .def_readwrite("mu", &mcpu::EnergyComponents::mu)
        .def_readwrite("backbone_torsion", &mcpu::EnergyComponents::backbone_torsion)
        .def_readwrite("sidechain_torsion", &mcpu::EnergyComponents::sidechain_torsion)
        .def_readwrite("hydrogen_bond", &mcpu::EnergyComponents::hydrogen_bond)
        .def_readwrite("aromatic", &mcpu::EnergyComponents::aromatic)
        .def_readwrite("native_contacts_bias", &mcpu::EnergyComponents::native_contacts_bias)
        .def("to_dict", &mcpu::EnergyComponents::to_dict)
        .def("get", &mcpu::EnergyComponents::get, py::arg("component"));
    py::class_<mcpu::XtcReporter, mcpu::Reporter, std::shared_ptr<mcpu::XtcReporter>>(m, "XtcReporter",
        "Writes coordinates to a GROMACS XTC trajectory every\n"
        "report_interval steps. Pass MCPUForceField.inverse_mapping so\n"
        "frames come out in the input topology order. XTC is the preferred\n"
        "format: it is compressed, and it is truncated correctly when a run\n"
        "resumes from a checkpoint (DCD is not).")
        .def(py::init<const std::string&, int, const std::vector<int>&, bool>(),
             py::arg("xtc_filename"),
             py::arg("report_interval"),
             py::arg("inverse_mapping") = std::vector<int>{},
             py::arg("append") = false)
        .def("flush", &mcpu::XtcReporter::flush)
        .def("n_frames_written", &mcpu::XtcReporter::n_frames_written)
        .def("filename", &mcpu::XtcReporter::filename,
             py::return_value_policy::reference_internal);
    py::class_<mcpu::EnergyReporter, mcpu::Reporter, std::shared_ptr<mcpu::EnergyReporter>>(m, "EnergyReporter",
        "Writes a CSV row every report_interval steps holding the per-group\n"
        "energies and the cumulative move accept/attempt counters.\n\n"
        "This is a writer, not a buffer: there is no accessor to read\n"
        "energies back out of it, so read the CSV. The Total column includes\n"
        "the native-contacts bias when one is enabled, which makes it right\n"
        "for monitoring and wrong for MBAR reweighting.")
        .def(py::init<const std::string&, int, bool>(),
             py::arg("energy_filename"),
             py::arg("report_interval"),
             py::arg("append") = false)
        .def("n_frames_written", &mcpu::EnergyReporter::n_frames_written)
        .def("filename", &mcpu::EnergyReporter::filename,
             py::return_value_policy::reference_internal)
        .def("set_walker_id", &mcpu::EnergyReporter::set_walker_id, py::arg("walker_id"))
        .def("walker_id", &mcpu::EnergyReporter::walker_id);
    py::class_<mcpu::SimulationReporter, mcpu::Reporter, std::shared_ptr<mcpu::SimulationReporter>>(m, "SimulationReporter",
        "Prints a short progress line to stdout every report_interval\n"
        "steps. For interactive use only -- it takes no filename and has no\n"
        "verbosity control. Use EnergyReporter for anything you intend to\n"
        "keep or parse.")
        .def(py::init<int>(), py::arg("report_interval"));

    py::class_<Solution>(m, "Solution")
        .def(py::init<>())                     // Bind the empty default constructor
        .def_readwrite("r_n", &Solution::r_n)  // Change from readonly to readwrite
        .def_readwrite("r_a", &Solution::r_a)
        .def_readwrite("r_c", &Solution::r_c);

    py::class_<TripeptideSolver>(m, "TripeptideSolver")
        .def(py::init<>())
        .def("initialize", &TripeptideSolver::initialize, 
             "Initialize solver with bond lengths, angles, and omega torsions",
             py::arg("b_len"), py::arg("b_ang"), py::arg("t_ang"))
        .def("solve", &TripeptideSolver::solve,
             "Solve for 3D coordinates given 4 anchor points",
             py::arg("r_n1"), py::arg("r_a1"), py::arg("r_a3"), py::arg("r_c3"))
        .def("get_xi", &TripeptideSolver::get_xi)
        .def("get_eta", &TripeptideSolver::get_eta)
        .def("get_delta", &TripeptideSolver::get_delta)
        .def("get_polynomial_coefficients", &TripeptideSolver::get_polynomial_coefficients)
        .def("calculate_jacobian", &TripeptideSolver::calculate_jacobian, py::arg("solution"));

    py::class_<mcpu::RotamerComponent>(m, "RotamerComponent")
        .def(py::init<>())
        .def_readwrite("log_weight", &mcpu::RotamerComponent::log_weight)
        .def_readwrite("mean", &mcpu::RotamerComponent::mean)
        .def_readwrite("sigma", &mcpu::RotamerComponent::sigma);

    py::class_<mcpu::RotamerLibrary>(m, "RotamerLibrary")
        .def(py::init<>())
        .def("add_residue_type", &mcpu::RotamerLibrary::add_residue_type,
             py::arg("amino_idx"), py::arg("weights"), py::arg("means"), py::arg("sigmas"),
             "Registers amino_idx's rotamer table: per-row weight (need not "
             "already sum to 1) plus per-chi (mean, sigma) in radians.")
        .def("num_rows", &mcpu::RotamerLibrary::num_rows, py::arg("amino_idx"))
        .def("sample_row", &mcpu::RotamerLibrary::sample_row,
             py::arg("amino_idx"), py::arg("u01"))
        .def("row", &mcpu::RotamerLibrary::row, py::arg("amino_idx"), py::arg("row_idx"),
             py::return_value_policy::reference_internal)
        .def("log_mixture_density", &mcpu::RotamerLibrary::log_mixture_density,
             py::arg("amino_idx"), py::arg("ntorsions"), py::arg("chi"));

    py::class_<mcpu::RamaComponent>(m, "RamaComponent")
        .def(py::init<>())
        .def_readwrite("log_weight", &mcpu::RamaComponent::log_weight)
        .def_readwrite("mean", &mcpu::RamaComponent::mean)
        .def_readwrite("cov", &mcpu::RamaComponent::cov)
        .def_readwrite("chol_l11", &mcpu::RamaComponent::chol_l11)
        .def_readwrite("chol_l21", &mcpu::RamaComponent::chol_l21)
        .def_readwrite("chol_l22", &mcpu::RamaComponent::chol_l22);

    py::class_<mcpu::RamaMixtureLibrary>(m, "RamaMixtureLibrary")
        .def(py::init<int>(), py::arg("n_wrap") = 1)
        .def("add_residue_type", &mcpu::RamaMixtureLibrary::add_residue_type,
             py::arg("amino_idx"), py::arg("weights"), py::arg("means"), py::arg("covariances"),
             "Registers amino_idx's (phi, psi) mixture: per-row weight (need "
             "not already sum to 1), per-row (phi, psi) mean in radians, and "
             "per-row covariance {c11, c12, c22} in radians^2.")
        .def("num_rows", &mcpu::RamaMixtureLibrary::num_rows, py::arg("amino_idx"))
        .def("sample_row", &mcpu::RamaMixtureLibrary::sample_row,
             py::arg("amino_idx"), py::arg("u01"))
        .def("row", &mcpu::RamaMixtureLibrary::row, py::arg("amino_idx"), py::arg("row_idx"),
             py::return_value_policy::reference_internal)
        .def("log_mixture_density", &mcpu::RamaMixtureLibrary::log_mixture_density,
             py::arg("amino_idx"), py::arg("phi_psi"))
        .def("n_wrap", &mcpu::RamaMixtureLibrary::n_wrap);

    py::class_<BackboneTorsionAngles>(m, "BackboneTorsionAngles")
        .def(py::init<float, float, float, float>())
        .def_readwrite("phi",  &BackboneTorsionAngles::phi)
        .def_readwrite("psi",  &BackboneTorsionAngles::psi)
        .def_readwrite("p_ca",  &BackboneTorsionAngles::pCA)
        .def_readwrite("b_ca",  &BackboneTorsionAngles::bCA);
    
    py::class_<SidechainTorsionAngles>(m, "SidechainTorsionAngles")
        .def(py::init<>())
        .def_readwrite("chi_angles", &SidechainTorsionAngles::chi_angles);

    py::class_<State>(m, "State")
        .def(py::init<int, int>(), py::arg("atoms"), py::arg("residues"))
        .def(py::init<const State&>())
        .def_property(
            "coords",
            [](const State& s) { return s.coords_as_eigen(); },
            [](State& s, const Eigen::Matrix3Xf& m) { s.set_coords_from_eigen(m); })
        .def_property_readonly("current_energy", &State::getEnergy)
        .def_readwrite("backbone_torsions",  &State::backbone_torsions)
        .def_readwrite("sidechain_torsions", &State::sidechain_torsions);

    py::class_<ProposalPatch>(m, "ProposalPatch")
        .def(py::init<int>(), py::arg("num_atoms"))
        .def_readwrite("is_valid", &ProposalPatch::is_valid)
        .def_readwrite("is_rigid", &ProposalPatch::is_rigid)
        .def_readwrite("first_affected_residue", &ProposalPatch::first_affected_residue)
        .def_readwrite("last_affected_residue", &ProposalPatch::last_affected_residue)
        .def_readwrite("bb_atom_moved", &ProposalPatch::bb_atom_moved)
        .def_readwrite("sc_atom_moved", &ProposalPatch::sc_atom_moved)
        .def_readwrite("o_atom_moved", &ProposalPatch::o_atom_moved)
        .def_readwrite("h_atom_moved", &ProposalPatch::h_atom_moved)
        .def_readwrite("moving_atoms", &ProposalPatch::moving_atoms)
        .def_readwrite("moved_indices", &ProposalPatch::moved_indices)
        .def("mark_moved", &ProposalPatch::mark_moved)
        .def("add_distorted_bb_residue", &ProposalPatch::add_distorted_bb_residue)
        .def("add_distorted_sc_residue", &ProposalPatch::add_distorted_sc_residue);

    py::enum_<MoveKind>(m, "MoveKind")
        .value("Pivot", MoveKind::Pivot)
        .value("KIC", MoveKind::KIC)
        .value("Sidechain", MoveKind::Sidechain)
        .value("Other", MoveKind::Other);


    py::class_<EnergyWeights>(m, "EnergyWeights")
        .def(py::init<>())
        .def_readwrite("use_legacy_weights", &EnergyWeights::use_legacy_weights)
        .def_readwrite("hbond_rdthree", &EnergyWeights::hbond_rdthree)
        .def("set_legacy_defaults", &EnergyWeights::set_legacy_defaults)
        .def("set_unweighted", &EnergyWeights::set_unweighted)
        .def("set_use_legacy_weights", &EnergyWeights::set_use_legacy_weights, py::arg("on"))
        .def("set_energy_weight", &EnergyWeights::set_energy_weight,
             py::arg("group_id"), py::arg("w"))
        .def("weight_for_group", &EnergyWeights::weight_for_group, py::arg("group_id"))
        .def("outer_weight", &EnergyWeights::outer_weight, py::arg("group_id"))
        .def_property_readonly_static("LEGACY_MU",
            [](py::object) { return EnergyWeights::kLegacyMu; })
        .def_property_readonly_static("LEGACY_BB_TOR",
            [](py::object) { return EnergyWeights::kLegacyBbTor; })
        .def_property_readonly_static("LEGACY_SC_TOR",
            [](py::object) { return EnergyWeights::kLegacyScTor; })
        .def_property_readonly_static("LEGACY_HBOND",
            [](py::object) { return EnergyWeights::kLegacyHBond; })
        .def_property_readonly_static("LEGACY_ARO",
            [](py::object) { return EnergyWeights::kLegacyAro; })
        .def_property_readonly_static("LEGACY_RDTHREE",
            [](py::object) { return EnergyWeights::kLegacyRdthree; });

    py::class_<Context>(m, "Context",
        "Owns the mutable state of a simulation: coordinates, the running\n"
        "energy, neighbour lists and the per-group energy weights.\n\n"
        "Constructed from a finished System. Note that set_positions() does\n"
        "NOT seed the running total energy -- call calculate_total_energy(-1)\n"
        "once afterwards if you drive a Context directly. Simulation does\n"
        "this for you on its first step().\n\n"
        "Members beyond positions, energy and state are performance and\n"
        "diagnostic knobs, and are not part of the stable API.")
        .def(py::init<std::shared_ptr<System>>(), py::arg("system"),
         py::keep_alive<1, 2>())
        .def("get_system",
             static_cast<System& (Context::*)()>(&Context::getSystem),
             py::return_value_policy::reference_internal)
        .def("get_state",
             static_cast<State& (Context::*)()>(&Context::getState),
             py::return_value_policy::reference_internal)
        .def("set_positions", &Context::setPositions)
        .def_property(
            "coords",
            &Context::coords_for_python,
            &Context::set_coords_from_python,
            "Coordinates in external (build) order by default; set_output_internal_order(True) for storage order.")
        .def("coords_for_python", &Context::coords_for_python)
        .def("set_coords_from_python", &Context::set_coords_from_python)
        .def("set_atom_reorder_mode",
             py::overload_cast<const std::string&>(&Context::set_atom_reorder_mode),
             py::arg("mode"),
             "Atom locality reorder: \"off\" (default) or \"init_only\".")
        .def("get_atom_reorder_mode", &Context::get_atom_reorder_mode)
        .def("set_output_internal_order", &Context::set_output_internal_order, py::arg("on"))
        .def("output_internal_order", &Context::output_internal_order)
        .def("atom_permutation_info",
             [](const Context& c) {
                 const auto info = c.atom_permutation_info();
                 const auto& perm = c.atom_permutation();
                 py::dict d;
                 d["enabled"] = info.enabled;
                 d["mode"] = info.mode;
                 d["permutation_checksum"] = info.permutation_checksum;
                 d["n_atoms"] = info.n_atoms;
                 d["n_res"] = info.n_res;
                 d["int_to_ext"] = perm.int_to_ext;
                 d["ext_to_int"] = perm.ext_to_int;
                 return d;
             })
        .def("calculate_total_energy", &Context::calculate_total_energy, py::arg("target_group") = -1)
        .def("calculate_total_energy_raw", &Context::calculate_total_energy_raw,
             py::arg("target_group") = -1)
        .def("calculate_delta_energy", &Context::calculate_delta_energy,
             py::arg("proposed_state"), py::arg("patch"))
        .def("has_hard_constraint_violation",
             &Context::has_hard_constraint_violation)
        .def("has_steric_clash", &Context::has_steric_clash)
        .def("energy_breakdown",
             [](const Context& c, bool weighted) {
                 const EnergyBreakdown bd = c.energy_breakdown();
                 py::dict d;
                 d["raw_total"] = bd.raw_total;
                 d["weighted_total"] = bd.weighted_total;
                 py::dict by_group;
                 const auto& src = weighted ? bd.weighted_by_group : bd.raw_by_group;
                 for (const auto& kv : src) {
                     by_group[py::int_(kv.first)] = kv.second;
                 }
                 d["by_group"] = by_group;
                 d["weighted"] = weighted;
                 d["use_legacy_weights"] = c.use_legacy_weights();
                 return d;
             },
             py::arg("weighted") = true,
             "Return per-group energies. weighted=True uses legacy outer weights "
             "(incl. HBond RDTHREE_CON); weighted=False returns raw per-potential energies.")
        .def("set_use_legacy_weights", &Context::set_use_legacy_weights, py::arg("on"))
        .def("use_legacy_weights", &Context::use_legacy_weights)
        .def("set_energy_weight", &Context::set_energy_weight,
             py::arg("group_id"), py::arg("w"),
             "Set outer weight for an energy group. For group 4 (HBond), the effective "
             "multiplier is outer * hbond_rdthree when use_legacy_weights is True.")
        .def("get_energy_weights",
             [](const Context& c) {
                 py::dict d;
                 for (int g = 1; g < EnergyWeights::kMaxTrackedGroup; ++g) {
                     d[py::int_(g)] = c.energyWeights().weight_for_group(g);
                 }
                 d["use_legacy_weights"] = c.use_legacy_weights();
                 d["hbond_rdthree"] = c.energyWeights().hbond_rdthree;
                 return d;
             })
        .def("set_q_bias", &Context::setQBias, py::arg("k_bias"), py::arg("n_target"),
             "Harmonic umbrella on hard native-contact count N: "
             "U = 0.5 * k_bias * (N - n_target)^2. "
             "(Legacy name; prefer set_native_contacts_bias.)")
        .def("set_native_contacts_bias", &Context::setNativeContactsBias,
             py::arg("k_bias"), py::arg("n_target"),
             "Harmonic umbrella on hard native-contact count N: "
             "U = 0.5 * k_bias * (N - n_target)^2.")
        .def("native_contacts_bias_k", &Context::getQBiasK)
        .def("native_contacts_bias_n_target", &Context::getQBiasTarget)
        .def("neighbor_aabb_rebuilds",
             [](const Context& c) {
                 return c.neighborStats().num_aabb_rebuild_accept;
             })
        .def("neighbor_dense_cap_fallbacks",
             [](const Context& c) {
                 return c.neighborStats().num_dense_cap_fallback;
             })
        .def("hbond_index_ok",
             [](const Context& c) {
                 return c.neighbors().count_hbond_candidate_mismatches(c.getState().coords_soa) == 0;
             })
        .def("hbond_uses_fallback",
             [](const Context& c) { return c.neighbors().hbondUsesFallback(); })
        .def("print_neighbor_audit",
             [](const Context& c) {
                 c.neighbors().maybe_print_neighbor_audit("Context::print_neighbor_audit");
             })
        .def("hbond_backend_name",
             [](const Context& c) { return c.neighbors().hbond_backend_name(); })
        .def("mu_backend_name",
             [](const Context& c) { return c.neighbors().mu_backend_name(); })
        .def("set_mu_skin", &Context::set_mu_skin, py::arg("skin"))
        .def("mu_skin", &Context::mu_skin)
        .def("mu_verlet_enabled", &Context::mu_verlet_enabled)
        .def("set_mu_verlet_enabled", &Context::set_mu_verlet_enabled, py::arg("on"),
             "Enable/disable Mu Verlet without changing skin (denselist geometry).")
        .def("set_skip_rigid_mm", &Context::set_skip_rigid_mm, py::arg("on"),
             "Skip moved-moved Mu pairs for rigid pivots (ΔE_mm=0).")
        .def("skip_rigid_mm", &Context::skip_rigid_mm)
        .def("set_use_cell_pair", &Context::set_use_cell_pair, py::arg("on"),
             "Cell-pair denselist Mu (default true). False = per-atom walks.")
        .def("use_cell_pair", &Context::use_cell_pair)
        .def("set_cell_pair_min_moved", &Context::set_cell_pair_min_moved,
             py::arg("n"),
             "Min moved atoms to use cell-pair (default 20; SC uses per-atom).")
        .def("cell_pair_min_moved", &Context::cell_pair_min_moved)
        .def("set_clash_first_min_moved", &Context::set_clash_first_min_moved,
             "Minimum moved-atom count for the Mu clash-first pass (default 50). "
             "Below it the pass is skipped; it costs ~10% on very small moves and "
             "gains ~14% on large pivots.")
        .def("clash_first_min_moved", &Context::clash_first_min_moved)
        .def_property_readonly(
            "mu_potential",
            [](Context& c) -> m08::MuPotential* { return c.mu_potential(); },
            py::return_value_policy::reference_internal,
            "First MuPotential, or None.")
        .def("set_mm_clash_margin", &Context::set_mm_clash_margin, py::arg("margin_r2"),
             "ADDED: MM clash margin (Å²). Also: MCPU_MM_CLASH_MARGIN.")
        .def("mm_clash_margin", &Context::mm_clash_margin)
        .def("set_mm_double_boundary", &Context::set_mm_double_boundary, py::arg("on"),
             "ADDED: double MM boundary clash check. Also: MCPU_MM_DOUBLE_BOUNDARY=1.")
        .def("mm_double_boundary", &Context::mm_double_boundary)
        .def("set_verlet_moved_threshold", &Context::set_verlet_moved_threshold,
             py::arg("n"),
             "Force CellOnly when n_moved > n (0 ⇒ always CellOnly).")
        .def("verlet_moved_threshold", &Context::verlet_moved_threshold)
        .def("set_verlet_partial_threshold", &Context::set_verlet_partial_threshold,
             py::arg("n"),
             "Partial Verlet CSR rebuild on accept when n_moved ≤ n (default 50).")
        .def("verlet_partial_threshold", &Context::verlet_partial_threshold)
        .def("set_invalidate_verlet_on_pivot_accept",
             &Context::set_invalidate_verlet_on_pivot_accept, py::arg("on"))
        .def("invalidate_verlet_on_pivot_accept",
             &Context::invalidate_verlet_on_pivot_accept)
        .def("set_mu_cell_size_scale", &Context::set_mu_cell_size_scale, py::arg("scale"))
        .def("mu_cell_size_scale", &Context::mu_cell_size_scale)
        .def("set_mu_cell_size_angstrom", &Context::set_mu_cell_size_angstrom,
             py::arg("angstrom"))
        .def("mu_cell_size_angstrom", &Context::mu_cell_size_angstrom)
        .def("set_mu_cell_size_min_angstrom", &Context::set_mu_cell_size_min_angstrom,
             py::arg("angstrom"))
        .def("mu_cell_size_min_angstrom", &Context::mu_cell_size_min_angstrom)
        .def("effective_mu_cell_size_A", &Context::effective_mu_cell_size_A)
        .def("mu_cell_size_A",
             [](const Context& c) { return c.neighbors().mu_cell_size_A(); })
        .def("maybe_rebuild_verlet", &Context::maybe_rebuild_verlet,
             "Rebuild Mu Verlet CSR if skin>0 (for microbench / debug).")
        .def("invalidate_verlet_pivot_accept", &Context::invalidate_verlet_pivot_accept,
             "Mark Mu Verlet dirty (same path as pivot-accept invalidation).")
        .def("reset_neighbor_proxy_stats", &Context::reset_neighbor_proxy_stats)
        .def("print_neighbor_proxy_stats",
             [](const Context& c, const std::string& tag) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                 c.print_neighbor_proxy_stats(tag.c_str());
#pragma GCC diagnostic pop
             },
             py::arg("tag") = "neighbor-proxy",
             "Print neighbor-list proxy statistics (developer tuning only). "
             "DEPRECATED: will be removed in a future version.")
        .def("set_proxy_print_every",
             [](Context& c, int n) {
                 if (PyErr_WarnEx(PyExc_DeprecationWarning,
                     "set_proxy_print_every() is deprecated and has no effect. "
                     "Auto-print was removed from Integrator.run() to prevent log spam "
                     "under REMD. Call print_neighbor_proxy_stats() explicitly if needed.",
                     1) < 0) {
                     throw py::error_already_set();
                 }
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
                 c.set_proxy_print_every(n);
#pragma GCC diagnostic pop
             },
             py::arg("n"),
             "DEPRECATED: was used to set auto-print interval for neighbor-list stats. "
             "Now a no-op. Will be removed in a future version.")
        .def("reset_coord_sync_stats", &Context::reset_coord_sync_stats)
        .def("coord_sync_stats",
             [](const Context& c) {
                 const auto& s = c.coordSyncStats();
                 py::dict d;
                 d["num_coords_eigen_materializations"] = s.num_coords_eigen_materializations;
                 d["num_coords_eigen_writes_back"] = s.num_coords_eigen_writes_back;
                 return d;
             })
        .def("neighbor_proxy_stats",
             [](const Context& c) {
                 const auto& s = c.neighborStats();
                 const float skin = c.mu_skin();
                 const auto& cfg = c.neighborConfig();
                 const auto r = s.derive(
                     s.num_steps_executed, skin,
                     cfg.verlet_warn_min_use_rate, cfg.verlet_warn_max_rebuild_rate);
                 py::dict d;
                 d["mu_num_candidates_iterated"] = s.mu_num_candidates_iterated;
                 d["mu_num_pair_distance_checks"] = s.mu_num_pair_distance_checks;

                 d["mu_num_pairs_within_rcut"] = s.mu_num_pairs_within_rcut;
                 d["mu_num_pairs_evaluated"] = s.mu_num_pairs_evaluated();
                 d["hbond_num_candidates_iterated"] = s.hbond_num_candidates_iterated;
                 d["hbond_num_geom_checks"] = s.hbond_num_geom_checks;
                 d["neighbor_num_cell_visits"] = s.neighbor_num_cell_visits;
                 d["mu_span_slots_scanned"] = s.mu_span_slots_scanned;
                 d["mu_stencil_cells_culled"] = s.mu_stencil_cells_culled;
                 d["mu_eval_pair_calls"] = s.mu_eval_pair_calls;
                 d["elided_rigid_mm"] = s.elided_rigid_mm;
                 d["skip_rigid_mm"] = cfg.skip_rigid_mm;
                 d["mu_cell_size_scale"] = cfg.mu_cell_size_scale;
                 d["mu_cell_size_angstrom"] = cfg.mu_cell_size_angstrom;
                 d["mu_cell_size_min_angstrom"] = cfg.mu_cell_size_min_angstrom;
                 d["effective_mu_cell_size_A"] = c.effective_mu_cell_size_A();
                 d["mu_cell_size_A"] = c.neighbors().mu_cell_size_A();
                 d["neighbor_offsets_count"] =
                     c.neighbors().denseActive()
                         ? static_cast<std::uint64_t>(
                               c.neighbors().muGrid().grid().neighbor_offsets_count())
                         : s.neighbor_offsets_count;
                 d["mu_stencil_radius"] =
                     c.neighbors().denseActive()
                         ? c.neighbors().muGrid().grid().stencil_radius()
                         : 0;
                 // ADDED: live Mu grid occupancy for cell-size tuning
                 if (c.neighbors().denseActive()) {
                     const auto& g = c.neighbors().muGrid().grid();
                     int n_occ = 0, sum = 0, mx = 0;
                     const int ncells = static_cast<int>(g.num_cells());
                     for (int ci = 0; ci < ncells; ++ci) {
                         const int n = g.cell_atom_count(ci);
                         if (n <= 0) continue;
                         ++n_occ;
                         sum += n;
                         if (n > mx) mx = n;
                     }
                     d["mu_grid_nx"] = g.nx();
                     d["mu_grid_ny"] = g.ny();
                     d["mu_grid_nz"] = g.nz();
                     d["mu_grid_n_cells"] = g.num_cells();
                     d["mu_grid_n_occupied"] = n_occ;
                     d["mu_grid_avg_occupancy"] =
                         n_occ > 0 ? static_cast<double>(sum) / n_occ : 0.0;
                     d["mu_grid_max_occupancy"] = mx;
                     d["mu_grid_peak_occupancy"] = g.peak_cell_occupancy();
                     d["mu_grid_contiguous"] = g.use_contiguous();
                     d["mu_grid_cell_capacity"] = OpenCellGrid::CELL_CAPACITY;
                 } else {
                     d["mu_grid_n_cells"] = 0;
                     d["mu_grid_n_occupied"] = 0;
                     d["mu_grid_avg_occupancy"] = 0.0;
                     d["mu_grid_max_occupancy"] = 0;
                     d["mu_grid_peak_occupancy"] = 0;
                     d["mu_grid_contiguous"] = false;
                     d["mu_grid_cell_capacity"] = OpenCellGrid::CELL_CAPACITY;
                 }
                 d["num_verlet_used"] = s.num_verlet_used();
                 d["num_verlet_fallback_cell"] = s.num_verlet_fallback_cell();
                 d["num_verlet_rebuilds"] = s.num_verlet_rebuilds;
                 d["num_verlet_partial_rebuilds"] = s.num_verlet_partial_rebuilds;
                 d["num_verlet_partial_affected_sum"] = s.num_verlet_partial_affected_sum;
                 d["num_verlet_invalidate_pivot_accept"] = s.num_verlet_invalidate_pivot_accept;
                 d["num_pivot_accepts"] = s.num_pivot_accepts;
                 d["num_pivot_accepts_keep_verlet_valid"] = s.num_pivot_accepts_keep_verlet_valid;
                 d["num_pivot_accepts_dirty_verlet"] = s.num_pivot_accepts_dirty_verlet;
                 d["mu_verlet_enabled"] = cfg.mu_verlet_enabled;
                 d["invalidate_verlet_on_pivot_accept"] = cfg.invalidate_verlet_on_pivot_accept;
                 d["num_verlet_rebuild_due_to_pivot_accept"] = s.num_verlet_rebuild_due_to_pivot_accept;
                 d["num_verlet_rebuild_due_to_disp_acc_exceeded"] =
                     s.num_verlet_rebuild_due_to_disp_acc_exceeded;
                 d["num_verlet_rebuild_due_to_dirty_flag"] = s.num_verlet_rebuild_due_to_dirty_flag;
                 d["num_verlet_rebuild_due_to_autoexpand_accept"] =
                     s.num_verlet_rebuild_due_to_autoexpand_accept;
                 d["verlet_edges_total"] = s.verlet_edges_total;
                 d["verlet_avg_degree"] = s.verlet_avg_degree;
                 d["num_verlet_neigh_reallocs"] = s.num_verlet_neigh_reallocs;
                 d["num_verlet_offsets_reallocs"] = s.num_verlet_offsets_reallocs;
                 d["num_aabb_rebuild_accept"] = s.num_aabb_rebuild_accept;
                 d["num_trial_fallback"] = s.num_trial_fallback;
                 d["num_reject_hard_disp"] = s.num_reject_hard_disp;
                 d["num_reject_out_of_box"] = s.num_reject_out_of_box;
                 d["num_steps_executed"] = s.num_steps_executed;
                 d["total_steps"] = r.total_steps;
                 d["mu_skin"] = r.mu_skin;
                 d["verlet_use_rate"] = r.verlet_use_rate;
                 d["rebuild_rate_per_step"] = r.rebuild_rate_per_step;
                 d["avg_mu_pair_checks_per_step"] = r.avg_mu_pair_checks_per_step;
                 d["avg_mu_pairs_within_rcut_per_step"] = r.avg_mu_pairs_within_rcut_per_step;
                 d["avg_mu_pairs_per_step"] = r.avg_mu_pairs_per_step;
                 d["avg_mu_candidates_per_step"] = r.avg_mu_candidates_per_step;
                 d["avg_cell_visits_per_step"] = r.avg_cell_visits_per_step;
                 d["avg_hbond_geom_checks_per_step"] = r.avg_hbond_geom_checks_per_step;
                 d["low_use_rate"] = r.low_use_rate;
                 d["high_rebuild_rate"] = r.high_rebuild_rate;
                 py::dict derived;
                 derived["total_steps"] = r.total_steps;
                 derived["mu_skin"] = r.mu_skin;
                 derived["verlet_use_rate"] = r.verlet_use_rate;
                 derived["rebuild_rate_per_step"] = r.rebuild_rate_per_step;
                 derived["avg_mu_pair_checks_per_step"] = r.avg_mu_pair_checks_per_step;
                 derived["avg_mu_pairs_within_rcut_per_step"] = r.avg_mu_pairs_within_rcut_per_step;
                 derived["avg_mu_pairs_per_step"] = r.avg_mu_pairs_per_step;
                 derived["avg_mu_candidates_per_step"] = r.avg_mu_candidates_per_step;
                 derived["avg_cell_visits_per_step"] = r.avg_cell_visits_per_step;
                 derived["avg_hbond_geom_checks_per_step"] = r.avg_hbond_geom_checks_per_step;
                 derived["low_use_rate"] = r.low_use_rate;
                 derived["high_rebuild_rate"] = r.high_rebuild_rate;
                 d["derived"] = derived;
                 return d;

             });


    py::class_<mcpu::MCIntegrator>(m, "Integrator",
        "Metropolis Monte Carlo move engine: backbone pivot, continuous\n"
        "sidechain, rotamer-library and kinematic-closure loop moves.\n\n"
        "temperature is a DIMENSIONLESS reduced parameter, roughly 0.3\n"
        "(cold, folded) to 0.6 (hot, unfolded) -- it is not Kelvin. The\n"
        "constructor default of 300.0 is about 500x the useful hot end, so\n"
        "always pass one explicitly.\n\n"
        "Call set_seed(): the trajectory is fully determined by it.")
        .def(py::init<float, float, float>(), py::arg("temperature") = 300.0f,
             py::arg("step_size_rad") = 0.1f,
             // Negative = "same as step_size_rad", so omitting it reproduces the
             // pre-knob single-amplitude behavior exactly.
             py::arg("sidechain_step_size_rad") = -1.0f)
        .def("set_sidechain_step_size_rad", &mcpu::MCIntegrator::set_sidechain_step_size_rad,
             py::arg("sigma_rad"),
             "Continuous-sidechain chi amplitude in radians; negative restores "
             "'same as backbone'. Legacy has two independent amplitudes "
             "(MC_STEP_SIZE 2 deg backbone, SIDECHAIN_NOISE 10 deg chi), so a "
             "like-for-like comparison needs this set separately. Ignored by the "
             "rotamer_library sidechain mode, which takes per-chi widths from the "
             "library rows.")
        .def("sidechain_step_size_rad", &mcpu::MCIntegrator::sidechain_step_size_rad)
        .def("backbone_step_size_rad", &mcpu::MCIntegrator::backbone_step_size_rad)
        .def("run", &mcpu::MCIntegrator::run, py::arg("context"), py::arg("num_steps"), py::arg("step_offset") = 0)
        .def("set_seed", &mcpu::MCIntegrator::set_seed, py::arg("seed"))
        .def("get_rng_state", &mcpu::MCIntegrator::get_rng_state,
             "Serialize mt19937 RNG state for checkpoint/resume.")
        .def("set_rng_state", &mcpu::MCIntegrator::set_rng_state, py::arg("state"),
             "Restore mt19937 RNG state previously returned by get_rng_state.")
        .def("set_sidechain_move_mode", &mcpu::MCIntegrator::set_sidechain_move_mode,
             py::arg("mode"),
             "Selects the Sidechain-slot proposal algorithm: 'continuous' "
             "(default) or 'rotamer_library'.")
        .def("sidechain_move_mode", &mcpu::MCIntegrator::sidechain_move_mode)
        .def("set_pivot_rama_probability", &mcpu::MCIntegrator::set_pivot_rama_probability,
             py::arg("p"),
             "Fraction of Pivot-slot attempts using the knowledge-based "
             "(phi,psi) rama-mixture proposal instead of the continuous "
             "single-dihedral pivot. Default 0.05; p=0.0 recovers exact "
             "legacy behavior (including RNG-draw count).")
        .def("pivot_rama_probability", &mcpu::MCIntegrator::pivot_rama_probability)
        .def("set_move_weights", &mcpu::MCIntegrator::set_move_weights,
             py::arg("pivot"), py::arg("kic"), py::arg("sidechain"),
             "Relative probabilities of the Pivot / KIC / Sidechain move slots, "
             "normalized internally. Default (0.25, 0.25, 0.50), which is the "
             "mix this integrator has always used; passing it explicitly is "
             "byte-identical to not calling this at all. Exactly one RNG draw "
             "is consumed per step whatever the weights are.\n\n"
             "Pass sidechain=0.0 for a force field whose residues have no chi "
             "angles (a backbone-only one): otherwise every sidechain proposal "
             "is a silent no-op and that share of the run is discarded. run() "
             "raises on that combination rather than letting it happen.")
        .def("move_weights",
             [](const mcpu::MCIntegrator& self) {
                 const std::array<float, 3> w = self.move_weights();
                 return py::make_tuple(w[0], w[1], w[2]);
             },
             "(pivot, kic, sidechain) slot probabilities, normalized to sum 1.")
        .def("set_pivot_rama_schedule", &mcpu::MCIntegrator::set_pivot_rama_schedule,
             py::arg("t_low"), py::arg("t_high"), py::arg("p_min"), py::arg("p_max"),
             "Sets pivot_rama_probability() from a piecewise-linear schedule "
             "evaluated once against this Integrator's own (fixed) "
             "temperature, in pyMCPU's reduced-temperature units.")
        .def("set_use_pooled_proposal", &mcpu::MCIntegrator::set_use_pooled_proposal,
             py::arg("on"),
             "If True (default when compiled with MCPU_USE_POOLED_PROPOSAL=1): "
             "DynamicOnly pooled proposal + sparse patch reset. "
             "If False: emulate vanilla Full State copy + per-step patch alloc + reject restore.")
        .def("use_pooled_proposal", &mcpu::MCIntegrator::use_pooled_proposal)
        .def_property(
            "use_sparse_proposal",
            &mcpu::MCIntegrator::use_sparse_proposal,
            &mcpu::MCIntegrator::set_use_sparse_proposal,
            "If True (default): skip per-step copy_dynamic_from; O(n_moved) restore on reject.")
        .def("set_use_sparse_proposal", &mcpu::MCIntegrator::set_use_sparse_proposal,
             py::arg("on"))
        .def("last_move_kind", &mcpu::MCIntegrator::last_move_kind,
             "ADDED: last proposed move kind string (Pivot/KIC/Sidechain/Other).")
        .def("last_move_is_rigid", &mcpu::MCIntegrator::last_move_is_rigid,
             "ADDED: whether the last proposal was a rigid body move.")
        .def("last_moved_indices", &mcpu::MCIntegrator::last_moved_indices,
             py::return_value_policy::copy,
             "ADDED: atom indices moved in the last proposal.")
        .def("last_delta_energy", &mcpu::MCIntegrator::last_delta_energy,
             "ADDED: ΔE of last accepted move (0 if rejected/invalid).")
        .def("last_log_jacobian_weight", &mcpu::MCIntegrator::last_log_jacobian_weight,
             "MH correction term from the most recent debug_force_rama_pivot_to call.")
        .def("reject_restore_enabled", &mcpu::MCIntegrator::reject_restore_enabled)
        .def("proposal_is_dynamic_only", &mcpu::MCIntegrator::proposal_is_dynamic_only)
        .def("proposal_lifecycle_info",
             [](const mcpu::MCIntegrator& integ) {
                 const auto info = integ.proposal_lifecycle_info();
                 py::dict d;
                 d["pooled_proposal_compiled_in"] = info.pooled_proposal_compiled_in;
                 d["use_pooled_proposal"] = info.use_pooled_proposal;
                 d["proposal_dynamic_only"] = info.proposal_dynamic_only;
                 d["reject_restore_enabled"] = info.reject_restore_enabled;
                 return d;
             })
        .def("set_step_stats_verbose", &mcpu::MCIntegrator::set_step_stats_verbose,
             py::arg("on"))
        .def("step_stats_verbose", &mcpu::MCIntegrator::step_stats_verbose)
        .def("reset_step_stats", &mcpu::MCIntegrator::reset_step_stats)
        .def("step_stats",
             [](const mcpu::MCIntegrator& integ) {
                 const auto& s = integ.step_stats();
                 py::dict d;
                 d["copy_dynamic_ns"] = s.copy_dynamic_ns;
                 d["commit_ns"] = s.commit_ns;
                 d["delta_energy_ns"] = s.delta_energy_ns;
                 d["step_total_ns"] = s.step_total_ns;
                 d["gen_pivot_ns"] = s.gen_pivot_ns;
                 d["gen_kic_ns"] = s.gen_kic_ns;
                 d["gen_sc_ns"] = s.gen_sc_ns;
                 {
                     py::list ba, bc, bn;
                     for (int i = 0; i < StepStats::kNBins; ++i) {
                         ba.append(s.bin_attempts[i]);
                         bc.append(s.bin_accepts[i]);
                         bn.append(s.bin_ns[i]);
                     }
                     d["bin_attempts"] = ba;
                     d["bin_accepts"] = bc;
                     d["bin_ns"] = bn;
                 }
                 d["moved_atoms_sum"] = s.moved_atoms_sum;
                 d["n_steps"] = s.n_steps;
                 d["n_valid_moves"] = s.n_valid_moves;
                 d["n_accepts"] = s.n_accepts;
                 d["mu_eval_pair_calls"] = s.mu_eval_pair_calls;
                 py::list by_kind;
                 static const char* kKindNames[] = {"pivot", "kic", "sidechain"};
                 for (int k = 0; k < 3; ++k) {
                     const auto& mk = s.mu_by_kind[k];
                     py::dict dmk;
                     dmk["kind"] = kKindNames[k];
                     dmk["ns"] = mk.ns;
                     dmk["eval_pair_calls"] = mk.eval_pair_calls;
                     dmk["pair_distance_checks"] = mk.pair_distance_checks;
                     dmk["pairs_within_rcut"] = mk.pairs_within_rcut;
                     dmk["eval_pair_nonzero"] = mk.eval_pair_nonzero;
                     dmk["n_steps"] = mk.n_steps;
                     dmk["moved_atoms_sum"] = mk.moved_atoms_sum;
                     if (mk.n_steps > 0) {
                         const double inv = 1.0 / static_cast<double>(mk.n_steps);
                         dmk["avg_moved_atoms"] =
                             static_cast<double>(mk.moved_atoms_sum) * inv;
                         dmk["avg_eval_pair_calls"] =
                             static_cast<double>(mk.eval_pair_calls) * inv;
                         dmk["avg_ns"] = static_cast<double>(mk.ns) * inv;
                         dmk["avg_pair_distance_checks"] =
                             static_cast<double>(mk.pair_distance_checks) * inv;
                         dmk["avg_pairs_within_rcut"] =
                             static_cast<double>(mk.pairs_within_rcut) * inv;
                         if (mk.eval_pair_calls > 0) {
                             dmk["eval_pair_nonzero_frac"] =
                                 static_cast<double>(mk.eval_pair_nonzero) /
                                 static_cast<double>(mk.eval_pair_calls);
                         }
                         if (mk.pair_distance_checks > 0) {
                             dmk["within_rcut_frac"] =
                                 static_cast<double>(mk.pairs_within_rcut) /
                                 static_cast<double>(mk.pair_distance_checks);
                         }
                     }
                     by_kind.append(dmk);
                 }
                 d["mu_by_kind"] = by_kind;
                 d["verlet_used"] = s.verlet_used;
                 d["verlet_fallback_cell"] = s.verlet_fallback_cell;
                 d["verlet_rebuilds"] = s.verlet_rebuilds;
                 d["verlet_partial_rebuilds"] = s.verlet_partial_rebuilds;
                 d["verlet_partial_affected_sum"] = s.verlet_partial_affected_sum;
                 {
                     const auto trials = s.verlet_used + s.verlet_fallback_cell;
                     py::dict vs;
                     vs["n_verlet_queries"] = s.verlet_used;
                     vs["n_cellonly_queries"] = s.verlet_fallback_cell;
                     vs["n_full_rebuilds"] = s.verlet_rebuilds;
                     vs["n_partial_rebuilds"] = s.verlet_partial_rebuilds;
                     vs["n_affected_atoms_sum"] = s.verlet_partial_affected_sum;
                     if (trials > 0) {
                         vs["reuse_rate"] =
                             static_cast<double>(s.verlet_used) /
                             static_cast<double>(trials);
                     }
                     if (s.verlet_partial_rebuilds > 0) {
                         vs["avg_affected_atoms"] =
                             static_cast<double>(s.verlet_partial_affected_sum) /
                             static_cast<double>(s.verlet_partial_rebuilds);
                     }
                     d["verlet_stats"] = vs;
                 }
                 {
                     const auto& b = s.pivot_mu_breakdown;
                     py::dict pb;
                     pb["cell_walk_ns"] = b.cell_walk_ns;
                     pb["r2_filter_ns"] = b.r2_filter_ns;
                     pb["eval_pair_ns"] = b.eval_pair_ns;
                     pb["overhead_ns"] = b.overhead_ns;
                     pb["candidates"] = b.candidates;
                     pb["in_cutoff"] = b.in_cutoff;
                     pb["n_pivot_steps"] = b.n_pivot_steps;
                     if (b.n_pivot_steps > 0) {
                         const double inv =
                             1.0 / static_cast<double>(b.n_pivot_steps);
                         pb["avg_cell_walk_ns"] =
                             static_cast<double>(b.cell_walk_ns) * inv;
                         pb["avg_r2_filter_ns"] =
                             static_cast<double>(b.r2_filter_ns) * inv;
                         pb["avg_eval_pair_ns"] =
                             static_cast<double>(b.eval_pair_ns) * inv;
                         pb["avg_overhead_ns"] =
                             static_cast<double>(b.overhead_ns) * inv;
                         pb["avg_candidates"] =
                             static_cast<double>(b.candidates) * inv;
                         pb["avg_in_cutoff"] =
                             static_cast<double>(b.in_cutoff) * inv;
                         if (b.candidates > 0) {
                             pb["in_cutoff_frac"] =
                                 static_cast<double>(b.in_cutoff) /
                                 static_cast<double>(b.candidates);
                         }
                         pb["avg_walk_empty_cells"] =
                             static_cast<double>(b.walk_empty_cells) * inv;
                         pb["avg_walk_nonempty_cells"] =
                             static_cast<double>(b.walk_nonempty_cells) * inv;
                         pb["avg_walk_atom_visits"] =
                             static_cast<double>(b.walk_atom_visits) * inv;
                         pb["avg_walk_probe_ns"] =
                             static_cast<double>(b.walk_probe_ns) * inv;
                         pb["avg_walk_oob_cells"] =
                             static_cast<double>(b.walk_oob_cells) * inv;
                         const auto cell_vis =
                             b.walk_empty_cells + b.walk_nonempty_cells;
                         if (cell_vis > 0) {
                             pb["walk_empty_frac"] =
                                 static_cast<double>(b.walk_empty_cells) /
                                 static_cast<double>(cell_vis);
                         }
                         if (b.walk_nonempty_cells > 0) {
                             pb["avg_atoms_per_nonempty_cell"] =
                                 static_cast<double>(b.walk_atom_visits) /
                                 static_cast<double>(b.walk_nonempty_cells);
                         }
                     }
                     pb["cell_pairs"] = b.cell_pairs;
                     pb["cell_pairs_empty"] = b.cell_pairs_empty;
                     pb["n_groups"] = b.n_groups;
                     pb["group_atoms"] = b.group_atoms;
                     pb["cell_pair_evals"] = b.cell_pair_evals;
                     if (b.cell_pair_evals > 0) {
                         const double inv_cp =
                             1.0 / static_cast<double>(b.cell_pair_evals);
                         pb["avg_cell_pairs"] =
                             static_cast<double>(b.cell_pairs) * inv_cp;
                         pb["avg_cell_pairs_empty"] =
                             static_cast<double>(b.cell_pairs_empty) * inv_cp;
                         pb["avg_n_groups"] =
                             static_cast<double>(b.n_groups) * inv_cp;
                     }
                     if (b.n_groups > 0) {
                         pb["avg_atoms_per_group"] =
                             static_cast<double>(b.group_atoms) /
                             static_cast<double>(b.n_groups);
                     }
                     const auto cp_tot = b.cell_pairs + b.cell_pairs_empty;
                     if (cp_tot > 0) {
                         pb["cell_pair_empty_frac"] =
                             static_cast<double>(b.cell_pairs_empty) /
                             static_cast<double>(cp_tot);
                     }
                     d["pivot_mu_breakdown"] = pb;
                 }
                 {
                     const auto& b = s.cell_pair_breakdown;
                     py::dict cpb;
                     cpb["group_build_ns"] = b.group_build_ns;
                     cpb["new_walk_ns"] = b.new_walk_ns;
                     cpb["new_r2_ns"] = b.new_r2_ns;
                     cpb["new_eval_ns"] = b.new_eval_ns;
                     cpb["old_walk_ns"] = b.old_walk_ns;
                     cpb["old_r2_ns"] = b.old_r2_ns;
                     cpb["old_eval_ns"] = b.old_eval_ns;
                     cpb["mmguard_ns"] = b.mmguard_ns;
                    cpb["movedbits_ns"] = b.movedbits_ns;
                    cpb["skipmask_ns"] = b.skipmask_ns;
                    cpb["clash_aborts"] = b.clash_aborts;
                     cpb["full_evals"] = b.full_evals;
                     cpb["new_r2_checks"] = b.new_r2_checks;
                     cpb["old_r2_checks"] = b.old_r2_checks;
                     cpb["new_eval_calls"] = b.new_eval_calls;
                     cpb["old_eval_calls"] = b.old_eval_calls;
                     cpb["n_steps"] = b.n_steps;
                     cpb["new_n_groups"] = b.new_n_groups;
                     cpb["new_group_atoms"] = b.new_group_atoms;
                     cpb["cell_pairs"] = b.cell_pairs;
                     cpb["cell_pairs_empty"] = b.cell_pairs_empty;
                     if (b.n_steps > 0) {
                         const double inv =
                             1.0 / static_cast<double>(b.n_steps);
                         cpb["avg_group_build_ns"] =
                             static_cast<double>(b.group_build_ns) * inv;
                         cpb["avg_new_walk_ns"] =
                             static_cast<double>(b.new_walk_ns) * inv;
                         cpb["avg_new_r2_ns"] =
                             static_cast<double>(b.new_r2_ns) * inv;
                         cpb["avg_new_eval_ns"] =
                             static_cast<double>(b.new_eval_ns) * inv;
                         cpb["avg_old_walk_ns"] =
                             static_cast<double>(b.old_walk_ns) * inv;
                         cpb["avg_old_r2_ns"] =
                             static_cast<double>(b.old_r2_ns) * inv;
                         cpb["avg_old_eval_ns"] =
                             static_cast<double>(b.old_eval_ns) * inv;
                         cpb["avg_new_r2_checks"] =
                             static_cast<double>(b.new_r2_checks) * inv;
                         cpb["avg_old_r2_checks"] =
                             static_cast<double>(b.old_r2_checks) * inv;
                         cpb["avg_new_eval_calls"] =
                             static_cast<double>(b.new_eval_calls) * inv;
                         cpb["avg_old_eval_calls"] =
                             static_cast<double>(b.old_eval_calls) * inv;
                         cpb["avg_new_n_groups"] =
                             static_cast<double>(b.new_n_groups) * inv;
                         cpb["clash_abort_rate"] =
                             static_cast<double>(b.clash_aborts) * inv;
                     }
                     if (b.new_n_groups > 0) {
                         cpb["avg_atoms_per_new_group"] =
                             static_cast<double>(b.new_group_atoms) /
                             static_cast<double>(b.new_n_groups);
                     }
                     const auto cp_tot = b.cell_pairs + b.cell_pairs_empty;
                     if (cp_tot > 0) {
                         cpb["cell_pair_empty_frac"] =
                             static_cast<double>(b.cell_pairs_empty) /
                             static_cast<double>(cp_tot);
                     }
                     d["cell_pair_breakdown"] = cpb;
                 }
                 const auto verlet_trials = s.verlet_used + s.verlet_fallback_cell;
                 if (verlet_trials > 0) {
                     d["verlet_use_rate"] =
                         static_cast<double>(s.verlet_used) /
                         static_cast<double>(verlet_trials);
                 }
                 if (s.n_steps > 0) {
                     d["verlet_rebuild_rate_per_step"] =
                         static_cast<double>(s.verlet_rebuilds) /
                         static_cast<double>(s.n_steps);
                 }
                 py::dict by_group;
                 static const char* kNames[] = {
                     "0", "mu", "backbone_torsion", "sidechain_torsion",
                     "hydrogen_bond", "aromatic", "native_contacts_bias", "7"
                 };
                 for (int g = 1; g <= 6; ++g) {
                     by_group[kNames[g]] = s.energy_delta_ns[g];
                 }
                 d["energy_delta_ns"] = by_group;
                 if (s.n_steps > 0) {
                     const double inv = 1.0 / static_cast<double>(s.n_steps);
                     const double total = static_cast<double>(s.step_total_ns);
                     d["avg_moved_atoms"] = static_cast<double>(s.moved_atoms_sum) * inv;
                     if (total > 0.0) {
                         d["copy_dynamic_pct"] =
                             100.0 * static_cast<double>(s.copy_dynamic_ns) / total;
                         d["delta_energy_pct"] =
                             100.0 * static_cast<double>(s.delta_energy_ns) / total;
                         d["commit_pct"] =
                             100.0 * static_cast<double>(s.commit_ns) / total;
                     }
                 }
                 return d;
             })
        .def("last_accept_bits",
             [](const mcpu::MCIntegrator& integ) {
                 return integ.last_accept_bits();
             })
        .def("num_pivot_resample_pro_phi", &mcpu::MCIntegrator::num_pivot_resample_pro_phi)
        .def("num_sc_resample_pro", &mcpu::MCIntegrator::num_sc_resample_pro)
        .def("get_bb_attempted", &mcpu::MCIntegrator::get_bb_attempted)
        .def("get_bb_accepted", &mcpu::MCIntegrator::get_bb_accepted)
        .def("get_sc_attempted", &mcpu::MCIntegrator::get_sc_attempted)
        .def("get_sc_accepted", &mcpu::MCIntegrator::get_sc_accepted)
        .def("get_rotamer_attempted", &mcpu::MCIntegrator::get_rotamer_attempted)
        .def("get_rotamer_accepted", &mcpu::MCIntegrator::get_rotamer_accepted)
        .def("get_kic_attempted", &mcpu::MCIntegrator::get_kic_attempted)
        .def("get_kic_accepted", &mcpu::MCIntegrator::get_kic_accepted)
        .def("get_rama_pivot_attempted", &mcpu::MCIntegrator::get_rama_pivot_attempted)
        .def("get_rama_pivot_accepted", &mcpu::MCIntegrator::get_rama_pivot_accepted)
        .def("get_kic_presolve_zero", &mcpu::MCIntegrator::get_kic_presolve_zero)
        .def("get_kic_jacobian_invalid", &mcpu::MCIntegrator::get_kic_jacobian_invalid)
        .def("get_kic_geometry_invalid", &mcpu::MCIntegrator::get_kic_geometry_invalid)
        .def("get_steric_rejected", &mcpu::MCIntegrator::get_steric_rejected)
        .def("move_stats",
             [](const mcpu::MCIntegrator& integ) {
                 py::dict d;
                 d["num_propose_pivot"] = integ.get_bb_attempted();
                 d["num_accept_pivot"] = integ.get_bb_accepted();
                 d["num_propose_kic"] = integ.get_kic_attempted();
                 d["num_accept_kic"] = integ.get_kic_accepted();
                 d["num_propose_sc"] = integ.get_sc_attempted();
                 d["num_accept_sc"] = integ.get_sc_accepted();
                 d["num_propose_rotamer"] = integ.get_rotamer_attempted();
                 d["num_accept_rotamer"] = integ.get_rotamer_accepted();
                 d["num_propose_rama_pivot"] = integ.get_rama_pivot_attempted();
                 d["num_accept_rama_pivot"] = integ.get_rama_pivot_accepted();
                 d["kic_presolve_zero"] = integ.get_kic_presolve_zero();
                 d["kic_jacobian_invalid"] = integ.get_kic_jacobian_invalid();
                 d["kic_geometry_invalid"] = integ.get_kic_geometry_invalid();
                 d["steric_rejected"] = integ.get_steric_rejected();
                 return d;
             })
        .def("debug_force_pivot", &mcpu::MCIntegrator::debug_force_pivot,
             py::arg("context"), py::arg("residue"), py::arg("is_phi"))
        .def("debug_force_sc", &mcpu::MCIntegrator::debug_force_sc,
             py::arg("context"), py::arg("residue"))
        .def("debug_force_rotamer", &mcpu::MCIntegrator::debug_force_rotamer,
             py::arg("context"), py::arg("residue"))
        .def("debug_force_rama_pivot", &mcpu::MCIntegrator::debug_force_rama_pivot,
             py::arg("context"), py::arg("residue"))
        .def("debug_force_rama_pivot_to", &mcpu::MCIntegrator::debug_force_rama_pivot_to,
             py::arg("context"), py::arg("residue"), py::arg("phi"), py::arg("psi"),
             "Test-only: forces the rama-mixture pivot move to an explicit "
             "(phi, psi) target instead of drawing one from the mixture.")
        .def("verify_physics_consistency", &mcpu::MCIntegrator::verify_physics_consistency,
             py::arg("context"), py::arg("num_steps"), py::arg("atol") = 1e-3f)
        .def("add_reporter", &mcpu::MCIntegrator::add_reporter)
        .def("clear_reporters", &mcpu::MCIntegrator::clear_reporters)
        .def("num_reporters", &mcpu::MCIntegrator::num_reporters)
        .def("set_fixed_residues", &mcpu::MCIntegrator::setFixedResidues,
             py::arg("residue_indices"), py::arg("n_residues"),
             "Mark residues as fixed (0-based engine indices). "
             "Fixed residues will not be moved by any MC proposal.")
        .def("clear_fixed_residues", &mcpu::MCIntegrator::clearFixedResidues)
        .def("get_fixed_residues", &mcpu::MCIntegrator::getFixedResidues)
        .def("get_fixed_residue_mask", &mcpu::MCIntegrator::fixedResidueMask)
        .def("has_fixed_residues", &mcpu::MCIntegrator::hasFixedResidues)
        .def("get_fixed_rejected", &mcpu::MCIntegrator::get_fixed_rejected);

    // ----------------------------------------------------------------------
    // build_info(): what this binary was actually compiled with
    // ----------------------------------------------------------------------
    // Every value below comes from either a compiler-predefined macro or the
    // generated BuildConfig.h, which CMake fills from the same variables that
    // produced the flags. There are deliberately NO fallback literals: a
    // missing define is a #error, not a plausible-looking default.
    //
    // This replaces build_flags(), which returned six of its eight keys as
    // hardcoded C++ literals -- `d["MCPU_UNSAFE_MATH"] = false;` ignored the
    // actual define, so it would have reported "safe" even after someone
    // enabled fast math. It also exposed no -march, compiler or LTO state,
    // which is exactly what a cross-build comparison needs.
#if !defined(MCPU_BUILD_ARCH_TIER) || !defined(MCPU_BUILD_LTO)
#error "BuildConfig.h was not generated; configure through CMake."
#endif
#if !defined(MCPU_FAST_MU_DELTA) || !defined(MCPU_USE_POOLED_PROPOSAL)
#error "Feature-flag defines missing; configure through CMake."
#endif
    m.def(
        "build_info",
        []() {
            py::dict arch;
            arch["tier"] = MCPU_BUILD_ARCH_TIER;
            arch["requested"] = MCPU_BUILD_ARCH_REQUESTED;
            arch["march"] = MCPU_BUILD_ARCH_MARCH;
            arch["flags"] = MCPU_BUILD_ARCH_FLAGS;

            // Read from the compiler's own macros, so this is an INDEPENDENT
            // witness of what -march did. If `arch.tier` says v4 while
            // isa.avx512f is false, the flag did not take effect and the two
            // halves of this dict disagree -- which a test can assert on.
            py::dict isa;
#if defined(__AVX__)
            isa["avx"] = true;
#else
            isa["avx"] = false;
#endif
#if defined(__AVX2__)
            isa["avx2"] = true;
#else
            isa["avx2"] = false;
#endif
#if defined(__FMA__)
            isa["fma"] = true;
#else
            isa["fma"] = false;
#endif
#if defined(__BMI2__)
            isa["bmi2"] = true;
#else
            isa["bmi2"] = false;
#endif
#if defined(__F16C__)
            isa["f16c"] = true;
#else
            isa["f16c"] = false;
#endif
#if defined(__AVX512F__)
            isa["avx512f"] = true;
#else
            isa["avx512f"] = false;
#endif
#if defined(__AVX512VL__)
            isa["avx512vl"] = true;
#else
            isa["avx512vl"] = false;
#endif

            py::dict compiler;
#if defined(__clang__)
            compiler["id"] = "Clang";
            compiler["version"] = std::to_string(__clang_major__) + "." +
                                  std::to_string(__clang_minor__) + "." +
                                  std::to_string(__clang_patchlevel__);
#elif defined(__GNUC__)
            compiler["id"] = "GNU";
            compiler["version"] = std::to_string(__GNUC__) + "." +
                                  std::to_string(__GNUC_MINOR__) + "." +
                                  std::to_string(__GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
            compiler["id"] = "MSVC";
            compiler["version"] = std::to_string(_MSC_VER);
#else
            compiler["id"] = "unknown";
            compiler["version"] = "";
#endif
            compiler["cxx_standard"] = static_cast<long>(__cplusplus);
            // The libstdc++ date stamp. This project spends a lot of README on
            // the `CXXABI_x.y.z not found` failure mode; having it here makes
            // that diagnosable in one call instead of with ldd and strings.
#if defined(__GLIBCXX__)
            compiler["glibcxx"] = static_cast<long>(__GLIBCXX__);
#else
            compiler["glibcxx"] = py::none();
#endif

            py::dict build;
            build["type"] = MCPU_BUILD_TYPE;
            build["lto"] = (MCPU_BUILD_LTO != 0);
            build["cmake_version"] = MCPU_BUILD_CMAKE_VERSION;
            // Two independent facts a single "Release" string cannot express:
            // RelWithDebInfo sets both, and a CXXFLAGS=-O0 override sets only
            // the second.
#if defined(__OPTIMIZE__)
            build["optimized"] = true;
#else
            build["optimized"] = false;
#endif
#if defined(NDEBUG)
            build["assertions"] = false;
#else
            build["assertions"] = true;
#endif

            // Makes the configure log's "SAFE math only" claim verifiable
            // rather than merely asserted.
            py::dict fp;
            fp["fp_contract"] = MCPU_BUILD_FP_CONTRACT;
#if defined(__FAST_MATH__)
            fp["fast_math"] = true;
#else
            fp["fast_math"] = false;
#endif
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
            fp["finite_math_only"] = true;
#else
            fp["finite_math_only"] = false;
#endif
#if defined(__ASSOCIATIVE_MATH__)
            fp["associative_math"] = true;
#else
            fp["associative_math"] = false;
#endif
#if defined(__RECIPROCAL_MATH__)
            fp["reciprocal_math"] = true;
#else
            fp["reciprocal_math"] = false;
#endif

            py::dict features;
            features["MCPU_FAST_MU_DELTA"] = (MCPU_FAST_MU_DELTA != 0);
            features["MCPU_USE_POOLED_PROPOSAL"] = (MCPU_USE_POOLED_PROPOSAL != 0);
#if defined(EIGEN_NO_DEBUG)
            features["EIGEN_NO_DEBUG"] = true;
#else
            features["EIGEN_NO_DEBUG"] = false;
#endif

            py::dict deps;
            deps["eigen"] = std::to_string(EIGEN_WORLD_VERSION) + "." +
                            std::to_string(EIGEN_MAJOR_VERSION) + "." +
                            std::to_string(EIGEN_MINOR_VERSION);
            deps["pybind11"] = std::to_string(PYBIND11_VERSION_MAJOR) + "." +
                               std::to_string(PYBIND11_VERSION_MINOR);

            py::dict d;
            d["arch"] = arch;
            d["isa"] = isa;
            d["compiler"] = compiler;
            d["build"] = build;
            d["fp"] = fp;
            d["features"] = features;
            d["deps"] = deps;
            return d;
        },
        "What this binary was compiled with: ISA baseline, compiler, build "
        "type, LTO, FP policy and feature flags. Every value derives from a "
        "real macro -- see build_info() in src/bindings/bindings.cpp.");

    m.def(
        "build_flags",
        []() {
            // Deprecated alias, kept for one release because
            // scripts/install_check.py calls it. Crucially it now reports the
            // REAL values: a deprecated function that lies is worse than a
            // removed one.
            py::module_::import("warnings").attr("warn")(
                "mcpu_core.build_flags() is deprecated; use build_info(). The "
                "unsafe_math_* keys were previously hardcoded literals.",
                py::module_::import("builtins").attr("DeprecationWarning"), 2);
            py::dict d;
            d["MCPU_USE_POOLED_PROPOSAL"] = (MCPU_USE_POOLED_PROPOSAL != 0);
            d["MCPU_FAST_MU_DELTA"] = (MCPU_FAST_MU_DELTA != 0);
#if defined(__FAST_MATH__)
            d["unsafe_math_enabled"] = true;
#else
            d["unsafe_math_enabled"] = false;
#endif
            d["MCPU_UNSAFE_MATH"] = d["unsafe_math_enabled"];
            d["fp_contract"] = MCPU_BUILD_FP_CONTRACT;
            d["arch_march"] = MCPU_BUILD_ARCH_MARCH;
            return d;
        },
        "DEPRECATED: use build_info().");
    m.def("reset_coord_sync_stats", []() { mcpu::coord_sync_stats().reset(); });
    m.def("coord_sync_stats", []() {
        const auto& s = mcpu::coord_sync_stats();
        py::dict d;
        d["num_coords_eigen_materializations"] = s.num_coords_eigen_materializations;
        d["num_coords_eigen_writes_back"] = s.num_coords_eigen_writes_back;
        return d;
    });


    py::class_<PotentialDeltaCheck>(m, "PotentialDeltaCheck")
        .def_readonly("energy_group", &PotentialDeltaCheck::energy_group)
        .def_readonly("delta_incremental", &PotentialDeltaCheck::delta_incremental)
        .def_readonly("delta_direct", &PotentialDeltaCheck::delta_direct)
        .def_readonly("passed", &PotentialDeltaCheck::passed)
        .def_readonly("message", &PotentialDeltaCheck::message);

    py::class_<PhysicsVerifier>(m, "PhysicsVerifier")
        .def_static("verify_potential_delta", &PhysicsVerifier::verify_potential_delta,
                    py::arg("context"), py::arg("old_state"), py::arg("proposed_state"),
                    py::arg("patch"), py::arg("energy_group"), py::arg("atol") = 1e-3f)
        .def_static("verify_all_potential_deltas", &PhysicsVerifier::verify_all_potential_deltas,
                    py::arg("context"), py::arg("old_state"), py::arg("proposed_state"),
                    py::arg("patch"), py::arg("atol") = 1e-3f)
        .def_static("verify_mc_energy_consistency", &PhysicsVerifier::verify_mc_energy_consistency,
                    py::arg("integrator"), py::arg("context"),
                    py::arg("num_steps"), py::arg("atol") = 1e-3f);

    py::class_<BlockIndices>(m, "BlockIndices")
        .def(py::init<>())
        .def_readwrite("bb_start",     &BlockIndices::bb_start)
        .def_readwrite("c_start",      &BlockIndices::c_start)
        .def_readwrite("sc_start",     &BlockIndices::sc_start)
        .def_readwrite("o_start",      &BlockIndices::o_start)
        .def_readwrite("h_start",      &BlockIndices::h_start)
        .def_readwrite("amide_donor",  &BlockIndices::amide_donor)
        .def_readwrite("sc_count",     &BlockIndices::sc_count)
        .def_readwrite("res_begin",    &BlockIndices::res_begin)
        .def_readwrite("res_end",      &BlockIndices::res_end)
        .def("has_sidechain",          &BlockIndices::has_sidechain)
        .def("has_hydrogen",           &BlockIndices::has_hydrogen)
        .def("has_oxygen",             &BlockIndices::has_oxygen)
        .def("has_explicit_h",         &BlockIndices::has_explicit_h)
        .def("ca_atom",                &BlockIndices::ca_atom)
        .def("c_atom",                 &BlockIndices::c_atom)
        .def("has_residue_span",       &BlockIndices::has_residue_span);

    py::class_<DownstreamCache>(m, "DownstreamCache")
        .def(py::init<>())
        .def_readwrite("first_sc_of_residue", &DownstreamCache::first_sc_of_residue)
        .def_readwrite("first_o_of_residue",  &DownstreamCache::first_o_of_residue)
        .def_readwrite("first_h_of_residue",  &DownstreamCache::first_h_of_residue);

    py::class_<System, std::shared_ptr<System>>(m, "System",
        "Immutable topology plus the registered energy terms.\n\n"
        "Built by MCPUForceField.create_system() rather than assembled by\n"
        "hand: the engine requires atoms pre-grouped into contiguous\n"
        "backbone/oxygen/sidechain segments, and the five knowledge-based\n"
        "potentials are a fitted set only meaningful together, so\n"
        "create_system registers all of them.\n\n"
        "get_num_atoms() can exceed the input heavy-atom count: each\n"
        "glycine carries one extra bookkeeping slot.")
        .def(py::init<int, int>())
        .def("add_potential",             &System::addPotential)
        .def("get_num_atoms",         &System::getNumAtoms)
        .def("get_num_residues",      &System::getNumResidues)
        .def("get_potentials",           &System::getPotentials, py::return_value_policy::reference_internal)
        .def("set_atom_counts",       &System::setAtomCounts)
        .def("set_virtual_amide_h",    &System::setVirtualAmideH, py::arg("on"))
        .def("virtual_amide_h",       &System::virtualAmideH)
        .def("get_total_h_atoms",      &System::getTotalHAtoms)
        .def("set_block_indices",     &System::setBlockIndices)
        .def("set_torsions_per_residue", &System::setTorsionsPerResidue)
        .def("get_torsions_per_residue", &System::getTorsionsPerResidue, py::return_value_policy::reference_internal)
        .def("set_chi_atom_indices",   &System::setChiAtomIndices)
        .def("get_chi_atom_indices",   &System::getChiAtomIndices, py::return_value_policy::reference_internal)
        .def("set_chi_moved_atom_ranges", &System::setChiMovedAtomRanges)
        .def("get_chi_moved_atom_ranges", &System::getChiMovedAtomRanges, py::return_value_policy::reference_internal)
        .def("set_rotamer_library",  &System::setRotamerLibrary)
        .def("get_rotamer_library",  &System::getRotamerLibrary, py::return_value_policy::reference_internal)
        .def("set_rama_mixture_library", &System::setRamaMixtureLibrary)
        .def("get_rama_mixture_library", &System::getRamaMixtureLibrary, py::return_value_policy::reference_internal)
        .def("set_downstream_cache",  &System::setDownstreamCache)
        .def("set_is_proline",
             [](System& s, const std::vector<int>& flags) {
                 std::vector<uint8_t> u(flags.begin(), flags.end());
                 s.setIsProline(std::move(u));
             },
             py::arg("flags"))
        .def("is_proline",          &System::is_proline, py::arg("res_id"))
        .def("set_amino_index",
             [](System& s, const std::vector<int>& idx) {
                 std::vector<uint8_t> u(idx.begin(), idx.end());
                 s.setAminoIndex(std::move(u));
             },
             py::arg("indices"))
        .def("amino_index",         &System::amino_index, py::arg("res_id"))
        .def("set_secondary_structure", &System::setSecondaryStructure, py::arg("ss"))
        .def("secondary_structure", [](const System& s, int res_id) {
                 return std::string(1, s.secondary_structure(res_id));
             }, py::arg("res_id"))
        .def("residue_contiguous_layout", &System::residueContiguousLayout)
        .def("get_block_indices",     &System::getBlockIndices, py::return_value_policy::reference_internal)
        .def_readwrite("atom_to_residue", &System::atom_to_residue)
        .def("set_energy_ignored_residues",
             [](System& s, const std::vector<int>& residues, const std::string& mode) {
                 mcpu::EnergyMaskMode m;
                 if (mode == "ignore_all") m = mcpu::EnergyMaskMode::IgnoreAll;
                 else if (mode == "clash_only") m = mcpu::EnergyMaskMode::ClashOnly;
                 else throw std::invalid_argument(
                     "mode must be 'ignore_all' or 'clash_only', got: " + mode);
                 s.set_energy_ignored_residues(residues, m);
             },
             py::arg("residues"), py::arg("mode") = "ignore_all")
        .def("clear_energy_ignored_residues", &System::clear_energy_ignored_residues)
        .def("is_residue_energy_ignored", &System::is_residue_energy_ignored, py::arg("res"))
        .def("energy_mask_mode", [](const System& s) -> std::string {
            return s.energy_mask_mode() == mcpu::EnergyMaskMode::IgnoreAll
                ? "ignore_all" : "clash_only";
        })
        .def("__repr__", [](const System& s) {
            return "<System: " + std::to_string(s.getNumAtoms()) +
                " atoms, " + std::to_string(s.getNumResidues()) + " residues>";
        });
    
    // Potentials
    py::class_<Potential, std::shared_ptr<Potential>>(m, "Potential")
    .def("set_energy_group", &Potential::setEnergyGroup)
    .def("get_energy_group", &Potential::getEnergyGroup)
    .def("set_enabled", &Potential::setEnabled, py::arg("enabled"))
    .def("is_enabled", &Potential::isEnabled);
    // @note: Matrices are copied from numpy arrays at construction.
    //        This is a one-time cost — simulation performance is unaffected.
    py::class_<m08::MuPotential, Potential, std::shared_ptr<m08::MuPotential>>(m, "MuPotential",
        "Pairwise contact/solvation potential between MCPU atom types\n"
        "(energy group 1, default outer weight 0.4).\n\n"
        "Dominates run time. A hard-core overlap returns a sentinel energy\n"
        "that rejects the move before the Metropolis test is reached.")
        .def(py::init<Eigen::MatrixXf, Eigen::MatrixXf, Eigen::MatrixXf, std::vector<int>, std::vector<int>>(),
             py::arg("energies"), py::arg("dist_sq"), py::arg("hard_core"),
             py::arg("types"), py::arg("to_residue"))
        .def("cache_necessary_data", &m08::MuPotential::cache_necessary_data,
             py::arg("topo_contact_mask"), py::arg("topo_clash_mask"), py::arg("coords"))
        .def(
            "set_topology_atom_meta",
            &m08::MuPotential::set_topology_atom_meta,
            py::arg("res_index"), py::arg("is_sidechain"), py::arg("atom_role"),
            py::arg("res_class"),
            "Layer 1 per-atom topology metadata for three-layer eval.")
        .def_property(
            "use_topo_flags",
            &m08::MuPotential::use_topo_flags,
            &m08::MuPotential::set_use_topo_flags,
            "Layered v2: precomputed topo_flag_ (default True). "
            "False = v1 on-the-fly Layer 1 decode. MCPU_TOPO_FLAGS=0.")
        .def("verify_layered_eval_consistency",
             &m08::MuPotential::verify_layered_eval_consistency,
             "Compare layered v1 vs v2 for all pairs × test r².")
        .def("bench_eval_pair_only", &m08::MuPotential::bench_eval_pair_only,
             py::arg("n_iter") = 1000000,
             "Microbench eval_pair ns/call (stderr).")
        .def_property_readonly(
            "clash_exception_count",
            &m08::MuPotential::clash_exception_count)
        .def_property_readonly(
            "topo_flag_size_mb", &m08::MuPotential::topo_flag_size_mb)
        .def_property_readonly("type_params_size_kb", &m08::MuPotential::type_params_size_kb)
        .def_property_readonly(
            "mu_exact_cutoff", &m08::MuPotential::mu_exact_cutoff,
            "Denselist query cutoff (Å) from max(type_params_ contact/hard).")
        .def_property_readonly(
            "mu_cutoff_sq", &m08::MuPotential::mu_cutoff_sq,
            "mu_exact_cutoff² used in denselist r² prefilter.")
        .def_property(
            "mm_clash_margin",
            &m08::MuPotential::mm_clash_margin,
            &m08::MuPotential::set_mm_clash_margin,
            "ADDED: MM clash margin Å² (MCPU_MM_CLASH_MARGIN).")
        .def_property(
            "mm_double_boundary",
            &m08::MuPotential::mm_double_boundary,
            &m08::MuPotential::set_mm_double_boundary,
            "ADDED: double MM boundary check (MCPU_MM_DOUBLE_BOUNDARY).")
        .def("calculate_energy_change",
             [](const m08::MuPotential& mu, const Context& context,
                const State& old_state, const State& new_state,
                const ProposalPatch& patch) {
                 return mu.calculateEnergyChange(
                     context, old_state, new_state, patch).delta_energy;
             },
             py::arg("context"), py::arg("old_state"), py::arg("new_state"),
             py::arg("patch"));
    py::class_<forces::TripletPotential, Potential, std::shared_ptr<forces::TripletPotential>>(m, "TripletPotential",
        "Backbone virtual-torsion statistics over a 4D (phi, psi, pCA, bCA)\n"
        "bin grid (energy group 2, default outer weight 1.35). Requires at\n"
        "least three residues.")
        .def(py::init<std::vector<float>>(), py::arg("loaded_params"));
    py::class_<forces::SidechainTripletPotential, Potential, std::shared_ptr<forces::SidechainTripletPotential>>(m, "SidechainTripletPotential",
        "Sidechain chi1-chi4 torsion statistics (energy group 3, default\n"
        "outer weight 2.5). Residues with no rotatable chi angles are\n"
        "skipped.")
        .def(py::init<std::vector<float>>(), py::arg("loaded_params"));
    py::class_<forces::HBondPotential, Potential, std::shared_ptr<forces::HBondPotential>>(m, "HBondPotential",
        "Directional 7D hydrogen-bond potential (energy group 4).\n\n"
        "The configured outer weight is 1.35, but the EFFECTIVE multiplier\n"
        "is 2.7: RDTHREE_CON (2.0) is applied in the group weight rather\n"
        "than inside the term. Builds its own virtual amide hydrogens.")
        .def(py::init<std::vector<float>, std::vector<float>>(),
             py::arg("loaded_params"), py::arg("seq_dep_params"));
    py::class_<forces::AromaticPotential, Potential, std::shared_ptr<forces::AromaticPotential>>(m, "AromaticPotential",
        "Ring-ring aromatic stacking for PHE and TRP (energy group 5,\n"
        "default outer weight 5.0).")
        .def(py::init<std::vector<std::array<int, 3>>, std::vector<float>>(),
             py::arg("aromatic_atom_indices"), py::arg("loaded_params"));
    py::class_<forces::QBiasPotential, Potential, std::shared_ptr<forces::QBiasPotential>>(
            m, "NativeContactsBiasPotential",
            "Harmonic umbrella bias on the native-contact count N (energy\n"
            "group 6): U = 0.5 * k * (N - N0)^2.\n\n"
            "Opt-in. Set k and N0 per replica with\n"
            "Context.set_native_contacts_bias(). The underlying C++ type is\n"
            "QBiasPotential, Q being the native-contact fraction.")
        .def(py::init<std::vector<int>, std::vector<int>, float>(),
             py::arg("ca_atom_i"), py::arg("ca_atom_j"), py::arg("q_cutoff"))
        .def("num_pairs", &forces::QBiasPotential::numPairs)
        .def("initialize_pair_cache", &forces::QBiasPotential::initializePairCache,
             py::arg("state"), py::arg("cache"));

    // ---- KORP lineage -----------------------------------------------------
    py::class_<forces::OrientationalPairMap,
               std::shared_ptr<forces::OrientationalPairMap>>(
            m, "OrientationalPairMap",
            "Engine-side view of a KORP 6D energy map.\n\n"
            "Built by pymcpu.forcefields.builders.korp_builder from a map that\n"
            "pymcpu.forcefields.korp_map has already parsed and validated. The\n"
            "energy table is referenced in place, not copied -- pass the numpy\n"
            "memmap and it stays shared through the OS page cache across every\n"
            "rank on a node. This object keeps that array alive.")
        .def(py::init([](float cutoff, float min_r, int nslices,
                         std::vector<float> br,
                         std::vector<int> shell_ncells,
                         std::vector<int> shell_nchi,
                         std::vector<float> shell_dchi,
                         std::vector<std::int64_t> shell_offset,
                         std::vector<int> ring_offset,
                         std::vector<float> ring_theta,
                         std::vector<float> ring_dpsi,
                         std::vector<int> ring_ncells,
                         std::vector<int> ring_first_cell,
                         std::vector<int> smapping,
                         std::vector<float> fmapping,
                         py::array_t<float, py::array::c_style> table) {
                 // Not forcecast: a table of the wrong dtype or layout must be
                 // an error here, because forcing it would build a temporary
                 // and leave this object pointing at freed memory.
                 std::vector<std::int8_t> smap;
                 smap.reserve(smapping.size());
                 for (int v : smapping) smap.push_back(static_cast<std::int8_t>(v));
                 return std::make_shared<forces::OrientationalPairMap>(
                     cutoff, min_r, nslices, std::move(br),
                     std::move(shell_ncells), std::move(shell_nchi),
                     std::move(shell_dchi), std::move(shell_offset),
                     std::move(ring_offset), std::move(ring_theta),
                     std::move(ring_dpsi), std::move(ring_ncells),
                     std::move(ring_first_cell), std::move(smap),
                     std::move(fmapping),
                     table.data(), static_cast<std::size_t>(table.size()));
             }),
             py::arg("cutoff"), py::arg("min_r"), py::arg("nslices"),
             py::arg("br"), py::arg("shell_ncells"), py::arg("shell_nchi"),
             py::arg("shell_dchi"), py::arg("shell_offset"),
             py::arg("ring_offset"), py::arg("ring_theta"), py::arg("ring_dpsi"),
             py::arg("ring_ncells"), py::arg("ring_first_cell"),
             py::arg("smapping"), py::arg("fmapping"), py::arg("table"),
             py::keep_alive<1, 17>())
        .def_property_readonly("cutoff", &forces::OrientationalPairMap::cutoff)
        .def_property_readonly("min_r", &forces::OrientationalPairMap::min_r)
        .def_property_readonly("num_shells", &forces::OrientationalPairMap::num_shells)
        .def_property_readonly("num_slices", &forces::OrientationalPairMap::num_slices)
        .def("slice_for_separation",
             &forces::OrientationalPairMap::slice_for_separation,
             py::arg("separation"));

    py::class_<forces::OrientationalPairPotential, Potential,
               std::shared_ptr<forces::OrientationalPairPotential>>(
            m, "OrientationalPairPotential",
            "KORP's 6D orientation-dependent residue-pair energy (energy\n"
            "group 7, default outer weight 1.0).\n\n"
            "One frame per residue from its own N, CA and C; the pair\n"
            "coordinate is CA-CA. No sidechain atom is read, which is what\n"
            "makes this usable as a backbone-only force field.\n\n"
            "korp_type indexes KORP's own residue ordering (alphabetical by\n"
            "ONE-letter code), which is not pyMCPU's AMINO_INDEX ordering.\n"
            "seq_number must be PDB residue numbers: KORP takes sequence\n"
            "separation from those rather than from array position.")
        .def(py::init([](std::shared_ptr<forces::OrientationalPairMap> map,
                         std::vector<int> n_atom,
                         std::vector<int> ca_atom,
                         std::vector<int> c_atom,
                         std::vector<int> korp_type,
                         std::vector<int> seq_number,
                         std::vector<int> chain_id) {
                 std::vector<std::uint8_t> types, chains;
                 types.reserve(korp_type.size());
                 for (int v : korp_type) types.push_back(static_cast<std::uint8_t>(v));
                 chains.reserve(chain_id.size());
                 for (int v : chain_id) chains.push_back(static_cast<std::uint8_t>(v));
                 return std::make_shared<forces::OrientationalPairPotential>(
                     std::move(map), std::move(n_atom), std::move(ca_atom),
                     std::move(c_atom), std::move(types), std::move(seq_number),
                     std::move(chains));
             }),
             py::arg("map"), py::arg("n_atom"), py::arg("ca_atom"),
             py::arg("c_atom"), py::arg("korp_type"), py::arg("seq_number"),
             py::arg("chain_id"))
        .def_property_readonly("num_residues",
                               &forces::OrientationalPairPotential::num_residues)
        .def_property_readonly("cutoff_angstrom",
                               &forces::OrientationalPairPotential::cutoff_angstrom)
        .def("set_rigid_skip_enabled",
             &forces::OrientationalPairPotential::set_rigid_skip_enabled,
             py::arg("on"),
             "Testing hook: disable the moved-moved elision so the delta path\n"
             "enumerates every changed pair. Both paths must give the same\n"
             "answer; comparing them is what catches a residue wrongly\n"
             "classified as rigidly moved.")
        .def_property_readonly("rigid_skip_enabled",
                               &forces::OrientationalPairPotential::rigid_skip_enabled);

    py::class_<forces::CalphaExcludedVolumePotential, Potential,
               std::shared_ptr<forces::CalphaExcludedVolumePotential>>(
            m, "CalphaExcludedVolumePotential",
            "CA-CA excluded-volume filter for the KORP force field (energy\n"
            "group 8).\n\n"
            "KORP carries no hard-core repulsion, so on its own it lets a\n"
            "chain pass through itself during MC. This term contributes\n"
            "exactly zero to every accepted state and returns the clash\n"
            "sentinel otherwise, so it filters without shifting the ensemble.")
        .def(py::init([](std::vector<int> ca_atom,
                         std::vector<int> seq_number,
                         std::vector<int> chain_id,
                         int min_separation, float min_distance) {
                 std::vector<std::uint8_t> chains;
                 chains.reserve(chain_id.size());
                 for (int v : chain_id) chains.push_back(static_cast<std::uint8_t>(v));
                 return std::make_shared<forces::CalphaExcludedVolumePotential>(
                     std::move(ca_atom), std::move(seq_number), std::move(chains),
                     min_separation, min_distance);
             }),
             py::arg("ca_atom"), py::arg("seq_number"), py::arg("chain_id"),
             py::arg("min_separation") = 3, py::arg("min_distance") = 3.2f)
        .def_property_readonly("num_residues",
                               &forces::CalphaExcludedVolumePotential::num_residues)
        .def_property_readonly("min_distance",
                               &forces::CalphaExcludedVolumePotential::min_distance)
        .def_property_readonly("min_separation",
                               &forces::CalphaExcludedVolumePotential::min_separation);
}