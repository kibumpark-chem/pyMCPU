#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h> // Necessary for Eigen matrices
#include <pybind11/stl.h>   // Necessary for std::vector
#include <array>
#include <sstream>
#include <iomanip>

#include "pymcpu/BuildConfig.h"  // GENERATED -- see cmake/BuildConfig.h.in

#include "pymcpu/State.h"
#include "pymcpu/System.h"
#include "pymcpu/Context.h"
#include "pymcpu/neighbor/OpenCellGrid.h"
#include "pymcpu/EnergyWeights.h"
#include "pymcpu/Integrator.h"

// Potentials
#include "pymcpu/forces/mcpu/mcpu08/MuPotential.h"
#include "pymcpu/forces/mcpu/common/TripletPotential.h"
#include "pymcpu/forces/mcpu/common/SideChainTripletPotential.h"
#include "pymcpu/forces/mcpu/common/HydrogenBondPotential.h"
#include "pymcpu/forces/mcpu/common/AromaticPotential.h"
#include "pymcpu/forces/bias/QBiasPotential.h"

// Tripeptide Closure
#include "pymcpu/moves/TripeptideClosure.h"
#include "pymcpu/moves/RotamerLibrary.h"
#include "pymcpu/moves/RamaMixtureLibrary.h"

#include "pymcpu/ProposalPatch.h"
#include "pymcpu/testing/PhysicsVerifier.h"
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

    // The clash exception is defined in Python (pymcpu.simulation), next to the
    // checks that raise it from Python; look it up when one is thrown.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const StericClashError& e) {
            PyObject* type = PyExc_RuntimeError;
            py::object clash_error;
            try {
                clash_error = py::module_::import("pymcpu.simulation").attr("StericClashError");
                type = clash_error.ptr();
            } catch (const py::error_already_set&) {
            }
            PyErr_SetString(type, e.what());
        }
    });

    // How far under a hard-core cutoff a whole state may hold a pair (A); a
    // move is tested against the cutoff itself. See Potential.h.
    m.attr("STATE_CLASH_BUFFER_A") = mcpu::kStateClashBufferA;

    // Reporters (bindings match current reporter headers only)
    py::class_<mcpu::Reporter, std::shared_ptr<mcpu::Reporter>>(m, "Reporter",
        "Base class of the output reporters. Exposed so that\n"
        "Simulation.add_reporter has a type to accept; not intended to be\n"
        "subclassed from Python.");
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
        "Writes a CSV row every report_interval steps: step, total, each\n"
        "energy term by name (System.energy_terms()), then <kind>_accepted and\n"
        "<kind>_attempted for each move kind in use (Integrator.move_counts()),\n"
        "then walker_id. Counts are cumulative.\n\n"
        "The header is written when the first run() starts. With append=True\n"
        "an existing header must match, or run() raises before any move.\n"
        "The total column includes the native-contacts bias when one is\n"
        "enabled, which makes it right for monitoring and wrong for MBAR\n"
        "reweighting.")
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
        .def("get_polynomial_coefficients", &TripeptideSolver::get_polynomial_coefficients)
        .def("calculate_jacobian", &TripeptideSolver::calculate_jacobian, py::arg("solution"))
        // Closures dropped by the 1e-6 rad N-CA-C check in the last solve().
        .def("last_rejected", &TripeptideSolver::last_rejected);

    py::class_<mcpu::RotamerComponent>(m, "RotamerComponent")
        .def(py::init<>())
        .def_readwrite("log_weight", &mcpu::RotamerComponent::log_weight)
        .def_readwrite("mean", &mcpu::RotamerComponent::mean)
        // log_mixture_density reads log_sigma, cached from sigma at load; a
        // write through row(...).sigma must refresh it or the density would
        // keep using the old sigma while the proposal samples the new one.
        .def_property(
            "sigma",
            [](const mcpu::RotamerComponent& c) { return c.sigma; },
            [](mcpu::RotamerComponent& c, const std::array<float, 4>& sigma) {
                c.sigma = sigma;
                for (size_t i = 0; i < 4; ++i) c.log_sigma[i] = std::log(sigma[i]);
            });

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
            [](State& s, const Eigen::Matrix3Xf& m) { s.set_coords_from_eigen(m); },
            "Coordinates, shape (3, n_atoms), float32, in Angstrom, in storage "
            "order and the engine frame: for a Context's state, Context.coords "
            "(build order by default) equals these plus Context.frame_offset. "
            "Writing them discards this State's Mu contact list. To move a "
            "Context, use Context.set_positions or Context.coords, which also "
            "refresh its neighbour grids; writing ctx.get_state().coords does "
            "not.")
        .def_property_readonly("current_energy", &State::getEnergy)
        .def_readwrite("backbone_torsions",  &State::backbone_torsions)
        .def_readwrite("sidechain_torsions", &State::sidechain_torsions);

    py::class_<ProposalPatch>(m, "ProposalPatch")
        .def(py::init<int>(), py::arg("num_atoms"))
        .def_readwrite("is_valid", &ProposalPatch::is_valid)
        .def_readwrite("is_rigid", &ProposalPatch::is_rigid)
        .def_readwrite("moving_atoms", &ProposalPatch::moving_atoms)
        // moved_indices and mark_moved are the two ways Python can hand the
        // engine a moved-atom list, so both are checked here, off the engine's
        // hot path: an index outside [0, num_atoms) would write past
        // moving_atoms, and a duplicate breaks the per-cell moved counts in
        // Mu's delta (see ProposalPatch::mark_moved).
        .def_property(
            "moved_indices",
            [](const ProposalPatch& p) { return p.moved_indices; },
            [](ProposalPatch& p, const std::vector<int>& idx) {
                std::vector<std::uint8_t> seen(p.moving_atoms.size(), 0);
                for (int i : idx) {
                    if (i < 0 || static_cast<size_t>(i) >= seen.size())
                        throw py::index_error(
                            "moved_indices: atom index " + std::to_string(i) +
                            " is outside [0, " + std::to_string(seen.size()) + ")");
                    if (seen[static_cast<size_t>(i)]++)
                        throw py::value_error(
                            "moved_indices: atom index " + std::to_string(i) +
                            " is listed twice; the list must not hold duplicates");
                }
                p.moved_indices = idx;
                p.clear_moved_ranges();
            })
        .def(
            "mark_moved",
            [](ProposalPatch& p, int i) {
                if (i < 0 || static_cast<size_t>(i) >= p.moving_atoms.size())
                    throw py::index_error(
                        "mark_moved: atom index " + std::to_string(i) +
                        " is outside [0, " + std::to_string(p.moving_atoms.size()) + ")");
                // Marking an atom twice is a no-op, as the mask already was.
                if (p.moving_atoms[static_cast<size_t>(i)] == 0) p.mark_moved(i);
            },
            py::arg("i"),
            "Mark atom i as moved. Marking an already-marked atom does nothing.")
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
        .def("set_use_legacy_weights", &EnergyWeights::set_use_legacy_weights, py::arg("on"))
        .def("set_energy_weight", &EnergyWeights::set_energy_weight,
             py::arg("group_id"), py::arg("w"))
        .def("weight_for_group", &EnergyWeights::weight_for_group, py::arg("group_id"))
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
        "Constructed from a finished System. The Integrator recomputes the\n"
        "total energy on entry whenever the coordinates or anything that\n"
        "defines the energy (mask, weights, bias, enabled potentials) changed\n"
        "since the last full recompute, so current_energy is exact after a run.\n\n"
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
        .def("set_positions", &Context::setPositions,
             py::arg("coords"), py::pos_only(), py::kw_only(),
             py::arg("frame_offset") = py::none(),
             "Place the atoms: coords, shape (3, n_atoms), in Angstrom, build "
             "order, the caller's frame. They are rounded to float32 once, in "
             "the engine frame (coords - frame_offset).\n\n"
             "The first placement fixes frame_offset: if any coordinate is "
             "64 A or more from the origin, each axis whose coordinates all "
             "have one sign is shifted toward it by a whole number of A, and "
             "every coordinate output adds the shift back. For float32 input "
             "that shift is exact. Pass frame_offset (A, shape (3,)) to set it "
             "instead; it then stays for later placements too. Entry is exact "
             "only for float32 input with an offset of whole A per axis, of "
             "the axis's sign and at most twice its smallest |coordinate| "
             "(such as another Context's frame_offset for the same start).")
        .def_property_readonly(
            "frame_offset",
            [](const Context& c) { return Eigen::Vector3d(c.frame_offset()); },
            "Shift (A, float64, shape (3,)) from the engine frame to the "
            "caller's: Context.coords = engine coordinates + frame_offset. "
            "Zero when every coordinate of the first placement is within 64 A "
            "of the origin; otherwise each axis whose coordinates all have "
            "one sign is shifted. set_positions(..., frame_offset=) sets it, "
            "and recenter() moves it with the chain.")
        .def("recenter", &Context::recenter,
             py::arg("min_reach_A") = kFrameShiftMinA,
             "Move the engine frame to a chain that drifted away from the "
             "origin, and return the shift (A, float64, shape (3,)): zero, "
             "with nothing changed, unless an engine coordinate reaches "
             "min_reach_A. Each axis is shifted by the whole-A midpoint of its "
             "coordinate range; frame_offset grows by the shift and "
             "Context.coords stay the same. The shift is exact for every atom "
             "that ends no farther from the origin than it started; one that "
             "ends farther out is rounded once, by at most half a float32 step "
             "there. The energy is recomputed. A stale energy is recomputed "
             "first, as the next run would (raising on a clash), and a state "
             "the last recompute found clashing is left alone. A shift whose "
             "rounding leaves a hard-core clash is undone exactly and returns "
             "zero. Simulation.step calls it after each periodic full "
             "recompute (Simulation.recenter_frame).")
        .def_property(
            "coords",
            &Context::coords_for_python,
            &Context::set_coords_from_python,
            "Coordinates, shape (3, n_atoms), float64, in Angstrom and the "
            "caller's frame: engine + frame_offset, exact in float64 except "
            "for an engine coordinate within a few 1e-6 A of zero. External "
            "(build) order by default; set_output_internal_order(True) for "
            "storage order. Setting them is set_positions in the same order.")
        .def("coords_for_python", &Context::coords_for_python)
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
                 py::dict by_name;
                 for (const auto& [g, name] : c.getSystem().energyTerms()) {
                     auto it = src.find(g);
                     by_name[py::str(name)] = it != src.end() ? it->second : 0.0f;
                 }
                 d["by_name"] = by_name;
                 d["weighted"] = weighted;
                 d["use_legacy_weights"] = c.use_legacy_weights();
                 return d;
             },
             py::arg("weighted") = true,
             "Return per-term energies, keyed by group (by_group) and by name "
             "(by_name). weighted=True uses legacy outer weights (incl. HBond "
             "RDTHREE_CON); weighted=False returns raw per-potential energies.")
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
        .def_property_readonly("energy_resyncs", &Context::energy_resyncs,
             "How many full recomputes the Integrator has done on entry because "
             "the coordinates or the energy definition changed.")
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
        .def("neighbor_grid_rebuilds",
             [](const Context& c) { return c.neighborStats().num_grid_rebuilds; },
             "Full rebuilds of the neighbour grids since the last "
             "reset_neighbor_proxy_stats(): one per set_positions and one "
             "after each cell overflow.")
        .def("hbond_index_ok",
             [](const Context& c) {
                 return c.neighbors().count_hbond_candidate_mismatches(c.getState().coords_soa) == 0;
             })
        .def("set_hbond_ledger_check",
             [](Context& c, bool on) { c.getHBondWorkspace().ledger_check = on; },
             "Debug: on every H-bond delta, re-score both states over all pairs "
             "touching an affected residue and count disagreements with the ledger.")
        .def("hbond_ledger_check_counts",
             [](Context& c) {
                 const auto& w = c.getHBondWorkspace();
                 return py::make_tuple(w.ledger_checks, w.ledger_mismatches);
             })
        .def("hbond_uses_fallback",
             [](const Context& c) { return c.neighbors().hbondUsesFallback(); })
        .def("hbond_backend_name",
             [](const Context& c) { return c.neighbors().hbond_backend_name(); })
        .def("mu_backend_name",
             [](const Context& c) { return c.neighbors().mu_backend_name(); })
        .def("set_skip_rigid_mm", &Context::set_skip_rigid_mm, py::arg("on"),
             "Skip re-measuring the pairs a rigid pivot carries (both atoms "
             "moved): their distances change only by rounding, so Mu re-decides "
             "just the carried pairs on its contact list. The H-bond term also "
             "keeps the energy of a donor-acceptor pair whose backbone geometry "
             "(residues r-1 to r+1 on both sides) moved as one body, and carries "
             "its ledger entry. Default True; False evaluates them all exactly, "
             "as a reference.")
        .def("skip_rigid_mm", &Context::skip_rigid_mm)
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
        .def("reset_neighbor_proxy_stats", &Context::reset_neighbor_proxy_stats)
        .def("neighbor_proxy_stats",
             [](const Context& c) {
                 const auto& s = c.neighborStats();
                 const auto& cfg = c.neighborConfig();
                 const auto r = s.derive(s.num_steps_executed);
                 py::dict d;
                 d["mu_num_pair_distance_checks"] = s.mu_num_pair_distance_checks;

                 d["mu_num_pairs_within_rcut"] = s.mu_num_pairs_within_rcut;
                 d["hbond_num_candidates_iterated"] = s.hbond_num_candidates_iterated;
                 d["hbond_num_geom_checks"] = s.hbond_num_geom_checks;
                 d["neighbor_num_cell_visits"] = s.neighbor_num_cell_visits;
                 d["mu_eval_pair_calls"] = s.mu_eval_pair_calls;
                 d["elided_rigid_mm"] = s.elided_rigid_mm;
                 d["skip_rigid_mm"] = cfg.skip_rigid_mm;
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
                 d["mu_grid_active"] = c.neighbors().denseActive();
                 d["mu_grid_overflows"] = s.mu_grid_overflows;
                 d["hbond_grid_overflows"] = s.hbond_grid_overflows;
                 // Live Mu grid occupancy, for cell-size tuning.
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
                     d["mu_grid_n_atoms"] = sum;
                     d["mu_grid_peak_occupancy"] = g.peak_cell_occupancy();
                     d["mu_grid_cell_capacity"] = OpenCellGrid::CELL_CAPACITY;
                 } else {
                     d["mu_grid_n_cells"] = 0;
                     d["mu_grid_n_occupied"] = 0;
                     d["mu_grid_avg_occupancy"] = 0.0;
                     d["mu_grid_max_occupancy"] = 0;
                     d["mu_grid_n_atoms"] = 0;
                     d["mu_grid_peak_occupancy"] = 0;
                     d["mu_grid_cell_capacity"] = OpenCellGrid::CELL_CAPACITY;
                 }
                 d["num_grid_rebuilds"] = s.num_grid_rebuilds;
                 d["num_reject_hard_disp"] = s.num_reject_hard_disp;
                 d["num_steps_executed"] = s.num_steps_executed;
                 d["total_steps"] = r.total_steps;
                 d["avg_mu_pair_checks_per_step"] = r.avg_mu_pair_checks_per_step;
                 d["avg_mu_pairs_within_rcut_per_step"] = r.avg_mu_pairs_within_rcut_per_step;
                 d["avg_mu_pairs_per_step"] = r.avg_mu_pairs_per_step;
                 d["avg_cell_visits_per_step"] = r.avg_cell_visits_per_step;
                 d["avg_hbond_geom_checks_per_step"] = r.avg_hbond_geom_checks_per_step;
                 py::dict derived;
                 derived["total_steps"] = r.total_steps;
                 derived["avg_mu_pair_checks_per_step"] = r.avg_mu_pair_checks_per_step;
                 derived["avg_mu_pairs_within_rcut_per_step"] = r.avg_mu_pairs_within_rcut_per_step;
                 derived["avg_mu_pairs_per_step"] = r.avg_mu_pairs_per_step;
                 derived["avg_cell_visits_per_step"] = r.avg_cell_visits_per_step;
                 derived["avg_hbond_geom_checks_per_step"] = r.avg_hbond_geom_checks_per_step;
                 d["derived"] = derived;
                 return d;

             });


    py::class_<mcpu::MCIntegrator>(m, "Integrator",
        "Metropolis Monte Carlo move engine: backbone pivot, continuous\n"
        "sidechain, rotamer-library and kinematic-closure loop moves.\n\n"
        "temperature is required: a DIMENSIONLESS reduced parameter, not Kelvin.\n"
        "Where a protein unfolds depends on the protein; chignolin melts at\n"
        "about 0.65 to 0.7.\n\n"
        "Call set_seed(): the same seed, input and build reproduce a run exactly.")
        .def(py::init<double, float, float>(), py::arg("temperature"),
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
             "Selects the Sidechain-slot proposal algorithm: 'rotamer_library' "
             "(default) or 'continuous'.")
        .def("sidechain_move_mode", &mcpu::MCIntegrator::sidechain_move_mode)
        .def("set_pivot_rama_probability", &mcpu::MCIntegrator::set_pivot_rama_probability,
             py::arg("p"),
             "Fraction of Pivot-slot attempts using the knowledge-based "
             "(phi,psi) rama-mixture proposal instead of the continuous "
             "single-dihedral pivot. Default 0.0 (opt-in); at p=0.0 no extra "
             "RNG draw is consumed.")
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
        .def("last_move_kind", &mcpu::MCIntegrator::last_move_kind,
             "Kind of the last proposed move (Pivot/KIC/Sidechain/Other), from "
             "run() or the last debug_force_* call that proposed a move.")
        .def("last_move_is_rigid", &mcpu::MCIntegrator::last_move_is_rigid,
             "Whether the last proposed move was a rigid body move.")
        .def("last_moved_indices", &mcpu::MCIntegrator::last_moved_indices,
             py::return_value_policy::copy,
             "Atom indices moved by the last proposed move.")
        .def("last_delta_energy", &mcpu::MCIntegrator::last_delta_energy,
             "After run(): the energy change of the last step that moved "
             "atoms if it was accepted, else 0. After a debug_force_* call: "
             "the forced proposal's energy change (the move is not committed).")
        .def("last_log_jacobian_weight", &mcpu::MCIntegrator::last_log_jacobian_weight,
             "Metropolis-Hastings correction term of the last forced proposal "
             "from a debug_force_* call (0 for a symmetric move).")
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
                 py::dict by_group;
                 // Only groups 1..7 have a timing slot (energy_delta_ns is [8]).
                 for (const auto& [g, name] : s.energy_terms) {
                     if (g >= 1 && g < 8) by_group[py::str(name)] = s.energy_delta_ns[g];
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
        // KIC reverse-check refusals and proline-phi skips.
        .def("get_kic_reverse_missing", &mcpu::MCIntegrator::get_kic_reverse_missing)
        .def("get_kic_proline_skipped", &mcpu::MCIntegrator::get_kic_proline_skipped)
        .def("get_steric_rejected", &mcpu::MCIntegrator::get_steric_rejected)
        .def("get_move_counters",
             [](const mcpu::MCIntegrator& integ) {
                 py::dict d;
                 for (const auto& [name, value] : integ.get_move_counters()) d[py::str(name)] = value;
                 return d;
             },
             "Every move counter as {name: count}, for checkpointing. Restore "
             "with set_move_counters().")
        .def("set_move_counters",
             [](mcpu::MCIntegrator& integ, const py::dict& counters) {
                 std::vector<std::pair<std::string, long long>> v;
                 for (const auto& kv : counters) {
                     v.emplace_back(py::cast<std::string>(kv.first), py::cast<long long>(kv.second));
                 }
                 integ.set_move_counters(v);
             },
             py::arg("counters"),
             "Restore counters saved by get_move_counters(). All counters are "
             "reset to 0 first; an unknown name raises ValueError and changes "
             "nothing.")
        .def("move_counts",
             [](const mcpu::MCIntegrator& integ, bool include_unused) {
                 py::dict d;
                 for (const auto& c : integ.move_counts()) {
                     if (c.in_use || include_unused) {
                         d[py::str(c.name)] = py::make_tuple(c.accepted, c.attempted);
                     }
                 }
                 return d;
             },
             py::arg("include_unused") = false,
             "Per-move-kind counts as {kind: (accepted, attempted)}, each move "
             "counted once: pivot, rama_pivot, kic, sidechain, rotamer. Kinds "
             "the current move weights and sidechain mode cannot propose are "
             "left out unless include_unused=True.")
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
                 d["kic_reverse_missing"] = integ.get_kic_reverse_missing();
                 d["kic_proline_skipped"] = integ.get_kic_proline_skipped();
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
#if !defined(MCPU_BUILD_ARCH_TIER) || !defined(MCPU_BUILD_LTO) || !defined(MCPU_BUILD_JCC_PAD)
#error "BuildConfig.h was not generated; configure through CMake."
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
            // EFFECTIVE, like lto: on only when CMake's probes showed the
            // assembler that emits the shipped code honours the option.
            build["jcc_pad"] = (MCPU_BUILD_JCC_PAD != 0);
            build["jcc_pad_mode"] = MCPU_BUILD_JCC_PAD_MODE;
            build["jcc_pad_reason"] = MCPU_BUILD_JCC_PAD_REASON;
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
            d["deps"] = deps;
            return d;
        },
        "What this binary was compiled with: ISA baseline, compiler, build "
        "type, LTO, FP policy and feature flags. Every value derives from a "
        "real macro -- see build_info() in src/bindings/bindings.cpp.");

    // Test hook for the cell grid on its own: file the atoms of ``xyz``
    // (3 x N, A) in a grid over the box [lo, hi), then list, for each probe
    // (3 x M), the atoms one walk visits, in visit order. ``walk`` is
    // "stencil" (for_each_neighbor), "spans" (the moved-aware stencil walk
    // with no atom moved) or "within" (the radius walk, with ``radius``).
    // Returns (lists, (nx, ny, nz), overflowed).
    m.def("_cell_grid_walk",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> xyz,
           float cell, float query_radius, std::array<float, 3> lo,
           std::array<float, 3> hi, std::uint64_t max_cells,
           py::array_t<float, py::array::c_style | py::array::forcecast> probes,
           const std::string& walk, float radius) {
            if (xyz.ndim() != 2 || xyz.shape(0) != 3 || probes.ndim() != 2 ||
                probes.shape(0) != 3)
                throw std::invalid_argument("xyz and probes must be 3 x N");
            const auto a = xyz.unchecked<2>();
            const auto p = probes.unchecked<2>();
            const int n = static_cast<int>(xyz.shape(1));
            BoxBounds b;
            b.lo = Eigen::Vector3f(lo[0], lo[1], lo[2]);
            b.hi = Eigen::Vector3f(hi[0], hi[1], hi[2]);
            b.valid = true;
            OpenCellGrid g(cell, n);
            if (!g.configure(b, cell, max_cells, query_radius))
                throw std::invalid_argument("invalid box");
            for (int i = 0; i < n; ++i) g.insert(i, a(0, i), a(1, i), a(2, i));
            const std::vector<std::uint8_t> none(static_cast<size_t>(g.num_cells()) + 8, 0);
            std::vector<std::vector<int>> out(static_cast<size_t>(probes.shape(1)));
            for (py::ssize_t k = 0; k < probes.shape(1); ++k) {
                std::vector<int>& ids = out[static_cast<size_t>(k)];
                auto span = [&](const int* id, const float*, const float*,
                                const float*, int count) {
                    ids.insert(ids.end(), id, id + count);
                    return true;
                };
                if (walk == "stencil")
                    g.for_each_neighbor(p(0, k), p(1, k), p(2, k),
                                        [&](int j) { ids.push_back(j); });
                else if (walk == "spans")
                    g.for_each_neighbor_cell_span_while_unmoved(
                        p(0, k), p(1, k), p(2, k), none.data(), span);
                else if (walk == "within")
                    g.for_each_cell_span_within_fast_unmoved(
                        p(0, k), p(1, k), p(2, k), radius, none.data(), span);
                else
                    throw std::invalid_argument("unknown walk");
            }
            return py::make_tuple(out, py::make_tuple(g.nx(), g.ny(), g.nz()),
                                  g.overflowed());
        },
        py::arg("xyz"), py::arg("cell"), py::arg("query_radius"), py::arg("lo"),
        py::arg("hi"), py::arg("max_cells"), py::arg("probes"), py::arg("walk"),
        py::arg("radius") = 0.f);


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
        "For MCPUForceField, get_num_atoms() is the input heavy-atom\n"
        "count, plus one slot per explicit amide hydrogen when\n"
        "virtual_amide_h=False.")
        .def(py::init<int, int>())
        .def("add_potential",             &System::addPotential)
        .def("get_num_atoms",         &System::getNumAtoms)
        .def("get_num_residues",      &System::getNumResidues)
        .def("get_potentials",           &System::getPotentials, py::return_value_policy::reference_internal)
        .def("energy_terms",
             [](const System& s) {
                 py::dict d;
                 for (const auto& [g, name] : s.energyTerms()) {
                     d[py::int_(g)] = name;
                 }
                 return d;
             },
             "Energy terms as {group: name}, sorted by group. Unnamed groups "
             "are reported as 'group_<n>'.")
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
        // KIC closure targets from the START coordinates (3 x n_atoms, Angstrom,
        // build order). MCPUForceField.create_system calls it; KIC refuses to run without it.
        .def("set_kic_reference", &System::setKicReference, py::arg("start_coords"))
        .def("has_kic_reference", &System::hasKicReference)
        .def("get_kic_reference", [](const System& s) {
                 const auto& r = s.kicReference();
                 py::dict d;
                 d["len_na"] = r.len_na;   d["len_ac"] = r.len_ac;   d["ang_nac"] = r.ang_nac;
                 d["len_cn"] = r.len_cn;   d["ang_acn"] = r.ang_acn; d["ang_cna"] = r.ang_cna;
                 d["omega"] = r.omega;
                 return d;
             })
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
    .def("set_name", &Potential::setName, py::arg("name"),
         "Name this energy term (e.g. 'mu'); see System.energy_terms().")
    .def("get_name", &Potential::getName)
    .def("set_enabled", &Potential::setEnabled, py::arg("enabled"))
    .def("is_enabled", &Potential::isEnabled);
    // The per-atom-pair matrices are reduced to one entry per atom-type pair
    // at construction and not kept.
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
        .def_property_readonly(
            "mu_exact_cutoff", &m08::MuPotential::mu_exact_cutoff,
            "Neighbour query cutoff (Å): the largest contact cutoff widened by "
            "the contact list's 0.05 Å near-miss band, or the largest hard-core "
            "cutoff if larger, times 1.0001.")
        .def_property_readonly(
            "contact_list_rebuilds", &m08::MuPotential::contact_list_rebuilds,
            "Times a state's contact list was rebuilt from its coordinates "
            "(diagnostic; shared by the replicas that share this potential).")
        .def_property_readonly(
            "clist_fallbacks", &m08::MuPotential::clist_fallbacks,
            "Moves the contact list could not follow (a grid cell overflowed, "
            "or a rigid move carried past the list's drift budget); each "
            "accepted one costs an O(N^2) list rebuild (diagnostic; shared "
            "like contact_list_rebuilds).")
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
        .def("num_pairs", &forces::QBiasPotential::numPairs);
}