#include "pymcpu/reporters/EnergyReporter.h"
#include "pymcpu/Integrator.h"
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace mcpu {

std::map<std::string, double> EnergyComponents::to_dict() const {
    return {
        {"total", total},
        {"mu", mu},
        {"backbone_torsion", backbone_torsion},
        {"sidechain_torsion", sidechain_torsion},
        {"hydrogen_bond", hydrogen_bond},
        {"aromatic", aromatic},
        {"native_contacts_bias", native_contacts_bias},
    };
}

double EnergyComponents::get(const std::string& component) const {
    auto d = to_dict();
    auto it = d.find(component);
    if (it == d.end()) {
        throw std::invalid_argument("Unknown energy component: " + component);
    }
    return it->second;
}

EnergyReporter::EnergyReporter(const std::string& energy_filename, int report_interval,
                               bool append)
    : Reporter(report_interval), filename_(energy_filename) {
    
    energy_file_.open(energy_filename, append ? (std::ios::out | std::ios::app)
                                              : (std::ios::out | std::ios::trunc));
    if (!energy_file_.is_open()) {
        throw std::runtime_error("Failed to open energy file for writing: " + energy_filename);
    }
    
    // OpenMM-style single monitoring CSV: energy components (Context) + MC
    // move counts (Integrator). Total includes NativeContactsBias (biased).
    if (!append) {
        energy_file_ << "Step,Total,Mu,BackboneTorsion,SidechainTorsion,HydrogenBond,Aromatic,"
                        "NativeContactsBias,PivotAccepted,PivotAttempted,SidechainAccepted,"
                        "SidechainAttempted,KicAccepted,KicAttempted,WalkerId\n";
        energy_file_.flush();
        if (energy_file_.fail()) {
            throw std::runtime_error("Failed to write header to energy file: " + energy_filename);
        }
    }
}

EnergyReporter::~EnergyReporter() {
    if (energy_file_.is_open()) {
        energy_file_.close();
    }
}

void EnergyReporter::report(int step, const Context& context, const MCIntegrator& integrator) {
    EnergyComponents components;
    components.total = static_cast<double>(context.getState().getEnergy());

    if (per_component_) {
        EnergyBreakdown bd = context.energy_breakdown();
        auto get_group = [&](int g) -> double {
            auto it = bd.weighted_by_group.find(g);
            return it != bd.weighted_by_group.end() ? static_cast<double>(it->second) : 0.0;
        };
        components.mu = get_group(1);
        components.backbone_torsion = get_group(2);
        components.sidechain_torsion = get_group(3);
        components.hydrogen_bond = get_group(4);
        components.aromatic = get_group(5);
        components.native_contacts_bias = get_group(6);
    }

    const long long p_acc = integrator.get_bb_accepted();
    const long long p_att = integrator.get_bb_attempted();
    const long long sc_acc = integrator.get_sc_accepted();
    const long long sc_att = integrator.get_sc_attempted();
    const long long kic_acc = integrator.get_kic_accepted();
    const long long kic_att = integrator.get_kic_attempted();

    // Format one complete line in memory, then write once. This keeps each
    // record contiguous (important on NFS when many ranks write concurrently
    // to different files, and avoids torn multi-operator << sequences).
    std::ostringstream line;
    line.setf(std::ios::fixed, std::ios::floatfield);
    line << std::setprecision(8)
         << step << ","
         << components.total << ","
         << components.mu << ","
         << components.backbone_torsion << ","
         << components.sidechain_torsion << ","
         << components.hydrogen_bond << ","
         << components.aromatic << ","
         << components.native_contacts_bias << ","
         << p_acc << "," << p_att << ","
         << sc_acc << "," << sc_att << ","
         << kic_acc << "," << kic_att << ","
         << walker_id_ << "\n";
    const std::string s = line.str();
    energy_file_.write(s.data(), static_cast<std::streamsize>(s.size()));
    if (energy_file_.fail()) {
        throw std::runtime_error("Failed to write frame to energy file at step " + std::to_string(step));
    }
    energy_file_.flush();
    if (energy_file_.fail()) {
        throw std::runtime_error("Failed to flush energy file at step " + std::to_string(step));
    }
    ++n_frames_written_;
}

} // namespace mcpu
