#include "pymcpu/reporters/SimulationReporter.h"
#include "pymcpu/Integrator.h" 
#include <iostream>

namespace mcpu {

SimulationReporter::SimulationReporter(int report_interval)
    : Reporter(report_interval) {
}

void SimulationReporter::report(int step, const Context& context, const MCIntegrator& integrator) {
    
    // Read raw counts from the integrator
    long long p_acc = integrator.get_bb_accepted();
    long long p_att = integrator.get_bb_attempted();

    long long sc_acc = integrator.get_sc_accepted();
    long long sc_att = integrator.get_sc_attempted();

    long long kic_acc = integrator.get_kic_accepted();
    long long kic_att = integrator.get_kic_attempted();
    
    double energy = context.getState().getEnergy();
    
    std::cout << "--- Step: " << step << " ---\n"
              << "Pivot Moves:     " << p_acc << " / " << p_att << "\n"
              << "Sidechain Moves: " << sc_acc << " / " << sc_att << "\n"
              << "Concerted Moves: " << kic_acc << " / " << kic_att << "\n"
              << "Energy:          " << energy << "\n\n";
}

} // namespace mcpu