#include "pymcpu/reporters/SimulationReporter.h"
#include "pymcpu/Integrator.h" 
#include <iostream>
#include <string>

namespace mcpu {

SimulationReporter::SimulationReporter(int report_interval)
    : Reporter(report_interval) {
}

void SimulationReporter::report(int step, const Context& context, const MCIntegrator& integrator) {
    std::cout << "--- Step: " << step << " ---\n";
    for (const auto& c : integrator.move_counts()) {
        if (!c.in_use) continue;
        std::cout << c.name << ":" << std::string(12 - std::string(c.name).size(), ' ')
                  << c.accepted << " / " << c.attempted << "\n";
    }
    // Flushed so progress shows up in job logs as the run goes.
    std::cout << "energy:      " << context.getState().getEnergy() << "\n\n" << std::flush;
}

} // namespace mcpu