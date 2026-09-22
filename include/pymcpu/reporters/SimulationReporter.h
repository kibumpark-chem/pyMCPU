#pragma once
#include "pymcpu/reporters/Reporter.h"

namespace mcpu {

class SimulationReporter : public Reporter {
public:
    SimulationReporter(int report_interval);
    ~SimulationReporter() override = default;

    void report(int step, const Context& context, const MCIntegrator& integrator) override;
};

} // namespace mcpu