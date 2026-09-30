#pragma once
#include "pymcpu/Context.h"

namespace mcpu {

// Forward declaration of your integrator so reporters can read its stats
class MCIntegrator; 

class Reporter {
protected:
    int report_interval_;
public:
    Reporter(int interval) : report_interval_(interval) {}
    virtual ~Reporter() = default;

    /// Called once at the start of every MCIntegrator::run(), before any
    /// move and before the step-0 report. A reporter that must validate its
    /// output against the System (EnergyReporter's CSV header) does it here,
    /// so a problem stops the run before it starts rather than mid-way.
    virtual void begin_run(const Context& /*context*/, const MCIntegrator& /*integrator*/) {}

    // The virtual function every specific logger will implement
    virtual void report(int step, const Context& context, const MCIntegrator& integrator) = 0;
    
    int get_interval() const { return report_interval_; }
};

} // namespace mcpu