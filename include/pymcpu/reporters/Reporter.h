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

    // The virtual function every specific logger will implement
    virtual void report(int step, const Context& context, const MCIntegrator& integrator) = 0;
    
    int get_interval() const { return report_interval_; }
};

} // namespace mcpu