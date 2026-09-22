#pragma once
#include "pymcpu/reporters/Reporter.h"
#include "pymcpu/EnergyWeights.h"
#include <string>
#include <fstream>
#include <map>
#include <vector>

namespace mcpu {

struct EnergyComponents {
    double total = 0.0;
    double mu = 0.0;
    double backbone_torsion = 0.0;
    double sidechain_torsion = 0.0;
    double hydrogen_bond = 0.0;
    double aromatic = 0.0;
    double native_contacts_bias = 0.0;

    std::map<std::string, double> to_dict() const;
    double get(const std::string& component) const;
};

class EnergyReporter : public Reporter {
private:
    std::ofstream energy_file_;
    bool per_component_ = true;
    std::string filename_;
    long long n_frames_written_ = 0;
    int walker_id_ = -1;

public:
    EnergyReporter(const std::string& energy_filename, int report_interval,
                   bool append = false);
    ~EnergyReporter() override;

    void report(int step, const Context& context, const MCIntegrator& integrator) override;

    void set_walker_id(int walker_id) noexcept { walker_id_ = walker_id; }
    [[nodiscard]] int walker_id() const noexcept { return walker_id_; }

    [[nodiscard]] long long n_frames_written() const noexcept { return n_frames_written_; }
    [[nodiscard]] const std::string& filename() const noexcept { return filename_; }
};

} // namespace mcpu