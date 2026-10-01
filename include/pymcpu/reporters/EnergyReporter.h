#pragma once
#include "pymcpu/reporters/Reporter.h"
#include <cstddef>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace mcpu {

/// Writes one CSV row per report: step, the weighted total, each energy term
/// by name, accepted/attempted counts for each move kind in use, walker_id.
///
/// The columns come from the System's energy terms and the integrator's move
/// kinds, so the header is written at the first run() (begin_run), not at
/// construction. In append mode an existing header must match exactly, and a
/// missing or empty file gets a fresh one.
class EnergyReporter : public Reporter {
private:
    std::ofstream energy_file_;
    std::string filename_;
    long long n_frames_written_ = 0;
    int walker_id_ = -1;

    /// First line of the file when appending to one that already had content;
    /// empty when the header still has to be written.
    std::string existing_header_;
    bool header_ready_ = false;
    std::vector<std::pair<int, std::string>> terms_;
    /// Indices into MCIntegrator::move_counts() that have columns.
    std::vector<std::size_t> move_columns_;

    void ensure_header(const Context& context, const MCIntegrator& integrator);
    void check_terms_unchanged(const Context& context) const;

public:
    EnergyReporter(const std::string& energy_filename, int report_interval,
                   bool append = false);
    ~EnergyReporter() override;

    void begin_run(const Context& context, const MCIntegrator& integrator) override;
    void report(int step, const Context& context, const MCIntegrator& integrator) override;

    void set_walker_id(int walker_id) noexcept { walker_id_ = walker_id; }
    [[nodiscard]] int walker_id() const noexcept { return walker_id_; }

    [[nodiscard]] long long n_frames_written() const noexcept { return n_frames_written_; }
    [[nodiscard]] const std::string& filename() const noexcept { return filename_; }
};

} // namespace mcpu
