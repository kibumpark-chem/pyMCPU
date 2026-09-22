#pragma once
#include "pymcpu/reporters/Reporter.h"
#include <string>
#include <vector>

// Forward declare the C struct so we don't have to include the messy C header here!
struct XDRFILE;

namespace mcpu {

class XtcReporter : public Reporter {
private:
    XDRFILE* xtc_file_;
    float precision_;

    std::vector<int> inverse_mapping_;
    int original_natoms_;
    std::string filename_;
    long long n_frames_written_ = 0;

public:
    XtcReporter(const std::string& xtc_filename, 
                int report_interval,
                const std::vector<int>& inverse_mapping = {},
                bool append = false);
    ~XtcReporter() override;

    void report(int step, const Context& context, const MCIntegrator& integrator) override;

    /// Push pending XTC bytes to the OS (stdio fflush; not fsync).
    void flush();

    [[nodiscard]] long long n_frames_written() const noexcept { return n_frames_written_; }
    [[nodiscard]] const std::string& filename() const noexcept { return filename_; }
};

} // namespace mcpu