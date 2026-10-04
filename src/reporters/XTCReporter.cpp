#include "pymcpu/reporters/XTCReporter.h"
#include <algorithm>
#include <vector>
#include <stdexcept>

// Include the C library here, completely hidden from the rest of your C++ project
extern "C" {
    #include "pymcpu/reporters/xdrfiles/xdrfile.h"
    #include "pymcpu/reporters/xdrfiles/xdrfile_xtc.h"
}

namespace mcpu {

XtcReporter::XtcReporter(const std::string& xtc_filename,
                         int report_interval,
                         const std::vector<int>& inverse_mapping,
                         bool append)
    : Reporter(report_interval),
      precision_(1000.0f),
      inverse_mapping_(inverse_mapping),
      filename_(xtc_filename) {
    
    xtc_file_ = xdrfile_open(xtc_filename.c_str(), append ? "a" : "w");
    if (!xtc_file_) {
        throw std::runtime_error("Failed to open XTC file for writing: " + xtc_filename);
    }

    if (!inverse_mapping_.empty()) {
        auto max_idx_iter = std::max_element(inverse_mapping_.begin(), inverse_mapping_.end());
        original_natoms_ = (max_idx_iter != inverse_mapping_.end() && *max_idx_iter >= 0) 
                           ? *max_idx_iter + 1 
                           : 0;
    } else {
        original_natoms_ = -1; 
    }
}

XtcReporter::~XtcReporter() {
    if (xtc_file_) {
        xdrfile_close(xtc_file_);
        xtc_file_ = nullptr;
    }
}

void XtcReporter::flush() {
    if (!xtc_file_) {
        return;
    }
    if (xdrfile_flush(xtc_file_) != 0) {
        throw std::runtime_error("Failed to flush XTC file");
    }
}

void XtcReporter::report(int step, const Context& context, const MCIntegrator& integrator) {
    // 1. Get coordinates using the public getter (internal storage order)
    const auto& soa = context.getState().coords_soa;
    int internal_natoms = soa.n;
    const auto& perm = context.atom_permutation();
    const bool map_ext = !perm.is_identity() && !context.output_internal_order();
    // Written in the user's frame: engine coordinate + frame offset, in double.
    const Eigen::Vector3d& offset = context.frame_offset();
    auto nm = [&offset](float engine, int d) {
        const double user = offset[d] != 0.0 ? engine + offset[d] : engine;
        return static_cast<float>(user / 10.0);
    };

    // 2. Prepare the C-style array for xdrfile (converting Angstroms to Nanometers)
    int out_natoms = (original_natoms_ > 0) ? original_natoms_ : internal_natoms;
    std::vector<rvec> x(out_natoms);

    if (inverse_mapping_.empty()) {
        for (int i = 0; i < internal_natoms; ++i) {
            const int out_i = map_ext ? perm.to_external(i) : i;
            if (out_i < 0 || out_i >= out_natoms) continue;
            x[out_i][0] = nm(soa.x[static_cast<size_t>(i)], 0);
            x[out_i][1] = nm(soa.y[static_cast<size_t>(i)], 1);
            x[out_i][2] = nm(soa.z[static_cast<size_t>(i)], 2);
        }
    } else {
        for (int i = 0; i < internal_natoms; ++i) {
            const int ext = map_ext ? perm.to_external(i) : i;
            if (ext < 0 || ext >= static_cast<int>(inverse_mapping_.size())) continue;
            int out_idx = inverse_mapping_[static_cast<size_t>(ext)];
            if (out_idx >= 0 && out_idx < out_natoms) {
                x[out_idx][0] = nm(soa.x[static_cast<size_t>(i)], 0);
                x[out_idx][1] = nm(soa.y[static_cast<size_t>(i)], 1);
                x[out_idx][2] = nm(soa.z[static_cast<size_t>(i)], 2);
            }
        }
    }

    // 3. Define the bounding box (zeroed out for gas-phase/implicit solvent)
    matrix box = { {0.0f, 0.0f, 0.0f}, 
                   {0.0f, 0.0f, 0.0f}, 
                   {0.0f, 0.0f, 0.0f} };

    // 4. Write to XTC (precision_ is passed by value)
    if (write_xtc(xtc_file_, out_natoms, step, static_cast<float>(step), box, x.data(), precision_) != exdrOK) {
        throw std::runtime_error("Failed to write frame to XTC file at step " + std::to_string(step));
    }
    if (xdrfile_flush(xtc_file_) != 0) {
        throw std::runtime_error("Failed to flush XTC file at step " + std::to_string(step));
    }
    ++n_frames_written_;
    (void)integrator;
}

} // namespace mcpu