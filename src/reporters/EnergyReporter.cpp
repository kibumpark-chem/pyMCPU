#include "pymcpu/reporters/EnergyReporter.h"
#include "pymcpu/Integrator.h"
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mcpu {

namespace {
// The first line of an existing, non-empty file, without its line ending, or
// "" when the file is missing or empty. A first line with no newline at all
// is returned with a marker so it can never equal a real header.
std::string read_existing_header(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) return {};
    std::string line;
    if (!std::getline(in, line)) return {};
    const bool had_newline = !in.eof();
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!had_newline) return line + " <no line ending>";
    return line;
}
}  // namespace

EnergyReporter::EnergyReporter(const std::string& energy_filename, int report_interval,
                               bool append)
    : Reporter(report_interval), filename_(energy_filename) {
    // Read before opening: opening with ios::app creates the file.
    if (append) existing_header_ = read_existing_header(energy_filename);

    energy_file_.open(energy_filename, append ? (std::ios::out | std::ios::app)
                                              : (std::ios::out | std::ios::trunc));
    if (!energy_file_.is_open()) {
        throw std::runtime_error("Failed to open energy file for writing: " + energy_filename);
    }
}

EnergyReporter::~EnergyReporter() {
    if (energy_file_.is_open()) {
        energy_file_.close();
    }
}

void EnergyReporter::begin_run(const Context& context, const MCIntegrator& integrator) {
    ensure_header(context, integrator);
}

void EnergyReporter::ensure_header(const Context& context, const MCIntegrator& integrator) {
    const auto moves = integrator.move_counts();
    if (header_ready_) {
        check_terms_unchanged(context);
        for (std::size_t i = 0; i < moves.size(); ++i) {
            const bool has_column = std::find(move_columns_.begin(), move_columns_.end(), i) !=
                                    move_columns_.end();
            if (moves[i].in_use && !has_column) {
                throw std::runtime_error(
                    "energy file " + filename_ + " has no columns for the '" +
                    std::string(moves[i].name) + "' move, which the current move "
                    "weights now use. Write this run to a new energy file.");
            }
        }
        return;
    }

    terms_ = context.getSystem().energyTerms();
    move_columns_.clear();
    std::ostringstream header;
    header << "step,total";
    for (const auto& term : terms_) header << "," << term.second;
    for (std::size_t i = 0; i < moves.size(); ++i) {
        if (!moves[i].in_use) continue;
        move_columns_.push_back(i);
        header << "," << moves[i].name << "_accepted," << moves[i].name << "_attempted";
    }
    header << ",walker_id";
    const std::string expected = header.str();

    if (!existing_header_.empty()) {
        if (existing_header_ != expected) {
            std::ostringstream msg;
            // Before per-term columns, every energy file started with this
            // fixed header. No setting can make a run match it.
            if (existing_header_.rfind("Step,Total,", 0) == 0) {
                msg << "energy file " << filename_ << " was written by an earlier "
                    << "pyMCPU, whose fixed columns (Step,Total,Mu,...,WalkerId) differ "
                    << "from this version's, so this run cannot append to it. Resume "
                    << "into a new file: change the output prefix or directory, or move "
                    << "this file aside.";
            } else {
                const auto w = integrator.move_weights();
                msg << "energy file " << filename_ << " already has a different header, so "
                    << "appending would misalign its columns.\n"
                    << "  in the file: " << existing_header_ << "\n"
                    << "  this run:    " << expected << "\n"
                    << "  (move weights pivot/kic/sidechain = " << w[0] << "/" << w[1] << "/"
                    << w[2] << ", sidechain mode " << integrator.sidechain_move_mode() << ")\n"
                    << "Resume with the same force field and move settings, or write to a "
                    << "new file.";
            }
            throw std::runtime_error(msg.str());
        }
    } else {
        energy_file_ << expected << "\n";
        energy_file_.flush();
        if (energy_file_.fail()) {
            throw std::runtime_error("Failed to write header to energy file: " + filename_);
        }
    }
    header_ready_ = true;
}

void EnergyReporter::check_terms_unchanged(const Context& context) const {
    if (context.getSystem().energyTerms() != terms_) {
        throw std::runtime_error(
            "energy terms changed after the header of energy file " + filename_ +
            " was written (a potential was added or renamed). Write to a new file.");
    }
}

void EnergyReporter::report(int step, const Context& context, const MCIntegrator& integrator) {
    // run() calls begin_run first; this also covers a report() made without it.
    if (!header_ready_) ensure_header(context, integrator);
    check_terms_unchanged(context);

    const EnergyBreakdown bd = context.energy_breakdown();
    const auto moves = integrator.move_counts();

    // Format one complete line in memory, then write once. This keeps each
    // record contiguous (important on NFS when many ranks write concurrently
    // to different files, and avoids torn multi-operator << sequences).
    std::ostringstream line;
    line.setf(std::ios::fixed, std::ios::floatfield);
    // total is the cached weighted total, including the native-contacts bias.
    line << std::setprecision(8) << step << ","
         << static_cast<double>(context.getState().getEnergy());
    for (const auto& term : terms_) {
        auto it = bd.weighted_by_group.find(term.first);
        line << "," << (it != bd.weighted_by_group.end() ? static_cast<double>(it->second) : 0.0);
    }
    for (std::size_t i : move_columns_) {
        line << "," << moves[i].accepted << "," << moves[i].attempted;
    }
    line << "," << walker_id_ << "\n";
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
