#include "pymcpu/forces/korp/common/OrientationalPairMap.h"

#include <cmath>

namespace mcpu::forces {

OrientationalPairMap::OrientationalPairMap(
    float cutoff, float min_r, int nslices,
    std::vector<float> br,
    std::vector<int> shell_ncells,
    std::vector<int> shell_nchi,
    std::vector<float> shell_dchi,
    std::vector<std::int64_t> shell_offset,
    std::vector<int> ring_offset,
    std::vector<float> ring_theta,
    std::vector<float> ring_dpsi,
    std::vector<int> ring_ncells,
    std::vector<int> ring_first_cell,
    std::vector<std::int8_t> smapping,
    std::vector<float> fmapping,
    const float* table, std::size_t table_size)
    : cutoff_(cutoff), cutoff_sq_(cutoff * cutoff), min_r_(min_r),
      nr_(static_cast<int>(shell_ncells.size())), nslices_(nslices),
      br_(std::move(br)),
      shell_ncells_(std::move(shell_ncells)), shell_nchi_(std::move(shell_nchi)),
      shell_dchi_(std::move(shell_dchi)), shell_offset_(std::move(shell_offset)),
      ring_offset_(std::move(ring_offset)),
      ring_theta_(std::move(ring_theta)), ring_dpsi_(std::move(ring_dpsi)),
      ring_ncells_(std::move(ring_ncells)), ring_first_cell_(std::move(ring_first_cell)),
      smapping_(std::move(smapping)), fmapping_(std::move(fmapping)),
      table_(table), table_size_(table_size)
{
    // Shape checks, not a re-parse. The Python loader already validated the
    // file itself; what can still go wrong here is the handover, and an
    // inconsistent one would read neighbouring bins rather than fail.
    const auto need = [](bool ok, const char* what) {
        if (!ok) throw std::invalid_argument(std::string("OrientationalPairMap: ") + what);
    };
    need(nr_ > 0, "no radial shells");
    need(br_.size() == static_cast<std::size_t>(nr_) + 1, "br must have nr+1 entries");
    need(shell_nchi_.size() == static_cast<std::size_t>(nr_), "shell_nchi size != nr");
    need(shell_dchi_.size() == static_cast<std::size_t>(nr_), "shell_dchi size != nr");
    need(shell_offset_.size() == static_cast<std::size_t>(nr_) + 1, "shell_offset size != nr+1");
    need(ring_offset_.size() == static_cast<std::size_t>(nr_) + 1, "ring_offset size != nr+1");
    need(smapping_.size() == kSeparationTableSize, "smapping must have 10 entries");
    need(fmapping_.size() == kSeparationTableSize, "fmapping must have 10 entries");
    need(nslices_ > 0, "no slices");
    need(table != nullptr, "null table");

    const std::size_t rings = static_cast<std::size_t>(ring_offset_.back());
    need(ring_theta_.size() == rings, "ring_theta size != total rings");
    need(ring_dpsi_.size() == rings, "ring_dpsi size != total rings");
    need(ring_ncells_.size() == rings, "ring_ncells size != total rings");
    need(ring_first_cell_.size() == rings, "ring_first_cell size != total rings");

    ring_cos_theta_.resize(ring_theta_.size());
    for (std::size_t i = 0; i < ring_theta_.size(); ++i) {
        ring_cos_theta_[i] = std::cos(static_cast<double>(ring_theta_[i]));
    }

    block_stride_ = shell_offset_.back();
    need(block_stride_ > 0, "empty block stride");

    const std::size_t expect =
        static_cast<std::size_t>(nslices_) * kNumTypes * kNumTypes
        * static_cast<std::size_t>(block_stride_);
    need(table_size_ == expect,
         "table size does not match nslices * 20 * 20 * block_stride");

    for (int s = 0; s < nr_; ++s) {
        const std::size_t u = static_cast<std::size_t>(s);
        need(shell_ncells_[u] > 0 && shell_nchi_[u] > 0, "empty shell");
        need(shell_dchi_[u] > 0.f, "non-positive dchi");
    }
    for (std::int8_t s : smapping_) {
        need(s < static_cast<std::int8_t>(nslices_), "smapping refers to a missing slice");
    }
}

} // namespace mcpu::forces
