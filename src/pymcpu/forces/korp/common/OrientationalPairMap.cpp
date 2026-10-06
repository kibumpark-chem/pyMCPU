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

    // Bin-edge directions for the azimuth and dihedral bins (see angle_bin).
    // Edge i of a bin of width w sits at angle i * w, i.e. at atan2-angle
    // i * w - pi. A bin wider than pi/2 that is not the whole circle could put
    // a tested edge more than pi from the vector, where the cross product's
    // sign no longer says which side it is on; no released map has one.
    const auto add_edges = [](std::vector<Edge>& out, float width, int n) {
        for (int i = 0; i <= n; ++i) {
            const double e = static_cast<double>(i) * static_cast<double>(width) - M_PI;
            out.push_back(Edge{std::cos(e), std::sin(e)});
        }
    };
    const auto check_width = [&](float width, int n) {
        need(n > 0 && width > 0.f, "empty angular bin");
        need(n == 1 || static_cast<double>(width) <= 0.5 * M_PI + 1e-6,
             "angular bins wider than pi/2 must cover the whole circle");
    };
    psi_edge_offset_.resize(rings);
    ring_inv_dpsi_.resize(rings);
    for (std::size_t k = 0; k < rings; ++k) {
        check_width(ring_dpsi_[k], ring_ncells_[k]);
        psi_edge_offset_[k] = static_cast<int>(psi_edge_.size());
        ring_inv_dpsi_[k] = 1.f / ring_dpsi_[k];
        add_edges(psi_edge_, ring_dpsi_[k], ring_ncells_[k]);
    }
    chi_edge_offset_.resize(static_cast<std::size_t>(nr_));
    shell_inv_dchi_.resize(static_cast<std::size_t>(nr_));
    for (int s = 0; s < nr_; ++s) {
        const std::size_t u = static_cast<std::size_t>(s);
        check_width(shell_dchi_[u], shell_nchi_[u]);
        chi_edge_offset_[u] = static_cast<int>(chi_edge_.size());
        shell_inv_dchi_[u] = 1.f / shell_dchi_[u];
        add_edges(chi_edge_, shell_dchi_[u], shell_nchi_[u]);
    }

    // Padded boundaries for the branch-free shell and ring counts, used when
    // the map fits them and its boundaries are ordered (then the count and
    // upstream's loop agree); otherwise bins() runs upstream's loops.
    bool fits = nr_ < kMaxShells;
    for (int s = 0; fits && s < nr_; ++s) {
        fits = br_[static_cast<std::size_t>(s)] < br_[static_cast<std::size_t>(s) + 1];
        const int lo = ring_offset_[static_cast<std::size_t>(s)];
        const int hi = ring_offset_[static_cast<std::size_t>(s) + 1];
        fits = fits && hi > lo && hi - lo <= kMaxRings + 1;
        for (int i = lo; fits && i + 1 < hi; ++i) {
            fits = ring_cos_theta_[static_cast<std::size_t>(i)]
                   >= ring_cos_theta_[static_cast<std::size_t>(i) + 1];
        }
    }
    padded_ = fits;
    if (padded_) {
        for (int s = 0; s < kMaxShells; ++s) {
            br_pad_[s] = s <= nr_ ? static_cast<double>(br_[static_cast<std::size_t>(s)])
                                  : HUGE_VAL;
        }
        ring_cos_pad_.assign(static_cast<std::size_t>(nr_) * kMaxRings, -HUGE_VAL);
        for (int s = 0; s < nr_; ++s) {
            const int lo = ring_offset_[static_cast<std::size_t>(s)];
            const int nring = ring_offset_[static_cast<std::size_t>(s) + 1] - lo;
            for (int i = 0; i + 1 < nring; ++i) {
                ring_cos_pad_[static_cast<std::size_t>(s) * kMaxRings
                              + static_cast<std::size_t>(i)] =
                    ring_cos_theta_[static_cast<std::size_t>(lo + i)];
            }
        }
    }
}

} // namespace mcpu::forces
