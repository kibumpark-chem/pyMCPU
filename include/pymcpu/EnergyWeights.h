#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace mcpu {

/// Per-energy-group outer energy weights matching legacy MCPU.
///
/// Legacy references
/// -----------------
/// Total energy (`energy.h` `ResetEnergies`):
///   E = weight_potential * E_pot
///     + weight_clash * nclashes
///     + weight_hbond * E_hbond
///     + TOR_WEIGHT * E_tor
///     + SCT_WEIGHT * E_sct
///     + ARO_WEIGHT * E_aro
///     + E_constraint
///
/// Macros (`define.h`):
///   POTNTL_WEIGHT 0.4   → Mu / contact (E_pot)          → energy group 1
///   HBOND_WEIGHT  1.35  → HBond outer weight            → energy group 4
///   TOR_WEIGHT    1.35  → backbone torsion              → energy group 2
///   SCT_WEIGHT    2.50  → sidechain torsion             → energy group 3
///   ARO_WEIGHT    5.0   → aromatic                      → energy group 5
///   RDTHREE_CON   2.0   → applied inside legacy HBond
///                         (`hbonds.h` HydrogenBonds / FoldHydrogenBonds):
///                         E_hbond = (table_sum / 1000) * RDTHREE_CON
///                         then outer × HBOND_WEIGHT.
///
/// pyMCPU HBondPotential already returns (table_sum / 1000).  To match legacy's
/// final HBond contribution we therefore apply
///   weight_effective[4] = HBOND_WEIGHT * RDTHREE_CON
/// when use_legacy_weights is true.  Clash count weighting is not used;
/// Mu hard-core clashes remain a large sentinel inside MuPotential.
///
/// Energy groups with no legacy outer weight (e.g. QBias group 6) default to 1.
struct EnergyWeights {
    static constexpr int kMaxTrackedGroup = 16;

    /// Legacy compile-time defaults from define.h / backbone.c.
    static constexpr float kLegacyMu      = 0.4f;   // POTNTL_WEIGHT
    static constexpr float kLegacyBbTor   = 1.35f;  // TOR_WEIGHT
    static constexpr float kLegacyScTor   = 2.50f;  // SCT_WEIGHT
    static constexpr float kLegacyHBond   = 1.35f;  // HBOND_WEIGHT
    static constexpr float kLegacyAro     = 5.0f;   // ARO_WEIGHT
    static constexpr float kLegacyRdthree = 2.0f;   // RDTHREE_CON

    bool use_legacy_weights = true;

    /// Outer weights by energy group index (1-based groups used by builders).
    /// Index 0 unused.  Effective HBond weight also folds in RDTHREE when
    /// use_legacy_weights is true (see weight_for_group).
    std::array<float, kMaxTrackedGroup> outer{};

    float hbond_rdthree = kLegacyRdthree;

    EnergyWeights() { set_legacy_defaults(); }

    void set_legacy_defaults() noexcept {
        use_legacy_weights = true;
        outer.fill(1.0f);
        outer[1] = kLegacyMu;
        outer[2] = kLegacyBbTor;
        outer[3] = kLegacyScTor;
        outer[4] = kLegacyHBond;
        outer[5] = kLegacyAro;
        // group 6+ remain 1.0
        hbond_rdthree = kLegacyRdthree;
    }

    void set_unweighted() noexcept {
        use_legacy_weights = false;
        outer.fill(1.0f);
        hbond_rdthree = 1.0f;
    }

    void set_use_legacy_weights(bool on) noexcept {
        if (on) {
            set_legacy_defaults();
        } else {
            set_unweighted();
        }
    }

    void set_energy_weight(int group_id, float w) {
        if (group_id < 0 || group_id >= kMaxTrackedGroup) {
            throw std::out_of_range("EnergyWeights::set_energy_weight: group_id out of range");
        }
        outer[static_cast<size_t>(group_id)] = w;
    }

    [[nodiscard]] float outer_weight(int group_id) const noexcept {
        if (group_id < 0 || group_id >= kMaxTrackedGroup) return 1.0f;
        return outer[static_cast<size_t>(group_id)];
    }

    /// Multiplier applied to raw potential energy / ΔE for this group.
    /// Group 4 (HBond): outer × hbond_rdthree (legacy RDTHREE_CON when defaults used).
    [[nodiscard]] float weight_for_group(int group_id) const noexcept {
        const float w = outer_weight(group_id);
        if (group_id == 4) {
            return w * hbond_rdthree;
        }
        return w;
    }

    [[nodiscard]] std::unordered_map<int, float> as_map() const {
        std::unordered_map<int, float> m;
        for (int g = 1; g < kMaxTrackedGroup; ++g) {
            m[g] = weight_for_group(g);
        }
        return m;
    }
};

struct EnergyBreakdown {
    /// Raw (unweighted) per-group energies from Potential::calculateEnergy.
    std::unordered_map<int, float> raw_by_group;
    /// weight_for_group(g) * raw for each group.
    std::unordered_map<int, float> weighted_by_group;
    float raw_total = 0.0f;
    float weighted_total = 0.0f;
};

}  // namespace mcpu
