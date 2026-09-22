#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcpu {

enum class AtomReorderMode : int {
    Off = 0,
    InitOnly = 1
};

inline AtomReorderMode atom_reorder_mode_from_string(const std::string& s) {
    if (s == "off" || s.empty()) return AtomReorderMode::Off;
    if (s == "init_only") return AtomReorderMode::InitOnly;
    throw std::invalid_argument(
        "atom_reorder_mode must be \"off\" or \"init_only\", got: " + s);
}

inline const char* atom_reorder_mode_to_string(AtomReorderMode m) noexcept {
    switch (m) {
        case AtomReorderMode::InitOnly: return "init_only";
        case AtomReorderMode::Off:
        default: return "off";
    }
}

/// Maps between external (build/PDB) atom ids and internal (hot-path SoA) ids.
///
/// External id = atom index at System build time (BB|O|SC layout order).
/// Internal id = storage index after optional init-only locality permutation.
struct AtomPermutation {
    std::vector<int> int_to_ext;  // size N: internal i → external e
    std::vector<int> ext_to_int;  // size N: external e → internal i

    [[nodiscard]] int n_atoms() const noexcept {
        return static_cast<int>(int_to_ext.size());
    }

    [[nodiscard]] bool enabled() const noexcept {
        return !int_to_ext.empty() && !is_identity();
    }

    [[nodiscard]] bool is_identity() const noexcept {
        const int n = n_atoms();
        if (n == 0) return true;
        for (int i = 0; i < n; ++i) {
            if (int_to_ext[static_cast<size_t>(i)] != i) return false;
        }
        return true;
    }

    [[nodiscard]] int to_internal(int ext_id) const {
        return ext_to_int.at(static_cast<size_t>(ext_id));
    }
    [[nodiscard]] int to_external(int int_id) const {
        return int_to_ext.at(static_cast<size_t>(int_id));
    }

    static AtomPermutation identity(int n) {
        AtomPermutation p;
        p.int_to_ext.resize(static_cast<size_t>(n));
        p.ext_to_int.resize(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            p.int_to_ext[static_cast<size_t>(i)] = i;
            p.ext_to_int[static_cast<size_t>(i)] = i;
        }
        return p;
    }

    /// Build inverse maps; throws if not a permutation of [0, N).
    void finalize_from_int_to_ext() {
        const int n = n_atoms();
        ext_to_int.assign(static_cast<size_t>(n), -1);
        for (int i = 0; i < n; ++i) {
            const int e = int_to_ext[static_cast<size_t>(i)];
            if (e < 0 || e >= n) {
                throw std::runtime_error("AtomPermutation: int_to_ext out of range");
            }
            if (ext_to_int[static_cast<size_t>(e)] != -1) {
                throw std::runtime_error("AtomPermutation: duplicate external id");
            }
            ext_to_int[static_cast<size_t>(e)] = i;
        }
    }

    void validate_inverses() const {
        const int n = n_atoms();
        if (static_cast<int>(ext_to_int.size()) != n) {
            throw std::runtime_error("AtomPermutation: size mismatch");
        }
        for (int i = 0; i < n; ++i) {
            const int e = int_to_ext[static_cast<size_t>(i)];
            if (e < 0 || e >= n) throw std::runtime_error("AtomPermutation: bad int_to_ext");
            if (ext_to_int[static_cast<size_t>(e)] != i) {
                throw std::runtime_error("AtomPermutation: inverses inconsistent");
            }
        }
        for (int e = 0; e < n; ++e) {
            const int i = ext_to_int[static_cast<size_t>(e)];
            if (i < 0 || i >= n) throw std::runtime_error("AtomPermutation: bad ext_to_int");
            if (int_to_ext[static_cast<size_t>(i)] != e) {
                throw std::runtime_error("AtomPermutation: inverses inconsistent");
            }
        }
    }

    /// FNV-1a 64-bit over int_to_ext (stable checksum for perf packets).
    [[nodiscard]] std::uint64_t checksum() const noexcept {
        std::uint64_t h = 14695981039346656037ull;
        for (int v : int_to_ext) {
            const auto u = static_cast<std::uint32_t>(v);
            h ^= static_cast<std::uint64_t>(u);
            h *= 1099511628211ull;
        }
        return h;
    }
};

}  // namespace mcpu
