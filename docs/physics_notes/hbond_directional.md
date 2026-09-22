# Directional hydrogen-bond kernel (KERNEL-4)

**Python API:** :class:`~pymcpu.HBondPotential` (see :doc:`/api/forces`).

Temperature in the Metropolis criterion is a **dimensionless**
reduced parameter (typical 0.3–0.6); H-bond energies are
**unitless** table lookups.

## Overview

Knowledge-based backbone hydrogen bonds use a **7-dimensional** geometric descriptor per donor–acceptor pair. Energy is table lookup multiplied by sequence-dependent weights, with long-range `beta_favor` and optional H–O distance penalty.

## Atom roles

For donor residue `i` and acceptor residue `j`:

| Role | Atoms |
|------|--------|
| Donor N | `N(i)` |
| Donor CA | `CA(i)` |
| Donor C | `C(i)` |
| C_prev | `C(i-1)` — previous residue carbon for H estimation and φ |
| Acceptor O | `O(j)` |
| Acceptor N, CA, C | `N(j)`, `CA(j)`, `C(j)` |
| Neighbors | `CA(i±1)`, `N(j+1)`, etc. for CA–CA filters and antiparallel angles |

**H is not stored.** It is estimated inline from backbone atoms (legacy `hbonds.h:395–401`):

```
vec_NCA = normalize(CA_don - N_don)
vec_NC  = normalize(C_prev - N_don)
h_dir   = -normalize(vec_NCA + vec_NC)
H_pos   = N_don + 1.0 Å × h_dir
```

## 7D descriptor and binning

Table shape: `hbond_E[3][9][9][9][9][9][9]` (flat 1,594,323 entries). Dimension order:

`[helix_sheet][pCA_d][bCA_d][pCA_a][bCA_a][PH][bH]`

| Index | Meaning |
|-------|---------|
| `helix_sheet` (jj1) | 0 = i−4 helix pair; 1 = parallel β; 2 = antiparallel β |
| jj2–jj5 | Plane / bisector angles between donor and acceptor peptide planes (20° bins, 9 bins each) |
| jj6–jj7 | PH and bH plane/bisector angles (20° bins) |

Binning uses **`safe_bin(angle_deg, 20.0f, 9)`** from `torsion_bin.hpp` (truncation + clamp). No chi-style epsilon, `fmod`, or half-bin offset.

**Helix/sheet classification (jj1):**

- `|res_don − res_acc| == 4` → jj1 = 0
- Else: angle between `CA(i−1)→CA(i+1)` and `CA(j+1)→CA(j−1)`; if `< 90°` → 1 (parallel), else 2 (antiparallel)

## Energy formula

Per pair (after all filters):

```
raw = seq_hb[hs][aa_don][aa_acc] × hbond_E[hs][jj2..jj7]

if |res_don − res_acc| > 4:
    raw ×= kBetaFavor    // 3.0f

if d_HO² > kHbInnerSq:
    raw ×= kHbPenalty    // 1.0f at default (no effect)

E_pair = raw / 1000.0f × kRdthreeCon    // kRdthreeCon = 2.0f
```

Weighted MC delta:

```
delta_hbond = kHbondWeight × Σ_pairs (E_new − E_old)    // kHbondWeight = 1.35f
```

Constants from legacy `define.h` / `backbone.c`.

## Filters (application order)

1. **Terminal residue:** donor or acceptor at chain terminus → skip
2. **Sequence separation:** `|res_don − res_acc| < 4` → skip
3. **H–O distance:** estimated H to acceptor O; `d_HO² > kHbCutoffSq` (6.25 Å²) → skip
4. **CA–CA precheck:** seq_sep-dependent minima over four CA–CA pairs (5.8²/5.5² for sep=4; 6.0²/5.4² for sep>4)
5. **Helix filter (sep>4):** `secstr[donor]=='H'` or `secstr[acceptor]=='H'` → skip
6. **Ramachandran filter:** seq_sep-dependent φ/ψ and secondary-structure checks (legacy `hbonds.h:462–495`)

For benchmark 1uao, `secstr` is all `'C'`; filters 5–6 never trigger.

## SWE-1 Option A — delta over affected pairs only

Legacy `FoldHydrogenBonds()` recomputes all pairs. pyMCPU computes:

```
delta = kHbondWeight × Σ_{affected (don,acc)} [E_new(don,acc) − E_old(don,acc)]
```

Unmoved pairs cancel. Complexity is **O(moved × shell_neighbors)** via `grid_h` / `grid_o` neighbor gating, not O(N²).

Pair deduplication uses `hbond_pair_stamp` in `MCWorkspace` (epoch reset). No read/write of `hbond_frame_cache`.

Sidechain-only moves return **0** (backbone H-bonds unchanged).

## Regression tolerance

CASE-2 uses **rel_error < 1e−4** (not 1e−5). Trigonometric geometry (cross products, normalizations) with mixed float/double accumulation is a known precision source versus legacy.

## Constants summary

| Name | Value |
|------|-------|
| `kHbondWeight` | 1.35f |
| `kRdthreeCon` | 2.0f |
| `kBetaFavor` | 3.0f |
| `kHbInnerSq` / `kHbCutoffSq` | 6.25f (2.5 Å)² |
| `kHbPenalty` | 1.0f |
