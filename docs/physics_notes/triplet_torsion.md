# Backbone triplet torsion (KERNEL-2)

**Python API:** :class:`~pymcpu.TripletPotential` and
:class:`~pymcpu.SidechainTripletPotential` (see :doc:`/api/forces`).

Temperature in the Metropolis criterion is a **dimensionless**
reduced parameter (typical 0.3–0.6); torsion energies are
**unitless** (`table / 1000` × weight).

## Four angles

Legacy `check_bb()` (`loop.h:914–955`) defines four angles per triplet index `i` covering residues `(i, i+1, i+2)`. All values are stored in **degrees**.

| pyMCPU | Legacy | Definition |
|--------|--------|------------|
| `pCA[i]` | `a_PCA[i]` | Angle between CA–N/O plane normals at residues `i` and `i+2` |
| `bCA[i]` | `a_bCA[i]` | Angle between bisectors of those CA–N/O vector pairs |
| `phi[i]` | `phim[i]` | Dihedral C(`i`)–N(`i+1`)–CA(`i+1`)–C(`i+1`), then **+180°** |
| `psi[i]` | `psim[i]` | Dihedral N(`i+1`)–CA(`i+1`)–C(`i+1`)–N(`i+2`), then **+180°** |

`TorsionData` stores `phi[]` and `psi[]` **with the +180° shift already applied** (range ~0–360°). The kernel must **not** add 180° again.

## Bin indexing

Legacy (`energy.h:74–77`):

```c
x = (int)(a_PCA[i] / 30.0);
y = (int)(a_bCA[i] / 30.0);
z = (int)(phim[i] / 60.0);
w = (int)(psim[i] / 60.0);
```

C++ equivalent: `static_cast<int>(angle / bin_width)` — **truncation toward zero**, not `std::floor()`.

| Angle | Bin width | Bins |
|-------|-----------|------|
| pCA, bCA | 30° | 6 |
| phi, psi | 60° | 6 |

Table shape: `torsion_E[triplet_index][pCA_bin][bCA_bin][phi_bin][psi_bin]`.

## Triplet index

`triplet_index = i` is the **first residue** of the triplet `(i, i+1, i+2)`.

Valid range: `0 <= i <= num_residues - 3`.

A disturbed backbone residue `r` affects triplets `{r-2, r-1, r}` (clamped to valid range). Duplicate triplets from multiple distorted residues are evaluated **once** (epoch stamp deduplication).

## Energy scale and weight

```text
E_triplet = table[triplet][bins...] / 1000.0
delta_backbone = TOR_WEIGHT * sum(new_E - old_E)
```

`TOR_WEIGHT = 1.35` (`define.h:26`). Default unset table cells = `1000`.

## Bin index safety

The legacy code does not clamp bin indices. An angle outside the expected range `[0, num_bins × bin_width)` accesses memory beyond the table bounds (undefined behavior). The new implementation clamps to `[0, num_bins-1]` which is safer and produces the nearest-boundary table value instead of a crash or corrupted energy. The clamp only changes behaviour for out-of-range angles, which a well-folded structure does not produce, so it does not perturb parity against the legacy energies.

## Triplet deduplication

When `bb_distorted[]` contains residues `5` and `6`, both disturb triplet `4` (= residues 4,5,6). Without deduplication, triplet 4 would be counted twice and `delta_E` would be wrong. The kernel uses an epoch-based `triplet_stamp` (LOCK-5) to visit each triplet at most once per proposal.

## Sidechain moves

Legacy computes `dE_tor` only when `sidechain_step == 0` (`move.h:131–132`). Sidechain moves leave `delta_backbone = 0`.

---

## Sidechain Triplet (KERNEL-3)

### Four χ angles per triplet middle residue

Legacy `sctenergy()` (`energy.h:85–125`) evaluates one sidechain triplet per index `i` covering residues `(i, i+1, i+2)`. The **middle** residue `i+1` supplies χ torsion angles. Table shape:

```text
sct_E[triplet_index][chi1_bin][chi2_bin][chi3_bin][chi4_bin]
```

12 bins per χ dimension (0–11), 30° per bin.

### Chi angle normalization pipeline (exact legacy)

All normalization happens at bin time inside `chi_to_bin()` — χ values in `TorsionData` are stored in **radians** with no offset applied at storage time. Absent χ angles are stored as `0.0` and treated as bin 0 for unused dimensions.

```text
1. ang = chi_radians * RAD2DEG
2. Wrap to (-180°, 180°]
3. if (ang < 0) ang += 180° + 0.00001f
   else         ang += 180° - 0.00001f
4. ang += 15°   (half-bin offset)
5. ang = fmod(ang, 360°)
6. bin = safe_bin(ang, 30°, 12)
```

The **epsilon values** (`±0.00001f`) are intentional boundary guards. Angles at exactly 0° and ±180° must fall into the same bins as legacy; without the epsilons, bin indices differ at boundary cases and regression tests fail.

### Absent χ handling (two mechanisms)

**MECHANISM-1 (residue skip):** If `ntorsions_per_residue[r] == 0` (GLY, ALA), skip the entire triplet row — return 0 contribution. No table lookup.

**MECHANISM-2 (unused dimensions → bin 0):** For χ dimensions `j >= ntorsions`, use bin 0 (not a sentinel). The table row is still evaluated with bin 0 for unused dimensions. Applies to residues with 1, 2, or 3 χ angles.

### Triplet index mapping

A sidechain move on residue `r` affects **only** triplet `r - 1` (middle residue = `r`):

```text
triplet_index = r - 1
Valid when: 1 <= r <= num_residues - 2
```

If `r == 0` or `r == num_residues - 1`, no valid triplet exists → return 0. Unlike backbone, there is no deduplication — exactly one triplet per `sc_distorted` residue.

### Energy scale and weight

```text
E_triplet = sct_E[triplet][bins...] / 1000.0
delta_sidechain = SCT_WEIGHT * sum(new_E - old_E)
```

`SCT_WEIGHT = 2.50` (`define.h:27`). Default unset table cells = `1000`.

### Bin index safety

As with backbone, legacy does not clamp χ bin indices. The new implementation uses `safe_bin()` after the `fmod` step to clamp to `[0, 11]`.
