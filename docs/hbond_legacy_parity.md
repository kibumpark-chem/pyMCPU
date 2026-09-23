# Hydrogen-bond potential: legacy MCPU parity

> **On the legacy inputs.** The MCPU parameter tables and run logs cited
> below are *not* distributed with pyMCPU. They are inputs to a C binary
> that is not in this repository and is not rebuildable here, so shipping
> them would imply a reproducibility path that does not exist. Every number
> quoted below is recorded inline, along with the arithmetic that reconciles
> it against the legacy run's own printed total — which is what makes the
> comparison checkable without the files. The five ladder tests in
> `tests/legacy_parity/test_hbond_ablation_ladder.py` skip in their absence;
> the assertions about pyMCPU's own H-bond energy do not depend on them.

pyMCPU's `HBondPotential` (`src/pymcpu/forces/mcpu/common/HydrogenBondPotential.cpp`,
`include/pymcpu/utils/hydrogen_bond_utils.h`) is a from-scratch C++ rewrite of legacy
MCPU's `HydrogenBonds()`/`FoldHydrogenBonds()`
(`dbfold_actin/MCPU/src_mpi_umbrella/hbonds.h`). This document records the exact
atom/gate/formula mapping between the two, and the ablation ladder used to verify
the rewrite bug-for-bug against the legacy reference on
`examples/actin/input_pdb/acta.pdb`.

An independent NumPy oracle transliterating `hbonds.h` directly from the legacy
source (not from pyMCPU's compiled `.bin` tables) lives at
`tests/legacy_parity/helpers/legacy_hbond_oracle.py`, with inline citations of the
exact legacy line ranges each function mirrors. `tests/legacy_parity/test_hbond_ablation_ladder.py`
and `tests/legacy_parity/test_hbond_engine_parity.py` pin every rung of the ladder
below to an exact numeric target, via `tests/legacy_parity/framework.py`'s
`LegacyReference`/`assert_legacy_parity`.

## Ablation ladder

Weighted group-4 (hydrogen_bond) energy on `acta.pdb`, `use_legacy_weights=True`
(`HBOND_WEIGHT = 1.35`):

| Stage | Change | Weighted E |
|---|---|---|
| A (pre-fix) | — | -216.4995 |
| 1 | fix `ang_CACA` (units + transposed operands) | -224.0406 |
| 2 | + Ramachandran gates + correct `'H'`-gate placement | -217.1853 |
| 3 | + sequence-dependent scaling (`seq_hb`) + `beta_favor` | **-172.9607 = legacy parity** |
| 4 | + real secondary structure via DSSP (opt-in) | -172.20 (unchanged at default) |

Legacy reference: a completed legacy MCPU run, `acta_T_0.600.log`, STEP 0,
`hbond = -128.12` (raw) → `-128.12 * 1.35 = -172.962` (weighted).

## Donor / acceptor atom mapping

Legacy indexes hydrogen-bond partners by a flat atom-offset table (decoded from
`hydrogen_jPL3h.data`'s header, `hbonds.h` ~180-230). pyMCPU loads the same
eight atoms per side directly by residue-relative name lookup
(`HydrogenBondUtils::construct_donor`/`construct_acceptor`):

| Legacy offset | pyMCPU field | Meaning |
|---|---|---|
| donor2 | `donor.CA` | donor residue Cα |
| donor5 | `donor.prev_CA` | donor's preceding residue Cα |
| donor7 | `donor.next_CA` | donor's following residue Cα |
| donor.N/.C | `donor.N`/`donor.C` | donor backbone N, C |
| donor.prev_N/.prev_C | `donor.prev_N`/`donor.prev_C` | preceding residue N, C |
| donor.H | `donor.H` | amide H (explicit or virtual, `compute_virtual_amide_H`) |
| acceptor3 | `acceptor.CA` | acceptor residue Cα |
| acceptor6 | `acceptor.next_CA` | acceptor's following residue Cα |
| acceptor8 | `acceptor.prev_CA` | acceptor's preceding residue Cα |
| acceptor.N/.C/.O | `acceptor.N`/`.C`/`.O` | acceptor backbone N, C, O |
| acceptor.next_N/.next_C | `acceptor.next_N`/`.next_C` | following residue N, C |

## Gate order (`HydrogenBondUtils::is_hydrogen_bond`)

Applied in this order; any failure contributes exactly 0 (same as legacy's
`NO_HBOND`), not a soft penalty:

1. **H···O distance cutoff** (`is_close_enough_for_hbond`): `HBOND_CUTOFF = 2.5 Å`.
2. **CA-CA orientation prefilter** (`passes_ca_geometry_gate`, `hbonds.h` ~409-423):
   four squared CA-CA distances (`donor.{prev_CA,CA}` × `acceptor.{CA,next_CA}`),
   reduced to `min1`/`min2`/`min3`, checked against helix (`res_idx_diff == 4`,
   cutoffs `5.8`/`5.5 Å`) or sheet (`res_idx_diff > 4`, cutoffs `6.0`/`5.4 Å`)
   thresholds.
   - **`'H'` secondary-structure gate**: for `res_idx_diff > 4` only — legacy
     `hbonds.h` ~456-459 applies this exclusively in the long-range branch. A
     prior version of this code applied it unconditionally, which is invisible
     while SS is hardcoded to `'C'` but would silently reject ~80% of helical
     (`res_idx_diff == 4`) H-bonds the moment real SS data is enabled.
3. **Ramachandran hard-reject gates** (`passes_ramachandran_gate`,
   `hbonds.h` ~424-460): donor/acceptor φ/ψ, computed directly from the same
   struct coordinates (not the `State::backbone_torsions` cache, which doesn't
   cover chain termini):
   - `Dphi = dihedral(prev_C, N, CA, C)`, `Dpsi = dihedral(prev_N, prev_CA, prev_C, N)`
   - `Aphi = dihedral(C, next_N, next_CA, next_C)`, `Apsi = dihedral(N, CA, C, next_N)`
   - all four converted to degrees and shifted `+180` (legacy convention).
   - `res_idx_diff == 4`: reject if secstr ∈ {`'E'`, `'L'`} at donor **or**
     acceptor, **and** (`Dphi < 180 and Dpsi < 180`) **or** (`Aphi < 180 and Apsi < 180`).
   - `res_idx_diff > 4`: reject if `Dphi > 150`, or `30 < Dpsi < 210`, or
     `Aphi > 150`, or `30 < Apsi < 210`, or secstr ∈ {`'L'`} at donor or acceptor.

## `ang_CACA` (helix/sheet type index, `hydrogen_bond_indices`)

Legacy: `ang_CACA = Angle(donor7 - donor5, acceptor6 - acceptor8)` — the angle
between each side's **own** chain axis (`next_CA - prev_CA`), not a cross-chain
comparison. Compared in **degrees** against a **90°** threshold.

A prior version of this code computed `angle(donor.prev_CA - acceptor.prev_CA,
donor.next_CA - acceptor.next_CA)` (transposed, cross-chain) against a `0.5`
threshold in **radians** — doubly wrong. Fixed to
`calculate_a_CACA(donor.next_CA, acceptor.next_CA, donor.prev_CA, acceptor.prev_CA)`
(intra-chain axis vectors) compared against `HBOND_CACA_HELIX_SHEET_THRESHOLD = π/2`.

`indices[0]` (the table's first, `helix_sheet` dimension): `0` when
`res_idx_diff == 4`; else `1` if `ang_CACA < π/2` else `2`.

## Sequence-dependent scaling + `beta_favor`

Legacy (`hbonds.h`, `HydrogenBonds()`/`FoldHydrogenBonds()`):

```c
if (abs(i - j) > 4)
    e += beta_favor * seq_hb[helix_sheet][GetAminoNumber(res_i)][GetAminoNumber(res_j)]
                     * hbond_E[hbond_index];
else
    e += seq_hb[helix_sheet][GetAminoNumber(res_i)][GetAminoNumber(res_j)]
                     * hbond_E[hbond_index];
```

(`HB_PENALTY` — legacy's other conditional multiplier — is confirmed dead code:
`HB_INNER == HB_CUTOFF`, so its branch is unreachable given legacy's own
constants; not ported.)

pyMCPU (`HBondPotential::evaluate_directional`):

```cpp
float e = get(indices[0], indices[1], ..., indices[6]);
e *= seq_dep_factor(indices[0], sys.amino_index(r_don), sys.amino_index(r_acc));
if (indices[0] != 0) { e *= BETA_FAVOR; }  // BETA_FAVOR = 3.0f; indices[0]==0 only for res_idx_diff==4
```

`seq_dep_factor` indexes a flat `(3, 20, 20)` table (`mcpu_params/hbond_seq_dep.bin`,
produced by `scripts/convert_hbond_seq_dep.py` from the already-bundled
the legacy `seq_dep_hb_mu_low.energy` table, sign-flipped per
legacy's `seq_hb[type][a][b] = -1.0 * value` convention). `amino_index` is a
per-residue `uint8_t` on `System` (`System::setAminoIndex`/`amino_index`),
populated from residue name via the same alphabetical order as legacy's
`GetAminoNumber()` (`pymcpu/forcefields/builders/hbond_builder.py::AMINO_ORDER`).

## Secondary structure (opt-in)

`System::secondary_structure_` (empty ⇒ every residue reads `'C'`, the
byte-identical default). Enable real secondary structure with
`MCPUForceField(..., compute_dssp=True)`: computes
`mdtraj.compute_dssp(traj, simplified=True)` at build time and maps DSSP's H/E
codes through unchanged, mapping simplified `'C'`/`'NA'` to the configurable
`dssp_coil_state` (default `'C'`). Legacy's fourth secstr state, `'L'`
("confident predicted coil", not "low confidence" — verified from usage in
`hbonds.h`'s Ramachandran gates), has no DSSP equivalent and is never inferred
automatically; it's only reachable via a literal legacy `.sec_str` file (not
currently implemented — DSSP is pyMCPU's SS source of truth).

## Resolved regression: RNG-cache checkpoint desync

Introducing non-uniform sequence-dependent scaling exposed a pre-existing,
unrelated determinism bug (it only became *visible* here because the changed
H-bond energetics shifted which move gets accepted at the exact step where the
bug mattered): `tests/physics/test_engine_determinism.py`'s round-trip test
would fail — two independently-built engines, given the same restored
`(coords, rng_state, current_step)`, diverged on the very next MC move.

Root cause: `MCIntegrator::angle_dist` (`std::normal_distribution<float>`)
caches a spare Gaussian across calls (libstdc++'s Marsaglia-polar
implementation) without consuming the underlying `mt19937` stream on every
other call. `get_rng_state()`/`set_rng_state()`/`set_seed()` only serialized
the `mt19937` state, never this cache, so a restored engine could silently
lose a pending spare and permanently desync from the continuing one. Fixed by
calling `angle_dist.reset()` from all three of those methods
(`include/pymcpu/Integrator.h`, `src/pymcpu/Integrator.cpp`). This also fixes
the same latent risk in `pymcpu/checkpointing.py`, `mpi_replica_exchange.py`,
and `we/propagator.py`, which all rely on the same RNG save/restore contract.

## Files

- `include/pymcpu/utils/hydrogen_bond_utils.h` — gates, `ang_CACA`, table indices.
- `src/pymcpu/forces/mcpu/common/HydrogenBondPotential.{h,cpp}` — `HBondPotential`,
  `seq_dep_factor`, `BETA_FAVOR`, `evaluate_directional`.
- `include/pymcpu/System.h` — `amino_index_`, `secondary_structure_` storage.
- `pymcpu/forcefields/builders/hbond_builder.py` — `AMINO_ORDER`, seq-dep table loader.
- `pymcpu/forcefields/mcpu.py` — DSSP computation, `amino_index`/SS plumbing into `System`.
- `tests/legacy_parity/helpers/legacy_hbond_oracle.py`,
  `tests/legacy_parity/test_hbond_ablation_ladder.py`,
  `tests/legacy_parity/test_hbond_engine_parity.py` — independent verification
  oracle and the pinned ablation-ladder targets.
- `scripts/convert_hbond_seq_dep.py` — legacy text table → `hbond_seq_dep.bin`.
- `scripts/validate_energy.py --legacy-hbond` — human-readable pyMCPU-vs-legacy-log
  side-by-side on the actin structure.
