# Aromatic stacking (KERNEL-5)

**Python API:** :class:`~pymcpu.AromaticPotential` (see :doc:`/api/forces`).

Temperature in the Metropolis criterion is a **dimensionless**
reduced parameter (typical 0.3–0.6); aromatic energies are
**unitless** (`aromatic_E[bin] / 1000` × `ARO_WEIGHT`).

## Aromatic residue types

Only **PHE** and **TRP** participate in aromatic stacking. **TYR** and **HIS** are excluded at code level (never registered in `aromatic_residues`). The parameter file `aromatic_noTYR.energy` matches this behavior.

## Ring atom definitions

| Residue | Ring center atoms | Normal vectors |
|---------|-------------------|----------------|
| PHE | CG, CE1, CE2 | (CG→CE1) × (CG→CE2) |
| TRP | CG, CZ2, CZ3 | (CG→CZ2) × (CG→CZ3) |

Ring center = arithmetic mean of the three atoms. Normal = normalized cross product of the two ring vectors from CG.

## Acute angle and 89.9° cap

The angle between ring plane normals is converted to degrees, folded to acute (`if angle > 90° then 180° − angle`), then capped:

```
if (angle_deg > 89.9°) angle_deg = 89.9°;
```

The cap is applied **before** `safe_bin()`. It ensures bin 8 covers 80–89.9°, not 80–90°, matching legacy behavior when normals are orthogonal.

## 1D lookup table

- Shape: `aromatic_E[9]` — nine bins
- Bin width: 10° (`aro_int = 10.0`)
- Range: 0°–89.9° after cap
- Default unset cells: **0** (not 1000)

## Energy formula

Per pair within 7.0 Å center distance:

```
E_pair = aromatic_E[bin] / 1000.0
delta_aromatic = ARO_WEIGHT × Σ_pairs (E_new − E_old)
```

`ARO_WEIGHT = 5.0`.

## Distance gate

Pairs are skipped when squared center distance ≥ (7.0 Å)² = 49.0 Å².

## Move gating

Unlike hbond (backbone only) or sidechain triplet (sidechain only), aromatic energy is evaluated for **all move types** in legacy. The pyMCPU kernel returns 0 only when no moved atom belongs to a registered aromatic residue.

## Delta kernel scope

`aromatic_delta()` iterates unique aromatic pairs where at least one residue was moved. Complexity is O(n_aro²) with typical n_aro ≪ N_atoms.
