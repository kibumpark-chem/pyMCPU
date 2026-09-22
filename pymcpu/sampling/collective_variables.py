"""Collective variables for biased sampling on top of the pyMCPU engine.

Provides the native-contacts CV on contact atoms (CA–CA or CB–CB):

* **N** — number (or soft effective number) of formed native contacts
* **Q** — normalized fraction ``N / n_contacts`` (convenience only)

Production bias / replica-exchange use hard-cutoff **N** to match C++
``NativeContactsBiasPotential``: ``U = 0.5 * k * (N - N0)^2``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import mdtraj as md

from pymcpu.config import ContactAtomMode, VALID_CONTACT_ATOM_MODES

if TYPE_CHECKING:  # avoid importing the engine just to use this module
    from pymcpu.forcefields.mcpu import MCPUForceField


def normalize_contact_atom_mode(mode: str | None) -> ContactAtomMode:
    """Validate contact atom mode (default: ``ca``)."""
    if mode is None or mode == "":
        return "ca"
    m = str(mode).strip().lower()
    if m not in VALID_CONTACT_ATOM_MODES:
        raise ValueError(
            f"contact_atom_mode must be one of {VALID_CONTACT_ATOM_MODES}, got {mode!r}"
        )
    return m  # type: ignore[return-value]


class NativeContactsCV:
    """Native-contacts collective variable on contact atoms (CA or CB).

    The *native contact set* can be defined in one of two ways:

    * **Derived** (default, ``native_contact_pairs=None``): pairs of residues
      whose sequence separation is at least ``min_seq_sep`` and whose
      reference contact-atom distance is below ``contact_cutoff`` (Å). Both
      are user-defined; the C++ bias potential receives the same precomputed pair
      list.
    * **Explicit**: pass ``native_contact_pairs`` — an ``(n_pairs, 2)``
      array-like of 0-based residue indices — to hand-specify the exact pair
      list instead. When given, it *replaces* the cutoff/``min_seq_sep``
      derivation entirely (not unioned with it), so ``contact_cutoff`` and
      ``min_seq_sep`` play no role in pair selection in this mode. The list
      is validated: it must be non-empty and shape ``(n_pairs, 2)``; every
      index must be in ``[0, n_res)``; no ``i == j`` self-pairs; no duplicate
      pairs, where ``(i, j)`` and ``(j, i)`` count as the same pair. The
      validated pairs are copied to ``self.native_contact_pairs`` for
      provenance/introspection (e.g. checkpoint logging, ``describe()``);
      it is ``None`` when the set was derived instead.

      Note that even in explicit-pairs mode, ``contact_cutoff`` keeps a
      second, independent role: it is the fallback value for ``q_cutoff``
      (the "is this contact currently formed" distance threshold used by
      :meth:`compute_N`/:meth:`compute_Q`) whenever ``q_cutoff`` is not
      given explicitly. A caller who wants full control over that threshold
      must still pass ``q_cutoff`` explicitly, or ``contact_cutoff`` will
      silently supply it.

    Atom selection is controlled by ``contact_atom_mode``:

    * ``"ca"`` — backbone CA–CA (default)
    * ``"cb"`` — CB–CB; residues without CB (e.g. GLY) use backbone CA

    Evaluation modes (``mode``):

    * ``"hard"`` (production / matches C++ bias):
          N = sum_i  [ r_i < q_cutoff ]
          Q = N / n_contacts
    * ``"soft"`` (analysis-only; Best–Hummer–Eaton):
          N_eff = sum_i  1 / (1 + exp(beta_c * (r_i - lam * r0_i)))
          Q = N_eff / n_contacts

    Soft mode must **not** be used for Metropolis exchange against the C++
    hard bias. All coordinates are in Å (engine internal units).

    ``fixed_residue_mask`` and ``energy_ignored_residue_mask`` behave
    differently depending on the mode:

    * **Derived mode**: a fixed–fixed pair (both residues in
      ``fixed_residue_mask``) is auto-excluded, as an efficiency/policy
      choice — a fixed-fixed distance can never change under MC. A pair
      touching an energy-ignored (linker/ghost) residue is likewise
      auto-excluded.
    * **Explicit mode**: ``fixed_residue_mask`` is validated for shape only
      and does **not** auto-exclude fixed-fixed pairs — a user who
      explicitly typed a pair made a deliberate choice (e.g. a regression
      check that a nominally-rigid contact stays formed) and should not have
      it silently vanish. ``energy_ignored_residue_mask``, however, still
      causes a hard ``ValueError`` if any explicit pair touches a masked
      residue: its coordinates are physically meaningless, so silently
      computing a "contact" there would look like real physics. This is a
      correctness guard, not a policy choice, so it applies unconditionally
      (unlike the derived-mode auto-exclusion above).
    """

    def __init__(
        self,
        ca_internal_idx: np.ndarray,
        ref_ca_xyz: np.ndarray,
        contact_cutoff: float = 8.0,
        min_seq_sep: int = 3,
        beta_c: float = 5.0,
        lam: float = 1.2,
        mode: str = "hard",
        q_cutoff: float | None = None,
        fixed_residue_mask: np.ndarray | None = None,
        energy_ignored_residue_mask: np.ndarray | None = None,
        contact_atom_mode: str = "ca",
        native_contact_pairs: Sequence[Sequence[int]] | np.ndarray | None = None,
    ):
        self.contact_atom_mode = normalize_contact_atom_mode(contact_atom_mode)
        self.ca_internal_idx = np.asarray(ca_internal_idx, dtype=np.int64)
        # NOTE: distances are handled in Angstroms, so beta_c carries units of
        # A^-1. The Best-Hummer-Eaton value (~5 nm^-1) corresponds to ~0.5 A^-1;
        # the 5.0 default here is intentionally sharper.
        self.beta_c = float(beta_c)
        self.lam = float(lam)
        if mode not in ("soft", "hard"):
            raise ValueError(f"Q mode must be 'soft' or 'hard', got {mode!r}")
        self.mode = mode
        self.q_cutoff = float(q_cutoff if q_cutoff is not None else contact_cutoff)

        ref_ca_xyz = np.asarray(ref_ca_xyz, dtype=np.float64)
        if ref_ca_xyz.ndim != 2 or ref_ca_xyz.shape[1] != 3:
            raise ValueError(
                f"ref_ca_xyz must have shape (n_res, 3), got {ref_ca_xyz.shape}."
            )
        n_res = ref_ca_xyz.shape[0]
        if self.ca_internal_idx.shape[0] != n_res:
            raise ValueError(
                "ca_internal_idx and ref_ca_xyz disagree on residue count: "
                f"{self.ca_internal_idx.shape[0]} contact-atom indices vs "
                f"{n_res} reference positions. "
                "Check that the engine residue ordering matches the reference PDB."
            )
        diff = ref_ca_xyz[:, None, :] - ref_ca_xyz[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))

        if native_contact_pairs is not None:
            # --- Explicit pairs: replaces cutoff/min_seq_sep derivation. ---
            pairs_arr = np.asarray(native_contact_pairs, dtype=np.int64)
            if pairs_arr.ndim != 2 or pairs_arr.shape[1] != 2:
                raise ValueError(
                    "native_contact_pairs must have shape (n_pairs, 2), got "
                    f"{pairs_arr.shape}."
                )
            if pairs_arr.shape[0] == 0:
                raise ValueError(
                    "native_contact_pairs must not be empty. Omit the argument "
                    "entirely (leave it None) to fall back to cutoff/min_seq_sep "
                    "derivation."
                )
            if np.any(pairs_arr < 0) or np.any(pairs_arr >= n_res):
                raise ValueError(
                    "native_contact_pairs contains an index outside [0, "
                    f"{n_res}): min={int(pairs_arr.min())}, "
                    f"max={int(pairs_arr.max())}."
                )
            pi = pairs_arr[:, 0]
            pj = pairs_arr[:, 1]
            if np.any(pi == pj):
                bad = pairs_arr[pi == pj]
                raise ValueError(
                    "native_contact_pairs contains self-pair(s) (i == j): "
                    f"{bad.tolist()}."
                )
            # Canonical undirected-pair key: (i, j) and (j, i) collide.
            keys = np.minimum(pi, pj) * n_res + np.maximum(pi, pj)
            unique_keys, counts = np.unique(keys, return_counts=True)
            if np.any(counts > 1):
                dup_keys = unique_keys[counts > 1]
                bad = pairs_arr[np.isin(keys, dup_keys)]
                raise ValueError(
                    "native_contact_pairs contains duplicate pair(s) (order-"
                    f"independent, i.e. (i, j) == (j, i)): {bad.tolist()}."
                )

            # energy_ignored_residue_mask: hard-reject (correctness guard, not
            # a policy choice) — an explicit pair touching a linker/ghost
            # residue would compute a "contact" from physically meaningless
            # coordinates.
            if energy_ignored_residue_mask is not None:
                emask = np.asarray(energy_ignored_residue_mask, dtype=bool)
                if emask.shape[0] != n_res:
                    raise ValueError(
                        f"energy_ignored_residue_mask length ({emask.shape[0]}) != n_res ({n_res})"
                    )
                either_ignored = emask[pi] | emask[pj]
                if np.any(either_ignored):
                    bad = pairs_arr[either_ignored]
                    raise ValueError(
                        "native_contact_pairs contains pair(s) touching an "
                        f"energy-ignored (linker/ghost) residue: {bad.tolist()}. "
                        "Such residues have physically meaningless coordinates, "
                        "so an explicit contact there cannot be computed."
                    )

            # fixed_residue_mask: shape-validated only. Unlike the derived
            # branch, fixed-fixed pairs are NOT auto-excluded here — a user
            # who explicitly typed a pair made a deliberate choice and should
            # not have it silently dropped.
            if fixed_residue_mask is not None:
                fmask = np.asarray(fixed_residue_mask, dtype=bool)
                if fmask.shape[0] != n_res:
                    raise ValueError(
                        f"fixed_residue_mask length ({fmask.shape[0]}) != n_res ({n_res})"
                    )

            self.pairs_i = pi
            self.pairs_j = pj
            self.r0 = dist[self.pairs_i, self.pairs_j]
            self.n_contacts = int(self.pairs_i.size)
            self.n_excluded_fixed_fixed = 0
            self.n_excluded_energy_ignored = 0
            self.native_contact_pairs = pairs_arr.copy()
        else:
            iu, ju = np.triu_indices(n_res, k=1)
            seq_sep_ok = (ju - iu) >= min_seq_sep
            within_cutoff = dist[iu, ju] < contact_cutoff
            keep = seq_sep_ok & within_cutoff

            # Exclude fixed–fixed contacts when a fixed mask is provided.
            if fixed_residue_mask is not None:
                fmask = np.asarray(fixed_residue_mask, dtype=bool)
                if fmask.shape[0] != n_res:
                    raise ValueError(
                        f"fixed_residue_mask length ({fmask.shape[0]}) != n_res ({n_res})"
                    )
                both_fixed = fmask[iu] & fmask[ju]
                keep = keep & ~both_fixed
                self.n_excluded_fixed_fixed = int(
                    np.sum(both_fixed & seq_sep_ok & within_cutoff)
                )
            else:
                self.n_excluded_fixed_fixed = 0

            # Exclude pairs involving any energy-ignored (linker/ghost) residue.
            if energy_ignored_residue_mask is not None:
                emask = np.asarray(energy_ignored_residue_mask, dtype=bool)
                if emask.shape[0] != n_res:
                    raise ValueError(
                        f"energy_ignored_residue_mask length ({emask.shape[0]}) != n_res ({n_res})"
                    )
                either_ignored = emask[iu] | emask[ju]
                keep = keep & ~either_ignored
                self.n_excluded_energy_ignored = int(
                    np.sum(either_ignored & seq_sep_ok & within_cutoff)
                )
            else:
                self.n_excluded_energy_ignored = 0

            self.pairs_i = iu[keep]
            self.pairs_j = ju[keep]
            self.r0 = dist[self.pairs_i, self.pairs_j]
            self.n_contacts = int(self.pairs_i.size)

            if self.n_contacts == 0:
                raise ValueError(
                    "No native contacts found. Check the reference, contact_cutoff "
                    f"({contact_cutoff} A) and min_seq_sep ({min_seq_sep})."
                )
            self.native_contact_pairs = None

    def atom_pair_indices(self) -> tuple[np.ndarray, np.ndarray]:
        """Engine-internal contact-atom index pairs used by the C++ bias potential."""
        return (
            self.ca_internal_idx[self.pairs_i].astype(np.int32),
            self.ca_internal_idx[self.pairs_j].astype(np.int32),
        )

    def _pair_distances(self, coords_3xn: np.ndarray) -> np.ndarray:
        coords_3xn = np.asarray(coords_3xn)
        if coords_3xn.ndim != 2 or coords_3xn.shape[0] != 3:
            raise ValueError(
                f"coords_3xn must have shape (3, n_atoms), got {coords_3xn.shape}."
            )
        ca = coords_3xn[:, self.ca_internal_idx].T  # (n_res, 3)
        d = ca[self.pairs_i] - ca[self.pairs_j]
        return np.sqrt(np.sum(d * d, axis=-1))

    def compute_N(self, coords_3xn: np.ndarray) -> float:
        """Native-contact count N (hard integer sum, or soft effective count)."""
        r = self._pair_distances(coords_3xn)
        if self.mode == "hard":
            return float(np.sum(r < self.q_cutoff))
        z = np.clip(self.beta_c * (r - self.lam * self.r0), -50.0, 50.0)
        return float(np.sum(1.0 / (1.0 + np.exp(z))))

    def compute_Q(self, coords_3xn: np.ndarray) -> float:
        """Normalized fraction Q = N / n_contacts."""
        return float(self.compute_N(coords_3xn) / self.n_contacts)

    def compute(self, coords_3xn: np.ndarray) -> float:
        """Alias for :meth:`compute_Q` (historical name). Prefer :meth:`compute_N`."""
        return self.compute_Q(coords_3xn)

    def fraction_to_count(self, q: float) -> float:
        """Map a fraction target Q* to count target N0 = Q* * n_contacts."""
        return float(q) * float(self.n_contacts)

    def count_to_fraction(self, n: float) -> float:
        """Map a count N to fraction Q = N / n_contacts."""
        return float(n) / float(self.n_contacts)


def attach_native_contacts_bias_potential(system, cv: NativeContactsCV):
    """Register the hard-N harmonic bias potential on an MCPU system.

    Context targets are set later per replica via
    ``context.set_native_contacts_bias(k, N0)``.
    """
    from pymcpu import mcpu_core

    if cv.mode != "hard":
        raise ValueError(
            "C++ NativeContactsBiasPotential is hard-cutoff only; "
            f"got NativeContactsCV(mode={cv.mode!r})."
        )
    atom_i, atom_j = cv.atom_pair_indices()
    potential = mcpu_core.NativeContactsBiasPotential(atom_i, atom_j, float(cv.q_cutoff))
    potential.set_energy_group(6)
    system.add_potential(potential)
    return potential


def build_contact_atom_index(
    forcefield: "MCPUForceField",
    mode: str = "ca",
) -> np.ndarray:
    """Return engine-internal contact-atom indices, ordered by residue.

    * ``ca`` — backbone CA (excludes the GLY sidechain CA duplicate).
    * ``cb`` — CB when present; otherwise backbone CA (GLY and any residue
      without CB). Never uses the GLY SC-slot CA duplicate.
    """
    mode_n = normalize_contact_atom_mode(mode)
    n_bb = forcefield.total_bb_atoms
    atoms = forcefield.ordered_atom_list

    if mode_n == "ca":
        ca = [
            (atom.residue_index, idx)
            for idx, atom in enumerate(atoms)
            if atom.name == "CA" and idx < n_bb
        ]
        ca.sort(key=lambda t: t[0])
        return np.array([idx for _, idx in ca], dtype=np.int64)

    # cb mode: one index per residue via BlockIndices
    n_res = int(forcefield.n_res)
    out = np.empty(n_res, dtype=np.int64)
    for r in range(n_res):
        block = forcefield.blocks[r]
        bb_ca = block.bb_start + 1  # N, CA, C layout
        sc = int(block.sc_start)
        if sc >= 0 and sc < len(atoms) and atoms[sc].name == "CB":
            out[r] = sc
        else:
            out[r] = bb_ca
    return out


def build_ca_index(forcefield: "MCPUForceField") -> np.ndarray:
    """Return engine-internal indices of the backbone CA atoms, ordered by residue.

    Thin wrapper around :func:`build_contact_atom_index` with ``mode="ca"``.
    """
    return build_contact_atom_index(forcefield, mode="ca")


def reference_contact_from_pdb(
    reference_pdb: str,
    mode: str = "ca",
) -> np.ndarray:
    """Load contact-atom coordinates (Angstroms), ordered by residue.

    * ``ca`` — all CA atoms.
    * ``cb`` — CB when present on the residue, else CA (GLY fallback).
    """
    mode_n = normalize_contact_atom_mode(mode)
    ref = md.load(reference_pdb)
    top = ref.topology
    xyz_nm = ref.xyz[0]

    if mode_n == "ca":
        ca_sel = top.select("name CA")
        if ca_sel.size == 0:
            raise ValueError(f"No CA atoms found in reference structure: {reference_pdb}")
        return xyz_nm[ca_sel, :] * 10.0

    coords: list[np.ndarray] = []
    for residue in top.residues:
        cb_idx = None
        ca_idx = None
        for atom in residue.atoms:
            if atom.name == "CB":
                cb_idx = atom.index
            elif atom.name == "CA":
                ca_idx = atom.index
        if cb_idx is not None:
            coords.append(xyz_nm[cb_idx])
        elif ca_idx is not None:
            coords.append(xyz_nm[ca_idx])
        else:
            raise ValueError(
                f"Residue {residue} in {reference_pdb} has neither CB nor CA"
            )
    if not coords:
        raise ValueError(f"No contact atoms found in reference structure: {reference_pdb}")
    return np.asarray(coords, dtype=np.float64) * 10.0


def reference_ca_from_pdb(reference_pdb: str) -> np.ndarray:
    """Load CA coordinates (Angstroms), ordered by residue, from a reference PDB."""
    return reference_contact_from_pdb(reference_pdb, mode="ca")


def kabsch_rmsd(mobile_xyz: np.ndarray, ref_xyz: np.ndarray) -> float:
    """RMSD (Angstrom) between ``mobile_xyz`` and ``ref_xyz`` after optimal
    (Kabsch) superposition. Both must be shape ``(n, 3)`` and in the same
    units. Raises on malformed input rather than returning a sentinel value —
    a silent ``0.0`` here would read as "perfectly superposed."
    """
    mobile_xyz = np.asarray(mobile_xyz, dtype=np.float64)
    ref_xyz = np.asarray(ref_xyz, dtype=np.float64)
    if mobile_xyz.shape != ref_xyz.shape:
        raise ValueError(
            f"mobile_xyz and ref_xyz shape mismatch: {mobile_xyz.shape} vs {ref_xyz.shape}"
        )
    if mobile_xyz.ndim != 2 or mobile_xyz.shape[1] != 3:
        raise ValueError(f"expected shape (n, 3), got {mobile_xyz.shape}")

    mob_c = mobile_xyz - mobile_xyz.mean(axis=0)
    ref_c = ref_xyz - ref_xyz.mean(axis=0)

    # Kabsch algorithm. cov = ref_c^T @ mob_c (note: reference on the left);
    # with that convention the SVD-derived `rot` below is already the
    # rotation that maps mob_c onto ref_c directly (mob_c @ rot), NOT its
    # transpose. Applying `mob_c @ rot.T` here is a common bug (verified
    # against scipy's Rotation.align_vectors: it leaves the full rotational
    # misalignment in the residual, silently inflating RMSD for any mobile
    # structure whose orientation differs from the reference).
    cov = ref_c.T @ mob_c
    u, _s, vt = np.linalg.svd(cov)
    d = np.linalg.det(vt.T @ u.T)
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T  # reflection fix
    rotated = mob_c @ rot
    return float(np.sqrt(np.mean(np.sum((rotated - ref_c) ** 2, axis=1))))


class CARMSDCV:
    """CA-RMSD-to-reference collective variable, mirroring :class:`NativeContactsCV`.

    Unlike :meth:`FoldingRunner._compute_rmsd_values`, :meth:`compute` raises
    on failure instead of returning ``0.0`` — as a progress coordinate, a
    silent zero would be indistinguishable from "perfectly folded."
    """

    def __init__(self, ca_internal_idx: np.ndarray, ref_ca_xyz: np.ndarray):
        self.ca_internal_idx = np.asarray(ca_internal_idx, dtype=np.int64)
        ref_ca_xyz = np.asarray(ref_ca_xyz, dtype=np.float64)
        if ref_ca_xyz.ndim != 2 or ref_ca_xyz.shape[1] != 3:
            raise ValueError(f"ref_ca_xyz must have shape (n_res, 3), got {ref_ca_xyz.shape}")
        if self.ca_internal_idx.shape[0] != ref_ca_xyz.shape[0]:
            raise ValueError(
                "ca_internal_idx and ref_ca_xyz disagree on residue count: "
                f"{self.ca_internal_idx.shape[0]} vs {ref_ca_xyz.shape[0]}"
            )
        self.ref_ca_xyz = ref_ca_xyz

    def compute(self, coords_3xn: np.ndarray) -> float:
        """CA-RMSD (Angstrom) of ``coords_3xn`` (engine order, shape (3, n_atoms))
        against the bound reference, after optimal superposition."""
        coords_3xn = np.asarray(coords_3xn)
        if coords_3xn.ndim != 2 or coords_3xn.shape[0] != 3:
            raise ValueError(f"coords_3xn must have shape (3, n_atoms), got {coords_3xn.shape}")
        mobile_ca = coords_3xn[:, self.ca_internal_idx].T  # (n_res, 3)
        return kabsch_rmsd(mobile_ca, self.ref_ca_xyz)
