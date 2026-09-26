"""Build a backbone-only engine System for the KORP terms.

This is the layout a real `KORPForceField` will produce, in miniature: the
`[N,CA,C]` segment, then the `[O]` segment, and an EMPTY sidechain segment.

Three details here are load-bearing rather than incidental, and all three are
easy to get subtly wrong:

* `sc_start` must be **-1**, never a valid index with `sc_count == 0`. The KIC
  move has a fallback that, given a non-negative `sc_start` with no count,
  computes a span reaching into other residues' atoms and rigid-transforms
  them. It stays in bounds, so nothing crashes; the geometry is just wrong.
* `first_sc_of_residue` must be the end of the O segment for every residue,
  because pivot rotates `[first_sc_of_residue[r], sc_end)` without checking.
  Anything smaller rotates part of the O segment a second time.
* O is carried purely so the move machinery and this segment bookkeeping work
  unchanged. KORP never reads it.
"""

from __future__ import annotations

import numpy as np

from pymcpu import mcpu_core


def build_backbone_system(coords):
    """Return ``(system, context)`` for ``coords`` of shape ``(n_res, 3, 3)``.

    Atom order is N,CA,C per residue, then one O per residue. The O positions
    are placed on CA: KORP does not read them and the CA-CA steric guard does
    not either, so their only job is to make the segment layout real.
    """
    coords = np.asarray(coords, dtype=np.float32)
    n_res = coords.shape[0]
    n_bb, n_o = 3 * n_res, n_res
    n_atoms = n_bb + n_o

    system = mcpu_core.System(n_atoms, n_res)
    system.set_atom_counts(n_bb, n_o, 0, 0)

    blocks = []
    for r in range(n_res):
        block = mcpu_core.BlockIndices()
        block.bb_start = 3 * r          # N; CA is always bb_start + 1
        block.c_start = 3 * r + 2
        block.o_start = n_bb + r
        block.sc_start = -1             # see the module docstring
        block.sc_count = 0
        block.h_start = -1
        blocks.append(block)
    system.set_block_indices(blocks)

    atom_to_res = []
    for r in range(n_res):
        atom_to_res.extend([r, r, r])
    atom_to_res.extend(range(n_res))
    system.atom_to_residue = atom_to_res

    system.set_torsions_per_residue([0] * n_res)

    cache = mcpu_core.DownstreamCache()
    cache.first_sc_of_residue = np.full(n_res, n_atoms, dtype=np.int32)
    cache.first_o_of_residue = np.asarray(
        [n_bb + r for r in range(n_res)], dtype=np.int32)
    cache.first_h_of_residue = np.full(n_res, n_atoms, dtype=np.int32)
    system.set_downstream_cache(cache)

    context = mcpu_core.Context(system)
    positions = np.zeros((3, n_atoms), dtype=np.float32)
    for r in range(n_res):
        for k in range(3):
            positions[:, 3 * r + k] = coords[r, k]
        positions[:, n_bb + r] = coords[r, 1]
    context.set_positions(positions)
    return system, context


def residue_atom_indices(n_res):
    """``(n_atom, ca_atom, c_atom)`` index lists for :func:`build_backbone_system`."""
    return (
        [3 * r for r in range(n_res)],
        [3 * r + 1 for r in range(n_res)],
        [3 * r + 2 for r in range(n_res)],
    )
