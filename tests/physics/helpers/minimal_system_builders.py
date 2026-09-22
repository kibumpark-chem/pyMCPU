"""Shared helpers for building minimal C++ systems in potential tests."""

from __future__ import annotations

import numpy as np

from pymcpu import mcpu_core


def _build_downstream_cache(
    blocks: list[mcpu_core.BlockIndices],
    *,
    n_atoms: int,
    sc_end: int,
) -> mcpu_core.DownstreamCache:
    """Mirror forcefield DownstreamCache construction for minimal fixtures.

    Integrator pivot moves index ``first_*_of_residue`` unconditionally on the
    legacy contiguous path; an empty cache is a null-deref segfault.
    """
    n_res = len(blocks)
    first_sc: list[int] = []
    first_o: list[int] = []
    first_h: list[int] = []
    for res_idx, block in enumerate(blocks):
        if block.o_start >= 0:
            first_o.append(int(block.o_start))
        else:
            # No oxygen segment: point at end so rotate_range is a no-op.
            first_o.append(n_atoms)
        if block.sc_start >= 0:
            first_sc.append(int(block.sc_start))
        else:
            next_valid = sc_end
            for lookahead in range(res_idx + 1, n_res):
                if blocks[lookahead].sc_start >= 0:
                    next_valid = int(blocks[lookahead].sc_start)
                    break
            first_sc.append(next_valid)
        if block.h_start >= 0:
            first_h.append(int(block.h_start))
        else:
            next_valid = n_atoms
            for lookahead in range(res_idx + 1, n_res):
                if blocks[lookahead].h_start >= 0:
                    next_valid = int(blocks[lookahead].h_start)
                    break
            first_h.append(next_valid)

    cache = mcpu_core.DownstreamCache()
    cache.first_sc_of_residue = np.asarray(first_sc, dtype=np.int32)
    cache.first_o_of_residue = np.asarray(first_o, dtype=np.int32)
    cache.first_h_of_residue = np.asarray(first_h, dtype=np.int32)
    return cache


def make_minimal_bb_blocks(n_res: int, atoms_per_res: int = 4) -> list[mcpu_core.BlockIndices]:
    """Legacy BB|O layout: [N,CA,C]*n_res then [O]*n_res.

    ``atoms_per_res`` is kept for call-site compatibility but the packed
    layout is always 3 BB + 1 O per residue (total 4*n_res atoms).
    """
    del atoms_per_res  # API compat; layout is fixed to BB|O
    blocks = []
    o_seg0 = n_res * 3
    for r in range(n_res):
        block = mcpu_core.BlockIndices()
        block.bb_start = r * 3
        block.o_start = o_seg0 + r
        block.sc_start = -1
        block.h_start = -1
        blocks.append(block)
    return blocks


def setup_minimal_bb_system(n_res: int, n_atoms: int) -> tuple[mcpu_core.System, mcpu_core.Context]:
    system = mcpu_core.System(n_atoms, n_res)
    system.set_atom_counts(n_res * 3, n_res, 0, 0)
    blocks = make_minimal_bb_blocks(n_res)
    system.set_block_indices(blocks)
    system.set_torsions_per_residue([0] * n_res)
    # atom_to_residue: BB segment then O segment
    atom_to_res = []
    for r in range(n_res):
        atom_to_res.extend([r, r, r])  # N, CA, C
    atom_to_res.extend(list(range(n_res)))  # O
    system.atom_to_residue = atom_to_res
    sc_end = n_res * 3 + n_res  # BB + O, no SC
    system.set_downstream_cache(
        _build_downstream_cache(blocks, n_atoms=n_atoms, sc_end=sc_end)
    )
    context = mcpu_core.Context(system)
    return system, context


def setup_minimal_sc_system(n_res: int, n_atoms: int) -> tuple[mcpu_core.System, mcpu_core.Context]:
    system = mcpu_core.System(n_atoms, n_res)
    blocks = []
    atom_to_res = []
    # Legacy BB|SC layout: [N,CA,C]*n_res then one SC atom per residue.
    sc_seg0 = n_res * 3
    for r in range(n_res):
        block = mcpu_core.BlockIndices()
        block.bb_start = r * 3
        block.sc_start = sc_seg0 + r
        block.sc_count = 1
        block.o_start = -1
        block.h_start = -1
        blocks.append(block)
    for r in range(n_res):
        atom_to_res.extend([r, r, r])
    atom_to_res.extend(list(range(n_res)))
    system.set_atom_counts(n_res * 3, 0, n_res, 0)
    system.set_block_indices(blocks)
    system.set_torsions_per_residue([1] * n_res)
    system.atom_to_residue = atom_to_res
    system.set_downstream_cache(
        _build_downstream_cache(blocks, n_atoms=n_atoms, sc_end=n_atoms)
    )
    context = mcpu_core.Context(system)
    return system, context
