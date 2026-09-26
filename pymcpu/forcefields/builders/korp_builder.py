"""Hand a parsed KORP map and a backbone topology to the C++ engine.

The map is parsed and validated in Python (:mod:`pymcpu.forcefields.korp_map`)
and the energy table is passed to C++ by reference, not copied. Keep the
returned map object alive for as long as any potential built from it: it owns
the reference that keeps the underlying numpy array (normally a memmap) from
being collected.
"""

from __future__ import annotations

import numpy as np

from pymcpu import mcpu_core
from pymcpu.forcefields.korp_map import KORP_RESIDUE_ORDER, KorpMap

__all__ = ["KorpPotentialBuilder"]

_KORP_INDEX = {name: i for i, name in enumerate(KORP_RESIDUE_ORDER)}


class KorpPotentialBuilder:
    """Builds the two C++ terms of the KORP force field."""

    @staticmethod
    def build_map(korp_map: KorpMap) -> "mcpu_core.OrientationalPairMap":
        arrays = korp_map.engine_arrays()
        table = np.asarray(korp_map.table)
        if table.dtype != np.dtype("<f4"):
            raise TypeError(
                f"KORP table must be little-endian float32, got {table.dtype}"
            )
        return mcpu_core.OrientationalPairMap(table=table, **arrays)

    @staticmethod
    def korp_types(res_names) -> list[int]:
        """Map three-letter residue names onto KORP's own type indices.

        KORP orders residues alphabetically by ONE-letter code, pyMCPU's
        ``AMINO_INDEX`` by three-letter code. The two disagree for most
        residues and neither ordering is visibly wrong at a glance, so this
        conversion is never done by hand.
        """
        try:
            return [_KORP_INDEX[name] for name in res_names]
        except KeyError as exc:
            raise ValueError(
                f"residue {exc.args[0]!r} has no KORP type; KORP is defined for "
                f"the 20 standard residues only"
            ) from exc

    @staticmethod
    def chain_codes(chain_ids) -> list[int]:
        """Chain tags as small integers, in first-seen order."""
        seen: dict = {}
        out = []
        for cid in chain_ids:
            if cid not in seen:
                if len(seen) >= 255:
                    raise ValueError("more than 255 chains is not supported")
                seen[cid] = len(seen)
            out.append(seen[cid])
        return out

    @classmethod
    def build_pair_potential(
        cls, engine_map, *, n_atom, ca_atom, c_atom, res_names, res_seq, chain_ids
    ) -> "mcpu_core.OrientationalPairPotential":
        return mcpu_core.OrientationalPairPotential(
            map=engine_map,
            n_atom=[int(i) for i in n_atom],
            ca_atom=[int(i) for i in ca_atom],
            c_atom=[int(i) for i in c_atom],
            korp_type=cls.korp_types(res_names),
            seq_number=[int(i) for i in res_seq],
            chain_id=cls.chain_codes(chain_ids),
        )

    @classmethod
    def build_steric_guard(
        cls, *, ca_atom, res_seq, chain_ids, min_separation=3, min_distance=3.2
    ) -> "mcpu_core.CalphaExcludedVolumePotential":
        return mcpu_core.CalphaExcludedVolumePotential(
            ca_atom=[int(i) for i in ca_atom],
            seq_number=[int(i) for i in res_seq],
            chain_id=cls.chain_codes(chain_ids),
            min_separation=int(min_separation),
            min_distance=float(min_distance),
        )
