"""Pure integer-arithmetic regression test for MPI exchange tag wraparound.

Moved out of the MPI checkpointing file: this doesn't touch MPI or
replica-exchange code at all, it's a standalone check of the modular
wraparound arithmetic ``MPIReplicaExchange`` uses to keep exchange tags
inside ``[0, TAG_UB]``.
"""

from __future__ import annotations


def test_exchange_tag_wrap_leaves_room_for_offset_pair() -> None:
    """``tag_base``..``tag_base + 2`` must stay within ``[0, TAG_UB]`` after
    wraparound.

    Regression for MPI_ERR_TAG on long REMD runs. 8388607 (2**23 - 1) is the
    Open MPI OFI transport's documented TAG_UB; the arithmetic itself only
    depends on TAG_UB being >= 3, so this reproduces the same modulus
    ``MPIReplicaExchange._attempt_mpi_exchange``/``_collect_exchange_record``
    use (``tag_mod = tag_ub - 1``), so ``tag_base + 2`` can never exceed
    ``tag_ub``.
    """
    tag_ub = 8388607  # Open MPI OFI TAG_UB = 2^23 - 1
    tag_mod = tag_ub - 1
    tag = 0
    for _ in range(tag_mod + 100):
        tag_base = tag
        tag = (tag + 10) % tag_mod
        assert 0 <= tag_base <= tag_ub - 2
        assert tag_base + 2 <= tag_ub
        collect_tag = tag
        tag = (tag + 1) % tag_mod
        assert 0 <= collect_tag <= tag_ub
