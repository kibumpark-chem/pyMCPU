"""Sampling drivers and collective variables for pyMCPU."""

from pymcpu.sampling.cv_factory import (
    CaRmsd,
    CollectiveVariable,
    CompositeCV,
    NativeContactsN,
    NativeContactsQ,
    TwoStateDelta,
    TwoStateRmsd,
    build_cv,
    custom_cv_from_spec,
)
from pymcpu.sampling.engine_session import (
    EngineSession,
    build_forcefield,
    compute_fingerprint,
)
from pymcpu.sampling.seeding import derive_seed
from pymcpu.sampling.collective_variables import (
    CARMSDCV,
    NativeContactsCV,
    attach_native_contacts_bias_potential,
    build_ca_index,
    build_contact_atom_index,
    reference_ca_from_pdb,
    reference_contact_from_pdb,
)
from pymcpu.sampling.folding import FoldingRunner
from pymcpu.sampling.replica_exchange import (
    ExchangeRecord,
    Replica,
    ReplicaExchange,
    ReplicaState,
    RunSummary,
    evaluate_exchange_acceptance,
    evaluate_exchange_delta,
    get_coords,
    make_n_targets,
    make_q_targets,
    make_temperature_ladder,
    swap_context_coordinates,
)

try:
    from pymcpu.sampling.mpi_replica_exchange import MPIReplicaExchange, partition_replicas
except Exception:  # pragma: no cover — ImportError or missing libmpi at runtime
    MPIReplicaExchange = None
    partition_replicas = None

__all__ = [
    "CARMSDCV",
    "CaRmsd",
    "CollectiveVariable",
    "CompositeCV",
    "EngineSession",
    "ExchangeRecord",
    "FoldingRunner",
    "MPIReplicaExchange",
    "NativeContactsCV",
    "NativeContactsN",
    "NativeContactsQ",
    "Replica",
    "ReplicaExchange",
    "ReplicaState",
    "RunSummary",
    "TwoStateDelta",
    "TwoStateRmsd",
    "attach_native_contacts_bias_potential",
    "build_ca_index",
    "build_contact_atom_index",
    "build_cv",
    "build_forcefield",
    "compute_fingerprint",
    "custom_cv_from_spec",
    "derive_seed",
    "evaluate_exchange_acceptance",
    "evaluate_exchange_delta",
    "get_coords",
    "make_n_targets",
    "make_q_targets",
    "make_temperature_ladder",
    "partition_replicas",
    "reference_ca_from_pdb",
    "reference_contact_from_pdb",
    "swap_context_coordinates",
]
