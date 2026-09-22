"""Concurrent materialization must be serialized, not raced.

``scripts/job_template.slurm`` launches N MPI ranks per node against a shared
``$HOME``, so N processes can call ``ensure_params`` simultaneously on a cold
cache. Two things must hold: no rank may observe a partial parameter file, and
N ranks must not each decode 678 MiB.

This uses ``multiprocessing`` with the *spawn* start method rather than MPI, so
it runs in ordinary CI and also covers the non-MPI cases that share the same
hazard: WESTPA workers, ``xargs -P``, SLURM job arrays, and two unrelated runs
sharing a cache.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pytest

from pymcpu.params import bundled_tables_path, required_files

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        bundled_tables_path("mcpu_v1") is None,
        reason="no in-wheel parameter archive (run scripts/encode_params.py)",
    ),
]

_N_WORKERS = 4


def _worker(cache_dir: str, report_path: str) -> None:
    """Materialize, recording whether *this* process did the decoding.

    Must be module-level and picklable for the spawn start method.
    """
    import logging

    os.environ["MCPU_CACHE_DIR"] = cache_dir

    did_materialize = {"value": False}

    class _Sentinel(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "materialized" in record.getMessage():
                did_materialize["value"] = True

    logging.getLogger("pymcpu.params").addHandler(_Sentinel())
    logging.getLogger("pymcpu.params").setLevel(logging.INFO)

    import pymcpu.params as params

    result: dict[str, object] = {"pid": os.getpid()}
    try:
        root = params.materialize_from_wheel("mcpu_v1", verify=False, timeout=600.0)
        result["root"] = str(root)
        result["materialized"] = did_materialize["value"]
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        result["error"] = f"{type(exc).__name__}: {exc}"
    Path(report_path).write_text(json.dumps(result))


@pytest.mark.slow
def test_concurrent_materialization_is_serialized(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir()
    ctx = mp.get_context("spawn")

    reports = [tmp_path / f"report_{i}.json" for i in range(_N_WORKERS)]
    procs = [
        ctx.Process(target=_worker, args=(str(cache), str(r)))
        for r in reports
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=900)

    assert all(not p.is_alive() for p in procs), "a worker hung"
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    results = [json.loads(r.read_text()) for r in reports if r.is_file()]
    assert len(results) == _N_WORKERS, f"only {len(results)} workers reported"

    errors = [r["error"] for r in results if "error" in r]
    assert not errors, f"workers failed: {errors}"

    # 1. every worker agrees on the same content-addressed directory
    roots = {r["root"] for r in results}
    assert len(roots) == 1, f"workers disagreed on the params root: {roots}"
    root = Path(next(iter(roots)))

    # 2. exactly one worker actually did the decoding
    n_materialized = sum(bool(r["materialized"]) for r in results)
    assert n_materialized == 1, (
        f"{n_materialized} workers materialized; the mkdir lock should admit "
        "exactly one"
    )

    # 3. the result is complete and internally consistent -- no worker can
    #    have observed (or left) a partial file
    for relpath in required_files("mcpu_v1").values():
        path = root / relpath
        assert path.is_file(), f"missing {relpath}"
        if relpath.endswith(".bin"):
            assert path.stat().st_size % 4 == 0, f"{relpath} is not whole float32s"
            # cheap sanity: decodes without error and is finite
            head = np.fromfile(path, dtype=np.float32, count=1024)
            assert np.isfinite(head).all(), f"{relpath} has non-finite values"

    # 4. no staging or lock directories left behind
    leftovers = [p.name for p in root.parent.iterdir() if p.name.startswith(".")]
    assert not leftovers, f"lock/staging not cleaned up: {leftovers}"
