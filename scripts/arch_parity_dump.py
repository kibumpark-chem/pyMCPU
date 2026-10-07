#!/usr/bin/env python3
"""Fingerprint a pyMCPU build's Monte Carlo behaviour, bit-exactly, to a file.

Why this exists
---------------
The five ``scripts/parity_*.py`` oracles each import ``pymcpu`` once and then
run twice *in the same interpreter*, toggling a runtime setter between the two
runs. That makes them excellent at proving two code paths agree, and
structurally incapable of proving two *separately compiled binaries* agree --
which is exactly what a change to ``-march`` or ``-ffp-contract`` needs.

So this script writes the fingerprint to disk. Record from build A, rebuild,
compare against build B::

    python scripts/arch_parity_dump.py --out ref.json
    # ... rebuild with a different flag ...
    python scripts/arch_parity_dump.py --compare ref.json

What fails the comparison
-------------------------
Energies (hex floats), the accept-bit stream, the coordinate hash, the
per-move accept counts and the step totals must match exactly. Internal work
counters (``proxy_stats``, ``mu_by_kind`` and the pair-call entry of
``step_ints``) are printed when they differ but do not fail the run: a change
that skips provably irrelevant work reaches the same trajectory with smaller
counters, and that is a pass. The step totals (``n_steps``,
``n_valid_moves``, ``n_accepts``, ``moved_atoms_sum``) are strict: each must be
present on both sides and equal. Any other ``step_ints`` key is a work counter,
including one only one side recorded (a counter added or retired since the
reference was recorded).

Why the numbers are stored as hex floats
----------------------------------------
Nothing else in this repo prints a bit-faithful float -- every parity script
uses ``.4f``/``.6f``/``:g``, and ``scripts/validate_energy.py`` uses
``:>16.6f``.

The reason that matters is the opposite of the intuitive one. ``%.6f``
round-trips a float32 *fine* at actin's magnitudes: at |x| ~ 323 the ULP is
3.05e-5, so half-ULP is coarser than 1e-6 and the decimal uniquely determines
the value. Decimal formats fail at *small* magnitudes, where half-ULP drops
below the format's granularity -- below |x| ~ 32 for ``%.6f``. So the exposed
case is **chignolin** (per-group energies ~-8 to -32) and any near-zero
component (aromatic, native-contacts bias), not actin.

The general rule is that every fixed-decimal format has a magnitude-dependent
failure floor, so a precision that is correct for one case silently degrades
for another. ``float.hex()`` has no such floor: it is exact at every
magnitude, round-trips through ``float.fromhex``, and a one-ULP difference is
legible by eye in a diff.

Why a subprocess
----------------
Each case runs in a fresh child interpreter. Two reasons:

1. Isolation. A case cannot inherit engine state, RNG state or neighbour-list
   state from a previous case, so a difference is attributable to the build.
2. Build selection. ``--import-root`` lets you point at a specific build tree,
   which is the only reliable way to choose between two installed builds:
   ``PYTHONPATH`` alone does *not* work, because an editable
   scikit-build-core install registers a meta-path finder that precedes
   ``sys.path``, and ``pymcpu/__init__.py`` additionally globs
   ``mcpu_core*.so`` and takes the first match. The child records
   ``mcpu_core.__file__`` so a comparison can prove the two runs really did
   load different binaries.

What is deliberately NOT recorded
---------------------------------
* ``Integrator.step_stats()`` wholesale -- it mixes wall-clock ``*_ns`` and
  ``*_pct`` fields that differ run to run on identical builds. Only the
  deterministic integer subset is kept.
* ``Integrator.get_rng_state()`` mid-run -- it is non-const and resets the
  distribution caches (``src/pymcpu/Integrator.cpp:337-339``), discarding a
  cached spare Gaussian and thereby *changing the subsequent draw sequence*.
  Sampling it would perturb the very thing being measured.
* Anything from the ``derived`` sub-dict of ``neighbor_proxy_stats()``, which
  is floating-point rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: Bump when the record shape changes in a way that invalidates old references.
FORMAT_VERSION = 2

#: Integer counters from ``Context.neighbor_proxy_stats()`` worth fingerprinting.
#: These move when contact-list membership moves, which is the failure mode a
#: 1-ULP change in a squared distance actually produces
#: (``MuPotential.h:419`` -> ``pending_contact_add`` -> ``state.mu_contact_list``).
_PROXY_KEYS = (
    "mu_num_pair_distance_checks",
    "mu_num_pairs_within_rcut",
    "mu_eval_pair_calls",
    "hbond_num_candidates_iterated",
    "hbond_num_geom_checks",
    "neighbor_num_cell_visits",
    "elided_rigid_mm",
)

#: (label, pdb-relative-path, seed, steps). Actin carries the most weight: a
#: threshold straddle needs a dense neighbourhood to be likely, and
#: chignolin at 77 atoms is thin by comparison.
DEFAULT_CASES = (
    ("chignolin-s42", "pymcpu/data/1uao.pdb", 42, 2000),
    ("chignolin-s1337", "pymcpu/data/1uao.pdb", 1337, 2000),
    ("actin-s42", "examples/actin/input_pdb/acta.pdb", 42, 400),
    ("actin-s1337", "examples/actin/input_pdb/acta.pdb", 1337, 400),
)


# ---------------------------------------------------------------------------
# Child: runs exactly one case and prints one JSON object on stdout.
# ---------------------------------------------------------------------------
def extract_mu_by_kind(stats: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Pull the per-move-kind Mu counters out of ``Integrator.step_stats()``.

    Split out of the recorder so the guards below can be exercised directly.
    Every one of them exists because of a bug that made this function
    silently return something useless; see
    ``tests/packaging/test_arch_parity_oracle_guards.py``.
    """
    by_kind: dict[str, dict[str, int]] = {}
    raw_by_kind = stats.get("mu_by_kind")
    # bindings.cpp binds this as a py::list of dicts, each carrying its own
    # "kind" name ("pivot"/"kic"/"sidechain") -- NOT as a dict keyed by kind.
    # Accept both shapes: an earlier version of this function only handled the
    # dict form and therefore recorded {} silently, losing the one counter this
    # script exists to read.
    if isinstance(raw_by_kind, dict):
        raw_entries = [
            (kind, payload) for kind, payload in raw_by_kind.items()
        ]
    elif isinstance(raw_by_kind, (list, tuple)):
        raw_entries = [
            (payload.get("kind", f"kind{i}") if isinstance(payload, dict) else i,
             payload)
            for i, payload in enumerate(raw_by_kind)
        ]
    else:
        raise SystemExit(
            f"step_stats()['mu_by_kind'] has unexpected type "
            f"{type(raw_by_kind).__name__}; this harness cannot record the "
            f"contact-membership counter and would report a false PASS."
        )
    for kind, payload in raw_entries:
        if not isinstance(payload, dict):
            continue
        by_kind[str(kind)] = {
            k: int(v)
            for k, v in payload.items()
            # Integers only -- but note `ns` IS an integer, so an int-only
            # filter is not sufficient on its own. Wall-clock fields must be
            # excluded by NAME or every comparison reports a spurious
            # divergence in this section even when the trajectory is
            # bit-identical. (`avg_ns` is a float and would be dropped by the
            # int test; raw `ns` would not.)
            if isinstance(v, int)
            and not isinstance(v, bool)
            and k != "ns"
            and not k.endswith("_ns")
        }
    # `by_kind` being non-empty is NOT sufficient: entries that carry a "kind"
    # name but no integer counters produce {"pivot": {}}, which is truthy while
    # recording nothing. Require that at least one actual counter survived.
    if not any(counters for counters in by_kind.values()):
        raise SystemExit(
            "step_stats()['mu_by_kind'] yielded no per-kind counters "
            f"(parsed keys: {sorted(by_kind)}); refusing to record a "
            "fingerprint that silently omits eval_pair_nonzero."
        )
    return by_kind


def _run_case_in_child(pdb: str, seed: int, steps: int) -> dict[str, Any]:
    import mdtraj as md
    import numpy as np

    import pymcpu as mc
    from pymcpu import mcpu_core
    from pymcpu.forcefields.mcpu import MCPUForceField

    traj = md.load(pdb)
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    forcefield = MCPUForceField(heavy, param_set="mcpu08")
    system = forcefield.create_system(heavy.topology)

    integrator = mcpu_core.Integrator(temperature=0.6, step_size_rad=0.1)
    integrator.set_seed(seed)
    sim = mc.Simulation(heavy.topology, system, integrator)
    sim.context.set_positions((forcefield.coords[0] * 10.0).T.astype(np.float32))

    # Disable the periodic full recompute: it would resync `current_energy` and
    # mask exactly the incremental-path divergence we are trying to detect.
    sim.full_energy_every_steps = 10**9

    ctx = sim.context
    ctx.reset_neighbor_proxy_stats()
    before = ctx.energy_breakdown(weighted=False)
    # The WEIGHTED breakdown and the weight vector are recorded as well, and
    # this is not redundant. Until this was added, the fingerprint held only
    # `weighted=False` values, so the per-group outer weights entered it ONLY
    # through their effect on accept bits. That makes the oracle nearly blind
    # to a weight error: unlike a coordinate perturbation (which amplifies at
    # ~145 steps per e-folding), a wrong weight is
    # a STATIC offset. Its only route into the fingerprint is flipping a
    # Metropolis comparison, with per-step probability ~beta*dw*|dE_group|;
    # at one ULP of the Mu weight (2.98e-08) and T=0.6 that is ~5e-08..5e-07
    # per step, or ~1e-03 over all 4,800 steps this harness samples. So a
    # one-ULP weight regression would have been reported as PASS roughly 999
    # times in 1000.
    before_w = ctx.energy_breakdown(weighted=True)
    weights = {str(g): float(w).hex()
               for g, w in dict(ctx.get_energy_weights()).items()}

    sim.step(steps)

    after = ctx.energy_breakdown(weighted=False)
    after_w = ctx.energy_breakdown(weighted=True)
    bits = "".join(str(int(b)) for b in integrator.last_accept_bits())

    coords = np.ascontiguousarray(ctx.get_state().coords)
    coords_digest = hashlib.sha256(coords.view(np.uint8)).hexdigest()

    proxy = dict(ctx.neighbor_proxy_stats())

    # `eval_pair_nonzero` counts the epsilon-free `r2 <= g.contact_r2` compare
    # in eval_pair, so a +/-1 delta would be the signature of a
    # single contact-membership flip.
    #
    # IMPORTANT -- it reads 0 in the shipped configuration, so do NOT rely on
    # it. The default-enabled contact list routes the delta through
    # `calculateEnergyChange_clist`, while the flush that publishes these
    # counters lives in `calculateEnergyChange_fast`, now only a fallback for
    # moves that leave the dense grid or run under an energy mask. Measured on chignolin/300 steps: `eval_pair_calls`,
    # `pair_distance_checks`, `pairs_within_rcut` and `eval_pair_nonzero` are
    # all 0 by default, and count every pair under `MCPU_CONTACT_LIST=0`,
    # which sends every move through the all-pairs moved-vs-all delta. That
    # run checks the physics, not the trajectory: moved-vs-all sums the old
    # and new energies separately, so a move that changes no contact gets a
    # rounding-size delta instead of exactly 0, and a positive one draws a
    # Metropolis number and shifts the random stream (actin, seed 42: the
    # accept bits part at step 53). Use it as a SEPARATE diagnostic run; the
    # primary fingerprint must stay on the path users actually execute.
    #
    # What still carries signal in this section: `n_steps` and
    # `moved_atoms_sum`. The comparison's real witnesses are the accept-bit
    # stream, the hex-float energies, the coordinate hash and `move_stats`.
    stats = dict(integrator.step_stats())
    by_kind = extract_mu_by_kind(stats)
    # Whitelisted, because step_stats() mixes wall-clock fields throughout and
    # its keys are conditionally present (guarded on n_steps > 0 etc.), so a
    # key-set diff would produce false positives.
    step_ints = {
        k: int(stats[k])
        for k in (
            "n_steps", "n_valid_moves", "n_accepts", "moved_atoms_sum",
            "mu_eval_pair_calls",
        )
        if isinstance(stats.get(k), int) and not isinstance(stats.get(k), bool)
    }

    return {
        "pdb": os.path.basename(pdb),
        "seed": seed,
        "steps": steps,
        "n_atoms": int(system.get_num_atoms()),
        "n_residues": int(system.get_num_residues()),
        "energy_before": {str(k): float(v).hex() for k, v in before["by_group"].items()},
        "energy_after": {str(k): float(v).hex() for k, v in after["by_group"].items()},
        "raw_total_before": float(before["raw_total"]).hex(),
        "raw_total_after": float(after["raw_total"]).hex(),
        "weighted_before": {
            str(k): float(v).hex() for k, v in before_w["by_group"].items()
        },
        "weighted_after": {
            str(k): float(v).hex() for k, v in after_w["by_group"].items()
        },
        "weighted_total_before": float(before_w["weighted_total"]).hex(),
        "weighted_total_after": float(after_w["weighted_total"]).hex(),
        "energy_weights": weights,
        "accept_bits": bits,
        "accept_bits_sha256": hashlib.sha256(bits.encode()).hexdigest(),
        "n_accepts": bits.count("1"),
        "coords_sha256": coords_digest,
        "move_stats": {k: int(v) for k, v in dict(integrator.move_stats()).items()},
        "proxy_stats": {
            k: int(proxy[k]) for k in _PROXY_KEYS if isinstance(proxy.get(k), int)
        },
        "step_ints": step_ints,
        "mu_by_kind": by_kind,
        "mu_backend": ctx.mu_backend_name(),
        "hbond_backend": ctx.hbond_backend_name(),
    }


def _build_identity() -> dict[str, Any]:
    """Whatever the loaded extension can tell us about how it was built.

    ``build_info()`` is the function that reports the arch baseline, compiler
    and LTO state. It may not exist yet -- this script has to be usable to
    record a reference from a build that predates it, which is the whole point
    of recording a reference *before* changing anything.
    """
    import pymcpu
    from pymcpu import mcpu_core

    identity: dict[str, Any] = {
        "pymcpu_version": pymcpu.__version__,
        "mcpu_core_file": mcpu_core.__file__,
        "python": sys.version.split()[0],
    }
    # A sha256 of the .so is what actually proves two records came from two
    # builds. `mcpu_core.__file__` can be the same path with different content
    # after a rebuild in place, which is the normal case here.
    try:
        identity["so_sha256"] = hashlib.sha256(
            Path(mcpu_core.__file__).read_bytes()
        ).hexdigest()
    except OSError:
        identity["so_sha256"] = None
    # The emitted ISA, read from the binary rather than trusted from CMake.
    # -march is silently dropped for Debug builds. Zero zmm/kmask means AVX-512 is genuinely absent.
    try:
        dis = subprocess.run(
            ["objdump", "-d", mcpu_core.__file__],
            capture_output=True, text=True, check=False,
        ).stdout
        identity["isa_counts"] = {
            "zmm": dis.count("%zmm"),
            "kmask": sum(dis.count(f"%k{i}") for i in range(8)),
            "ymm": dis.count("%ymm"),
            "fma": dis.count("vfmadd") + dis.count("vfmsub"),
        }
    except (OSError, subprocess.SubprocessError):
        identity["isa_counts"] = None
    fn = getattr(mcpu_core, "build_info", None)
    identity["build_info"] = None  # an extension that predates build_info()
    if fn is not None:
        try:
            identity["build_info"] = {k: v for k, v in dict(fn()).items()}
        except Exception as exc:  # noqa: BLE001 -- diagnostic only
            identity["build_info"] = f"<raised {type(exc).__name__}: {exc}>"
    return identity


# ---------------------------------------------------------------------------
# Parent
# ---------------------------------------------------------------------------
def _child_env(import_root: str | None) -> dict[str, str]:
    """A child environment where the only variable is the build under test.

    Three classes of confound are removed here, each of which would otherwise
    be attributed to the compiler flag:

    1. **Engine env vars.** Many ``MCPU_*`` variables change engine
       behaviour, and several latch in ``static const bool`` lambdas read
       once per process. ``MCPU_CONTACT_LIST=0`` is the sharpest: it sends
       every Mu delta down the all-pairs moved-vs-all scan instead of the
       contact list. All are cleared; the ones we need are then set
       explicitly.
    2. **Parameter resolution.** ``ensure_params`` has six steps, and two of
       them key off ``PACKAGE_ROOT``. If one build resolves parameters from a
       wheel and the other from the repo tree, the *potentials* could differ.
       ``MCPU_PARAMS_DIR`` is step 1 and outranks everything, so pinning it
       makes the parameter set a constant.
    3. **BLAS threading.** numpy is in the force-field build path, and a
       different thread count can reorder a reduction -- changing the
       parameter matrices handed to the potentials before a single MC step
       runs.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCPU_")}

    # Pin the parameter set. Resolve it once in the parent so both children
    # get a byte-identical root rather than each racing to materialize one.
    try:
        sys.path.insert(0, str(ROOT))
        from pymcpu.params import ensure_params

        env["MCPU_PARAMS_DIR"] = str(ensure_params("mcpu08"))
    except Exception:  # noqa: BLE001 -- fall back to the child's own resolution
        pass
    env["MCPU_NO_DOWNLOAD"] = "1"

    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    if import_root:
        # Beat the editable install's meta-path finder. The finder is
        # registered by a .pth in USER site-packages, so disabling user site
        # stops it from ever being processed -- killing both the meta-path
        # finder and its path injection in one move. PYTHONPATH then supplies
        # the chosen build plus the dependencies.
        env["PYTHONNOUSERSITE"] = "1"
        deps = [p for p in sys.path if p.endswith("site-packages")]
        # ROOT is required, not optional: disabling user site also removes the
        # editable install's path injection, so the repo's own Python layer
        # would be unimportable. Order matters -- import_root first so its
        # .so wins, then ROOT for the Python package, then the dependencies.
        #
        # import_root holds ONLY pymcpu/mcpu_core*.so and no __init__.py, so
        # CPython treats it as a namespace portion and discards it once it
        # finds the real package in ROOT. `pymcpu` therefore resolves to the
        # repo tree in both runs -- isolating the .so and nothing else -- and
        # `from . import mcpu_core` falls through to _find_mcpu_core_so(),
        # which now globs exactly one candidate.
        env["PYTHONPATH"] = os.pathsep.join([import_root, str(ROOT), *deps])
    return env


def _spawn(case: tuple[str, str, int, int], import_root: str | None) -> dict[str, Any]:
    label, rel_pdb, seed, steps = case
    pdb = ROOT / rel_pdb
    if not pdb.exists():
        return {"label": label, "skipped": f"missing {rel_pdb}"}

    env = _child_env(import_root)

    proc = subprocess.run(
        [sys.executable, __file__, "--_child", str(pdb), str(seed), str(steps)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(ROOT),
        check=False,
    )
    if proc.returncode != 0:
        return {
            "label": label,
            "failed": f"child exited {proc.returncode}",
            "stderr": proc.stderr[-2000:],
        }
    # Anything else the child prints (warnings, opt-in diagnostics) would mix
    # with the payload, so it is framed rather than assumed to be the whole of
    # stdout.
    marker = "@@ARCH_PARITY_JSON@@"
    _, _, payload = proc.stdout.rpartition(marker)
    try:
        record = json.loads(payload)
    except json.JSONDecodeError as exc:
        return {"label": label, "failed": f"unparseable child output: {exc}"}
    record["label"] = label
    return record


def _record(import_root: str | None, cases) -> dict[str, Any]:
    # The identity probe MUST run in the same environment as the cases, or it
    # reports the default build while the cases ran against --import-root --
    # producing a record that names the wrong binary.
    ident_proc = subprocess.run(
        [sys.executable, __file__, "--_identity"],
        capture_output=True,
        text=True,
        env=_child_env(import_root),
        cwd=str(ROOT),
        check=False,
    )
    marker = "@@ARCH_PARITY_JSON@@"
    _, _, payload = ident_proc.stdout.rpartition(marker)
    try:
        identity = json.loads(payload)
    except json.JSONDecodeError:
        identity = {"error": ident_proc.stderr[-1000:]}

    return {
        "format_version": FORMAT_VERSION,
        "build": identity,
        "cases": [_spawn(c, import_root) for c in cases],
    }


#: Fields whose difference is a *physics* difference. Everything else in a
#: record is identity or metadata and is reported but never fails a comparison.
_PHYSICS_FIELDS = (
    "n_atoms",
    "n_residues",
    "energy_before",
    "energy_after",
    "raw_total_before",
    "raw_total_after",
    # Weighted quantities and the weight vector itself. Without these the
    # fingerprint could not witness a per-group outer weight at all.
    "weighted_before",
    "weighted_after",
    "weighted_total_before",
    "weighted_total_after",
    "energy_weights",
    "accept_bits",
    "coords_sha256",
    "move_stats",
    "step_ints",
    "mu_backend",
    "hbond_backend",
)

#: Internal work counters: how many candidates, cells, geometry checks and
#: rebuilds the engine spent getting to the answer. They are reported when
#: they differ but do not fail the comparison, so a change that reaches the
#: same energies, accept bits and coordinates with less work still passes.
#: ``proxy_stats`` and ``mu_by_kind`` are informational as a whole;
#: in ``step_ints`` only ``_STEP_STRICT_KEYS`` fail the comparison.
_WORK_COUNTER_FIELDS = ("proxy_stats", "mu_by_kind")
_STEP_STRICT_KEYS = ("n_steps", "n_valid_moves", "n_accepts", "moved_atoms_sum")


def _split_step_ints(case: dict[str, Any]) -> tuple[dict, dict]:
    """``case``'s ``step_ints`` split into (strict, work-counter) parts.

    The strict part always holds every ``_STEP_STRICT_KEYS`` entry, ``None``
    when the case did not record it, so a step total missing on one side is a
    divergence. Every other key is a work counter.
    """
    step = case.get("step_ints") or {}
    strict = {k: step.get(k) for k in _STEP_STRICT_KEYS}
    work = {k: v for k, v in step.items() if k not in strict}
    return strict, work


def _work_counter_diffs(ref_case: dict[str, Any], cur_case: dict[str, Any]) -> list[str]:
    """One line per work counter that differs between the two cases."""
    lines = []
    pairs = [(f, ref_case.get(f) or {}, cur_case.get(f) or {}) for f in _WORK_COUNTER_FIELDS]
    pairs.append(("step_ints", _split_step_ints(ref_case)[1],
                  _split_step_ints(cur_case)[1]))
    for field, ra, rb in pairs:
        for key in sorted(set(ra) | set(rb)):
            if ra.get(key) != rb.get(key):
                lines.append(f"{field}[{key}]: ref {ra.get(key)} vs cur {rb.get(key)}")
    return lines


def _compare(ref: dict[str, Any], cur: dict[str, Any]) -> int:
    if ref.get("format_version") != cur.get("format_version"):
        print(
            f"FAIL: format_version {ref.get('format_version')} != "
            f"{cur.get('format_version')}; re-record the reference",
            file=sys.stderr,
        )
        return 2

    rb, cb = ref.get("build") or {}, cur.get("build") or {}
    print(f"reference build : {rb.get('mcpu_core_file')}")
    print(f"current build   : {cb.get('mcpu_core_file')}")
    for label, side in (("reference", rb), ("current", cb)):
        info = side.get("build_info")
        arch = "<build_info() absent in this build>"
        if isinstance(info, dict):
            arch = info.get("arch") or info.get("MCPU_ARCH") or "?"
        isa = side.get("isa_counts") or {}
        isa_txt = (
            f"zmm={isa.get('zmm')} kmask={isa.get('kmask')} "
            f"ymm={isa.get('ymm')} fma={isa.get('fma')}"
            if isa else "isa=<unknown>"
        )
        print(f"  {label:9} arch={arch}  {isa_txt}")
        print(f"  {label:9} so_sha256={str(side.get('so_sha256'))[:16]}")

    # The digest -- not the path -- is what proves two builds were compared.
    # A rebuild in place keeps the path and changes the content, which is the
    # normal workflow here, so a path comparison would give a false sense of
    # having tested something.
    # A build can DECLARE a compile flag that was never applied. This happened
    # twice while developing this script: a CMake `if(MCPU_FP_CONTRACT AND ...)`
    # treated the literal string "off" as boolean false, so the cache recorded
    # MCPU_FP_CONTRACT=off while the compiler line carried no -ffp-contract at
    # all. The resulting comparison looked like "off changes nothing" when in
    # truth nothing had been changed. The declared setting is therefore never
    # trusted on its own -- it is cross-checked against the FMA count, which is
    # the observable consequence of contraction.
    r_isa = (rb.get("isa_counts") or {})
    c_isa = (cb.get("isa_counts") or {})
    r_fma, c_fma = r_isa.get("fma"), c_isa.get("fma")

    def _declared_fp(side: dict[str, Any]) -> str | None:
        info = side.get("build_info")
        if not isinstance(info, dict):
            return None
        # build_info() puts this in its own top-level "fp" section, NOT under
        # "build" -- reading the wrong key here would make this whole guard a
        # silent no-op, which is the failure mode it exists to catch.
        fp = info.get("fp")
        if isinstance(fp, dict) and fp.get("fp_contract") is not None:
            return str(fp["fp_contract"])
        return None

    r_fp, c_fp = _declared_fp(rb), _declared_fp(cb)
    if r_fp is not None and c_fp is not None and r_fma is not None:
        if r_fp != c_fp and r_fma == c_fma:
            print()
            print(
                f"WARNING: the two builds declare DIFFERENT FP contraction "
                f"(ref={r_fp!r} vs cur={c_fp!r}) yet emit the SAME number of "
                f"FMA instructions ({r_fma}). The flag was almost certainly "
                f"not applied to one of them, so any 'no difference' result "
                f"below measures nothing. Verify the compiler command line "
                f"before believing this comparison."
            )
        elif r_fp == c_fp and r_fma != c_fma:
            print()
            print(
                f"NOTE: both builds declare FP contraction {r_fp!r} but emit "
                f"different FMA counts ({r_fma} vs {c_fma}); the difference "
                f"comes from something other than this flag (compiler version, "
                f"-march, or a source change)."
            )

    ref_d, cur_d = rb.get("so_sha256"), cb.get("so_sha256")
    same_build = bool(ref_d) and ref_d == cur_d
    if same_build:
        print()
        print(
            "NOTE: both records have the SAME .so digest. This is a valid "
            "self-consistency (null) test -- it proves the oracle is "
            "deterministic -- but it proves nothing about a flag change."
        )
    print()

    ref_cases = {c.get("label"): c for c in ref.get("cases", [])}
    failures = 0
    harness_errors = 0
    compared = 0
    skipped = 0
    work_only = 0
    for cur_case in cur.get("cases", []):
        label = cur_case.get("label")
        ref_case = ref_cases.get(label)
        if ref_case is None:
            print(f"  {label:18} SKIP  (absent from reference)")
            skipped += 1
            continue
        if "skipped" in cur_case or "skipped" in ref_case:
            print(f"  {label:18} SKIP  ({cur_case.get('skipped') or ref_case.get('skipped')})")
            skipped += 1
            continue
        if "failed" in cur_case:
            print(f"  {label:18} ERROR {cur_case['failed']}")
            print(cur_case.get("stderr", ""))
            harness_errors += 1
            continue

        compared += 1
        diffs = [
            f for f in _PHYSICS_FIELDS
            if f != "step_ints" and ref_case.get(f) != cur_case.get(f)
        ]
        if _split_step_ints(ref_case)[0] != _split_step_ints(cur_case)[0]:
            diffs.append("step_ints")
        work = _work_counter_diffs(ref_case, cur_case)
        if not diffs:
            print(f"  {label:18} IDENTICAL  ({cur_case['steps']} steps, "
                  f"{cur_case['n_atoms']} atoms, {cur_case['n_accepts']} accepts)")
            if work:
                work_only += 1
                print("      work counters differ (informational, not a failure):")
                for line in work:
                    print(f"        {line}")
            continue

        failures += 1
        print(f"  {label:18} DIVERGED in {', '.join(diffs)}")

        # Localize: the accept-bit stream is the most direct witness, and its
        # first differing index IS the MC step at which the builds parted.
        a, b = ref_case.get("accept_bits", ""), cur_case.get("accept_bits", "")
        if a != b:
            if len(a) != len(b):
                print(f"      accept-bit length {len(a)} vs {len(b)}")
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            if first is not None:
                lo = max(0, first - 12)
                print(f"      first divergence at MC step {first} "
                      f"({first / max(len(a), 1):.1%} through the run)")
                print(f"        ref ...{a[lo:first]}[{a[first]}]{a[first + 1:first + 13]}...")
                print(f"        cur ...{b[lo:first]}[{b[first]}]{b[first + 1:first + 13]}...")
                print(f"      accepts: ref {a.count('1')} vs cur {b.count('1')}")
        for field in ("energy_before", "energy_after"):
            ra, rb = ref_case.get(field, {}), cur_case.get(field, {})
            for group in sorted(set(ra) | set(rb)):
                if ra.get(group) != rb.get(group):
                    print(f"      {field} group {group}:")
                    print(f"        ref {ra.get(group)}")
                    print(f"        cur {rb.get(group)}")
        for field in ("move_stats", "step_ints"):
            ra, rb = ref_case.get(field, {}), cur_case.get(field, {})
            if field == "step_ints":
                ra = _split_step_ints(ref_case)[0]
                rb = _split_step_ints(cur_case)[0]
            for key in sorted(set(ra) | set(rb)):
                if ra.get(key) != rb.get(key):
                    print(f"      {field}[{key}]: ref {ra.get(key)} vs cur {rb.get(key)}")
        for line in work:
            print(f"      (work counter) {line}")

    print()
    # Exit codes are distinct on purpose. Every existing parity_*.py conflates
    # "the physics diverged" with "the harness broke", which is how a broken
    # oracle reads as a clean result.
    #   0 = bit-identical      1 = genuine divergence      2 = harness failure
    # A verdict of PASS having compared NOTHING is the worst possible output,
    # and it was reachable: every SKIP path (case absent from the reference, or
    # `skipped` because an input PDB was missing, _spawn) bypasses both
    # counters below, so four skips printed "PASS: every case bit-identical".
    # Coverage is therefore part of the verdict, not a footnote.
    if compared == 0:
        print("HARNESS ERROR: zero cases were actually compared "
              f"({skipped} skipped). This is not a PASS -- it is a no-op. "
              "Check that the reference was recorded with the same case set "
              "and that every input PDB is present.")
        return 2
    print(f"compared {compared} case(s); skipped {skipped}")
    if harness_errors:
        print(f"HARNESS ERROR: {harness_errors} case(s) failed to run")
        print("This is not a physics result. Fix the harness and re-run.")
        return 2
    if failures:
        print(f"FAIL: {failures} case(s) diverged")
        print(
            "Reading the result: an `energy_before` difference means the "
            "builds disagree before a single MC step -- a static evaluation "
            "difference. A difference only in `accept_bits`/`energy_after` "
            "means they agree statically and part during sampling, which "
            "points at the incremental delta path.\n"
            "CAUTION: mu_by_kind's pair counters (eval_pair_nonzero and "
            "friends) read 0 unless MCPU_CONTACT_LIST=0, because the shipped "
            "contact-list delta path bypasses their instrumentation -- so "
            "their agreement is not evidence of anything. The witnesses that "
            "matter are accept_bits, the hex-float energies and coords_sha256."
        )
        return 1
    if work_only:
        print(f"note: {work_only} case(s) reached identical results with "
              "different internal work counters (informational)")
    if same_build:
        print("PASS: every case bit-identical (same build -- null test only)")
    else:
        print("PASS: every case bit-identical across two different builds")
    return 0


def main() -> int:
    # Child modes are internal; they print a framed JSON payload and exit.
    if "--_child" in sys.argv:
        idx = sys.argv.index("--_child")
        pdb, seed, steps = sys.argv[idx + 1], int(sys.argv[idx + 2]), int(sys.argv[idx + 3])
        record = _run_case_in_child(pdb, seed, steps)
        print("@@ARCH_PARITY_JSON@@" + json.dumps(record))
        return 0
    if "--_identity" in sys.argv:
        print("@@ARCH_PARITY_JSON@@" + json.dumps(_build_identity()))
        return 0

    parser = argparse.ArgumentParser(
        description="Bit-exact cross-build fingerprint for pyMCPU.",
        epilog="Record from one build, then compare from another. Same compiler "
               "for both, or you are measuring the compiler and not the flag.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--out", metavar="PATH", help="record a reference to PATH")
    mode.add_argument("--compare", metavar="PATH", help="compare this build against PATH")
    parser.add_argument(
        "--import-root",
        metavar="DIR",
        help="import pymcpu from DIR instead of the default (selects a build)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="chignolin only, fewer steps -- for checking the harness itself",
    )
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.quick:
        cases = tuple(
            (label, pdb, seed, 300) for label, pdb, seed, _ in DEFAULT_CASES[:2]
        )

    current = _record(args.import_root, cases)

    if args.out:
        Path(args.out).write_text(json.dumps(current, indent=1, sort_keys=True))
        n_ok = sum(1 for c in current["cases"] if "accept_bits" in c)
        print(f"wrote {args.out}  ({n_ok}/{len(cases)} cases recorded)")
        for case in current["cases"]:
            if "accept_bits" in case:
                print(f"  {case['label']:18} {case['n_atoms']:5} atoms  "
                      f"{case['steps']:5} steps  {case['n_accepts']:5} accepts  "
                      f"E={case['raw_total_after']}")
            else:
                print(f"  {case['label']:18} {case.get('skipped') or case.get('failed')}")
        return 0

    ref = json.loads(Path(args.compare).read_text())
    return _compare(ref, current)


if __name__ == "__main__":
    raise SystemExit(main())
