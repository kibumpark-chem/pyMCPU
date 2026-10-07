#!/usr/bin/env python3
"""Accept or reject a pyMCPU build that is *correct but not bit-identical*.

``scripts/arch_parity_dump.py`` proves two builds are the same to the last
bit. That is the right gate for refactors, and the wrong gate for speedups
that legitimately change float rounding (a different summation order, a
grid-based recompute, a faster root refinement, another compiler): those
flip one Metropolis decision somewhere and the trajectories diverge, so a
step-by-step comparison says nothing. This script applies the tolerance
criteria agreed for such changes instead::

    python scripts/tolerance_check.py --import-root A --import-root B [--mode quick|full]

Build A is the reference (normally current main), build B the candidate.
Every case runs in a fresh child interpreter, exactly as arch_parity_dump
does, so the only variable is the binary that ``--import-root`` selects.

The four checks
---------------
1. ``static``   Per-group energies (``energy_breakdown``, unweighted and
   weighted, plus the totals) on fixed structures: chignolin, actin, every
   ``--pdb`` structure, actin with an energy mask, and KORP on actin. Pass: ``|a-b| <= max(rtol*max(|a|,|b|), atol)`` with
   rtol=1e-5 and atol=1e-4 energy units.

   Why atol=1e-4: the per-term relative test is meaningless for a term near
   zero (the clash group is exactly 0, aromatic and some torsion groups are
   O(1)). 1e-4 only takes over below |E| = atol/rtol = 10. It is
   (a) far above rounding noise for such a term -- the terms are float32
   (``Potential::calculateEnergy`` returns float) and a near-zero term is a
   sum of O(10-100) contributions of O(1), so reordering it moves it by
   ~sqrt(n)*6e-8 ~ 1e-6, i.e. 1e-4 is ~100x headroom; and (b) physically
   negligible -- at the production temperature T=0.6 it changes a Boltzmann
   factor by 1.7e-4, well below what sampling can resolve.

2. ``running``  Long runs with no intermediate full recompute; at the end
   ``|current_energy - calculate_total_energy(-1)| <= max(1e-3, 1e-5*|E|)``
   for EACH build (a self-consistency check of the delta paths, not an
   A-vs-B one).

   Energy sums and the running total are double, so a correct build stays
   within about 1e-10 of a full recompute (actin: 3e-11 after 1e7 steps;
   KORP: exactly 0) and passes either criterion with room to spare. The
   1e-5*|E| allowance is for reference builds from before that change,
   which accumulated in float32 and random-walked past a flat 1e-3 (actin
   8.1e-3 after 1M steps, KORP actin 9.8e-3 after 20k), still below 1e-5
   relative. A delta-path bug is systematic and grows linearly instead --
   see the validation notes for what it looks like.
   ``--running-rtol 0`` restores the flat absolute criterion.
   Cases: the ``--proteins`` structures (default chignolin and actin) with
   the default move mix, actin under an energy mask
   (masked runs bypass the contact list), and KORP on actin.

3. ``sampling`` Several independent seeds per build; per seed the mean
   energy, the native-contact fraction Q (``NativeContactsCV``, CA contacts
   of the start structure) and the acceptance rate of each move kind, all
   after a burn-in. Build B uses *different* seeds from build A (offset
   ``--seed-offset``), so main-vs-main is a genuine test of the statistics
   rather than a bit-identical replay. Standard error per build is the
   larger of the block-average SE (10 blocks per seed, pooled) and the
   seed-to-seed SE; pass if ``|mean_A - mean_B| <= z * sqrt(SE_A^2+SE_B^2)``
   (z=3.0 in full mode, 3.5 in quick mode). About 10 comparisons run at
   once and the SE comes from only 8 seeds, so the threshold has to carry a
   multiple-comparison margin: with 4 seeds x 250k steps, main-vs-main
   reached z=2.78 on one acceptance rate. Quick-mode runs are short and
   still relaxing from the start structure, hence the higher z there. The effect size is reported as the
   difference in units of the observable's per-sample standard deviation.

4. ``korp``     The compiled KORP term against the reference ``korpe``
   energies shipped with the map bundle (``$KORP_MAP_PATH``, default
   ``~/.cache/pymcpu/korp/Korp6Dv1/korp6Dv1.bin``): each build within 1e-6
   relative (the repo's test tolerance; main currently sits at ~4e-8), and
   the two builds within the static criterion. Skipped with a note when the
   map is absent.

Structures
----------
Chignolin (``examples/data/1uao.pdb``) and actin
(``examples/actin/input_pdb/acta.pdb``) ship with the repository and are
always used, read from the reference build's tree. Pass more with ``--pdb``
(repeatable; the file stem becomes the label, usable in ``--proteins``).
The tolerances were validated on T4 lysozyme and four AlphaFold models in
the 150-450 residue range that matters for production runs. They are not
committed (CC-BY 4.0 / wwPDB files of 0.2-0.5 MB each); fetch them with::

    mkdir -p ~/tolcheck && cd ~/tolcheck
    curl -sLO https://files.rcsb.org/download/2LZM.pdb            # T4L, 164 res
    for u in P00918 P00338 P00558; do                              # CA2, LDH-A, PGK1
        curl -sLO https://alphafold.ebi.ac.uk/files/AF-$u-F1-model_v4.pdb
    done

The validation runs used chain A heavy atoms, first altloc, with AlphaFold
terminal tails of pLDDT < 50 trimmed (CA2 259, LDH-A 332, PGK1 417
residues); the checker itself accepts any structure the force field loads.
To repeat the validated quick configuration::

    python scripts/tolerance_check.py --import-root MAIN --import-root NEW \
        --pdb ~/tolcheck/2LZM.pdb --proteins 2LZM,actin

Modes: ``quick`` (PR check) and ``full`` (release); every count is
overridable (--running-steps, --sampling-steps, --seeds, ...). ``--save``
writes all raw records; ``--ref FILE`` reuses build A's records from an
earlier ``--save`` of the same configuration instead of re-running A.
Exit status: 0 all PASS, 1 some FAIL, 2 a case crashed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MARKER = "@@TOLERANCE_JSON@@"
DEFAULT_KORP_MAP = Path.home() / ".cache/pymcpu/korp/Korp6Dv1/korp6Dv1.bin"

KORP_REFERENCE = {  # korpe_gcc --only_score, see tests/physics/forces/test_korp_reference_parity.py
    "CASP12DCsel20/T0860D1.pdb": -3693.586739,
    "CASP12DCsel20/T0860D1_s026m1.pdb": -1004.099531,
    "CASP12DCsel20/T0860D1_s119m1.pdb": -2444.948077,
    "rcd6/1CEO.pdb": -11463.957486,
}
KORP_GROUP = 7
MOVE_KINDS = ("pivot", "kic", "sc", "rotamer", "rama_pivot")

MODES = {
    #            running steps, korp running steps, sampling steps/seed, seeds, sample_every
    "quick": dict(running_steps=50_000, korp_running_steps=5_000,
                  sampling_steps=150_000, seeds=8, sample_every=1_000, z=3.5),
    "full": dict(running_steps=200_000, korp_running_steps=20_000,
                 sampling_steps=2_000_000, seeds=8, sample_every=1_000, z=3.0),
}


# --------------------------------------------------------------------------
# child side
# --------------------------------------------------------------------------
def _build_sim(task: dict[str, Any]):
    import warnings

    import mdtraj as md
    import numpy as np

    import pymcpu as mc
    from pymcpu import mcpu_core

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traj = md.load(task["pdb"])
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    korp = task.get("ff", "mcpu08") == "korp"
    if korp:
        from pymcpu.forcefields.korp import KORPForceField
        ff = KORPForceField(heavy)
        system = ff.create_system(heavy.topology)
        top = ff.output_topology
    else:
        from pymcpu.forcefields.mcpu import MCPUForceField
        ff = MCPUForceField(heavy, param_set="mcpu08")
        system = ff.create_system(heavy.topology)
        top = heavy.topology
    if task.get("mask"):
        system.set_energy_ignored_residues(list(range(int(task["mask"]))), "ignore_all")
    integ = mcpu_core.Integrator(temperature=float(task.get("T", 0.6)),
                                 step_size_rad=0.05 if korp else 0.1)
    integ.set_seed(int(task.get("seed", 1)))
    sim = mc.Simulation(top, system, integ)
    sim.context.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    if korp:
        ff.apply_energy_weights(sim.context)
        integ.set_move_weights(0.5, 0.5, 0.0)
    return ff, system, integ, sim


def _breakdown(ctx) -> dict[str, Any]:
    u = ctx.energy_breakdown(weighted=False)
    w = ctx.energy_breakdown(weighted=True)
    return {
        "unweighted": {str(k): float(v) for k, v in u["by_group"].items()},
        "weighted": {str(k): float(v) for k, v in w["by_group"].items()},
        "raw_total": float(u["raw_total"]),
        "weighted_total": float(w["weighted_total"]),
    }


def _child(task: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    from pymcpu import mcpu_core

    t0 = time.perf_counter()
    ff, system, integ, sim = _build_sim(task)
    ctx = sim.context
    out: dict[str, Any] = {"so": mcpu_core.__file__, "n_atoms": int(system.get_num_atoms()),
                           "n_residues": int(system.get_num_residues())}
    kind = task["kind"]
    if kind in ("static", "korp"):
        out.update(_breakdown(ctx))
    elif kind == "running":
        sim.full_energy_every_steps = 10**9    # never recompute inside the run
        chunk = 10_000                         # one production exchange interval
        done = 0
        t1 = time.perf_counter()
        while done < task["steps"]:
            n = min(chunk, task["steps"] - done)
            sim.step(n)
            done += n
        out["mc_seconds"] = time.perf_counter() - t1
        out["running"] = float(ctx.get_state().current_energy)
        out["full"] = float(ctx.calculate_total_energy(-1))
        out["steps"] = done
    elif kind == "sampling":
        from pymcpu.sampling.collective_variables import NativeContactsCV, build_ca_index
        ca = build_ca_index(ff)
        coords0 = np.asarray(ctx.get_state().coords, dtype=np.float64)
        cv = NativeContactsCV(ca, coords0[:, ca].T.copy())
        every = int(task["sample_every"])
        # production recomputes the full energy every 1-10 exchanges of 10k steps
        sim.full_energy_every_steps = int(task.get("recompute_every", 50_000))
        n_samples = int(task["steps"]) // every
        n_burn = int(n_samples * float(task.get("burn_frac", 0.2)))
        energies, qs, stats = [], [], []
        t1 = time.perf_counter()
        for k in range(n_samples):
            sim.step(every)
            if k >= n_burn:
                st = ctx.get_state()
                energies.append(float(st.current_energy))
                qs.append(float(cv.compute_Q(np.asarray(st.coords, dtype=np.float64))))
            if k == n_burn - 1 or k >= n_burn:
                ms = dict(integ.move_stats())
                stats.append({m: [int(ms.get(f"num_propose_{m}", 0)),
                                  int(ms.get(f"num_accept_{m}", 0))] for m in MOVE_KINDS})
        out["mc_seconds"] = time.perf_counter() - t1
        out.update(energy=energies, q=qs, move_stats=stats, steps=int(task["steps"]))
    out["wall_seconds"] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------
# parent side: environment, scheduling
# --------------------------------------------------------------------------
def _child_env(import_root: str, params_dir: str | None, korp_map: Path | None) -> dict[str, str]:
    """Same isolation as arch_parity_dump._child_env: no MCPU_* knobs, one
    parameter directory for both builds, single-threaded BLAS."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("MCPU_")}
    if params_dir:
        env["MCPU_PARAMS_DIR"] = params_dir
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    deps = [p for p in sys.path if p.endswith("site-packages")]
    env["PYTHONPATH"] = os.pathsep.join([import_root, *deps])
    if korp_map is not None:
        env["KORP_MAP_PATH"] = str(korp_map)
    return env


def _resolve_params_dir(import_root: str) -> str | None:
    proc = subprocess.run(
        [sys.executable, "-c",
         "from pymcpu.params import ensure_params; print(ensure_params('mcpu08'))"],
        capture_output=True, text=True, env=_child_env(import_root, None, None), check=False)
    return proc.stdout.strip().splitlines()[-1] if proc.returncode == 0 and proc.stdout.strip() else None


def _parse_cpus(spec: str | None) -> list[int]:
    if not spec:
        return sorted(os.sched_getaffinity(0))
    cpus: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def _run_tasks(jobs: list[tuple[str, dict[str, Any]]], envs: dict[str, dict[str, str]],
               cpus: list[int], verbose: bool) -> list[dict[str, Any]]:
    """Run (build, task) pairs, one child per CPU, longest first."""
    free: queue.Queue[int] = queue.Queue()
    for c in cpus:
        free.put(c)

    def one(job):
        build, task = job
        cpu = free.get()
        try:
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--_child", json.dumps(task)],
                capture_output=True, text=True, env=envs[build], check=False,
                preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
            rec: dict[str, Any] = {"build": build, "task": task}
            if proc.returncode != 0:
                rec["failed"] = f"child exited {proc.returncode}: {proc.stderr[-1500:]}"
            else:
                rec.update(json.loads(proc.stdout.rpartition(MARKER)[2]))
            if verbose:
                print(f"  [{build}] {task['kind']:8s} {task['label']:24s} "
                      f"{time.perf_counter() - t0:7.1f} s cpu{cpu}"
                      + ("  FAILED" if "failed" in rec else ""), flush=True)
            return rec
        finally:
            free.put(cpu)

    cost = {"sampling": 3, "running": 2, "static": 1, "korp": 1}
    order = sorted(jobs, key=lambda j: -cost[j[1]["kind"]] * j[1].get("steps", 1))
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        return list(pool.map(one, order))


def _plan(args, korp_map: Path | None) -> dict[str, list[dict[str, Any]]]:
    """Tasks per build. 'A' and 'B' differ only in sampling seeds."""
    root_a = Path(args.import_root[0])
    actin = root_a / "examples/actin/input_pdb/acta.pdb"
    structures = {"chignolin": root_a / "examples/data/1uao.pdb", "actin": actin}
    for p in args.pdb:
        structures[Path(p).stem] = Path(p).expanduser()
    missing = [str(v) for v in structures.values() if not v.exists()]
    if missing:
        raise SystemExit(f"structure(s) not found: {', '.join(missing)}")
    unknown = [k for k in args.proteins.split(",") if k not in structures]
    if unknown:
        raise SystemExit(f"--proteins names no structure: {', '.join(unknown)} "
                         f"(known: {', '.join(structures)})")
    dyn = {k: structures[k] for k in args.proteins.split(",")}

    sel = set(args.only.split(","))
    static, running, sampling, korp = [], [], [], []
    if "static" in sel:
        for name, pdb in structures.items():
            static.append(dict(kind="static", label=name, pdb=str(pdb)))
        static.append(dict(kind="static", label="actin+mask30", pdb=str(actin), mask=30))
        if korp_map:
            static.append(dict(kind="static", label="actin/korp", pdb=str(actin), ff="korp"))
    if "running" in sel:
        for name, pdb in dyn.items():
            running.append(dict(kind="running", label=name, pdb=str(pdb), seed=11,
                                steps=args.running_steps))
        running.append(dict(kind="running", label="actin+mask30", pdb=str(actin), seed=11,
                            mask=30, steps=args.running_steps))
        if korp_map:
            running.append(dict(kind="running", label="actin/korp", pdb=str(actin), ff="korp",
                                seed=11, steps=args.korp_running_steps))
    if "sampling" in sel:
        for name, pdb in dyn.items():
            for i in range(args.seeds):
                sampling.append(dict(kind="sampling", label=f"{name}/s{i}", group=name,
                                     pdb=str(pdb), seed=101 + i, steps=args.sampling_steps,
                                     sample_every=args.sample_every, T=args.temperature,
                                     recompute_every=args.recompute_every))
    if "korp" in sel and korp_map:
        for rel in KORP_REFERENCE:
            p = korp_map.parent / rel
            if p.exists():
                korp.append(dict(kind="korp", label=rel, pdb=str(p), ff="korp"))
    a = static + running + sampling + korp
    b = [dict(t, seed=t["seed"] + args.seed_offset) if t["kind"] == "sampling" else dict(t)
         for t in a]
    return {"A": a, "B": b}


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, check: str, case: str, ok: bool, detail: str) -> None:
        self.rows.append((check, case, ok, detail))

    def summary(self) -> dict[str, list[int]]:
        s: dict[str, list[int]] = {}
        for check, _, ok, _ in self.rows:
            s.setdefault(check, [0, 0])[0 if ok else 1] += 1
        return s


def _close(a: float, b: float, rtol: float, atol: float) -> tuple[bool, float]:
    d = abs(a - b)
    return d <= max(rtol * max(abs(a), abs(b)), atol), d


def _cmp_energies(rep, check, case, ra, rb, rtol, atol):
    worst, worst_key, ok_all = 0.0, "", True
    for which in ("unweighted", "weighted"):
        for g in sorted(set(ra[which]) | set(rb[which]), key=lambda x: int(x)):
            a, b = ra[which].get(g, 0.0), rb[which].get(g, 0.0)
            ok, d = _close(a, b, rtol, atol)
            rel = d / max(abs(a), abs(b), atol / rtol)
            if not ok or rel > worst:
                worst, worst_key = max(worst, rel), f"{which}[{g}] {a:.6g} vs {b:.6g}"
            ok_all &= ok
    for tot in ("raw_total", "weighted_total"):
        ok, d = _close(ra[tot], rb[tot], rtol, atol)
        ok_all &= ok
        rel = d / max(abs(ra[tot]), abs(rb[tot]), atol / rtol)
        if rel > worst:
            worst, worst_key = rel, f"{tot} {ra[tot]:.6g} vs {rb[tot]:.6g}"
    rep.add(check, case, ok_all, f"max scaled diff {worst:.2e} at {worst_key}")


def _block_stats(series: list[list[float]], nb: int = 10):
    """Per-seed series -> (mean, se, per-sample sd)."""
    import numpy as np
    seed_means, block_se2, pooled = [], [], []
    for x in series:
        x = np.asarray(x, dtype=np.float64)
        if len(x) < nb:
            continue
        blocks = np.array([b.mean() for b in np.array_split(x, nb)])
        seed_means.append(blocks.mean())
        block_se2.append(blocks.var(ddof=1) / nb)
        pooled.append(x)
    n = len(seed_means)
    mean = float(np.mean(seed_means))
    se_block = math.sqrt(sum(block_se2)) / n
    se_seed = float(np.std(seed_means, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
    sd = float(np.concatenate(pooled).std(ddof=1))
    return mean, max(se_block, se_seed), sd


def _acceptance_series(stats: list[dict[str, list[int]]], kind: str, nb: int = 10):
    """Cumulative move_stats snapshots -> per-block acceptance rates."""
    import numpy as np
    prop = np.array([s[kind][0] for s in stats], dtype=np.float64)
    acc = np.array([s[kind][1] for s in stats], dtype=np.float64)
    if prop[-1] - prop[0] <= 0:
        return None
    edges = np.linspace(0, len(prop) - 1, nb + 1).round().astype(int)
    dp, da = np.diff(prop[edges]), np.diff(acc[edges])
    return list(da / np.maximum(dp, 1.0))


def compare(recs: list[dict[str, Any]], args) -> Report:
    rep = Report()
    by = {(r["build"], r["task"]["kind"], r["task"]["label"]): r for r in recs}
    for (build, kind, label), r in sorted(by.items()):
        if "failed" in r:
            rep.add(kind, f"{label} [{build}]", False, "CRASH " + r["failed"][-300:])
    labels = sorted({(k, lab) for (_, k, lab) in by})
    for kind, label in labels:
        ra, rb = by.get(("A", kind, label)), by.get(("B", kind, label))
        if not ra or not rb or "failed" in ra or "failed" in rb:
            continue
        if kind == "static":
            _cmp_energies(rep, "static", label, ra, rb, args.rtol, args.atol)
        elif kind == "running":
            for build, r in (("A", ra), ("B", rb)):
                d = abs(r["running"] - r["full"])
                tol = max(args.running_tol, args.running_rtol * abs(r["full"]))
                rep.add("running", f"{label} [{build}]", d <= tol,
                        f"|running-full|={d:.2e} (tol {tol:.1e}, rel {d / abs(r['full']):.1e}) "
                        f"after {r['steps']} steps "
                        f"(E={r['full']:.3f}, {1e6 * r['mc_seconds'] / r['steps']:.1f} us/step)")
        elif kind == "korp":
            exp = KORP_REFERENCE[label]
            for build, r in (("A", ra), ("B", rb)):
                e = r["unweighted"].get(str(KORP_GROUP), float("nan"))
                rel = abs(e - exp) / abs(exp)
                rep.add("korp", f"{label} [{build}]", rel <= args.korp_rtol,
                        f"{e:.6f} vs korpe {exp:.6f}, rel {rel:.1e}")
            _cmp_energies(rep, "korp", f"{label} A-vs-B", ra, rb, args.rtol, args.atol)
    # sampling: aggregate seeds per protein
    groups = sorted({r["task"]["group"] for r in recs if r["task"]["kind"] == "sampling"})
    for g in groups:
        runs = {b: [r for r in recs if r["task"]["kind"] == "sampling"
                    and r["task"]["group"] == g and r["build"] == b and "failed" not in r]
                for b in ("A", "B")}
        if len(runs["A"]) < 2 or len(runs["B"]) < 2:
            rep.add("sampling", g, False, "fewer than 2 successful seeds per build")
            continue
        observables: dict[str, dict[str, list[list[float]]]] = {"energy": {}, "Q": {}}
        for b in ("A", "B"):
            observables["energy"][b] = [r["energy"] for r in runs[b]]
            observables["Q"][b] = [r["q"] for r in runs[b]]
        for kind in MOVE_KINDS:
            per = {b: [s for s in (_acceptance_series(r["move_stats"], kind) for r in runs[b])
                       if s is not None] for b in ("A", "B")}
            if per["A"] and per["B"] and all(per != v for v in observables.values()):
                observables[f"acc_{kind}"] = per   # rotamer can alias sc: report once
        for name, per in observables.items():
            ma, sa, da = _block_stats(per["A"])
            mb, sb, db = _block_stats(per["B"])
            se = math.sqrt(sa * sa + sb * sb)
            diff = mb - ma
            z = abs(diff) / se if se > 0 else (0.0 if diff == 0 else math.inf)
            sd = math.sqrt((da * da + db * db) / 2) or 1.0
            rep.add("sampling", f"{g} {name}", z <= args.z,
                    f"A {ma:.5g}+-{sa:.2g}  B {mb:.5g}+-{sb:.2g}  diff {diff:+.3g} "
                    f"z={z:.2f}  effect {diff / sd:+.3f} sd")
    return rep


# --------------------------------------------------------------------------
def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--_child":
        res = _child(json.loads(sys.argv[2]))
        print(MARKER + json.dumps(res))
        return 0

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--import-root", action="append", required=True,
                   help="build tree to import pymcpu from; give twice: reference A, candidate B")
    p.add_argument("--mode", choices=sorted(MODES), default="quick")
    p.add_argument("--only", default="static,running,sampling,korp",
                   help="comma list of checks to run")
    p.add_argument("--proteins", default="chignolin,actin",
                   help="labels of the structures for the running and sampling checks")
    p.add_argument("--pdb", action="append", default=[], metavar="PATH",
                   help="extra structure (repeatable); its file stem is its label")
    p.add_argument("--korp-map", default=os.environ.get("KORP_MAP_PATH", str(DEFAULT_KORP_MAP)))
    for k in ("running_steps", "korp_running_steps", "sampling_steps", "seeds", "sample_every"):  # noqa
        p.add_argument("--" + k.replace("_", "-"), type=int, default=None)
    p.add_argument("--recompute-every", type=int, default=50_000,
                   help="sampling: steps between full recomputes (production: 10k-100k)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed-offset", type=int, default=1000,
                   help="build B's sampling seeds = A's + this (0 = same seeds)")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-4)
    p.add_argument("--running-tol", type=float, default=1e-3,
                   help="running check: absolute floor on |running - full|")
    p.add_argument("--running-rtol", type=float, default=1e-5,
                   help="running check: relative allowance (0 = absolute 1e-3 only; "
                        "current main needs it, see module docstring)")
    p.add_argument("--korp-rtol", type=float, default=1e-6)
    p.add_argument("--z", type=float, default=None,
                   help="sampling threshold in combined SEs (quick 3.5, full 3.0)")
    p.add_argument("--cpus", default=None, help="CPU list for children, e.g. 24-29")
    p.add_argument("--save", metavar="PATH", help="write every raw record as JSON")
    p.add_argument("--ref", metavar="PATH", help="reuse build A's records from a --save file")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()
    if len(args.import_root) != 2:
        p.error("give --import-root exactly twice (A then B)")
    for k, v in MODES[args.mode].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    korp_map = Path(args.korp_map).expanduser()
    if not korp_map.is_file():
        print(f"NOTE: no KORP map at {korp_map}; KORP cases skipped")
        korp_map = None
    params_dir = _resolve_params_dir(args.import_root[0])
    envs = {b: _child_env(r, params_dir, korp_map) for b, r in zip("AB", args.import_root)}
    plan = _plan(args, korp_map)
    cpus = _parse_cpus(args.cpus)
    config = {k: getattr(args, k) for k in ("mode", "only", "proteins", "running_steps",
              "korp_running_steps", "sampling_steps", "seeds", "sample_every",
              "recompute_every", "temperature", "seed_offset")}
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]

    recs_a: list[dict[str, Any]] = []
    if args.ref:
        saved = json.load(open(args.ref))
        if saved.get("config_hash") != config_hash:
            print(f"--ref {args.ref} was recorded with another configuration; re-running A")
        else:
            recs_a = [r for r in saved["records"] if r["build"] == "A"]
    jobs = [("B", t) for t in plan["B"]] + ([] if recs_a else [("A", t) for t in plan["A"]])
    print(f"tolerance_check: mode={args.mode} A={args.import_root[0]} B={args.import_root[1]}")
    print(f"  {len(jobs)} child runs on CPUs {cpus}  (params {params_dir})", flush=True)
    t0 = time.perf_counter()
    recs = recs_a + _run_tasks(jobs, envs, cpus, not args.quiet)
    elapsed = time.perf_counter() - t0

    sos = {r["build"]: r.get("so") for r in recs if r.get("so")}
    print(f"  A loaded {sos.get('A')}\n  B loaded {sos.get('B')}")
    if sos.get("A") == sos.get("B"):
        print("  NOTE: A and B loaded the same binary (self-test)")
    rep = compare(recs, args)
    width = max(len(c) for _, c, _, _ in rep.rows) if rep.rows else 10
    for check in ("static", "running", "sampling", "korp"):
        for c, case, ok, detail in rep.rows:
            if c == check:
                print(f"{'PASS' if ok else 'FAIL'}  {c:8s} {case:{width}s}  {detail}")
    summ = rep.summary()
    verdict = all(v[1] == 0 for v in summ.values())
    print("summary: " + ", ".join(f"{k} {v[0]} pass/{v[1]} fail" for k, v in sorted(summ.items()))
          + f"  ({elapsed:.0f} s wall)")
    print("OVERALL", "PASS" if verdict else "FAIL")
    if args.save:
        json.dump({"config": config, "config_hash": config_hash, "records": recs,
                   "rows": rep.rows}, open(args.save, "w"))
    if any("failed" in r for r in recs):
        return 2
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
