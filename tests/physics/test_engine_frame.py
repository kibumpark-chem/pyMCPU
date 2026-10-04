"""A structure placed far from the origin runs in a shifted engine frame.

Coordinates are float32, and every move rounds each coordinate it changes at
its absolute value, so far from the origin bond lengths drift, KIC moves fail
their reversibility check and rigidly carried pairs can slip under their
hard-core cutoff. A Context therefore shifts a structure that reaches 64 A or
more from the origin next to it, by a whole number of A per axis it does not
straddle, which is exact for float32 input; every coordinate output adds the
shift back, in float64. See docs/api/context.rst, "The engine frame".
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path

import mdtraj as md
import numpy as np
import pytest

from pymcpu import mcpu_core
from pymcpu.config import EngineSpec
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from pymcpu.sampling import EngineSession, get_coords, swap_context_coordinates
from tests.fixtures.context_builders import resolve_test_pdb


def _heavy(pdb) -> md.Trajectory:
    traj = md.load(str(pdb))
    return traj.atom_slice(traj.topology.select("not element H"))


@functools.lru_cache(maxsize=None)
def _actin(offset_a: float):
    """Actin (heavy atoms) translated by offset_a on every axis, and its force
    field: built once per offset, as each build takes about 6 s."""
    heavy = _heavy(resolve_test_pdb())
    heavy.xyz = (heavy.xyz + np.float32(offset_a / 10.0)).astype(np.float32)
    return heavy, MCPUForceField(heavy)


def _chignolin_at(offset_a: float):
    """Chignolin translated by offset_a on every axis BEFORE the force field
    is built, as a far input file would be."""
    heavy = _heavy(default_example_pdb())
    heavy.xyz = (heavy.xyz + np.float32(offset_a / 10.0)).astype(np.float32)
    ff = MCPUForceField(heavy)
    return ff, heavy.topology, (ff.coords[0] * 10.0).T.astype(np.float32)


def _context(ff, topology, coords, **kwargs):
    ctx = mcpu_core.Context(ff.create_system(topology))
    ctx.set_positions(coords, **kwargs)
    ctx.calculate_total_energy(-1)
    return ctx


def _expected_offset(u: np.ndarray) -> np.ndarray:
    """The rule of include/pymcpu/utils/FrameOffset.h, in numpy."""
    u = u.astype(np.float64)
    s = np.zeros(3)
    if np.abs(u).max() < 64.0:
        return s
    for d in range(3):
        lo, hi = u[d].min(), u[d].max()
        if lo > 0:
            s[d] = min(np.floor(0.5 * (lo + hi) + 0.5), np.floor(2 * lo))
        elif hi < 0:
            s[d] = max(np.ceil(0.5 * (lo + hi) - 0.5), np.ceil(2 * hi))
    return s


@pytest.mark.parametrize("pdb", ["chignolin", "actin"])
def test_inputs_near_the_origin_are_not_shifted(pdb) -> None:
    if pdb == "chignolin":
        heavy = _heavy(default_example_pdb())
        ff = MCPUForceField(heavy)
    else:
        heavy, ff = _actin(0.0)
    x = (ff.coords[0] * 10.0).T.astype(np.float32)
    x[0, 0] = np.float32(-0.0)
    ctx = _context(ff, heavy.topology, x)
    assert np.array_equal(ctx.frame_offset, np.zeros(3))
    assert np.array_equal(np.asarray(ctx.get_state().coords), x)
    assert np.signbit(ctx.coords[0, 0])
    assert ctx.coords.dtype == np.float64


@pytest.mark.parametrize(
    ("offset", "capped"),
    [
        ((200.0, 180.0, 220.0), False),
        ((-3000.0, 0.0, 4000.0), False),
        ((70.0, 0.0, 9.0), True),
        ((-70.0, 0.0, -9.0), True),
    ],
    ids=["cryo-em-box", "far", "capped", "capped-negative"],
)
def test_a_far_input_runs_near_the_origin(offset, capped: bool) -> None:
    heavy = _heavy(default_example_pdb())
    ff = MCPUForceField(heavy)
    x = (ff.coords[0] * 10.0).T.astype(np.float32)
    far = (x + np.asarray(offset, dtype=np.float32)[:, None]).astype(np.float32)

    ctx = _context(ff, heavy.topology, far)
    shift = _expected_offset(far)
    assert np.array_equal(ctx.frame_offset, shift)
    if capped:
        # z lies on one side, close to 0: the shift stops at twice its
        # smallest |z| instead of reaching the middle. That cap is what keeps
        # the entry exact.
        z = far[2].astype(np.float64)
        middle = np.floor(0.5 * (z.min() + z.max()) + 0.5) if z.min() > 0 else \
            np.ceil(0.5 * (z.min() + z.max()) - 0.5)
        assert shift[2] != middle
    assert np.abs(np.asarray(ctx.get_state().coords)).max() < 20.0
    # Exact both ways: the engine holds far - shift, and coords gives far back.
    assert np.array_equal(np.asarray(ctx.get_state().coords),
                          (far.astype(np.float64) - shift[:, None]).astype(np.float32))
    assert np.array_equal(ctx.coords, far.astype(np.float64))

    # Same distances, same energies as running it where it was.
    kept = _context(ff, heavy.topology, far, frame_offset=(0.0, 0.0, 0.0))
    shifted, unshifted = ctx.energy_breakdown(False), kept.energy_breakdown(False)
    for term in ("mu", "backbone_torsion", "sidechain_torsion"):
        assert shifted["by_name"][term] == unshifted["by_name"][term], term
    assert shifted["raw_total"] == pytest.approx(unshifted["raw_total"], abs=1e-4)


def test_kic_works_far_from_the_origin() -> None:
    """Chignolin 4000 A out. Kept there, KIC's reversibility check fails
    about 2000 times in 20k steps and only ~10 KIC moves are accepted
    (1 failure and ~430 accepts at the origin); in the engine frame it runs as
    it does at the origin."""
    ff, top, coords = _chignolin_at(4000.0)

    def run(**kwargs):
        ctx = _context(ff, top, coords, **kwargs)
        integ = mcpu_core.Integrator(temperature=0.6)
        integ.set_seed(7)
        integ.run(ctx, 20000, 0)
        return integ.move_stats()

    shifted = run()
    assert shifted["kic_reverse_missing"] <= 10
    assert shifted["num_accept_kic"] >= 300
    kept = run(frame_offset=(0.0, 0.0, 0.0))
    assert kept["kic_reverse_missing"] >= 500


def test_the_frame_is_fixed_by_the_first_placement() -> None:
    ff, top, far = _chignolin_at(4000.0)
    ctx = _context(ff, top, far)
    first = ctx.frame_offset.copy()
    assert np.all(first > 3000.0)

    moved = (far + np.float32(37.0)).astype(np.float32)
    ctx.set_positions(moved)
    assert np.array_equal(ctx.frame_offset, first)
    assert np.array_equal(ctx.coords, moved.astype(np.float64))

    ctx.set_positions(far, frame_offset=(0.0, 0.0, 0.0))
    assert np.array_equal(ctx.frame_offset, np.zeros(3))
    assert np.array_equal(np.asarray(ctx.get_state().coords), far)
    ctx.set_positions(far)  # an explicit offset sticks too
    assert np.array_equal(ctx.frame_offset, np.zeros(3))


def test_a_far_restart_continues_bit_for_bit() -> None:
    """(coords, rng state) taken from a far run, placed into a fresh Context
    after its start structure, continue the run exactly."""
    ff, top, far = _chignolin_at(4000.0)
    a = _context(ff, top, far)
    integ_a = mcpu_core.Integrator(temperature=0.6)
    integ_a.set_seed(11)
    integ_a.run(a, 2000, 0)
    snapshot, rng = get_coords(a), integ_a.get_rng_state()
    assert snapshot.dtype == np.float64
    integ_a.run(a, 2000, 2000)

    b = _context(ff, top, far)
    b.set_positions(snapshot)
    b.calculate_total_energy(-1)
    integ_b = mcpu_core.Integrator(temperature=0.6)
    integ_b.set_rng_state(rng)
    integ_b.run(b, 2000, 2000)
    assert np.array_equal(get_coords(a), get_coords(b))


def test_a_far_engine_session_restarts_bit_for_bit(tmp_path) -> None:
    heavy = _heavy(default_example_pdb())
    heavy.xyz = (heavy.xyz + np.float32(400.0)).astype(np.float32)
    pdb = str(tmp_path / "far.pdb")
    heavy.save_pdb(pdb)
    spec = EngineSpec(pdb=pdb, cv=({"type": "native_contacts_q", "reference_pdb": pdb},))

    a = EngineSession(spec)
    a.set_seed(123)
    a.step(500)
    snapshot, rng, step = a.coords(), a.get_rng_state(), a.current_step
    assert np.abs(snapshot).min() > 3000.0  # the caller's frame
    a.step(500)

    b = EngineSession(spec)
    b.set_coords(snapshot)
    b.restore_rng_state(rng)
    b.current_step = step
    b.step(500)
    assert np.array_equal(a.coords(), b.coords())


def test_a_far_replica_swap_is_exact() -> None:
    ff, top, far = _chignolin_at(4000.0)
    a, b = _context(ff, top, far), _context(ff, top, far)
    for ctx, seed in ((a, 1), (b, 2)):
        integ = mcpu_core.Integrator(temperature=0.6)
        integ.set_seed(seed)
        integ.run(ctx, 500, 0)
    engine_a = np.asarray(a.get_state().coords).copy()
    engine_b = np.asarray(b.get_state().coords).copy()
    swap_context_coordinates(a, b, get_coords(a), get_coords(b))
    assert np.array_equal(np.asarray(a.get_state().coords), engine_b)
    assert np.array_equal(np.asarray(b.get_state().coords), engine_a)


def test_the_coords_setter_shifts_once_after_a_reorder() -> None:
    """After an init_only reorder the coords setter takes its own path; it must
    apply the frame offset exactly once, in either output order."""
    heavy, ff = _actin(4000.0)
    ctx = mcpu_core.Context(ff.create_system(heavy.topology))
    ctx.set_atom_reorder_mode("init_only")
    ctx.set_positions((ff.coords[0] * 10.0).T.astype(np.float32))
    assert np.all(ctx.frame_offset > 3000.0)
    engine = np.asarray(ctx.get_state().coords).copy()
    for internal_order in (False, True):
        ctx.set_output_internal_order(internal_order)
        out = ctx.coords
        ctx.coords = out
        assert np.array_equal(np.asarray(ctx.get_state().coords), engine), internal_order
        assert np.array_equal(ctx.coords, out), internal_order


def test_xtc_frames_are_in_the_callers_frame(tmp_path) -> None:
    import pymcpu as mc

    ff, top, far = _chignolin_at(4000.0)
    sim = mc.Simulation(top, ff.create_system(top), mc.Integrator(0.6))
    sim.context.set_positions(far)
    path = tmp_path / "far.xtc"
    sim.add_xtc_reporter(str(path), 100)
    sim.step(100)  # the reporter flushes every frame
    frame = md.load(str(path), top=top).xyz[-1].T * 10.0
    np.testing.assert_allclose(frame, sim.context.coords, atol=1e-2)


def test_a_far_replica_exchange_checkpoint_restores_bit_for_bit(tmp_path) -> None:
    """Checkpoints hold float64 coordinates in the caller's frame, and a fresh
    ReplicaExchange that loads one gets the engine coordinates back exactly."""
    from pymcpu.checkpointing import load_checkpoint
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    traj = md.load(str(default_example_pdb()))
    traj.xyz = (traj.xyz + np.float32(400.0)).astype(np.float32)
    pdb = str(tmp_path / "far.pdb")
    traj.save_pdb(pdb)

    def build(name: str) -> ReplicaExchange:
        return ReplicaExchange(
            pdb, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0,
            log_interval=10, output_prefix="rex", output_dir=tmp_path / name,
            seed=7, checkpoint_dir=tmp_path / "ckpt", checkpoint_interval=1,
        )

    rex = build("a")
    rex.run(2, 50, verbose=False, write_logs=False)
    engine = [np.asarray(r.simulation.context.get_state().coords).copy() for r in rex.replicas]
    assert all(np.all(r.simulation.context.frame_offset > 3000.0) for r in rex.replicas)
    saved = load_checkpoint(tmp_path / "ckpt" / "last.chk")["replica_coords"]
    assert all(np.asarray(c).dtype == np.float64 for c in saved)

    fresh = build("b")
    fresh.load_checkpoint(tmp_path / "ckpt")
    for rep, before in zip(fresh.replicas, engine):
        assert np.array_equal(np.asarray(rep.simulation.context.get_state().coords), before)


def test_far_coordinates_left_far_are_noted_once() -> None:
    """The note is once per process, so it is checked in a child process."""
    code = (
        "import numpy as np, mdtraj as md\n"
        "from pymcpu import mcpu_core\n"
        "from pymcpu.forcefields.mcpu import MCPUForceField\n"
        "from pymcpu.runners import default_example_pdb\n"
        "t = md.load(str(default_example_pdb()))\n"
        "h = t.atom_slice(t.topology.select('not element H'))\n"
        "ff = MCPUForceField(h)\n"
        "x = (ff.coords[0] * 10.0).T.astype(np.float32)\n"
        "ctx = mcpu_core.Context(ff.create_system(h.topology))\n"
        "ctx.set_positions(x + np.float32(4000.0))\n"
        "ctx.set_positions(x + np.float32(2000.0), frame_offset=(0.0, 0.0, 0.0))\n"
        "ctx.set_positions(x + np.float32(3000.0), frame_offset=(0.0, 0.0, 0.0))\n"
    )
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2])
    assert run.returncode == 0, run.stderr[-2000:]
    err = run.stderr
    assert err.count("NOTE: coordinates reach") == 1
    assert "frame_offset 0 0 0 A" in err
