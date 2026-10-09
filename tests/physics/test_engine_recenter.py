"""The engine frame follows a chain that drifts away from the origin.

Every move rounds the coordinates it changes at their absolute value, so a
chain that drifts far out during a run (an unfolded, hot one does) loses
precision. Context.recenter shifts the engine frame back onto the chain once
a coordinate reaches 64 A, by the whole-A midpoint of each axis.
Simulation.step calls it after each periodic full recompute, and the drivers
before every checkpoint save (Simulation.recompute_and_recenter). A walker
keeps its own frame: replica swaps and checkpoints carry the offset with the
coordinates. See docs/api/context.rst, "The engine frame".
"""

from __future__ import annotations

import functools

import mdtraj as md
import numpy as np
import pytest

import pymcpu as mc
from pymcpu import mcpu_core
from pymcpu.checkpointing import load_checkpoint, save_checkpoint, saved_frame_offset
from pymcpu.config import EngineSpec
from pymcpu.forcefields.mcpu import MCPUForceField
from pymcpu.runners import default_example_pdb
from pymcpu.sampling import EngineSession, get_coords, swap_context_coordinates

FAR = np.float32(150.0)


@functools.lru_cache(maxsize=None)
def _chignolin():
    """Chignolin heavy atoms, its force field, and its start coordinates
    (float32, A, (3, n)); its native structure lies within 7 A of the origin."""
    traj = md.load(str(default_example_pdb()))
    heavy = traj.atom_slice(traj.topology.select("not element H"))
    ff = MCPUForceField(heavy)
    return ff, heavy.topology, (ff.coords[0] * 10.0).T.astype(np.float32)


def _context(coords, **kwargs) -> mcpu_core.Context:
    ff, top, _ = _chignolin()
    ctx = mcpu_core.Context(ff.create_system(top))
    ctx.set_positions(coords, **kwargs)
    ctx.calculate_total_energy(-1)
    assert not ctx.has_steric_clash()
    return ctx


def _engine(ctx) -> np.ndarray:
    return np.asarray(ctx.get_state().coords).copy()


def _far_context() -> mcpu_core.Context:
    """Chignolin 150 A out on every axis, run unshifted (an explicit zero
    offset), as a chain that drifted there would be."""
    return _context(_chignolin()[2] + FAR, frame_offset=(0.0, 0.0, 0.0))


def test_recenter_does_nothing_below_the_threshold() -> None:
    ctx = _context(_chignolin()[2])
    engine, energy = _engine(ctx), ctx.get_state().current_energy
    resyncs = ctx.energy_resyncs
    shift = ctx.recenter()
    assert shift.dtype == np.float64 and shift.shape == (3,)
    assert not np.any(shift)
    assert np.array_equal(_engine(ctx), engine)
    assert ctx.get_state().current_energy == energy
    assert np.array_equal(ctx.frame_offset, np.zeros(3))

    # A far chain is left alone too while its reach is under min_reach_A,
    # and without a recompute (the energy below is the running one).
    far = _far_context()
    integ = mcpu_core.Integrator(temperature=0.6)
    integ.set_seed(3)
    integ.run(far, 200, 0)
    engine, energy = _engine(far), far.get_state().current_energy
    assert not np.any(far.recenter(min_reach_A=1000.0))
    assert np.array_equal(_engine(far), engine)
    assert far.get_state().current_energy == energy
    assert far.energy_resyncs == resyncs


def test_a_far_chain_is_centred() -> None:
    ctx = _far_context()
    before = _engine(ctx).astype(np.float64)
    user, energy = ctx.coords.copy(), ctx.get_state().current_energy
    expected = np.round(0.5 * (before.min(axis=1) + before.max(axis=1)))
    assert np.all(expected > 140.0)

    shift = ctx.recenter()
    assert np.array_equal(shift, expected)
    assert np.array_equal(ctx.frame_offset, expected)
    after = _engine(ctx).astype(np.float64)
    assert np.all(np.abs(0.5 * (after.min(axis=1) + after.max(axis=1))) <= 0.5)
    assert np.abs(after).max() < 10.0
    # Every atom moved toward the origin, so the shift is exact: the engine
    # coordinates are the old ones minus the shift, and the caller's
    # coordinates are unchanged bit for bit.
    assert np.array_equal(after, before - expected[:, None])
    assert np.array_equal(ctx.coords, user)
    # The energy was recomputed in the new frame; distances are unchanged, so
    # it is the same energy, and it is what a full recompute gives.
    assert ctx.get_state().current_energy == pytest.approx(energy, abs=1e-9)
    assert ctx.calculate_total_energy(-1) == ctx.get_state().current_energy
    assert not ctx.has_steric_clash()
    # Centred, it stays put.
    assert not np.any(ctx.recenter(min_reach_A=0.0))


def test_an_atom_moved_outward_is_rounded_once() -> None:
    """A chain straddling the origin: atoms on the far side of the midpoint
    end farther out and are rounded once, by at most half a float32 step;
    every other atom is shifted exactly."""
    x = _chignolin()[2].copy()
    x[0] += np.float32(90.0)
    x[0, :4] -= np.float32(140.0)  # the first residue crosses the origin
    ctx = _context(x, frame_offset=(0.0, 0.0, 0.0))
    before, user = _engine(ctx).astype(np.float64), ctx.coords.copy()

    shift = ctx.recenter()
    assert shift[0] != 0.0
    after = _engine(ctx).astype(np.float64)
    inward = np.abs(after) <= np.abs(before)
    assert np.array_equal(after[inward], (before - shift[:, None])[inward])
    assert np.array_equal(ctx.coords[inward], user[inward])
    outward = ~inward
    assert outward.any()
    err = np.abs(after - (before - shift[:, None]))[outward]
    half_step = 0.5 * np.spacing(np.abs(after[outward]).astype(np.float32))
    assert np.all(err <= half_step)
    assert np.abs(after).max() <= np.abs(before).max()


def test_a_shift_that_rounds_a_pair_under_the_hard_core_is_undone() -> None:
    """Rounding the atoms that end farther out can put a pair just above its
    state cutoff under it. Such a shift is undone, and the state is exactly
    as it was. The case is built: a pair 55 A out on the far side of a chain
    200 A out, set within a few float32 steps of its state cutoff."""
    ff, top, x = _chignolin()
    i, j = 1, x.shape[1] - 2
    base = x.copy()
    base[0] += np.float32(200.0)
    ctx = mcpu_core.Context(ff.create_system(top))

    def place(xi, xj) -> bool:
        c = base.copy()
        c[:, i] = np.float32([xi, 0.5, 0.5])
        c[:, j] = np.float32([xj, 0.5, 0.5])
        ctx.set_positions(c, frame_offset=(0.0, 0.0, 0.0))
        ctx.calculate_total_energy(-1)
        return ctx.has_steric_clash()

    undone = 0
    # Atom i lands on the float32 grid of its new position, so only j rounds;
    # which side of the cutoff it rounds to depends on the low bits of xi.
    for k in range(4):
        xi = np.float32(-55.0 + k * 2.0**-17)
        clash, clean = float(xi) - 2.0, float(xi) - 3.5
        assert place(xi, clash) and not place(xi, clean)
        while True:  # the cutoff, to one float32 step
            mid = np.float32(0.5 * (clash + clean))
            if mid in (np.float32(clash), np.float32(clean)):
                break
            if place(xi, mid):
                clash = float(mid)
            else:
                clean = float(mid)
        xj = np.float32(clean)
        for _ in range(3):
            assert not place(xi, xj)
            engine, user = _engine(ctx), ctx.coords.copy()
            energy = ctx.get_state().current_energy
            if not np.any(ctx.recenter()):
                undone += 1
                assert np.array_equal(_engine(ctx), engine)
                assert np.array_equal(ctx.coords, user)
                assert np.array_equal(ctx.frame_offset, np.zeros(3))
                assert ctx.get_state().current_energy == energy
                assert not ctx.has_steric_clash()
                # A full recompute agrees, and the next run starts cleanly.
                assert ctx.calculate_total_energy(-1) == pytest.approx(energy, abs=1e-9)
                mcpu_core.Integrator(temperature=0.6).run(ctx, 10, 0)
            else:
                assert not ctx.has_steric_clash()
            xj = np.nextafter(xj, np.float32(-1000.0))
    assert undone > 0


@pytest.mark.parametrize("enabled", [True, False])
def test_simulation_recentres_after_the_periodic_recompute(enabled: bool) -> None:
    ff, top, x = _chignolin()
    sim = mc.Simulation(top, ff.create_system(top), mc.Integrator(0.6))
    sim.integrator.set_seed(5)
    sim.full_energy_every_steps = 100
    sim.recenter_frame = enabled
    sim.context.set_positions(x + FAR, frame_offset=(0.0, 0.0, 0.0))
    sim.step(99)  # no recompute yet, so no recentring
    assert np.array_equal(sim.context.frame_offset, np.zeros(3))
    user = sim.context.coords.copy()
    sim.step(1)
    if enabled:
        assert np.all(sim.context.frame_offset > 140.0)
        assert np.abs(_engine(sim.context)).max() < 20.0
    else:
        assert np.array_equal(sim.context.frame_offset, np.zeros(3))
    # The last step may have moved atoms, but recentring moved none of them.
    moved = sim.last_moved_indices
    keep = np.setdiff1d(np.arange(user.shape[1]), moved)
    assert np.array_equal(sim.context.coords[:, keep], user[:, keep])
    assert sim.recompute_energy() == pytest.approx(sim.context.get_state().current_energy, abs=1e-9)


def test_a_swap_between_frames_is_exact() -> None:
    a, b = _far_context(), _context(_chignolin()[2])
    a.recenter()
    for ctx, seed in ((a, 1), (b, 2)):
        integ = mcpu_core.Integrator(temperature=0.6)
        integ.set_seed(seed)
        integ.run(ctx, 300, 0)
    engine_a, engine_b = _engine(a), _engine(b)
    offset_a, offset_b = a.frame_offset.copy(), b.frame_offset.copy()
    assert not np.array_equal(offset_a, offset_b)

    swap_context_coordinates(a, b, get_coords(a), get_coords(b))
    assert np.array_equal(_engine(a), engine_b) and np.array_equal(a.frame_offset, offset_b)
    assert np.array_equal(_engine(b), engine_a) and np.array_equal(b.frame_offset, offset_a)
    swap_context_coordinates(a, b, get_coords(a), get_coords(b))
    assert np.array_equal(_engine(a), engine_a) and np.array_equal(a.frame_offset, offset_a)
    assert np.array_equal(_engine(b), engine_b) and np.array_equal(b.frame_offset, offset_b)


def _chignolin_pdb(tmp_path) -> str:
    path = str(tmp_path / "chignolin.pdb")
    md.load(str(default_example_pdb())).save_pdb(path)
    return path


def test_replica_exchange_checkpoints_keep_each_frame(tmp_path) -> None:
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    pdb = _chignolin_pdb(tmp_path)

    def build(name: str) -> ReplicaExchange:
        return ReplicaExchange(
            pdb, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0,
            log_interval=10, output_prefix="rex", output_dir=tmp_path / name,
            seed=7, checkpoint_dir=tmp_path / "ckpt", checkpoint_interval=1,
        )

    rex = build("a")
    rex.run(2, 50, verbose=False, write_logs=False)

    # A checkpoint written before offsets were saved restores as it always
    # did: in the fresh Context's frame, the one it was written from.
    old = load_checkpoint(tmp_path / "ckpt" / "last.chk")
    assert len(old.pop("replica_frame_offsets")) == 2
    save_checkpoint(old, tmp_path / "old", filename="last.chk")
    fresh = build("b")
    fresh.load_checkpoint(tmp_path / "old")
    for rep, before in zip(fresh.replicas, rex.replicas):
        ctx = rep.simulation.context
        assert np.array_equal(_engine(ctx), _engine(before.simulation.context))
        assert np.array_equal(ctx.frame_offset, np.zeros(3))

    # Give each replica its own frame, far out, as drifting chains would get.
    for k, rep in enumerate(rex.replicas):
        ctx = rep.simulation.context
        ctx.set_positions(get_coords(ctx) + 100.0 * (k + 1), frame_offset=(0.0, 0.0, 0.0))
        ctx.calculate_total_energy(-1)
        assert np.all(ctx.recenter() > 90.0)
    engine = [_engine(r.simulation.context) for r in rex.replicas]
    offsets = [r.simulation.context.frame_offset.copy() for r in rex.replicas]
    assert not np.array_equal(offsets[0], offsets[1])
    rex.save_checkpoint(tmp_path / "new")

    fresh = build("c")
    fresh.load_checkpoint(tmp_path / "new")
    for rep, e, o in zip(fresh.replicas, engine, offsets):
        assert np.array_equal(_engine(rep.simulation.context), e)
        assert np.array_equal(rep.simulation.context.frame_offset, o)


def test_checkpoint_saves_recentre_and_a_resume_continues_exactly(tmp_path) -> None:
    # The drivers recompute before every save, and that restarts the count
    # toward the periodic recompute: when saves come more often (as with the
    # defaults), the periodic recompute never comes, and the saves have to
    # move the frame.
    from pymcpu.sampling.replica_exchange import ReplicaExchange

    pdb = _chignolin_pdb(tmp_path)

    def build(name: str, far: bool) -> ReplicaExchange:
        rex = ReplicaExchange(
            pdb, temperatures=[0.5, 0.6], n_targets=[0.0], k_bias=0.0,
            log_interval=10, output_prefix="rex", output_dir=tmp_path / name,
            seed=7, checkpoint_dir=tmp_path / name / "ckpt", checkpoint_interval=1,
            full_energy_every_steps=100,
        )
        for k, rep in enumerate(rex.replicas if far else ()):
            ctx = rep.simulation.context
            ctx.set_positions(get_coords(ctx) + 150.0 + 20.0 * k, frame_offset=(0.0, 0.0, 0.0))
            ctx.calculate_total_energy(-1)
        return rex

    whole = build("whole", far=True)
    whole.run(3, 50, verbose=False, write_logs=False)

    # 50 steps, then a save: no periodic recompute yet, so the save moved it.
    part = build("part", far=True)
    part.run(1, 50, verbose=False, write_logs=False)
    for rep in part.replicas:
        ctx = rep.simulation.context
        assert np.all(ctx.frame_offset > 140.0)
        assert np.abs(_engine(ctx)).max() < 20.0
    saved = load_checkpoint(tmp_path / "part" / "ckpt" / "last.chk")
    for k, rep in enumerate(part.replicas):
        assert np.array_equal(saved_frame_offset(saved, k), rep.simulation.context.frame_offset)

    # The checkpoint holds the recentred state, so the run resumed from it
    # is the uninterrupted one.
    resumed = build("part", far=False)
    resumed.run(3, 50, verbose=False, write_logs=False, resume=tmp_path / "part" / "ckpt")
    for a, b in zip(whole.replicas, resumed.replicas):
        assert np.array_equal(_engine(a.simulation.context), _engine(b.simulation.context))
        assert np.array_equal(a.simulation.context.frame_offset, b.simulation.context.frame_offset)


def test_a_folding_checkpoint_keeps_its_frame(tmp_path) -> None:
    from pymcpu.sampling.folding import FoldingRunner

    pdb = _chignolin_pdb(tmp_path)

    def runner(resume: bool) -> FoldingRunner:
        return FoldingRunner(
            pdb, output_dir=str(tmp_path / "out"), checkpoint_dir=str(tmp_path / "ck"),
            seed=1, report_interval=5, steps_per_cycle=5, checkpoint_interval=1,
            resume=resume, verbose=False,
        )

    first = runner(resume=False)
    ctx = first.simulation.context
    ctx.set_positions(get_coords(ctx) + 150.0, frame_offset=(0.0, 0.0, 0.0))
    ctx.calculate_total_energy(-1)
    first.save_checkpoint(cycle=1)  # recentres first, then saves
    assert np.all(ctx.frame_offset > 140.0)
    engine, offset = _engine(ctx), ctx.frame_offset.copy()

    second = runner(resume=True)
    second.load_checkpoint(str(tmp_path / "ck" / "last.chk"))
    assert np.array_equal(_engine(second.simulation.context), engine)
    assert np.array_equal(second.simulation.context.frame_offset, offset)


def test_an_engine_session_restart_keeps_its_frame(tmp_path) -> None:
    pdb = _chignolin_pdb(tmp_path)
    spec = EngineSpec(pdb=pdb, cv=({"type": "native_contacts_q", "reference_pdb": pdb},))
    a = EngineSession(spec)
    a.set_coords(a.coords() + 150.0, frame_offset=(0.0, 0.0, 0.0))
    a._ensure_sim().full_energy_every_steps = 100
    a.set_seed(123)
    a.step(300)
    snapshot, offset = a.coords(), a.frame_offset()
    rng, step = a.get_rng_state(), a.current_step
    assert np.all(offset > 140.0)
    a.step(300)

    b = EngineSession(spec)
    b._ensure_sim().full_energy_every_steps = 100
    b.set_coords(snapshot, frame_offset=offset)
    b.restore_rng_state(rng)
    b.current_step = step
    b.step(300)
    assert np.array_equal(a.frame_offset(), b.frame_offset())
    assert np.array_equal(a.coords(), b.coords())
