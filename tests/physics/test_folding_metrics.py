"""Self-consistency tests for ``FoldingRunner``'s folding-progress metrics.

Covers native-contact detection, Q (fraction of native contacts formed) and
CA RMSD-to-native computation, convergence-window logic, and the removal of
the ``FoldingBias`` / ``BasinTracker`` skeletons (old checkpoints that carry
their fields still load). None of this compares against legacy MCPU -- legacy
MCPU has no folding-progress-tracking feature to compare against, so every
check here is an internal consistency property (e.g. "the native structure
scores near-perfect on its own metric", "RMSD is never negative", "a window
of all-high Q values is reported as converged").

``FoldingRunner`` instances here are built via ``_make_minimal_folding_runner``
(``__new__`` + hand-set attributes, no real MC engine) so metric methods can
be unit-tested without constructing a full ``Simulation``/forcefield. The
run()-orchestration side of folding checkpointing (call ordering, resume,
convergence-triggered saves) lives in
``tests/integration/checkpointing/test_folding_checkpoint.py`` instead, built
against a real ``FoldingRunner``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

mdtraj = pytest.importorskip(
    "mdtraj", reason="mdtraj not installed -- skipping folding metric tests"
)
import mdtraj as md  # noqa: E402

from pymcpu.checkpointing import CheckpointConfig, FoldingCheckpointState, load_checkpoint  # noqa: E402
from pymcpu.sampling.folding import FoldingRunner  # noqa: E402

# ── Shared test-runner construction ──────────────────────────────

#: Read straight from FoldingRunner.__init__'s own signature so this helper
#: can't silently drift from production defaults (the old version restated
#: contact_cutoff_ang=8.0, min_seq_sep=4, etc. as separate literals here).
_INIT_DEFAULTS = inspect.signature(FoldingRunner.__init__).parameters


def _ca_blocks_forcefield(pdb: str | Path) -> SimpleNamespace:
    """Stand-in force field whose engine CA indices are the PDB's CA indices.

    ``build_contact_atom_index`` only reads ``blocks`` (CA = ``bb_start + 1``)
    and ``total_sc_atoms``, so this is enough for ``NativeContactsCV``.
    """
    top = md.load(str(pdb)).topology
    blocks = [
        SimpleNamespace(bb_start=atom.index - 1, sc_start=-1)
        for atom in top.atoms
        if atom.name == "CA"
    ]
    return SimpleNamespace(blocks=blocks, total_sc_atoms=0, output_topology=top)


def _make_minimal_folding_runner(
    reference_pdb: str | Path | None = None,
    pdb_path: str | Path | None = None,
) -> FoldingRunner:
    """Lightweight FoldingRunner for metric unit tests (no full MC engine)."""
    if pdb_path is None:
        pdb_path = reference_pdb

    if reference_pdb is not None:
        n_atoms = int(md.load(str(reference_pdb)).n_atoms)
    else:
        n_atoms = 77  # chignolin (1uao) atom count; arbitrary when no ref PDB

    mock_sim = MagicMock()
    mock_sim.get_coords.return_value = np.random.rand(n_atoms, 3).astype(np.float32)
    mock_sim.current_step = 0
    mock_sim.context = MagicMock()
    mock_sim.context.has_steric_clash.return_value = False  # a MagicMock is truthy

    runner = FoldingRunner.__new__(FoldingRunner)
    runner.simulation = mock_sim
    runner.replicas = [mock_sim]
    runner.forcefield = (
        _ca_blocks_forcefield(reference_pdb) if reference_pdb is not None else None
    )
    runner.contact_atom_mode = "ca"
    runner.system = MagicMock()
    runner.forcefield_name = "mcpu08"
    runner.system.get_num_atoms.return_value = n_atoms
    runner.pdb_path = Path(pdb_path) if pdb_path is not None else None
    runner.reference_pdb = str(reference_pdb) if reference_pdb is not None else None
    runner.contact_cutoff_ang = _INIT_DEFAULTS["contact_cutoff_ang"].default
    runner.min_seq_sep = _INIT_DEFAULTS["min_seq_sep"].default
    runner.q_threshold = _INIT_DEFAULTS["q_threshold"].default
    runner.convergence_window = _INIT_DEFAULTS["convergence_window"].default
    runner._native_contacts = None
    runner._q_cv = None
    runner._ca_internal_idx = None
    runner.native_contact_pairs = None
    runner.convergence_history = []
    runner.checkpoint_config = CheckpointConfig()
    runner.traj_reporters = {}
    runner.traj_frame_counts = {}
    runner.sample_writer = None
    runner.seed = _INIT_DEFAULTS["seed"].default
    runner.temperature = _INIT_DEFAULTS["temperature"].default
    runner.fixed_residues = []
    runner.linker_residues = []
    runner.linker_energy_mode = _INIT_DEFAULTS["linker_energy_mode"].default
    return runner


def _save_checkpoint_with_random_coords(runner: FoldingRunner, tmp_path: Path, monkeypatch, cycle: int) -> None:
    """Point checkpoint_dir at tmp_path, stub get_coords with random data, and save.

    Shared setup for the checkpoint round-trip tests below. get_coords is
    monkeypatched because these runners come from _make_minimal_folding_runner
    (mocked simulation, no real MC context), but save_checkpoint calls the
    real pymcpu.sampling.folding.get_coords helper.
    """
    runner.checkpoint_config.checkpoint_dir = str(tmp_path)
    n_atoms = int(runner.simulation.get_coords().shape[0])
    monkeypatch.setattr(
        "pymcpu.sampling.folding.get_coords",
        lambda ctx: np.random.rand(3, n_atoms).astype(np.float64),
    )
    runner.save_checkpoint(cycle=cycle)


def _save_and_reload(runner: FoldingRunner, tmp_path: Path, monkeypatch, cycle: int) -> FoldingCheckpointState:
    """Save via _save_checkpoint_with_random_coords, then reload as FoldingCheckpointState."""
    _save_checkpoint_with_random_coords(runner, tmp_path, monkeypatch, cycle)
    return FoldingCheckpointState.from_dict(load_checkpoint(str(tmp_path / "last.chk")))


def test_chignolin_pdb_path_is_actually_chignolin(chignolin_pdb_path: str) -> None:
    """Guard against the chignolin_pdb_path fixture regressing to point at a
    much larger structure (e.g. actin), which would silently invalidate every
    near-native Q/RMSD assertion below."""
    traj = md.load(chignolin_pdb_path)
    n_res = traj.topology.n_residues
    assert n_res <= 15, (  # chignolin (1uao) is 10 residues
        f"chignolin_pdb_path should point at chignolin (~10 residues), but "
        f"got {n_res} residues. Path: {chignolin_pdb_path}."
    )


# ── Native contact detection ──────────────────────────────────────


class TestNativeContacts:
    def test_non_empty(self, chignolin_pdb_path: str) -> None:
        """chignolin should have a non-empty native contact list at 8 Å / sep 4."""
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        contacts = runner._compute_native_contacts()
        assert isinstance(contacts, list)
        assert len(contacts) > 0

    def test_respects_min_seq_sep(self, chignolin_pdb_path: str) -> None:
        """No contact pair should have |i_res - j_res| < min_seq_sep."""
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        contacts = runner._compute_native_contacts()
        ref = md.load(str(chignolin_pdb_path))
        top = ref.topology

        for i, j in contacts:
            sep = abs(top.atom(i).residue.index - top.atom(j).residue.index)
            assert sep >= runner.min_seq_sep

    def test_cached_after_first_call(self, chignolin_pdb_path: str) -> None:
        """_compute_native_contacts result is cached in self._native_contacts."""
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        assert runner._native_contacts is None
        c1 = runner._compute_native_contacts()
        assert runner._native_contacts is not None
        assert runner._compute_native_contacts() == c1

    def test_empty_without_ref_pdb(self) -> None:
        """Without a reference PDB, _compute_native_contacts returns [] (never None)."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        assert runner._compute_native_contacts() == []

    def test_native_contact_pairs_replace_the_cutoff(
        self, chignolin_pdb_path: str
    ) -> None:
        """Explicit ``native_contact_pairs`` must replace cutoff/min_seq_sep
        derivation entirely, not just supplement it.

        Picks residue pair (0, 1): adjacent residues (sequence separation 1),
        which the cutoff-based derivation would always reject since it is
        below the default ``min_seq_sep`` (4). ``contact_cutoff_ang`` is also
        forced to 0.0, a distance no real atom pair can be below, so the
        cutoff-derived branch would additionally return zero contacts for
        that reason alone. Getting back exactly the explicit pair proves both
        knobs were bypassed, not merely under-triggered.
        """
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        engine_ca = runner._get_engine_ca_indices()
        assert len(engine_ca) >= 2, "need at least 2 CA atoms to form a pair"

        runner.contact_cutoff_ang = 0.0
        runner.min_seq_sep = 1000
        runner.native_contact_pairs = [(0, 1)]
        runner._native_contacts = None  # ensure a fresh (uncached) computation

        contacts = runner._compute_native_contacts()

        assert contacts == [(int(engine_ca[0]), int(engine_ca[1]))]

    def test_raises_when_the_cv_cannot_be_built(self, chignolin_pdb_path: str) -> None:
        """A CV that cannot be built is an error, not a silent switch to CA."""
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        runner.forcefield = None
        with pytest.raises(RuntimeError, match="need a force field"):
            runner._compute_native_contacts()

    def test_cv_build_errors_propagate(self, chignolin_pdb_path: str) -> None:
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        runner.contact_atom_mode = "cb"  # the stand-in force field has no CB
        with pytest.raises(ValueError, match="needs CB atoms"):
            runner._compute_native_contacts()


# ── Q (fraction of native contacts) ───────────────────────────────


class TestComputeQ:
    def test_returns_array_shape_1(self, chignolin_pdb_path: str) -> None:
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        q = runner._compute_Q_values()
        assert isinstance(q, np.ndarray)
        assert q.shape == (1,)

    def test_in_range_0_1(self, chignolin_pdb_path: str) -> None:
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        q = runner._compute_Q_values()
        assert np.all(q >= 0.0) and np.all(q <= 1.0)

    def test_native_structure_scores_near_1(self, chignolin_pdb_path: str) -> None:
        """Feeding the native coordinates back into the Q calculator should
        recover every native contact (self-vs-self)."""
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        ref = md.load(str(chignolin_pdb_path))
        native_coords_ang = ref.xyz[0] * 10.0
        runner.simulation.get_coords = lambda: native_coords_ang

        q = runner._compute_Q_values()
        # Freshly measured on the current engine: self-vs-self gives exactly
        # Q=1.0 (every native contact distance is trivially below cutoff).
        # The old test asserted a loose >=0.90 with no stated rationale;
        # tightened here to the actual measured value (with a hair of slack
        # for platform float rounding) since a real unit-mismatch regression
        # would show up far below this, not as noise around it.
        assert q[0] >= 0.999, f"Expected Q~1.0 for native structure, got {q[0]:.6f}"

    def test_returns_array_without_ref_pdb(self) -> None:
        """Without reference PDB, returns np.array([0.0]) not None."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        q = runner._compute_Q_values()
        assert isinstance(q, np.ndarray)
        assert q[0] == 0.0


# ── RMSD to native ─────────────────────────────────────────────────


class TestComputeRmsd:
    def test_returns_array_shape_1(self, chignolin_pdb_path: str) -> None:
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        rmsd = runner._compute_rmsd_values()
        assert isinstance(rmsd, np.ndarray)
        assert rmsd.shape == (1,)

    def test_non_negative(self, chignolin_pdb_path: str) -> None:
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        rmsd = runner._compute_rmsd_values()
        assert rmsd[0] >= 0.0

    def test_native_structure_scores_near_zero(self, chignolin_pdb_path: str) -> None:
        """Native coordinates should give CA RMSD near 0 after Kabsch (true
        self-superposition, bounded only by floating-point precision)."""
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        ref = md.load(str(chignolin_pdb_path))
        native_coords_ang = ref.xyz[0] * 10.0
        runner.simulation.get_coords = lambda: native_coords_ang

        rmsd = runner._compute_rmsd_values()
        # Freshly measured: ~1.5e-15 Å (machine precision) on this engine.
        # The old test asserted a loose <0.1 Å with no stated rationale;
        # tightened to 1e-6 Å -- far above float rounding noise, far below
        # any real structural deviation, so this still catches a broken
        # Kabsch alignment while not masking a real regression the way the
        # old 0.1 Å band would.
        assert rmsd[0] < 1e-6, f"Expected RMSD~0 for native structure, got {rmsd[0]:.3e} Å"

    def test_returns_zero_without_ref_pdb(self) -> None:
        """Without reference PDB, returns np.array([0.0]) not None."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        rmsd = runner._compute_rmsd_values()
        assert rmsd[0] == 0.0


# ── Convergence-window logic ────────────────────────────────────────


class TestConvergenceWindow:
    def test_off_without_a_threshold(self) -> None:
        """The early stop is opt-in: with no q_threshold (the default) a
        history of folded cycles never stops the run."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        assert runner.q_threshold is None
        runner.convergence_history = [1.0] * 50
        assert runner._check_folding_convergence() is False

    def test_false_on_empty_history(self) -> None:
        runner = _make_minimal_folding_runner(reference_pdb=None)
        runner.convergence_history = []
        assert runner._check_folding_convergence() is False

    def test_false_when_window_not_yet_full(self) -> None:
        runner = _make_minimal_folding_runner(reference_pdb=None)
        runner.convergence_window = 10
        runner.q_threshold = 0.75
        runner.convergence_history = [0.9] * 5
        assert runner._check_folding_convergence() is False

    def test_true_when_full_window_above_threshold(self) -> None:
        runner = _make_minimal_folding_runner(reference_pdb=None)
        runner.convergence_window = 10
        runner.q_threshold = 0.75
        runner.convergence_history = [0.8] * 10
        assert runner._check_folding_convergence() is True

    def test_false_when_one_recent_entry_below_threshold(self) -> None:
        runner = _make_minimal_folding_runner(reference_pdb=None)
        runner.convergence_window = 5
        runner.q_threshold = 0.75
        runner.convergence_history = [0.8, 0.8, 0.7, 0.8, 0.8]
        assert runner._check_folding_convergence() is False

    def test_only_checks_recent_window(self) -> None:
        """Old below-threshold entries outside the window must not block convergence."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        runner.convergence_window = 5
        runner.q_threshold = 0.75
        runner.convergence_history = [0.3] * 10 + [0.9] * 5
        assert runner._check_folding_convergence() is True


class TestConvergenceHistoryUpdates:
    def test_each_call_appends_one_float(self, chignolin_pdb_path: str) -> None:
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        assert runner.convergence_history == []
        runner._update_convergence_history()
        assert len(runner.convergence_history) == 1
        assert isinstance(runner.convergence_history[0], float)
        runner._update_convergence_history()
        assert len(runner.convergence_history) == 2

    def test_entries_in_range_0_1(self, chignolin_pdb_path: str) -> None:
        runner = _make_minimal_folding_runner(chignolin_pdb_path)
        for _ in range(5):
            runner._update_convergence_history()
        assert all(0.0 <= val <= 1.0 for val in runner.convergence_history)


# ── Q / convergence-history in the checkpoint payload ────────────────


def test_save_checkpoint_captures_q_values(chignolin_pdb_path: str, tmp_path: Path, monkeypatch) -> None:
    """Q values must be stored in the saved checkpoint (not silently dropped
    behind a guard that only fires when some optional attribute is set)."""
    runner = _make_minimal_folding_runner(chignolin_pdb_path)
    loaded = _save_and_reload(runner, tmp_path, monkeypatch, cycle=1)
    q = loaded.native_contacts_fraction
    assert q is not None
    assert isinstance(q, np.ndarray)
    assert q.shape == (1,)


def test_save_checkpoint_captures_convergence_history(
    chignolin_pdb_path: str, tmp_path: Path, monkeypatch
) -> None:
    """Convergence history must survive checkpoint round-trip."""
    runner = _make_minimal_folding_runner(chignolin_pdb_path)
    runner.convergence_history = [0.5, 0.6, 0.7]
    loaded = _save_and_reload(runner, tmp_path, monkeypatch, cycle=3)
    assert loaded.convergence_history == [0.5, 0.6, 0.7]


# ── FoldingBias / BasinTracker were removed ──────────────────────────


class TestSkeletonsRemoved:
    """``FoldingBias`` and ``BasinTracker`` never acted on a run
    (``enable_bias=True`` ran unbiased), so they are gone, with the six
    FoldingRunner arguments that built them."""

    def test_not_exported(self) -> None:
        import pymcpu.sampling as sampling

        for name in ("FoldingBias", "BasinTracker"):
            assert not hasattr(sampling, name)
            assert name not in sampling.__all__
        with pytest.raises(ModuleNotFoundError):
            __import__("pymcpu.sampling.folding_bias")

    @pytest.mark.parametrize(
        "name", ["enable_bias", "bias_k", "bias_r0", "bias_mode", "enable_basins", "n_basins"]
    )
    def test_folding_runner_argument_is_gone(self, name: str) -> None:
        assert name not in _INIT_DEFAULTS

    def test_an_old_checkpoint_with_their_fields_loads(self, tmp_path: Path, monkeypatch) -> None:
        """A checkpoint written while they existed carries folding_bias_params,
        basin_assignments and folding_events; a runner loads it and ignores them."""
        from pymcpu.checkpointing import save_checkpoint

        runner_save = _make_minimal_folding_runner(reference_pdb=None)
        runner_save.convergence_history = [0.4, 0.5]
        _save_checkpoint_with_random_coords(runner_save, tmp_path, monkeypatch, cycle=5)
        old = load_checkpoint(str(tmp_path / "last.chk"))
        old.update(
            folding_bias_params={"k": 2.0, "r0": 0.8, "mode": "harmonic"},
            basin_assignments=np.array([1], dtype=np.int32),
            folding_events=[{"cycle": 4, "Q": 0.5}],
        )
        save_checkpoint(old, checkpoint_dir=str(tmp_path / "old"), filename="last.chk")

        runner_load = _make_minimal_folding_runner(reference_pdb=None)
        state = runner_load.load_checkpoint(str(tmp_path / "old" / "last.chk"))

        assert state.cycle == 5
        assert runner_load.convergence_history == [0.4, 0.5]
        for name in ("folding_bias", "basin_tracker", "folding_events"):
            assert not hasattr(runner_load, name)
