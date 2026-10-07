"""Self-consistency tests for ``FoldingRunner``'s folding-progress metrics.

Covers native-contact detection, Q (fraction of native contacts formed) and
CA RMSD-to-native computation, convergence-window logic, and the
``FoldingBias`` / ``BasinTracker`` skeleton classes (including their
checkpoint round-trip). None of this compares against legacy MCPU -- legacy
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
from pymcpu.sampling.folding_bias import BasinTracker, FoldingBias  # noqa: E402

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
    enable_bias: bool = False,
    bias_k: float = 0.0,
    bias_r0: float = 0.0,
    bias_mode: str = "none",
    enable_basins: bool = False,
    n_basins: int = 1,
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
    runner.folding_events = []
    runner.checkpoint_config = CheckpointConfig()
    runner.traj_reporters = {}
    runner.traj_frame_counts = {}
    runner.sample_writer = None
    runner.seed = _INIT_DEFAULTS["seed"].default
    runner.temperature = _INIT_DEFAULTS["temperature"].default
    runner.fixed_residues = []
    runner.linker_residues = []
    runner.linker_energy_mode = _INIT_DEFAULTS["linker_energy_mode"].default
    runner.folding_bias = (
        FoldingBias(k=bias_k, r0=bias_r0, mode=bias_mode) if enable_bias else None
    )
    runner.basin_tracker = (
        BasinTracker(n_basins=n_basins, n_replicas=1) if enable_basins else None
    )
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


# ── FoldingBias ────────────────────────────────────────────────────


class TestFoldingBias:
    def test_default_mode_is_none(self) -> None:
        b = FoldingBias()
        assert b.mode == "none"
        assert b.k == 0.0
        assert b.r0 == 0.0

    def test_apply_none_mode_returns_zero(self) -> None:
        b = FoldingBias(mode="none", k=5.0, r0=0.8)
        assert b.apply(np.zeros((10, 3)), q=0.5) == 0.0

    def test_apply_harmonic(self) -> None:
        b = FoldingBias(mode="harmonic", k=2.0, r0=0.8)
        energy = b.apply(np.zeros((10, 3)), q=0.6)
        expected = 0.5 * 2.0 * (0.6 - 0.8) ** 2  # closed-form harmonic well
        assert abs(energy - expected) < 1e-9

    def test_get_set_params_round_trip(self) -> None:
        b1 = FoldingBias(k=3.0, r0=0.7, mode="harmonic")
        params = b1.get_params()

        b2 = FoldingBias()
        b2.set_params(params)

        assert b2.k == 3.0
        assert b2.r0 == 0.7
        assert b2.mode == "harmonic"

    def test_extra_params_preserved_round_trip(self) -> None:
        """Extra kwargs (forward-compatible funnel-bias params) must survive
        get/set round-trip."""
        b = FoldingBias(k=1.0, r0=0.5, mode="funnel", width=0.3, offset=0.1)
        params = b.get_params()
        assert params["width"] == 0.3
        assert params["offset"] == 0.1

        b2 = FoldingBias()
        b2.set_params(params)
        assert b2.extra["width"] == 0.3
        assert b2.extra["offset"] == 0.1


class TestBasinTracker:
    def test_default_assigns_all_to_basin_zero(self) -> None:
        bt = BasinTracker(n_basins=3, n_replicas=4)
        bt.update(np.array([0.5, 0.6, 0.7, 0.8]))
        assert all(bt.assignments == 0)

    def test_get_assignments_returns_a_copy(self) -> None:
        bt = BasinTracker(n_basins=2, n_replicas=2)
        a1 = bt.get_assignments()
        a1[0] = 99
        assert bt.assignments[0] == 0, "get_assignments must return a copy, not a reference"


# ── FoldingRunner <-> FoldingBias/BasinTracker wiring ────────────────


class TestFoldingRunnerBiasWiring:
    def test_bias_and_basins_none_by_default(self) -> None:
        """With enable_bias=False/enable_basins=False (defaults), both are None."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        assert runner.folding_bias is None
        assert runner.basin_tracker is None

    def test_bias_and_basins_set_when_enabled(self) -> None:
        runner = _make_minimal_folding_runner(
            reference_pdb=None,
            enable_bias=True,
            bias_k=1.5,
            bias_r0=0.7,
            bias_mode="harmonic",
            enable_basins=True,
            n_basins=2,
        )
        assert isinstance(runner.folding_bias, FoldingBias)
        assert isinstance(runner.basin_tracker, BasinTracker)
        assert runner.folding_bias.k == 1.5
        assert runner.folding_bias.mode == "harmonic"


class TestFoldingBiasCheckpointRoundTrip:
    def test_params_survive_round_trip_when_enabled(self, tmp_path: Path, monkeypatch) -> None:
        runner = _make_minimal_folding_runner(
            reference_pdb=None, enable_bias=True, bias_k=2.0, bias_r0=0.8, bias_mode="harmonic"
        )
        loaded = _save_and_reload(runner, tmp_path, monkeypatch, cycle=1)
        assert loaded.folding_bias_params == {"k": 2.0, "r0": 0.8, "mode": "harmonic"}

    def test_stores_none_when_disabled(self, tmp_path: Path, monkeypatch) -> None:
        """Regression coverage for a past bug where save_checkpoint gated
        this on hasattr(self, 'folding_bias') -- always True since the
        attribute exists whether or not bias is enabled, so the guard never
        actually distinguished enabled from disabled. Checked here by
        constructing a disabled runner and asserting the checkpoint really
        does come back None, not by grepping source text."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        loaded = _save_and_reload(runner, tmp_path, monkeypatch, cycle=1)
        assert loaded.folding_bias_params is None


class TestBasinTrackerCheckpointRoundTrip:
    def test_assignments_survive_round_trip_when_enabled(self, tmp_path: Path, monkeypatch) -> None:
        runner = _make_minimal_folding_runner(reference_pdb=None, enable_basins=True, n_basins=3)
        runner.basin_tracker.assignments = np.array([0, 2, 1], dtype=np.int32)
        loaded = _save_and_reload(runner, tmp_path, monkeypatch, cycle=2)
        np.testing.assert_array_equal(loaded.basin_assignments, [0, 2, 1])

    def test_assignments_restored_into_a_fresh_runner(self, tmp_path: Path, monkeypatch) -> None:
        """load_checkpoint must restore basin_tracker.assignments when enabled."""
        runner_save = _make_minimal_folding_runner(reference_pdb=None, enable_basins=True, n_basins=2)
        runner_save.basin_tracker.assignments = np.array([1, 0], dtype=np.int32)
        _save_checkpoint_with_random_coords(runner_save, tmp_path, monkeypatch, cycle=5)

        runner_load = _make_minimal_folding_runner(reference_pdb=None, enable_basins=True, n_basins=2)
        runner_load.checkpoint_config.checkpoint_dir = str(tmp_path)
        runner_load.load_checkpoint(str(tmp_path / "last.chk"))
        np.testing.assert_array_equal(runner_load.basin_tracker.assignments, [1, 0])

    def test_stores_none_when_disabled(self, tmp_path: Path, monkeypatch) -> None:
        """Save-side counterpart of the folding_bias disabled-guard regression
        test above, for basin_tracker: a runner with basins disabled must
        checkpoint basin_assignments=None, not crash on a hasattr guard that
        can't tell disabled from enabled."""
        runner = _make_minimal_folding_runner(reference_pdb=None)
        loaded = _save_and_reload(runner, tmp_path, monkeypatch, cycle=1)
        assert loaded.basin_assignments is None

    def test_disabled_loader_ignores_a_checkpoints_basin_data(self, tmp_path: Path, monkeypatch) -> None:
        """Load-side counterpart: a runner loading a checkpoint that *does*
        carry basin_assignments (saved by a basins-enabled run) must not
        crash when its own basin_tracker is None -- the restore is skipped,
        not force-applied to a tracker that doesn't exist."""
        runner_save = _make_minimal_folding_runner(reference_pdb=None, enable_basins=True, n_basins=2)
        runner_save.basin_tracker.assignments = np.array([1, 0], dtype=np.int32)
        _save_checkpoint_with_random_coords(runner_save, tmp_path, monkeypatch, cycle=5)

        runner_load = _make_minimal_folding_runner(reference_pdb=None)  # basins disabled
        runner_load.checkpoint_config.checkpoint_dir = str(tmp_path)
        runner_load.load_checkpoint(str(tmp_path / "last.chk"))  # must not raise
        assert runner_load.basin_tracker is None
