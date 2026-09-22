"""
folding_bias.py

Skeleton classes for folding funnel bias and basin tracking.
Both classes are fully serializable for checkpointing.

Extend FoldingBias.apply() and BasinTracker.update() with real
energy / clustering logic when needed.
"""

from __future__ import annotations

import numpy as np


# ── FoldingBias ───────────────────────────────────────────────


class FoldingBias:
    """
    Folding funnel bias potential.

    Currently a no-op skeleton. Extend apply() with a real
    energy term (harmonic well, funnel metadynamics, etc.).

    Checkpointing:
        get_params() returns a plain dict that is stored in
        FoldingCheckpointState.folding_bias_params.
        set_params() restores state from that dict.
        Both methods must remain inverse of each other.
    """

    def __init__(
        self,
        k: float = 0.0,
        r0: float = 0.0,
        mode: str = "none",
        **extra,
    ):
        """
        Args:
            k:     Force constant (kcal/mol or reduced units).
            r0:    Reference value (Q target or distance target).
            mode:  "harmonic", "funnel", or "none" (default).
            extra: Any additional parameters stored verbatim.
        """
        self.k = float(k)
        self.r0 = float(r0)
        self.mode = str(mode)
        self.extra = dict(extra)

    # ── Serialization ────────────────────────────────────────

    def get_params(self) -> dict:
        """
        Return all parameters as a JSON-serializable dict.
        This dict is stored verbatim in the checkpoint.
        """
        params = {"k": self.k, "r0": self.r0, "mode": self.mode}
        params.update(self.extra)
        return params

    def set_params(self, params: dict) -> None:
        """
        Restore parameters from a checkpoint dict.
        Unknown keys are stored in self.extra (forward-compatible).
        """
        self.k = float(params.get("k", self.k))
        self.r0 = float(params.get("r0", self.r0))
        self.mode = str(params.get("mode", self.mode))
        known = {"k", "r0", "mode"}
        self.extra = {k: v for k, v in params.items() if k not in known}

    # ── Energy ───────────────────────────────────────────────

    def apply(self, coords: np.ndarray, q: float) -> float:
        """
        Compute bias energy for current coordinates and Q value.

        Args:
            coords: (n_atoms, 3) coordinate array (Angstrom).
            q:      Current native contact fraction Q in [0, 1].

        Returns:
            Bias energy as a float (same units as k).
            Returns 0.0 in skeleton ("none") mode.
        """
        if self.mode == "harmonic":
            return 0.5 * self.k * (q - self.r0) ** 2
        if self.mode == "funnel":
            # Placeholder — extend with real funnel potential
            return 0.0
        return 0.0  # mode == "none"

    def __repr__(self) -> str:
        return (
            f"FoldingBias(mode={self.mode!r}, "
            f"k={self.k}, r0={self.r0})"
        )


# ── BasinTracker ──────────────────────────────────────────────


class BasinTracker:
    """
    Basin / cluster assignment tracker.

    Tracks which free-energy basin each replica currently occupies.
    Currently a skeleton that assigns all replicas to basin 0.
    Extend update() with real clustering (e.g. k-means on Q,
    RMSD-based, or MSM state assignments).

    Checkpointing:
        self.assignments is saved as-is in
        FoldingCheckpointState.basin_assignments (numpy array).
        Restored directly in load_checkpoint().
    """

    def __init__(self, n_basins: int = 1, n_replicas: int = 1):
        """
        Args:
            n_basins:   Number of distinct basins to track.
            n_replicas: Number of replicas (sets initial array size).
        """
        self.n_basins = int(n_basins)
        self.n_replicas = int(n_replicas)
        self.assignments = np.zeros(n_replicas, dtype=np.int32)

    def update(self, q_values: np.ndarray) -> None:
        """
        Update basin assignments given current Q values.

        Args:
            q_values: np.ndarray of shape (n_replicas,) in [0, 1].

        Skeleton behaviour: all replicas assigned to basin 0.
        Override with real clustering logic when needed.
        """
        self.assignments = np.zeros(len(q_values), dtype=np.int32)

    def get_assignments(self) -> np.ndarray:
        """Return a copy of current basin assignments."""
        return self.assignments.copy()

    def __repr__(self) -> str:
        return (
            f"BasinTracker(n_basins={self.n_basins}, "
            f"assignments={self.assignments.tolist()})"
        )
