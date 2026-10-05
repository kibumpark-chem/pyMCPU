"""HDF5 / NPZ writers and reduced-potential helpers for pymbar."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

KB_DEFAULT = 1.0


def reduced_potentials_from_arrays(
    energy_unbiased: np.ndarray,
    N: np.ndarray,
    temperatures: np.ndarray,
    n_targets: np.ndarray,
    k_bias: float,
    *,
    kB: float = KB_DEFAULT,
) -> np.ndarray:
    """Return ``u[k, n] = (E_n + 0.5*k*(N_n - N0_k)^2) / (kB * T_k)``.

    Parameters
    ----------
    energy_unbiased :
        Shape ``(n_samples,)`` unbiased energies.
    N :
        Shape ``(n_samples,)`` hard native-contact counts.
    temperatures :
        Shape ``(n_temps,)``.
    n_targets :
        Shape ``(n_windows,)`` umbrella centers N0.
    k_bias :
        Harmonic strength on N.
    kB :
        Boltzmann constant in simulation units (default 1.0).

    Returns
    -------
    u_kn :
        Shape ``(n_states, n_samples)`` with
        ``n_states = n_temps * n_windows`` and state ordering
        ``k = temp_index * n_windows + n_index``.
    """
    e = np.asarray(energy_unbiased, dtype=np.float64).reshape(-1)
    n_vals = np.asarray(N, dtype=np.float64).reshape(-1)
    if e.shape != n_vals.shape:
        raise ValueError("energy_unbiased and N must have the same length")

    temps = np.asarray(temperatures, dtype=np.float64).reshape(-1)
    targets = np.asarray(n_targets, dtype=np.float64).reshape(-1)
    if temps.size < 1 or targets.size < 1:
        raise ValueError("temperatures and n_targets must be non-empty")

    n_samples = e.size
    n_temps = temps.size
    n_win = targets.size
    n_states = n_temps * n_win
    u = np.empty((n_states, n_samples), dtype=np.float64)

    bias = 0.5 * float(k_bias) * (n_vals[None, :] - targets[:, None]) ** 2
    # bias shape: (n_win, n_samples)
    for t_idx, temp in enumerate(temps):
        denom = float(kB) * float(temp)
        if denom == 0.0:
            raise ValueError("temperature (and kB) must be non-zero")
        for q_idx in range(n_win):
            k = t_idx * n_win + q_idx
            u[k, :] = (e + bias[q_idx]) / denom
    return u


def _load_arrays(path: Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to read HDF5 RE analysis files; "
                "install with: pip install h5py"
            ) from exc
        with h5py.File(path, "r") as h5:
            walker = h5["walker_id"] if "walker_id" in h5 else h5["replica_index"]
            data = {
                "state_index": np.asarray(h5["state_index"]),
                "energy_unbiased": np.asarray(h5["energy_unbiased"]),
                "N": np.asarray(h5["N"]),
                "cycle": np.asarray(h5["cycle"]),
                "walker_id": np.asarray(walker),
                "attrs": dict(h5.attrs),
            }
            # Normalize bytes attrs
            attrs = {}
            for key, val in data["attrs"].items():
                if isinstance(val, bytes):
                    attrs[key] = val.decode()
                elif hasattr(val, "tolist"):
                    attrs[key] = np.asarray(val)
                else:
                    attrs[key] = val
            data["attrs"] = attrs
            return data

    # NPZ fallback (+ optional meta.json)
    npz_path = path if path.suffix.lower() == ".npz" else path.with_suffix(".npz")
    if not npz_path.exists():
        raise FileNotFoundError(f"Analysis file not found: {path}")
    loaded = np.load(npz_path, allow_pickle=False)
    meta_path = npz_path.with_name(npz_path.stem + "_meta.json")
    if not meta_path.exists():
        meta_path = npz_path.with_name("meta.json")
    attrs: dict[str, Any] = {}
    if meta_path.exists():
        attrs = json.loads(meta_path.read_text())
    walker_key = "walker_id" if "walker_id" in loaded.files else "replica_index"
    return {
        "state_index": np.asarray(loaded["state_index"]),
        "energy_unbiased": np.asarray(loaded["energy_unbiased"]),
        "N": np.asarray(loaded["N"]),
        "cycle": np.asarray(loaded["cycle"]),
        "walker_id": np.asarray(loaded[walker_key]),
        "attrs": attrs,
    }


def reduced_potentials(path: str | Path) -> np.ndarray:
    """Load an RE analysis file and reconstruct reduced potentials ``u_kn``."""
    data = _load_arrays(Path(path))
    attrs = data["attrs"]
    temperatures = np.asarray(attrs["temperatures"], dtype=np.float64)
    n_targets = np.asarray(attrs["n_targets"], dtype=np.float64)
    k_bias = float(attrs["k_bias"])
    kB = float(attrs.get("kB", KB_DEFAULT))
    return reduced_potentials_from_arrays(
        data["energy_unbiased"],
        data["N"],
        temperatures,
        n_targets,
        k_bias,
        kB=kB,
    )


@dataclass
class _Buffers:
    state_index: list[int]
    energy_unbiased: list[float]
    N: list[float]
    cycle: list[int]
    walker_id: list[int]


class RexSampleWriter:
    """Accumulate per-cycle RE samples and flush to HDF5 or NPZ+meta.json.

    MBAR uses ``state_index``, ``energy_unbiased``, and ``N``. ``walker_id`` is
    for mixing diagnostics (which configuration lineage sits at each state).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        temperatures: Sequence[float],
        n_targets: Sequence[float],
        k_bias: float,
        n_contacts: int | float,
        kB: float = KB_DEFAULT,
        bias_cv: str = "hard_N",
    ):
        self.path = Path(path)
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.n_targets = np.asarray(n_targets, dtype=np.float64)
        self.k_bias = float(k_bias)
        self.n_contacts = float(n_contacts)
        self.kB = float(kB)
        self.bias_cv = str(bias_cv)
        self._buf = _Buffers([], [], [], [], [])
        self._closed = False
        self.n_samples_written = 0

    def append(
        self,
        *,
        state_index: int,
        energy_unbiased: float,
        N: float,
        cycle: int,
        walker_id: int,
    ) -> None:
        """Buffer one sample record. Raises ``RuntimeError`` once closed."""
        if self._closed:
            raise RuntimeError("RexSampleWriter is closed")
        self._buf.state_index.append(int(state_index))
        self._buf.energy_unbiased.append(float(energy_unbiased))
        self._buf.N.append(float(N))
        self._buf.cycle.append(int(cycle))
        self._buf.walker_id.append(int(walker_id))
        self.n_samples_written = len(self._buf.state_index)

    @property
    def n_samples(self) -> int:
        """Number of records buffered so far."""
        return len(self._buf.state_index)

    def n_frames_written(self) -> int:
        """Consistent interface with XtcReporter / EnergyReporter."""
        return int(self.n_samples_written)

    def load_existing(self, path: str | Path | None = None) -> int:
        """
        Seed the in-memory buffer from an on-disk HDF5/NPZ analysis file.

        Used after resume truncation so subsequent ``append`` / ``flush``
        preserve pre-checkpoint samples instead of overwriting them.
        """
        if self._closed:
            raise RuntimeError("RexSampleWriter is closed")
        src = Path(path) if path is not None else self.path
        candidates = [src]
        if src.suffix.lower() in {".h5", ".hdf5", ""}:
            candidates.append(src.with_suffix(".npz"))
        elif src.suffix.lower() == ".npz":
            candidates.append(src.with_suffix(".h5"))

        existing = next((p for p in candidates if p.is_file()), None)
        if existing is None:
            return 0

        data = _load_arrays(existing)
        n = int(np.asarray(data["state_index"]).reshape(-1).size)
        self._buf = _Buffers(
            [int(x) for x in np.asarray(data["state_index"]).reshape(-1).tolist()],
            [float(x) for x in np.asarray(data["energy_unbiased"]).reshape(-1).tolist()],
            [float(x) for x in np.asarray(data["N"]).reshape(-1).tolist()],
            [int(x) for x in np.asarray(data["cycle"]).reshape(-1).tolist()],
            [int(x) for x in np.asarray(data["walker_id"]).reshape(-1).tolist()],
        )
        self.n_samples_written = n
        self.path = existing
        return n

    def _meta(self) -> dict[str, Any]:
        return {
            "k_bias": self.k_bias,
            "n_contacts": self.n_contacts,
            "temperatures": self.temperatures.tolist(),
            "n_targets": self.n_targets.tolist(),
            "kB": self.kB,
            "bias_cv": self.bias_cv,
        }

    def flush(self) -> Path:
        """Write buffered samples. Prefers HDF5; falls back to NPZ + meta.json."""
        arrays = {
            "state_index": np.asarray(self._buf.state_index, dtype=np.int32),
            "energy_unbiased": np.asarray(self._buf.energy_unbiased, dtype=np.float64),
            "N": np.asarray(self._buf.N, dtype=np.float64),
            "cycle": np.asarray(self._buf.cycle, dtype=np.int32),
            "walker_id": np.asarray(self._buf.walker_id, dtype=np.int32),
        }
        meta = self._meta()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        use_h5 = self.path.suffix.lower() in {".h5", ".hdf5", ""}
        target = self.path if self.path.suffix else self.path.with_suffix(".h5")

        if use_h5:
            try:
                import h5py

                with h5py.File(target, "w") as h5:
                    for key, arr in arrays.items():
                        h5.create_dataset(key, data=arr)
                    h5.attrs["k_bias"] = self.k_bias
                    h5.attrs["n_contacts"] = self.n_contacts
                    h5.attrs["temperatures"] = self.temperatures
                    h5.attrs["n_targets"] = self.n_targets
                    h5.attrs["kB"] = self.kB
                    h5.attrs["bias_cv"] = self.bias_cv
                self.path = target
                self.n_samples_written = int(arrays["state_index"].shape[0])
                return target
            except Exception:
                # ImportError or ABI mismatch — NPZ fallback
                target = self.path.with_suffix(".npz") if self.path.suffix else Path(
                    str(self.path) + ".npz"
                )

        np.savez_compressed(target, **arrays)
        meta_path = target.with_name(target.stem + "_meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        self.path = target
        self.n_samples_written = int(arrays["state_index"].shape[0])
        return target

    def close(self) -> Path:
        """Flush, mark the writer closed, and return the path written."""
        out = self.flush()
        self._closed = True
        return out

    def __enter__(self) -> RexSampleWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            self.close()


def open_writer(
    path: str | Path,
    *,
    temperatures: Sequence[float],
    n_targets: Sequence[float],
    k_bias: float,
    n_contacts: int | float,
    **kwargs: Any,
) -> RexSampleWriter:
    return RexSampleWriter(
        path,
        temperatures=temperatures,
        n_targets=n_targets,
        k_bias=k_bias,
        n_contacts=n_contacts,
        **kwargs,
    )
