"""Reader for the KORP 6D knowledge-based energy map, plus a reference scorer.

The map file is not distributed with pyMCPU. See :func:`load_korp_map` for how
it is located and why it is not vendored.

Provenance
----------
Everything here is transcribed from the reference implementation published by
the Chacon lab (``github.com/chaconlab/Korp``, ``sbg/src/libenergy/korpe.cpp``):
``readMapHeader``, ``readMeshes``, ``getMeshes``, ``getMap``, ``readKORP``,
``frameCoord``, ``frames2ic`` and ``contact2bins``. Where the published paper
(Lopez-Blanco & Chacon, Bioinformatics 2019, 35(17):3013-3019) and that code
disagree, the code wins -- it is what built the released map. See
:func:`residue_frame` for the one place this actually matters.

This module is the *reference*, not the hot path. The production energy lives in
C++ (``forces/korp/common/``); :func:`score_structure` here is O(N^2) and exists
to validate that C++ against the upstream ``korpe`` binary.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "KORP_RESIDUE_ORDER",
    "KorpMap",
    "KorpMapError",
    "load_korp_map",
    "residue_frame",
    "pair_coordinates",
    "score_structure",
]


class KorpMapError(ValueError):
    """The map file is absent, truncated, or not a 6D residue-level KORP map."""


#: KORP's residue ordering: alphabetical by ONE-letter code. This is NOT
#: pyMCPU's ordering (``AMINO_INDEX`` is alphabetical by three-letter code), and
#: silently using the wrong one produces a plausible but wrong energy.
#: Transcribed from ``korpm.h``'s ``aasAA[]``.
KORP_RESIDUE_ORDER = (
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",
)
_KORP_INDEX = {name: i for i, name in enumerate(KORP_RESIDUE_ORDER)}

#: ``korpe.h``: sequence separations above this are all treated as non-bonding.
BONDING_THR = 9

#: The only frame model this reader accepts. Model 10 builds the frame from
#: N/CA/C and uses CA for distances, which is what makes KORP sidechain
#: independent. Models 11-14 move the origin to CB/O/C/N; model 11 would need CB
#: and would break that property, so they are rejected rather than guessed at.
_FRAME_MODEL_NCAC = 10

# Frame model 10: one frame per residue, 20 interacting residue types.
_NFT = 1
_NTYPES = 20

_HEADER_STRUCT = struct.Struct("<i f b i i f i b b b b i i")
assert _HEADER_STRUCT.size == 37, _HEADER_STRUCT.size


@dataclass(frozen=True)
class _Shell:
    """One radial shell's angular tessellation (``getMeshes``)."""

    ncells: int
    nring: int
    ncellsring: np.ndarray  # int32[nring]
    icell: np.ndarray       # int32[nring], first flat cell index of each ring
    theta: np.ndarray       # float32[nring], ring UPPER latitude boundaries
    dpsi: np.ndarray        # float32[nring], azimuthal bin width per ring
    dchi: float

    @property
    def nchi(self) -> int:
        # getMap sizes the chi axis with roundf(2*pi/dchi); match it exactly.
        return int(round(2.0 * np.pi / self.dchi))


@dataclass
class KorpMap:
    """A parsed KORP 6D map: header, tessellation, and the energy table."""

    path: Path
    sha256: str | None
    # header
    dimensions: int
    cutoff: float
    frame_model: int
    nonbonding: int
    nonbonding2: int
    bonding_factor: float
    ngauss: int
    fullgauss: bool
    use_ji: bool
    each_bonding: bool
    use_bonding: bool
    # tessellation
    nr: int
    br: np.ndarray          # float32[nr+1], shell boundaries; br[0] == min_r
    shells: tuple[_Shell, ...]
    # slice mapping
    nslices: int
    smapping: np.ndarray    # int8[10]; -1 == excluded, else slice index
    fmapping: np.ndarray    # float32[10]; per-slice weight
    # payload
    table: np.ndarray       # float32, flat
    shell_off: np.ndarray   # int64[nr+1], float offsets within one (s,i,j) block
    block_stride: int       # floats per (s,i,j) block

    @property
    def min_r(self) -> float:
        return float(self.br[0])

    def block_base(self, s: int, a: int, b: int) -> int:
        """Flat float offset of the ``(slice, typeA, typeB)`` block."""
        return ((s * _NTYPES + a) * _NTYPES + b) * self.block_stride

    def lookup(self, s: int, a: int, b: int, ir: int,
               cell_a: int, cell_b: int, ic: int) -> float:
        sh = self.shells[ir]
        off = (self.block_base(s, a, b) + int(self.shell_off[ir])
               + (cell_a * sh.ncells + cell_b) * sh.nchi + ic)
        return float(self.table[off])

    def engine_arrays(self) -> dict:
        """Flatten the tessellation into what the C++ map view takes.

        The per-shell ring arrays are ragged in general -- the format allows a
        different mesh per radial shell -- so they are concatenated with a
        ``ring_offset`` index rather than assumed rectangular. The released
        map happens to use the same 6-ring mesh in all ten shells, but relying
        on that would make a differently-binned map read neighbouring cells
        instead of failing.
        """
        ring_offset = [0]
        theta, dpsi, ncells, first_cell = [], [], [], []
        for shell in self.shells:
            theta.extend(float(x) for x in shell.theta)
            dpsi.extend(float(x) for x in shell.dpsi)
            ncells.extend(int(x) for x in shell.ncellsring)
            first_cell.extend(int(x) for x in shell.icell)
            ring_offset.append(len(theta))
        return {
            "cutoff": float(self.cutoff),
            "min_r": float(self.min_r),
            "nslices": int(self.nslices),
            "br": [float(x) for x in self.br],
            "shell_ncells": [int(s.ncells) for s in self.shells],
            "shell_nchi": [int(s.nchi) for s in self.shells],
            "shell_dchi": [float(s.dchi) for s in self.shells],
            "shell_offset": [int(x) for x in self.shell_off],
            "ring_offset": ring_offset,
            "ring_theta": theta,
            "ring_dpsi": dpsi,
            "ring_ncells": ncells,
            "ring_first_cell": first_cell,
            "smapping": [int(x) for x in self.smapping],
            "fmapping": [float(x) for x in self.fmapping],
        }

    def slice_for_separation(self, sep: int) -> int:
        """``smapping``: slice index for a sequence separation, or -1 to skip."""
        if sep > BONDING_THR:
            sep = 0
        return int(self.smapping[sep])


def _read_exact(fh, struct_or_dtype, count=None):
    if count is None:
        data = fh.read(struct_or_dtype.size)
        if len(data) != struct_or_dtype.size:
            raise KorpMapError("truncated map: header ended early")
        return struct_or_dtype.unpack(data)
    nbytes = np.dtype(struct_or_dtype).itemsize * count
    data = fh.read(nbytes)
    if len(data) != nbytes:
        raise KorpMapError("truncated map: mesh block ended early")
    return np.frombuffer(data, dtype=struct_or_dtype, count=count).copy()


def _derive_slice_mapping(nonbonding, nonbonding2, bonding_factor,
                          each_bonding, use_bonding):
    """Transcribed from ``readKORP``. Returns (nslices, smapping, fmapping)."""
    fmapping = np.zeros(10, dtype=np.float32)
    fmapping[0] = 1.0  # non-bonding is the reference and is always 1.0
    if each_bonding:
        nslices = 1 + nonbonding2 - nonbonding
        fmapping[1:nslices] = bonding_factor
    elif use_bonding:
        nslices = 2
        fmapping[1] = bonding_factor
    else:
        nslices = 1
    # NOTE: readKORP doubles nslices when (dimensions == 3 && use_ji). This
    # reader only accepts 6D maps, so that branch cannot apply here.

    smapping = np.zeros(10, dtype=np.int8)
    bi = 1
    for i in range(1, 10):
        if i <= nonbonding:
            smapping[i] = -1  # excluded entirely
        elif i > nonbonding2 or not use_bonding:
            smapping[i] = 0   # non-bonding
        elif each_bonding:
            smapping[i] = bi
            bi += 1
        else:
            smapping[i] = 1   # one bonding map for all local contacts
    return nslices, smapping, fmapping


def load_korp_map(path, *, mmap: bool = True, sha256: bool = False) -> KorpMap:
    """Parse a KORP 6D map file.

    Parameters
    ----------
    path
        The ``korp6Dv1.bin`` energy map. pyMCPU does not ship it: at 316 MiB it
        is well over PyPI's per-file limit, so it is supplied by whoever runs
        the simulation. The chaconlab.org distribution is BSD-3-Clause and does
        permit redistribution, so this is a packaging decision rather than a
        licensing one -- but note the same file is also published under
        NPOSL-3.0 in the ``korpm`` repository, so it is worth recording which
        copy you have.
    mmap
        Memory-map the table instead of reading it. Default True, which matters
        for replica exchange: the OS page cache then shares one copy across every
        rank on the node instead of giving each its own ~316 MiB.
    sha256
        Also digest the file, so a recorded energy is traceable to a specific
        map. Costs a full read, so it is off by default.

    Raises
    ------
    KorpMapError
        If the file is missing, truncated, or is not a 6D N/CA/C-frame binned
        map. Every check is fatal on purpose: a map this reader half-understands
        would score silently wrong rather than fail.
    """
    path = Path(path)
    if not path.is_file():
        raise KorpMapError(f"KORP energy map not found: {path}")

    with open(path, "rb") as fh:
        (dimensions, cutoff, frame_model, nonbonding, nonbonding2,
         bonding_factor, ngauss, fullgauss, use_ji, each_bonding,
         _unused_c, _unused_i0, _unused_i1) = _read_exact(fh, _HEADER_STRUCT)

        if dimensions != 6:
            raise KorpMapError(
                f"{path}: dimensions={dimensions}; this reader handles only the "
                "6D orientational map."
            )
        if frame_model != _FRAME_MODEL_NCAC:
            raise KorpMapError(
                f"{path}: frame_model={frame_model}, expected "
                f"{_FRAME_MODEL_NCAC} (N/CA/C frame, CA distances). Models "
                "11-14 place the frame origin on CB/O/C/N; model 11 needs CB "
                "and so is not sidechain independent, which is the property "
                "pyMCPU's backbone-only KORP force field relies on."
            )
        if ngauss > 0:
            raise KorpMapError(
                f"{path}: ngauss={ngauss} marks a Gaussian-mixture map; only "
                "binned maps are supported."
            )

        use_bonding = nonbonding2 >= 0

        nr = _read_exact(fh, struct.Struct("<i"))[0]
        br = _read_exact(fh, np.dtype("<f4"), nr + 1)
        _scell = _read_exact(fh, np.dtype("<i4"), nr)

        shells = []
        for r in range(nr):
            ncells, nring = _read_exact(fh, struct.Struct("<ii"))
            ncellsring = _read_exact(fh, np.dtype("<i4"), nring)
            icell = _read_exact(fh, np.dtype("<i4"), nring)
            theta = _read_exact(fh, np.dtype("<f4"), nring)
            dpsi = _read_exact(fh, np.dtype("<f4"), nring)
            dchi = _read_exact(fh, struct.Struct("<f"))[0]
            if ncells != int(_scell[r]):
                raise KorpMapError(
                    f"{path}: shell {r} declares ncells={ncells} but the radial "
                    f"block said {int(_scell[r])}."
                )
            shells.append(_Shell(int(ncells), int(nring), ncellsring, icell,
                                 theta, dpsi, float(dchi)))
        shells = tuple(shells)

        if abs(float(br[nr]) - float(cutoff)) > 1e-4:
            raise KorpMapError(
                f"{path}: outer shell boundary {float(br[nr])} disagrees with "
                f"the header cutoff {float(cutoff)}."
            )

        nslices, smapping, fmapping = _derive_slice_mapping(
            nonbonding, nonbonding2, bonding_factor, each_bonding, use_bonding
        )

        # Per-shell float offsets inside one (slice, typeA, typeB) block.
        shell_off = np.zeros(nr + 1, dtype=np.int64)
        for r, sh in enumerate(shells):
            shell_off[r + 1] = shell_off[r] + sh.ncells * sh.ncells * sh.nchi
        block_stride = int(shell_off[nr])

        payload_offset = fh.tell()
        n_floats = nslices * _NFT * _NFT * _NTYPES * _NTYPES * block_stride
        expected = payload_offset + 4 * n_floats
        actual = path.stat().st_size
        if actual != expected:
            # This single equality is a complete structural checksum of the
            # decode: a misread nr, cell count, chi count or slice count all
            # land here rather than silently shifting every lookup.
            raise KorpMapError(
                f"{path}: size {actual} != expected {expected} "
                f"(prefix {payload_offset} + {n_floats} floats from "
                f"nslices={nslices}, nr={nr}, "
                f"cells={[s.ncells for s in shells]}, "
                f"nchi={[s.nchi for s in shells]}). The header parsed, so the "
                "layout assumption is what is wrong."
            )

    if mmap:
        table = np.memmap(path, dtype="<f4", mode="r",
                          offset=payload_offset, shape=(n_floats,))
    else:
        table = np.fromfile(path, dtype="<f4", offset=payload_offset,
                            count=n_floats)

    digest = None
    if sha256:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digest = h.hexdigest()

    return KorpMap(
        path=path, sha256=digest,
        dimensions=int(dimensions), cutoff=float(cutoff),
        frame_model=int(frame_model), nonbonding=int(nonbonding),
        nonbonding2=int(nonbonding2), bonding_factor=float(bonding_factor),
        ngauss=int(ngauss), fullgauss=bool(fullgauss), use_ji=bool(use_ji),
        each_bonding=bool(each_bonding), use_bonding=bool(use_bonding),
        nr=int(nr), br=br, shells=shells,
        nslices=int(nslices), smapping=smapping, fmapping=fmapping,
        table=table, shell_off=shell_off, block_stride=block_stride,
    )


def residue_frame(n, ca, c):
    """Local orthonormal frame for one residue, from its own N, CA and C.

    Returns ``(origin, R)`` where ``R``'s columns are ``(vx, vy, vz)`` and the
    origin is CA.

    The published Eq. (2) and the released code disagree here, and the code
    wins because the code is what built the map. The paper writes
    ``vy ~ vz x (N - CA)``; ``frameCoord`` computes ``vy ~ vz x (C - CA)``.
    Since ``(N-CA) = lambda*vz - (C-CA)``, those two cross products are exact
    negatives, so the paper's frame is the code's rotated 180 degrees about
    ``vz``. Both are right-handed, so nothing looks wrong -- but every psi
    angle shifts by pi and lands in a different bin.
    """
    n = np.asarray(n, dtype=np.float64)
    ca = np.asarray(ca, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    r12 = n - ca
    r13 = c - ca
    vz = r12 + r13
    vz /= np.linalg.norm(vz)
    vy = np.cross(vz, r13)
    vy /= np.linalg.norm(vy)
    vx = np.cross(vy, vz)
    return ca, np.stack([vx, vy, vz], axis=1)


def _dihedral_unit_negated(ua, ub, uc):
    """``dihedral3DunitN``: the variant whose first cross product is negated."""
    v1 = -np.cross(ua, ub)
    v2 = np.cross(ub, uc)
    v3 = np.cross(v1, ub)
    return np.arctan2(np.dot(v3, v2), np.dot(v1, v2))


def pair_coordinates(frame_a, frame_b):
    """The six KORP coordinates ``(d, thetaA, psiA, thetaB, psiB, chi)``.

    ``frame_a`` must be the lower residue index: the map is not symmetric under
    swapping the partners.
    """
    (pa, ra), (pb, rb) = frame_a, frame_b
    vxa, vya, vza = ra[:, 0], ra[:, 1], ra[:, 2]
    vxb, vyb, vzb = rb[:, 0], rb[:, 1], rb[:, 2]

    rab = pb - pa
    rba = pa - pb
    d = float(np.linalg.norm(rab))

    def _ang(u, v):
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        return float(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)))

    theta_a = _ang(vza, rab)
    theta_b = _ang(vzb, rba)

    def _psi(r, vx, vy, vz):
        v2 = r - vz * (np.dot(r, vz) / np.dot(vz, vz))
        psi = _ang(vx, v2)
        if np.dot(vy, v2) < 0.0:
            psi = -psi
        return psi + np.pi

    psi_a = _psi(rab, vxa, vya, vza)
    psi_b = _psi(rba, vxb, vyb, vzb)
    chi = np.pi + _dihedral_unit_negated(vza, rab / d, vzb)
    return d, theta_a, psi_a, theta_b, psi_b, chi


def contact_bins(kmap: KorpMap, d, theta_a, psi_a, theta_b, psi_b, chi):
    """``contact2bins``: the six coordinates to flat bin indices.

    The caller must already have checked ``min_r < d < cutoff``; upstream's
    radial loop has no upper bound and walks off the end otherwise.
    """
    ir = 0
    while d > kmap.br[ir + 1]:
        ir += 1
    sh = kmap.shells[ir]

    def _cell(theta, psi):
        it = 0
        while theta > sh.theta[it] and it < sh.nring - 1:
            it += 1
        ip = int(psi / sh.dpsi[it])
        if ip >= sh.ncellsring[it]:
            ip = int(sh.ncellsring[it]) - 1
        return int(sh.icell[it]) + ip

    ic = int(chi / sh.dchi)
    if ic >= sh.nchi:
        ic = sh.nchi - 1
    return ir, _cell(theta_a, psi_a), _cell(theta_b, psi_b), ic


def score_structure(kmap: KorpMap, coords, res_names, res_seq=None,
                    chain_ids=None) -> float:
    """Reference KORP energy for one structure. O(N^2); for validation only.

    Parameters
    ----------
    coords
        ``(n_res, 3, 3)`` array of N, CA, C positions in Angstrom.
    res_names
        Three-letter residue names.
    res_seq
        PDB residue numbers. KORP derives sequence separation from these, not
        from array position, so renumbered or gapped input scores differently
        from upstream. Defaults to ``0..n-1``.
    chain_ids
        Per-residue chain identifiers; pairs across chains are always
        non-bonding. Defaults to a single chain.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_res = len(res_names)
    if coords.shape != (n_res, 3, 3):
        raise ValueError(f"coords must be (n_res, 3, 3), got {coords.shape}")
    if res_seq is None:
        res_seq = np.arange(n_res)
    if chain_ids is None:
        chain_ids = ["A"] * n_res

    try:
        types = [_KORP_INDEX[name] for name in res_names]
    except KeyError as exc:
        raise ValueError(f"residue {exc.args[0]!r} has no KORP type") from exc

    frames = [residue_frame(coords[i, 0], coords[i, 1], coords[i, 2])
              for i in range(n_res)]
    ca = np.array([f[0] for f in frames])

    cutoff2 = kmap.cutoff ** 2
    min_r = kmap.min_r
    total = 0.0  # accumulate in float64, as korpe does
    for i in range(n_res):
        for j in range(i + 1, n_res):
            delta = ca[j] - ca[i]
            d2 = float(delta @ delta)
            if d2 >= cutoff2:
                continue
            sep = 0 if chain_ids[i] != chain_ids[j] else abs(
                int(res_seq[j]) - int(res_seq[i]))
            s = kmap.slice_for_separation(sep)
            if s < 0:
                continue
            coords6 = pair_coordinates(frames[i], frames[j])
            if coords6[0] <= min_r:
                continue
            ir, cell_a, cell_b, ic = contact_bins(kmap, *coords6)
            total += (float(kmap.fmapping[s])
                      * kmap.lookup(s, types[i], types[j], ir, cell_a, cell_b, ic))
    return total
