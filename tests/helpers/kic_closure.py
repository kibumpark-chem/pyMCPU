"""The exact distribution of a chain whose only move is one KIC window.

All residues but four consecutive ones, a proline P and the three after it,
are fixed, and KIC is the only move. The engine can then draw one window and
driver only: the window P+1..P+3 with the psi driver (psi of P). The phi-driver
window P..P+3 would change the proline's phi and is refused, and every other
window would move a fixed residue. A state is the driver angle
theta = psi(P) and one closure of the window for that theta, so the states
form closed curves. Uniform measure on the seven torsions, restricted to these
curves, gives the closure J = 1 / |det M| per unit of theta (M: the twists of
the window's six torsion axes, as in docs/physics_notes/kic_jacobian.md), so
the exact distribution has density J exp(-E / T) per unit of theta, zero
where the engine's hard-core check finds a clash.

The closures are found here independently of the engine's degree-16
polynomial solver, from the start structure's KIC targets
(``System.get_kic_reference``):

* CA(P+2) lies on the circle of points at the right distances from CA(P+1)
  and CA(P+3); alpha is its angle on that circle.
* The peptide unit CA(P+1)..CA(P+2) is rigid and can turn about its CA-CA
  axis; psi1 is that turn, measured from the N(P+1) side. The N-CA-C angle
  of residue P+1 fixes cos(psi1) = q1(theta, alpha). Likewise psi3 for the
  unit CA(P+2)..CA(P+3): cos(psi3) = q3(theta, alpha).
* F(theta, alpha, psi1, psi3) = 0 is the N-CA-C angle of residue P+2.

These three smooth equations in x = (theta, alpha, psi1, psi3) define the
curves. Eliminating psi1 and psi3 (two signs each) gives a scalar equation in
(theta, alpha) that marching squares can scan, but the curves touch the
edges of each sign's domain tangentially where the two signs meet, and a grid
loses curve there. So the scan only seeds a pseudo-arclength continuation in
x, which follows each closed curve through those points. Per unit arc length
s the density is J |dtheta/ds| exp(-E / T); J alone diverges where theta
turns back along a curve, but the product does not.

The energy is binned, so the density jumps along a curve; the curve is
sampled every ``h / k`` radians and integrated by the trapezoid rule.
"""
from __future__ import annotations

import math
from pathlib import Path

import mdtraj as md
import numpy as np
from scipy.spatial import cKDTree

from pymcpu import mcpu_core
from pymcpu.forcefields.mcpu import MCPUForceField


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _dihedrals(p0, p1, p2, p3):
    b1 = _unit(p2 - p1)
    v = p0 - p1
    v = v - np.sum(v * b1, -1, keepdims=True) * b1
    w = p3 - p2
    w = w - np.sum(w * b1, -1, keepdims=True) * b1
    return np.arctan2(np.sum(np.cross(b1, v) * w, -1), np.sum(v * w, -1))


def _place(a, b, c, bond, angle, torsion):
    """Point d with |cd| = bond, angle bcd = angle and dihedral abcd = torsion."""
    bc = _unit(c - b)
    n = _unit(np.cross(b - a, bc))
    m = np.cross(n, bc)
    return c + bond * (-math.cos(angle) * bc + math.sin(angle) * (math.cos(torsion) * m
                                                                 + math.sin(torsion) * n))


def _frames(p1, center, p3):
    """Orthonormal frames (columns) built from three points, as the engine's
    rigid transfer builds them."""
    v1 = _unit(p1 - center)
    v2 = _unit(p3 - center)
    bis = _unit(v1 + v2)
    nrm = _unit(np.cross(v1, v2))
    return np.stack([bis, np.cross(nrm, bis), nrm], axis=-1)


def _rotate(points, center, axis, angle):
    """Rodrigues rotation of fixed points (3,) by angles (n,) -> (n, 3)."""
    v = points - center
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    return center + v * c + np.cross(axis, v) * s + axis * np.dot(v, axis) * (1 - c)


class KicWindow:
    """One KIC window with the psi driver; every other residue fixed.

    ``pdb`` and ``residues`` (first, last; inclusive, 0-based) choose the
    chain; ``proline`` is the 0-based index, within that chain, of the
    proline whose psi drives the window.
    """

    def __init__(self, pdb: str | Path, proline: int, residues: tuple[int, int] | None = None,
                 reorder: str = "off"):
        traj = md.load(str(pdb))
        select = "not element H"
        if residues is not None:
            select += f" and resid {residues[0]} to {residues[1]}"
        heavy = traj.atom_slice(traj.topology.select(select))
        self.ff = MCPUForceField(heavy)
        self.system = self.ff.create_system(heavy.topology)
        self.reorder = reorder
        self.start32 = (self.ff.coords[0] * 10.0).T.astype(np.float32)
        atoms = self.ff.ordered_atom_list
        self.res = np.array([a.residue_index for a in atoms])
        self.names = np.array([a.name for a in atoms])
        self.n_res = int(self.res.max()) + 1
        self.r = r = proline + 1
        self.free = (proline, r, r + 1, r + 2)
        if not (1 <= proline and r + 3 < self.n_res):
            raise ValueError("the window needs a fixed residue on each side")
        if not self.system.is_proline(proline) or any(self.system.is_proline(k) for k in self.free[1:]):
            raise ValueError("residue `proline` must be the only proline among the four free ones")
        self.fixed = [k for k in range(self.n_res) if k not in self.free]
        self.base = np.array(self.new_context().coords, dtype=np.float64)
        ref = {k: np.asarray(v, dtype=np.float64) for k, v in self.system.get_kic_reference().items()}
        I = self.atom
        b = self.base
        self.i_n0, self.i_a0, self.i_c0, self.i_o0 = I(r - 1, "N"), I(r - 1, "CA"), I(r - 1, "C"), I(r - 1, "O")
        self.i_n1, self.i_a1, self.i_c1, self.i_o1 = I(r, "N"), I(r, "CA"), I(r, "C"), I(r, "O")
        self.i_n2, self.i_a2, self.i_c2, self.i_o2 = I(r + 1, "N"), I(r + 1, "CA"), I(r + 1, "C"), I(r + 1, "O")
        self.i_n3, self.i_a3, self.i_c3 = I(r + 2, "N"), I(r + 2, "CA"), I(r + 2, "C")
        backbone = {"N", "CA", "C", "O"}
        self.sidechains = {k: [i for i in np.nonzero(self.res == k)[0] if self.names[i] not in backbone]
                           for k in (r, r + 1, r + 2)}
        # Driver: psi(r-1) turns N(r), CA(r) and O(r-1) about CA(r-1) -> C(r-1).
        self.drv_center = b[:, self.i_c0]
        self.drv_axis = _unit(b[:, self.i_c0] - b[:, self.i_a0])
        quads = [(I(r - 1, "N"), I(r - 1, "CA"), I(r - 1, "C"), I(r, "N"))]
        for k in (r, r + 1, r + 2):
            quads.append((I(k - 1, "C"), I(k, "N"), I(k, "CA"), I(k, "C")))
            quads.append((I(k, "N"), I(k, "CA"), I(k, "C"), I(k + 1, "N")))
        self._quads = np.array(quads)
        self.theta0 = float(self.torsions(b[None])[0, 0])
        self.a3, self.c3 = b[:, self.i_a3], b[:, self.i_c3]
        self.cos_nac = np.cos(ref["ang_nac"][r:r + 3])
        units = []
        for k in (r, r + 1):  # peptide unit CA(k) .. CA(k+1) in a local frame
            ca, c = np.zeros(3), np.array([ref["len_ac"][k], 0.0, 0.0])
            n = _place(np.array([0.0, 1.0, 0.0]), ca, c, ref["len_cn"][k], ref["ang_acn"][k], 0.0)
            ca2 = _place(ca, c, n, ref["len_na"][k + 1], ref["ang_cna"][k], ref["omega"][k])
            units.append((ca, c, n, ca2))
        ca, c, n, ca2 = units[0]   # seen from CA(r), axis to CA(r+1)
        e = _unit(ca2)
        f = _unit(c - np.dot(c, e) * e)
        g = np.cross(e, f)
        self.d1 = float(np.linalg.norm(ca2))
        self.u1_c = np.array([c @ e, c @ f, c @ g])
        self.u1_n = np.array([n @ e, n @ f, n @ g])
        self.cos_b1 = self.u1_c[0] / np.linalg.norm(self.u1_c)
        self.sin_b1 = math.sqrt(1.0 - self.cos_b1 ** 2)
        ca, c, n, ca2 = units[1]   # seen from CA(r+2), axis to CA(r+1)
        e = _unit(ca - ca2)
        f = _unit((n - ca2) - np.dot(n - ca2, e) * e)
        g = np.cross(e, f)
        self.d2 = float(np.linalg.norm(ca2))
        self.u2_n = np.array([(n - ca2) @ e, (n - ca2) @ f, (n - ca2) @ g])
        self.u2_c = np.array([(c - ca2) @ e, (c - ca2) @ f, (c - ca2) @ g])
        self.cos_b3 = self.u2_n[0] / np.linalg.norm(self.u2_n)
        self.sin_b3 = math.sqrt(1.0 - self.cos_b3 ** 2)
        # A fixed direction never parallel to CA(r) -> CA(r+2) sets alpha = 0.
        th = np.linspace(-math.pi, math.pi, 73)
        w = _unit(self.a3 - _rotate(b[:, self.i_a1], self.drv_center, self.drv_axis, th - self.theta0))
        self.circle_ref = max(np.eye(3), key=lambda v: np.min(np.linalg.norm(np.cross(w, v), axis=1)))

    def atom(self, residue: int, name: str) -> int:
        return int(np.nonzero((self.res == residue) & (self.names == name))[0][0])

    def new_context(self):
        ctx = mcpu_core.Context(self.system)
        ctx.set_atom_reorder_mode(self.reorder)
        ctx.set_positions(self.start32)
        ctx.calculate_total_energy(-1)
        return ctx

    # ------------------------------------------------------------------
    def closure(self, x, signs=None):
        """Window atoms and residuals for points x (n, 4) = (theta, alpha, psi1, psi3).

        With ``signs`` = (s1, s3), x holds (theta, alpha) only and psi1, psi3
        are solved from q1, q3 with those signs; the residual is then F alone
        (NaN where q1 or q3 is out of [-1, 1]). Otherwise the residuals are
        (cos psi1 - q1, cos psi3 - q3, F)."""
        theta, alpha = x[..., 0], x[..., 1]
        n1 = _rotate(self.base[:, self.i_n1], self.drv_center, self.drv_axis, (theta - self.theta0).ravel()).reshape(theta.shape + (3,))
        a1 = _rotate(self.base[:, self.i_a1], self.drv_center, self.drv_axis, (theta - self.theta0).ravel()).reshape(theta.shape + (3,))
        a3, c3 = self.a3, self.c3
        span = a3 - a1
        D = np.linalg.norm(span, axis=-1, keepdims=True)
        w = span / D
        xc = (self.d1 ** 2 - self.d2 ** 2 + D ** 2) / (2 * D)
        rho2 = self.d1 ** 2 - xc ** 2
        rho = np.sqrt(np.maximum(rho2, 0.0))
        av = _unit(self.circle_ref - np.sum(self.circle_ref * w, -1, keepdims=True) * w)
        bv = np.cross(w, av)
        a2 = a1 + xc * w + rho * (np.cos(alpha)[..., None] * av + np.sin(alpha)[..., None] * bv)
        e1 = _unit(a2 - a1)
        un1 = _unit(n1 - a1)
        p1 = np.sum(un1 * e1, -1, keepdims=True)
        perp1 = un1 - p1 * e1
        R1 = np.linalg.norm(perp1, axis=-1, keepdims=True)
        q1 = (self.cos_nac[0] - self.cos_b1 * p1) / (self.sin_b1 * R1)
        e3 = _unit(a2 - a3)
        uc3 = _unit(c3 - a3)
        p3 = np.sum(uc3 * e3, -1, keepdims=True)
        perp3 = uc3 - p3 * e3
        R3 = np.linalg.norm(perp3, axis=-1, keepdims=True)
        q3 = (self.cos_nac[2] - self.cos_b3 * p3) / (self.sin_b3 * R3)
        if signs is None:
            psi1, psi3 = x[..., 2:3], x[..., 3:4]
        else:
            psi1 = signs[0] * np.arccos(np.clip(q1, -1.0, 1.0))
            psi3 = signs[1] * np.arccos(np.clip(q3, -1.0, 1.0))
        nh1 = perp1 / R1
        f1 = np.cos(psi1) * nh1 + np.sin(psi1) * np.cross(e1, nh1)
        g1 = np.cross(e1, f1)
        c1 = a1 + self.u1_c[0] * e1 + self.u1_c[1] * f1 + self.u1_c[2] * g1
        n2 = a1 + self.u1_n[0] * e1 + self.u1_n[1] * f1 + self.u1_n[2] * g1
        nh3 = perp3 / R3
        f3 = np.cos(psi3) * nh3 + np.sin(psi3) * np.cross(e3, nh3)
        g3 = np.cross(e3, f3)
        n3 = a3 + self.u2_n[0] * e3 + self.u2_n[1] * f3 + self.u2_n[2] * g3
        c2 = a3 + self.u2_c[0] * e3 + self.u2_c[1] * f3 + self.u2_c[2] * g3
        F = np.sum(_unit(n2 - a2) * _unit(c2 - a2), -1) - self.cos_nac[1]
        atoms = dict(n1=n1, a1=a1, c1=c1, n2=n2, a2=a2, c2=c2, n3=n3)
        if signs is not None:
            ok = (rho2[..., 0] > 0) & (np.abs(q1[..., 0]) <= 1) & (np.abs(q3[..., 0]) <= 1)
            return atoms, np.where(ok, F, np.nan), psi1[..., 0], psi3[..., 0]
        G = np.stack([np.cos(psi1[..., 0]) - q1[..., 0], np.cos(psi3[..., 0]) - q3[..., 0], F], -1)
        return atoms, G

    def locate(self, c):
        """(theta, alpha, psi1, psi3) of the window in coordinates c (3, n_atoms)."""
        c = np.asarray(c, dtype=np.float64)
        theta = float(self.torsions(c[None])[0, 0])
        a1, n1, c1 = c[:, self.i_a1], c[:, self.i_n1], c[:, self.i_c1]
        a2, n3 = c[:, self.i_a2], c[:, self.i_n3]
        w = _unit(self.a3 - a1)
        av = _unit(self.circle_ref - np.dot(self.circle_ref, w) * w)
        alpha = math.atan2(np.dot(a2 - a1, np.cross(w, av)), np.dot(a2 - a1, av))

        def turn(axis, reference, moving):
            nh = _unit(reference - np.dot(reference, axis) * axis)
            f = _unit(moving - np.dot(moving, axis) * axis)
            return math.atan2(np.dot(np.cross(axis, nh), f), np.dot(nh, f))
        psi1 = turn(_unit(a2 - a1), _unit(n1 - a1), _unit(c1 - a1))
        psi3 = turn(_unit(a2 - self.a3), _unit(self.c3 - self.a3), _unit(n3 - self.a3))
        return np.array([theta, alpha, psi1, psi3])

    def coords(self, x):
        """Full coordinates (n, 3, n_atoms), build order, for curve points x (n, 4)."""
        atoms, _ = self.closure(x)
        n = len(x)
        c = np.repeat(self.base[None], n, axis=0)
        c[:, :, self.i_o0] = _rotate(self.base[:, self.i_o0], self.drv_center, self.drv_axis, x[:, 0] - self.theta0)
        for key, i in (("n1", self.i_n1), ("a1", self.i_a1), ("c1", self.i_c1), ("n2", self.i_n2),
                       ("a2", self.i_a2), ("c2", self.i_c2), ("n3", self.i_n3)):
            c[:, :, i] = atoms[key]
        # Sidechains move rigidly with N, CA, C of their residue; O(k) with CA(k), C(k), N(k+1).
        for k, idx in self.sidechains.items():
            if idx:
                iN, iA, iC = self.atom(k, "N"), self.atom(k, "CA"), self.atom(k, "C")
                rot = _frames(c[:, :, iN], c[:, :, iA], c[:, :, iC]) @ _frames(
                    self.base[:, iN], self.base[:, iA], self.base[:, iC]).T
                c[:, :, idx] = c[:, :, [iA]] + rot @ (self.base[:, idx] - self.base[:, [iA]])
        for k, io in ((self.r, self.i_o1), (self.r + 1, self.i_o2)):
            iA, iC, iNn = self.atom(k, "CA"), self.atom(k, "C"), self.atom(k + 1, "N")
            rot = _frames(c[:, :, iA], c[:, :, iC], c[:, :, iNn]) @ _frames(
                self.base[:, iA], self.base[:, iC], self.base[:, iNn]).T
            c[:, :, io] = c[:, :, iC] + rot @ (self.base[:, io] - self.base[:, iC])
        return c

    def jacobian(self, x):
        """J = 1 / |det M| for curve points x (n, 4)."""
        atoms, _ = self.closure(x)
        o = atoms["a1"]
        a3 = np.broadcast_to(self.a3, o.shape)
        c3 = np.broadcast_to(self.c3, o.shape)
        cols = []
        for n, a, c in ((atoms["n1"], atoms["a1"], atoms["c1"]), (atoms["n2"], atoms["a2"], atoms["c2"]),
                        (atoms["n3"], a3, c3)):
            u_phi, u_psi = _unit(a - n), _unit(c - a)
            cols += [np.concatenate([u_phi, np.cross(a - o, u_phi)], -1),
                     np.concatenate([u_psi, np.cross(c - o, u_psi)], -1)]
        return 1.0 / np.abs(np.linalg.det(np.stack(cols, -1)))

    def torsions(self, c):
        """psi(r-1), phi(r), psi(r), phi(r+1), psi(r+1), phi(r+2), psi(r+2) for coordinates (n, 3, n_atoms)."""
        q = self._quads
        return _dihedrals(*(np.moveaxis(c[:, :, q[:, j]], 1, -1) for j in range(4)))


# ----------------------------------------------------------------------
# The curves
# ----------------------------------------------------------------------
_EPS = 1e-7
_OFFSETS = np.zeros((9, 4))
for _k in range(4):
    _OFFSETS[1 + 2 * _k, _k], _OFFSETS[2 + 2 * _k, _k] = _EPS, -_EPS


def _residual_and_jacobian(window, X):
    """Residuals (m, 3) and their derivatives (m, 3, 4) by central differences."""
    m = len(X)
    _, G = window.closure((X[:, None, :] + _OFFSETS[None]).reshape(-1, 4))
    G = G.reshape(m, 9, 3)
    D = np.stack([(G[:, 1 + 2 * k] - G[:, 2 + 2 * k]) / (2 * _EPS) for k in range(4)], -1)
    return G[:, 0], D


def _tangent(D, previous=None):
    t = np.linalg.svd(D)[2][-1]
    return -t if previous is not None and t @ previous < 0 else t


def _wrap(d):
    return (d + math.pi) % (2 * math.pi) - math.pi


def _embed(X):
    return np.concatenate([np.cos(X), np.sin(X)], -1)


def _seeds(window, n_coarse=360, sub=4):
    """Points on every curve, by marching squares over (theta, alpha) for each sign pair."""
    h = 2 * math.pi / n_coarse
    th = -math.pi + h * np.arange(n_coarse)
    al = h * np.arange(n_coarse)
    o = np.arange(sub + 1) * (h / sub)
    TH, AL = np.meshgrid(th, al, indexing="ij")
    seeds = []
    for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        _, F, _, _ = window.closure(np.stack([TH, AL], -1), signs)
        corners = np.stack([F, np.roll(F, -1, 0), np.roll(F, -1, 1), np.roll(np.roll(F, -1, 0), -1, 1)])
        sg = np.where(np.isfinite(corners), np.sign(corners), 0)
        cells = np.argwhere((sg.max(0) > 0) & (sg.min(0) < 0))
        if len(cells) == 0:
            continue
        T, A = np.broadcast_arrays(th[cells[:, 0]][:, None, None] + o[None, :, None],
                                   al[cells[:, 1]][:, None, None] + o[None, None, :])
        _, Fs, P1, P3 = window.closure(np.stack([T, A], -1), signs)
        # a sign change along alpha between valid neighbours brackets a curve point
        fa, fb = Fs[:, :, :-1], Fs[:, :, 1:]
        hit = np.isfinite(fa) & np.isfinite(fb) & (np.sign(fa) != np.sign(fb))
        ci, ti, ai = np.nonzero(hit)
        frac = fa[ci, ti, ai] / (fa[ci, ti, ai] - fb[ci, ti, ai])
        seeds.append(np.c_[T[ci, ti, ai], A[ci, ti, ai] + frac * (h / sub),
                           P1[ci, ti, ai], P3[ci, ti, ai]])
    if not seeds:
        raise ValueError("the window has no closure for any driver angle")
    return np.concatenate(seeds)


def _trace(window, x0, h, max_length=500.0):
    """Follow the closed curve through x0; returns its points (n, 4) in order."""
    x = x0.copy()
    for _ in range(10):
        g, D = _residual_and_jacobian(window, x[None])
        x = x - np.linalg.pinv(D[0]) @ g[0]
    t = _tangent(_residual_and_jacobian(window, x[None])[1][0])
    start, points, length = x.copy(), [x.copy()], 0.0
    while length < max_length:
        step = h
        while True:
            guess = x + step * t
            y = guess.copy()
            for _ in range(10):
                g, D = _residual_and_jacobian(window, y[None])
                dy = np.linalg.solve(np.vstack([D[0], t]), -np.append(g[0], t @ (y - guess)))
                y = y + dy
                if np.abs(dy).max() < 1e-11:
                    break
            moved = np.linalg.norm(y - x)
            t_new = _tangent(_residual_and_jacobian(window, y[None])[1][0], t)
            if np.abs(dy).max() < 1e-11 and 0.5 * step < moved < 1.5 * step and t_new @ t > 0.95:
                break
            step *= 0.5
            if step < 1e-7:
                raise RuntimeError(f"continuation stalled at {x}")
        back = _wrap(start - x)
        along = back @ t
        if length > 20 * h and 0 < along <= moved and np.linalg.norm(back - along * t) < 1e-3:
            return np.array(points)
        length += moved
        points.append(y.copy())
        x, t = y, t_new
    raise RuntimeError("a closure curve did not close")


def exact_curve(window, h=0.005, k=25):
    """Points along every closure curve and their weights per unit of theta.

    Returns x (n, 4); ``dtheta`` (n,), the theta-measure of each point's share
    of its curve (|dtheta/ds| times its arc length); and the index of the
    curve each point is on (points of a curve are in order along it)."""
    seeds = _seeds(window)
    for _ in range(6):  # onto the curves, so that a traced curve accounts for its seeds
        g, D = _residual_and_jacobian(window, seeds)
        seeds = seeds - (np.linalg.pinv(D) @ g[..., None])[..., 0]
    g, _ = _residual_and_jacobian(window, seeds)
    seeds = seeds[np.abs(g).max(-1) < 1e-9]
    left = np.ones(len(seeds), bool)
    xs, dthetas, curve = [], [], []
    while left.any():
        P = _trace(window, seeds[np.argmax(left)], h)
        left[np.argmax(left)] = False
        near = cKDTree(_embed(P)).query(_embed(seeds[left]))[0] < 2.5 * h
        left[np.nonzero(left)[0][near]] = False
        if xs and cKDTree(_embed(np.concatenate(xs))).query(_embed(P[:1]))[0][0] < 2.5 * h:
            continue  # the same curve again
        # k points per continuation step, each projected back onto the curve
        Q = np.vstack([P, P[:1] + np.round((P[-1:] - P[:1]) / (2 * math.pi)) * 2 * math.pi])
        guess = (Q[:-1, None, :] + (np.arange(k) / k)[None, :, None] * np.diff(Q, axis=0)[:, None, :]).reshape(-1, 4)
        chord = np.repeat(_unit(np.diff(Q, axis=0)), k, axis=0)
        X = guess.copy()
        for _ in range(4):
            g, D = _residual_and_jacobian(window, X)
            r = np.concatenate([g, np.sum(chord * (X - guess), -1, keepdims=True)], -1)
            X = X - np.linalg.solve(np.concatenate([D, chord[:, None, :]], 1), r[..., None])[..., 0]
        g, D = _residual_and_jacobian(window, X)
        if np.abs(g).max() > 1e-9:
            raise RuntimeError("a curve point did not converge")
        t = np.linalg.svd(D)[2][:, -1, :]
        seg = np.linalg.norm(np.diff(np.vstack([X, X[:1] + (Q[-1] - Q[0])]), axis=0), axis=1)
        xs.append(X)
        dthetas.append(np.abs(t[:, 0]) * 0.5 * (seg + np.roll(seg, 1)))
        curve.append(np.full(len(X), len(xs) - 1))
    return np.concatenate(xs), np.concatenate(dthetas), np.concatenate(curve)


def closures_per_theta(x, curve):
    """How many closures each point's driver angle has: the curves' crossings of it."""
    theta = x[:, 0]
    grid = np.linspace(-math.pi, math.pi, 7201)
    count = np.zeros(len(grid))
    for c in np.unique(curve):
        t = _wrap(theta[curve == c])
        t2 = np.roll(t, -1)
        keep = np.abs(t2 - t) < math.pi
        lo, hi = np.minimum(t, t2)[keep], np.maximum(t, t2)[keep]
        # grid points g with lo < g <= hi, by a difference array
        diff = np.zeros(len(grid) + 1)
        np.add.at(diff, np.searchsorted(grid, lo, side="right"), 1)
        np.add.at(diff, np.searchsorted(grid, hi, side="right"), -1)
        count += np.cumsum(diff)[:-1]
    return np.interp(_wrap(theta), grid, count)


def evaluate(window, x, chunk=4000):
    """Engine energy, hard-core clash, J and the seven torsions at curve points x."""
    ctx = window.new_context()
    n = len(x)
    E, clash, J, tors = np.empty(n), np.zeros(n, bool), np.empty(n), np.empty((n, 7))
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        J[sl] = window.jacobian(x[sl])
        c = window.coords(x[sl])
        tors[sl] = window.torsions(c)
        for m in range(len(c)):
            ctx.coords = c[m]
            E[lo + m] = ctx.calculate_total_energy(-1)
            clash[lo + m] = ctx.has_steric_clash()
    return E, clash, J, tors
