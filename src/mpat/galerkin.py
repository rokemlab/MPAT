"""Bundles from fixed-step walkers, solved as a Lagrange-Galerkin scheme (``mpat --galerkin``),
on a GPU with NVIDIA Warp.

    mpat dwi.nii.gz dwi.bval dwi.bvec t1.nii.gz out/ --galerkin --galerkin-points 4 [--angle 30]
    mpat.galerkin.run(afq, bundles=["Left Arcuate"], points=4)

The walkers are probabilistic tracking's: from grey matter into the white matter, each takes
steps of ``step`` mm (0.5 by default, as pyAFQ; or a fraction of the smallest voxel side:
``"0.5vox"``), and after each step draws its new direction from the fODF of the voxel it landed in, within ``angle``
degrees of the old one (directions: a DIPY sphere).  It ends where a step leaves the white
matter: validly into grey matter or the brain's edge; otherwise (CSF, outside, an excluded
voxel, a turn with no fODF within the angle) it is lost.  (run's default, stop "interp": it
stops instead as pyAFQ's GPU tracker, where the WM PVE trilinear at its point is <= 0.4.)  No turning rate, no mean free path:
the model is the tracker's, without its sampling.

The scheme follows every walker at once, deterministically, and keeps where in a voxel they
are.  The white matter is cut into bins (``bins`` per voxel side: 1, 3, 5, ...), and in each
bin the walkers heading one direction are a density that is quadratic in position: 10
coefficients (their number, mean and spread) on the orthonormal Legendre basis of the bin.
One step, per direction:

- shift: the density moves by the step, exactly.  The bin is cut into pieces whose walkers
  visit the same bins in the same order (by the planes where the landing bin changes and
  those through the grid's edges along the step); each piece lands in one bin, where its
  density is projected back onto the quadratics (Galerkin: the 10 moments kept exactly),
  unless one of the voxels it passes is dead -- then its walkers leave the white matter there.
  The geometry is the same in every bin, so each piece's integrals (a 10 x 10 matrix) are
  computed once per direction, exactly (degree-5 quadrature on its tetrahedra);
- turn: in each bin, the walkers heading m move to m' in m's cone with probability
  f(m') w(m') / sum over the cone of f w (the voxel's fODF; w the directions' solid angles).

The scheme is linear, conserves walkers (up to those that leave), and its exact adjoint is its
transpose: the backward solve (the probability that a walker ends in a set of voxels) is the
same steps transposed, with the same pieces.  A bundle's density is then, per step, its
walkers' path length through each voxel times their probability of ending well after it --
summed over all steps once, from the forward walkers summed over the steps.  Densities may
dip a little below zero (a quadratic is not positive everywhere); maps are clipped at zero
and the negative mass is reported.

Walkers launch through each face between a valid end voxel and the white
matter, in every direction of the fODF there that enters, f w |Omega . n| per face, spread
evenly over the face (2 x 2 Gauss points per bin face) -- their first step is traced exactly
from there.  A solve runs until the walkers still in the white matter are below ``tol`` of
those launched (absolute values, bin by bin).

Bundles follow MPAT's rules (mpat.solve): end kinds, include ROIs by
inclusion-exclusion over avoid sets, exclude ROIs taken out of the white matter.  Per avoid
set and end kind: a forward solve from that kind's launches and a backward solve toward the
kinds it pairs with (as many steps as the forward took, so that both count the same walkers:
the transport identity).

Bins finer than the voxel (``bins`` 3, 5) keep a beam's position to a third, a fifth of a
voxel, at kb^3 times the memory and work.  So a bundle with ``bins`` > 1 is solved twice: on
whole voxels over all of the white matter, and then with the finer bins only where that
first density reaches ``floor`` x its peak, grown by ``dilate`` voxels (walkers leaving that
region are lost).

Speed (identical results): states only where the fODF is non-zero (a walker leaves a turn only
into such a direction), and landings only where a walker can land (from a step upwind, in a
direction of the fODF there); every step a gather (no atomics); pieces with the same visits
merged; empty sources skipped; the shifts' threads direction by direction (their reads
coalesce); the turns a thread block per bin, the bin's directions gathered once into shared
memory and the cones summed from there; each bundle solved only on the white matter its
exclusions leave it (Bundles: a hemisphere for one-sided bundles); each step replayed as a CUDA
graph; the tallies (ends, densities) once per solve from the walkers summed over its steps.

Outputs go to <output_dir>/mpat-galerkin (or ``out_dir``) under mpat.run's names, so the
collage, the viewer pages and the cores read them.
"""
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, field, replace
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np

from .solve import Clock, _endpoint_kinds

log = logging.getLogger("mpat")
_wp = None


def warp():
    """The ``warp`` module (initialised), or an ImportError saying how to get it."""
    global _wp
    if _wp is None:
        try:
            import warp as wp
        except ImportError as err:
            raise ImportError("mpat.galerkin needs NVIDIA Warp: pip install warp-lang "
                              "(or pip install 'mpat-tractography[gpu]')") from err
        wp.init()
        _wp = wp
    return _wp


def get_device(name=None):
    """A Warp device: ``None``, "cuda" or "gpu" for the first CUDA device, "cuda:1" for the
    second, "cpu" for Warp's CPU device (for testing)."""
    wp = warp()
    name = "cuda" if name in (None, "", "gpu", True) else str(name)
    if name.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError(
            "Warp sees no CUDA device (no NVIDIA GPU, or its driver is not loaded: "
            "`python -c 'import warp; warp.init()'` lists the devices Warp finds), or device='cpu' "
            "to try the GPU code on the CPU (slow; for tests).")
    return wp.get_device(name)

VALID_ENDS = (1, 3)     # grey matter, the brain's edge (mpat.chain's absorbing classes)
ODF_FLOOR = 1e-3        # visitation ODFs where the density reaches this x its peak (as mpat.afq)


class Directions:
    """The walkers' directions (a DIPY sphere that holds each direction's reverse), their solid
    angles (spherical Voronoi) and the turning cone of each (cptr into cidx)."""

    def __init__(self, sphere="symmetric362", angle=35.0):
        from dipy.core.sphere import Sphere
        from dipy.data import get_sphere
        from scipy.spatial import SphericalVoronoi

        sp = sphere if isinstance(sphere, Sphere) else get_sphere(name=sphere)
        v = np.asarray(sp.vertices, np.float64)
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        self.v, self.M = v, len(v)
        G = v @ v.T
        rev = np.argmin(G, axis=1)
        if np.abs(v[rev] + v).max() > 1e-6:
            raise ValueError(f"sphere {sphere!r} does not hold the reverse of each of its directions")
        self.rev = rev.astype(np.int32)
        area = SphericalVoronoi(v).calculate_areas()
        self.w = area * (4 * np.pi / area.sum())
        cone = G >= np.cos(np.deg2rad(angle)) - 1e-9
        self.cptr = np.concatenate([[0], np.cumsum(cone.sum(1))]).astype(np.int32)
        self.cidx = np.concatenate([np.flatnonzero(r) for r in cone]).astype(np.int32)


def fodf_on(sh_rows, dirs, chunk=20000):
    """The fODF of each row of SH coefficients on the walkers' directions: clipped at 0, values
    below PMF_THRESHOLD x its maximum zeroed (as the chain, and pyAFQ's tracker), normalised to
    integrate to 1 over the sphere (rows of zeros stay zero)."""
    from .geometry import PMF_THRESHOLD, _sh_basis

    B = _sh_basis(dirs.v, sh_rows.shape[1])
    out = np.empty((len(sh_rows), dirs.M), np.float32)
    for s in range(0, len(sh_rows), chunk):
        sf = np.maximum(np.asarray(sh_rows[s:s + chunk], np.float64) @ B, 0.0)
        sf[sf < PMF_THRESHOLD * sf.max(1, keepdims=True)] = 0.0
        tot = (sf @ dirs.w)[:, None]
        out[s:s + chunk] = np.divide(sf, tot, out=np.zeros_like(sf), where=tot > 0)
    return out


@dataclass
class TransportResult:
    name: str
    density: np.ndarray       # (X, Y, Z) path length (mm) of the bundle's walkers per voxel
    endpoints: np.ndarray     # (X, Y, Z) the bundle's walkers' ends per voxel (two per walker)
    seed_prob: np.ndarray     # (X, Y, Z) P(a walker launched from this voxel is in the bundle)
    odf_ijk: np.ndarray       # (n, 3) voxels with a visitation ODF
    odf_coef: np.ndarray      # (n, ncoef) its SH coefficients (descoteaux07, legacy, lmax 8)
    info: dict = field(default_factory=dict)


def _call(f, *args, **kw):
    """``f(*args, **kw)``, leaving out the keywords ``f`` does not take (mpat.afq's helpers
    differ a little between MPAT versions)."""
    params = inspect.signature(f).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        for k in [k for k in kw if k not in params]:
            if kw.pop(k) is not None:
                log.warning("%s takes no %s here; ignored", f.__name__, k)
    return f(*args, **kw)

LOG_SECONDS = 30.0        # seconds between a solve's progress lines (5 with verbose)
GEOMETRY_VERSION = "1"    # bump when the pieces change: invalidates the disk cache
_GEOMETRY = {}            # in-memory cache of the pieces, by key


# ------------------------------------------------------------------ the pieces of a step
SQ3, SQ5 = math.sqrt(3.0), math.sqrt(5.0)
EXPONENTS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [0, 2, 0], [0, 0, 2],
                      [1, 1, 0], [1, 0, 1], [0, 1, 1]])


def basis(x):
    """The 10 orthonormal quadratics of the unit cube (products of Legendre polynomials of total
    degree <= 2; the first is 1) at points x (n, 3): (n, 10)."""
    t = np.asarray(x, np.float64)
    P = np.stack([np.ones_like(t), SQ3 * (2 * t - 1), SQ5 * (6 * t * t - 6 * t + 1)], -1)
    return P[:, 0, EXPONENTS[:, 0]] * P[:, 1, EXPONENTS[:, 1]] * P[:, 2, EXPONENTS[:, 2]]


def _gm_rule(s=2, n=3):
    """Grundmann-Moeller rule of degree 2s + 1 on the n-simplex: barycentric points and weights
    (summing to 1: times the simplex's volume)."""
    d = 2 * s + 1
    pts, wts = [], []
    for i in range(s + 1):
        w = (-1) ** i * 2.0 ** (-2 * s) * (d + n - 2 * i) ** d / (math.factorial(i) * math.factorial(d + n - i))
        for beta in itertools.product(range(s - i + 1), repeat=n + 1):
            if sum(beta) == s - i:
                pts.append([(2 * b + 1) / (d + n - 2 * i) for b in beta])
                wts.append(w)
    return np.array(pts), np.array(wts) * math.factorial(n)


GM_BARY, GM_W = _gm_rule(2, 3)     # degree 5: a quadratic x path length x a quadratic, exactly
_E = np.eye(3)
KUHN = np.array([[[0, 0, 0], _E[p[0]], _E[p[0]] + _E[p[1]], [1, 1, 1]] for p in itertools.permutations(range(3))])


def _tet_vol(T):
    return np.einsum("ij,ij->i", T[:, 1] - T[:, 0], np.cross(T[:, 2] - T[:, 0], T[:, 3] - T[:, 0])) / 6.0


def _prism(a0, a1, a2, b0, b1, b2):
    return [np.stack(t, 1) for t in ((a0, a1, a2, b0), (a1, a2, b0, b1), (a2, b0, b1, b2))]


def _split(T, nrm, c, eps=1e-12):
    """Tetrahedra T (t, 4, 3) cut by the plane nrm . x = c into tetrahedra on either side."""
    s = T @ nrm - c
    s[np.abs(s) < eps] = 0.0
    cut = ((s > 0).sum(1) > 0) & ((s < 0).sum(1) > 0)
    if not cut.any():
        return T
    S = T[cut]
    o = np.argsort(-s[cut], axis=1, kind="stable")     # positive, zero, negative
    P = np.take_along_axis(S, o[:, :, None], 1)
    sv = np.take_along_axis(s[cut], o, 1)
    npos, nneg = (sv > 0).sum(1), (sv < 0).sum(1)

    def X(m, a, b):                                    # where edge a-b meets the plane
        ta, tb = sv[m, a], sv[m, b]
        return P[m, a] + (ta / (ta - tb))[:, None] * (P[m, b] - P[m, a])

    out = []
    m = (npos == 1) & (nneg == 3)
    if m.any():
        i1, i2, i3 = X(m, 0, 1), X(m, 0, 2), X(m, 0, 3)
        out += [np.stack([P[m, 0], i1, i2, i3], 1)] + _prism(P[m, 1], P[m, 2], P[m, 3], i1, i2, i3)
    m = (npos == 3) & (nneg == 1)
    if m.any():
        i0, i1, i2 = X(m, 3, 0), X(m, 3, 1), X(m, 3, 2)
        out += [np.stack([P[m, 3], i0, i1, i2], 1)] + _prism(P[m, 0], P[m, 1], P[m, 2], i0, i1, i2)
    m = (npos == 2) & (nneg == 2)
    if m.any():
        i02, i03, i12, i13 = X(m, 0, 2), X(m, 0, 3), X(m, 1, 2), X(m, 1, 3)
        out += _prism(P[m, 0], i02, i03, P[m, 1], i12, i13) + _prism(P[m, 2], i02, i12, P[m, 3], i03, i13)
    m = (npos == 1) & (nneg == 2)                      # p, z, n1, n2
    if m.any():
        i2, i3, z = X(m, 0, 2), X(m, 0, 3), P[m, 1]
        out += [np.stack([P[m, 0], z, i2, i3], 1), np.stack([z, P[m, 2], P[m, 3], i3], 1),
                np.stack([z, P[m, 2], i3, i2], 1)]
    m = (npos == 2) & (nneg == 1)                      # p1, p2, z, n
    if m.any():
        i0, i1, z = X(m, 3, 0), X(m, 3, 1), P[m, 2]
        out += [np.stack([P[m, 3], z, i0, i1], 1), np.stack([z, P[m, 0], P[m, 1], i1], 1),
                np.stack([z, P[m, 0], i1, i0], 1)]
    m = (npos == 1) & (nneg == 1)                      # p, z1, z2, n
    if m.any():
        i = X(m, 0, 3)
        out += [np.stack([P[m, 0], P[m, 1], P[m, 2], i], 1), np.stack([P[m, 3], P[m, 1], P[m, 2], i], 1)]
    new = np.concatenate(out)
    v0, v1 = np.abs(_tet_vol(S)).sum(), np.abs(_tet_vol(new)).sum()
    if abs(v0 - v1) > 1e-11 * v0 + 1e-15:
        raise RuntimeError(f"mpat.galerkin: a cut lost volume ({v0:.15g} -> {v1:.15g})")
    return np.concatenate([T[~cut], new[np.abs(_tet_vol(new)) > 1e-15]])


def _crossable(d, a):
    """The grid planes (integers) along axis a that a step d (bins) can cross from the unit bin."""
    if d[a] > 1e-12:
        return list(range(1, math.ceil(1 + d[a] - 1e-12)))
    if d[a] < -1e-12:
        return list(range(math.floor(d[a] + 1e-12) + 1, 1))
    return []


def _planes(d):
    """The planes cutting the unit bin into pieces: where the landing bin changes (x_a = n - d_a)
    and where the order of two crossings does (through the grid's edges, along d)."""
    out = []
    for a in range(3):
        if abs(d[a]) > 1e-12:
            for n in range(math.floor(d[a]) - 1, math.floor(d[a]) + 3):
                if 1e-12 < n - d[a] < 1 - 1e-12:
                    out.append((_E[a], n - d[a]))
    corners = np.array(list(itertools.product((0, 1), repeat=3)), np.float64)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        for n in _crossable(d, a):
            for m in _crossable(d, b):
                nrm = d[b] * _E[a] - d[a] * _E[b]
                c = d[b] * n - d[a] * m
                s = corners @ nrm - c
                if s.max() > 1e-12 and s.min() < -1e-12:
                    z = np.linalg.norm(nrm)
                    out.append((nrm / z, c / z))
    return out


def pieces(d, smm):
    """A step of d (in bins; smm mm long) from the unit bin, in pieces whose points visit the same
    bins in the same order.  Per piece: its cells (bin offsets, the source first, the landing
    bin last); A (10, 10): its walkers' coefficients in the landing bin from the source's,
    A[l, k] = int phi_l(x + d - land) phi_k(x); a (10,): its walkers, int phi_k; per cell B
    (ncell, 10), int phi_k x path length (mm), and W (ncell, 10, 10), int phi_k x path length x
    phi_l(x + d - land)."""
    d = np.asarray(d, np.float64)
    T = KUHN.astype(np.float64)
    for nrm, c in _planes(d):
        T = _split(T, nrm, c)
    cand = [(a, n) for a in range(3) for n in _crossable(d, a)]
    cen = T.mean(1)
    if cand:
        ax = np.array([a for a, _ in cand])
        nn = np.array([n for _, n in cand], np.float64)
        t = (nn[None, :] - cen[:, ax]) / d[ax][None, :]
        hit = (t > 0) & (t < 1)
        o = np.argsort(np.where(hit, t, np.inf), axis=1, kind="stable")
        keys = np.where(np.take_along_axis(hit, o, 1), o, -1)
    else:
        keys = np.zeros((len(T), 1), np.int64) - 1
    uk, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    X = np.einsum("qv,tvx->tqx", GM_BARY, T)
    Wq = np.abs(_tet_vol(T))[:, None] * GM_W[None, :]
    out = []
    for r, key in enumerate(uk):
        sel = inv == r
        x, w = X[sel].reshape(-1, 3), Wq[sel].ravel()
        cells, tcol = [np.zeros(3, np.int64)], [np.zeros(len(x))]
        for i in key[key >= 0]:
            a, n = cand[i]
            c = cells[-1].copy()
            c[a] += 1 if d[a] > 0 else -1
            cells.append(c)
            tcol.append((n - x[:, a]) / d[a])
        tcol.append(np.ones(len(x)))
        ell = smm * np.diff(np.stack(tcol, 1), axis=1)
        land = cells[-1]
        px, py = basis(x), basis(x + d - land)
        out.append(dict(cells=np.array(cells), land=land, A=(py * w[:, None]).T @ px, a=w @ px,
                        B=np.einsum("n,nq,nk->qk", w, ell, px), W=np.einsum("n,nq,nk,nl->qkl", w, ell, px, py)))
    # every walker in exactly one piece; path lengths add up to the step
    tot = sum(p["a"] for p in out)
    bad = np.abs(tot - np.eye(10)[0]).max()
    for p in out:
        bad = max(bad, np.abs(p["B"].sum(0) - smm * p["a"]).max(), np.abs(p["W"].sum(0) - smm * p["A"].T).max())
    if bad > 1e-9 * max(1.0, smm):
        raise RuntimeError(f"mpat.galerkin: the pieces of step {d} do not add up ({bad:.1e})")
    return out


def _pieces_flat(args):
    d, smm = args
    ps = pieces(d, smm)
    return (np.array([p["land"] for p in ps], np.int32).reshape(-1, 3),
            np.array([p["A"] for p in ps]), np.array([p["a"] for p in ps]),
            np.array([len(p["cells"]) for p in ps], np.int64),
            np.concatenate([p["cells"] for p in ps]).astype(np.int32),
            np.concatenate([p["B"] for p in ps]), np.concatenate([p["W"] for p in ps]))


def geometry(dvec, smm, procs=None):
    """The pieces of a step for every direction (dvec: (M, 3), the steps in bins), flattened:
    gptr (M + 1) into land (R, 3), A (R, 10, 10), a (R, 10), cptr (R + 1) into cell (C, 3),
    B (C, 10), W (C, 10, 10).  Cached in memory and on disk (MPAT_CACHE or ~/.cache/mpat)."""
    dvec = np.asarray(dvec, np.float64)
    key = hashlib.sha1(np.round(dvec, 10).tobytes() + np.float64(smm).tobytes()
                       + GEOMETRY_VERSION.encode()).hexdigest()[:24]
    if key in _GEOMETRY:
        return _GEOMETRY[key]
    root = Path(os.environ.get("MPAT_CACHE", Path.home() / ".cache" / "mpat"))
    path = root / f"galerkin-pieces-{key}.npz"
    if path.exists():
        try:
            with np.load(path) as z:
                geo = {k: z[k] for k in z.files}
            _GEOMETRY[key] = geo
            return geo
        except Exception as err:  # noqa: BLE001 -- a broken cache file: recompute
            log.warning("galerkin: ignoring the cached pieces %s (%s)", path, err)
    t0 = time.time()
    jobs = [(d, smm) for d in dvec]
    n = max(1, min(int(procs or os.cpu_count() or 1), 32))
    log.info("galerkin: cutting a step into pieces for %d directions in %d process(es) (once per voxel size, step and "
             "bins: cached in %s)", len(jobs), n if len(jobs) > 8 else 1, root)
    res = None
    if n > 1 and len(jobs) > 8:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor

            # spawned, not forked: the parent may hold a CUDA context
            with ProcessPoolExecutor(n, mp_context=mp.get_context("spawn")) as ex:
                res = list(ex.map(_pieces_flat, jobs, chunksize=max(1, len(jobs) // (4 * n))))
        except (ValueError, OSError, RuntimeError, ImportError) as err:     # one process, then
            log.info("galerkin: the pieces in one process (%s)", err)
            res = None
    if res is None:
        res = [_pieces_flat(j) for j in jobs]
    npc = np.array([len(r[0]) for r in res])
    ncl = np.concatenate([r[3] for r in res])
    geo = dict(gptr=np.concatenate([[0], np.cumsum(npc)]).astype(np.int32),
               land=np.concatenate([r[0] for r in res]).astype(np.int32),
               A=np.concatenate([r[1] for r in res]).astype(np.float32),
               a=np.concatenate([r[2] for r in res]).astype(np.float32),
               cptr=np.concatenate([[0], np.cumsum(ncl)]).astype(np.int32),
               cell=np.concatenate([r[4] for r in res]).astype(np.int32),
               B=np.concatenate([r[5] for r in res]).astype(np.float32),
               W=np.concatenate([r[6] for r in res]).astype(np.float32))
    log.info("galerkin: the pieces of a step in %d directions (%.1f per direction, at most %d; %.1f cells each) "
             "in %.0fs", len(dvec), npc.mean(), npc.max(), ncl.mean(), time.time() - t0)
    try:
        root.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp, **geo)
        os.replace(tmp, path)
    except OSError:
        pass
    _GEOMETRY[key] = geo
    return geo


# ------------------------------------------------------------------ the solver
def _secs(t):
    return f"{t:.0f}s" if t >= 10 else f"{t:.1f}s"


class _Progress:
    """A solve's progress lines, every ``every`` seconds: the step, the time per step and, from
    how fast the walkers leave (forward) or the steps to go (backward), an estimate of the rest."""

    def __init__(self, what, every, tol=None, total=None, label="walkers left {:.1e} of those launched"):
        self.what, self.every, self.tol, self.total, self.label = what, float(every), tol, total, label
        self.t0 = self.last = time.perf_counter()
        self.hist = []

    def step(self, n, left=None, extra=None):
        """Logs a line if ``every`` seconds have passed (returns whether it did); ``extra``, a function
        of the step, adds to the line (called only then)."""
        now = time.perf_counter()
        if left is not None:
            self.hist = (self.hist + [(n, left)])[-32:]
        if now - self.last < self.every:
            return False
        self.last = now
        el = now - self.t0
        ms = 1e3 * el / max(n, 1)
        more_txt = extra(n) if extra is not None else ""
        if self.total:                                   # backward: a known number of steps
            log.info("  %s: step %d of %d (%.0f%%), %.1f ms/step, about %s to go%s", self.what, n, self.total,
                     100.0 * n / self.total, ms, _secs((self.total - n) * ms / 1e3), more_txt)
            return True
        msg = f"  {self.what}: step {n}, " + self.label.format(left)
        (n0, l0), (n1, l1) = self.hist[0], self.hist[-1]
        if n1 > n0 and l0 > 0 and 0 < l1 < l0 and self.tol:
            q = (l1 / l0) ** (1.0 / (n1 - n0))           # per step, lately
            more = max(0.0, math.log(self.tol / l1) / math.log(q))
            msg += f" (-{100 * (1 - q):.2g}%/step: about {more:.0f} more steps, {_secs(more * ms / 1e3)})"
        log.info("%s, %.1f ms/step, %s so far%s", msg, ms, _secs(el), more_txt)
        return True


ARRIVAL_SHARES = (0.99, 0.999, 0.9999)     # verbose: the steps by which these shares of a solve's arrivals came


STEP_VOX_WARN = (0.15, 1.0)    # a step outside these voxels (of the smallest side) is logged as a warning


def parse_step(step):
    """A step length and its unit from ``step``: a number (mm), ``"0.5"`` or ``"0.5mm"`` (mm), or
    ``"0.4vox"`` (a fraction of the smallest voxel side).  Returns (value, "mm" | "vox")."""
    if isinstance(step, (int, float, np.floating, np.integer)) and not isinstance(step, bool):
        val, unit = float(step), "mm"
    else:
        m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(mm|vox|voxels?)?\s*", str(step))
        if m is None:
            raise ValueError(f"step: a length in mm (0.5, \"0.5mm\") or in voxels (\"0.5vox\"), not {step!r}")
        val, unit = float(m.group(1)), "vox" if (m.group(2) or "mm").startswith("vox") else "mm"
    if not val > 0:
        raise ValueError(f"step must be positive, not {step!r}")
    return val, unit


@dataclass
class GalerkinConfig:
    bins: int = 1                    # bins per voxel side (1, 3, 5, ...)
    step: float = 0.5                # step length, in step_unit
    step_unit: str = "mm"            # "mm", or "vox": a fraction of the smallest voxel side
    angle: float = 35.0              # largest turn per step, degrees
    sphere: str = "symmetric362"     # the walkers' directions: a DIPY sphere holding each reverse
    tol: float = 1e-6                # a solve stops when the walkers left are below this x those launched
    max_steps: int = 5000            # steps per solve, at most
    floor: float = 1e-3              # bins > 1: the fine solve where the voxel solve's density reaches this x peak
    dilate: int = 2                  # ... grown by this many voxels
    interp: str = "all"              # bins > 1: each bin its own fODF, interpolated between voxel centers
                                     # ("all", the default: from every voxel, as DIPY's trackers; "wm": from
                                     # white-matter voxels; "": each voxel's fODF in all its bins)
    points: int = 0                  # > 0: each bin's walkers turn by the fODF interpolated at points^3 Gauss
                                     # points of the bin (as DIPY's, where each walker is), not one per bin;
                                     # 3 at least (a constant fODF then turns exactly as without); any bins
    stop: str = ""                   # "": walkers leave the white matter's voxels (tissue labels); "interp": as
                                     # pyAFQ's GPU tracker, a walker stops where the stop map (WM PVE), trilinear
                                     # at its landing point, is <= stop_threshold, and ends in that voxel
                                     # (points only: decided at each Gauss point; run's default)
    stop_threshold: float = 0.4      # stop "interp": pyAFQ's (cuslines') threshold
    until: str = "ends"              # seeded bundles' forwards weighted by the partner (their backward known
                                     # first): stop when their expected arrivals still to come are below tol x
                                     # those expected ("ends", the default), or when the walkers left are below
                                     # tol x those launched ("walkers", as every other solve)
    prune: float = 1e-9              # seeded bundles: those forwards drop a bin's walkers when their expected
                                     # arrivals are below this x the solve's expected; the backwards drop a
                                     # bin's importance when it times the all-halves forward's visits there is
                                     # below this x the arrivals in that end kind (0: never).  What is dropped
                                     # is tallied (a bound on the arrivals lost).  Other bundles: no effect

    def key(self):
        d = asdict(self)
        if d["until"] == "walkers":
            d.pop("until")               # the defaults: as the keys of runs made before the options
        if not d["prune"]:
            d.pop("prune")
        if not d["interp"]:
            d.pop("interp")              # off: as the keys of runs made before the option
        if not d["points"]:
            d.pop("points")
        if not d["stop"]:
            d.pop("stop")
            d.pop("stop_threshold")
        return d


def _settled_free(dev):
    """Free bytes on a CUDA device once released arrays are back with the driver: unreferenced
    (collected, in case of reference cycles) and the device synchronized (when Warp's memory pool,
    release threshold 0, returns them).  Read before that, a model just released still counts."""
    import gc

    gc.collect()
    warp().synchronize_device(dev)
    return float(dev.free_memory)


def raw_sf(sh, ijk, shape, dirs, among=None, chunk=20000):
    """The fODF clipped at 0 (no floor, not normalised) on the walkers' directions, at the voxels
    ``ijk`` and their 26 neighbours (only those in ``among``, a mask, if given): (rows, M) float32
    and an (X, Y, Z) map of their rows (-1 elsewhere, read as no fODF).  What ``bin_fodf``
    interpolates."""
    from scipy import ndimage as ndi

    from .geometry import _sh_basis

    near = np.zeros(shape, bool)
    near[tuple(ijk.T)] = True
    near = ndi.binary_dilation(near, np.ones((3, 3, 3), bool))
    if among is not None:
        near &= np.asarray(among, bool)
    idx = np.argwhere(near)
    rows = np.full(shape, -1, np.int64)
    rows[tuple(idx.T)] = np.arange(len(idx))
    B = _sh_basis(dirs.v, sh.shape[-1])
    sf = np.empty((len(idx), dirs.M), np.float32)
    for s in range(0, len(idx), chunk):
        sf[s:s + chunk] = np.maximum(np.asarray(sh[tuple(idx[s:s + chunk].T)], np.float64) @ B, 0.0)
    return sf, rows


def trilinear_or_nan(vol, pos):
    """``vol`` (X, Y, Z) interpolated trilinearly at points ``pos`` (n, 3; voxel centres at integers)
    from the 8 voxel centres around each; NaN where one it weighs is off the grid (pyAFQ's tracker
    stops there: outside the image).  (n,)."""
    pos = np.asarray(pos, np.float64)
    lo = np.floor(pos).astype(np.int64)
    fr = pos - lo
    shape = np.asarray(vol.shape)
    out = np.zeros(len(pos))
    bad = np.any((lo < 0) | (lo >= shape) | ((lo + 1 >= shape) & (fr > 0)), axis=1)   # a corner it weighs, off
    q0 = np.where(bad[:, None], 0, lo)
    for c in itertools.product((0, 1), repeat=3):
        w = np.prod(np.where(np.asarray(c, bool), fr, 1.0 - fr), axis=1)
        q = np.minimum(q0 + np.asarray(c), shape - 1)      # a corner of weight 0 at the far edge: any voxel
        out += w * vol[q[:, 0], q[:, 1], q[:, 2]]
    out[bad] = np.nan
    return out


def stop_value(stop_map, pos):
    """The stop map at points ``pos`` (n, 3; the fODF grid's voxel coordinates), trilinear: on the
    fODF's grid (an array), or on its own grid (``(volume, A)``, A the 4x4 from the fODF's voxel
    coordinates to the volume's; e.g. the WM PVE at the T1's resolution).  NaN off its grid."""
    if isinstance(stop_map, tuple):
        vol, A = stop_map
        pos = np.asarray(pos, np.float64) @ np.asarray(A, np.float64)[:3, :3].T + np.asarray(A, np.float64)[:3, 3]
        return trilinear_or_nan(vol, pos)
    return trilinear_or_nan(stop_map, pos)


def point_domain(stop_map, shape, threshold, bins, points, chunk=20000):
    """The voxels (of the fODF's grid, ``shape``) with a Gauss point of the turns (``bins``^3 bins of
    ``points``^3 each) where the stop map exceeds ``threshold``: where walkers of stop "interp"
    can go on (elsewhere every point stops them).  For a stop map on its own grid (stop_value's
    (volume, A)); candidates first: the voxels a trilinear cell with a corner above the threshold
    reaches (only there can the map exceed it)."""
    vol, A = stop_map
    vol = np.asarray(vol)
    Ainv = np.linalg.inv(np.asarray(A, np.float64))
    node = np.argwhere(vol > threshold).astype(np.float64)
    cand = np.zeros(shape, bool)
    corners = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    for a0 in range(0, len(node), 200000):
        c = (node[a0:a0 + 200000, None, :] + corners[None]) @ Ainv[:3, :3].T + Ainv[:3, 3]
        lo = np.clip(np.floor(c.min(1) + 0.5).astype(np.int64), 0, np.asarray(shape) - 1)
        hi = np.clip(np.floor(c.max(1) + 0.5).astype(np.int64), 0, np.asarray(shape) - 1)
        span = (hi - lo).max(0) + 1
        for d in itertools.product(*[range(int(n)) for n in span]):
            q = lo + np.asarray(d)
            ok = np.all(q <= hi, axis=1)
            cand[tuple(q[ok].T)] = True
    qu, _ = gauss_points(points)
    k3 = bins ** 3
    rel = np.concatenate([(np.array([b // (bins * bins), (b // bins) % bins, b % bins])[None, :] + qu) / bins - 0.5
                          for b in range(k3)])
    ijk = np.argwhere(cand)
    keep = np.zeros(len(ijk), bool)
    for a0 in range(0, len(ijk), chunk):
        val = stop_value(stop_map, (ijk[a0:a0 + chunk, None, :] + rel[None]).reshape(-1, 3)).reshape(-1, len(rel))
        keep[a0:a0 + chunk] = (np.nan_to_num(val, nan=-np.inf) > threshold).any(1)
    out = np.zeros(shape, bool)
    out[tuple(ijk[keep].T)] = True
    return out


def interp_domain(vol, threshold):
    """The voxels where ``vol``, trilinear between voxel centres, exceeds ``threshold`` somewhere:
    walkers of stop "interp" are only there.  Exact: in each of a voxel's 8 octants the trilinear
    function is multilinear, so its largest value is at one of the 27 points of the voxel's
    centre, face and edge midpoints and corners (each the mean of 1, 2, 4 or 8 voxel values; one
    off the grid counts as no value: -inf)."""
    v = np.pad(np.asarray(vol, np.float64), 1, constant_values=-np.inf)
    X, Y, Z = np.asarray(vol).shape
    best = np.full((X, Y, Z), -np.inf)
    for o in itertools.product((-1, 0, 1), repeat=3):
        acc = np.zeros((X, Y, Z))
        n = 0
        for c in itertools.product(*[(0, d) if d else (0,) for d in o]):
            acc = acc + v[1 + c[0]:1 + c[0] + X, 1 + c[1]:1 + c[1] + Y, 1 + c[2]:1 + c[2] + Z]
            n += 1
        best = np.maximum(best, acc / n)
    return best > threshold

def point_fodf(sf, rows, pos, dirs, normalise=True):
    """The clipped fODF (``raw_sf``'s rows) interpolated trilinearly at points ``pos`` (n, 3; voxel
    centres at integers) from the 8 voxel centres around each (as DIPY's trackers; a voxel with no
    row counts as zero), then as ``fodf_on``: values below PMF_THRESHOLD x its maximum zeroed,
    normalised to integrate to 1 over the sphere (``normalise``).  (n, M) float32."""
    from .geometry import PMF_THRESHOLD

    pos = np.asarray(pos, np.float64)
    lo = np.floor(pos).astype(np.int64)
    fr = (pos - lo).astype(np.float32)
    shape = np.asarray(rows.shape)
    out = np.zeros((len(pos), dirs.M), np.float32)
    for c in itertools.product((0, 1), repeat=3):
        w = np.prod(np.where(np.asarray(c, bool), fr, 1.0 - fr), axis=1)
        if not w.any():
            continue
        q = lo + np.asarray(c)
        ok = (w > 0) & np.all((q >= 0) & (q < shape), axis=1)
        r = np.full(len(pos), -1, np.int64)
        r[ok] = rows[tuple(q[ok].T)]
        ok &= r >= 0
        out[ok] += w[ok, None] * sf[r[ok]]
    out[out < PMF_THRESHOLD * out.max(1, keepdims=True)] = 0.0
    if not normalise:
        return out
    tot = (out @ dirs.w.astype(np.float32))[:, None]
    return np.divide(out, tot, out=np.zeros_like(out), where=tot > 0)


class _PointFODF:
    """The fODF at points of every voxel, on the device (galerkin.point_fodf there, one point at a
    time for all voxels at once): which directions have fODF at any of them (the states), and
    each bin's weights from them."""

    def __init__(self, wp, K, dev, sf, rows, ijk, dirs, every=LOG_SECONDS):
        from .geometry import PMF_THRESHOLD

        self.wp, self.K, self.dev, self.every = wp, K, dev, every
        self.Nv, self.M = len(ijk), dirs.M
        self.thr = float(PMF_THRESHOLD)
        self.sf = wp.array(np.ascontiguousarray(sf, np.float32), dtype=wp.float32, device=dev)
        self.rows = wp.array(np.ascontiguousarray(rows, np.int32), dtype=wp.int32, device=dev)
        self.ijk = wp.array(np.ascontiguousarray(ijk, np.int32), dtype=wp.vec3i, device=dev)
        self.w = wp.array(np.asarray(dirs.w, np.float32), dtype=wp.float32, device=dev)
        self.fd = wp.zeros((self.Nv, self.M), dtype=wp.float32, device=dev)

    def _at(self, off):
        wp, K = self.wp, self.K
        wp.launch(K.pt_interp, dim=(self.Nv, self.M), inputs=[self.ijk, wp.vec3(*map(float, off)), self.rows, self.sf,
                                                               self.fd], device=self.dev)
        wp.launch(K.pt_norm, dim=self.Nv, inputs=[self.fd, self.w, self.thr, self.M], device=self.dev)

    def _progress(self, what, i, n, t0, last):
        now = time.perf_counter()
        if now - last >= self.every and i < n:
            self.wp.synchronize_device(self.dev)
            log.info("  %s: %d of %d points in %s", what, i, n, _secs(now - t0))
            return now
        return last

    def support(self, offs):
        """(Nv, M) bool: fODF at any of the points (offsets from the voxel centres)."""
        flags = self.wp.zeros((self.Nv, self.M), dtype=self.wp.uint8, device=self.dev)
        t0 = last = time.perf_counter()
        for i, off in enumerate(offs, 1):
            self._at(off)
            self.wp.launch(self.K.pt_flag, dim=(self.Nv, self.M), inputs=[self.fd, flags], device=self.dev)
            last = self._progress("fODF at the points (which directions)", i, len(offs), t0, last)
        return flags.numpy() > 0

    def tables(self, plist, evox, edir, lvox, ldir, conep, conei, ztab, efw):
        """For (bin, offset, weight) in plist: the states' f w there, times the weight, into efw[:,
        bin] (their mean over the bin's points), and (ztab given: one point per bin) the landings'
        1 / (cone's f w) into ztab[:, bin]."""
        wp, K = self.wp, self.K
        P, Lt = int(evox.shape[0]), int(lvox.shape[0])
        t0 = last = time.perf_counter()
        for i, (b, off, qw) in enumerate(plist, 1):
            self._at(off)
            wp.launch(K.pt_state, dim=P, inputs=[evox, edir, self.fd, efw, b, qw], device=self.dev)
            if ztab is not None:
                wp.launch(K.pt_landing, dim=Lt, inputs=[lvox, ldir, self.fd, conep, conei, ztab, b], device=self.dev)
            last = self._progress("fODF at the points (bin weights)", i, len(plist), t0, last)
        wp.synchronize_device(self.dev)

    def release(self):
        self.sf = self.rows = self.ijk = self.w = self.fd = None     # (the turns may hold on to some)


def _write(res, stem, affine, cfg, rows):
    b = f"{stem}_bundle-{re.sub(r'[^A-Za-z0-9]', '', res.name)}"
    maps = dict(density=(res.density, "map"), endpoints=(res.endpoints, "map"), seedprob=(res.seed_prob, "probmap"))
    if res.odf_coef is not None and len(res.odf_coef):
        odf = np.zeros(res.density.shape + res.odf_coef.shape[1:], np.float32)
        odf[tuple(res.odf_ijk.T)] = res.odf_coef
        maps["visitodf"] = (odf, "dwimap")
    for desc, (vol, suffix) in maps.items():
        nib.save(nib.Nifti1Image(vol, affine), f"{b}_desc-{desc}_{suffix}.nii.gz")
    with open(f"{b}_desc-summary.json", "w") as fh:
        json.dump(dict(bundle=res.name, config=cfg.key(), result=res.info, valid_ends=["gm", "edge"],
                       density="path length (mm) of the bundle's walkers in each voxel",
                       endpoints="the bundle walkers' ends per voxel (grey matter / edge; two per walker)",
                       visitodf="the bundle walkers' directions per voxel: SH (descoteaux07, legacy, lmax 8), "
                                "path length per steradian",
                       seedprob="P(a walker launched from this grey-matter / edge voxel is in the bundle)"),
                  fh, indent=1)
    i = res.info
    rows.append(dict(bundle=res.name, expected_streamlines=round(i["expected_bundle_streamlines"], 4),
                     launched=round(i["launched_walkers"], 1), solves=i["n_solves"], columns=i["n_columns"],
                     sweeps=i["sweeps"], seconds=i["seconds"]))


def gauss_points(g):
    """g^3 Gauss-Legendre points of the unit cube and their weights (summing to 1): exact for
    polynomials of degree 2g - 1 per axis (g >= 3: products of two quadratics)."""
    x, w = np.polynomial.legendre.leggauss(int(g))
    u, w = (x + 1) / 2, w / 2
    U = np.stack(np.meshgrid(u, u, u, indexing="ij"), -1).reshape(-1, 3)
    W = np.einsum("i,j,k->ijk", w, w, w).ravel()
    return U, W


def bin_fodf(sf, rows, ijk, kb, s, dirs):
    """The fODF in bin ``s`` (sub index as the kernels': (x * kb + y) * kb + z) of each voxel of
    ``ijk``: the clipped fODF interpolated trilinearly at the bin's center from the 8 voxel
    centers around it (as DIPY's trackers interpolate theirs; a voxel ``raw_sf`` has no row for
    counts as zero -- the bin's own voxel always weighs in), then as ``fodf_on``: values below PMF_THRESHOLD x its maximum zeroed, normalised to
    integrate to 1 over the sphere.  (N, M) float32."""
    b = np.array([s // (kb * kb), (s // kb) % kb, s % kb], np.float64)
    return point_fodf(sf, rows, ijk + (b + 0.5) / kb - 0.5, dirs)      # the bin's center


def _morton(ijk):
    """Morton (Z-order) codes of voxel indices (n, 3): neighbours in space stay close in the order."""
    code = np.zeros(len(ijk), np.int64)
    for bit in range(11):
        for a in range(3):
            code |= ((ijk[:, a].astype(np.int64) >> bit) & 1) << (3 * bit + a)
    return code


def _order(v):
    """The directions in bands of polar angle, by azimuth (snaking): consecutive ones stay close,
    so the shift's threads in a row of directions share their pieces' cells."""
    th = np.arccos(np.clip(v[:, 2], -1, 1))
    ph = np.arctan2(v[:, 1], v[:, 0])
    nb = max(1, round(math.sqrt(len(v) / 4)))
    band = np.minimum((th / np.pi * nb).astype(int), nb - 1)
    return np.lexsort((np.where(band % 2 == 1, -ph, ph), band))


class Galerkin:
    """The white matter (or ``region`` of it) in bins, the walkers' directions, the fODF and the
    pieces of a step, on a Warp device; the solves; and bundles from them.

        model = Galerkin(sh, zooms, transient, absorb, GalerkinConfig(bins=1), device="cuda")
        res = model.bundle(dict(name="IFOF", include=[...], exclude=..., start=..., end=...))
    """

    def __init__(self, sh, zooms, transient, absorb, cfg=None, device=None, graphs=None, region=None, procs=None,
                 verbose=False, log_every=None, stop_map=None):
        from . import _warp_galerkin

        t0 = time.time()
        stages = []
        tick = [time.time()]

        def stage(name):
            now = time.time()
            stages.append(f"{name} {_secs(now - tick[0])}")
            tick[0] = now

        self.verbose = bool(verbose)
        self.log_every = float(log_every) if log_every is not None else (5.0 if verbose else LOG_SECONDS)
        self.cfg = cfg = cfg or GalerkinConfig()
        self.wp = wp = warp()
        self.K = K = _warp_galerkin
        self.dev = dev = get_device(device)
        if dev.is_cpu and not wp.config.enable_cpu_blocks:
            wp.config.enable_cpu_blocks = True      # Warp's CPU device (tests): blocks with their threads and barriers
        if dev.is_cuda and wp.is_mempool_enabled(dev):
            wp.set_mempool_release_threshold(dev, 0)     # freed arrays go back: free memory tells the truth
        self.graphs = bool(dev.is_cuda) if graphs is None else bool(graphs)
        self.dirs = dirs = Directions(cfg.sphere, cfg.angle)
        M = dirs.M
        self.kb = kb = int(cfg.bins)
        if kb < 1:
            raise ValueError("bins must be a positive integer")
        self.k3 = k3 = kb ** 3
        transient = np.asarray(transient, bool)
        self.shape = shape = transient.shape
        self.V = int(np.prod(shape))
        self.zooms = zooms = np.abs(np.asarray(zooms, np.float64))[:3]
        if cfg.step_unit not in ("mm", "vox"):
            raise ValueError(f"step_unit: 'mm' or 'vox', not {cfg.step_unit!r}")
        if not float(cfg.step) > 0:
            raise ValueError(f"step must be positive, not {cfg.step!r}")
        side = float(zooms.min())
        self.smm = smm = float(cfg.step) * (side if cfg.step_unit == "vox" else 1.0)
        self.step_vox = smm / side
        lo, hi = STEP_VOX_WARN
        (log.warning if not lo <= self.step_vox <= hi else log.info)(
            "galerkin: step %.4g mm = %.4g voxels (smallest voxel side %.4g mm)%s", smm, self.step_vox, side,
            "" if lo <= self.step_vox <= hi else f": outside {lo:g}-{hi:g} voxels (shorter: many more steps per "
            f"solve; longer: a step may skip a voxel's fODF)")
        hbin = zooms / kb
        dvec = smm * dirs.v / hbin[None, :]
        dom = transient if region is None else transient & np.asarray(region, bool)
        ijk = np.argwhere(dom).astype(np.int64)
        self.ijk = ijk = ijk[np.argsort(_morton(ijk), kind="stable")]
        self.Nv = Nv = len(ijk)
        if Nv == 0:
            raise ValueError("no white matter to walk in")
        self.lin = np.ravel_multi_index(tuple(ijk.T), shape)
        vid = np.full(shape, -1, np.int32)
        vid[tuple(ijk.T)] = np.arange(Nv, dtype=np.int32)
        absorb = np.asarray(absorb)
        self.valid = np.isin(absorb, VALID_ENDS).ravel()
        if cfg.stop not in ("", "interp"):
            raise ValueError(f"stop: '' (the tissue labels' voxels) or 'interp', not {cfg.stop!r}")
        if cfg.until not in ("walkers", "ends"):
            raise ValueError(f"until: 'walkers' or 'ends', not {cfg.until!r}")
        if not cfg.prune >= 0:
            raise ValueError(f"prune: a share >= 0, not {cfg.prune!r}")
        self.stop = cfg.stop == "interp"
        if self.stop:                # every stop is an end, in whatever voxel (the ROIs decide)
            native = isinstance(stop_map, tuple)
            if stop_map is None or (not native and tuple(np.shape(stop_map)) != tuple(shape)):
                raise ValueError("stop 'interp' needs stop_map: the WM PVE on the fODF's grid, or (volume, 4x4 "
                                 "from the fODF's voxels to the volume's) on its own grid")
            self.valid[:] = True
        # the fODF on the directions, its support (the states) and its cones' mass
        stage("white matter")
        # interp: each bin its own fODF (bin_fodf); the states are where any bin of the voxel has
        # fODF, their f w and the landings' 1 / Z per bin (filled in below, once the landings are known)
        if cfg.interp not in ("", "wm", "all"):
            raise ValueError(f"interp: '' (off), 'wm' or 'all', not {cfg.interp!r}")
        self.points = npq = int(cfg.points)
        if npq and npq < 3:
            raise ValueError("points: 0 (off) or 3 or more per axis (fewer do not keep a constant fODF's turn exact)")
        self.interp = interp = (bool(cfg.interp) and kb > 1) or npq > 0
        self.ib = int(interp)
        self._raw = None
        if npq:         # the Gauss points of a bin (unit-bin coordinates) and where they lie in voxel coordinates
            self.qu, self.qw = gauss_points(npq)
            self.Q = len(self.qw)
        # the turns at points: a thread per direction (fused_point_kernels)
        if npq and M > K.FUSED_MAX_DIRS:
            raise ValueError(f"points: the turns at points take at most {K.FUSED_MAX_DIRS} directions, a thread "
                             f"each (this sphere: {M})")
        if self.stop and not npq:
            raise ValueError("stop 'interp' is decided at the points of the turns: points >= 3")
        if interp:      # "wm": the white matter's fODFs alone (the field the walkers know without interp)
            sf_raw, sf_rows = raw_sf(sh, ijk, shape, dirs, among=transient if cfg.interp == "wm" else None)
            # the points the fODF is read at: Gauss points of each bin (points) or its centre
            if npq:
                plist = [(b, (np.array([b // (kb * kb), (b // kb) % kb, b % kb]) + self.qu[q]) / kb
                          - 0.5, float(self.qw[q])) for b in range(k3) for q in range(self.Q)]
            else:
                plist = [(b, (np.array([b // (kb * kb), (b // kb) % kb, b % kb]) + 0.5) / kb - 0.5, 1.0)
                         for b in range(k3)]
            pf = _PointFODF(wp, K, dev, sf_raw, sf_rows, ijk, dirs, self.log_every)
            F = pf.support([o for _, o, _ in plist]).astype(np.float32)   # the states: fODF at any point
        else:
            F = fodf_on(np.asarray(sh)[tuple(ijk.T)], dirs)
        stage("fODF" + (f" in each of the {k3} bins" if interp else ""))
        self.order = _order(dirs.v)
        rank = np.empty(M, np.int64)
        rank[self.order] = np.arange(M)
        mm, vv = np.nonzero(F.T > 0)                       # direction by direction, voxels in order
        o = np.argsort(rank[mm], kind="stable")            # directions in a sweep over the sphere
        mm, vv = mm[o], vv[o]
        self.P = P = len(vv)
        if P == 0:
            raise ValueError("no fODF in the white matter")
        eidx = np.full((M, Nv), -1, np.int32)
        eidx[mm, vv] = np.arange(P, dtype=np.int32)
        efw = (F[vv, mm] * dirs.w[mm]).astype(np.float32)
        cone = np.zeros((M, M), np.float32)
        for m in range(M):
            cone[m, dirs.cidx[dirs.cptr[m]:dirs.cptr[m + 1]]] = 1.0
        zi = np.empty((M, Nv), np.float32)
        fw = F * dirs.w[None, :].astype(np.float32)
        for s in range(0, Nv, 50000):
            Z = fw[s:s + 50000] @ cone.T
            zi[:, s:s + 50000] = np.divide(1.0, Z, out=np.zeros_like(Z), where=Z > 0).T
        del F, fw
        stage("support and cones")
        # the pieces of a step
        geo = geometry(dvec, smm, procs)
        self.npieces = np.diff(geo["gptr"])
        stage("pieces")
        # Walkers land in (bin, m) only from a voxel within a step upwind where m is in the fODF's
        # support (states live nowhere else): zero zi elsewhere, and the per-(bin, direction)
        # kernels (fwd_shift, bwd_turn) skip it.  Nothing there is ever read, so the results are
        # the same; with wide cones most directions with fODF in their cone are such.
        zpos = float((zi > 0).mean())
        sup = eidx >= 0
        need = np.zeros((M, Nv), bool)
        dims = np.asarray(shape)
        offs = {}
        for m in range(M):
            land = geo["land"][geo["gptr"][m]:geo["gptr"][m + 1]]
            ax = [sorted({(s_ + l) // kb for s_ in range(kb) for l in set(land[:, a].tolist())}) for a in range(3)]
            for o in itertools.product(*ax):
                offs.setdefault(o, []).append(m)
        for o, ms in offs.items():
            src = ijk - np.asarray(o)
            ok = np.all((src >= 0) & (src < dims), axis=1)
            sv = np.full(Nv, -1, np.int64)
            sv[ok] = vid[tuple(src[ok].T)]
            ok = sv >= 0
            ms = np.asarray(ms)
            need[np.ix_(ms, np.flatnonzero(ok))] |= sup[np.ix_(ms, sv[ok])]
        zi[~need] = 0.0
        del need, sup
        self.landing_share = (float((zi > 0).mean()), zpos)
        self.state_share = P / float(Nv * M)
        stage("landings")
        # the launch faces: a valid end voxel (not white matter) next to the domain
        faces = []
        base = ijk * kb
        for a in range(3):
            a1, a2 = (a + 1) % 3, (a + 2) % 3
            for side in (-1, 1):
                q = ijk.copy()
                q[:, a] += side
                ok = np.all((q >= 0) & (q < np.asarray(shape)), axis=1)
                qq = np.where(ok[:, None], q, 0)
                lin = np.ravel_multi_index(tuple(qq.T), shape)
                ok &= self.valid[lin] & ~transient.ravel()[lin]
                n = int(ok.sum())
                if not n:
                    continue
                p0 = base[ok].astype(np.float64)
                b0 = base[ok].copy()
                if side == 1:                  # the face at the voxel's + side: walkers enter heading -
                    p0[:, a] += kb
                    b0[:, a] += kb - 1
                faces.append((np.flatnonzero(ok), q[ok], np.full(n, a), np.full(n, -side), p0, b0,
                              np.tile(_E[a1].astype(np.int64), (n, 1)), np.tile(_E[a2].astype(np.int64), (n, 1))))
        cat = [np.concatenate([f[i] for f in faces]) if faces else np.zeros((0, 3) if i in (1, 4, 5, 6, 7) else 0)
               for i in range(8)]
        self.nfaces = len(cat[0])
        g = 0.5 / math.sqrt(3.0)
        pts = [(i + 0.5 + su * g, j + 0.5 + sv * g) for i in range(kb) for j in range(kb)
               for su in (-1, 1) for sv in (-1, 1)]
        # the landings, voxel by voxel (the turns' buffer), and the threads' orders
        lv, lm = np.nonzero(zi.T > 0)
        self.Lt = Lt = len(lv)
        lzi = zi.T[lv, lm].astype(np.float32)
        lidx = np.full((Nv, M), -1, np.int32)
        lidx[lv, lm] = np.arange(Lt, dtype=np.int32)
        lperm = np.lexsort((lv, rank[lm])).astype(np.int32)      # the shift: direction by direction
        edir = mm.astype(np.int32)
        sperm = np.lexsort((mm, vv)).astype(np.int32)            # the turn: voxel by voxel
        vptr = np.concatenate([[0], np.cumsum(np.bincount(vv, minlength=Nv))]).astype(np.int32)
        lptr = np.concatenate([[0], np.cumsum(np.bincount(lv, minlength=Nv))]).astype(np.int32)
        if interp:                      # per bin (and point): filled on the device below, once the states are there
            efw = np.zeros((P, k3), np.float32)          # with points: their mean over the bin (launches, pictures)
            lzi = (np.ones if npq else np.zeros)((Lt, k3), np.float32)   # points: the turn divides, not the shift
            if npq:
                self._raw = (sf_raw, sf_rows)            # seeds read the fODF at their own point
            del sf_raw, sf_rows
        else:
            efw, lzi = efw.reshape(-1, 1), lzi.reshape(-1, 1)
        if dev.is_cuda:
            free = _settled_free(dev)
            need = self.need(Nv, M, P, Lt, k3, geo)
            self.state_bytes = 40.0 * P * k3
            if need - self.state_bytes > 0.9 * free:
                log.warning("galerkin: a bundle needs about %.1f GB on %s, seeded walkers %.1f GB, and %.1f GB is "
                            "free: expect out-of-memory errors; use fewer bins, a smaller sphere or a longer step",
                            need / 1e9, dev, (need - self.state_bytes) / 1e9, free / 1e9)
            elif need > 0.9 * free:
                log.warning("galerkin: a bundle needs about %.1f GB on %s and %.1f GB is free: expect out-of-memory "
                            "errors in bundles (seeded walkers need one state fewer, %.1f GB: they fit)",
                            need / 1e9, dev, free / 1e9, (need - self.state_bytes) / 1e9)
        stage("launch faces and landings")

        def put(a, dt):
            return wp.array(np.ascontiguousarray(a), dtype=dt, device=dev)

        v3i, v3 = wp.vec3i, wp.vec3
        self.d_vid = put(vid, wp.int32)
        self.d_vijk = put(ijk.astype(np.int32), v3i)
        self.d_eidx = put(eidx, wp.int32)
        self.d_eidxT = put(eidx.T, wp.int32)
        self.d_lidx = put(lidx, wp.int32)
        self.d_ldir, self.d_lvox = put(lm.astype(np.int32), wp.int32), put(lv.astype(np.int32), wp.int32)
        self.d_lzi = put(lzi, wp.float32)               # (Lt, 1), or (Lt, kb^3) with interp
        self.d_lperm, self.d_sperm = put(lperm, wp.int32), put(sperm, wp.int32)
        self.d_vptr, self.d_lptr = put(vptr, wp.int32), put(lptr, wp.int32)
        self.k_fwd_turn, self.k_bwd_turn = K.turn_kernels(M)
        self.d_edir, self.d_evox = put(edir, wp.int32), put(vv.astype(np.int32), wp.int32)
        self.d_efw = put(efw, wp.float32)               # (P, 1), or (P, kb^3) with interp
        if npq:                                          # the turns at points: their buffers
            self.d_phi = put(basis(self.qu).astype(np.float32), K.vec10)
            self.d_qw = put(self.qw.astype(np.float32), wp.float32)
            # each point's corner block (offset from the voxel's centre) and place in it
            off = np.zeros((k3, self.Q, 3), np.int32)
            frac = np.zeros((k3, self.Q, 3), np.float32)
            order = np.zeros((k3, self.Q), np.int32)
            for b in range(k3):
                rel = (np.array([b // (kb * kb), (b // kb) % kb, b % kb])[None, :] + self.qu) / kb - 0.5
                lo = np.floor(rel)
                off[b], frac[b] = lo.astype(np.int32), (rel - lo).astype(np.float32)
                order[b] = np.lexsort((off[b][:, 2], off[b][:, 1], off[b][:, 0]))     # points sharing corners together
            self.d_qoff = put(off.reshape(-1, 3), wp.vec3i)
            self.d_qfrac = put(frac.reshape(-1, 3), wp.vec3)
            self.d_qorder = put(order.ravel(), wp.int32)
            # the turns: a thread per direction, the bin's values in registers
            self.k_fwd_fused, self.k_bwd_fused = K.fused_point_kernels(M)
            self.fused_block = K.fused_block(M)
            if self.stop:       # each point's stop flag: the stop map, trilinear there, <= the threshold
                rel = np.stack([(np.array([b // (kb * kb), (b // kb) % kb, b % kb])[None, :] + self.qu) / kb - 0.5
                                for b in range(k3)]).reshape(-1, 3)
                flags = np.zeros((Nv, k3 * self.Q), np.uint8)
                for a0 in range(0, Nv, 20000):
                    pos = (ijk[a0:a0 + 20000, None, :] + rel[None]).reshape(-1, 3)
                    val = stop_value(stop_map, pos)
                    flags[a0:a0 + 20000] = (np.isnan(val) | (val <= cfg.stop_threshold)).reshape(-1, k3 * self.Q)
                self.d_stopq = put(flags, wp.uint8)
                log.info("galerkin: stopping as pyAFQ's (stop map <= %g, trilinear at each point): %.1f%% of the "
                         "%d points stop; %d voxels have points that do", cfg.stop_threshold, 100 * flags.mean(),
                         flags.size, int(flags.any(1).sum()))
                del flags
        del zi, eidx, lidx
        self.d_conep, self.d_conei = put(dirs.cptr, wp.int32), put(dirs.cidx, wp.int32)
        if npq:             # the turns at points: the cone indices as uint16 (directions < 2^16)
            self.d_conei_f = put(np.asarray(dirs.cidx).astype(np.uint16), wp.uint16)
        if interp:      # f w of each state and 1 / (its cone's f w) of each landing, at every point
            pf.tables(plist, self.d_evox, self.d_edir, self.d_lvox, self.d_ldir, self.d_conep, self.d_conei,
                      None if npq else self.d_lzi, self.d_efw)
            if npq:                 # the turns read the fODFs themselves
                self.d_sf, self.d_rows, self.d_w = pf.sf, pf.rows, pf.w
            pf.release()
            del pf
            stage(f"fODF at {self.Q} points of each of the {k3} bins" if npq else "fODF weights per bin")
        self.d_gptr, self.d_land = put(geo["gptr"], wp.int32), put(geo["land"], v3i)
        self.d_A = put(geo["A"], K.mat10)
        self.d_AT = put(np.transpose(geo["A"], (0, 2, 1)), K.mat10)
        self.d_amass = put(geo["a"], K.vec10)
        self.d_cptr, self.d_cell = put(geo["cptr"], wp.int32), put(geo["cell"], v3i)
        self.d_Wc, self.d_Bc = put(geo["W"], K.mat10), put(geo["B"], K.vec10)
        self.d_omega, self.d_dvec = put(dirs.v.astype(np.float32), v3), put(dvec.astype(np.float32), v3)
        self.d_fvox = put(cat[0].astype(np.int32), wp.int32)
        self.d_fgv = put(cat[1].reshape(-1, 3).astype(np.int32), v3i)
        self.d_fax, self.d_fsg = put(cat[2].astype(np.int32), wp.int32), put(cat[3].astype(np.int32), wp.int32)
        self.d_fp0 = put(cat[4].reshape(-1, 3).astype(np.float32), v3)
        self.d_fb0 = put(cat[5].reshape(-1, 3).astype(np.int32), v3i)
        self.d_fu = put(cat[6].reshape(-1, 3).astype(np.int32), v3i)
        self.d_fv = put(cat[7].reshape(-1, 3).astype(np.int32), v3i)
        self.d_gpts = put(np.array(pts, np.float32), wp.vec2)
        self.d_none3 = wp.zeros((1, 1, 1), dtype=wp.float32, device=dev)
        self.d_none2 = wp.zeros((1, 1), dtype=wp.float32, device=dev)
        if not self.stop:
            self.d_stopq = wp.zeros((1, 1), dtype=wp.uint8, device=dev)
        self.d_stopend = wp.zeros(Nv if self.stop else 1, dtype=wp.float64, device=dev)   # the walkers stopped
        self.use_stop = int(self.stop)
        self.nblk = 65536 if dev.is_cuda else 64
        self.ybuf = self.rho = None
        self.movie = None                # a SolveMovie recording the solves (mpat.solvemovie), or None
        self.d_ones = None
        stage("upload")
        if not getattr(Galerkin, "_kernels_on", {}).get(str(dev)):
            log.info("galerkin: loading the kernels on %s (the first run compiles them, a minute or two; then cached)",
                     dev)
            wp.load_module(K, device=dev)
            Galerkin._kernels_on = {**getattr(Galerkin, "_kernels_on", {}), str(dev): True}
            stage("kernels")
        log.info("galerkin setup: %s", ", ".join(stages))
        log.info("galerkin on %s: %d white-matter voxels (%s mm) x %d^3 bins, %d directions (sphere %s, %.1f per "
                 "turn within %g deg), step %.3g mm; fODF support %.0f%% (%d states x %d bins); %.1f pieces per step "
                 "(at most %d); walkers land in %.0f%% of (bin, direction) (%.0f%% with fODF in the cone); %d launch "
                 "faces; about %.1f GB per bundle; "
                 "set up in %.1fs", dev, Nv, "x".join(f"{z:g}" for z in zooms), kb, M, cfg.sphere,
                 (dirs.cptr[1:] - dirs.cptr[:-1]).mean(), cfg.angle, smm, 100.0 * P / (Nv * M), P, k3,
                 self.npieces.mean(), self.npieces.max(), 100 * self.landing_share[0], 100 * self.landing_share[1],
                 self.nfaces, self.need(Nv, M, P, Lt, k3, geo) / 1e9, time.time() - t0)

    def need(self, Nv, M, P, Lt, k3, geo):
        """Device bytes a bundle needs at the peak: the lookups (states and landings by voxel and
        direction), the pieces, the density by direction, two state arrays (forward: the walkers,
        stepped in place, and their sum over the steps; backward: the importance and that sum)
        and the turns' buffer (the landings)."""
        return (12.0 * Nv * M + 16.0 * P + 12.0 * Lt + sum(v.nbytes for v in geo.values()) * 1.2 + 4.0 * Nv * M
                + 2 * 40.0 * P * k3 + 40.0 * Lt * k3 + 8.0 * self.V * 3
                + (4.0 * (P + Lt) * k3 if self.interp else 0.0)
                + (6.0 * Nv * M if self.points else 0.0))

    def release(self):
        """Free the device arrays."""
        for k in [k for k in vars(self) if k.startswith("d_")]:
            setattr(self, k, None)
        self.ybuf = self.rho = None

    # -------------------------------------------------------------- device helpers
    def _vec10(self, n):
        """(n, kb^3) vec10 on the device, zeroed, viewing a float array (n, kb^3 x 10) -- its ``f``
        -- that the tiled turns gather from."""
        wp = self.wp
        f = wp.zeros((n, self.k3 * 10), dtype=wp.float32, device=self.dev)
        v = wp.array(ptr=f.ptr, dtype=self.K.vec10, shape=(n, self.k3), device=self.dev)
        v.f = f
        return v

    def _state(self):
        return self._vec10(self.P)

    def _picture(self, x, importance):
        """A state on the device as a (X, Y, Z) volume: the walkers per voxel, or (importance) the
        fODF-weighted mean importance per voxel (a probability)."""
        wp = self.wp
        if importance:
            w, ib = self.d_efw, self.ib
        else:
            if self.d_ones is None:
                self.d_ones = wp.array(np.ones((self.P, 1), np.float32), dtype=wp.float32, device=self.dev)
            w, ib = self.d_ones, 0
        out = wp.zeros(self.Nv, dtype=wp.float32, device=self.dev)
        wp.launch(self.K.voxel_sum, dim=(self.P, self.k3), inputs=[x, self.d_evox, w, ib, out], device=self.dev)
        vol = np.zeros(self.shape, np.float32)
        vol[tuple(self.ijk.T)] = np.maximum(out.numpy(), 0) / (self.k3 if importance else 1)
        return vol

    def _grid(self, a):
        return self.wp.array(np.asarray(a, np.float32).reshape(self.shape), dtype=self.wp.float32, device=self.dev)

    def _alive(self, avoid):
        return self.wp.array((~np.asarray(avoid, bool).ravel()[self.lin]).astype(np.uint8), dtype=self.wp.uint8,
                             device=self.dev)

    def _launch(self, mode, alive, bval=None, tgrid=None, sign=1.0, outv=None, rho=None):
        wp, K = self.wp, self.K
        wp.launch(K.launch, dim=(self.nfaces, int(self.d_gpts.shape[0])),
                  inputs=[int(mode), self.d_fvox, self.d_fgv, self.d_fax, self.d_fsg,
                          self.d_fp0, self.d_fb0, self.d_fu, self.d_fv, self.d_gpts, self.d_vid, alive, self.d_eidx,
                          self.d_efw, self.ib, self.d_omega, self.d_dvec, self.smm, self.kb,
                          self.d_none3 if bval is None else bval, self.d_none3 if tgrid is None else tgrid,
                          self.d_lidx, self.d_lzi, self.ybuf, float(sign), self.ybuf, outv,
                          self.d_none2 if rho is None else rho],
                  device=self.dev)

    def _tick(self, prof, key):
        # profiling one step: wait for the device, add the time since the last tick to prof[key]
        if prof is not None:
            self.wp.synchronize_device(self.dev)
            now = time.perf_counter()
            if key:
                prof[key] = prof.get(key, 0.0) + now - prof["_t"]
            prof["_t"] = now

    def _profile_line(self, what, prof):
        parts = [f"{k} {1e3 * v:.1f} ms" for k, v in prof.items() if k != "_t"]
        log.info("  %s: one step kernel by kernel (synchronized): %s", what, ", ".join(parts))

    def _fwd_step(self, x, xn, C, alive, red, prof=None, ex=None, vis=None, conn=None, vn=None):
        """One step from x into xn; C += xn and red = its walkers, or (C None) xn's walkers leaving
        the white matter into ex (and, given vis, per voxel into it).  ``conn`` (g, threshold, cred):
        before the sum, xn's expected arrivals under the importance g into cred, a bin's walkers
        dropped where theirs are below the threshold (Galerkin.connect).  ``vn`` (P, kb^3): += each bin's
        |xn| (the norm of its coefficients), the visits a backward's pruning is bounded by."""
        wp, K, dev = self.wp, self.K, self.dev
        self._tick(prof, None)
        red.zero_()
        wp.launch(K.fwd_shift, dim=(self.Lt, self.k3),
                  inputs=[self.d_lperm, self.d_ldir, self.d_lvox, self.d_lzi, self.ib, x, self.d_eidx, self.d_vid, alive,
                          self.d_vijk, self.d_gptr, self.d_land, self.d_A, self.d_cptr, self.d_cell, self.kb, self.ybuf],
                  device=dev)
        self._tick(prof, "shift")
        self._fwd_turn(xn)
        self._tick(prof, "turn")
        if conn is not None:
            self._connect(xn, *conn)
            self._tick(prof, "importance")
        if vn is not None:
            wp.launch(K.norm_accum, dim=(self.P, self.k3), inputs=[xn, vn], device=dev)
            self._tick(prof, "visits")
        if C is None:
            wp.launch(K.mass, dim=self.nblk, inputs=[xn, self.k3, self.P * self.k3, self.nblk, 0, red], device=dev)
            self._exit_tally(xn, alive, ex)
            if vis is not None:
                self._visit(xn, vis)
        else:
            wp.launch(K.accum, dim=self.nblk, inputs=[xn, C, self.k3, self.P * self.k3, self.nblk, 0, red], device=dev)
        self._tick(prof, "sum")

    def _connect(self, x, g, thr, cred):
        """cred = (expected arrivals of x's walkers kept, the same summed in absolute value, of those
        dropped, the same in absolute value): x . g per bin; a bin's walkers whose |x . g| < thr are
        dropped (set to 0) -- thr 0: none."""
        cred.zero_()
        self.wp.launch(self.K.connect, dim=self.nblk, inputs=[x, g, self.k3, self.P * self.k3, self.nblk, float(thr),
                                                              cred], device=self.dev)

    def _fwd_turn(self, xn):
        if self.points:
            from .geometry import PMF_THRESHOLD

            self.wp.launch_tiled(self.k_fwd_fused, dim=(self.Nv, self.k3),
                                 inputs=[self.d_vijk, self.d_rows, self.d_sf, self.d_w, float(PMF_THRESHOLD),
                                         self.d_qorder, self.d_qoff, self.d_qfrac, self.Q, self.d_qw, self.d_phi,
                                         self.d_lidx, self.d_eidxT, self.d_conep, self.d_conei_f, self.ybuf,
                                         self.d_stopq, self.use_stop, self.d_stopend, xn],
                                 block_dim=self.fused_block, device=self.dev)
            return
        self.wp.launch_tiled(self.k_fwd_turn, dim=(self.Nv, self.k3),
                             inputs=[self.d_lidx, self.ybuf.f, self.d_vptr, self.d_sperm, self.d_edir, self.d_efw,
                                     self.ib, self.d_conep, self.d_conei, xn], block_dim=self.K.TURN_BLOCK, device=self.dev)

    def _bwd_turn(self, g, alive, tgrid=None):
        if self.points:
            from .geometry import PMF_THRESHOLD

            if self.stop and tgrid is None:
                raise ValueError("stop 'interp': the backward turn needs what ending in each voxel is worth (tgrid)")
            self.wp.launch_tiled(self.k_bwd_fused, dim=(self.Nv, self.k3),
                                 inputs=[self.d_vijk, self.d_rows, self.d_sf, self.d_w, float(PMF_THRESHOLD),
                                         self.d_qorder, self.d_qoff, self.d_qfrac, self.Q, self.d_qw, self.d_phi,
                                         self.d_lidx, self.d_eidxT, self.d_conep, self.d_conei_f, alive, g,
                                         self.d_stopq, self.use_stop, self.d_none3 if tgrid is None else tgrid,
                                         self.ybuf],
                                 block_dim=self.fused_block, device=self.dev)
            return
        self.wp.launch_tiled(self.k_bwd_turn, dim=(self.Nv, self.k3),
                             inputs=[self.d_eidxT, g.f, self.d_efw, self.ib, alive, self.d_lptr, self.d_ldir,
                                     self.d_lzi, self.d_conep, self.d_conei, self.ybuf], block_dim=self.K.TURN_BLOCK,
                             device=self.dev)

    def _bwd_step(self, g, gn, alive, tgrid, prof=None, prune=None):
        """One step of the importance from g into gn; ``prune`` (vn, threshold, red): then a bin's
        importance is dropped where |gn| x vn (the all-halves forward's visits there, in norm) is below
        the threshold, red[0] += that product (a bound on the arrivals it was worth)."""
        wp, K, dev = self.wp, self.K, self.dev
        self._tick(prof, None)
        self._bwd_turn(g, alive, tgrid)
        self._tick(prof, "turn")
        wp.launch(K.bwd_shift, dim=(self.P, self.k3),
                  inputs=[self.d_edir, self.d_evox, self.d_vijk, self.d_vid, alive, tgrid, self.d_gptr, self.d_land,
                          self.d_AT, self.d_amass, self.d_cptr, self.d_cell, self.kb, self.d_lidx, self.ybuf, gn],
                  device=dev)
        self._tick(prof, "shift")
        if prune is not None:
            vn, thr, red = prune
            wp.launch(K.bwd_prune, dim=self.nblk, inputs=[gn, vn, self.k3, self.P * self.k3, self.nblk, float(thr),
                                                           red], device=dev)
            self._tick(prof, "prune")

    def _replay(self, graphs, parity, fn, *args):
        """fn(*args), as a CUDA graph replayed after its first (compiling) call."""
        wp = self.wp
        if not self.graphs:
            fn(*args)
            return
        if graphs[parity] is None:
            if graphs.get("warm") is None:
                fn(*args)                     # compiles the kernels
                graphs["warm"] = True
                return
            with wp.ScopedCapture(device=self.dev) as cap:
                fn(*args)
            graphs[parity] = cap.graph
        wp.capture_launch(graphs[parity])

    # -------------------------------------------------------------- solves
    def forward(self, alive, bval, what="forward", targets=None):
        """The walkers launched from the voxels of ``bval`` (a (X, Y, Z) float32 device array of
        weights), stepped until those left are below tol of the launched.  Returns None if none
        launch, else (C: the walkers summed over the steps, (P, kb^3) on the device; where they
        leave the white matter, (V,); the steps; info).  ``targets``: as _forward_from's."""
        wp, K, dev = self.wp, self.K, self.dev
        t0 = time.perf_counter()
        x = self._state()                         # in place: the shift reads it into the landings first
        ex = wp.zeros(self.shape, dtype=wp.float64, device=dev)
        self.d_stopend.zero_()                        # (stop "interp": the first turn's stops count too)
        self.ybuf.zero_()                             # the launches' first step, then their first turn
        self._launch(K.FORWARD, alive, bval=bval, outv=ex)
        self._fwd_turn(x)
        return self._forward_from(x, ex, alive, what, t0, targets=targets)

    def forward_from(self, x0, alive, what="forward", keep_sum=True, visits=False, targets=None, conn=None,
                     vn=None):
        """As forward, from walkers already in the white matter: ``x0``, a state (P, kb^3) on the
        device (overwritten: the walkers are stepped in place).  Returns as forward; with
        ``keep_sum=False`` C is None and no second state is held: where the walkers leave is
        tallied step by step instead (the same sum, one state less on the device, a kernel more
        per step), and with ``visits`` the walkers per voxel too (info["visits"], (X, Y, Z): the
        same as the picture of C, summed step by step).  ``conn``, ``vn``: as _forward_from's."""
        t0 = time.perf_counter()
        ex = self.wp.zeros(self.shape, dtype=self.wp.float64, device=self.dev)
        self.d_stopend.zero_()
        vis = None
        if visits and not keep_sum:
            vis = self.wp.zeros(self.Nv, dtype=self.wp.float32, device=self.dev)
            if self.d_ones is None:
                self.d_ones = self.wp.array(np.ones((self.P, 1), np.float32), dtype=self.wp.float32,
                                            device=self.dev)
        out = self._forward_from(x0, ex, alive, what, t0, keep_sum, vis, targets, conn, vn)
        if vis is not None and out is not None:
            vol = np.zeros(self.shape, np.float32)
            vol[tuple(self.ijk.T)] = np.maximum(vis.numpy(), 0)
            out[3]["visits"] = vol
        return out

    def _exits(self, ex):
        """Where a solve's walkers ended, (V,): leaving the domain (ex) and, with stop "interp", stopping
        at points inside it."""
        out = ex.numpy().ravel()
        if self.stop:
            out[self.lin] += self.d_stopend.numpy()
        return out

    def _exit_tally(self, x, alive, ex):
        self.wp.launch(self.K.exit_tally, dim=(self.P, self.k3),
                       inputs=[x, self.d_edir, self.d_evox, self.d_vijk, self.d_vid, alive, self.d_gptr, self.d_amass,
                               self.d_cptr, self.d_cell, self.kb, ex], device=self.dev)

    def _visit(self, x, vis):
        self.wp.launch(self.K.voxel_sum, dim=(self.P, self.k3), inputs=[x, self.d_evox, self.d_ones, 0, vis],
                       device=self.dev)

    def _arrived(self, ex, C, alive, targets):
        """The walkers that have ended so far in each of ``targets`` ((label, (V,) weights) pairs): where
        they left (ex, and the points they stopped at) or, given C (their sum over the steps so far),
        where C's leave in their next step."""
        sc = ex
        if C is not None:
            sc = self.wp.clone(ex)
            self._exit_tally(C, alive, sc)
        out = self._exits(sc)
        return [float(out @ tv) for _, tv in targets]

    def _subnormal_text(self, x):
        """Verbose: how many of a state's nonzero coefficients are subnormal floats (below 1.2e-38)."""
        red = self.wp.zeros(2, dtype=self.wp.float64, device=self.dev)
        self.wp.launch(self.K.subnormals, dim=self.nblk, inputs=[x, self.k3, self.P * self.k3, self.nblk, red],
                       device=self.dev)
        nz, sb = red.numpy()
        return (f"; subnormal: {sb / max(nz, 1.0):.1e} of the nonzero coefficients "
                f"({nz / (10.0 * self.P * self.k3):.0%} nonzero)")

    def _log_arrivals(self, what, targets, hist, final, nsteps, info):
        """Per target: the step (of those logged) from which on each of ARRIVAL_SHARES of its final
        arrivals had come (the coefficients are signed: the tally can overshoot before it settles)."""
        rows = []
        logged = [n for n, _ in hist]
        every = f"~{np.diff(logged).mean():.0f}" if len(logged) > 1 else "?"
        for i, (label, _) in enumerate(targets):
            fin = final[i]
            by = []
            for q in ARRIVAL_SHARES:                # the logged step from which on they stay at the share
                below = [j for j, (_, a) in enumerate(hist) if not a[i] >= q * fin]
                by.append(hist[below[-1] + 1][0] if below and below[-1] + 1 < len(hist) else
                          (hist[0][0] if not below else nsteps))
            rows.append(dict(target=label, arrived=fin, steps_by_share=dict(zip(map(str, ARRIVAL_SHARES), by)),
                             logged_at=logged))
            log.info("  %s: %.4g arrived in %s; %s of them by step %s (of %d; logged every %s steps)", what, fin,
                     label, "/".join(f"{100 * q:g}%" for q in ARRIVAL_SHARES), "/".join(str(b) for b in by), nsteps,
                     every)
        info["arrivals"] = rows

    def _forward_from(self, x, ex, alive, what, t0, keep_sum=True, vis=None, targets=None, conn=None, vn=None):
        """The stepping of forward and forward_from.  ``targets``: (label, (V,) weights) pairs whose
        arrivals (walkers ending there, weighted) are logged on each progress line and, at the end,
        the steps by which ARRIVAL_SHARES of them had come.  ``conn`` (g, until, prune): the
        walkers' importance (a state, the backward of where they may end, known first): their
        expected arrivals x . g are tallied each step; until "ends": the solve stops when those still
        to come (in absolute value, bin by bin) are below tol x those expected at the start; prune
        > 0: a bin's walkers are dropped when theirs are below prune x those expected.  ``vn``
        (P, kb^3) float32, zeroed: each bin's walkers summed over the steps in norm (|x| per step), the
        first state's too."""
        wp, K, dev, cfg = self.wp, self.K, self.dev, self.cfg
        targets = [(lab, np.asarray(tv, np.float64).ravel()) for lab, tv in (targets or ()) if np.any(tv)]
        hist = []
        C = self._state() if keep_sum else None
        red = wp.zeros(1, dtype=wp.float64, device=dev)
        if keep_sum:
            wp.launch(K.accum, dim=self.nblk, inputs=[x, C, self.k3, self.P * self.k3, self.nblk, 0, red], device=dev)
        else:
            wp.launch(K.mass, dim=self.nblk, inputs=[x, self.k3, self.P * self.k3, self.nblk, 0, red], device=dev)
            self._exit_tally(x, alive, ex)
            if vis is not None:
                self._visit(x, vis)
        if vn is not None:
            wp.launch(K.norm_accum, dim=(self.P, self.k3), inputs=[x, vn], device=dev)
        m1 = float(red.numpy()[0])
        info = dict(steps=1, converged=True, launched_mass=m1)
        imp = None
        if conn is not None:
            g, until, prune = conn
            cred = wp.zeros(4, dtype=wp.float64, device=dev)
            self._connect(x, g, 0.0, cred)
            e0 = float(cred.numpy()[0])
            imp = dict(g=g, until=until, prune=float(prune), expected=e0, thr=float(prune) * abs(e0), cred=cred,
                       to_come=e0, to_come_abs=float(cred.numpy()[1]), dropped=0.0, dropped_abs=0.0)
            info["importance"] = dict(until=until, prune=float(prune), expected_arrivals=e0)
        if not m1 > 0:
            exits = self._exits(ex)
            if exits.sum() <= 0:
                log.info("  %s: no walkers launched", what)
                return None
            return C, exits, 1, info
        if imp is not None and imp["until"] == "ends":
            log.info("  %s: %.4g walkers in the white matter after their first step (%s), %.4g arrivals expected; "
                     "stepping until fewer than %g of those are to come%s", what, m1, _secs(time.perf_counter() - t0),
                     imp["expected"], cfg.tol, f", dropping a bin's walkers below {imp['prune']:g} of them"
                     if imp["prune"] else "")
        else:
            log.info("  %s: %.4g walkers in the white matter after their first step (%s); stepping until fewer than "
                     "%g of them are left%s", what, m1, _secs(time.perf_counter() - t0), cfg.tol,
                     f", dropping a bin's walkers below {imp['prune']:g} of the {imp['expected']:.4g} expected arrivals"
                     if imp is not None and imp["prune"] else "")
        movie = self.movie
        if movie is not None:
            movie.begin("forward", what)
            movie.add(1, self._picture(x, False))
        graphs = {0: None, 1: None}
        left, n = 1.0, 1
        by_ends = imp is not None and imp["until"] == "ends"
        prog = _Progress(what, self.log_every, tol=cfg.tol,
                         **(dict(label="arrivals to come {:.1e} of those expected") if by_ends else {}))
        cconn = None if imp is None else (imp["g"], imp["thr"], imp["cred"])

        cur = [x]                                     # (the state, for the progress lines; let go with x)

        def extra(step):
            txt = ""
            if targets:
                arr = self._arrived(ex, C, alive, targets)
                hist.append((step, arr))
                txt += "; arrived " + ", ".join(f"{lab} {a:.4g}" for (lab, _), a in zip(targets, arr))
            if imp is not None:
                e = abs(imp["expected"]) or 1.0
                txt += (f"; walkers left {left_mass[0]:.1e}" if by_ends else
                        f"; arrivals to come {imp['to_come_abs'] / e:.1e} of those expected")
                if imp["prune"]:
                    txt += f", dropped {imp['dropped_abs'] / e:.1e} of them"
            if self.verbose:
                txt += self._subnormal_text(cur[0])
            return txt

        left_mass = [1.0]

        while True:
            if n >= cfg.max_steps:
                info["converged"] = False
                break
            if self.verbose and n == 2:                  # once per solve: the time per kernel
                prof = {}
                self._fwd_step(x, x, C, alive, red, prof, None if keep_sum else ex, vis, cconn, vn)
                self._profile_line(what, prof)
            else:
                self._replay(graphs, 0, self._fwd_step, x, x, C, alive, red, None, None if keep_sum else ex, vis,
                             cconn, vn)
            n += 1
            left = left_mass[0] = float(red.numpy()[0]) / m1
            if imp is not None:
                kept, kept_abs, dr, dr_abs = (float(v) for v in imp["cred"].numpy())
                imp.update(to_come=kept, to_come_abs=kept_abs, dropped=imp["dropped"] + dr,
                           dropped_abs=imp["dropped_abs"] + dr_abs)
                if by_ends:
                    left = kept_abs / (abs(imp["expected"]) or 1.0)
            if not np.isfinite(left):
                raise RuntimeError(f"{what}: the walkers' mass is not finite (step {n})")
            prog.step(n, left, extra)
            if movie is not None and movie.want(n):
                movie.add(n, self._picture(x, False), f", walkers left {left:.1e}")
            if left < cfg.tol:
                break
        if movie is not None:
            movie.end()
        del x, cur[:]
        if keep_sum:
            self._exit_tally(C, alive, ex)
        exits = self._exits(ex)
        if targets:
            final = [float(exits @ tv) for _, tv in targets]
            self._log_arrivals(what, targets, hist + [(n, final)], final, n, info)
        if imp is not None:
            e = abs(imp["expected"]) or 1.0
            info["importance"].update(to_come=imp["to_come"], to_come_abs=imp["to_come_abs"], dropped=imp["dropped"],
                                      dropped_abs=imp["dropped_abs"], walkers_left=left_mass[0])
            log.info("  %s: of the %.4g arrivals expected, %.1e still to come at the last step%s", what,
                     imp["expected"], imp["to_come_abs"] / e,
                     f", {imp['dropped_abs'] / e:.1e} dropped with their walkers ({imp['dropped'] / e:.1e} signed)"
                     if imp["prune"] else "")
        sec = time.perf_counter() - t0
        info.update(steps=n, remaining=left, seconds=round(sec, 2), ms_per_step=round(1e3 * sec / n, 2))
        log.info("  %s: %d steps in %s (%.1f ms/step); %.3g walkers left the white matter", what, n, _secs(sec),
                 1e3 * sec / n, exits.sum())
        if not info["converged"]:
            log.warning("%s: %.1e of the walkers still in the white matter after %d steps (--galerkin-max-steps); "
                        "they may circulate where they rarely leave", what, left, n)
        return C, exits, n, info

    def backward(self, alive, tgrid, nsteps, what="backward", prune=None):
        """The importance of each state (P, kb^3): what a walker there is worth at the end
        (``tgrid``: (X, Y, Z) float32 on the device, per voxel it may leave into) within
        ``nsteps`` steps -- the forward's transpose.  ``prune`` (vn, share, expected): each step drops
        a bin's importance where |g| x vn (P, kb^3: a forward's visits there, summed over its steps in
        norm) is below share x expected (that forward's arrivals here); what it was worth to that
        forward is at most the sum of those products (logged; info["prune"]).  Returns (g, info)."""
        t0 = time.perf_counter()
        g = self._state()                    # in place: the turn reads it into the landings first
        graphs = {0: None, 1: None}
        pr = None
        if prune is not None:
            vn, share, expected = prune
            pr = (vn, float(share) * float(expected), self.wp.zeros(1, dtype=self.wp.float64, device=self.dev))
        log.info("  %s: %d steps (as many as the forward took)%s", what, nsteps,
                 f", dropping a bin's importance below {prune[1]:g} of the {prune[2]:.4g} arrivals" if pr else "")
        prog = _Progress(what, self.log_every, total=nsteps)
        movie = self.movie
        if movie is not None:
            movie.begin("backward", what, total=nsteps)
        for it in range(nsteps):
            if self.verbose and it == 1:
                prof = {}
                self._bwd_step(g, g, alive, tgrid, prof, pr)
                self._profile_line(what, prof)
            else:
                self._replay(graphs, 0, self._bwd_step, g, g, alive, tgrid, None, pr)
            if it % 16 == 15 or self.verbose:
                self.wp.synchronize_device(self.dev)       # so that the progress lines tell the truth
                prog.step(it + 1, extra=(lambda _n: self._subnormal_text(g)) if self.verbose else None)
            if movie is not None and movie.want(it + 1):
                movie.add(it + 1, self._picture(g, True))
        self.wp.synchronize_device(self.dev)
        if movie is not None:
            movie.end()
        sec = time.perf_counter() - t0
        info = dict(steps=nsteps, seconds=round(sec, 2), ms_per_step=round(1e3 * sec / max(nsteps, 1), 2))
        if pr is not None:
            dropped = float(pr[2].numpy()[0])
            info["prune"] = dict(share=float(prune[1]), expected_arrivals=float(prune[2]), dropped_bound=dropped)
            log.info("  %s: importance dropped worth at most %.1e of the %.4g arrivals", what,
                     dropped / max(abs(float(prune[2])), 1e-300), float(prune[2]))
        log.info("  %s: %d steps in %s (%.1f ms/step)", what, nsteps, _secs(sec), 1e3 * sec / max(nsteps, 1))
        return g, info

    def _finish(self, alive, C, g, tgrid, bval, sign):
        """From a forward (C) and its backward (g): per launch voxel, its walkers' worth (V,);
        and sign x the bundle density of the walkers into self.rho."""
        wp, K, dev = self.wp, self.K, self.dev
        t0 = time.perf_counter()
        reach = wp.zeros(self.shape, dtype=wp.float64, device=dev)
        self._bwd_turn(g, alive, tgrid)
        self._launch(K.REACH, alive, tgrid=tgrid, outv=reach)
        self._launch(K.DENSITY, alive, bval=bval, tgrid=tgrid, sign=sign, outv=reach, rho=self.rho)
        wp.launch(K.density, dim=(self.P, self.k3),
                  inputs=[self.d_edir, self.d_evox, self.d_vijk, self.d_vid, alive, tgrid, self.d_gptr, self.d_land,
                          self.d_cptr, self.d_cell, self.d_Wc, self.d_Bc, self.kb, self.d_lidx, C, self.ybuf,
                          float(sign), self.rho], device=dev)
        out = reach.numpy().ravel()
        log.info("  ends and bundle density in %s", _secs(time.perf_counter() - t0))
        return out

    def launched(self, alive):
        """Per voxel next to the white matter: the walkers launched from it.  (V,)."""
        out = self.wp.zeros(self.shape, dtype=self.wp.float64, device=self.dev)
        self._launch(self.K.LAUNCHED, alive, outv=out)
        return out.numpy().ravel()

    # -------------------------------------------------------------- seeded walkers
    def seeded(self, seeds, targets, allowed=None, weights=None, both=True, what="seeds", density=False):
        """Walkers started at points inside the white matter, as a seeded tracker starts them.

        ``seeds``: (n, 3) voxel coordinates (voxel centers at integers).  Each seed heads m with
        probability f(m) w(m) / (the sum over its allowed directions), f its bin's fODF (the turn's
        weights); ``allowed``: an (n, M) bool of the directions a seed may start in (default all
        with fODF).  ``weights``: per seed (default 1).  ``targets``: a list of (X, Y, Z) masks of
        voxels a walker may leave the white matter into.

        Returns a dict:
        ends:   (T,) the walkers expected to end in each target;
        pairs:  (T, T) if ``both``: walkers in pairs, a seed's second heading the opposite way (as a
                bidirectional streamline): [i, j] = the expected pairs whose first half ends in
                target i and second in j (a streamline joining i and j is [i, j] + [j, i]);
        ends_reverse: (T,) as ends, for walkers heading the reverse of the first ones (a bidirectional
                streamline's second half);
        seed_ends: (n, T) each seed's share of ends (rows of seeds outside the white matter: 0);
        check:  (T,) |ends from the backward solves - ends counted by the forward| / the latter;
        seeds_used, steps, info.
        One forward solve (from all the seeds: it sets how many steps the backward ones take) and
        one backward solve per target; each holds one state on the device (the forward tallies
        where its walkers leave step by step), so seeds fit where a bundle (two) may not.
        ``density``: one more forward, from both halves of every seed at once (the first heading
        and its reverse), its walkers tallied per voxel step by step (still one state):
        visits: (X, Y, Z) its expected step starts per voxel (times the step: path length; the seed
                counts once per half);
        ends_both: (T,) its walkers ending in each target, counted forward (= ends + ends_reverse
                up to tol)."""
        wp, dev, dirs = self.wp, self.dev, self.dirs
        seeds = np.asarray(seeds, np.float64).reshape(-1, 3)
        n = len(seeds)
        weights = np.ones(n) if weights is None else np.asarray(weights, np.float64)
        g = seeds + 0.5                                 # the bins' frame: voxel i spans [i, i + 1)
        vox = np.floor(g).astype(np.int64)
        ok = np.all((vox >= 0) & (vox < np.asarray(self.shape)), axis=1)
        vid = self.d_vid.numpy()
        dv = np.full(n, -1, np.int64)
        dv[ok] = vid[tuple(vox[ok].T)]
        use = dv >= 0
        kb = self.kb
        gb = np.floor(g * kb).astype(np.int64)
        bsub = gb - vox * kb
        sbin = (bsub[:, 0] * kb + bsub[:, 1]) * kb + bsub[:, 2]
        phi = basis(g * kb - gb)                        # (n, 10): a point seed's coefficients per unit weight
        eidx = self.d_eidx.numpy()                       # (M, Nv)
        efw = self.d_efw.numpy()                         # (P, 1 or kb^3)
        rev = dirs.rev
        fpt = None
        if self.points:                                  # points: the fODF at the seed itself (as DIPY)
            fpt = np.zeros((n, dirs.M), np.float32)
            fpt[use] = point_fodf(*self._raw, seeds[use], dirs) * dirs.w.astype(np.float32)[None, :]
        lost_mass = 0.0
        sid, pst, pre, prob = [], [], [], []           # per seed state: seed, its state, its reverse's, probability
        for k in np.flatnonzero(use):
            ps = eidx[:, dv[k]]
            m = np.flatnonzero(ps >= 0)
            if allowed is not None:
                m = m[np.asarray(allowed[k], bool)[m]]
            if fpt is not None:          # directions with fODF at the seed but no state (none at the bin's points)
                al = np.ones(dirs.M, bool) if allowed is None else np.asarray(allowed[k], bool)
                tot = float(fpt[k][al].sum())
                if tot > 0:
                    lost_mass += weights[k] * (1.0 - float(fpt[k][m].sum()) / tot)
            if not len(m):
                use[k] = False
                continue
            fw = (fpt[k, m] if fpt is not None else efw[ps[m], sbin[k] * self.ib]).astype(np.float64)
            if not fw.sum() > 0:
                use[k] = False
                continue
            sid.append(np.full(len(m), k))
            pst.append(ps[m])
            pre.append(eidx[rev[m], dv[k]])
            prob.append(weights[k] * fw / fw.sum())
        if not sid:
            raise ValueError(f"{what}: no seed in the white matter with a direction to start in")
        sid, pst, pre, prob = map(np.concatenate, (sid, pst, pre, prob))
        sb = sbin[sid]
        log.info("%s: %d of %d seeds in the white matter, %d starting states; %d target(s)", what, int(use.sum()), n,
                 len(sid), len(targets))
        x0 = np.zeros((self.P, self.k3, 10), np.float32)
        np.add.at(x0, (pst, sb), (prob[:, None] * phi[sid]).astype(np.float32))
        alive = self._alive(np.zeros(self.V, bool))
        self.ybuf = self._vec10(self.Lt)
        try:
            xd = wp.array(x0, dtype=self.K.vec10, device=dev)
            del x0
            fw_ = self.forward_from(xd, alive, what + " forward", keep_sum=False)
            if fw_ is None:
                raise ValueError(f"{what}: no walkers")
            _, exits, nsteps, inf_f = fw_
            del xd
            T = len(targets)
            vals = np.zeros((T, len(sid)))
            rvals = np.zeros((T, len(sid)))
            fwd_ends, info_b = np.zeros(T), []
            for t, tm in enumerate(targets):
                tm = np.asarray(tm, np.float32)
                fwd_ends[t] = float((exits * tm.ravel()).sum())
                gg, ib_ = self.backward(alive, self._grid(tm), nsteps, f"{what} backward {t + 1}/{T}")
                gh = gg.numpy().reshape(self.P, self.k3, 10)
                del gg
                vals[t] = np.einsum("ij,ij->i", gh[pst, sb].astype(np.float64), phi[sid])
                have = pre >= 0
                rvals[t, have] = np.einsum("ij,ij->i", gh[pre[have], sb[have]].astype(np.float64), phi[sid[have]])
                del gh
                info_b.append(ib_)
            visits = ends_both = None
            if density:
                xb = np.zeros((self.P, self.k3, 10), np.float32)
                np.add.at(xb, (pst, sb), (prob[:, None] * phi[sid]).astype(np.float32))
                have = pre >= 0
                np.add.at(xb, (pre[have], sb[have]), (prob[have, None] * phi[sid[have]]).astype(np.float32))
                xd = wp.array(xb, dtype=self.K.vec10, device=dev)
                del xb
                fw2 = self.forward_from(xd, alive, what + " forward, both halves", keep_sum=False, visits=True)
                if fw2 is not None:
                    _, exits2, _, inf_d = fw2
                    visits = inf_d.pop("visits", None)
                    ends_both = np.array([float((exits2 * np.asarray(tm, np.float32).ravel()).sum())
                                          for tm in targets])
                del xd
        finally:
            self.ybuf = None
        ends = vals @ prob
        seed_ends = np.zeros((n, len(targets)))
        np.add.at(seed_ends, sid, (vals * prob).T)
        out = dict(ends=ends, ends_reverse=rvals @ prob, seed_ends=seed_ends, seeds_used=int(use.sum()),
                   steps=int(nsteps),
                   check=np.abs(ends - fwd_ends) / np.maximum(fwd_ends, 1e-300),
                   info=dict(forward=inf_f, backward=info_b, seed_mass_without_state=lost_mass))
        if both:
            out["pairs"] = (vals * prob) @ rvals.T
        if density:
            out["visits"], out["ends_both"] = visits, ends_both
        log.info("%s: expected ends %s; backward vs forward %.1e", what, np.array2string(ends, precision=4),
                 float(out["check"].max()))
        return out

    # -------------------------------------------------------------- bundles
    def bundle(self, spec):
        """One bundle: ``spec`` as mpat.afq.bundle_specs gives it (name; include, a list of
        masks; exclude, start, end: masks or None).  Returns a TransportResult."""
        wp, dev, dirs, V = self.wp, self.dev, self.dirs, self.V
        t0 = time.time()
        name = spec["name"]

        def flat(m):
            return None if m is None else np.asarray(m, bool).ravel()

        exc = flat(spec.get("exclude"))
        exc = np.zeros(V, bool) if exc is None else exc
        inc = [flat(m) for m in spec.get("include", ())]
        a_valid = self.valid & ~exc
        kinds, coup, rule = _endpoint_kinds(a_valid, flat(spec.get("start")), flat(spec.get("end")))
        nK = len(kinds)
        subsets = [T for r in range(len(inc) + 1) for T in combinations(range(len(inc)), r)]
        log.info("%s: %s; %d include ROI(s) -> %d avoid set(s) x %d end kind(s) x (forward + backward)",
                 name, rule, len(inc), len(subsets), nK)
        self.ybuf = self._vec10(self.Lt)
        self.rho = wp.zeros((dirs.M, self.Nv), dtype=wp.float32, device=dev)
        try:
            launched = self.launched(self._alive(exc))
            ends, good = np.zeros(V), np.zeros(V)
            check, solves = [], []

            def avoid_of(T):
                avoid = exc.copy()
                for i in T:
                    avoid |= inc[i]
                return avoid

            plan = sum(int((k & ~avoid_of(T)).any()) for T in subsets for k in kinds)
            log.info("%s: %d forward + backward pair(s) to solve on %s%s", name, plan, dev,
                     f" ({_settled_free(dev) / 1e9:.1f} of {float(dev.total_memory) / 1e9:.1f} GB free)"
                     if dev.is_cuda else "")
            done = 0
            for T in subsets:
                sign = (-1.0) ** len(T)
                avoid = avoid_of(T)
                alive = self._alive(avoid)
                kb = np.stack([k & ~avoid for k in kinds])
                tag = f" avoiding include {', '.join(str(i + 1) for i in T)}" if T else ""
                for k in range(nK):
                    if not kb[k].any():
                        continue
                    done += 1
                    what = f"[{done}/{plan}] {name}{tag}, end kind {k + 1}/{nK}"
                    bval = self._grid(kb[k])
                    tv = (coup[k][:, None] * kb).sum(0)          # what ending in each voxel is worth
                    fw = self.forward(alive, bval, what + " forward", targets=[("partner ROIs", tv)])
                    if fw is None:
                        continue
                    C, leave, nst, inf_f = fw
                    del fw                                # C alone holds the walkers: freed after its tallies
                    reach, inf_b = np.zeros(V), None
                    if tv.any():
                        tgrid = self._grid(tv)
                        g, inf_b = self.backward(alive, tgrid, nst, what + " backward")
                        reach = self._finish(alive, C, g, tgrid, bval, sign)
                        del g
                    del C
                    from_k = kb[k] * reach            # walkers launched in kind k that end in a partner ...
                    into = tv * leave                  # ... counted where they end
                    ends += sign * (from_k + into)
                    good += sign * from_k
                    a, b = from_k.sum(), into.sum()
                    check.append(abs(a - b) / max(a, b, 1e-300))
                    solves.append(dict(avoid=[i + 1 for i in T], kind=k + 1, forward=inf_f, backward=inf_b))
            rho_h = self.rho.numpy().T
        finally:
            self.ybuf = self.rho = None
        dens = rho_h.sum(1, dtype=np.float64)
        expected = float(good.sum())
        density = np.zeros(V, np.float32)
        density[self.lin] = np.clip(dens, 0, None)
        odf_ijk, odf_coef = visitation_odf(dirs, self.ijk, rho_h, dens)
        del rho_h
        seed = np.divide(good, launched, out=np.zeros(V), where=launched > 0)
        info = dict(model=f"fixed-step walkers, Lagrange-Galerkin (mpat.galerkin): quadratic densities in "
                          f"{self.kb}^3 bins per voxel",
                    config=self.cfg.key(), endpoint_rule=rule, step_mm=self.smm, bins_per_voxel_side=self.kb,
                    white_matter_voxels=self.Nv, expected_bundle_streamlines=expected,
                    launched_walkers=float((launched * a_valid).sum()),
                    endpoint_mass_check=float(ends.sum() / max(2 * expected, 1e-300)),
                    transport_identity=float(max(check) if check else 0.0),
                    n_solves=int(sum(1 + (s["backward"] is not None) for s in solves)), n_columns=nK,
                    sweeps=int(sum(s["forward"]["steps"] + (s["backward"] or {}).get("steps", 0) for s in solves)),
                    solves=solves, tolerance=self.cfg.tol,
                    negative_mass_before_clip=dict(density=float(-dens[dens < 0].sum()),
                                                   endpoints=float(-ends[ends < 0].sum())),
                    seconds=round(time.time() - t0, 1))
        log.info("%s: %.3g expected walkers in the bundle (of %.3g launched); endpoint check %.4f, transport identity "
                 "%.1e; %d steps in %.0fs", name, expected, info["launched_walkers"], info["endpoint_mass_check"],
                 info["transport_identity"], info["sweeps"], info["seconds"])
        shape = self.shape
        return TransportResult(name, density.reshape(shape), np.clip(ends, 0, None).astype(np.float32).reshape(shape),
                               np.clip(seed, 0, 1).astype(np.float32).reshape(shape), odf_ijk, odf_coef, info)

    # -------------------------------------------------------------- bundles from seeds (pyAFQ's seeding)
    def _starts(self, seeds, sidx, peak, chunk=200000):
        """The first headings of seeded streamlines, as pyAFQ's GPU tracker starts them: streamline i
        from seed ``sidx[i]`` along peak direction ``peak[i]`` (an index of the directions); its half
        2i heads within the cone of the peak, its half 2i + 1 within the cone of the reverse, each
        drawing its direction as a turn does there (f w over the cone's directions; with points, f at
        the seed itself).  The halves draw independently.  Returns a dict: per half h, ``hseed`` its
        seed and ``hlin`` its seed's voxel (flat index; -1 outside the grid); per (half, direction)
        entry ``eh`` its half, ``inv`` its (state, bin) as an index of ``uniq`` (state x kb^3 + bin)
        and ``prob``; ``phi`` the seeds' coefficients (n, 10); ``lost`` the seeds' heading mass with
        fODF at the seed but no state."""
        dirs, kb, k3 = self.dirs, self.kb, self.k3
        seeds = np.asarray(seeds, np.float64).reshape(-1, 3)
        n = len(seeds)
        g = seeds + 0.5                                 # the bins' frame: voxel i spans [i, i + 1)
        vox = np.floor(g).astype(np.int64)
        inb = np.all((vox >= 0) & (vox < np.asarray(self.shape)), axis=1)
        vid = self.d_vid.numpy()
        dv = np.full(n, -1, np.int64)
        dv[inb] = vid[tuple(vox[inb].T)]
        slin = np.full(n, -1, np.int64)
        slin[inb] = np.ravel_multi_index(tuple(vox[inb].T), self.shape)
        gb = np.floor(g * kb).astype(np.int64)
        bsub = gb - vox * kb
        sbin = (bsub[:, 0] * kb + bsub[:, 1]) * kb + bsub[:, 2]
        phi = basis(g * kb - gb)
        sidx, peak = np.asarray(sidx, np.int64), np.asarray(peak, np.int64)
        hseed = np.repeat(sidx, 2)
        hcen = np.stack([peak, dirs.rev[peak]], 1).ravel()
        nh = len(hseed)
        eidx = self.d_eidx.numpy()
        efw = None if self.points else self.d_efw.numpy()
        cptr, cidx = dirs.cptr.astype(np.int64), dirs.cidx.astype(np.int64)
        cnt = cptr[1:] - cptr[:-1]
        w = dirs.w.astype(np.float32)
        lost = 0.0
        EH, LIN, PR = [], [], []
        for a in range(0, nh, chunk):
            h = np.arange(a, min(a + chunk, nh))
            h = h[dv[hseed[h]] >= 0]
            if not len(h):
                continue
            k = cnt[hcen[h]]
            eh = np.repeat(h, k)
            pos = np.arange(int(k.sum())) - np.repeat(np.cumsum(k) - k, k)
            m = cidx[np.repeat(cptr[hcen[h]], k) + pos]
            s = hseed[eh]
            st = eidx[m, dv[s]].astype(np.int64)
            if self.points:                          # the fODF at the seed itself (as DIPY / cuslines)
                us, row = np.unique(s, return_inverse=True)
                F = point_fodf(*self._raw, np.asarray(seeds)[us], dirs)
                fall = (F[row, m] * w[m]).astype(np.float64)
                fw = np.where(st >= 0, fall, 0.0)
                hl = eh - a
                tall = np.bincount(hl, fall, minlength=chunk)
                tin = np.bincount(hl, fw, minlength=chunk)
                have = tall > 0
                lost += float((1.0 - tin[have] / tall[have]).sum())
            else:
                fw = np.zeros(len(st))
                ok = st >= 0
                fw[ok] = efw[st[ok], sbin[s[ok]] * self.ib]
            keep = fw > 0
            eh, st, s, fw = eh[keep], st[keep], s[keep], fw[keep]
            tot = np.bincount(eh - a, fw, minlength=chunk)
            EH.append(eh.astype(np.int32))
            LIN.append(st * k3 + sbin[s])
            PR.append((fw / tot[eh - a]).astype(np.float32))
        eh = np.concatenate(EH) if EH else np.zeros(0, np.int32)
        lin = np.concatenate(LIN) if LIN else np.zeros(0, np.int64)
        prob = np.concatenate(PR) if PR else np.zeros(0, np.float32)
        del EH, LIN, PR
        uniq, inv = np.unique(lin, return_inverse=True)
        used = np.zeros(nh, bool)
        used[eh] = True
        return dict(n=n, nh=nh, hseed=hseed, hlin=slin[hseed], eh=eh, inv=inv.astype(np.int32), uniq=uniq,
                    prob=prob, phi=phi, lost=lost, used=used, in_wm=int((dv >= 0).sum()))

    @staticmethod
    def _chunks(n, size=1 << 22):
        return ((a, min(a + size, n)) for a in range(0, n, size))

    def _start_values(self, st, gh):
        """What each half of ``_starts`` is worth under the backward solve ``gh`` (host, (P, kb^3, 10)):
        sum over its directions of prob x the state's importance at the seed.  (nh,)."""
        Gu = gh.reshape(-1, 10)[st["uniq"]].astype(np.float64)
        out = np.zeros(st["nh"])
        eh, inv, prob, phi, hseed = st["eh"], st["inv"], st["prob"], st["phi"], st["hseed"]
        for a, b in self._chunks(len(eh)):
            val = prob[a:b] * np.einsum("ij,ij->i", Gu[inv[a:b]], phi[hseed[eh[a:b]]])
            out += np.bincount(eh[a:b], val, minlength=st["nh"])
        return out

    def _start_state(self, st, weight):
        """The walkers of ``_starts``' halves, half h weighted by ``weight[h]``, as a state on the device."""
        nu = len(st["uniq"])
        acc = np.zeros((nu, 10))
        eh, inv, prob, phi, hseed = st["eh"], st["inv"], st["prob"], st["phi"], st["hseed"]
        for a, b in self._chunks(len(eh)):
            c = (np.asarray(weight, np.float64)[eh[a:b]] * prob[a:b])[:, None] * phi[hseed[eh[a:b]]]
            for j in range(10):
                acc[:, j] += np.bincount(inv[a:b], c[:, j], minlength=nu)
        x0 = np.zeros((self.P * self.k3, 10), np.float32)
        x0[st["uniq"]] = acc
        del acc
        return self.wp.array(x0.reshape(self.P, self.k3, 10), dtype=self.K.vec10, device=self.dev)

    def seeded_bundle(self, spec, seeds, starts):
        """One bundle from streamlines seeded as pyAFQ seeds them (its GPU tracker, cuslines): from
        each seed (``seeds``, (n, 3) voxel coordinates, centers at integers) one streamline per peak
        of the fODF there (``starts``: (seed index, peak direction index) per streamline, as
        seed_peaks gives them), its two halves heading within the cone of the peak and of its
        reverse, drawn independently.  The bundle is the streamlines whose two ends meet ``spec``'s
        end rule (as bundle's: start / end ROIs; include ROIs by inclusion-exclusion over avoid sets;
        a streamline through an excluded voxel, its seed's included, is out).  Returns a
        TransportResult: density and ends as bundle's, of the expected streamlines in the bundle;
        seed_prob per seed voxel, the share of its streamlines in the bundle.

        Solves: one forward from every half (it sets the backward solves' steps), then per avoid set
        a backward per end kind (each half's chance to end in it) and per end kind a forward from the
        halves weighted by their partner's chance to end in a matching kind, its density with the
        backward of what ending in each voxel is worth (the halves draw independently, so a pair's
        chance is the product)."""
        wp, dev, dirs, V = self.wp, self.dev, self.dirs, self.V
        t0 = time.time()
        name = spec["name"]

        def flat(m):
            return None if m is None else np.asarray(m, bool).ravel()

        exc = flat(spec.get("exclude"))
        exc = np.zeros(V, bool) if exc is None else exc
        inc = [flat(m) for m in spec.get("include", ())]
        a_valid = self.valid & ~exc
        kinds, coup, rule = _endpoint_kinds(a_valid, flat(spec.get("start")), flat(spec.get("end")))
        nK = len(kinds)
        subsets = [T for r in range(len(inc) + 1) for T in combinations(range(len(inc)), r)]
        sidx, peak = starts
        st = self._starts(seeds, sidx, peak)
        nh, ns = st["nh"], st["nh"] // 2
        hlin = st["hlin"]
        log.info("%s (seeded): %s; %d of %d seeds in this white matter, %d streamlines (%d halves with a heading, "
                 "%d entries); %d include ROI(s) -> %d avoid set(s) x %d end kind(s)", name, rule, st["in_wm"], st["n"],
                 ns, int(st["used"].sum()), len(st["eh"]), len(inc), len(subsets), nK)

        def out_of(avoid):            # halves whose seed is in ``avoid`` (or off the grid): none of theirs count
            o = hlin < 0
            o[hlin >= 0] = avoid[hlin[hlin >= 0]]
            return o

        by_imp = self.cfg.until != "walkers" or self.cfg.prune > 0     # the partner forwards steered by importance
        state_bytes = 40.0 * self.P * self.k3
        ends, good_seed = np.zeros(V), np.zeros(V)
        count_seed = np.zeros(V)
        np.add.at(count_seed, hlin[0::2][hlin[0::2] >= 0], 1.0)
        expected, check, solves = 0.0, [], []
        self.ybuf = self._vec10(self.Lt)
        self.rho = wp.zeros((dirs.M, self.Nv), dtype=wp.float32, device=dev)
        rho_h = None
        try:
            keep0 = (~out_of(exc) & st["used"]).astype(np.float64)
            alive0 = self._alive(exc)
            fw0 = None
            vn = None                    # prune: the all-halves walkers' visits per bin (in norm), for the backwards
            if keep0.any():
                x = self._start_state(st, keep0)
                if self.cfg.prune > 0:
                    vn = wp.zeros((self.P, self.k3), dtype=wp.float32, device=dev)
                fw0 = self.forward_from(x, alive0, f"{name} seeds forward (all halves)", keep_sum=False,
                                        targets=[(f"end kind {k + 1}", kinds[k] & ~exc) for k in range(nK)], vn=vn)
                del x
            if fw0 is not None:
                _, exits0, nsteps, inf0 = fw0
                arr0 = [float((exits0 * (kinds[k] & ~exc)).sum()) for k in range(nK)]    # each end kind's arrivals
                solves.append(dict(avoid=[], kind=0, forward=inf0, backward=None))
                plan = sum(int((k & ~self._avoid_of(exc, inc, T)).any()) for T in subsets for k in kinds)
                done = 0
                for T in subsets:
                    sign = (-1.0) ** len(T)
                    avoid = self._avoid_of(exc, inc, T)
                    alive = self._alive(avoid)
                    keep = (~out_of(avoid)).astype(np.float64)
                    kb_ = np.stack([k & ~avoid for k in kinds])
                    tag = f" avoiding include {', '.join(str(i + 1) for i in T)}" if T else ""
                    Vk = np.zeros((nK, nh))
                    gh = [None] * nK
                    for k in range(nK):
                        if not kb_[k].any():
                            continue
                        done += 1
                        g, inf_b = self.backward(alive, self._grid(kb_[k]), nsteps,
                                                 f"[{done}/{plan}] {name}{tag}, end kind {k + 1}/{nK} backward",
                                                 prune=(vn, self.cfg.prune, arr0[k])
                                                 if vn is not None and arr0[k] > 0 else None)
                        gh[k] = g.numpy().reshape(self.P, self.k3, 10)
                        del g
                        Vk[k] = self._start_values(st, gh[k]) * keep
                        solves.append(dict(avoid=[i + 1 for i in T], kind=k + 1, forward=None, backward=inf_b))
                        if not T:            # every half counted where it ends (forward) and at its start (backward)
                            a, b = Vk[k] @ keep0, float((exits0 * kb_[k]).sum())
                            check.append(abs(a - b) / max(a, b, 1e-300))
                    VA, VB = Vk[:, 0::2], Vk[:, 1::2]
                    pair = np.einsum("ki,kl,li->i", VA, coup, VB)        # each streamline's chance, both ends
                    expected += sign * float(pair.sum())
                    np.add.at(good_seed, hlin[0::2][hlin[0::2] >= 0], sign * pair[hlin[0::2] >= 0])
                    for col in range(nK):
                        tv = (coup[:, col][:, None] * kb_).sum(0)            # what a half's end is worth ...
                        if not tv.any():
                            continue
                        W = np.empty(nh)                                   # ... given its partner ends in col
                        W[0::2], W[1::2] = VB[col], VA[col]
                        W *= keep
                        if not np.any(W != 0):
                            continue
                        gl = None                                          # its backward: sum of the kinds'
                        for k in range(nK):
                            if coup[k, col] and gh[k] is not None:
                                part = gh[k] * np.float32(coup[k, col])
                                gl = part if gl is None else gl + part
                        g = None
                        steer = by_imp
                        if steer and dev.is_cuda and _settled_free(dev) < 1.1 * state_bytes + 2 * state_bytes:
                            steer = False                 # no room for the importance beside the walkers and their sum
                            log.warning("%s: too little GPU memory to steer this forward by its importance (until / "
                                        "prune): it runs as with until=walkers, prune=0", name)
                        if steer:                         # on the device before the forward: it steers it
                            g = self._state()
                            g.f.assign(np.ascontiguousarray(gl, np.float32).reshape(self.P, self.k3 * 10))
                        x = self._start_state(st, W)
                        fw = self.forward_from(x, alive, f"{name}{tag}, halves whose partner ends in kind "
                                                         f"{col + 1}/{nK} forward", targets=[("their own ends", tv)],
                                               conn=(g, self.cfg.until, self.cfg.prune) if steer else None)
                        del x
                        if fw is None:
                            continue
                        C, exits, _, inf_f = fw
                        del fw
                        if g is None:
                            g = self._state()
                            g.f.assign(np.ascontiguousarray(gl, np.float32).reshape(self.P, self.k3 * 10))
                        del gl
                        tgrid = self._grid(tv)
                        t1 = time.perf_counter()
                        self._bwd_turn(g, alive, tgrid)
                        wp.launch(self.K.density, dim=(self.P, self.k3),
                                  inputs=[self.d_edir, self.d_evox, self.d_vijk, self.d_vid, alive, tgrid, self.d_gptr,
                                          self.d_land, self.d_cptr, self.d_cell, self.d_Wc, self.d_Bc, self.kb,
                                          self.d_lidx, C, self.ybuf, float(sign), self.rho], device=dev)
                        wp.synchronize_device(dev)
                        log.info("  bundle density in %s", _secs(time.perf_counter() - t1))
                        del g, C
                        into = tv * exits
                        ends += sign * into
                        Gl = (coup[:, col][:, None] * Vk).sum(0)
                        a, b = float(W @ Gl), float(into.sum())
                        check.append(abs(a - b) / max(a, b, 1e-300))
                        solves.append(dict(avoid=[i + 1 for i in T], kind=col + 1, forward=inf_f, backward=None))
                    del gh
            del vn
            rho_h = self.rho.numpy().T
        finally:
            self.ybuf = self.rho = None
        dens = rho_h.sum(1, dtype=np.float64)
        density = np.zeros(V, np.float32)
        density[self.lin] = np.clip(dens, 0, None)
        odf_ijk, odf_coef = visitation_odf(dirs, self.ijk, rho_h, dens)
        del rho_h
        seed = np.divide(good_seed, count_seed, out=np.zeros(V), where=count_seed > 0)
        info = dict(model=f"fixed-step walkers, Lagrange-Galerkin (mpat.galerkin): quadratic densities in "
                          f"{self.kb}^3 bins per voxel; streamlines seeded as pyAFQ's (one per fODF peak per seed, "
                          f"halves within the cone of the peak and its reverse)",
                    config=self.cfg.key(), endpoint_rule=rule, step_mm=self.smm, step_voxels=self.step_vox,
                    bins_per_voxel_side=self.kb, white_matter_voxels=self.Nv, expected_bundle_streamlines=expected,
                    launched_walkers=float(ns),
                    seeding=dict(seeds=int(st["n"]), seeds_in_white_matter=int(st["in_wm"]), streamlines=int(ns),
                                 halves_with_a_heading=int(st["used"].sum()),
                                 seed_mass_without_state=float(st["lost"]),
                                 seedprob="P(a streamline seeded in this voxel is in the bundle)"),
                    endpoint_mass_check=float(ends.sum() / max(2 * expected, 1e-300)),
                    transport_identity=float(max(check) if check else 0.0),
                    n_solves=int(sum((s["forward"] is not None) + (s["backward"] is not None) for s in solves)),
                    n_columns=nK,
                    sweeps=int(sum((s["forward"] or {}).get("steps", 0) + (s["backward"] or {}).get("steps", 0)
                                   for s in solves)),
                    solves=solves, tolerance=self.cfg.tol,
                    negative_mass_before_clip=dict(density=float(-dens[dens < 0].sum()),
                                                   endpoints=float(-ends[ends < 0].sum())),
                    seconds=round(time.time() - t0, 1))
        log.info("%s (seeded): %.4g expected streamlines in the bundle (of %d); endpoint check %.4f, transport identity "
                 "%.1e; %d steps in %.0fs", name, expected, ns, info["endpoint_mass_check"], info["transport_identity"],
                 info["sweeps"], info["seconds"])
        shape = self.shape
        return TransportResult(name, density.reshape(shape), np.clip(ends, 0, None).astype(np.float32).reshape(shape),
                               np.clip(seed, 0, 1).astype(np.float32).reshape(shape), odf_ijk, odf_coef, info)

    @staticmethod
    def _avoid_of(exc, inc, T):
        avoid = exc.copy()
        for i in T:
            avoid |= inc[i]
        return avoid


def visitation_odf(dirs, ijk, rho_dir, dens, sh_order=8):
    """The bundle's walkers' directions where its density reaches ODF_FLOOR x its peak: SH
    coefficients (descoteaux07, legacy, symmetric), antipodally symmetrised, in path length per
    steradian, so that they integrate to the voxel's density."""
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix

    top = dens.max() if dens.size else 0.0
    keep = np.flatnonzero(dens >= ODF_FLOOR * top) if top > 0 else np.zeros(0, np.int64)
    _, inv = sh_to_sf_matrix(Sphere(xyz=dirs.v), sh_order_max=sh_order, return_inv=True)
    g = np.clip(rho_dir[keep].astype(np.float64), 0, None) / dirs.w[None, :]
    g = 0.5 * (g + g[:, dirs.rev])
    c = g @ inv
    c00 = dens[keep] / (2 * np.sqrt(np.pi))
    c *= np.divide(c00, c[:, 0], out=np.zeros_like(c00), where=c[:, 0] > 0)[:, None]
    return ijk[keep], c.astype(np.float32)


def inward(seeds, inside, dirs, margin=0.0):
    """Which directions point into ``inside`` (a mask, e.g. the white matter) from each seed:
    those along which the distance to its edge grows (the distance transform's gradient,
    interpolated at the seed), by more than ``margin`` (a cosine).  For seeds in an end ROI that
    start toward the white matter: Galerkin.seeded(..., allowed=inward(...), both=False).
    (n, M) bool; a seed where the gradient vanishes allows every direction."""
    from scipy import ndimage as ndi

    inside = np.asarray(inside, bool)
    d = ndi.distance_transform_edt(inside) - ndi.distance_transform_edt(~inside)
    grad = np.stack(np.gradient(d), -1)
    pts = np.asarray(seeds, np.float64).reshape(-1, 3)
    gv = np.stack([ndi.map_coordinates(grad[..., k], pts.T, order=1, mode="nearest") for k in range(3)], -1)
    norm = np.linalg.norm(gv, axis=1, keepdims=True)
    gv = np.divide(gv, norm, out=np.zeros_like(gv), where=norm > 0)
    cos = gv @ np.asarray(dirs.v).T
    out = cos > margin
    out[norm[:, 0] == 0] = True
    return out


SEED_PEAK_THRESHOLD = 0.25      # pyAFQ's GPU tracker (cuslines): a seed's peaks above this x its largest ...
SEED_PEAK_SEPARATION = 45.0     # ... and at least this many degrees apart


def seeds_from_mask(mask, count, rng=0):
    """``count`` seeds in ``mask`` as DIPY's random_seeds_from_mask(seed_count_per_voxel=False), which
    pyAFQ uses for random seeds: each voxel count // voxels seeds, the remaining count % voxels in as
    many voxels drawn without replacement, each seed uniform in its voxel.  (count, 3) voxel
    coordinates (centers at integers)."""
    where = np.argwhere(np.asarray(mask, bool))
    nv, count = len(where), int(count)
    if nv == 0:
        raise ValueError("the seed mask is empty")
    if count < 1:
        raise ValueError("the seed count must be positive")
    r = np.random.default_rng(rng)
    q, extra = divmod(count, nv)
    vox = np.concatenate([np.repeat(where, q, axis=0), where[r.choice(nv, extra, replace=False)]])
    return vox + r.random(vox.shape) - 0.5


def seed_peaks(sh, seeds, sphere="symmetric362", relative=SEED_PEAK_THRESHOLD, separation=SEED_PEAK_SEPARATION,
               chunk=50000):
    """The streamlines pyAFQ's GPU tracker starts at each seed: one per peak of the fODF there (clipped
    at 0, interpolated trilinearly from the 8 voxel centers around the seed, values below
    PMF_THRESHOLD x its maximum zeroed: point_fodf), as DIPY's peak_directions finds them on
    ``sphere`` (above ``relative`` x the largest, ``separation`` degrees apart, a direction and its
    reverse one).  Returns (seed index, peak direction index) per streamline: Galerkin.seeded_bundle's
    ``starts``."""
    from dipy.core.sphere import Sphere
    from dipy.data import get_sphere
    from dipy.direction.peaks import peak_directions

    sp = sphere if isinstance(sphere, Sphere) else get_sphere(name=sphere)
    dirs = Directions(sp)
    sh = np.asarray(sh)
    seeds = np.asarray(seeds, np.float64).reshape(-1, 3)
    shape = sh.shape[:3]
    vox = np.floor(seeds + 0.5).astype(np.int64)
    ok = np.all((vox >= 0) & (vox < np.asarray(shape)), axis=1)
    sid, pk = [], []
    t0, last = time.perf_counter(), time.perf_counter()
    for a in range(0, len(seeds), chunk):
        idx = np.arange(a, min(a + chunk, len(seeds)))
        idx = idx[ok[idx]]
        if not len(idx):
            continue
        sf, rows = raw_sf(sh, np.unique(vox[idx], axis=0), shape, dirs)
        F = point_fodf(sf, rows, seeds[idx], dirs, normalise=False)
        del sf, rows
        for j, f in zip(idx, F):
            if not f.max() > 0:
                continue
            _, _, p = peak_directions(f.astype(np.float64), sp, relative_peak_threshold=relative,
                                      min_separation_angle=separation)
            sid.extend([j] * len(p))
            pk.extend(p)
        if time.perf_counter() - last > LOG_SECONDS:
            last = time.perf_counter()
            log.info("  seed peaks: %d of %d seeds (%s)", idx[-1] + 1, len(seeds), _secs(last - t0))
    sid, pk = np.asarray(sid, np.int64), np.asarray(pk, np.int64)
    log.info("seed peaks: %d streamlines from %d seeds (%.2f per seed; relative threshold %g, %g deg apart) in %s",
             len(sid), len(seeds), len(sid) / max(len(seeds), 1), relative, separation,
             _secs(time.perf_counter() - t0))
    return sid, pk


def bundle_region(density, transient, floor=1e-3, dilate=2):
    """Where a voxel solve's bundle density reaches ``floor`` x its peak, grown by ``dilate``
    voxels (26-connected), within the white matter: the region of a solve in finer bins."""
    from scipy import ndimage as ndi

    top = float(density.max())
    if not top > 0:
        return None
    reg = density >= floor * top
    if dilate > 0:
        reg = ndi.binary_dilation(reg, structure=np.ones((3, 3, 3), bool), iterations=int(dilate))
    return reg & np.asarray(transient, bool)


SHARED_DOMAIN = 0.85     # a bundle whose exclusions leave more of the white matter than this uses the whole


class Bundles:
    """Bundles with ``cfg.bins`` bins per voxel side.  Each is solved in whole voxels on the white
    matter its exclusions leave it -- for a bundle kept to one side, a hemisphere: about half the
    work -- with the same results, since a walker entering an excluded voxel is lost either way
    (bundles that exclude little share one model of all of the white matter).  With bins > 1
    it is then solved in the finer bins within its bundle's region (bundle_region).
    ``bundle(spec)`` as Galerkin's."""

    def __init__(self, sh, zooms, transient, absorb, cfg, device=None, graphs=None, verbose=False, log_every=None,
                 stop_map=None):
        self.args = (sh, zooms, np.asarray(transient, bool), absorb)
        self.cfg, self.device, self.graphs = cfg, device, graphs
        self.kw = dict(verbose=verbose, log_every=log_every, stop_map=stop_map)
        self.coarse = cfg if cfg.bins == 1 else replace(cfg, bins=1)
        self.whole = None
        self.movie = None                # a SolveMovie for the next bundle (its models record into it)
        self.seeding = None              # (seeds, starts): every bundle from these seeded streamlines
                                         # (Galerkin.seeded_bundle; seeds_from_mask, seed_peaks), not launches

    def _solve(self, model, spec):
        if self.seeding is None:
            return model.bundle(spec)
        return model.seeded_bundle(spec, *self.seeding)

    def _model(self, cfg, region=None):
        sh, zooms, transient, absorb = self.args
        return Galerkin(sh, zooms, transient, absorb, cfg, device=self.device, graphs=self.graphs, region=region,
                        **self.kw)

    def _near_ends(self, spec):
        """spec["soft_max_len"] (mm, > 0): the voxels within that distance of both its start and its
        end ROI -- every point of a streamline no longer than that, from one to the other, is (a soft
        maximum length: walkers leaving them are lost, longer paths within them still count).  None
        without it; a ValueError if the bundle has no start or no end voxel."""
        L = float(spec.get("soft_max_len") or 0.0)
        if L <= 0:
            return None
        from scipy import ndimage as ndi

        sh, zooms, transient = self.args[:3]
        shape = transient.shape
        near = None
        for key in ("start", "end"):
            m = spec.get(key)
            m = None if m is None else np.asarray(m, bool).reshape(shape)
            if m is None or not m.any():
                raise ValueError(f"{spec['name']}: a soft maximum length ({L:g} mm) needs a start and an end ROI, and "
                                 f"its {key} ROI has no voxels")
            d = ndi.distance_transform_edt(~m, sampling=np.asarray(zooms, float)) <= L
            near = d if near is None else near & d
        return near

    def _voxels(self, spec, near=None):
        """The voxel model for a bundle, and whether it is the bundle's own (to release).  ``near``:
        _near_ends' voxels, the only ones it walks in."""
        transient = self.args[2]
        exc = spec.get("exclude")
        reg = transient if exc is None else transient & ~np.asarray(exc, bool)
        n, total = int(reg.sum()), int(transient.sum())
        if near is not None:
            reg = reg & near
            log.info("%s: within %g mm of both its start and its end ROI (soft maximum length): %d of the %d "
                     "white-matter voxels%s", spec["name"], float(spec["soft_max_len"]), int(reg.sum()), n,
                     "" if exc is None else " its exclusions leave it")
            if not reg.any():
                raise ValueError(f"{spec['name']}: no white matter within {float(spec['soft_max_len']):g} mm of both "
                                 f"its start and its end ROI")
            return self._model(self.coarse, reg), True
        if exc is None or n > SHARED_DOMAIN * total or n == 0:
            if self.whole is None:
                self.whole = self._model(self.coarse)
            return self.whole, False
        log.info("%s: solved on the %d white-matter voxels its exclusions leave it (of %d)", spec["name"], n, total)
        return self._model(self.coarse, reg), True

    def bundle(self, spec):
        if self.cfg.bins > 1:
            log.info("%s: first in whole voxels, for the region of its %d^3-bin solve", spec["name"], self.cfg.bins)
        near = self._near_ends(spec)
        model, own = self._voxels(spec, near)
        model.movie = self.movie
        try:
            res = self._solve(model, spec)
        finally:
            model.movie = None
            if own:
                model.release()
        if near is not None:
            res.info["soft_max_len"] = float(spec["soft_max_len"])
        if self.cfg.bins == 1:
            return res
        transient = self.args[2]
        exc = spec.get("exclude")

        def region(floor):
            reg = bundle_region(res.density, transient, floor, self.cfg.dilate)
            if reg is not None and exc is not None:
                reg &= ~np.asarray(exc, bool)
            if reg is not None and near is not None:
                reg &= near
            return reg

        floor = self.cfg.floor
        reg = region(floor)
        if reg is None or not reg.any():
            log.warning("%s: no bundle in whole voxels; not solved in %d^3 bins", spec["name"], self.cfg.bins)
            res.info["fine"] = "skipped: no bundle in whole voxels"
            return res
        free, need = self._free(), self._fine_bytes(int(reg.sum()), model)
        if free is not None and need > 0.9 * free:
            raise MemoryError(
                f"{spec['name']}: its {self.cfg.bins}^3-bin solve over {int(reg.sum())} voxels (density >= "
                f"{floor:g} x its peak, grown by {self.cfg.dilate}) needs about {need / 1e9:.1f} GB on the GPU and "
                f"{free / 1e9:.1f} GB is free.  Fewer bins (--galerkin-bins 2: about "
                f"{self._fine_bytes(int(reg.sum()), model, 2) / 1e9:.1f} GB), a smaller region (a higher "
                f"--galerkin-floor, a smaller --galerkin-dilate) or a smaller sphere would fit")
        log.info("%s: in %d^3 bins within %d voxels (density >= %g x its peak, grown by %d)", spec["name"],
                 self.cfg.bins, int(reg.sum()), floor, self.cfg.dilate)
        fine = self._model(self.cfg, reg)
        fine.movie = self.movie
        try:
            out = self._solve(fine, spec)
        finally:
            fine.release()
        out.info["voxel_solve"] = {k: res.info[k] for k in ("expected_bundle_streamlines", "launched_walkers",
                                                            "transport_identity", "endpoint_mass_check", "sweeps",
                                                            "seconds")}
        out.info["region_voxels"] = int(reg.sum())
        out.info["seconds"] = round(out.info["seconds"] + res.info["seconds"], 1)
        return out

    def _free(self):
        """Free bytes on the device (None on the CPU: no limit)."""
        model = self.whole
        dev = None
        if model is not None:
            dev = model.dev
        else:
            dev = get_device(self.device)
        if not dev.is_cuda:
            return None
        return _settled_free(dev)

    def _fine_bytes(self, n, voxels, bins=None):
        """Device bytes a solve in bins^3 bins (cfg.bins) over n voxels needs, from the voxel model's
        shares of states and landings per (voxel, direction)."""
        M, k3 = voxels.dirs.M, (bins or self.cfg.bins) ** 3
        P, Lt = voxels.state_share * n * M, voxels.landing_share[0] * n * M
        return (16.0 * n * M + 16.0 * P + 12.0 * Lt + 2 * 40.0 * P * k3 + 40.0 * Lt * k3 + 24.0 * voxels.V + 2e8
                + (4.0 * (P + Lt) * k3 if self.cfg.interp else 0.0)
                + (8.0 * (P + Lt) * k3 * int(self.cfg.points) ** 3 if self.cfg.points else 0.0))

    def release(self):
        if self.whole is not None:
            self.whole.release()


def _seeding(seed_mask, sh_img, sh, transient, sphere, count, threshold, rng):
    """run's seed_mask: the seeds (those in the white matter) and their streamlines' starts."""
    img = seed_mask if hasattr(seed_mask, "affine") else nib.load(str(seed_mask))
    if tuple(img.shape[:3]) != tuple(sh_img.shape[:3]) or not np.allclose(img.affine, sh_img.affine, atol=1e-4):
        raise ValueError(f"the seed mask's grid ({tuple(img.shape[:3])}, affine\n{img.affine}) is not the fODF's "
                         f"({tuple(sh_img.shape[:3])}, affine\n{sh_img.affine}): resample it first")
    data = np.asarray(img.dataobj)
    if data.ndim > 3:
        data = data.reshape(data.shape[:3] + (-1,))[..., 0]
    mask = data > threshold
    seeds = seeds_from_mask(mask, count, rng)
    vox = np.floor(seeds + 0.5).astype(np.int64)
    inwm = np.asarray(transient, bool)[tuple(vox.T)]
    log.info("seeds: %d in the %d voxels of %s above %g; %d (%.1f%%) in the white matter (the rest start no "
             "streamline here)", len(seeds), int(mask.sum()), getattr(seed_mask, "get_filename", lambda: None)()
             or seed_mask, threshold, int(inwm.sum()), 100.0 * inwm.mean())
    seeds = seeds[inwm]
    return seeds, seed_peaks(sh, seeds, sphere)


# ------------------------------------------------------------------ MPAT on top of pyAFQ
def run(afq, bundles=None, bins=1, step=0.5, angle=35.0, sphere="symmetric362", tol=1e-6, max_steps=5000,
        floor=1e-3, dilate=2, interp="all", points=0, device=None, viz=True, cores=True, synthseg_thalamus=True, no_midline=(),
        out_dir=None, fornix=True, viz_colormap=None, verbose=False, log_every=None, ends_only=False, movie=None,
        seed_mask=None, seed_count=1_000_000, seed_threshold=0.0, seed_rng=0, stop="interp", stop_threshold=0.4,
        stop_grid="fodf", until="ends", prune=1e-9, soft_max_len=0.0,
        **roi_kwargs):
    """Segment pyAFQ's bundles with fixed-step walkers, solved by Lagrange-Galerkin (module
    docstring), on a GPU with Warp.

    afq         ``ParticipantAFQ`` or ``GroupAFQ`` (every participant in turn)
    bundles     pyAFQ bundle names (default: all in the bundle dictionary)
    bins        bins per voxel side: 1, 3, 5, ... (> 1: a voxel solve first, then the finer bins
                within its bundle's region: ``floor``, ``dilate``)
    step        step length: mm (a number, "0.5" or "0.5mm"; default 0.5, as pyAFQ's step_size) or a
                fraction of the smallest voxel side ("0.5vox"); outside 0.15-1 voxel it warns
    angle       largest turn per step, degrees
    sphere      the walkers' directions: a DIPY sphere holding each direction's reverse
    tol         a solve stops when the walkers still in the white matter are below this x those
                launched (1e-4 is plenty for maps; 1e-6 for exact identities)
    max_steps   steps per solve, at most
    interp      bins > 1: each bin its own fODF, trilinear between voxel centers: "all" (default) from
                every voxel (grey matter too, as DIPY's trackers), "wm" from the white matter's voxels;
                "" or "off": each voxel's fODF in all its bins.  (One bin per voxel: its fODF either way.)
    points      > 0: each bin's walkers turn by the fODF interpolated at points^3 Gauss points of the bin
                (DIPY's: where each walker is), not one fODF per bin -- fine fODF detail without finer
                bins; 3 or more (0: off)
    device      the Warp device: None or "cuda" (the first GPU), "cuda:1", "cpu" (slow; testing)
    ends_only   drop the bundles' include ROIs (waypoints): only their start / end ROIs and exclusions
                (and hemisphere) define them; one avoid set instead of 2^(number of include ROIs).  The
                optic radiations' thalamic end is then cut to the LGN's eighth of the thalamus
                (mpat.afq.bundle_specs_iter; the same for every solver)
    movie       also write each bundle's solves as a video (mpat.solvemovie): at most this many frames
                per solve (e.g. 40); None: no videos
    seed_mask   a NIfTI (path or image) on the fODF's grid: the bundles from streamlines seeded in it as
                pyAFQ's GPU tracker seeds them (to match its maps; e.g. the seed mask pyAFQ wrote), not
                from launches at the white matter's edge: ``seed_count`` seeds where the mask exceeds
                ``seed_threshold`` (seeds_from_mask, as DIPY's random seeds; ``seed_rng``), one
                streamline per fODF peak at each (seed_peaks), its halves within the cone of the peak
                and of its reverse (Galerkin.seeded_bundle).  The solves cost the same for any count:
                more seeds, less sampling noise; the maps are for ``seed_count`` seeds
    stop        "interp" (default): as pyAFQ's GPU tracker, a walker stops where the WM PVE, trilinear at
                its landing point, is <= ``stop_threshold`` (default 0.4), and ends in that voxel whatever
                its tissue (CSF too); the start / end ROIs decide.  Walkers move in the voxels where the
                PVE exceeds the threshold somewhere; launches start at that domain's faces (from ROI
                voxels outside it).  Decided at each Gauss point of the turns: needs ``points`` (>= 3).
                "voxel": walkers move in the tissue labels' white-matter voxels (WM PVE > 0.4 per voxel)
                and end where they step into grey matter or the edge (CSF: lost)
    stop_grid   stop "interp": the WM PVE's grid it is interpolated on.  "fodf" (default): resampled to
                the fODF's grid first, as pyAFQ's tracker does; "native": on pyAFQ's PVE's own grid (the
                T1's resolution when it is there), mapped through the affines; walkers then go where a
                Gauss point of the turns is above the threshold (point_domain)
    until, prune  seeded bundles (seed_mask): how the forwards weighted by the partner stop, and what
                the solves drop (GalerkinConfig; defaults "ends", 1e-9; "walkers", 0: exact)
    soft_max_len  a soft maximum streamline length, mm (0: none), or {bundle name: mm}: a bundle's
                walkers move only within that distance of both its start and its end ROI (where every
                streamline no longer than that lies; walkers leaving are lost, longer paths within
                still count).  A bundle with it and no start or end voxel is an error
    verbose     progress every 5 s (not 30) and, once per solve, the time of each kernel in a step
    log_every   seconds between a solve's progress lines (default 30, or 5 with verbose)
    out_dir     outputs here instead of <pyAFQ output_dir>/mpat-galerkin
    viz, cores, viz_colormap, synthseg_thalamus, no_midline, fornix, **roi_kwargs: as mpat.run
    Returns {bundle: TransportResult} (a dict per participant for a GroupAFQ)."""
    kw = dict(bundles=bundles, bins=bins, step=step, angle=angle, sphere=sphere, tol=tol, max_steps=max_steps,
              floor=floor, dilate=dilate, interp=interp, points=points, device=device, viz=viz, cores=cores,
              synthseg_thalamus=synthseg_thalamus, no_midline=no_midline, fornix=fornix, viz_colormap=viz_colormap, verbose=verbose, log_every=log_every,
              ends_only=ends_only, movie=movie, seed_mask=seed_mask, seed_count=seed_count,
              seed_threshold=seed_threshold, seed_rng=seed_rng, stop=stop, stop_threshold=stop_threshold,
              stop_grid=stop_grid, until=until, prune=prune, soft_max_len=soft_max_len,
              **roi_kwargs)
    if hasattr(afq, "pAFQ_list"):
        return {f"sub-{s}_ses-{t}": run(p, **kw, out_dir=out_dir and str(Path(out_dir) / f"sub-{s}" / f"ses-{t}"))
                for s, t, p in zip(afq.valid_sub_list, afq.valid_ses_list, afq.pAFQ_list)}
    from . import afq as mafq
    from .afq import (
        _thalami_if_synthseg,
        roi_specs,
        stopped,
        tissue_labels,
        visualize,
        write_table,
    )

    t0 = time.time()
    clock = Clock()
    with clock("pyAFQ inputs"):
        sh_path = afq.export(f"{afq.export('tracking_params')['odf_model'].lower()}_params")
        sh_img = nib.load(sh_path)
        pve_img, bm_img = nib.load(afq.export("pve_internal")), nib.load(afq.export("brain_mask"))
    with clock("tissue labels"):
        transient, absorb, dseg = tissue_labels(sh_img, pve_img, bm_img)
    out = Path(out_dir) if out_dir else Path(afq.kwargs["output_dir"]) / "mpat-galerkin"
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)
    nib.save(nib.Nifti1Image(dseg, sh_img.affine), f"{stem}_desc-mpat_dseg.nii.gz")
    with clock("pyAFQ inputs"):
        thal = _thalami_if_synthseg(afq, sh_img, synthseg_thalamus)
    step_val, step_unit = parse_step(step)
    stop = "" if stop in (None, "", "voxel", "labels") else stop
    cfg = GalerkinConfig(bins=int(bins), step=step_val, step_unit=step_unit, angle=float(angle), sphere=sphere,
                         tol=float(tol), max_steps=int(max_steps), floor=float(floor), dilate=int(dilate),
                         interp="" if interp in (None, "", "off") else interp, points=int(points or 0),
                         stop=stop, stop_threshold=float(stop_threshold), until=until, prune=float(prune))
    stop_map = None
    if stop:
        if not cfg.points:
            raise ValueError("stop 'interp' (the default) is decided at the points of the turns: pass points >= 3 "
                             "(--galerkin-points), or stop='voxel' (--galerkin-stop voxel)")
        from .afq import _on_grid

        if stop_grid == "native":
            pv = np.asarray(pve_img.dataobj, np.float32)
            stop_map = (np.ascontiguousarray(pv[..., 2] if pv.ndim == 4 else pv),       # WM PVE (csf, gm, wm)
                        np.linalg.inv(pve_img.affine) @ sh_img.affine)
            dom = point_domain(stop_map, sh_img.shape[:3], cfg.stop_threshold, cfg.bins, cfg.points)
            zp = np.sqrt((pve_img.affine[:3, :3] ** 2).sum(0))
            log.info("stopping as pyAFQ's, on the PVE's own grid (%s, %s mm): walkers in the %d voxels with a turn "
                     "point where the WM PVE, trilinear, exceeds %g (the tissue labels' white matter: %d)",
                     "x".join(str(n) for n in pve_img.shape[:3]), "x".join(f"{z:.3g}" for z in zp), int(dom.sum()),
                     cfg.stop_threshold, int(transient.sum()))
        elif stop_grid == "fodf":
            stop_map = np.asarray(_on_grid(pve_img, sh_img), np.float32)[..., 2]       # WM PVE (csf, gm, wm)
            dom = interp_domain(stop_map, cfg.stop_threshold)
            log.info("stopping as pyAFQ's: walkers in the %d voxels where the WM PVE, trilinear, exceeds %g "
                     "somewhere (the tissue labels' white matter: %d)", int(dom.sum()), cfg.stop_threshold,
                     int(transient.sum()))
        else:
            raise ValueError(f"stop_grid: 'fodf' or 'native', not {stop_grid!r}")
        transient = dom
    sh = np.asarray(sh_img.dataobj, np.float32)
    zooms = sh_img.header.get_zooms()[:3]
    side = float(min(abs(z) for z in zooms))
    log.info("MPAT galerkin: step %g %s = %.4g mm = %.4g voxels (smallest voxel side %.4g mm)", step_val, step_unit,
             step_val * (side if step_unit == "vox" else 1.0), step_val / (1.0 if step_unit == "vox" else side), side)
    seeding = None
    if seed_mask is not None:
        with clock("seeds"):
            seeding = _seeding(seed_mask, sh_img, sh, transient, sphere, seed_count, seed_threshold, seed_rng)
    with clock("galerkin setup"):
        model = Bundles(sh, zooms, transient, absorb, cfg, device=device, verbose=verbose, log_every=log_every,
                        stop_map=stop_map)
        model.seeding = seeding
    rows, results = [], {}
    nb = len(bundles) if bundles else None
    specs = iter(_call(roi_specs, afq, sh_img, transient, stem, bundles, thalamus=thal, no_midline=no_midline,
                       ends_only=ends_only,
                       fornix=fornix, **roi_kwargs))
    interrupted = False
    try:
        while True:
            if verbose:
                log.info("making the next bundle's ROIs (pyAFQ)")
            with clock("bundle ROIs"):
                spec = next(specs, None)
            if spec is None:
                break
            i = len(results) + 1
            log.info("bundle %d%s: %s", i, f" of {nb}" if nb else "", spec["name"])
            L = soft_max_len.get(spec["name"], 0.0) if isinstance(soft_max_len, dict) else soft_max_len
            if L:
                spec = dict(spec, soft_max_len=float(L))
            tb = time.time()
            if movie:
                from .solvemovie import SolveMovie

                b = f"{stem}_bundle-{re.sub(r'[^A-Za-z0-9]', '', spec['name'])}"
                model.movie = SolveMovie(spec["name"], f"{b}_desc-solves.mp4", sh_img.affine, transient,
                                         frames=int(movie))
            with clock("bundle solves"):
                res = model.bundle(spec)
            with clock("writing outputs"):
                _write(res, stem, sh_img.affine, cfg, rows)
            results[res.name] = res
            write_table(stem, rows)
            if model.movie is not None:
                with clock("videos"):
                    model.movie.density(res.density)
                    model.movie.write()
                model.movie = None
            log.info("bundle %d%s (%s) done in %s; %s since the start", i, f" of {nb}" if nb else "", res.name,
                     _secs(time.time() - tb), _secs(time.time() - t0))
    except KeyboardInterrupt:       # keep what is done: its maps are written; the rest below
        interrupted = True
        stopped(len(results))
    finally:
        model.release()
    write_table(stem, rows)
    names = [re.sub(r"[^A-Za-z0-9]", "", n) for n in results]
    if cores and names:
        from .core import write_cores

        with clock("cores"):
            write_cores(out)
    if viz and names:
        try:
            t1 = afq.export("t1_masked")
        except Exception:  # noqa: BLE001 -- the collage falls back to the tissue labels
            t1 = None
        everything = mafq.bundles_on_disk(afq, out) if hasattr(mafq, "bundles_on_disk") else None
        with clock("viz"):
            _call(visualize, out, stem, names, sh_path, t1=t1, rois=True, colormap=viz_colormap,
                  all_bundles=everything)
    clock.report("MPAT galerkin time per stage", time.time() - t0)
    log.info("MPAT galerkin: %d bundles in %.0fs -> %s", len(results), time.time() - t0, out)
    if interrupted:
        raise KeyboardInterrupt
    return results


def run_cli(afq, a):
    """``mpat ... --galerkin``: ``run`` with the command line's options (``a``, argparse's
    namespace).  Options of the chain (--moments, --live, ...) do not apply."""
    off = [f for f, on in (("--moments", getattr(a, "moments", False)),
                           ("--live", getattr(a, "live", None) is not None),
                           ("--single-odf", getattr(a, "single_odf", False)),
                           ("--pyafq", getattr(a, "pyafq", False)),
                           ("--flux", getattr(a, "flux", False)), ("--fixels", getattr(a, "fixels", False)),
                           ("--ptt", getattr(a, "ptt", False)), ("--sift2", getattr(a, "sift2", False)),
                           ("--arbor", getattr(a, "arbor", False)),
                           ("--save-transitions", getattr(a, "save_transitions", None) is not None)) if on]
    if off:
        log.warning("--galerkin: %s ignored (chain options)", ", ".join(off))
    kw = {}
    return run(afq, bundles=a.bundles, bins=a.galerkin_bins, step=a.galerkin_step, angle=a.angle,
               sphere=a.galerkin_sphere, tol=a.tol, max_steps=a.galerkin_max_steps, floor=a.galerkin_floor,
               dilate=a.galerkin_dilate, interp=getattr(a, "galerkin_interp", None) or "all",
               points=getattr(a, "galerkin_points", 0) or 0,
               device=getattr(a, "gpu", None), viz=not a.no_viz,
               synthseg_thalamus=not a.mni_thalamus, no_midline=a.no_midline, out_dir=a.out_dir,
               fornix=not getattr(a, "no_fornix_exclude", False), viz_colormap=getattr(a, "colormap", None),
               verbose=getattr(a, "verbose", False), log_every=getattr(a, "galerkin_log_every", None),
               ends_only=getattr(a, "ends_only", False), movie=getattr(a, "movie", None),
               seed_mask=getattr(a, "galerkin_seed_mask", None), seed_count=getattr(a, "galerkin_seed_count", 1_000_000),
               seed_threshold=getattr(a, "galerkin_seed_threshold", 0.0), seed_rng=getattr(a, "galerkin_seed_rng", 0),
               stop=getattr(a, "galerkin_stop", "interp"), stop_threshold=getattr(a, "galerkin_stop_threshold", 0.4),
               stop_grid=getattr(a, "galerkin_stop_grid", "fodf"),
               until=getattr(a, "galerkin_until", "ends"), prune=getattr(a, "galerkin_prune", 1e-9),
               soft_max_len=getattr(a, "galerkin_soft_max_len", 0.0) or 0.0,
               **kw)
