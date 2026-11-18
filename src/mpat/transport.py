"""Bundles as linear transport (``mpat --transport``): walkers that run straight and turn,
solved on a GPU with NVIDIA Warp.

    mpat dwi.nii.gz dwi.bval dwi.bvec t1.nii.gz out/ --transport [--gpu cuda:1] [--bundles ...]
    mpat.transport.run(afq, bundles=["Left Inferior Fronto-occipital"])

The chain (mpat.chain) moves a walker one voxel at a time, into one of its 26 neighbours.  Here
a walker moves in a straight line in any of M directions (a DIPY sphere, 362 by default) and,
on average every ``mfp`` mm, turns into a new direction drawn from the local fODF within
``angle`` degrees of the old one.  The walkers' expected angular flux psi(x, Omega) obeys the
linear Boltzmann equation of neutron and radiative transport,

    Omega . grad psi + sigma psi = sigma int K(x; Omega' -> Omega) psi(x, Omega') dOmega',

    sigma = 1 / mfp,   K(x; Omega' -> Omega) = fODF(x, Omega) [angle(Omega, Omega') <= angle] / Z(x, Omega'),

solved here in its simplest deterministic form, with no acceleration:

- discrete ordinates: one psi per direction of the sphere (Voronoi solid angles as weights);
- step characteristics on the voxel grid: a voxel's psi from its upwind neighbours,
  psi = (src + sum_k a_k psi_upwind_k) / (sigma + sum_k a_k), a_k = |Omega_k| / h_k, swept plane
  by plane (every voxel of a diagonal plane at once, for every octant of directions);
- source iteration: the first sweep gives the walkers' straight runs from their launches, and
  sweep i adds the paths with exactly i turns (its source: the walkers turning out of the
  previous sweep's).  A solve stops when what the sweeps still to come would add (their
  increments extrapolated geometrically) is below ``tol`` of the total.

Walkers enter the white matter from grey matter and from the brain's edge: each such face of a
white-matter voxel emits walkers into it in every direction, as many as the fODF there,
cosine-weighted (about half a walker per face where fibres meet it head-on).  A walker ends where
it leaves the white matter: validly into grey matter or the edge; otherwise (CSF, outside, an
excluded voxel, a turn with no fODF within the angle) it is lost.  The forward solve gives
where the walkers launched from a set of voxels go (psi); the adjoint solve, the probability
that a walker at x heading Omega ends in a set of voxels (psi^dagger: the chain's backward
solve, swept the other way).  The bundle density is their product, summed over directions:
the path length (mm) per voxel of the walkers that are in the bundle.

Bundles follow MPAT's rules (mpat.solve): ends as the start / end ROIs require (one end in
start and the other in end, or one end in the endpoint ROI, or any two valid ends), include
ROIs by inclusion-exclusion over avoid sets, exclude ROIs (and avoid sets) taken out of the
white matter.  Per avoid set: one adjoint solve (a column per end kind), then a forward solve
per end kind.

``upsample``: solve on a finer grid than the fODF's (isotropic voxels of that many mm): the
SH coefficients interpolated linearly, the tissue labels made on that grid from their own
images, and the bundles' ROIs made on the fODF's grid (where pyAFQ warps them) and taken
nearest-neighbour onto it.  The sweeps' sideways blur shrinks with the
voxel size; memory and time grow with the number of voxels.

Outputs go to <output_dir>/mpat-transport (or ``out_dir``) under mpat.run's names, so the
collage, the viewer pages and the cores read them: per bundle its density (``density_map``),
the walkers' ends (``endpoints_map``), P(a walker launched from a voxel is in the bundle)
(``seedprob_probmap``), the walkers' directions per voxel (``visitodf_dwimap``) and a summary.
"""
from __future__ import annotations

import csv
import inspect
import json
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np

from .solve import Clock, _endpoint_kinds

log = logging.getLogger("mpat")

VALID_ENDS = (1, 3)     # grey matter, the brain's edge (mpat.chain's absorbing classes)
# a cell's six neighbour slots: x-, x+, y-, y+, z-, z+ (slot 2k: the - side of axis k)
SLOTS = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1]], np.int64)
REDUCE_BLOCKS = 4096    # row blocks per column in the sums on a GPU (64 on Warp's CPU device)
ODF_FLOOR = 1e-3        # visitation ODFs where the density reaches this x its peak (as mpat.afq)
LOG_EVERY = 100         # sweeps between progress lines


@dataclass
class TransportConfig:
    mfp: float = 2.0                 # mean straight run between turns, mm (the turning rate is 1 / mfp)
    angle: float = 35.0              # largest turn, degrees
    sphere: str = "symmetric362"     # the walkers' directions: a DIPY sphere holding each reverse
    tol: float = 1e-6                # what the remaining sweeps would add, relative: a solve stops below it
    max_iter: int = 5000             # sweeps per solve, at most

    def key(self):
        return asdict(self)


class Directions:
    """The walkers' directions (a DIPY sphere that holds each direction's reverse), their solid
    angles (spherical Voronoi), the turning cone, and the octants the sweeps go by."""

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
        neg = v < 0                      # octant bits: x < 0, y < 0, z < 0 (0 counts as +)
        self.octant = (neg[:, 0] * 1 + neg[:, 1] * 2 + neg[:, 2] * 4).astype(np.int32)
        groups = [np.flatnonzero(self.octant == o) for o in range(8)]
        self.Mo = max(len(g) for g in groups)
        self.odirs = np.full((8, self.Mo), -1, np.int32)
        for o, g in enumerate(groups):
            self.odirs[o, :len(g)] = g
        self.up = np.zeros((8, 3), np.int32)     # slot of the upwind / downwind neighbour
        self.dn = np.zeros((8, 3), np.int32)
        for o in range(8):
            for k in range(3):
                minus = (o >> k) & 1
                self.up[o, k] = 2 * k + minus
                self.dn[o, k] = 2 * k + 1 - minus


class Domain:
    """The white matter as the sweeps see it: its cells (in np.argwhere order), their six
    neighbours, and for each octant of directions the cells in sweep order, plane by plane."""

    def __init__(self, transient, zooms):
        self.shape = shape = tuple(int(s) for s in transient.shape)
        dims = np.asarray(shape, np.int64)
        self.ijk = np.argwhere(transient).astype(np.int64)
        self.N = N = len(self.ijk)
        if N == 0:
            raise ValueError("no white matter to walk in")
        self.lin = np.ravel_multi_index(tuple(self.ijk.T), shape)
        cid = np.full(int(dims.prod()), -1, np.int64)
        cid[self.lin] = np.arange(N)
        self.nbr_c = np.empty((N, 6), np.int32)
        self.nbr_v = np.empty((N, 6), np.int32)
        for s, off in enumerate(SLOTS):
            q = self.ijk + off
            ok = np.all((q >= 0) & (q < dims), axis=1)
            v = np.full(N, -1, np.int64)
            v[ok] = np.ravel_multi_index(tuple(q[ok].T), shape)
            self.nbr_v[:, s] = v
            self.nbr_c[:, s] = np.where(v >= 0, cid[np.maximum(v, 0)], -1)
        self.P = P = int(dims.sum() - 2)               # planes i' + j' + k' = 0 .. sum(dims) - 3
        self.order = np.empty((8, N), np.int32)
        self.pstart = np.empty((8, P + 1), np.int32)
        counts = np.zeros((8, P), np.int64)
        for o in range(8):
            key = np.zeros(N, np.int64)
            for k in range(3):
                key += (dims[k] - 1 - self.ijk[:, k]) if (o >> k) & 1 else self.ijk[:, k]
            self.order[o] = np.argsort(key, kind="stable")
            counts[o] = np.bincount(key, minlength=P)
            self.pstart[o] = np.concatenate([[0], np.cumsum(counts[o])])
        self.pmax = counts.max(0)
        self.planes = np.flatnonzero(self.pmax > 0)
        self.zooms = np.abs(np.asarray(zooms, np.float64))[:3]


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


def upsample_sh(sh_img, voxel_mm):
    """``sh_img`` on a grid of ``voxel_mm`` isotropic voxels over the same field of view.  Inside
    the fODF mask (taken nearest-neighbour, so it keeps its extent) the SH coefficients are
    interpolated linearly -- the fODF is linear in them -- from the voxels inside the mask only
    (normalised: the zeros outside do not shrink it at the edge); outside they stay zero."""
    from scipy import ndimage as ndi

    data = np.asarray(sh_img.dataobj, np.float32)
    zooms = np.abs(np.asarray(sh_img.header.get_zooms()[:3], np.float64))
    shape = np.asarray(data.shape[:3])
    r = float(voxel_mm) / zooms                        # fODF voxels per new voxel, per axis
    if np.any(r > 1):
        log.warning("--transport-upsample %g mm is coarser than the fODF's voxels (%s mm) along some axis",
                    voxel_mm, "x".join(f"{z:.3g}" for z in zooms))
    new = np.maximum(1, np.ceil(shape / r - 1e-6)).astype(np.int64)
    step = np.eye(4)
    step[:3, :3] = np.diag(r)
    step[:3, 3] = 0.5 * r - 0.5                        # the same field of view: voxel corners line up
    affine = sh_img.affine @ step
    mask = np.any(data != 0, axis=-1)
    maskf = mask.astype(np.float32)
    out = np.zeros((int(new.prod()), data.shape[-1]), np.float32)
    ii = np.arange(new[1] * new[2])
    for x in range(int(new[0])):                       # a slab at a time: index arrays stay small
        idx = np.stack([np.full(ii.size, x), ii // new[2], ii % new[2]])
        pts = idx * r[:, None] + step[:3, 3][:, None]
        near = np.clip(np.rint(pts).astype(np.int64), 0, (shape - 1)[:, None])
        inside = mask[tuple(near)]
        if not inside.any():
            continue
        rows = x * ii.size + np.flatnonzero(inside)
        p = pts[:, inside]
        wsum = ndi.map_coordinates(maskf, p, order=1, mode="nearest")    # > 0: the nearest voxel is in
        for j in range(data.shape[-1]):
            out[rows, j] = ndi.map_coordinates(data[..., j], p, order=1, mode="nearest") / wsum
    img = nib.Nifti1Image(out.reshape(tuple(new) + (data.shape[-1],)), affine)
    img.header.set_xyzt_units(*sh_img.header.get_xyzt_units())
    return img


@dataclass
class TransportResult:
    name: str
    density: np.ndarray       # (X, Y, Z) path length (mm) of the bundle's walkers per voxel
    endpoints: np.ndarray     # (X, Y, Z) the bundle's walkers' ends per voxel (two per walker)
    seed_prob: np.ndarray     # (X, Y, Z) P(a walker launched from this voxel is in the bundle)
    odf_ijk: np.ndarray       # (n, 3) voxels with a visitation ODF
    odf_coef: np.ndarray      # (n, ncoef) its SH coefficients (descoteaux07, legacy, lmax 8)
    info: dict = field(default_factory=dict)


class Transport:
    """The white matter, the walkers' directions and the fODF on them, on a Warp device; the
    solves; and bundles from them.

        model = Transport(sh, zooms, transient, absorb, TransportConfig(), device="cuda")
        res = model.bundle(dict(name="IFOF", include=[...], exclude=..., start=..., end=...))
    """

    def __init__(self, sh, zooms, transient, absorb, cfg=None, device=None, graphs=None):
        from . import _warp_transport
        from .gpu import get_device, warp

        t0 = time.time()
        self.cfg = cfg = cfg or TransportConfig()
        self.wp = wp = warp()
        self.K = K = _warp_transport
        self.dev = dev = get_device(device)
        # each sweep's ~(X + Y + Z) launches replayed as one graph: on a GPU (or on Warp's CPU
        # device, where Warp captures graphs too, to test this path)
        self.graphs = bool(dev.is_cuda) if graphs is None else bool(graphs)
        self.dirs = dirs = Directions(cfg.sphere, cfg.angle)
        self.dom = dom = Domain(transient, zooms)
        self.shape, self.V = dom.shape, int(np.prod(dom.shape))
        self.valid = np.isin(np.asarray(absorb), VALID_ENDS).ravel()
        self.sigma = 1.0 / float(cfg.mfp)
        N, M = dom.N, dirs.M
        fod = fodf_on(np.asarray(sh)[np.asarray(transient, bool)], dirs)

        def put(a, dt):
            return wp.array(np.ascontiguousarray(a), dtype=dt, device=dev)

        self.d_order, self.d_pstart = put(dom.order, wp.int32), put(dom.pstart, wp.int32)
        self.d_nbr_c, self.d_nbr_v = put(dom.nbr_c, wp.int32), put(dom.nbr_v, wp.int32)
        self.d_odirs, self.d_up, self.d_dn = put(dirs.odirs, wp.int32), put(dirs.up, wp.int32), put(dirs.dn, wp.int32)
        self.d_octant, self.d_rev = put(dirs.octant, wp.int32), put(dirs.rev, wp.int32)
        self.d_cptr, self.d_cidx = put(dirs.cptr, wp.int32), put(dirs.cidx, wp.int32)
        self.d_acoef = put(np.abs(dirs.v) / dom.zooms[None, :], wp.float32)
        self.d_hlen = put(dom.zooms, wp.float32)
        self.d_w = put(dirs.w, wp.float32)
        self.d_fod = put(fod, wp.float32)
        del fod
        self.d_wiz = wp.zeros((N, M), dtype=wp.float32, device=dev)
        wp.launch(K.cone_norm, dim=(N, M), inputs=[self.d_fod, self.d_w, self.d_cptr, self.d_cidx, self.d_wiz],
                  device=dev)
        self.d_none = wp.zeros((1, 1, 1), dtype=wp.float32, device=dev)
        self.nblk = REDUCE_BLOCKS if dev.is_cuda else 64
        col = N * M * 4
        log.info("transport on %s: %d white-matter voxels (%s mm) x %d directions (sphere %s, %.1f per turn within "
                 "%g deg), mean run %g mm; %.0f MB per angular-flux column, a bundle needs about %.1f GB (two end "
                 "kinds; %.1f GB with three); set up in %.1fs", dev, N, "x".join(f"{z:g}" for z in dom.zooms), M,
                 cfg.sphere, (dirs.cptr[1:] - dirs.cptr[:-1]).mean(), cfg.angle, cfg.mfp, col / 1e6,
                 self.need(2) / 1e9, self.need(3) / 1e9, time.time() - t0)
        if dev.is_cuda:
            free = float(dev.free_memory)
            if self.need(2) > free:
                log.warning("a bundle needs about %.1f GB on %s and %.1f GB is free: expect out-of-memory errors; "
                            "use a coarser --transport-upsample or a smaller --transport-sphere", self.need(2) / 1e9,
                            dev, free / 1e9)

    def need(self, n_kinds):
        """Device bytes a bundle with ``n_kinds`` end kinds needs at the peak: the fODF, its cone
        normalisers and the density by direction, plus the adjoint solve's two arrays (one
        column per kind), or, after it, the adjoint and a forward solve's two (one column)."""
        col = self.dom.N * self.dirs.M * 4
        return col * (3 + max(2 * n_kinds, n_kinds + 2)) + 4 * self.V * 3 * n_kinds

    def release(self):
        """Free the device arrays."""
        for k in [k for k in vars(self) if k.startswith("d_")]:
            setattr(self, k, None)

    # -------------------------------------------------------------- one source iteration
    def _sweep(self, adjoint, old, new, alive, bval, ncol, red):
        """new = one more turn: the walkers turning out of ``old`` (or, adjoint, the importance
        of turning), swept along their straight runs; ``red``: what it added and the total."""
        wp, K, dev, dom = self.wp, self.K, self.dev, self.dom
        N, M = dom.N, self.dirs.M
        if adjoint:
            wp.launch(K.turn_adjoint, dim=(N, M), inputs=[old, self.d_fod, self.d_wiz, self.d_w, self.d_rev,
                                                         self.d_cptr, self.d_cidx, self.sigma, ncol, new], device=dev)
        else:
            wp.launch(K.turn_forward, dim=(N, M), inputs=[old, self.d_fod, self.d_wiz, self.d_cptr, self.d_cidx,
                                                         self.sigma, ncol, new], device=dev)
        rest = [self.d_order, self.d_pstart, self.d_odirs, self.d_up, self.d_nbr_c, self.d_nbr_v, self.d_acoef,
                self.d_hlen, self.sigma, alive, bval, 0 if adjoint else 1, self.d_fod, ncol, new]
        Mo = self.dirs.Mo
        for d in dom.planes:
            wp.launch(K.sweep_plane, dim=(8, int(dom.pmax[d]), Mo), inputs=[int(d), *rest], device=dev)
        wp.launch(K.zero2d64, dim=(ncol, 2), inputs=[red], device=dev)
        wp.launch(K.increments, dim=(ncol, self.nblk), inputs=[new, old, self.d_w, N, M, self.nblk, red], device=dev)

    def solve(self, adjoint, alive, bval, what="solve"):
        """Source iteration for the columns of ``bval`` ((ncol, V) float32 on the device):
        forward, the walkers launched from each column's voxels (weighted by its values); or
        adjoint, the probability of ending in them.  ``alive``: (N,) uint8 on the device, the
        cells walkers may be in.  Returns (the angular flux on the device, (ncol, N, M), indexed
        by the sweep's direction: the reverse of the walker's for an adjoint; info)."""
        wp, dev, tol = self.wp, self.dev, self.cfg.tol
        ncol = int(bval.shape[0])
        N, M = self.dom.N, self.dirs.M
        bufs = [wp.zeros((ncol, N, M), dtype=wp.float32, device=dev) for _ in range(2)]
        red = wp.zeros((ncol, 2), dtype=wp.float64, device=dev)
        graphs = [None, None]
        t0 = time.perf_counter()
        prev, rem = None, np.full(ncol, np.inf)
        info = dict(sweeps=0, converged=False)
        it = 0
        for it in range(self.cfg.max_iter):
            old, new = bufs[it % 2], bufs[(it + 1) % 2]
            if self.graphs and it >= 1:          # after the first sweep (which compiles): replay a graph
                if graphs[it % 2] is None:
                    with wp.ScopedCapture(device=dev) as cap:
                        self._sweep(adjoint, old, new, alive, bval, ncol, red)
                    graphs[it % 2] = cap.graph
                wp.capture_launch(graphs[it % 2])
            else:
                self._sweep(adjoint, old, new, alive, bval, ncol, red)
            r = red.numpy()
            inc, tot = r[:, 0], r[:, 1]
            rem = np.full(ncol, np.inf)
            if prev is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    q = inc / prev
                ok = (tot > 0) & (prev > 0) & (q < 1)
                rem[ok] = np.maximum(inc[ok], 0.0) * np.maximum(q[ok], 0.0) / (1.0 - q[ok]) / tot[ok]
                rem[(tot > 0) & (inc <= 0)] = 0.0     # the sweep added nothing float32 can hold
            rem[tot <= 0] = 0.0                   # nothing launched (or nothing to reach)
            prev = inc
            worst = float(rem.max())
            if not np.all(np.isfinite(tot)):
                raise RuntimeError(f"{what}: the sweeps diverge (sweep {it + 1})")
            if (it + 1) % LOG_EVERY == 0:
                log.info("  %s sweep %d: still to come %.1e", what, it + 1, worst)
            if worst < tol:
                info["converged"] = True
                break
        sec = time.perf_counter() - t0
        info.update(sweeps=it + 1, remaining=float(rem.max()), seconds=round(sec, 2),
                    ms_per_sweep=round(1e3 * sec / (it + 1), 2))
        if not info["converged"]:
            log.warning("%s: not converged after %d sweeps (still to come %.1e of the total); walkers "
                        "may circulate in white matter they rarely leave", what, it + 1, info["remaining"])
        psi = bufs[(it + 1) % 2]
        bufs[it % 2] = None
        return psi, info

    # -------------------------------------------------------------- tallies at the white matter's faces
    def _launch_tally(self, J, alive):
        """Per voxel next to the white matter: the walkers launched from it (J None), or, per
        column of the adjoint J, those of them that end in that column's targets.  (ncol, V)."""
        wp, K, dev = self.wp, self.K, self.dev
        ncol = 1 if J is None else int(J.shape[0])
        out = wp.zeros((ncol, self.V), dtype=wp.float32, device=dev)
        wp.launch(K.launch_tally, dim=(self.dom.N, self.dirs.M),
                  inputs=[self.d_none if J is None else J, alive, self.d_nbr_c, self.d_nbr_v, self.d_up,
                          self.d_octant, self.d_acoef, self.d_hlen, self.d_fod, self.d_w, self.d_rev, ncol,
                          1 if J is None else 0, out], device=dev)
        return out.numpy().astype(np.float64)

    def _exit_tally(self, psi, alive):
        """Per voxel next to the white matter and column of the forward psi: the walkers that
        leave the white matter into it.  (ncol, V)."""
        wp, K, dev = self.wp, self.K, self.dev
        ncol = int(psi.shape[0])
        out = wp.zeros((ncol, self.V), dtype=wp.float32, device=dev)
        wp.launch(K.exit_tally, dim=(self.dom.N, self.dirs.M),
                  inputs=[psi, alive, self.d_nbr_c, self.d_nbr_v, self.d_dn, self.d_octant, self.d_acoef, self.d_w,
                          ncol, out], device=dev)
        return out.numpy().astype(np.float64)

    def _alive(self, avoid):
        return self.wp.array((~avoid[self.dom.lin]).astype(np.uint8), dtype=self.wp.uint8, device=self.dev)

    # -------------------------------------------------------------- bundles
    def bundle(self, spec):
        """One bundle: ``spec`` as mpat.afq.bundle_specs gives it (name; include, a list of
        masks; exclude, start, end: masks or None; on the fODF grid).  Returns a TransportResult."""
        wp, K, dev, dom, dirs = self.wp, self.K, self.dev, self.dom, self.dirs
        t0 = time.time()
        name = spec["name"]
        N, M, V = dom.N, dirs.M, self.V

        def flat(m):
            return None if m is None else np.asarray(m, bool).ravel()

        exc = flat(spec.get("exclude"))
        exc = np.zeros(V, bool) if exc is None else exc
        inc = [flat(m) for m in spec.get("include", ())]
        a_valid = self.valid & ~exc
        kinds, coup, rule = _endpoint_kinds(a_valid, flat(spec.get("start")), flat(spec.get("end")))
        nK = len(kinds)
        subsets = [T for r in range(len(inc) + 1) for T in combinations(range(len(inc)), r)]
        log.info("%s: %s; %d include ROI(s) -> %d avoid set(s) x (adjoint + forward), %d column(s) each",
                 name, rule, len(inc), len(subsets), nK)
        rho = wp.zeros((N, M), dtype=wp.float32, device=dev)
        launched = self._launch_tally(None, self._alive(exc))[0]        # walkers from each voxel
        ends = np.zeros(V)          # the bundle walkers' ends per voxel (both ends)
        good = np.zeros(V)          # per launch voxel: its walkers that are in the bundle
        check, solves = [], []
        for T in subsets:
            sign = (-1.0) ** len(T)
            avoid = exc.copy()
            for i in T:
                avoid |= inc[i]
            alive = self._alive(avoid)
            kb = np.stack([k & ~avoid for k in kinds])
            bval = wp.array(kb.astype(np.float32), dtype=wp.float32, device=dev)
            tag = f" avoiding include {', '.join(str(i + 1) for i in T)}" if T else ""
            J, info_a = self.solve(True, alive, bval, f"{name} adjoint{tag}")
            reach = self._launch_tally(J, alive)      # launches from each voxel that end in kind l
            leave = np.zeros((nK, V))                 # walkers from kind k leaving into each voxel
            info_f = []
            for k in range(nK):                       # one launch set at a time: less memory
                if not kb[k].any():
                    continue
                bk = wp.array(kb[k:k + 1].astype(np.float32), dtype=wp.float32, device=dev)
                psi, inf = self.solve(False, alive, bk, f"{name} forward{tag}, end kind {k + 1}/{nK}")
                leave[k] = self._exit_tally(psi, alive)[0]
                row = wp.array(np.asarray(coup[k:k + 1], np.float32), dtype=wp.float32, device=dev)
                wp.launch(K.add_density, dim=(N, M), inputs=[psi, J, self.d_rev, self.d_w, row, 1, nK, sign, rho],
                          device=dev)
                del psi
                info_f.append(inf)
            del J
            for k in range(nK):
                for l in range(nK):
                    if coup[k, l] == 0:
                        continue
                    from_k = kb[k] * reach[l]          # walkers launched in kind k that end in kind l ...
                    into_l = kb[l] * leave[k]          # ... counted where they end
                    ends += sign * coup[k, l] * (from_k + into_l)
                    good += sign * coup[k, l] * from_k
                    a, b = from_k.sum(), into_l.sum()
                    check.append(abs(a - b) / max(a, b, 1e-300))
            solves.append(dict(avoid=[i + 1 for i in T], adjoint=info_a, forward=info_f))
        rho_h = rho.numpy()
        rho = None
        dens = rho_h.sum(1, dtype=np.float64)
        expected = float(good.sum())
        density = np.zeros(V, np.float32)
        density[dom.lin] = np.clip(dens, 0, None)
        odf_ijk, odf_coef = self._visitation_odf(rho_h, dens)
        del rho_h
        seed = np.divide(good, launched, out=np.zeros(V), where=launched > 0)
        info = dict(model="linear transport (mpat.transport)", config=self.cfg.key(), endpoint_rule=rule,
                    expected_bundle_streamlines=expected,
                    launched_walkers=float((launched * a_valid).sum()),
                    endpoint_mass_check=float(ends.sum() / max(2 * expected, 1e-300)),
                    transport_identity=float(max(check) if check else 0.0),
                    n_solves=int(sum(1 + len(s["forward"]) for s in solves)), n_columns=nK,
                    sweeps=int(sum(s["adjoint"]["sweeps"] + sum(f["sweeps"] for f in s["forward"]) for s in solves)),
                    solves=solves, tolerance=self.cfg.tol,
                    negative_mass_before_clip=dict(density=float(-dens[dens < 0].sum()),
                                                   endpoints=float(-ends[ends < 0].sum())),
                    seconds=round(time.time() - t0, 1))
        log.info("%s: %.3g expected walkers in the bundle (of %.3g launched); endpoint check %.4f, "
                 "transport identity %.1e; %d sweeps in %.0fs", name, expected, info["launched_walkers"],
                 info["endpoint_mass_check"], info["transport_identity"], info["sweeps"], info["seconds"])
        shape = self.shape
        return TransportResult(name, density.reshape(shape), np.clip(ends, 0, None).astype(np.float32).reshape(shape),
                               np.clip(seed, 0, 1).astype(np.float32).reshape(shape), odf_ijk, odf_coef, info)

    def _visitation_odf(self, rho_dir, dens, sh_order=8):
        """The bundle's walkers' directions where its density reaches ODF_FLOOR x its peak: SH
        coefficients (descoteaux07, legacy, symmetric), antipodally symmetrised, in path length
        per steradian, so that they integrate to the voxel's density."""
        from dipy.core.sphere import Sphere
        from dipy.reconst.shm import sh_to_sf_matrix

        top = dens.max() if dens.size else 0.0
        keep = np.flatnonzero(dens >= ODF_FLOOR * top) if top > 0 else np.zeros(0, np.int64)
        _, inv = sh_to_sf_matrix(Sphere(xyz=self.dirs.v), sh_order_max=sh_order, return_inv=True)
        g = np.clip(rho_dir[keep].astype(np.float64), 0, None) / self.dirs.w[None, :]
        g = 0.5 * (g + g[:, self.dirs.rev])
        c = g @ inv
        c00 = dens[keep] / (2 * np.sqrt(np.pi))
        c *= np.divide(c00, c[:, 0], out=np.zeros_like(c00), where=c[:, 0] > 0)[:, None]
        return self.dom.ijk[keep], c.astype(np.float32)


# ------------------------------------------------------------------ MPAT on top of pyAFQ, as transport
def _call(f, *args, **kw):
    """``f(*args, **kw)``, leaving out the keywords ``f`` does not take (mpat.afq's helpers
    differ a little between MPAT versions)."""
    params = inspect.signature(f).parameters
    if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
        for k in [k for k in kw if k not in params]:
            if kw.pop(k) is not None:
                log.warning("%s takes no %s here; ignored", f.__name__, k)
    return f(*args, **kw)


def nearest_voxels(coarse_img, fine_img):
    """For every voxel of ``fine_img``'s grid, the linear index of the ``coarse_img`` voxel its
    centre lies in (exact at the field's edge too, where resampling takes points half a voxel
    out for outside)."""
    T = np.linalg.solve(coarse_img.affine, fine_img.affine)          # fine index -> coarse index
    sc, sf = np.asarray(coarse_img.shape[:3]), tuple(int(n) for n in fine_img.shape[:3])
    out = np.empty(int(np.prod(sf)), np.int64)
    ii = np.arange(sf[1] * sf[2])
    for x in range(sf[0]):
        idx = np.stack([np.full(ii.size, x), ii // sf[2], ii % sf[2], np.ones(ii.size, np.int64)])
        c = np.clip(np.rint(T[:3] @ idx).astype(np.int64), 0, (sc - 1)[:, None])
        out[x * ii.size:(x + 1) * ii.size] = np.ravel_multi_index(tuple(c), tuple(sc))
    return out


def _upsampled_roi_specs(afq, coarse_img, fine_img, stem, bundles, thalamus, no_midline, fornix, **kw):
    """mpat.afq.roi_specs for a solve on a finer grid than the fODF's: every bundle's ROIs made on
    the fODF's grid -- pyAFQ warps template ROIs onto the grid its mapping was made on (the
    DWI's) and labels them with the grid it is given, so they must be made there -- and then
    taken nearest-neighbour onto ``fine_img``'s grid, where their masks are written (as
    roi_specs writes them)."""
    from .afq import bundle_specs_iter, parc_file_of

    shape = tuple(coarse_img.shape[:3])
    near = nearest_voxels(coarse_img, fine_img)

    def up(m):
        if isinstance(m, np.ndarray) and m.shape == shape:
            return np.asarray(m).ravel()[near].reshape(fine_img.shape[:3])
        if isinstance(m, list):
            return [up(x) for x in m]
        return m

    for spec in _call(bundle_specs_iter, afq, coarse_img, np.zeros(shape, bool), bundles, thalamus=thalamus,
                      no_midline=no_midline, fornix=fornix, hypo_file=f"{stem}_model-synthseg2hypo_dseg.nii.gz",
                      parc_file=parc_file_of(stem), **kw):
        spec = {k: up(v) for k, v in spec.items()}
        b = f"{stem}_bundle-{re.sub(r'[^A-Za-z0-9]', '', spec['name'])}"

        def save(m, desc):
            nib.save(nib.Nifti1Image(np.asarray(m, np.uint8), fine_img.affine), f"{b}_desc-{desc}_mask.nii.gz")

        for k in ("start", "end"):
            if spec.get(k) is not None:
                save(spec[k], k)
        if spec["include"]:
            lab = np.zeros(fine_img.shape[:3], np.uint8)
            for i, m in enumerate(spec["include"], 1):
                lab[m & (lab == 0)] = i
            save(lab, "include")
        exc = spec.pop("exclude_rois", None)
        if exc is not None:
            save(exc, "exclude")
        yield spec


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


def run(afq, bundles=None, tol=1e-6, angle=35.0, mfp=2.0, sphere="symmetric362", device=None, viz=True,
        cores=True, synthseg_thalamus=True, no_midline=(), out_dir=None, fornix=True, max_iter=5000,
        viz_colormap=None, upsample=None, **roi_kwargs):
    """Segment pyAFQ's bundles as linear transport (module docstring), on a GPU with Warp.

    afq         ``ParticipantAFQ`` or ``GroupAFQ`` (every participant in turn)
    bundles     pyAFQ bundle names (default: all in the bundle dictionary)
    tol         a solve stops when what its remaining sweeps would add is below this (relative)
    angle       largest turn, degrees
    mfp         mean straight run between turns, mm
    sphere      the walkers' directions: a DIPY sphere holding each direction's reverse
                ("symmetric362", "symmetric642", "repulsion724", ...)
    device      the Warp device: None or "cuda" (the first GPU), "cuda:1", "cpu" (slow; testing)
    upsample    solve on isotropic voxels of this many mm (e.g. 1.0) instead of the fODF's grid
                (module docstring); outputs are on that grid
    max_iter    sweeps per solve, at most (a turn each: long bundles with a short mfp need many)
    out_dir     outputs here instead of <pyAFQ output_dir>/mpat-transport
    viz, cores  the collage and viewer pages, and the bundle cores, as mpat.run writes them
    viz_colormap  NiiVue colour map of the density on the viewer pages (as mpat.run's)
    synthseg_thalamus, no_midline, fornix, **roi_kwargs (midline_tol, where MPAT has it): the
                bundles' ROIs exactly as mpat.run makes them (mpat.afq.bundle_specs)
    Returns {bundle: TransportResult} (a dict per participant for a GroupAFQ)."""
    if hasattr(afq, "pAFQ_list"):
        kw = dict(bundles=bundles, tol=tol, angle=angle, mfp=mfp, sphere=sphere, device=device, viz=viz, cores=cores,
                  synthseg_thalamus=synthseg_thalamus, no_midline=no_midline, fornix=fornix, max_iter=max_iter,
                  viz_colormap=viz_colormap, upsample=upsample, **roi_kwargs)
        return {f"sub-{s}_ses-{t}": run(p, **kw, out_dir=out_dir and str(Path(out_dir) / f"sub-{s}" / f"ses-{t}"))
                for s, t, p in zip(afq.valid_sub_list, afq.valid_ses_list, afq.pAFQ_list)}
    from . import afq as mafq
    from .afq import _thalami_if_synthseg, roi_specs, tissue_labels, visualize

    t0 = time.time()
    clock = Clock()
    with clock("pyAFQ inputs"):
        sh_path = afq.export(f"{afq.export('tracking_params')['odf_model'].lower()}_params")
        sh_img = nib.load(sh_path)
        pve_img, bm_img = nib.load(afq.export("pve_internal")), nib.load(afq.export("brain_mask"))
    own_img = sh_img                   # the fODF's grid: where pyAFQ's ROIs are made
    if upsample:
        with clock("upsampling the fODF"):
            sh_img = upsample_sh(own_img, float(upsample))
        log.info("solving on %g mm voxels (the fODF's: %s mm): %s voxels", float(upsample),
                 "x".join(f"{z:.3g}" for z in own_img.header.get_zooms()[:3]),
                 "x".join(str(n) for n in sh_img.shape[:3]))
    with clock("tissue labels"):
        transient, absorb, dseg = tissue_labels(sh_img, pve_img, bm_img)
    out = Path(out_dir) if out_dir else Path(afq.kwargs["output_dir"]) / "mpat-transport"
    out.mkdir(parents=True, exist_ok=True)
    stem = str(out / Path(afq.kwargs["base_fname"]).name)
    nib.save(nib.Nifti1Image(dseg, sh_img.affine), f"{stem}_desc-mpat_dseg.nii.gz")
    with clock("pyAFQ inputs"):
        thal = _thalami_if_synthseg(afq, own_img, synthseg_thalamus)
    cfg = TransportConfig(mfp=mfp, angle=angle, sphere=sphere, tol=tol, max_iter=max_iter)
    with clock("transport setup"):
        model = Transport(np.asarray(sh_img.dataobj, np.float32), sh_img.header.get_zooms()[:3], transient,
                          absorb, cfg, device=device)
    rows, results = [], {}
    if upsample:
        specs = _upsampled_roi_specs(afq, own_img, sh_img, stem, bundles, thal, no_midline, fornix, **roi_kwargs)
    else:
        specs = _call(roi_specs, afq, sh_img, transient, stem, bundles, thalamus=thal, no_midline=no_midline,
                      fornix=fornix, **roi_kwargs)
    specs = iter(specs)
    try:
        while True:
            with clock("bundle ROIs"):
                spec = next(specs, None)
            if spec is None:
                break
            with clock("bundle solves"):
                res = model.bundle(spec)
            with clock("writing outputs"):
                _write(res, stem, sh_img.affine, cfg, rows)
            results[res.name] = res
    finally:
        model.release()
    if rows:
        with open(f"{stem}_desc-mpat_table.tsv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
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
            # the per-bundle pages draw the fODF only on its own grid (not upsampled)
            _call(visualize, out, stem, names, None if upsample else sh_path, t1=t1, rois=True,
                  colormap=viz_colormap, all_bundles=everything)
    clock.report("MPAT transport time per stage", time.time() - t0)
    log.info("MPAT transport: %d bundles in %.0fs -> %s", len(results), time.time() - t0, out)
    return results


def run_cli(afq, a):
    """``mpat ... --transport``: ``run`` with the command line's options (``a``, argparse's
    namespace).  Options of the chain alone (--moments, --live, --single-odf, --max-length,
    --gpu-backend, --gpu-weights, --pyafq, --flux, --fixels, --ptt and its settings, --sift2,
    --arbor, --save-transitions) do not apply."""
    off = [f for f, on in (("--moments", getattr(a, "moments", False)),
                           ("--live", getattr(a, "live", None) is not None),
                           ("--single-odf", getattr(a, "single_odf", False)),
                           ("--pyafq", getattr(a, "pyafq", False)),
                           ("--flux", getattr(a, "flux", False)), ("--fixels", getattr(a, "fixels", False)),
                           ("--ptt", getattr(a, "ptt", False)), ("--sift2", getattr(a, "sift2", False)),
                           ("--arbor", getattr(a, "arbor", False)),
                           ("--save-transitions", getattr(a, "save_transitions", None) is not None)) if on]
    if off:
        log.warning("--transport: %s ignored (chain options)", ", ".join(off))
    kw = {}
    if getattr(a, "midline_tol", None) is not None:
        kw["midline_tol"] = a.midline_tol
    return run(afq, bundles=a.bundles, tol=a.tol, angle=a.angle, mfp=a.transport_mfp, sphere=a.transport_sphere,
               upsample=getattr(a, "transport_upsample", None), max_iter=getattr(a, "transport_max_iter", 5000),
               device=getattr(a, "gpu", None), viz=not a.no_viz, synthseg_thalamus=not a.mni_thalamus,
               no_midline=a.no_midline, out_dir=a.out_dir, fornix=not getattr(a, "no_fornix_exclude", False),
               viz_colormap=getattr(a, "colormap", None), **kw)
