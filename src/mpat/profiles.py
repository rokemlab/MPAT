"""Tract profiles: scalar values along a bundle's core, from its MPAT density map (issue 6).

For one bundle and its core (``mpat.core``, n nodes in world mm):

1. Voxels.  Every voxel with density > 0, with no density floor, connectivity or distance
   cutoff: each one gets a place on the profile, and counts in proportion to its density (step 4),
   so faint voxels and stray islands count for little without being dropped.
2. Position along the core.  Each support voxel's centre is projected onto the nearest point of
   the core polyline -- onto its segments, not its vertices -- for a continuous arc length s and
   that segment's direction u.  Past either end s runs on along the end segment (s < 0, or beyond
   the core's length) instead of stopping at the end node.  There is no graph search: a voxel goes
   where it lies along the tract, and where a bundle bends back on itself (a U) a voxel still lies
   on its own arm's segments even when a node of the other arm is nearer to it.  The exception is
   a tight hairpin: a voxel past one end of the core that is nearer another part of the core than
   that end goes to the nearer part.
3. Spreading.  A voxel is a cube, not a point, so it covers a stretch of the core and is shared
   between the nodes along that stretch.  Projected onto u, a uniform cube is the sum of three
   uniform lengths |a_k . u| (a_k the affine's columns; D (|u_x| + |u_y| + |u_z|) in all for
   isotropic voxels of side D), and its share of node n is the fraction of that projected mass
   in n's bin, between the midpoints to n's neighbours (the end nodes' bins are open), from the
   exact piecewise-polynomial CDF (``_box_cdf``).  That is the limit of supersampling the voxel
   infinitely finely and binning the subpoints by arc length, exact while the core is straight
   across the voxel; a voxel touches only the two or three bins it spans, and its shares sum to
   one.  Without this, nodes closer together than a voxel get no voxels (about half the nodes of
   a 100 mm bundle on 2 mm voxels).  Nodes are never moved or renumbered; two that coincide split
   their stretch of the core at the point they share.
   ``kernel="linear"`` (the default) spreads each voxel's trilinear hat instead of its cube: per
   axis a tent two voxels wide, the cube convolved with itself, so six uniform lengths instead of
   three.  That is how each voxel's value reaches the profile when the scalar is sampled with
   trilinear interpolation, as pyAFQ samples it along streamlines.  On a curved phantom with
   voxel-level noise in the scalar, the box kernel's profiles were 3.1x as rough as the true
   profile (pyAFQ's: 2.0x without streamline-length variation); the linear kernel's 1.5x, with
   the lowest RMSE of the three.
4. Weights.  d_i = density / peak x the voxel's share of the node.  ``weighting``:
   "density"   w_i = d_i (the default): the expected value over MPAT's walkers -- the mean that
               all of MPAT's streamlines see in the node's stretch -- with no cross-section geometry.
   "gaussian"  w_i = d_i exp(-D_i^2 / 2), D_i the Mahalanobis distance of the voxel centre from the
               node's d-weighted mean in its d-weighted covariance (pinvh, so thin cross-sections
               are kept).  Favours the bundle's core, finite everywhere: a one-voxel node, or a
               voxel at the centre, just gets d_i.
   "mahalanobis"  w_i = d_i / D_i, the same D_i: pyAFQ's ``gaussian_weights`` (inverse
               Mahalanobis distance) with every sum weighted by d.  As there, a voxel exactly at the
               mean gets zero weight; a node whose weights are all zero (one voxel) is weighted by
               d alone.  1 / D is unbounded near the centre, so with a few dozen voxels per node the
               one or two nearest it carry the profile: on a curved phantom with voxel-level noise
               (linear kernel) its profiles were 4.5x as rough as the true profile, against 3.7x
               for "gaussian" and 1.5x for "density".  Unlike pyAFQ, no voxels are removed first
               (pyAFQ drops streamlines more than 3 SD from any node's centre before weighting).
5. Profile.  P_n = sum w_i s_i / sum w_i over node n's voxels with a finite scalar s_i; NaN if
   there are none (info lists nodes with no voxels in ``empty_nodes``).  Scalars are put on the
   density grid with ``afq._on_grid`` (linear); voxels outside a scalar's grid have no value for it.
"""
from __future__ import annotations

import csv
import json
import os.path as op

import nibabel as nib
import numpy as np

from .afq import _on_grid
from .core import bundle_files

BLOCK = 1 << 20     # voxel x segment pairs per block of the nearest-segment search (memory bound)


def _project(x, core):
    """(s, u) of each point x (m, 3) -- see step 2: arc length along the core of its nearest point
    on the core polyline, and the unit direction of that segment; or None if the core has no
    segment of length > 0."""
    seg = np.diff(core, axis=0)
    L = np.linalg.norm(seg, axis=1)
    js = np.flatnonzero(L > 0)                               # coincident nodes make no segment
    if not len(js):
        return None
    s_node = np.r_[0, np.cumsum(L)][js]
    L, T, start = L[js], seg[js] / L[js, None], core[js]
    s, j = np.empty(len(x)), np.empty(len(x), int)
    step = max(1, BLOCK // len(js))
    for a in range(0, len(x), step):
        rel = x[a:a + step, None] - start                    # (m, J, 3)
        t = np.einsum("mjk,jk->mj", rel, T)                  # along each segment's line
        near = np.square(rel - np.clip(t, 0, L)[..., None] * T).sum(-1).argmin(1)
        t = t[np.arange(len(near)), near]
        # the end segments run on past the core's ends; every other point stays on its segment
        t = np.clip(t, np.where(near == 0, -np.inf, 0), np.where(near == len(js) - 1, np.inf, L[near]))
        s[a:a + step], j[a:a + step] = s_node[near] + t, near
    return s, T[j]


def _box_cdf(z, w):
    """P(U_1 + ... + U_p <= z) for independent U_k uniform on [-w_k/2, w_k/2]: z (m, K), w (m, p).
    The box-spline formula sum_e (-1)^|e| (y - e.w)_+^q / (q! prod w), y = z + sum(w) / 2, over the
    q widths that are not ~0 (an edge across u adds no length).  Widths under 1e-4 of the largest
    are dropped (CDF error < 1e-4): keeping them would cost the alternating sum its precision."""
    from math import factorial

    w = -np.sort(-w, axis=1)
    q = (w > 1e-4 * w[:, :1]).sum(1)
    out = np.empty(z.shape)
    for k in range(1, w.shape[1] + 1):
        i = q == k
        if not i.any():
            continue
        wk = w[i, :k]
        tot = wk.sum(1)[:, None]
        y = np.clip(z[i] + tot / 2, 0, tot)
        f = sum((-1) ** sum(e) * np.clip(y - (wk @ np.array(e))[:, None], 0, None) ** k
                for e in np.ndindex(*[2] * k))
        out[i] = np.clip(f / (np.prod(wk, 1)[:, None] * factorial(k)), 0, 1)
    return out


def _spread(x, core, affine, kernel="linear"):
    """(voxel index, node, share) of every voxel's shares of the nodes -- see step 3.  A core
    collapsed to a single point gives every voxel wholly to node 0.  ``kernel``: "box" spreads
    the voxel's cube; "linear" its trilinear hat (each axis a tent two voxels wide: the cube
    convolved with itself), as pyAFQ's trilinear sampling of the scalar spreads each voxel."""
    if kernel not in ("box", "linear"):
        raise ValueError(f"kernel must be 'box' or 'linear', not {kernel!r}")
    v = np.arange(len(x))
    su = _project(x, core)
    if su is None:
        return v, np.zeros(len(x), int), np.ones(len(x))
    s, u = su
    s_node = np.r_[0, np.cumsum(np.linalg.norm(np.diff(core, axis=0), axis=1))]
    edges = np.r_[-np.inf, (s_node[:-1] + s_node[1:]) / 2, np.inf]      # node n spans edges n..n+1
    w = np.abs(u @ affine[:3, :3])                           # the cube's three edges along u
    if kernel == "linear":
        w = np.repeat(w, 2, axis=1)                          # a tent is two boxes
    half = w.sum(1) / 2
    lo = np.searchsorted(edges, s - half) - 1
    hi = np.searchsorted(edges, s + half) - 1
    nodes = lo[:, None] + np.arange((hi - lo).max() + 1 if len(v) else 1)
    past = nodes > hi[:, None]                               # padding beyond the voxel's last node
    nodes[past] = 0
    F = _box_cdf(edges[nodes + 1] - s[:, None], w) - _box_cdf(edges[nodes] - s[:, None], w)
    F[past] = 0
    keep = F > 1e-12
    return np.broadcast_to(v[:, None], F.shape)[keep], nodes[keep], F[keep]


def _mahalanobis(x, d):
    """Mahalanobis distance of each x_i from the d-weighted mean, in the d-weighted population
    covariance (pseudo-inverted).  pyAFQ's ``gaussian_weights`` (population covariance, pinvh),
    with every sum weighted by d:
    https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/_fixes.py#L335-L350"""
    from scipy.linalg import pinvh

    r = x - d @ x / d.sum()
    S = (d[:, None] * r).T @ r / d.sum()
    D2 = np.einsum("ij,jk,ik->i", r, pinvh(0.5 * (S + S.T)), r)
    return np.sqrt(np.clip(D2, 0, None))


def _weights(x, d, owner, n, weighting="density"):
    """Unnormalised profile weight of every voxel (step 4)."""
    if weighting == "density":
        return d
    if weighting not in ("gaussian", "mahalanobis"):
        raise ValueError(f"weighting must be 'density', 'gaussian' or 'mahalanobis', not {weighting!r}")
    w = np.zeros(len(x))
    for k in range(n):
        i = np.flatnonzero(owner == k)
        if not len(i):
            continue
        D = _mahalanobis(x[i], d[i])
        if weighting == "gaussian":
            w[i] = d[i] * np.exp(-0.5 * D ** 2)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            wk = d[i] / D
        # inverse distances that are not finite get no weight, as in pyAFQ:
        # https://github.com/tractometry/pyAFQ/blob/65a43a88ff1cf1e681a04a47010c9bb8dbb07ed8/AFQ/_fixes.py#L355-L357
        wk[~np.isfinite(wk)] = 0
        w[i] = wk if wk.sum() > 0 else d[i]
    return w


def tract_profiles(density, affine, core, scalars, n_nodes=100,
                   weighting="density", kernel="linear"):
    """Profiles of one bundle: ({scalar: (n,) values}, info).

    density   3-D bundle density map (NaN / negative read as 0, as bundle_core does)
    core      (n, 3) core nodes in world mm (``Core.xyz``), or None when there is none
    scalars   {name: nibabel image}
    n_nodes   the node count when there is no core; otherwise the core's own count is kept
    weighting "density" (default), "gaussian" or "mahalanobis" (step 4)
    kernel    "linear" (default) or "box": how each voxel is spread along the core (step 3)
    """
    d = np.clip(np.nan_to_num(np.asarray(density, float)), 0, None)   # as bundle_core
    if d.max() <= 0 or core is None:
        status = "empty" if d.max() <= 0 else "no core"
        return {k: np.full(n_nodes, np.nan) for k in scalars}, dict(status=status)
    core = np.asarray(core, float)
    n = len(core)
    dn = d / d.max()
    ijk = np.argwhere(dn > 0)
    x = nib.affines.apply_affine(affine, ijk)
    vox, node, share = _spread(x, core, affine, kernel)
    w = _weights(x[vox], dn[tuple(ijk.T)][vox] * share, node, n, weighting)
    out = {}
    ref = nib.Nifti1Image(d, affine)
    for name, img in scalars.items():
        s = _on_grid(img, ref)[tuple(ijk.T)].astype(float)
        # outside the scalar's own grid there is no value (resampling would read 0 there)
        own = nib.affines.apply_affine(np.linalg.inv(img.affine), x)
        s[~np.all((own > -0.5) & (own < np.array(img.shape[:3]) - 0.5), axis=1)] = np.nan
        s = s[vox]
        ok = np.isfinite(s) & (w > 0)
        num = np.bincount(node[ok], w[ok] * s[ok], n)
        den = np.bincount(node[ok], w[ok], n)
        out[name] = np.divide(num, den, out=np.full(n, np.nan), where=den > 0)
    count = np.bincount(node, minlength=n)
    return out, dict(status="ok", n_voxels=len(ijk), empty_nodes=np.flatnonzero(count == 0).tolist())


def write_profiles(folder, scalars, weighting="density", kernel="linear"):
    """Profiles of every MPAT bundle under ``folder`` from its saved core (``write_cores``), as
    pyAFQ-style ``<stem>_desc-profiles_tractography.csv`` (tractID, nodeID, one column per scalar;
    missing values empty) and a ``.json`` with each bundle's status.

    scalars   {column name: path of a scalar image}, e.g. pyAFQ's exported ``dti_fa``
    weighting, kernel  passed to ``tract_profiles``
    """
    files = bundle_files(str(folder))
    out = next(iter(files.values()))["prefix"].split("_bundle-")[0] + "_desc-profiles_tractography.csv"
    images = {k: nib.load(p) for k, p in scalars.items()}
    rows, meta = [], {}
    for key, f in files.items():
        summary = f["prefix"] + "_desc-summary.json"
        name = json.load(open(summary)).get("bundle", key) if op.exists(summary) else key
        img = nib.load(f["density"])
        core_tsv = f["prefix"] + "_desc-core_path.tsv"
        core = np.loadtxt(core_tsv, skiprows=1, usecols=(1, 2, 3), ndmin=2) if op.exists(core_tsv) else None
        prof, meta[name] = tract_profiles(img.dataobj, img.affine, core, images, weighting=weighting,
                                          kernel=kernel)
        for i, values in enumerate(zip(*prof.values())):
            rows.append([name, i] + ["" if np.isnan(v) else repr(float(v)) for v in values])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tractID", "nodeID", *images])
        w.writerows(rows)
    with open(out[: -len(".csv")] + ".json", "w") as fh:
        json.dump(dict(method="mpat.profiles (see its docstring); nodeID 0 is the core's start end",
                       weighting=weighting, kernel=kernel,
                       scalars={k: str(p) for k, p in scalars.items()}, bundles=meta), fh, indent=1)
    return out
