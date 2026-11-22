"""A Monte Carlo of mpat.galerkin's walkers (numba): the same launches, steps, turns and ends,
sampled walker by walker -- the reference the scheme's densities converge to."""
import numpy as np
from numba import njit


@njit(cache=True)
def _walk(p, m, alive_grid, F, vidg, w, v, cptr, cidx, inv_zoom, smm, maxsteps, pv, pl):
    """One walker from point p (voxel coordinates) heading m.  Fills pv/pl with the voxels it
    passes and the path length (mm) in each; returns (number of entries, the voxel it left into
    as a linear index, or -1 if lost)."""
    X, Y, Z = alive_grid.shape
    n = 0
    for _ in range(maxsteps):
        d0 = smm * v[m, 0] * inv_zoom[0]
        d1 = smm * v[m, 1] * inv_zoom[1]
        d2 = smm * v[m, 2] * inv_zoom[2]
        cx, cy, cz = int(np.floor(p[0])), int(np.floor(p[1])), int(np.floor(p[2]))
        big = 1e30
        tx, ty, tz = big, big, big
        sx, sy, sz = 0, 0, 0
        if d0 > 0:
            tx, sx = (cx + 1 - p[0]) / d0, 1
        elif d0 < 0:
            tx, sx = (cx - p[0]) / d0, -1
        if d1 > 0:
            ty, sy = (cy + 1 - p[1]) / d1, 1
        elif d1 < 0:
            ty, sy = (cy - p[1]) / d1, -1
        if d2 > 0:
            tz, sz = (cz + 1 - p[2]) / d2, 1
        elif d2 < 0:
            tz, sz = (cz - p[2]) / d2, -1
        tc = 0.0
        while True:
            tn = min(tx, ty, tz)
            pv[n] = (cx * Y + cy) * Z + cz
            pl[n] = smm * (min(tn, 1.0) - tc)
            n += 1
            if tn >= 1.0:
                break
            if tx <= ty and tx <= tz:
                cx += sx
                tx += sx / d0
            elif ty <= tz:
                cy += sy
                ty += sy / d1
            else:
                cz += sz
                tz += sz / d2
            tc = tn
            if cx < 0 or cy < 0 or cz < 0 or cx >= X or cy >= Y or cz >= Z:
                return n, -1
            if not alive_grid[cx, cy, cz]:
                return n, (cx * Y + cy) * Z + cz
        p[0] += d0
        p[1] += d1
        p[2] += d2
        c = vidg[cx, cy, cz]
        tot = 0.0
        for q in range(cptr[m], cptr[m + 1]):
            tot += F[c, cidx[q]] * w[cidx[q]]
        if tot <= 0:
            return n, -1
        u = np.random.random() * tot
        acc = 0.0
        mn = cidx[cptr[m + 1] - 1]
        for q in range(cptr[m], cptr[m + 1]):
            acc += F[c, cidx[q]] * w[cidx[q]]
            if u < acc:
                mn = cidx[q]
                break
        m = mn
    return n, -1


@njit(cache=True)
def _run(seed, fvox, fgv, fax, fsg, ijk, alive_grid, F, vidg, w, v, cptr, cidx, inv_zoom, smm, nper, src, tgt,
         maxsteps, dens, ends):
    np.random.seed(seed)
    pv = np.empty(maxsteps * 8, np.int64)
    pl = np.empty(maxsteps * 8, np.float64)
    p = np.empty(3)
    good = 0.0
    launched = 0.0
    for f in range(len(fvox)):
        g = fgv[f]
        if src[g] == 0:
            continue
        c = fvox[f]
        a = fax[f]
        for m in range(len(w)):
            cs = v[m, a] * fsg[f]
            if cs <= 0 or F[c, m] <= 0:
                continue
            W = cs * F[c, m] * w[m] / nper
            launched += W * nper
            for _ in range(nper):
                for k in range(3):
                    p[k] = ijk[c, k] + np.random.random()
                p[a] = ijk[c, a] + (0.0 if fsg[f] > 0 else 1.0)
                n, e = _walk(p, m, alive_grid, F, vidg, w, v, cptr, cidx, inv_zoom, smm, maxsteps, pv, pl)
                if e >= 0 and tgt[e] != 0:
                    good += W
                    ends[g] += W
                    ends[e] += W
                    for i in range(n):
                        dens[pv[i]] += W * pl[i]
    return good, launched


def monte_carlo(model, sh, kinds_src_tgt, nper=200, seed=0, maxsteps=4000):
    """Monte Carlo of a mpat.galerkin.Galerkin model's walkers (its domain, directions, faces;
    the fODF from ``sh``, the volume's SH coefficients), for pairs (source mask, target mask) over the volume: the walkers launched from
    source voxels that end in target voxels.  Returns (their number, the launched, their density
    (V,), their ends (V,))."""
    from mpat.galerkin import fodf_on

    shape = model.shape
    V = int(np.prod(shape))
    F = fodf_on(np.asarray(sh)[tuple(model.ijk.T)], model.dirs).astype(np.float64)
    vidg = np.full(shape, -1, np.int64)
    vidg[tuple(model.ijk.T)] = np.arange(model.Nv)
    alive_grid = vidg >= 0
    fvox = model.d_fvox.numpy().astype(np.int64)
    fgv3 = model.d_fgv.numpy().astype(np.int64)
    fgv = np.ravel_multi_index(tuple(fgv3.T), shape) if len(fgv3) else np.zeros(0, np.int64)
    fax = model.d_fax.numpy().astype(np.int64)
    fsg = model.d_fsg.numpy().astype(np.int64)
    d = model.dirs
    dens, ends = np.zeros(V), np.zeros(V)
    good = launched = 0.0
    for i, (s, t) in enumerate(kinds_src_tgt):
        gd, la = _run(seed + i, fvox, fgv, fax, fsg, model.ijk.astype(np.int64), alive_grid, F, vidg, d.w, d.v,
                      d.cptr.astype(np.int64), d.cidx.astype(np.int64), 1.0 / model.zooms, model.smm, nper,
                      np.asarray(s, np.int8).ravel(), np.asarray(t, np.int8).ravel(), maxsteps, dens, ends)
        good += gd
        launched += la
    return good, launched, dens, ends


@njit(cache=True)
def _run_seeds(seed, pts, sv, allowed, F, vidg, alive_grid, w, v, cptr, cidx, rev, inv_zoom, smm, nper, tlab, T,
               maxsteps, ends, pairs):
    np.random.seed(seed)
    pv = np.empty(maxsteps * 8, np.int64)
    pl = np.empty(maxsteps * 8, np.float64)
    p = np.empty(3)
    M = len(w)
    for k in range(len(pts)):
        c = sv[k]
        tot = 0.0
        for m in range(M):
            if allowed[k, m]:
                tot += F[c, m] * w[m]
        if tot <= 0:
            continue
        for _ in range(nper):
            u = np.random.random() * tot
            acc = 0.0
            m0 = -1
            for m in range(M):
                if allowed[k, m]:
                    acc += F[c, m] * w[m]
                    if u < acc:
                        m0 = m
                        break
            if m0 < 0:
                continue
            e = np.full(2, -1, np.int64)
            for h in range(2):
                for q in range(3):
                    p[q] = pts[k, q]
                _, x = _walk(p, m0 if h == 0 else rev[m0], alive_grid, F, vidg, w, v, cptr, cidx, inv_zoom, smm,
                             maxsteps, pv, pl)
                if x >= 0 and tlab[x] > 0:
                    e[h] = tlab[x] - 1
            if e[0] >= 0:
                ends[e[0]] += 1.0 / nper
                if e[1] >= 0:
                    pairs[e[0], e[1]] += 1.0 / nper


def monte_carlo_seeds(model, sh, seeds, targets, allowed=None, nper=2000, seed=0, maxsteps=4000):
    """Monte Carlo of Galerkin.seeded: walkers from points (voxel coordinates, centers at
    integers), each heading m with probability f(m) w(m) over its allowed directions, a second
    walker heading the opposite way; ``targets``: disjoint masks.  Returns (ends (T,): walkers of
    the first heading ending in each target, per seed summed; pairs (T, T): both)."""
    from mpat.galerkin import fodf_on

    shape = model.shape
    F = fodf_on(np.asarray(sh)[tuple(model.ijk.T)], model.dirs).astype(np.float64)
    vidg = np.full(shape, -1, np.int64)
    vidg[tuple(model.ijk.T)] = np.arange(model.Nv)
    pts = np.asarray(seeds, np.float64).reshape(-1, 3) + 0.5       # the walk's frame: voxel i spans [i, i + 1)
    sv = vidg[tuple(np.floor(pts).astype(np.int64).T)]
    keep = sv >= 0
    pts, sv = pts[keep], sv[keep]
    d = model.dirs
    al = np.ones((len(pts), d.M), np.uint8) if allowed is None else np.asarray(allowed, np.uint8)[keep]
    tlab = np.zeros(int(np.prod(shape)), np.int64)
    for t, tm in enumerate(targets):
        tlab[np.asarray(tm, bool).ravel()] = t + 1
    T = len(targets)
    ends, pairs = np.zeros(T), np.zeros((T, T))
    _run_seeds(seed, pts, sv, al, F, vidg, vidg >= 0, d.w, d.v, d.cptr.astype(np.int64), d.cidx.astype(np.int64),
               d.rev.astype(np.int64), 1.0 / model.zooms, model.smm, nper, tlab, T, maxsteps, ends, pairs)
    return ends, pairs


# ------------------------------------------------------------------ the fODF at each walker's point
@njit(cache=True)
def _pmf_point(x, y, z, R, rowmap, thr, out):
    """The clipped fODF interpolated trilinearly at (x, y, z) (voxel centres at integers) from
    the 8 voxel centres around it (a voxel with no row counts as zero), values below thr x its
    maximum zeroed (galerkin.point_fodf, unnormalised)."""
    M = out.shape[0]
    for j in range(M):
        out[j] = 0.0
    fx, fy, fz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    rx, ry, rz = x - fx, y - fy, z - fz
    X, Y, Z = rowmap.shape
    for c in range(8):
        ix, wx = (fx + 1, rx) if c & 1 else (fx, 1.0 - rx)
        iy, wy = (fy + 1, ry) if c & 2 else (fy, 1.0 - ry)
        iz, wz = (fz + 1, rz) if c & 4 else (fz, 1.0 - rz)
        wt = wx * wy * wz
        if wt <= 0 or ix < 0 or iy < 0 or iz < 0 or ix >= X or iy >= Y or iz >= Z:
            continue
        r = rowmap[ix, iy, iz]
        if r < 0:
            continue
        for j in range(M):
            out[j] += wt * R[r, j]
    mx = 0.0
    for j in range(M):
        mx = max(mx, out[j])
    for j in range(M):
        if out[j] < thr * mx:
            out[j] = 0.0


@njit(cache=True)
def _walk_pt(p, m, alive_grid, R, rowmap, thr, w, v, cptr, cidx, inv_zoom, smm, maxsteps, pmf):
    """As _walk, each turn by the fODF at the walker's own point.  Returns the voxel it left
    into (linear index) or -1 if lost."""
    X, Y, Z = alive_grid.shape
    for _ in range(maxsteps):
        d0 = smm * v[m, 0] * inv_zoom[0]
        d1 = smm * v[m, 1] * inv_zoom[1]
        d2 = smm * v[m, 2] * inv_zoom[2]
        cx, cy, cz = int(np.floor(p[0])), int(np.floor(p[1])), int(np.floor(p[2]))
        big = 1e30
        tx, ty, tz = big, big, big
        sx, sy, sz = 0, 0, 0
        if d0 > 0:
            tx, sx = (cx + 1 - p[0]) / d0, 1
        elif d0 < 0:
            tx, sx = (cx - p[0]) / d0, -1
        if d1 > 0:
            ty, sy = (cy + 1 - p[1]) / d1, 1
        elif d1 < 0:
            ty, sy = (cy - p[1]) / d1, -1
        if d2 > 0:
            tz, sz = (cz + 1 - p[2]) / d2, 1
        elif d2 < 0:
            tz, sz = (cz - p[2]) / d2, -1
        while True:
            tn = min(tx, ty, tz)
            if tn >= 1.0:
                break
            if tx <= ty and tx <= tz:
                cx += sx
                tx += sx / d0
            elif ty <= tz:
                cy += sy
                ty += sy / d1
            else:
                cz += sz
                tz += sz / d2
            if cx < 0 or cy < 0 or cz < 0 or cx >= X or cy >= Y or cz >= Z:
                return -1
            if not alive_grid[cx, cy, cz]:
                return (cx * Y + cy) * Z + cz
        p[0] += d0
        p[1] += d1
        p[2] += d2
        _pmf_point(p[0] - 0.5, p[1] - 0.5, p[2] - 0.5, R, rowmap, thr, pmf)
        tot = 0.0
        for q in range(cptr[m], cptr[m + 1]):
            tot += pmf[cidx[q]] * w[cidx[q]]
        if tot <= 0:
            return -1
        u = np.random.random() * tot
        acc = 0.0
        mn = cidx[cptr[m + 1] - 1]
        for q in range(cptr[m], cptr[m + 1]):
            acc += pmf[cidx[q]] * w[cidx[q]]
            if u < acc:
                mn = cidx[q]
                break
        m = mn
    return -1


@njit(cache=True)
def _run_seeds_pt(seed, pts, allowed, R, rowmap, thr, alive_grid, w, v, cptr, cidx, rev, inv_zoom, smm, nper, tlab,
                  maxsteps, ends, pairs):
    np.random.seed(seed)
    p = np.empty(3)
    M = len(w)
    pmf = np.empty(M)
    for k in range(len(pts)):
        _pmf_point(pts[k, 0] - 0.5, pts[k, 1] - 0.5, pts[k, 2] - 0.5, R, rowmap, thr, pmf)
        tot = 0.0
        for m in range(M):
            if allowed[k, m]:
                tot += pmf[m] * w[m]
        if tot <= 0:
            continue
        p0 = pmf.copy()
        for _ in range(nper):
            u = np.random.random() * tot
            acc = 0.0
            m0 = -1
            for m in range(M):
                if allowed[k, m]:
                    acc += p0[m] * w[m]
                    if u < acc:
                        m0 = m
                        break
            if m0 < 0:
                continue
            e = np.full(2, -1, np.int64)
            for h in range(2):
                for q in range(3):
                    p[q] = pts[k, q]
                x = _walk_pt(p, m0 if h == 0 else rev[m0], alive_grid, R, rowmap, thr, w, v, cptr, cidx, inv_zoom,
                             smm, maxsteps, pmf)
                if x >= 0 and tlab[x] > 0:
                    e[h] = tlab[x] - 1
            if e[0] >= 0:
                ends[e[0]] += 1.0 / nper
                if e[1] >= 0:
                    pairs[e[0], e[1]] += 1.0 / nper


def monte_carlo_seeds_points(model, sh, seeds, targets, allowed=None, nper=2000, seed=0, maxsteps=4000, among=None):
    """As monte_carlo_seeds, each walker turning (and starting) by the fODF interpolated at its own
    point, as DIPY's tracker and Galerkin with points: the reference those converge to."""
    from mpat.galerkin import raw_sf
    from mpat.geometry import PMF_THRESHOLD

    shape = model.shape
    R, rowmap = raw_sf(np.asarray(sh), model.ijk, shape, model.dirs, among=among)
    vidg = np.full(shape, -1, np.int64)
    vidg[tuple(model.ijk.T)] = np.arange(model.Nv)
    pts = np.asarray(seeds, np.float64).reshape(-1, 3) + 0.5       # the walk's frame: voxel i spans [i, i + 1)
    keep = vidg[tuple(np.floor(pts).astype(np.int64).T)] >= 0
    pts = pts[keep]
    d = model.dirs
    al = np.ones((len(pts), d.M), np.uint8) if allowed is None else np.asarray(allowed, np.uint8)[keep]
    tlab = np.zeros(int(np.prod(shape)), np.int64)
    for t, tm in enumerate(targets):
        tlab[np.asarray(tm, bool).ravel()] = t + 1
    T = len(targets)
    ends, pairs = np.zeros(T), np.zeros((T, T))
    _run_seeds_pt(seed, pts, al, R.astype(np.float64), rowmap.astype(np.int64), float(PMF_THRESHOLD), vidg >= 0, d.w,
                  d.v, d.cptr.astype(np.int64), d.cidx.astype(np.int64), d.rev.astype(np.int64), 1.0 / model.zooms,
                  model.smm, nper, tlab, maxsteps, ends, pairs)
    return ends, pairs


# ------------------------------------------------------------------ seeded streamlines, pyAFQ's GPU tracker's
@njit(cache=True)
def _stop_val(x, y, z, S):
    """S trilinear at (x, y, z) (voxel centres at integers); -1 if a corner is off the grid."""
    fx, fy, fz = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    X, Y, Z = S.shape
    rx, ry, rz = x - fx, y - fy, z - fz
    if fx < 0 or fy < 0 or fz < 0 or fx >= X or fy >= Y or fz >= Z:
        return -1.0
    if (fx + 1 >= X and rx > 0) or (fy + 1 >= Y and ry > 0) or (fz + 1 >= Z and rz > 0):
        return -1.0
    fx1, fy1, fz1 = min(fx + 1, X - 1), min(fy + 1, Y - 1), min(fz + 1, Z - 1)
    out = 0.0
    for c in range(8):
        ix, wx = (fx1, rx) if c & 1 else (fx, 1.0 - rx)
        iy, wy = (fy1, ry) if c & 2 else (fy, 1.0 - ry)
        iz, wz = (fz1, rz) if c & 4 else (fz, 1.0 - rz)
        out += wx * wy * wz * S[ix, iy, iz]
    return out


@njit(cache=True)
def _walk_pt_stop(p, m, alive_grid, R, rowmap, thr, w, v, cptr, cidx, inv_zoom, smm, maxsteps, pmf, S, sthr, use):
    """As _walk_pt; with ``use``, after each step the walker stops where S (trilinear at its point)
    is <= sthr, ending in the voxel there.  Returns the voxel it ended in, or -1 if lost."""
    X, Y, Z = alive_grid.shape
    for _ in range(maxsteps):
        d0 = smm * v[m, 0] * inv_zoom[0]
        d1 = smm * v[m, 1] * inv_zoom[1]
        d2 = smm * v[m, 2] * inv_zoom[2]
        cx, cy, cz = int(np.floor(p[0])), int(np.floor(p[1])), int(np.floor(p[2]))
        big = 1e30
        tx, ty, tz = big, big, big
        sx, sy, sz = 0, 0, 0
        if d0 > 0:
            tx, sx = (cx + 1 - p[0]) / d0, 1
        elif d0 < 0:
            tx, sx = (cx - p[0]) / d0, -1
        if d1 > 0:
            ty, sy = (cy + 1 - p[1]) / d1, 1
        elif d1 < 0:
            ty, sy = (cy - p[1]) / d1, -1
        if d2 > 0:
            tz, sz = (cz + 1 - p[2]) / d2, 1
        elif d2 < 0:
            tz, sz = (cz - p[2]) / d2, -1
        while True:
            tn = min(tx, ty, tz)
            if tn >= 1.0:
                break
            if tx <= ty and tx <= tz:
                cx += sx
                tx += sx / d0
            elif ty <= tz:
                cy += sy
                ty += sy / d1
            else:
                cz += sz
                tz += sz / d2
            if cx < 0 or cy < 0 or cz < 0 or cx >= X or cy >= Y or cz >= Z:
                return -1
            if not alive_grid[cx, cy, cz]:
                return (cx * Y + cy) * Z + cz
        p[0] += d0
        p[1] += d1
        p[2] += d2
        if use:
            sv = _stop_val(p[0] - 0.5, p[1] - 0.5, p[2] - 0.5, S)
            if sv <= sthr:
                qx, qy, qz = int(np.floor(p[0])), int(np.floor(p[1])), int(np.floor(p[2]))
                return (qx * Y + qy) * Z + qz
        _pmf_point(p[0] - 0.5, p[1] - 0.5, p[2] - 0.5, R, rowmap, thr, pmf)
        tot = 0.0
        for q in range(cptr[m], cptr[m + 1]):
            tot += pmf[cidx[q]] * w[cidx[q]]
        if tot <= 0:
            return -1
        u = np.random.random() * tot
        acc = 0.0
        mn = cidx[cptr[m + 1] - 1]
        for q in range(cptr[m], cptr[m + 1]):
            acc += pmf[cidx[q]] * w[cidx[q]]
            if u < acc:
                mn = cidx[q]
                break
        m = mn
    return -1


@njit(cache=True)
def _run_bundle_pt(seed, pts, peaks, R, rowmap, thr, alive_grid, w, v, cptr, cidx, rev, inv_zoom, smm, nper, tlab,
                   maxsteps, S, sthr, use, out):
    """Per streamline (seed pts[k], peak peaks[k]): nper samples of two independent halves, each
    starting within the cone of the peak / its reverse (the fODF at the seed), walked to their
    ends; out[k] = the share joining target 1 and target 2."""
    np.random.seed(seed)
    p = np.empty(3)
    M = len(w)
    pmf = np.empty(M)
    p0 = np.empty(M)
    for k in range(len(pts)):
        _pmf_point(pts[k, 0] - 0.5, pts[k, 1] - 0.5, pts[k, 2] - 0.5, R, rowmap, thr, p0)
        hit = 0
        for _ in range(nper):
            e = np.full(2, -1, np.int64)
            for h in range(2):
                c = peaks[k] if h == 0 else rev[peaks[k]]
                tot = 0.0
                for q in range(cptr[c], cptr[c + 1]):
                    tot += p0[cidx[q]] * w[cidx[q]]
                if tot <= 0:
                    break
                u = np.random.random() * tot
                acc = 0.0
                m0 = cidx[cptr[c + 1] - 1]
                for q in range(cptr[c], cptr[c + 1]):
                    acc += p0[cidx[q]] * w[cidx[q]]
                    if u < acc:
                        m0 = cidx[q]
                        break
                for q in range(3):
                    p[q] = pts[k, q]
                x = _walk_pt_stop(p, m0, alive_grid, R, rowmap, thr, w, v, cptr, cidx, inv_zoom, smm, maxsteps, pmf,
                                  S, sthr, use)
                if x >= 0:
                    e[h] = tlab[x]
            if (e[0] == 1 and e[1] == 2) or (e[0] == 2 and e[1] == 1):
                hit += 1
        out[k] = hit / nper


def monte_carlo_seeded_bundle(model, sh, seeds, starts, start, end, nper=200, seed=0, maxsteps=4000, stop_map=None,
                              stop_threshold=0.4):
    """Galerkin.seeded_bundle's expected streamlines joining ``start`` and ``end`` (disjoint masks),
    sampled: each streamline's halves walked one by one, turning by the fODF at their own point,
    and (``stop_map``) stopping as pyAFQ's tracker where it is <= ``stop_threshold``.  Returns
    (expected, its standard error)."""
    from mpat.galerkin import raw_sf
    from mpat.geometry import PMF_THRESHOLD

    shape = model.shape
    R, rowmap = raw_sf(np.asarray(sh), model.ijk, shape, model.dirs)
    alive = np.zeros(shape, bool)
    alive[tuple(model.ijk.T)] = True
    sid, pk = (np.asarray(a, np.int64) for a in starts)
    pts = np.asarray(seeds, np.float64).reshape(-1, 3)[sid] + 0.5      # the walk's frame
    keep = alive[tuple(np.floor(pts).astype(np.int64).T)]
    pts, pk = pts[keep], pk[keep]
    tlab = np.zeros(int(np.prod(shape)), np.int64)
    tlab[np.asarray(start, bool).ravel()] = 1
    tlab[np.asarray(end, bool).ravel()] = 2
    d = model.dirs
    use = stop_map is not None
    S = np.asarray(stop_map if use else np.zeros((1, 1, 1)), np.float64)
    out = np.zeros(len(pts))
    _run_bundle_pt(seed, pts, pk, R.astype(np.float64), rowmap.astype(np.int64), float(PMF_THRESHOLD), alive, d.w,
                   d.v, d.cptr.astype(np.int64), d.cidx.astype(np.int64), d.rev.astype(np.int64), 1.0 / model.zooms,
                   model.smm, nper, tlab, maxsteps, S, float(stop_threshold), use, out)
    return float(out.sum()), float(np.sqrt((out * (1 - out)).sum() / nper))
