"""NVIDIA Warp kernels for mpat.galerkin (imported only there: ``pip install warp-lang``).

The white matter is cut into bins, ``kb`` per voxel side; a bin is ``(voxel, sub)``, ``sub`` =
(sx kb + sy) kb + sz, and its global bin coordinates are ``voxel_ijk kb + (sx, sy, sz)``.  The
walkers in a bin heading one direction are a quadratic density there: 10 coefficients on the
orthonormal Legendre basis of the unit cube (``vec10``; coefficient 0 is the walkers' number).

States live on the fODF's support: one entry ``p`` per (voxel, direction) where the fODF is
not zero (``edir``, ``evox``; ``eidx[m, voxel]`` / ``eidxT[voxel, m]``: the entry, -1 if none)
-- a walker leaves a turn only into such a direction -- with ``kb^3`` bins each: arrays
(entry, sub) of vec10, direction by direction, voxels in Morton order.  What the turns read
(``y``: the walkers just landed, before they turn; ``h``: the importance of landing) lives on
the landings: one entry ``l`` per (voxel, direction) a walker can land in (``ldir``,
``lvox``; ``lidx[voxel, m]``), voxel by voxel, so that a turn finds all it needs of a voxel
together (in cache).  The shifts take their threads direction by direction (neighbouring
threads: neighbouring voxels, the same pieces), the turns voxel by voxel (``lperm``,
``sperm``: those orders).

A step's geometry is per direction (mpat.galerkin.geometry): its pieces ``r`` (``gptr[m]`` ..
``gptr[m + 1]``), each the part of a bin whose walkers visit the same bins in the same order:
``cell[cptr[r]]`` .. ``cell[cptr[r + 1] - 1]``, bin offsets from the source bin (the source
first, the landing bin ``land[r]`` last); ``A[r]`` maps a source bin's coefficients to the
landing bin's (``AT``: its transpose), ``amass[r]`` to the piece's walkers; per cell ``Wc``
(path length x the importance at landing) and ``Bc`` (path length) for the densities.

``vid``: the domain index of each voxel of the volume (-1 outside it); ``alive``: the domain
voxels walkers may be in for this solve.  A piece whose cells all lie in live voxels moves its
walkers; otherwise they leave the white matter into its first dead voxel (counted there, or,
backward, worth ``tgrid`` there).
"""
import warp as wp

wp.set_module_options({"enable_backward": False})     # no autodiff: half the code to compile

vec10 = wp.types.vector(length=10, dtype=wp.float32)
mat10 = wp.types.matrix(shape=(10, 10), dtype=wp.float32)

SQ3 = wp.constant(1.7320508075688772)
SQ5 = wp.constant(2.23606797749979)
BIAS = wp.constant(64)            # bins added before integer division: it never sees a negative
BIG = wp.constant(1.0e30)
LAUNCHED, FORWARD, REACH, DENSITY = 0, 1, 2, 3


@wp.func
def fdiv(a: int, k: int):
    return (a + BIAS * k) // k - BIAS


@wp.func
def vox(b: wp.vec3i, k: int):
    return wp.vec3i(fdiv(b[0], k), fdiv(b[1], k), fdiv(b[2], k))


@wp.func
def look(vid: wp.array3d(dtype=wp.int32), v: wp.vec3i):
    # domain index of voxel v; -1 not in the domain, -2 outside the volume
    if v[0] < 0 or v[1] < 0 or v[2] < 0:
        return -2
    if v[0] >= vid.shape[0] or v[1] >= vid.shape[1] or v[2] >= vid.shape[2]:
        return -2
    return vid[v[0], v[1], v[2]]


@wp.func
def same(a: wp.vec3i, b: wp.vec3i):
    return a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


@wp.func
def sublin(b: wp.vec3i, v: wp.vec3i, k: int):
    # sub index of bin b in its voxel v
    return ((b[0] - v[0] * k) * k + (b[1] - v[1] * k)) * k + (b[2] - v[2] * k)


@wp.func
def f3(a: wp.vec3i):
    return wp.vec3(float(a[0]), float(a[1]), float(a[2]))


@wp.func
def subvec(s: int, k: int):
    return wp.vec3i(s // (k * k), (s // k) % k, s % k)


@wp.func
def basis(y: wp.vec3):
    # the orthonormal quadratics of the unit cube at y
    x1 = SQ3 * (2.0 * y[0] - 1.0)
    y1 = SQ3 * (2.0 * y[1] - 1.0)
    z1 = SQ3 * (2.0 * y[2] - 1.0)
    x2 = SQ5 * (6.0 * y[0] * y[0] - 6.0 * y[0] + 1.0)
    y2 = SQ5 * (6.0 * y[1] * y[1] - 6.0 * y[1] + 1.0)
    z2 = SQ5 * (6.0 * y[2] * y[2] - 6.0 * y[2] + 1.0)
    return vec10(1.0, x1, y1, z1, x2, y2, z2, x1 * y1, x1 * z1, y1 * z1)


@wp.func
def fate(src: wp.vec3i, r: int, cptr: wp.array(dtype=wp.int32), cell: wp.array(dtype=wp.vec3i), kb: int,
         vid: wp.array3d(dtype=wp.int32), alive: wp.array(dtype=wp.uint8)):
    # piece r from the bin src (in a live voxel): (index of its first cell in a dead voxel, or
    # cptr[r + 1] if none; the domain index of that voxel or of the landing voxel)
    prev = vox(src, kb)
    lc = int(-1)
    for i in range(cptr[r] + 1, cptr[r + 1]):
        v = vox(src + cell[i], kb)
        if not same(v, prev):
            prev = v
            c = look(vid, v)
            if c < 0:
                return wp.vec2i(i, c)
            if alive[c] == wp.uint8(0):
                return wp.vec2i(i, c)
            lc = c
    if lc < 0:
        lc = look(vid, prev)
    return wp.vec2i(cptr[r + 1], lc)


@wp.func
def tval(tgrid: wp.array3d(dtype=wp.float32), v: wp.vec3i):
    # what a walker leaving the white matter into voxel v is worth (0 outside the volume)
    if v[0] < 0 or v[1] < 0 or v[2] < 0:
        return 0.0
    if v[0] >= tgrid.shape[0] or v[1] >= tgrid.shape[1] or v[2] >= tgrid.shape[2]:
        return 0.0
    return tgrid[v[0], v[1], v[2]]


# ------------------------------------------------------------------ one step, forward
@wp.kernel
def fwd_shift(lperm: wp.array(dtype=wp.int32), ldir: wp.array(dtype=wp.int32), lvox: wp.array(dtype=wp.int32),
              lzi: wp.array2d(dtype=wp.float32), ib: int, x: wp.array2d(dtype=vec10),
              eidx: wp.array2d(dtype=wp.int32), vid: wp.array3d(dtype=wp.int32), alive: wp.array(dtype=wp.uint8),
              vijk: wp.array(dtype=wp.vec3i), gptr: wp.array(dtype=wp.int32), land: wp.array(dtype=wp.vec3i),
              A: wp.array(dtype=mat10), cptr: wp.array(dtype=wp.int32), cell: wp.array(dtype=wp.vec3i), kb: int,
              y: wp.array2d(dtype=vec10)):
    # y(l, sub) = zi(l) sum over pieces r of l's direction m: A_r x(bin - land_r, m), if all of r's
    # cells from there are live: the walkers landing there heading m (gathered: no atomics),
    # divided by their cone's fODF mass (the turn's normaliser; per bin with ib = 1, else per voxel)
    t, s = wp.tid()
    l = lperm[t]
    m = ldir[l]
    cv = lvox[l]
    out = vec10()
    if alive[cv] == wp.uint8(0):
        y[l, s] = out
        return
    L = vijk[cv] * kb + subvec(s, kb)
    for r in range(gptr[m], gptr[m + 1]):
        src = L - land[r]
        sv = vox(src, kb)
        cs = look(vid, sv)
        if cs < 0:
            continue
        if alive[cs] == wp.uint8(0):
            continue
        p = eidx[m, cs]
        if p < 0:
            continue
        xv = x[p, sublin(src, sv, kb)]
        if wp.length_sq(xv) == 0.0:
            continue
        ok = int(1)
        prev = sv
        for i in range(cptr[r] + 1, cptr[r + 1] - 1):
            v = vox(src + cell[i], kb)
            if not same(v, prev):
                prev = v
                c = look(vid, v)
                if c < 0:
                    ok = 0
                    break
                if alive[c] == wp.uint8(0):
                    ok = 0
                    break
        if ok == 1:
            out = out + A[r] @ xv
    y[l, s] = lzi[l, s * ib] * out


@wp.kernel
def accum(xn: wp.array2d(dtype=vec10), C: wp.array2d(dtype=vec10), k3: int, ntot: int, nblk: int, slot: int,
          red: wp.array(dtype=wp.float64)):
    # C += xn; red[slot] += sum |walkers| of xn (bin by bin), in nblk strided blocks
    i0 = wp.tid()
    acc = wp.float64(0.0)
    for i in range(i0, ntot, nblk):
        p = i // k3
        s = i - p * k3
        v = xn[p, s]
        C[p, s] = C[p, s] + v
        acc += wp.float64(wp.abs(v[0]))
    wp.atomic_add(red, slot, acc)


@wp.kernel
def mass(xn: wp.array2d(dtype=vec10), k3: int, ntot: int, nblk: int, slot: int, red: wp.array(dtype=wp.float64)):
    # red[slot] += sum |walkers| of xn (accum without the sum over the steps)
    i0 = wp.tid()
    acc = wp.float64(0.0)
    for i in range(i0, ntot, nblk):
        p = i // k3
        s = i - p * k3
        acc += wp.float64(wp.abs(xn[p, s][0]))
    wp.atomic_add(red, slot, acc)


@wp.kernel
def subnormals(x: wp.array2d(dtype=vec10), k3: int, ntot: int, nblk: int, red: wp.array(dtype=wp.float64)):
    # red[0] += the nonzero coefficients of x, red[1] += those below the smallest normal float
    i0 = wp.tid()
    nz = wp.float64(0.0)
    sb = wp.float64(0.0)
    for i in range(i0, ntot, nblk):
        p = i // k3
        s = i - p * k3
        v = x[p, s]
        for j in range(10):
            a = wp.abs(v[j])
            if a > 0.0:
                nz += wp.float64(1.0)
                if a < 1.17549435e-38:
                    sb += wp.float64(1.0)
    wp.atomic_add(red, 0, nz)
    wp.atomic_add(red, 1, sb)


@wp.kernel
def connect(x: wp.array2d(dtype=vec10), g: wp.array2d(dtype=vec10), k3: int, ntot: int, nblk: int, thr: float,
            red: wp.array(dtype=wp.float64)):
    # per bin c = x . g, the walkers' expected arrivals (orthonormal coefficients: the integral of
    # x g); |c| < thr (thr > 0): the bin's walkers dropped, red[2] += c, red[3] += |c|; else red[0]
    # += c, red[1] += |c|
    i0 = wp.tid()
    kept = wp.float64(0.0)
    kabs = wp.float64(0.0)
    drop = wp.float64(0.0)
    dabs = wp.float64(0.0)
    for i in range(i0, ntot, nblk):
        p = i // k3
        s = i - p * k3
        v = x[p, s]
        c = wp.float64(0.0)
        gv = g[p, s]
        for j in range(10):
            c += wp.float64(v[j]) * wp.float64(gv[j])
        if thr > 0.0 and wp.abs(c) < wp.float64(thr):
            x[p, s] = vec10()
            drop += c
            dabs += wp.abs(c)
        else:
            kept += c
            kabs += wp.abs(c)
    wp.atomic_add(red, 0, kept)
    wp.atomic_add(red, 1, kabs)
    wp.atomic_add(red, 2, drop)
    wp.atomic_add(red, 3, dabs)


@wp.kernel
def norm_accum(x: wp.array2d(dtype=vec10), vn: wp.array2d(dtype=wp.float32)):
    # vn += |x| bin by bin (the norm of the coefficients: |integral of x h| <= |x| |h|)
    p, s = wp.tid()
    vn[p, s] = vn[p, s] + wp.length(x[p, s])


@wp.kernel
def bwd_prune(g: wp.array2d(dtype=vec10), vn: wp.array2d(dtype=wp.float32), k3: int, ntot: int, nblk: int,
              thr: float, red: wp.array(dtype=wp.float64)):
    # a bin's importance dropped where |g| vn (vn: a forward's visits there, in norm) < thr, red[0] += it
    i0 = wp.tid()
    drop = wp.float64(0.0)
    for i in range(i0, ntot, nblk):
        p = i // k3
        s = i - p * k3
        c = wp.float64(wp.length(g[p, s])) * wp.float64(vn[p, s])
        if c < wp.float64(thr):
            if c > wp.float64(0.0):
                drop += c
            g[p, s] = vec10()
    wp.atomic_add(red, 0, drop)


# ------------------------------------------------------------------ one step, backward (the transpose)
@wp.kernel
def bwd_shift(edir: wp.array(dtype=wp.int32), evox: wp.array(dtype=wp.int32), vijk: wp.array(dtype=wp.vec3i),
              vid: wp.array3d(dtype=wp.int32), alive: wp.array(dtype=wp.uint8), tgrid: wp.array3d(dtype=wp.float32),
              gptr: wp.array(dtype=wp.int32), land: wp.array(dtype=wp.vec3i), AT: wp.array(dtype=mat10),
              amass: wp.array(dtype=vec10), cptr: wp.array(dtype=wp.int32), cell: wp.array(dtype=wp.vec3i), kb: int,
              lidx: wp.array2d(dtype=wp.int32), h: wp.array2d(dtype=vec10), gn: wp.array2d(dtype=vec10)):
    # gn(p) = sum over p's direction's pieces: A_r^T h(landing bin), or, where the piece leaves
    # the white matter, tgrid(its first dead voxel) times its walkers
    p, s = wp.tid()
    m = edir[p]
    cv = evox[p]
    out = vec10()
    if alive[cv] != wp.uint8(0):
        src = vijk[cv] * kb + subvec(s, kb)
        for r in range(gptr[m], gptr[m + 1]):
            fr = fate(src, r, cptr, cell, kb, vid, alive)
            if fr[0] < cptr[r + 1]:
                tv = tval(tgrid, vox(src + cell[fr[0]], kb))
                if tv != 0.0:
                    out = out + tv * amass[r]
            else:
                l = lidx[fr[1], m]
                if l >= 0:
                    lb = src + land[r]
                    out = out + AT[r] @ h[l, sublin(lb, vox(lb, kb), kb)]
    gn[p, s] = out


# ------------------------------------------------------------------ the fODF at points, on the device
# (which directions have fODF at the points, and the bins' weights: one point at a time, for every
# voxel at once -- galerkin.point_fodf, there on the host)
@wp.kernel
def pt_interp(ijk: wp.array(dtype=wp.vec3i), off: wp.vec3, rows: wp.array3d(dtype=wp.int32),
              sf: wp.array2d(dtype=wp.float32), fd: wp.array2d(dtype=wp.float32)):
    # fd(v, m) = the clipped fODF at voxel v's point (its centre + off, voxel centres at integers),
    # trilinear from the 8 voxel centres around it (a voxel with no row counts as zero)
    v, m = wp.tid()
    p = f3(ijk[v]) + off
    lx = int(wp.floor(p[0]))
    ly = int(wp.floor(p[1]))
    lz = int(wp.floor(p[2]))
    fx = p[0] - float(lx)
    fy = p[1] - float(ly)
    fz = p[2] - float(lz)
    acc = float(0.0)
    for c in range(8):
        cx = (c >> 2) & 1
        cy = (c >> 1) & 1
        cz = c & 1
        wx = 1.0 - fx
        if cx == 1:
            wx = fx
        wy = 1.0 - fy
        if cy == 1:
            wy = fy
        wz = 1.0 - fz
        if cz == 1:
            wz = fz
        wt = wx * wy * wz
        ix = lx + cx
        iy = ly + cy
        iz = lz + cz
        if wt > 0.0 and ix >= 0 and iy >= 0 and iz >= 0 and ix < rows.shape[0] and iy < rows.shape[1] \
                and iz < rows.shape[2]:
            r = rows[ix, iy, iz]
            if r >= 0:
                acc += wt * sf[r, m]
    fd[v, m] = acc


@wp.kernel
def pt_norm(fd: wp.array2d(dtype=wp.float32), w: wp.array(dtype=wp.float32), thr: float, M: int):
    # as fodf_on: values below thr x the maximum zeroed, normalised to integrate to 1, times w
    v = wp.tid()
    mx = float(0.0)
    for m in range(M):
        mx = wp.max(mx, fd[v, m])
    cut = thr * mx
    tot = float(0.0)
    for m in range(M):
        x = fd[v, m]
        if x < cut:
            x = 0.0
            fd[v, m] = 0.0
        tot += x * w[m]
    for m in range(M):
        x = float(0.0)
        if tot > 0.0:
            x = fd[v, m] / tot * w[m]
        fd[v, m] = x


@wp.kernel
def pt_flag(fd: wp.array2d(dtype=wp.float32), flags: wp.array2d(dtype=wp.uint8)):
    v, m = wp.tid()
    if fd[v, m] > 0.0:
        flags[v, m] = wp.uint8(1)


@wp.kernel
def pt_state(evox: wp.array(dtype=wp.int32), edir: wp.array(dtype=wp.int32), fd: wp.array2d(dtype=wp.float32),
             efw: wp.array2d(dtype=wp.float32), b: int, qw: float):
    # each state's f w at the point, times qw, into efw[:, b] (the bin's mean)
    p = wp.tid()
    efw[p, b] = efw[p, b] + qw * fd[evox[p], edir[p]]


@wp.kernel
def pt_landing(lvox: wp.array(dtype=wp.int32), ldir: wp.array(dtype=wp.int32), fd: wp.array2d(dtype=wp.float32),
               conep: wp.array(dtype=wp.int32), conei: wp.array(dtype=wp.int32), tab: wp.array2d(dtype=wp.float32),
               col: int):
    # each landing's 1 / (its cone's f w at the point), 0 if none
    l = wp.tid()
    v = lvox[l]
    m = ldir[l]
    z = float(0.0)
    for c in range(conep[m], conep[m + 1]):
        z += fd[v, conei[c]]
    x = float(0.0)
    if z > 0.0:
        x = 1.0 / z
    tab[l, col] = x


# ------------------------------------------------------------------ pictures of the solves
@wp.kernel
def voxel_sum(x: wp.array2d(dtype=vec10), evox: wp.array(dtype=wp.int32), w: wp.array2d(dtype=wp.float32),
              ib: int, out: wp.array(dtype=wp.float32)):
    # out(voxel) += w(p) x(p, sub)[0]: per voxel, the walkers (w = 1) or the fODF-weighted
    # importance (w = f w), over its directions and bins
    p, s = wp.tid()
    v = x[p, s][0] * w[p, s * ib]
    if v != 0.0:
        wp.atomic_add(out, evox[p], v)


# ------------------------------------------------------------------ the turns in shared memory
_TURNS = {}
TURN_BLOCK = 64         # threads per voxel (and bin) in the turns


def turn_kernels(M):
    """The turns for M directions, one thread block per bin: the bin's landings (forward) or its
    states (backward) gathered once into a dense (M, 10) tile in shared memory (missing ones
    read as zero), then every thread sums its cones from there, with no lookups.  M must be a
    compile-time constant, so the kernels are made (and cached) per sphere size."""
    if M in _TURNS:
        return _TURNS[M]

    @wp.kernel(module="unique")
    def fwd_turn_tiled(lidx: wp.array2d(dtype=wp.int32), yf: wp.array2d(dtype=wp.float32),
                       vptr: wp.array(dtype=wp.int32), sperm: wp.array(dtype=wp.int32), edir: wp.array(dtype=wp.int32),
                       efw: wp.array2d(dtype=wp.float32), ib: int, conep: wp.array(dtype=wp.int32),
                       conei: wp.array(dtype=wp.int32), xn: wp.array2d(dtype=vec10)):
        # xn(p) = f(p) w(p) sum over the cone of p's direction of y(voxel, m): the walkers turning
        # into p's direction; yf: the landings as floats (landing, sub x 10)
        v, s, lane = wp.tid()
        ind = wp.tile_load(lidx[v], shape=M)
        Y = wp.tile_load_indexed(yf, ind, shape=(M, 10), offset=(0, s * 10), axis=0, storage="shared")
        for t in range(vptr[v] + lane, vptr[v + 1], wp.block_dim()):
            p = sperm[t]
            mp = edir[p]
            acc = vec10()
            for q in range(conep[mp], conep[mp + 1]):
                i = conei[q]
                acc = acc + vec10(Y[i, 0], Y[i, 1], Y[i, 2], Y[i, 3], Y[i, 4], Y[i, 5], Y[i, 6], Y[i, 7], Y[i, 8],
                                  Y[i, 9])
            xn[p, s] = efw[p, s * ib] * acc

    @wp.kernel(module="unique")
    def bwd_turn_tiled(eidxT: wp.array2d(dtype=wp.int32), gf: wp.array2d(dtype=wp.float32),
                       efw: wp.array2d(dtype=wp.float32), ib: int, alive: wp.array(dtype=wp.uint8),
                       lptr: wp.array(dtype=wp.int32), ldir: wp.array(dtype=wp.int32),
                       lzi: wp.array2d(dtype=wp.float32),
                       conep: wp.array(dtype=wp.int32), conei: wp.array(dtype=wp.int32), h: wp.array2d(dtype=vec10)):
        # h(l) = zi(l) sum over the cone of l's direction of f(m') w(m') g(voxel, m'): the
        # importance of landing (gf: the states as floats; efw: f w per state, one column, or per bin)
        v, s, lane = wp.tid()
        ind = wp.tile_load(eidxT[v], shape=M)
        G = wp.tile_load_indexed(gf, ind, shape=(M, 10), offset=(0, s * 10), axis=0, storage="shared")
        Fw = wp.tile_load_indexed(efw, ind, shape=(M, 1), offset=(0, s * ib), axis=0, storage="shared")
        live = alive[v] != wp.uint8(0)
        for l in range(lptr[v] + lane, lptr[v + 1], wp.block_dim()):
            out = vec10()
            if live:
                m = ldir[l]
                for q in range(conep[m], conep[m + 1]):
                    i = conei[q]
                    fw = Fw[i, 0]
                    if fw != 0.0:
                        out = out + fw * vec10(G[i, 0], G[i, 1], G[i, 2], G[i, 3], G[i, 4], G[i, 5], G[i, 6],
                                               G[i, 7], G[i, 8], G[i, 9])
                out = lzi[l, s * ib] * out
            h[l, s] = out

    _TURNS[M] = (fwd_turn_tiled, bwd_turn_tiled)
    return _TURNS[M]


@wp.func
def _corner(rows: wp.array3d(dtype=wp.int32), sf: wp.array2d(dtype=wp.float32), ix: int, iy: int, iz: int, m: int,
            M: int):
    # the clipped fODF of voxel (ix, iy, iz) in direction m (0 off the grid, outside the rows, or m >= M)
    val = float(0.0)
    if m < M and ix >= 0 and iy >= 0 and iz >= 0 and ix < rows.shape[0] and iy < rows.shape[1] \
            and iz < rows.shape[2]:
        r = rows[ix, iy, iz]
        if r >= 0:
            val = sf[r, m]
    return val


def fused_block(M):
    """Threads per bin of the turns at points: one per direction, padded to a whole warp."""
    return ((M + 31) // 32) * 32


FUSED_MAX_DIRS = 1024   # the turns at points: a thread per direction, at most 1024 in a CUDA block


def fused_regs(block, regs_per_sm=65536, blocks=2):
    """The register cap that lets ``blocks`` blocks of ``block`` threads share an SM (64K registers
    on every recent NVIDIA GPU; allocated in steps of 8 per thread), or 0 (no cap needed: 255 or
    more).  symmetric362 (384 threads): 80, measured fastest on an RTX 5080 (Laptop): 5.9 / 5.1 ms
    per forward / backward turn on ISMRM's ILF, against 7.6 / 6.5 uncapped (96 registers) and 6.3
    / 5.3 at 64."""
    cap = (regs_per_sm // (blocks * block)) // 8 * 8
    return 0 if cap >= 255 else int(cap)


def fused_point_kernels(M):
    """The turns at points, a thread block per bin:
        xn(p)_j = sum_q wq(q) fq(p, q) phi_j(u_q) sum_{l in the cone of p's direction} zq(l, q) (y(l) . phi(u_q))
    -- each part of a bin's walkers turns by the fODF where it is (fq: f w at Gauss point q, the
    clipped fODF trilinear from the 8 voxel centres around it, thresholded and normalised as
    galerkin.point_fodf; zq: 1 / its cone's f w there), projected back onto the quadratics.
    Backward, its exact transpose:
        h(l)_i = sum_q zq(l, q) phi_i(u_q) sum_{p in the cone of l's direction} wq(q) fq(p, q) (g(p) . phi(u_q))
    The fODF at the points is computed in the turn (no tables).  With ``use_stop``, a point flagged
    in ``stopq`` (Nv, kb^3 Q) is where walkers stop (pyAFQ's interpolated stopping): forward, the
    walkers landed there end in its voxel (added to ``ends``) and do not turn; backward, landing
    there is worth tgrid of its voxel.  Laid out for latency: a thread per direction (block ``fused_block(M)``) keeps in its
    registers what is fixed over the bin -- its landing and state entries, w, its cone's range,
    the walkers that landed (forward) or the importance of its state (backward), its 8 corner
    fODF values and its accumulated coefficients -- so that the loop over the points reads only
    the two shared tiles the cone sums need (the fODF at the point, and the walkers per landing)
    and the cone indices (uint16).  Shared memory: 2 floats per direction (was 10 floats and a
    vec10).  The active set: a bin with nothing to turn (no walkers landed; backward, states worth
    nothing and no stop-point source) writes zeros and returns -- exact (zero in, zero out), and
    most of the domain early in a solve launched from ROIs.  Every shared write is made by every thread (a block-wide sync follows it).
    Their registers per thread are capped so that two blocks share an SM (fused_regs: 80 at 384
    threads); MPAT_GALERKIN_FUSED_REGS=N sets another cap (0: the compiler's choice)."""
    import os

    MP = fused_block(M)
    regs = os.environ.get("MPAT_GALERKIN_FUSED_REGS")
    regs = fused_regs(MP) if regs is None else int(regs or 0)          # 0: the compiler's choice
    key = ("fused", M, regs)
    if key in _TURNS:
        return _TURNS[key]
    opts = {"cuda_max_registers": regs} if regs > 0 else {}

    @wp.kernel(module="unique", **opts)
    def fwd_fused(vijk: wp.array(dtype=wp.vec3i), rows: wp.array3d(dtype=wp.int32), sf: wp.array2d(dtype=wp.float32),
                  w: wp.array(dtype=wp.float32), thr: float, qorder: wp.array(dtype=wp.int32),
                  qoff: wp.array(dtype=wp.vec3i), qfrac: wp.array(dtype=wp.vec3), Q: int,
                  wq: wp.array(dtype=wp.float32), phi: wp.array(dtype=vec10), lidx: wp.array2d(dtype=wp.int32),
                  eidxT: wp.array2d(dtype=wp.int32), conep: wp.array(dtype=wp.int32), conei: wp.array(dtype=wp.uint16),
                  y: wp.array2d(dtype=vec10), stopq: wp.array2d(dtype=wp.uint8), use_stop: int,
                  ends: wp.array(dtype=wp.float64), xn: wp.array2d(dtype=vec10)):
        v, s, m = wp.tid()
        F = wp.tile_zeros(shape=MP, dtype=wp.float32, storage="shared")
        A = wp.tile_zeros(shape=MP, dtype=wp.float32, storage="shared")
        SY = wp.tile_zeros(shape=1, dtype=wp.float32, storage="shared")
        vi = vijk[v]
        # fixed over the bin: this thread's direction m
        l = int(-1)
        p = int(-1)
        wm = float(0.0)
        q0 = int(0)
        q1 = int(0)
        if m < M:
            l = lidx[v, m]
            p = eidxT[v, m]
            wm = w[m]
            q0 = conep[m]
            q1 = conep[m + 1]
        yv = vec10()
        if l >= 0:
            yv = y[l, s]
        # the active set: a bin no walker has landed in (all exactly zero) turns none -- its states get 0
        F[m] = wp.length_sq(yv)
        if wp.tile_sum(F)[0] == 0.0:
            if p >= 0:
                xn[p, s] = vec10()
            return
        out = vec10()
        c0 = float(0.0)
        c1 = float(0.0)
        c2 = float(0.0)
        c3 = float(0.0)
        c4 = float(0.0)
        c5 = float(0.0)
        c6 = float(0.0)
        c7 = float(0.0)
        px = int(-9)
        py = int(-9)
        pz = int(-9)
        stopped = float(0.0)
        for t in range(Q):
            SY[0] = 0.0         # a barrier (Warp syncs after a shared write): the last point's cone sums are done
            k = qorder[s * Q + t]
            st = int(0)
            if use_stop != 0:
                st = int(stopq[v, s * Q + k])
            if st != 0:             # the walkers landed at this point stop here (they end in this voxel)
                a = float(0.0)
                if l >= 0:
                    a = wq[k] * wp.dot(yv, phi[k])
                A[m] = a
                stopped += wp.tile_sum(A)[0]
                continue
            off = qoff[s * Q + k]
            if off[0] != px or off[1] != py or off[2] != pz:        # the 8 corners of these points
                px = off[0]
                py = off[1]
                pz = off[2]
                bx = vi[0] + px
                by = vi[1] + py
                bz = vi[2] + pz
                c0 = _corner(rows, sf, bx, by, bz, m, M)
                c1 = _corner(rows, sf, bx, by, bz + 1, m, M)
                c2 = _corner(rows, sf, bx, by + 1, bz, m, M)
                c3 = _corner(rows, sf, bx, by + 1, bz + 1, m, M)
                c4 = _corner(rows, sf, bx + 1, by, bz, m, M)
                c5 = _corner(rows, sf, bx + 1, by, bz + 1, m, M)
                c6 = _corner(rows, sf, bx + 1, by + 1, bz, m, M)
                c7 = _corner(rows, sf, bx + 1, by + 1, bz + 1, m, M)
            fr = qfrac[s * Q + k]
            gx = 1.0 - fr[0]
            gy = 1.0 - fr[1]
            gz = 1.0 - fr[2]
            val = (gx * (gy * (gz * c0 + fr[2] * c1) + fr[1] * (gz * c2 + fr[2] * c3))
                   + fr[0] * (gy * (gz * c4 + fr[2] * c5) + fr[1] * (gz * c6 + fr[2] * c7)))
            F[m] = val                                                          # the fODF at the point
            cut = thr * wp.tile_max(F)[0]
            x = val
            if x < cut:
                x = 0.0
            fw = x * wm                                                         # thresholded, times w
            A[m] = fw
            tot = wp.tile_sum(A)[0]
            if tot > 0.0:
                f = fw / tot
                F[m] = f                                                        # f w, normalised
                a = float(0.0)                                                  # each landing: its walkers there
                if l >= 0:                                                      # / its cone's f w
                    z = float(0.0)
                    for q in range(q0, q1):
                        z += F[int(conei[q])]
                    if z > 0.0:
                        a = wp.dot(yv, phi[k]) / z
                A[m] = a
                if p >= 0 and f != 0.0:                                         # each state: f w x its cone's walkers
                    acc = float(0.0)
                    for q in range(q0, q1):
                        acc += A[int(conei[q])]
                    out = out + (wq[k] * f * acc) * phi[k]
        if p >= 0:
            xn[p, s] = out
        if m == 0 and stopped != 0.0:
            wp.atomic_add(ends, v, wp.float64(stopped))

    @wp.kernel(module="unique", **opts)
    def bwd_fused(vijk: wp.array(dtype=wp.vec3i), rows: wp.array3d(dtype=wp.int32), sf: wp.array2d(dtype=wp.float32),
                  w: wp.array(dtype=wp.float32), thr: float, qorder: wp.array(dtype=wp.int32),
                  qoff: wp.array(dtype=wp.vec3i), qfrac: wp.array(dtype=wp.vec3), Q: int,
                  wq: wp.array(dtype=wp.float32), phi: wp.array(dtype=vec10), lidx: wp.array2d(dtype=wp.int32),
                  eidxT: wp.array2d(dtype=wp.int32), conep: wp.array(dtype=wp.int32), conei: wp.array(dtype=wp.uint16),
                  alive: wp.array(dtype=wp.uint8), g: wp.array2d(dtype=vec10), stopq: wp.array2d(dtype=wp.uint8),
                  use_stop: int, tgrid: wp.array3d(dtype=wp.float32), h: wp.array2d(dtype=vec10)):
        v, s, m = wp.tid()
        F = wp.tile_zeros(shape=MP, dtype=wp.float32, storage="shared")
        B = wp.tile_zeros(shape=MP, dtype=wp.float32, storage="shared")
        SY = wp.tile_zeros(shape=1, dtype=wp.float32, storage="shared")
        vi = vijk[v]
        l = int(-1)
        p = int(-1)
        wm = float(0.0)
        q0 = int(0)
        q1 = int(0)
        if m < M:
            l = lidx[v, m]
            p = eidxT[v, m]
            wm = w[m]
            q0 = conep[m]
            q1 = conep[m + 1]
        gv = vec10()
        if p >= 0:
            gv = g[p, s]
        tv0 = float(0.0)
        if use_stop != 0:
            tv0 = tgrid[vi[0], vi[1], vi[2]]
        # the active set: a bin whose states are worth exactly nothing, with no stop-point source
        # (tgrid 0 here), gives its landings 0
        F[m] = wp.length_sq(gv)
        if wp.tile_sum(F)[0] == 0.0 and tv0 == 0.0:
            if l >= 0:
                h[l, s] = vec10()
            return
        out = vec10()
        c0 = float(0.0)
        c1 = float(0.0)
        c2 = float(0.0)
        c3 = float(0.0)
        c4 = float(0.0)
        c5 = float(0.0)
        c6 = float(0.0)
        c7 = float(0.0)
        px = int(-9)
        py = int(-9)
        pz = int(-9)
        nq = int(0)
        if alive[v] != wp.uint8(0):
            nq = Q
        tv = float(0.0)
        if use_stop != 0:
            tv = tgrid[vi[0], vi[1], vi[2]]
        for t in range(nq):
            SY[0] = 0.0         # a barrier: the last point's cone sums are done before F and B are rewritten
            k = qorder[s * Q + t]
            st = int(0)
            if use_stop != 0:
                st = int(stopq[v, s * Q + k])
            if st != 0:             # landing at this point: the walker ends here, worth tgrid of this voxel
                if l >= 0:
                    out = out + (wq[k] * tv) * phi[k]
                continue
            off = qoff[s * Q + k]
            if off[0] != px or off[1] != py or off[2] != pz:
                px = off[0]
                py = off[1]
                pz = off[2]
                bx = vi[0] + px
                by = vi[1] + py
                bz = vi[2] + pz
                c0 = _corner(rows, sf, bx, by, bz, m, M)
                c1 = _corner(rows, sf, bx, by, bz + 1, m, M)
                c2 = _corner(rows, sf, bx, by + 1, bz, m, M)
                c3 = _corner(rows, sf, bx, by + 1, bz + 1, m, M)
                c4 = _corner(rows, sf, bx + 1, by, bz, m, M)
                c5 = _corner(rows, sf, bx + 1, by, bz + 1, m, M)
                c6 = _corner(rows, sf, bx + 1, by + 1, bz, m, M)
                c7 = _corner(rows, sf, bx + 1, by + 1, bz + 1, m, M)
            fr = qfrac[s * Q + k]
            gx = 1.0 - fr[0]
            gy = 1.0 - fr[1]
            gz = 1.0 - fr[2]
            val = (gx * (gy * (gz * c0 + fr[2] * c1) + fr[1] * (gz * c2 + fr[2] * c3))
                   + fr[0] * (gy * (gz * c4 + fr[2] * c5) + fr[1] * (gz * c6 + fr[2] * c7)))
            F[m] = val
            cut = thr * wp.tile_max(F)[0]
            x = val
            if x < cut:
                x = 0.0
            fw = x * wm
            B[m] = fw
            tot = wp.tile_sum(B)[0]
            if tot > 0.0:
                f = fw / tot
                F[m] = f
                b = float(0.0)                                                  # each state: wq f w (g . phi)
                if p >= 0 and f != 0.0:
                    b = wq[k] * f * wp.dot(gv, phi[k])
                B[m] = b
                if l >= 0:                                                      # each landing: its cone's,
                    z = float(0.0)                                              # over its cone's f w
                    acc = float(0.0)
                    for q in range(q0, q1):
                        i = int(conei[q])
                        z += F[i]
                        acc += B[i]
                    if z > 0.0:
                        out = out + (acc / z) * phi[k]
        if l >= 0:
            h[l, s] = out

    _TURNS[key] = (fwd_fused, bwd_fused)
    return _TURNS[key]


# ------------------------------------------------------------------ tallies, once per solve
@wp.kernel
def exit_tally(C: wp.array2d(dtype=vec10), edir: wp.array(dtype=wp.int32), evox: wp.array(dtype=wp.int32),
               vijk: wp.array(dtype=wp.vec3i), vid: wp.array3d(dtype=wp.int32), alive: wp.array(dtype=wp.uint8),
               gptr: wp.array(dtype=wp.int32), amass: wp.array(dtype=vec10), cptr: wp.array(dtype=wp.int32),
               cell: wp.array(dtype=wp.vec3i), kb: int, out: wp.array3d(dtype=wp.float64)):
    # the walkers of C (summed over the steps) leaving the white matter, into each voxel
    p, s = wp.tid()
    cv = evox[p]
    if alive[cv] == wp.uint8(0):
        return
    cvec = C[p, s]
    if wp.length_sq(cvec) == 0.0:
        return
    m = edir[p]
    src = vijk[cv] * kb + subvec(s, kb)
    for r in range(gptr[m], gptr[m + 1]):
        fr = fate(src, r, cptr, cell, kb, vid, alive)
        if fr[0] < cptr[r + 1] and fr[1] != -2:
            v = vox(src + cell[fr[0]], kb)
            wp.atomic_add(out, v[0], v[1], v[2], wp.float64(wp.dot(amass[r], cvec)))


@wp.kernel
def density(edir: wp.array(dtype=wp.int32), evox: wp.array(dtype=wp.int32), vijk: wp.array(dtype=wp.vec3i),
            vid: wp.array3d(dtype=wp.int32), alive: wp.array(dtype=wp.uint8), tgrid: wp.array3d(dtype=wp.float32),
            gptr: wp.array(dtype=wp.int32), land: wp.array(dtype=wp.vec3i), cptr: wp.array(dtype=wp.int32),
            cell: wp.array(dtype=wp.vec3i), Wc: wp.array(dtype=mat10), Bc: wp.array(dtype=vec10), kb: int,
            lidx: wp.array2d(dtype=wp.int32), C: wp.array2d(dtype=vec10), h: wp.array2d(dtype=vec10), sign: float,
            rho: wp.array2d(dtype=wp.float32)):
    # rho(m, voxel) += sign x path length there of C's walkers (heading m) times their
    # importance after the step: at landing (h) or where they leave (tgrid)
    p, s = wp.tid()
    cv = evox[p]
    if alive[cv] == wp.uint8(0):
        return
    cvec = C[p, s]
    if wp.length_sq(cvec) == 0.0:
        return
    m = edir[p]
    src = vijk[cv] * kb + subvec(s, kb)
    for r in range(gptr[m], gptr[m + 1]):
        fr = fate(src, r, cptr, cell, kb, vid, alive)
        if fr[0] == cptr[r + 1]:
            l = lidx[fr[1], m]
            if l < 0:
                continue
            lb = src + land[r]
            hv = h[l, sublin(lb, vox(lb, kb), kb)]
            if wp.length_sq(hv) != 0.0:
                for i in range(cptr[r], cptr[r + 1]):
                    c = look(vid, vox(src + cell[i], kb))
                    wp.atomic_add(rho, m, c, sign * wp.dot(cvec, Wc[i] @ hv))
        else:
            tv = tval(tgrid, vox(src + cell[fr[0]], kb))
            if tv != 0.0:
                for i in range(cptr[r], fr[0]):
                    c = look(vid, vox(src + cell[i], kb))
                    wp.atomic_add(rho, m, c, sign * tv * wp.dot(Bc[i], cvec))


# ------------------------------------------------------------------ launches
@wp.func
def trace(P: wp.vec3, B0: wp.vec3i, d: wp.vec3, kb: int, vid: wp.array3d(dtype=wp.int32),
          alive: wp.array(dtype=wp.uint8), dep: float, rho: wp.array2d(dtype=wp.float32), m: int):
    # a step from the point P (bin coordinates) in bin B0 along d: (landing bin, 0), or (the
    # first dead voxel, 1), or (that voxel, 2) outside the volume.  dep != 0: adds dep x the
    # path length (in units of the step) in each live voxel to rho(m, voxel)
    cx = B0[0]
    cy = B0[1]
    cz = B0[2]
    tx = float(BIG)
    ty = float(BIG)
    tz = float(BIG)
    sx = int(0)
    sy = int(0)
    sz = int(0)
    if d[0] > 0.0:
        tx = (float(cx + 1) - P[0]) / d[0]
        sx = 1
    elif d[0] < 0.0:
        tx = (float(cx) - P[0]) / d[0]
        sx = -1
    if d[1] > 0.0:
        ty = (float(cy + 1) - P[1]) / d[1]
        sy = 1
    elif d[1] < 0.0:
        ty = (float(cy) - P[1]) / d[1]
        sy = -1
    if d[2] > 0.0:
        tz = (float(cz + 1) - P[2]) / d[2]
        sz = 1
    elif d[2] < 0.0:
        tz = (float(cz) - P[2]) / d[2]
        sz = -1
    v = vox(B0, kb)
    lc = look(vid, v)
    tc = float(0.0)
    for _it in range(256):
        tn = wp.min(tx, wp.min(ty, tz))
        if dep != 0.0:
            wp.atomic_add(rho, m, lc, dep * (wp.min(tn, 1.0) - tc))
        if tn >= 1.0:
            return wp.vec4i(cx, cy, cz, 0)
        if tx <= ty and tx <= tz:
            cx += sx
            tx += float(sx) / d[0]
        elif ty <= tz:
            cy += sy
            ty += float(sy) / d[1]
        else:
            cz += sz
            tz += float(sz) / d[2]
        tc = tn
        nv = vox(wp.vec3i(cx, cy, cz), kb)
        if not same(nv, v):
            v = nv
            c = look(vid, v)
            if c == -2:
                return wp.vec4i(v[0], v[1], v[2], 2)
            if c < 0:
                return wp.vec4i(v[0], v[1], v[2], 1)
            if alive[c] == wp.uint8(0):
                return wp.vec4i(v[0], v[1], v[2], 1)
            lc = c
    return wp.vec4i(cx, cy, cz, 0)


@wp.kernel
def launch(mode: int, fvox: wp.array(dtype=wp.int32), fgv: wp.array(dtype=wp.vec3i), fax: wp.array(dtype=wp.int32),
           fsg: wp.array(dtype=wp.int32), fp0: wp.array(dtype=wp.vec3), fb0: wp.array(dtype=wp.vec3i),
           fu: wp.array(dtype=wp.vec3i), fv: wp.array(dtype=wp.vec3i), gpts: wp.array(dtype=wp.vec2),
           vid: wp.array3d(dtype=wp.int32), alive: wp.array(dtype=wp.uint8), eidx: wp.array2d(dtype=wp.int32),
           efw: wp.array2d(dtype=wp.float32), ib: int, omega: wp.array(dtype=wp.vec3),
           dvec: wp.array(dtype=wp.vec3), smm: float, kb: int, bval: wp.array3d(dtype=wp.float32),
           tgrid: wp.array3d(dtype=wp.float32), lidx: wp.array2d(dtype=wp.int32), lzi: wp.array2d(dtype=wp.float32),
           h: wp.array2d(dtype=vec10), sign: float,
           y: wp.array2d(dtype=vec10), outv: wp.array3d(dtype=wp.float64), rho: wp.array2d(dtype=wp.float32)):
    # Walkers launched through face f (from the grey-matter voxel fgv into the white-matter
    # voxel fvox) at its point g, in each direction of the fODF there that enters: f w |Omega.n|
    # per face, shared evenly by its points.  Their first step, by mode: LAUNCHED, count them
    # (outv, at the launch voxel); FORWARD (bval-weighted), project
    # those that land onto their landing bin (y, divided by their cone's fODF mass as
    # fwd_shift's) and count those that leave (outv, where);
    # REACH, add up their importance (h, or tgrid where they leave) at the launch voxel (outv);
    # DENSITY (bval-weighted), the bundle density of their first step (rho)
    f, g = wp.tid()
    cv = fvox[f]
    if alive[cv] == wp.uint8(0):
        return
    gv = fgv[f]
    bv = float(1.0)
    if mode == 1 or mode == 3:
        bv = bval[gv[0], gv[1], gv[2]]
        if bv == 0.0:
            return
    pt = gpts[g]
    P = fp0[f] + pt[0] * f3(fu[f]) + pt[1] * f3(fv[f])
    B0 = fb0[f] + int(wp.floor(pt[0])) * fu[f] + int(wp.floor(pt[1])) * fv[f]
    ax = fax[f]
    sg = float(fsg[f])
    sb = sublin(B0, vox(B0, kb), kb) * ib          # the fODF of the bin the walkers start in
    share = bv / float(gpts.shape[0])
    acc = float(0.0)
    for m in range(omega.shape[0]):
        cosv = omega[m][ax] * sg
        if cosv <= 0.0:
            continue
        e = eidx[m, cv]
        if e < 0:
            continue
        W = cosv * efw[e, sb] * share
        if mode == 0:
            acc += W
            continue
        d = dvec[m]
        res = trace(P, B0, d, kb, vid, alive, 0.0, rho, m)
        lb = wp.vec3i(res[0], res[1], res[2])
        if mode == 1:
            if res[3] == 0:
                lv = vox(lb, kb)
                l = lidx[look(vid, lv), m]
                if l >= 0:
                    sl = sublin(lb, lv, kb)
                    wp.atomic_add(y, l, sl, (W * lzi[l, sl * ib]) * basis(P + d - f3(lb)))
            elif res[3] == 1:
                wp.atomic_add(outv, lb[0], lb[1], lb[2], wp.float64(W))
            continue
        val = float(0.0)
        if res[3] == 0:
            lv = vox(lb, kb)
            l = lidx[look(vid, lv), m]
            if l >= 0:
                val = wp.dot(basis(P + d - f3(lb)), h[l, sublin(lb, lv, kb)])
        elif res[3] == 1:
            val = tval(tgrid, lb)
        if mode == 2:
            acc += W * val
        elif val != 0.0:
            trace(P, B0, d, kb, vid, alive, sign * W * val * smm, rho, m)
    if (mode == 0 or mode == 2) and acc != 0.0:
        wp.atomic_add(outv, gv[0], gv[1], gv[2], wp.float64(acc))
