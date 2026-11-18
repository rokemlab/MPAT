"""NVIDIA Warp kernels for mpat.transport (imported only there: ``pip install warp-lang``).

Angular fluxes are float32 arrays (column, white-matter cell, direction).  A cell's neighbours
are six slots, x-, x+, y-, y+, z-, z+: ``nbr_c`` the neighbouring white-matter cell (-1 if
none), ``nbr_v`` the neighbouring voxel (linear index into the volume, -1 outside it).  A
direction's octant is its signs (bit 0: x < 0, bit 1: y < 0, bit 2: z < 0); ``up[o, k]`` /
``dn[o, k]`` are the slots of the upwind / downwind neighbour along axis k for octant o.
``alive``: the cells walkers may be in for this solve (not excluded, not in the avoid set);
``bval``: per column and voxel, the boundary value outside them (launches, or targets).
Adjoint fluxes are indexed by the direction they are swept in, the reverse of the walker's.
"""
import warp as wp

wp.set_module_options({"enable_backward": False})     # no autodiff: half the code to compile


@wp.kernel
def sweep_plane(d: int, order: wp.array2d(dtype=wp.int32), pstart: wp.array2d(dtype=wp.int32),
                odirs: wp.array2d(dtype=wp.int32), up: wp.array2d(dtype=wp.int32),
                nbr_c: wp.array2d(dtype=wp.int32), nbr_v: wp.array2d(dtype=wp.int32),
                acoef: wp.array2d(dtype=wp.float32), hlen: wp.array(dtype=wp.float32), sigma: float,
                alive: wp.array(dtype=wp.uint8), bval: wp.array2d(dtype=wp.float32), use_fod: int,
                fod: wp.array2d(dtype=wp.float32), ncol: int, psi: wp.array3d(dtype=wp.float32)):
    # One plane of an upwind (step characteristics) sweep: psi holds the source on entry and
    # the angular flux on exit,  psi = (src + sum_k a_k psi_upwind_k) / (sigma + sum_k a_k).
    # Thread (octant, cell of plane d in that octant's order, direction of that octant).
    o, p, j = wp.tid()
    i0 = pstart[o, d]
    if p >= pstart[o, d + 1] - i0:
        return
    m = odirs[o, j]
    if m < 0:
        return
    c = order[o, i0 + p]
    if alive[c] == wp.uint8(0):
        for col in range(ncol):
            psi[col, c, m] = 0.0
        return
    den = sigma + acoef[m, 0] + acoef[m, 1] + acoef[m, 2]
    for col in range(ncol):
        num = psi[col, c, m]
        for k in range(3):
            a = acoef[m, k]
            if a > 0.0:
                s = up[o, k]
                n = nbr_c[c, s]
                inside = int(0)
                if n >= 0:
                    if alive[n] != wp.uint8(0):
                        inside = 1
                if inside == 1:
                    num += a * psi[col, n, m]
                else:
                    v = nbr_v[c, s]
                    if v >= 0:
                        b = bval[col, v]
                        if b != 0.0:
                            if use_fod != 0:            # launches: as many as the fODF where they enter
                                b = b * hlen[k] * fod[c, m]
                            num += a * b
        psi[col, c, m] = num / den


@wp.kernel
def turn_forward(psi: wp.array3d(dtype=wp.float32), fod: wp.array2d(dtype=wp.float32),
                 wiz: wp.array2d(dtype=wp.float32), cptr: wp.array(dtype=wp.int32),
                 cidx: wp.array(dtype=wp.int32), sigma: float, ncol: int, src: wp.array3d(dtype=wp.float32)):
    # walkers turning into direction m:  src(c, m) = sigma fod(c, m) sum_{m' in cone(m)} psi(c, m') w(m') / Z(c, m')
    c, m = wp.tid()
    f = fod[c, m]
    k0 = cptr[m]
    k1 = cptr[m + 1]
    for col in range(ncol):
        acc = float(0.0)
        if f > 0.0:
            for k in range(k0, k1):
                n = cidx[k]
                acc += psi[col, c, n] * wiz[c, n]
        src[col, c, m] = sigma * f * acc


@wp.kernel
def turn_adjoint(J: wp.array3d(dtype=wp.float32), fod: wp.array2d(dtype=wp.float32),
                 wiz: wp.array2d(dtype=wp.float32), w: wp.array(dtype=wp.float32),
                 rev: wp.array(dtype=wp.int32), cptr: wp.array(dtype=wp.int32),
                 cidx: wp.array(dtype=wp.int32), sigma: float, ncol: int, src: wp.array3d(dtype=wp.float32)):
    # the importance of a walker heading m = rev(n) that turns:
    #   src(c, n) = sigma / Z(c, m) sum_{m2 in cone(m)} fod(c, m2) w(m2) J(c, rev(m2))
    c, n = wp.tid()
    mf = rev[n]
    wz = wiz[c, mf]
    if wz == 0.0:                     # no fODF within the angle: a walker that turns is lost
        for col in range(ncol):
            src[col, c, n] = 0.0
        return
    iz = wz / w[mf]
    k0 = cptr[n]
    k1 = cptr[n + 1]
    for col in range(ncol):
        acc = float(0.0)
        for k in range(k0, k1):
            n2 = cidx[k]
            m2 = rev[n2]
            acc += fod[c, m2] * w[m2] * J[col, c, n2]
        src[col, c, n] = sigma * iz * acc


@wp.kernel
def cone_norm(fod: wp.array2d(dtype=wp.float32), w: wp.array(dtype=wp.float32), cptr: wp.array(dtype=wp.int32),
              cidx: wp.array(dtype=wp.int32), wiz: wp.array2d(dtype=wp.float32)):
    # wiz(c, m) = w(m) / Z(c, m),  Z(c, m) = sum_{m' in cone(m)} fod(c, m') w(m')  (0 where Z = 0)
    c, m = wp.tid()
    z = float(0.0)
    for k in range(cptr[m], cptr[m + 1]):
        n = cidx[k]
        z += fod[c, n] * w[n]
    if z > 0.0:
        wiz[c, m] = w[m] / z
    else:
        wiz[c, m] = 0.0


@wp.kernel
def zero2d64(a: wp.array2d(dtype=wp.float64)):
    i, j = wp.tid()
    a[i, j] = wp.float64(0.0)


@wp.kernel
def increments(cur: wp.array3d(dtype=wp.float32), prev: wp.array3d(dtype=wp.float32), w: wp.array(dtype=wp.float32),
               ncell: int, M: int, nblk: int, out: wp.array2d(dtype=wp.float64)):
    # per column: what this sweep added (sum w (cur - prev)) and the total (sum w cur), in nblk
    # row-strided blocks
    col, b = wp.tid()
    inc = wp.float64(0.0)
    tot = wp.float64(0.0)
    for i in range(b, ncell * M, nblk):
        c = i / M
        m = i - c * M
        x = cur[col, c, m]
        ww = wp.float64(w[m])
        inc += ww * wp.float64(x - prev[col, c, m])
        tot += ww * wp.float64(x)
    wp.atomic_add(out, col, 0, inc)
    wp.atomic_add(out, col, 1, tot)


@wp.kernel
def add_density(psi: wp.array3d(dtype=wp.float32), J: wp.array3d(dtype=wp.float32), rev: wp.array(dtype=wp.int32),
                w: wp.array(dtype=wp.float32), coup: wp.array2d(dtype=wp.float32), nkf: int, nkj: int, sign: float,
                rho: wp.array2d(dtype=wp.float32)):
    # rho(c, m) += sign w(m) sum_kl coup(k, l) psi_k(c, m) J_l(c, rev m): path length (mm) of the
    # bundle's walkers in cell c heading m (nkf forward columns k, nkj adjoint columns l)
    c, m = wp.tid()
    mr = rev[m]
    acc = float(0.0)
    for k in range(nkf):
        pk = psi[k, c, m]
        if pk != 0.0:
            for l in range(nkj):
                acc += coup[k, l] * pk * J[l, c, mr]
    rho[c, m] = rho[c, m] + sign * w[m] * acc


@wp.kernel
def exit_tally(psi: wp.array3d(dtype=wp.float32), alive: wp.array(dtype=wp.uint8),
               nbr_c: wp.array2d(dtype=wp.int32), nbr_v: wp.array2d(dtype=wp.int32), dn: wp.array2d(dtype=wp.int32),
               octant: wp.array(dtype=wp.int32), acoef: wp.array2d(dtype=wp.float32), w: wp.array(dtype=wp.float32),
               ncol: int, out: wp.array2d(dtype=wp.float32)):
    # walkers leaving the white matter into each voxel: w a_k psi through the downwind faces
    c, m = wp.tid()
    if alive[c] == wp.uint8(0):
        return
    o = octant[m]
    for k in range(3):
        a = acoef[m, k]
        if a > 0.0:
            s = dn[o, k]
            if nbr_c[c, s] < 0:
                v = nbr_v[c, s]
                if v >= 0:
                    for col in range(ncol):
                        x = psi[col, c, m]
                        if x != 0.0:
                            wp.atomic_add(out, col, v, w[m] * a * x)


@wp.kernel
def launch_tally(J: wp.array3d(dtype=wp.float32), alive: wp.array(dtype=wp.uint8),
                 nbr_c: wp.array2d(dtype=wp.int32), nbr_v: wp.array2d(dtype=wp.int32), up: wp.array2d(dtype=wp.int32),
                 octant: wp.array(dtype=wp.int32), acoef: wp.array2d(dtype=wp.float32),
                 hlen: wp.array(dtype=wp.float32), fod: wp.array2d(dtype=wp.float32), w: wp.array(dtype=wp.float32),
                 rev: wp.array(dtype=wp.int32), ncol: int, ones: int, out: wp.array2d(dtype=wp.float32)):
    # walkers launched from each voxel into the white matter (ones = 1), or those of them that
    # reach the adjoint's targets (weighted by J)
    c, m = wp.tid()
    if alive[c] == wp.uint8(0):
        return
    f = fod[c, m]
    if f == 0.0:
        return
    o = octant[m]
    mr = rev[m]
    for k in range(3):
        a = acoef[m, k]
        if a > 0.0:
            s = up[o, k]
            if nbr_c[c, s] < 0:
                v = nbr_v[c, s]
                if v >= 0:
                    q = a * hlen[k] * f * w[m]
                    if ones != 0:
                        wp.atomic_add(out, 0, v, q)
                    else:
                        for col in range(ncol):
                            wp.atomic_add(out, col, v, q * J[col, c, mr])
