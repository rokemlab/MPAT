"""NVIDIA Warp kernels for mpat.gpu (imported only there: ``pip install warp-lang``).

Solution arrays are row-major: rows are states (times the values a state carries: 1, or 4 on
the moment chain), columns are solution columns.  Kernels over (rows, columns) are launched
with the column varying fastest, so neighbouring threads read neighbouring columns of a row.
Sparse matrices are cut into segments of states whose entries fit Warp's 32-bit indices:
``s0`` is a segment's first state and ``ptr`` its row pointers from 0.  Reductions over rows
run in ``nblk`` row-strided blocks per column (``part``), summed by ``sum_parts``.

Kernels mirror mpat.kernel and mpat.moments; float literals are cast (Warp's are float32).
"""
from typing import Any

import warp as wp

wp.set_module_options({"enable_backward": False})     # no autodiff: half the code to compile

F0 = wp.constant(wp.float64(0.0))


# ------------------------------------------------------------------ the chain: stencil storage
@wp.kernel
def stencil_backward(ptr: wp.array(dtype=wp.int32), slot: wp.array(dtype=Any), w: wp.array(dtype=Any),
                     tn: wp.array2d(dtype=wp.int32), tb: wp.array2d(dtype=wp.int32),
                     nbr: wp.array2d(dtype=wp.int32), nb: int, s0: int,
                     x: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    # out = Q x on the segment's states: each state gathers from its targets
    i, c = wp.tid()
    s = s0 + i
    vox = s / nb
    cls = s - vox * nb
    acc = F0
    for k in range(ptr[i], ptr[i + 1]):
        sl = wp.int32(slot[k])
        j = nbr[vox, tn[cls, sl]] * nb + tb[cls, sl]
        acc += wp.float64(w[k]) * x[j, c]
    out[s, c] = acc


@wp.kernel
def stencil_forward(ptr: wp.array(dtype=wp.int32), slot: wp.array(dtype=Any), w: wp.array(dtype=Any),
                    tn: wp.array2d(dtype=wp.int32), tb: wp.array2d(dtype=wp.int32),
                    nbr: wp.array2d(dtype=wp.int32), nb: int, s0: int,
                    x: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    # out += Q^T x from the segment's states: each state scatters to its targets (out zeroed first)
    i, c = wp.tid()
    s = s0 + i
    v = x[s, c]
    if v == F0:
        return
    vox = s / nb
    cls = s - vox * nb
    for k in range(ptr[i], ptr[i + 1]):
        sl = wp.int32(slot[k])
        j = nbr[vox, tn[cls, sl]] * nb + tb[cls, sl]
        wp.atomic_add(out, j, c, wp.float64(w[k]) * v)


for _slot in (wp.uint8, wp.uint16):
    for _w in (wp.float32, wp.float64):
        for _k in (stencil_backward, stencil_forward):
            wp.overload(_k, {"slot": wp.array(dtype=_slot), "w": wp.array(dtype=_w)})


# ------------------------------------------------------------------ the moment chain: 4 x 4 blocks
# blk[k, 4 * i + r]: value r of a transition's source to value i of its target
@wp.kernel
def bsr_backward(ptr: wp.array(dtype=wp.int32), ind: wp.array(dtype=wp.int32), blk: wp.array2d(dtype=wp.float32),
                 s0: int, x: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    i, c = wp.tid()
    s = s0 + i
    a0 = F0
    a1 = F0
    a2 = F0
    a3 = F0
    for k in range(ptr[i], ptr[i + 1]):
        j = ind[k] * 4
        x0 = x[j, c]
        x1 = x[j + 1, c]
        x2 = x[j + 2, c]
        x3 = x[j + 3, c]
        a0 += (wp.float64(blk[k, 0]) * x0 + wp.float64(blk[k, 4]) * x1 + wp.float64(blk[k, 8]) * x2
               + wp.float64(blk[k, 12]) * x3)
        a1 += (wp.float64(blk[k, 1]) * x0 + wp.float64(blk[k, 5]) * x1 + wp.float64(blk[k, 9]) * x2
               + wp.float64(blk[k, 13]) * x3)
        a2 += (wp.float64(blk[k, 2]) * x0 + wp.float64(blk[k, 6]) * x1 + wp.float64(blk[k, 10]) * x2
               + wp.float64(blk[k, 14]) * x3)
        a3 += (wp.float64(blk[k, 3]) * x0 + wp.float64(blk[k, 7]) * x1 + wp.float64(blk[k, 11]) * x2
               + wp.float64(blk[k, 15]) * x3)
    b = s * 4
    out[b, c] = a0
    out[b + 1, c] = a1
    out[b + 2, c] = a2
    out[b + 3, c] = a3


@wp.kernel
def bsr_forward(ptr: wp.array(dtype=wp.int32), ind: wp.array(dtype=wp.int32), blk: wp.array2d(dtype=wp.float32),
                s0: int, x: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    i, c = wp.tid()
    s = s0 + i
    b = s * 4
    x0 = x[b, c]
    x1 = x[b + 1, c]
    x2 = x[b + 2, c]
    x3 = x[b + 3, c]
    if x0 == F0 and x1 == F0 and x2 == F0 and x3 == F0:
        return
    for k in range(ptr[i], ptr[i + 1]):
        j = ind[k] * 4
        wp.atomic_add(out, j, c, wp.float64(blk[k, 0]) * x0 + wp.float64(blk[k, 1]) * x1
                      + wp.float64(blk[k, 2]) * x2 + wp.float64(blk[k, 3]) * x3)
        wp.atomic_add(out, j + 1, c, wp.float64(blk[k, 4]) * x0 + wp.float64(blk[k, 5]) * x1
                      + wp.float64(blk[k, 6]) * x2 + wp.float64(blk[k, 7]) * x3)
        wp.atomic_add(out, j + 2, c, wp.float64(blk[k, 8]) * x0 + wp.float64(blk[k, 9]) * x1
                      + wp.float64(blk[k, 10]) * x2 + wp.float64(blk[k, 11]) * x3)
        wp.atomic_add(out, j + 3, c, wp.float64(blk[k, 12]) * x0 + wp.float64(blk[k, 13]) * x1
                      + wp.float64(blk[k, 14]) * x2 + wp.float64(blk[k, 15]) * x3)


# ------------------------------------------------------------------ absorption
@wp.kernel
def csr_gather(ptr: wp.array(dtype=wp.int32), ind: wp.array(dtype=wp.int32), val: wp.array(dtype=wp.float64),
               s0: int, a: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    # out[s] = sum_k val[k] a[ind[k]]: the probability of being absorbed next step, weighted by a
    i, c = wp.tid()
    s = s0 + i
    acc = F0
    for k in range(ptr[i], ptr[i + 1]):
        acc += val[k] * a[ind[k], c]
    out[s, c] = acc


@wp.kernel
def csr_scatter(ptr: wp.array(dtype=wp.int32), ind: wp.array(dtype=wp.int32), val: wp.array(dtype=wp.float64),
                s0: int, x: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    # out[ind[k]] += val[k] x[s]: the mass absorbed from occupancy x (out zeroed first)
    i, c = wp.tid()
    s = s0 + i
    v = x[s, c]
    if v == F0:
        return
    for k in range(ptr[i], ptr[i + 1]):
        wp.atomic_add(out, ind[k], c, val[k] * v)


@wp.kernel
def rblk_gather(ptr: wp.array(dtype=wp.int32), ind: wp.array(dtype=wp.int32), blk: wp.array2d(dtype=wp.float32),
                s0: int, a: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    i, c = wp.tid()
    s = s0 + i
    a0 = F0
    a1 = F0
    a2 = F0
    a3 = F0
    for k in range(ptr[i], ptr[i + 1]):
        v = a[ind[k], c]
        a0 += wp.float64(blk[k, 0]) * v
        a1 += wp.float64(blk[k, 1]) * v
        a2 += wp.float64(blk[k, 2]) * v
        a3 += wp.float64(blk[k, 3]) * v
    b = s * 4
    out[b, c] = a0
    out[b + 1, c] = a1
    out[b + 2, c] = a2
    out[b + 3, c] = a3


@wp.kernel
def rblk_scatter(ptr: wp.array(dtype=wp.int32), ind: wp.array(dtype=wp.int32), blk: wp.array2d(dtype=wp.float32),
                 s0: int, x: wp.array2d(dtype=wp.float64), out: wp.array2d(dtype=wp.float64)):
    i, c = wp.tid()
    s = s0 + i
    b = s * 4
    x0 = x[b, c]
    x1 = x[b + 1, c]
    x2 = x[b + 2, c]
    x3 = x[b + 3, c]
    if x0 == F0 and x1 == F0 and x2 == F0 and x3 == F0:
        return
    for k in range(ptr[i], ptr[i + 1]):
        wp.atomic_add(out, ind[k], c, wp.float64(blk[k, 0]) * x0 + wp.float64(blk[k, 1]) * x1
                      + wp.float64(blk[k, 2]) * x2 + wp.float64(blk[k, 3]) * x3)


# ------------------------------------------------------------------ elementwise and reductions
@wp.kernel
def mul_mask(x: wp.array2d(dtype=wp.float64), mask: wp.array2d(dtype=wp.uint8)):
    i, c = wp.tid()
    if mask[i, c] == wp.uint8(0):
        x[i, c] = F0


@wp.kernel
def masked_copy(x: wp.array2d(dtype=wp.float64), mask: wp.array2d(dtype=wp.uint8), out: wp.array2d(dtype=wp.float64)):
    i, c = wp.tid()
    if mask[i, c] == wp.uint8(0):
        out[i, c] = F0
    else:
        out[i, c] = x[i, c]


@wp.kernel
def copy_cols(x: wp.array2d(dtype=wp.float64), c0: int, out: wp.array2d(dtype=wp.float64)):
    # out = x[:, c0 : c0 + out.shape[1]] (a bundle's columns of its group's, to read on the host)
    i, c = wp.tid()
    out[i, c] = x[i, c0 + c]


@wp.kernel
def zero_rows(x: wp.array2d(dtype=wp.float64), kill: wp.array(dtype=wp.uint8), nc: int):
    # the rows of states past the soft maximum length (walkers end there)
    i, c = wp.tid()
    if kill[i / nc] != wp.uint8(0):
        x[i, c] = F0


@wp.kernel
def accumulate(nxt: wp.array2d(dtype=wp.float64), mask: wp.array2d(dtype=wp.uint8),
               total: wp.array2d(dtype=wp.float64), nc: int, nblk: int, part: wp.array2d(dtype=wp.float64)):
    # nxt *= mask; total += nxt; part: the mass nxt carries (mass rows), per block of rows
    b, c = wp.tid()
    nrows = nxt.shape[0]
    acc = F0
    i = b
    while i < nrows:
        v = nxt[i, c]
        if mask[i, c] == wp.uint8(0):
            v = F0
        nxt[i, c] = v
        total[i, c] = total[i, c] + v
        if i % nc == 0:
            acc += wp.abs(v)
        i += nblk
    part[b, c] = acc


@wp.kernel
def step_source(nxt: wp.array2d(dtype=wp.float64), mask: wp.array2d(dtype=wp.uint8),
                src: wp.array2d(dtype=wp.float64), cur: wp.array2d(dtype=wp.float64), nc: int, nblk: int,
                part: wp.array2d(dtype=wp.float64)):
    # nxt = src + mask * nxt (nxt holding A cur); part: |nxt - cur| (mass rows), per block of rows
    b, c = wp.tid()
    nrows = nxt.shape[0]
    acc = F0
    i = b
    while i < nrows:
        v = src[i, c]
        if mask[i, c] != wp.uint8(0):
            v = v + nxt[i, c]
        nxt[i, c] = v
        if i % nc == 0:
            acc += wp.abs(v - cur[i, c])
        i += nblk
    part[b, c] = acc


@wp.kernel
def col_mass(x: wp.array2d(dtype=wp.float64), nc: int, nblk: int, part: wp.array2d(dtype=wp.float64)):
    b, c = wp.tid()
    nrows = x.shape[0]
    acc = F0
    i = b
    while i < nrows:
        if i % nc == 0:
            acc += wp.abs(x[i, c])
        i += nblk
    part[b, c] = acc


@wp.kernel
def sum_parts(part: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    c = wp.tid()
    acc = F0
    for b in range(part.shape[0]):
        acc += part[b, c]
    out[c] = acc


@wp.kernel
def row_mass(x: wp.array2d(dtype=wp.float64), minus: wp.array2d(dtype=wp.float64), use_minus: int, nc: int,
             out: wp.array(dtype=wp.float64)):
    # per state: |x| (or |x - minus|) of its mass row, summed over the columns
    s = wp.tid()
    i = s * nc
    acc = F0
    for c in range(x.shape[1]):
        v = x[i, c]
        if use_minus != 0:
            v = v - minus[i, c]
        acc += wp.abs(v)
    out[s] = acc


@wp.kernel
def row_sums(x: wp.array2d(dtype=wp.float64), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    acc = F0
    for c in range(x.shape[1]):
        acc += x[i, c]
    out[i] = acc


@wp.kernel
def rows_density(S: wp.array2d(dtype=wp.float64), beta: wp.array2d(dtype=wp.float64),
                 colsign: wp.array(dtype=wp.float64), c0: int, rows_per: int, out: wp.array(dtype=wp.float64)):
    # per group of rows_per rows (a state, or a voxel's states): sum of S * beta * colsign over
    # them and a bundle's columns c0 .. c0 + len(colsign)
    v = wp.tid()
    acc = F0
    for r in range(v * rows_per, (v + 1) * rows_per):
        for c in range(colsign.shape[0]):
            acc += S[r, c0 + c] * beta[r, c0 + c] * colsign[c]
    out[v] = acc


# ------------------------------------------------------------------ the forward start
# Launch entries (one per (half-walk, first exit)): ent_state is the transient state reached
# (-1: an absorbing voxel, ent_abs), ent_vox its padded voxel.  bits[plan, voxel]: bit 0 the
# plan's exclude ROI, bit 1 + i its include ROI i; a column's avoid set is taboo where
# bits & col_mask[column] != 0.
@wp.kernel
def half_values(lptr: wp.array(dtype=wp.int32), ent_state: wp.array(dtype=wp.int32),
                ent_abs: wp.array(dtype=wp.int32), lval: wp.array2d(dtype=wp.float64), nc: int,
                beta: wp.array2d(dtype=wp.float64), absw: wp.array2d(dtype=wp.float64),
                hseed: wp.array(dtype=wp.int32), bits: wp.array2d(dtype=wp.uint8),
                col_plan: wp.array(dtype=wp.int32), col_mask: wp.array(dtype=wp.int32),
                vals: wp.array2d(dtype=wp.float64)):
    # vals[h, c]: P(half h completes column c's requirement) (0 if seeded in the avoid set)
    h, c = wp.tid()
    acc = F0
    for k in range(lptr[h], lptr[h + 1]):
        st = ent_state[k]
        if st >= 0:
            row = st * nc
            for r in range(nc):
                acc += lval[k, r] * beta[row + r, c]
        else:
            acc += lval[k, 0] * absw[ent_abs[k], c]
    if (wp.int32(bits[col_plan[c], hseed[h]]) & col_mask[c]) != 0:
        acc = F0
    vals[h, c] = acc


@wp.kernel
def pair_terms(vals: wp.array2d(dtype=wp.float64), M: wp.array2d(dtype=wp.float64),
               signs: wp.array(dtype=wp.float64), pair_w: wp.array(dtype=wp.float64),
               hseed: wp.array(dtype=wp.int32), bits: wp.array2d(dtype=wp.uint8), plan: int,
               xmask: wp.array(dtype=wp.int32), c0: int, nK: int,
               coef: wp.array2d(dtype=wp.float64), pbx: wp.array2d(dtype=wp.float64)):
    # one plan's streamline p, avoid set x: its share of P(pair in bundle) (pbx), and each
    # half's forward-start weight (its partner's completion, through the end-kind form M)
    p, x = wp.tid()
    h0 = 2 * p
    h1 = h0 + 1
    col = c0 + x * nK
    acc = F0
    for k in range(nK):
        for l in range(nK):
            acc += vals[h0, col + k] * M[k, l] * vals[h1, col + l]
    pbx[p, x] = signs[x] * acc
    sm = xmask[x]
    ok0 = wp.float64(1.0)
    if (wp.int32(bits[plan, hseed[h0]]) & sm) != 0:
        ok0 = F0
    ok1 = wp.float64(1.0)
    if (wp.int32(bits[plan, hseed[h1]]) & sm) != 0:
        ok1 = F0
    pw = pair_w[p]
    for k in range(nK):
        t0 = F0
        t1 = F0
        for l in range(nK):
            t0 += vals[h1, col + l] * M[k, l]
            t1 += vals[h0, col + l] * M[k, l]
        coef[h0, col + k] = t0 * pw * ok0
        coef[h1, col + k] = t1 * pw * ok1


@wp.kernel
def scatter_init(lptr: wp.array(dtype=wp.int32), ent_state: wp.array(dtype=wp.int32),
                 ent_abs: wp.array(dtype=wp.int32), ent_vox: wp.array(dtype=wp.int32),
                 lval: wp.array2d(dtype=wp.float64), nc: int, coef: wp.array2d(dtype=wp.float64),
                 bits: wp.array2d(dtype=wp.uint8), col_plan: wp.array(dtype=wp.int32),
                 col_mask: wp.array(dtype=wp.int32), pi0: wp.array2d(dtype=wp.float64),
                 a0: wp.array2d(dtype=wp.float64)):
    # pi0 += coef x each launch (not into the avoid set); a0: absorbed on the launch step
    h, c = wp.tid()
    cf = coef[h, c]
    if cf == F0:
        return
    plan = col_plan[c]
    cm = col_mask[c]
    for k in range(lptr[h], lptr[h + 1]):
        st = ent_state[k]
        if st >= 0:
            if (wp.int32(bits[plan, ent_vox[k]]) & cm) == 0:
                row = st * nc
                for r in range(nc):
                    wp.atomic_add(pi0, row + r, c, cf * lval[k, r])
        else:
            wp.atomic_add(a0, ent_abs[k], c, cf * lval[k, 0])
