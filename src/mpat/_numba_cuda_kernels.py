"""numba-cuda kernels for mpat.gpu (imported only there: ``pip install "numba-cuda[cu13]"``).

The same kernels as mpat._warp_kernels, argument for argument and with the same thread layout,
so that the two backends differ only in their compilers: rows are states (times the values a
state carries), columns are solution columns, one thread per (row, column) with the column
varying fastest.  Every kernel takes its launch dimensions last (``n0`` and, for 2-D launches,
``n1``) and runs a grid-stride loop over them (64-bit indices), so launches of any size work.
Early exits are ``continue``s of that loop.

``lto=False``: numba-cuda would otherwise compile to LTO-IR whenever nvJitLink is installed, and
LTO-IR links only when nvJitLink is at least as new as the NVVM that made it, which breaks when
another package in the process loads a different nvJitLink or NVVM.  These kernels link nothing
in, so LTO gains them nothing; PTX is what Warp's kernels go through too.
"""
from numba import config, cuda



def _jit(f):
    # cuda.jit(f, ...), not cuda.jit(...)(f): numba-cuda 0.30 drops lto= in the second form.
    # The simulator takes no lto=.
    return cuda.jit(f, cache=True) if config.ENABLE_CUDASIM else cuda.jit(f, cache=True, lto=False)


# ------------------------------------------------------------------ the chain: stencil storage
@_jit
def stencil_backward(ptr, slot, w, tn, tb, nbr, nb, s0, x, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        s = s0 + i
        vox = s // nb
        cls = s - vox * nb
        acc = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            sl = slot[k]
            j = nbr[vox, tn[cls, sl]] * nb + tb[cls, sl]
            acc += float(w[k]) * x[j, c]
        out[s, c] = acc


@_jit
def stencil_forward(ptr, slot, w, tn, tb, nbr, nb, s0, x, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        s = s0 + i
        v = x[s, c]
        if v == 0.0:
            continue
        vox = s // nb
        cls = s - vox * nb
        for k in range(ptr[i], ptr[i + 1]):
            sl = slot[k]
            j = nbr[vox, tn[cls, sl]] * nb + tb[cls, sl]
            cuda.atomic.add(out, (j, c), float(w[k]) * v)


# ------------------------------------------------------------------ the moment chain: 4 x 4 blocks
@_jit
def bsr_backward(ptr, ind, blk, s0, x, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        s = s0 + i
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            j = ind[k] * 4
            x0 = x[j, c]
            x1 = x[j + 1, c]
            x2 = x[j + 2, c]
            x3 = x[j + 3, c]
            a0 += float(blk[k, 0]) * x0 + float(blk[k, 4]) * x1 + float(blk[k, 8]) * x2 + float(blk[k, 12]) * x3
            a1 += float(blk[k, 1]) * x0 + float(blk[k, 5]) * x1 + float(blk[k, 9]) * x2 + float(blk[k, 13]) * x3
            a2 += float(blk[k, 2]) * x0 + float(blk[k, 6]) * x1 + float(blk[k, 10]) * x2 + float(blk[k, 14]) * x3
            a3 += float(blk[k, 3]) * x0 + float(blk[k, 7]) * x1 + float(blk[k, 11]) * x2 + float(blk[k, 15]) * x3
        b = s * 4
        out[b, c] = a0
        out[b + 1, c] = a1
        out[b + 2, c] = a2
        out[b + 3, c] = a3


@_jit
def bsr_forward(ptr, ind, blk, s0, x, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        b = (s0 + i) * 4
        x0 = x[b, c]
        x1 = x[b + 1, c]
        x2 = x[b + 2, c]
        x3 = x[b + 3, c]
        if x0 == 0.0 and x1 == 0.0 and x2 == 0.0 and x3 == 0.0:
            continue
        for k in range(ptr[i], ptr[i + 1]):
            j = ind[k] * 4
            cuda.atomic.add(out, (j, c), float(blk[k, 0]) * x0 + float(blk[k, 1]) * x1
                            + float(blk[k, 2]) * x2 + float(blk[k, 3]) * x3)
            cuda.atomic.add(out, (j + 1, c), float(blk[k, 4]) * x0 + float(blk[k, 5]) * x1
                            + float(blk[k, 6]) * x2 + float(blk[k, 7]) * x3)
            cuda.atomic.add(out, (j + 2, c), float(blk[k, 8]) * x0 + float(blk[k, 9]) * x1
                            + float(blk[k, 10]) * x2 + float(blk[k, 11]) * x3)
            cuda.atomic.add(out, (j + 3, c), float(blk[k, 12]) * x0 + float(blk[k, 13]) * x1
                            + float(blk[k, 14]) * x2 + float(blk[k, 15]) * x3)


# ------------------------------------------------------------------ absorption
@_jit
def csr_gather(ptr, ind, val, s0, a, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        acc = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            acc += val[k] * a[ind[k], c]
        out[s0 + i, c] = acc


@_jit
def csr_scatter(ptr, ind, val, s0, x, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        v = x[s0 + i, c]
        if v == 0.0:
            continue
        for k in range(ptr[i], ptr[i + 1]):
            cuda.atomic.add(out, (ind[k], c), val[k] * v)


@_jit
def rblk_gather(ptr, ind, blk, s0, a, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        a0 = 0.0
        a1 = 0.0
        a2 = 0.0
        a3 = 0.0
        for k in range(ptr[i], ptr[i + 1]):
            v = a[ind[k], c]
            a0 += float(blk[k, 0]) * v
            a1 += float(blk[k, 1]) * v
            a2 += float(blk[k, 2]) * v
            a3 += float(blk[k, 3]) * v
        b = (s0 + i) * 4
        out[b, c] = a0
        out[b + 1, c] = a1
        out[b + 2, c] = a2
        out[b + 3, c] = a3


@_jit
def rblk_scatter(ptr, ind, blk, s0, x, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        b = (s0 + i) * 4
        x0 = x[b, c]
        x1 = x[b + 1, c]
        x2 = x[b + 2, c]
        x3 = x[b + 3, c]
        if x0 == 0.0 and x1 == 0.0 and x2 == 0.0 and x3 == 0.0:
            continue
        for k in range(ptr[i], ptr[i + 1]):
            cuda.atomic.add(out, (ind[k], c), float(blk[k, 0]) * x0 + float(blk[k, 1]) * x1
                            + float(blk[k, 2]) * x2 + float(blk[k, 3]) * x3)


# ------------------------------------------------------------------ elementwise and reductions
@_jit
def fill_zero(a, n0):
    for i in range(cuda.grid(1), n0, cuda.gridsize(1)):
        a[i] = 0


@_jit
def mul_mask(x, mask, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        if mask[i, c] == 0:
            x[i, c] = 0.0


@_jit
def masked_copy(x, mask, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        out[i, c] = x[i, c] if mask[i, c] != 0 else 0.0


@_jit
def copy_cols(x, c0, out, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        out[i, c] = x[i, c0 + c]


@_jit
def zero_rows(x, kill, nc, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        i = idx // n1
        c = idx - i * n1
        if kill[i // nc] != 0:
            x[i, c] = 0.0


@_jit
def accumulate(nxt, mask, total, nc, nblk, part, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        b = idx // n1
        c = idx - b * n1
        acc = 0.0
        for i in range(b, nxt.shape[0], nblk):
            v = nxt[i, c] if mask[i, c] != 0 else 0.0
            nxt[i, c] = v
            total[i, c] += v
            if i % nc == 0:
                acc += abs(v)
        part[b, c] = acc


@_jit
def step_source(nxt, mask, src, cur, nc, nblk, part, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        b = idx // n1
        c = idx - b * n1
        acc = 0.0
        for i in range(b, nxt.shape[0], nblk):
            v = src[i, c]
            if mask[i, c] != 0:
                v += nxt[i, c]
            nxt[i, c] = v
            if i % nc == 0:
                acc += abs(v - cur[i, c])
        part[b, c] = acc


@_jit
def col_mass(x, nc, nblk, part, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        b = idx // n1
        c = idx - b * n1
        acc = 0.0
        for i in range(b, x.shape[0], nblk):
            if i % nc == 0:
                acc += abs(x[i, c])
        part[b, c] = acc


@_jit
def sum_parts(part, out, n0):
    for c in range(cuda.grid(1), n0, cuda.gridsize(1)):
        acc = 0.0
        for b in range(part.shape[0]):
            acc += part[b, c]
        out[c] = acc


@_jit
def row_mass(x, minus, use_minus, nc, out, n0):
    for s in range(cuda.grid(1), n0, cuda.gridsize(1)):
        i = s * nc
        acc = 0.0
        for c in range(x.shape[1]):
            v = x[i, c]
            if use_minus != 0:
                v -= minus[i, c]
            acc += abs(v)
        out[s] = acc


@_jit
def row_sums(x, out, n0):
    for i in range(cuda.grid(1), n0, cuda.gridsize(1)):
        acc = 0.0
        for c in range(x.shape[1]):
            acc += x[i, c]
        out[i] = acc


@_jit
def rows_density(S, beta, colsign, c0, rows_per, out, n0):
    for v in range(cuda.grid(1), n0, cuda.gridsize(1)):
        acc = 0.0
        for r in range(v * rows_per, (v + 1) * rows_per):
            for c in range(colsign.shape[0]):
                acc += S[r, c0 + c] * beta[r, c0 + c] * colsign[c]
        out[v] = acc


# ------------------------------------------------------------------ the forward start
@_jit
def half_values(lptr, ent_state, ent_abs, lval, nc, beta, absw, hseed, bits, col_plan, col_mask, vals, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        h = idx // n1
        c = idx - h * n1
        acc = 0.0
        for k in range(lptr[h], lptr[h + 1]):
            st = ent_state[k]
            if st >= 0:
                row = st * nc
                for r in range(nc):
                    acc += lval[k, r] * beta[row + r, c]
            else:
                acc += lval[k, 0] * absw[ent_abs[k], c]
        if (bits[col_plan[c], hseed[h]] & col_mask[c]) != 0:
            acc = 0.0
        vals[h, c] = acc


@_jit
def pair_terms(vals, M, signs, pair_w, hseed, bits, plan, xmask, c0, nK, coef, pbx, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        p = idx // n1
        x = idx - p * n1
        h0 = 2 * p
        h1 = h0 + 1
        col = c0 + x * nK
        acc = 0.0
        for k in range(nK):
            for m in range(nK):
                acc += vals[h0, col + k] * M[k, m] * vals[h1, col + m]
        pbx[p, x] = signs[x] * acc
        sm = xmask[x]
        ok0 = 0.0 if (bits[plan, hseed[h0]] & sm) != 0 else 1.0
        ok1 = 0.0 if (bits[plan, hseed[h1]] & sm) != 0 else 1.0
        pw = pair_w[p]
        for k in range(nK):
            t0 = 0.0
            t1 = 0.0
            for m in range(nK):
                t0 += vals[h1, col + m] * M[k, m]
                t1 += vals[h0, col + m] * M[k, m]
            coef[h0, col + k] = t0 * pw * ok0
            coef[h1, col + k] = t1 * pw * ok1


@_jit
def scatter_init(lptr, ent_state, ent_abs, ent_vox, lval, nc, coef, bits, col_plan, col_mask, pi0, a0, n0, n1):
    for idx in range(cuda.grid(1), n0 * n1, cuda.gridsize(1)):
        h = idx // n1
        c = idx - h * n1
        cf = coef[h, c]
        if cf == 0.0:
            continue
        plan = col_plan[c]
        cm = col_mask[c]
        for k in range(lptr[h], lptr[h + 1]):
            st = ent_state[k]
            if st >= 0:
                if (bits[plan, ent_vox[k]] & cm) == 0:
                    row = st * nc
                    for r in range(nc):
                        cuda.atomic.add(pi0, (row + r, c), cf * lval[k, r])
            else:
                cuda.atomic.add(a0, (ent_abs[k], c), cf * lval[k, 0])
