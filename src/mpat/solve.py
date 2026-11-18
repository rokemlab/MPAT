"""Bundle segmentation with include / exclude / endpoint ROIs, analytically.

A streamline from a seed is a pair of half-walks launched along +p and -p for each fODF peak
p.  The bundle event is

    passes every include ROI (either half) AND avoids every exclude ROI (both halves)
    AND both ends are valid (grey matter or brain edge) AND >= 1 end in the endpoint ROI.

Include ROIs are handled by inclusion-exclusion over "avoid" sets, P(hit A and B) =
sum_T (-1)^|T| P(avoid union(T)), and avoiding a set is an exclusion ROI.  For each avoid
set X and end kind k, one backward solve beta = (I - M Q)^-1 M r gives every state's
probability of completing the requirements (shared by all seeds); one forward solve from
every half's launch, weighted by its partner's completion probability, gives the occupancy.
Occupancy x completion is the bundle density (transition path theory's reactive density).
Each solve is a column; bundles are packed into groups of columns that share every pass
over the transition matrix.

``Interleaved`` advances a group's backward and forward solves together instead of one after
the other: the density then exists, as a lower bound that grows, from the first steps (the
live view, mpat.live), and a solve can start from an earlier solution.

The solves' arrays and the operations on them are an ``ops`` object: ``HostOps`` (numpy
arrays, numba kernels; the default) or ``mpat.gpu.WarpOps`` (arrays on a GPU, NVIDIA Warp
kernels).  The algorithm is the same for both.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from . import kernel as K
from .chain import VALID_ENDS

log = logging.getLogger("mpat")

class Clock:
    """Wall time per stage, summed over calls: ``with clock("forward"): ...``."""

    def __init__(self):
        self.t = {}

    def add(self, key, seconds):
        self.t[key] = self.t.get(key, 0.0) + seconds

    @contextmanager
    def __call__(self, key):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(key, time.perf_counter() - t0)

    def report(self, title, total=None):
        """One log line per stage, largest first, with its share of ``total`` (default: the sum)."""
        total = total or sum(self.t.values()) or 1e-9
        log.info("%s (%.0fs):", title, total)
        for k, v in sorted(self.t.items(), key=lambda kv: -kv[1]):
            log.info("  %-24s %8.1fs  %5.1f%%", k, v, 100 * v / total)


STALL_WINDOW, STALL_RATIO = 200, 0.9   # stalled: remaining mass fell < 10% in 200 steps
TAIL_MIN_STEPS = 30        # power steps before the Krylov tail may take over (walkers in transit)
TAIL_SWITCH_AFTER = 500    # ... when power iteration would still need more steps than this
TAIL_MAX_PRODUCTS = 3000   # for the whole Krylov tail, all rounds together
TAIL_ROUNDS = 4            # Krylov rounds on what the previous one left, before power iteration
FLOAT_BLOCKS_PER_COLUMN = 6    # float64 arrays per solution column at the peak
OVERLENGTH_SURVIVAL = 0.5      # states where more walkers than this outlast the soft max length


# ------------------------------------------------------------------ Krylov tail
def _cdot(a, b):
    return np.einsum("ij,ij->j", a, b)


def bicgstab(apply, b, tol_abs, rows, max_mv=TAIL_MAX_PRODUCTS, restart=50):
    """BiCGSTAB on every column of b in lockstep (one product per half-step shared by all
    columns).  Converged columns are frozen; a column whose residual has not halved in
    ``restart`` iterations restarts.  Returns (x, products, residual 1-norm over ``rows``)."""
    k = b.shape[1]
    x, r, rh = np.zeros_like(b), b.copy(), b.copy()
    p, v, t = np.zeros_like(b), np.zeros_like(b), np.empty_like(b)
    rho, alpha, omega = np.ones(k), np.ones(k), np.ones(k)
    res = np.abs(r[rows]).sum(0)
    done = res <= tol_abs
    checkpoint = res.copy()
    mv = 0

    def div(a, d):
        ok = (d != 0) & ~done
        return np.where(ok, a / np.where(d != 0, d, 1.0), 0.0)

    while not done.all() and mv < max_mv:
        rho_new = _cdot(rh, r)
        beta = div(rho_new, rho) * div(alpha, omega)
        p *= beta
        p -= v * (beta * omega)
        p += r
        apply(p, v)
        alpha = div(rho_new, _cdot(rh, v))
        r -= v * alpha
        x += p * alpha
        apply(r, t)
        mv += 2
        omega = div(_cdot(t, r), _cdot(t, t))
        x += r * omega
        r -= t * omega
        rho = np.where(done, rho, rho_new)
        res = np.abs(r[rows]).sum(0)
        if not np.isfinite(res).all():
            break
        stuck = (alpha == 0) & (omega == 0) & ~done
        if (mv // 2) % restart == 0:
            stuck |= ~done & (res > 0.5 * checkpoint)
            checkpoint = res.copy()
        if stuck.any():
            rh[:, stuck], p[:, stuck], v[:, stuck] = r[:, stuck], 0.0, 0.0
            rho[stuck] = alpha[stuck] = omega[stuck] = 1.0
        done |= res <= tol_abs
    return x, mv, res


def _bicgstab_cols(apply, b, tol_abs, rows, cols, ncols, max_mv=TAIL_MAX_PRODUCTS):
    """bicgstab on columns ``cols`` of an operator ``apply`` that works on all ``ncols``
    columns (the others zero; the operator acts column by column)."""
    def sub(X, out):
        full = np.zeros((X.shape[0], ncols))
        full[:, cols] = X
        res = np.empty_like(full)
        apply(full, res)
        out[...] = res[:, cols]

    return bicgstab(sub, b, tol_abs, rows, max_mv=max_mv)


def _crawling(hist, it, tol, window=10):
    """The remaining mass decays geometrically and power iteration would still need more
    than TAIL_SWITCH_AFTER steps to reach ``tol``."""
    if it < TAIL_MIN_STEPS or it % window or len(hist) < 2 * window + 1 or hist[-1] <= 0:
        return False
    q1 = (hist[-1] / hist[-1 - window]) ** (1.0 / window)
    q0 = (hist[-1 - window] / hist[-1 - 2 * window]) ** (1.0 / window)
    return 0 < q1 < 1 and abs(q1 - q0) < 0.02 and np.log(tol / hist[-1]) / np.log(q1) > TAIL_SWITCH_AFTER


def solve_series(mul, x0, mask, tol, max_iter, what, ncomp=1, verbose=False, accel=False, krylov=True,
                 on_step=None, ops=None):
    """total = sum_t (M A)^t x0 per column (M = diag(mask)), A x = mul(x, out), by power
    iteration until the remaining mass (mass rows, relative to the largest column's start) is
    below ``tol``.  Stops early when it stalls (mass circulating in states that almost never
    get absorbed) and reports where that mass sits as ``info["trapped"]``.  ``x0`` is consumed.
    ``verbose``: log the time per step (matrix product / masking and summing) every 100 steps.
    ``accel``: False (default), True / "mr" (invariant-preserving Periodic Minimal Residual
    extrapolation), or "aitken" (scalar Aitken delta-squared extrapolation).
    ``krylov``: when power iteration crawls or stalls, solve the rest of the series with BiCGSTAB.
    ``on_step``: optional callback ``on_step(it, res, total, term)`` invoked after each iteration
        (``term``: that iteration's term of the series, i.e. the current wavefront; numpy arrays).
    ``ops``: the arrays' operations (default: numpy arrays on the host, ``HostOps``; arrays on a
    GPU: ``mpat.gpu.DeviceOps``).  Extrapolation and the Krylov tail run on host arrays only.
    Returns (total, info)."""
    ops = ops or HostOps(ncomp=ncomp)
    host = isinstance(ops, HostOps)
    use_accel = (accel is not False and accel is not None)
    if use_accel and not host:
        log.warning("%s solve: extrapolation (accel) needs host arrays; plain power iteration on the GPU", what)
        use_accel = False
    accel_kind = "mr" if accel is True else str(accel)
    burn_in, period, omega_max = 25, 8, 25.0
    t_start = time.perf_counter()
    t_mul = t_acc = 0.0
    w_mul = w_acc = 0.0      # since the last verbose line
    total = ops.copy(x0)
    term, nxt = x0, ops.empty_like(x0)
    mask = ops.contiguous(mask)
    rows = slice(None, None, ncomp)
    scale = max(float(ops.col_mass(x0).max()), 1e-300)
    info = dict(iterations=0, residual=0.0, stalled=False, extrapolated_jumps=0)
    hist, ref, tried = [], None, (not krylov) or use_accel or not host
    for it in range(1, max_iter + 1):
        t0 = time.perf_counter()
        mul(term, nxt)
        t1 = time.perf_counter()
        do_extrap = use_accel and (it >= burn_in and (it - burn_in) % period == 0)
        if not do_extrap:
            res = float(ops.accumulate(nxt, mask, total).max()) / scale
        else:
            stats = K._step_stats(term, nxt, mask, ncomp)
            u1, w1, uu, uw, ww = stats[:, 0], stats[:, 1], stats[:, 2], stats[:, 3], stats[:, 4]
            res = float(w1.max()) / scale
            C = x0.shape[1]
            omega = np.ones(C)
            if res >= tol:
                for c in range(C):
                    if w1[c] / scale < tol:
                        continue
                    if accel_kind == "mr":
                        rr = uu[c] - 2.0 * uw[c] + ww[c]
                        ur = uw[c] - uu[c]
                        if rr > 1e-300 and ur < 0.0:
                            om = 1.0 - ur / rr
                            if om > 1.05:
                                omega[c] = min(om, omega_max)
                    elif accel_kind == "aitken":
                        if u1[c] > 1e-300 and 0.1 * u1[c] < w1[c] < 0.9999 * u1[c]:
                            lam = w1[c] / u1[c]
                            omega[c] = min(1.0 / (1.0 - lam), omega_max)
            res = float(K._extrapolate_accumulate(term, nxt, total, omega, ncomp).max()) / scale
            if np.any(omega > 1.05):
                info["extrapolated_jumps"] += 1
        t2 = time.perf_counter()
        t_mul += t1 - t0
        t_acc += t2 - t1
        w_mul += t1 - t0
        w_acc += t2 - t1
        term, nxt = nxt, term
        hist.append(res)
        info.update(iterations=it, residual=res)
        if on_step is not None:
            if host:
                on_step(it, res, total, term)
            else:
                on_step(it, res, ops.to_host(total), ops.to_host(term))
        if it % 100 == 0:
            log.info("  %s step %d: remaining mass %.2e", what, it, res)
            if verbose:
                log.info("  [timing] %s steps %d-%d: %.1f ms/step (matrix product %.1f, mask+sum %.1f); "
                         "%d rows x %d columns", what, it - 99, it, 10 * (w_mul + w_acc), 10 * w_mul,
                         10 * w_acc, x0.shape[0], x0.shape[1])
                w_mul = w_acc = 0.0
        if res < tol:
            break
        if not np.isfinite(res) or res > 1e6:
            raise RuntimeError(f"{what} solve diverges (remaining mass {res:.1e} after {it} steps)")
        stall = False
        if it % STALL_WINDOW == 0:
            stall = ref is not None and res > STALL_RATIO * ref
            ref = res
        if not tried and (stall or _crawling(hist, it, tol)):
            tried = True
            tol_abs = tol * scale
            act = np.flatnonzero(np.abs(term[rows]).sum(0) > tol_abs)
            tmp = np.empty((term.shape[0], act.size))
            msk = np.ascontiguousarray(mask[:, act])

            def apply(X, out):
                mul(np.ascontiguousarray(X), tmp)
                tmp[...] *= msk
                np.subtract(X, tmp, out=out)

            log.info("  %s: switching to the Krylov tail at step %d (remaining mass %.1e, %s; %d columns)",
                     what, it, res, "stalled" if stall else "decaying too slowly for power iteration", act.size)
            t0_k = time.time()
            pending = np.ascontiguousarray(term[:, act])
            added = np.ones(act.size, bool)
            live = np.ones(act.size, bool)
            mv_all, rounds = 0, 0
            resid = np.empty_like(pending)
            for rounds in range(1, TAIL_ROUNDS + 1):
                cols = np.flatnonzero(live)
                budget = TAIL_MAX_PRODUCTS - mv_all
                if not cols.size or budget < 2:
                    break
                every = cols.size == act.size
                b = pending if every else np.ascontiguousarray(pending[:, cols])
                r0 = np.abs(b[rows]).sum(0)
                y, mv, _ = bicgstab(apply, b, tol_abs, rows, max_mv=budget) if every else \
                    _bicgstab_cols(apply, b, tol_abs, rows, cols, act.size, max_mv=budget)
                mv_all += mv
                if every:
                    apply(y, resid)
                    np.subtract(b, resid, out=resid)
                    r = resid
                else:
                    full = np.zeros_like(pending)
                    full[:, cols] = y
                    apply(full, resid)
                    del full
                    r = b - resid[:, cols]
                r1 = np.abs(r[rows]).sum(0)
                ok = np.isfinite(r1) & (r1 < 0.5 * r0)
                acc = cols[ok]
                total[:, act[acc]] += y[:, ok] - np.where(added[acc], 1.0, 0.0) * b[:, ok]
                del y
                pending[:, acc] = r[:, ok]
                added[acc] = False
                live[cols[~ok]] = False
                live[acc[r1[ok] <= tol_abs]] = False
                del r
            del tmp, resid
            add = ~added
            total[:, act[add]] += pending[:, add]
            term[:, act] = pending
            left = np.abs(term[rows]).sum(0)
            res = float(left.max()) / scale
            info.update(tail_started_at=it, tail_products=int(mv_all), tail_rounds=rounds, residual=res,
                        tail_seconds=round(time.time() - t0_k, 1))
            log.info("  %s: Krylov tail from step %d, %d products in %d round(s), remaining %.1e (%.0fs)",
                     what, it, mv_all, rounds, res, time.time() - t0_k)
            if res < tol:
                break
            n_left = int((left[act] > tol_abs).sum())
            log.info("  %s: %d of %d columns above tol after the Krylov tail continue with power "
                     "iteration", what, n_left, act.size)
            stall = False
        if stall:
            info["stalled"] = True
            log.warning("%s solve stalled at step %d: remaining mass %.2e circulates in states "
                        "that (almost) never get absorbed; stopping here", what, it, res)
            break
    if info["residual"] >= tol:     # stalled or max_iter reached
        info["trapped"] = ops.row_mass(term) / scale
    n = max(info["iterations"], 1)
    info.update(seconds=round(time.perf_counter() - t_start, 1), ms_per_step=round(1e3 * (t_mul + t_acc) / n, 2))
    if verbose:
        log.info("  [timing] %s: %d steps in %.1fs, %.1f ms/step (matrix product %.1f, mask+sum %.1f)",
                 what, n, info["seconds"], 1e3 * (t_mul + t_acc) / n, 1e3 * t_mul / n, 1e3 * t_acc / n)
    return total, info


# ------------------------------------------------------------------ soft maximum length
def overlength_states(chain, max_length_mm, zooms, ops=None):
    """Direction states from which most walkers are still walking after ``max_length_mm``:
    the survival probability u_T = Q^T 1 (Q: one step among white-matter states) after
    T = max_length / smallest voxel side steps -- every step crosses into a neighbouring voxel,
    so a walk of T steps is at least that long.  Traps (white matter the turning limit makes
    hard to leave, e.g. a ring around a cavity) keep u_T near 1, ordinary states drain
    within a tract's length.  Bundle ROIs only add ways out, so this holds for every bundle.
    ``ops``: where the products run (default: the host).
    Returns (kill: bool per state, survival per state, T)."""
    ops = ops or HostOps(chain)
    T = int(np.ceil(max_length_mm / float(np.min(np.abs(zooms)))))
    nc = chain.ncomp
    u = np.zeros((chain.n_states * nc, 1))
    u[::nc] = 1.0                                        # the mass rows (moments: position-averaged)
    u = ops.from_host(u)
    out = ops.empty_like(u)
    for _ in range(T):
        ops.mul("backward", u, out)
        u, out = out, u
    surv = ops.to_host(u)[::nc, 0].copy()
    return surv >= OVERLENGTH_SURVIVAL, surv, T


# ------------------------------------------------------------------ launches
@dataclass
class Launches:
    """First exits of every half-walk of every seed (independent of the bundle)."""
    seeds: np.ndarray      # (S, 3) voxel indices
    seed_lin: np.ndarray   # (S,) padded linear indices
    pair_seed: np.ndarray  # (P,) seed of each streamline (peak pair)
    pair_w: np.ndarray     # (P,) weight; sums to 1 per seed
    pair_d: np.ndarray     # (P, 2) the two signed peak directions (sphere vertices)
    hseed: np.ndarray      # (2P,) padded linear index of each half's seed (partner = h ^ 1)
    lptr: np.ndarray       # (2P + 1,) sparse first exits: keys n * nb + b, values (ncomp,)
    lkey: np.ndarray
    lval: np.ndarray
    dead: np.ndarray       # (2P,)

    @property
    def nbytes(self):
        return self.lval.nbytes + self.lkey.nbytes + self.lptr.nbytes


def prepare_launches(chain, seeds, chunk=4096):
    """One streamline per seed voxel, split over its fODF peak pairs {p, -p}."""
    t0 = time.time()
    seeds = np.asarray(seeds, np.int64).reshape(-1, 3)
    seed_lin = chain.lin(seeds)
    anti = chain.geo.antipode()
    pair_seed, pair_w, pair_d = [], [], []
    for si, v in enumerate(seed_lin):
        peaks = chain.seed_peaks(v)
        pairs = {}
        for p in peaks:
            key = (min(p, anti[p]), max(p, anti[p]))
            pairs[key] = pairs.get(key, 0.0) + 1.0 / peaks.size
        for (p, q), w in pairs.items():
            pair_seed.append(si), pair_w.append(w), pair_d.append((p, q))
    pair_seed = np.asarray(pair_seed, np.int64)
    pair_w = np.asarray(pair_w, np.float64)
    pair_d = np.asarray(pair_d, np.int64).reshape(-1, 2)
    P, nc, nb = pair_seed.size, chain.ncomp, chain.nb
    counts = np.zeros(2 * P, np.int64)
    keys, vals = [], []
    dead = np.empty(2 * P)
    for s0 in range(0, P, chunk):
        n = min(P, s0 + chunk) - s0
        o = np.empty((2 * n, 26, nb, nc))
        d = np.empty(2 * n)
        v_chunk = seed_lin[pair_seed[s0:s0 + n]]
        if n > 0:
            chg = np.flatnonzero(v_chunk[1:] != v_chunk[:-1]) + 1
            starts = np.concatenate([[0], chg])
            u_vlin = v_chunk[starts]
            dptr = 2 * np.concatenate([starts, [n]]).astype(np.int64)
        else:
            u_vlin = np.zeros(0, np.int64)
            dptr = np.zeros(1, np.int64)
        chain.launch_halves(u_vlin, dptr, pair_d[s0:s0 + n].reshape(-1), o, d)
        hh, nn, bb = np.nonzero(np.any(o != 0, axis=-1))
        counts[2 * s0:2 * (s0 + n)] = np.bincount(hh, minlength=2 * n)
        keys.append((nn * nb + bb).astype(np.int16))
        vals.append(o[hh, nn, bb])
        dead[2 * s0:2 * (s0 + n)] = d
    L = Launches(seeds, seed_lin, pair_seed, pair_w, pair_d, np.repeat(seed_lin[pair_seed], 2),
                 np.concatenate([[0], np.cumsum(counts)]).astype(np.int64),
                 np.concatenate(keys) if keys else np.zeros(0, np.int16),
                 np.concatenate(vals) if vals else np.zeros((0, nc)), dead)
    log.info("launches: %d streamlines (peak pairs) from %d seeds in %.0fs (%.0f MB)", P, len(seeds),
             time.time() - t0, L.nbytes / 1e6)
    return L


# ------------------------------------------------------------------ bundles
@dataclass
class BundleResult:
    name: str
    seed_vox: np.ndarray     # (S, 3)
    seed_prob: np.ndarray    # (S,) P(the streamline from this seed is in the bundle)
    density_t: np.ndarray    # (Nt,) expected visits by bundle streamlines, all seeds
    endpoints_a: np.ndarray  # (Na,) expected bundle endpoints per absorbing voxel
    info: dict = field(default_factory=dict)
    density_class: np.ndarray = None  # (Nt, 26) the visits split by the walker's direction class
    fields: object = None    # mpat.fields.BundleFields when solved with keep_fields=True
    flux26: np.ndarray = None        # (Nt, 26) expected steps to each of 26 neighbours (when ncomp == 1)
    start_mask: np.ndarray = None    # optional start ROI mask on the chain grid
    end_mask: np.ndarray = None      # optional end ROI mask on the chain grid


@dataclass
class SolveStep:
    """What a :func:`segment_bundles` ``on_step`` callback gets after each series iteration."""
    phase: str               # "backward" (completion probabilities) or "forward" (occupancy)
    it: int                  # iteration number within the phase
    res: float               # relative residual
    total: np.ndarray        # solution so far, (n_states * ncomp, columns)
    term: np.ndarray         # this iteration's term of the series: the current wavefront
    colsign: np.ndarray      # inclusion-exclusion sign of each column
    n_kinds: int             # end kinds of the first bundle (its columns 0 .. n_kinds - 1 avoid nothing)
    beta: np.ndarray = None  # the backward solution; forward phase only


def _endpoint_kinds(a_valid, a_start, a_end):
    """Disjoint end categories per half and the non-negative bilinear form M >= 0:
    P(both ends acceptable) = sum_kl M[k, l] V_k(h1) V_l(h2)."""
    if a_start is None and a_end is None:
        return [a_valid], np.array([[1.0]]), "both ends valid"
    if a_start is None or a_end is None:
        E = a_valid & (a_end if a_end is not None else a_start)
        R = a_valid & ~E
        return [E, R], np.array([[1.0, 1.0], [1.0, 0.0]]), "one end in the endpoint ROI"
    S, E = a_valid & a_start, a_valid & a_end
    if not (S & E).any():
        return [S, E], np.array([[0.0, 1.0], [1.0, 0.0]]), "one end in start, the other in end"
    B = S & E
    return [S & ~B, E & ~B, B], np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]), "one end in start, the other in end"


class _Plan:
    """A bundle's avoid sets, end kinds and solution columns."""

    def __init__(self, chain, spec, valid_classes):
        self.name, self.t0 = spec["name"], time.time()
        self.start_mask = spec.get("start")
        self.end_mask = spec.get("end")
        pad = np.zeros(chain.pshape, bool)

        def padded(m):
            p = pad.copy()
            p[1:-1, 1:-1, 1:-1] = m
            return p.ravel()

        inc = [padded(m) for m in spec.get("include", ())]
        exc = padded(spec["exclude"]) if spec.get("exclude") is not None else pad.ravel()
        self.subsets = [T for r in range(len(inc) + 1) for T in combinations(range(len(inc)), r)]
        self.signs = np.array([(-1.0) ** len(T) for T in self.subsets])
        self.nX = len(self.subsets)
        self.taboo = np.zeros((pad.size, self.nX), np.bool_)
        for x, T in enumerate(self.subsets):
            tb = exc.copy()
            for i in T:
                tb |= inc[i]
            self.taboo[:, x] = tb
        a_valid = np.isin(chain.a_class, np.asarray(valid_classes))
        ends = [padded(spec[k])[chain.a_lin] if spec.get(k) is not None else None for k in ("start", "end")]
        kinds, self.M, self.end_rule = _endpoint_kinds(a_valid, *ends)
        self.nK = len(kinds)
        self.C = self.nX * self.nK
        self.xcol = np.repeat(np.arange(self.nX), self.nK)
        self.kcol = np.tile(np.arange(self.nK), self.nX)
        self.absw = np.stack([kinds[self.kcol[c]] & ~self.taboo[chain.a_lin, self.xcol[c]]
                              for c in range(self.C)], 1).astype(np.float64)
        st = np.repeat(chain.t_lin, chain.nb * chain.ncomp)
        self.mask = (~self.taboo[st][:, self.xcol]).astype(np.uint8)
        kill = getattr(chain, "kill", None)      # states past the soft maximum length: walkers end there
        if kill is not None:
            self.mask[np.repeat(kill, chain.ncomp)] = 0


def max_columns(chain, launches, max_mem, blocks=FLOAT_BLOCKS_PER_COLUMN):
    """Solution columns that fit in ``max_mem`` bytes next to the chain and the launches."""
    per_col = chain.n_states * chain.ncomp * (8 * blocks + 1)
    free = max_mem - chain.nbytes - launches.nbytes - 1e9
    return int(max(1, min(256, free // per_col)))


FLOAT_BLOCKS_INTERLEAVED = 7   # r, beta, pi0, S and a work array per solve, at the start


# ------------------------------------------------------------------ the arrays' operations
class HostOps:
    """The solves' arrays and their operations on the host: numpy arrays, numba kernels.

    Every solve goes through one of these (``mpat.gpu.WarpOps`` is the same on a GPU):
    allocating and copying arrays, a matrix product with the chain, one step of a series
    (product, mask, sum, and what the step added per column), and the column groups
    (``group``: masks, absorption, the forward start, the bundles' maps)."""

    name = "numba"

    def __init__(self, chain=None, launches=None, ncomp=None):
        self.chain, self.L = chain, launches
        self.nc = int(ncomp if ncomp is not None else (chain.ncomp if chain is not None else 1))

    # arrays
    def empty_like(self, a):
        return np.empty_like(a)

    def copy(self, a):
        return a.copy()

    def contiguous(self, a):
        return np.ascontiguousarray(a)

    def masked(self, a, mask):
        return a * mask

    def from_host(self, a):
        return np.asarray(a)

    def to_host(self, a):
        return a

    # the solves
    def mul(self, what, x, out):
        return (self.chain.mul_forward if what == "forward" else self.chain.mul_backward)(x, out)

    def accumulate(self, nxt, mask, total):
        """nxt *= mask; total += nxt; per column, the mass nxt carries (mass rows)."""
        return K.mask_accumulate(nxt, mask, total, self.nc)

    def step(self, nxt, mask, src, cur):
        """nxt = src + mask * nxt (nxt holding A cur); per column, |nxt - cur| (mass rows)."""
        return K.step_source(nxt, mask, src, cur, self.nc)

    def col_mass(self, x):
        return np.abs(x[::self.nc]).sum(0)

    def row_mass(self, x, minus=None):
        """Per state, |x| (or |x - minus|) of its mass row, summed over the columns (float32)."""
        d = x[::self.nc] if minus is None else x[::self.nc] - minus[::self.nc]
        return np.abs(d).sum(1).astype(np.float32)

    # column groups
    def max_columns(self, max_mem, live=False):
        return max_columns(self.chain, self.L, max_mem, FLOAT_BLOCKS_INTERLEAVED if live else FLOAT_BLOCKS_PER_COLUMN)

    def group(self, plans):
        return HostGroup(self.chain, plans, self.L)

    def same_place(self, other):
        """Whether arrays of ``other`` can be used here (an earlier solution, as a start)."""
        return type(other) is type(self)


def make_ops(chain, launches=None, backend=None, device=None, **kwargs):
    """The solves' ops: ``backend`` "numba" (default; the host), or on a GPU "warp" or
    "numba-cuda" (``mpat.gpu.DeviceOps`` on ``device``, default the first CUDA device;
    ``kwargs`` go to it)."""
    backend = (backend or "numba").lower()
    if backend in ("numba", "host", "cpu"):
        return HostOps(chain, launches)
    if backend in ("warp", "gpu", "cuda", "numba-cuda"):
        from .gpu import DeviceOps

        return DeviceOps(chain, launches, device=device, backend="numba-cuda" if backend == "numba-cuda" else "warp",
                         **kwargs)
    raise ValueError(f"unknown backend {backend!r}: 'numba', 'warp' or 'numba-cuda'")


class HostGroup:
    """A column group on the host: the columns' offsets, the transient mask and the
    absorption right-hand side (the plans' own masks are released), the forward start from a
    backward solution, and each bundle's maps from the two solutions."""

    def __init__(self, chain, plans, L):
        self.chain, self.plans, self.L = chain, list(plans), L
        self.offs = np.cumsum([0] + [p.C for p in self.plans])
        self.mask = np.concatenate([p.mask for p in self.plans], axis=1)
        for p in self.plans:
            p.mask = None
        r = np.concatenate([chain.absorb_rhs(p.absw) for p in self.plans], axis=1)
        r *= self.mask
        self.r = r

    def forward_start(self, beta, pi0=None):
        """The forward solve's start from the backward solution ``beta``: every half-walk's
        launch, weighted by its partner's probability of completing the bundle's requirements.
        Returns (pi0, [(mass absorbed on the launch step, seed probabilities, pair
        probabilities)] per plan); ``pi0`` is refilled in place if given."""
        chain, L = self.chain, self.L
        nb, Ns, nc = chain.nb, chain.n_states, chain.ncomp
        P = L.pair_seed.size
        if pi0 is None:
            pi0 = np.zeros((Ns * nc, int(self.offs[-1])))
        else:
            pi0[:] = 0.0
        starts = []
        for p, o0 in zip(self.plans, self.offs[:-1]):
            seed_ok = (~p.taboo[L.hseed][:, p.xcol]).astype(np.float64)
            vals = np.empty((L.hseed.size, p.C))
            K.half_values(L.hseed, L.lptr, L.lkey, L.lval, chain.label, chain.noff, chain.tindex,
                          chain.aindex, nb, beta, p.absw, seed_ok, vals, int(o0))
            V = vals.reshape(P, 2, p.nX, p.nK)
            pb = np.einsum("pxk,kl,pxl->px", V[:, 0], p.M, V[:, 1]) @ p.signs   # P(pair in bundle)
            seed_prob = np.bincount(L.pair_seed, weights=L.pair_w * pb, minlength=len(L.seeds))
            coef = np.einsum("phxl,kl->phxk", V[:, ::-1], p.M).reshape(2 * P, p.C)
            coef *= np.repeat(L.pair_w, 2)[:, None] * seed_ok
            a0 = np.zeros((chain.a_lin.size, p.C))
            K.scatter_init(L.hseed, L.lptr, L.lkey, L.lval, coef, chain.label, chain.noff, chain.tindex,
                           chain.aindex, nb, p.taboo, p.xcol, pi0, a0, int(o0))
            starts.append((a0, seed_prob, pb))
        if getattr(chain, "kill", None) is not None:
            pi0[np.repeat(chain.kill, nc)] = 0.0
        return pi0, starts

    def density(self, S, beta, i):
        """Plan ``i``'s density per white-matter voxel from the solutions (step 0 not included)."""
        ch, p = self.chain, self.plans[i]
        return K.voxel_density(S, beta, p.signs[p.xcol], int(self.offs[i]), ch.t_lin.size, ch.nb * ch.ncomp)

    def columns(self, X, i):
        """Plan ``i``'s columns of a group array, as a contiguous array."""
        return np.ascontiguousarray(X[:, int(self.offs[i]):int(self.offs[i + 1])])

    def keep(self, x):
        """A copy of a group array that outlives the solve consuming it (``mpat.fields``'
        launch, with keep_fields)."""
        return x.copy()

    def launch_absorbed(self, a0, i):
        """Plan ``i``'s mass absorbed on the launch step: the forward start's ``a0``."""
        return a0

    def maps(self, S, beta, starts):
        """Per plan, in order: (density per white-matter voxel and direction class, step 0 not
        included; expected bundle endpoints per absorbing voxel)."""
        ch = self.chain
        for p, o0, o1, (a0, _, _) in zip(self.plans, self.offs[:-1], self.offs[1:], starts):
            colsign = p.signs[p.xcol]
            S_p = np.ascontiguousarray(S[:, o0:o1])
            dcls = K.paired_density(S_p, np.ascontiguousarray(beta[:, o0:o1]), colsign, ch.n_states, ch.ncomp)
            yield dcls.reshape(ch.t_lin.size, ch.nb), ((a0 + ch.absorbed_from(S_p)) * p.absw) @ colsign


def _trapped(chain, info, what, names, on_trapped):
    """Hand the mass a solve could not finish (per voxel) to ``on_trapped``."""
    m = info.pop("trapped", None)
    if m is None or on_trapped is None:
        return
    vol = chain.to_volume(m.reshape(-1, chain.nb).sum(1))
    on_trapped(what, names, vol)


def _solve(ops, x0, mask, tol, max_iter, what, verbose=False, accel=False, on_step=None):
    """A group's ``what`` ("backward" / "forward") series on ``ops``, or, given a Chain (as
    mpat.trxviz_bridge does), on the host."""
    if not hasattr(ops, "group"):
        ops = HostOps(ops)
    return solve_series(lambda x, out: ops.mul(what, x, out), x0, mask, tol, max_iter, what, ops.nc, verbose,
                        accel=accel, krylov=False, on_step=on_step, ops=ops)


def _bundle_results(chain, plans, L, group, S, beta, starts, tol, info_b, info_f, on_result, gclock,
                    launch=None, arc_length=False, zooms=None):
    """The BundleResult of every plan of a solved column group (``on_result`` called with
    each), from the group's maps (``group.maps``); on the single-value chain also each
    bundle's edge flux (``flux26``), and with ``launch`` (the forward solve's start) its
    solution columns themselves (``fields``, mpat.fields.BundleFields)."""
    names = [p.name for p in plans]
    P = L.pair_seed.size
    t_dens = time.perf_counter()
    out = {}
    ti = chain.tindex[L.seed_lin]
    pt = ti[L.pair_seed]
    on = pt >= 0
    chord_w = None
    if arc_length:
        from .geometry import class_chord_lengths

        chord_w = class_chord_lengths(chain.geo, zooms)
    maps = group.maps(S, beta, starts)
    for i, (p, (dcls, endpoints), (a0, seed_prob, pb)) in enumerate(zip(plans, maps, starts)):
        # step 0 at the seed, half along each direction of the streamline's peak
        w0 = 0.5 * (L.pair_w * pb)[on]
        for k in (0, 1):
            np.add.at(dcls, (pt[on], chain.geo.vbin[L.pair_d[on, k]]), w0)
        if chord_w is not None:
            dcls = dcls * chord_w[None, :]
        density = dcls.sum(1)
        info = dict(n_seeds=int(len(L.seeds)), n_streamlines=int(P),
                    expected_bundle_streamlines=float(seed_prob.sum()), endpoint_rule=p.end_rule,
                    endpoint_mass_check=float(endpoints.sum() / max(2 * seed_prob.sum(), 1e-300)),
                    n_columns=int(p.C), tolerance=tol, backward=info_b, forward=info_f,
                    solved_with=[n for n in names if n != p.name],
                    negative_mass_before_clip=dict(density=float(-density[density < 0].sum()),
                                                   seed_prob=float(-seed_prob[seed_prob < 0].sum())),
                    seconds=round(time.time() - p.t0, 1))
        flux26 = S_p = beta_p = None
        if chain.ncomp == 1:
            S_p, beta_p = group.columns(S, i), group.columns(beta, i)
            f26 = np.empty((chain.t_lin.size, 26), np.float64)
            K.stencil_flux(chain.Q_ptr, chain.Q_slot, chain.Q_w, chain.tmpl_n, chain.tmpl_b, chain.nbr_t,
                           chain.nb, S_p, beta_p, p.signs[p.xcol], f26)
            flux26 = np.clip(f26, 0, None).astype(np.float32)
        res = BundleResult(p.name, L.seeds, np.clip(seed_prob, 0, None), np.clip(density, 0, None),
                           np.clip(endpoints, 0, None), info, np.clip(dcls, 0, None).astype(np.float32),
                           flux26=flux26, start_mask=p.start_mask, end_mask=p.end_mask)
        if launch is not None:
            from .fields import BundleFields

            res.fields = BundleFields(chain, p, S_p, beta_p, group.columns(launch, i), group.launch_absorbed(a0, i))
        del S_p, beta_p
        if on_result is not None:
            tw = time.perf_counter()
            on_result(res)
            gclock.add("writing outputs", time.perf_counter() - tw)
            t_dens += time.perf_counter() - tw
        out[p.name] = res
    gclock.add("densities and endpoints", time.perf_counter() - t_dens)
    return out


def _group_done(gclock, clock, t0, info_b, info_f):
    log.info("group done in %.0fs (backward %d steps, forward %d steps)", time.time() - t0,
             info_b["iterations"], info_f["iterations"])
    log.info("  group time: " + ", ".join(f"{k} {v:.3g}s" for k, v in
                                          sorted(gclock.t.items(), key=lambda kv: -kv[1]) if v >= 0.005))
    if clock is not None:
        for k, v in gclock.t.items():
            clock.add(k, v)


def _solve_group(chain, plans, L, tol, max_iter, on_result, on_trapped=None, verbose=False, clock=None,
                 ops=None, keep_fields=False, accel=False, arc_length=False, zooms=None, on_step=None):
    ops = ops or HostOps(chain, L)
    t0 = time.time()
    gclock = Clock()
    names = [p.name for p in plans]
    colsign_all = np.concatenate([p.signs[p.xcol] for p in plans])
    n_kinds = plans[0].nK

    def step_hook(phase, beta=None):
        if on_step is None:
            return None
        if beta is not None and not isinstance(ops, HostOps):
            beta = ops.to_host(beta)
        return lambda it, res, total, term: on_step(SolveStep(phase, it, res, total, term, colsign_all,
                                                               n_kinds, beta))

    if len(plans) > 1:
        log.info("solving %d bundles together (%d columns): %s", len(plans), sum(p.C for p in plans),
                 ", ".join(names))
    with gclock("masks and absorption"):
        g = ops.group(plans)
    with gclock("backward solve"):
        beta, info_b = _solve(ops, g.r, g.mask, tol, max_iter, "backward", verbose, accel=accel,
                              on_step=step_hook("backward"))
    g.r = None                      # consumed by the solve
    with gclock("trapped maps"):
        _trapped(chain, info_b, "backward", names, on_trapped)
    with gclock("seed weights (forward start)"):
        pi0, starts = g.forward_start(beta)
    launch = g.keep(pi0) if keep_fields else None      # solve_series consumes pi0
    with gclock("forward solve"):
        S, info_f = _solve(ops, pi0, g.mask, tol, max_iter, "forward", verbose, accel=accel,
                           on_step=step_hook("forward", beta))
    with gclock("trapped maps"):
        _trapped(chain, info_f, "forward", names, on_trapped)
    del pi0
    g.mask = None
    out = _bundle_results(chain, plans, L, g, S, beta, starts, tol, info_b, info_f, on_result, gclock,
                          launch=launch, arc_length=arc_length, zooms=zooms)
    _group_done(gclock, clock, t0, info_b, info_f)
    return out


# ------------------------------------------------------------------ both solves together
class Interleaved:
    """A column group's backward and forward solves advanced together, so that the bundle
    density exists from the first steps (a live view), and a solve can start from an earlier
    solution (ROIs edited by hand).

    Both solves are run as fixed-point steps x <- b + M A x.  Started from x = b these are the
    partial sums of the power series ``solve_series`` adds up, one more term per step:

        backward   beta <- M r   + M Q   beta
        forward    S    <- pi0   + M Q^T S

    The forward start pi0 depends on beta (each half-walk's launch is weighted by its
    partner's probability of completing the bundle's requirements), so it is recomputed from
    the current beta every ``source_every`` steps, and once more when the backward solve has
    converged; from then on the forward solve runs on alone to the tolerance.  The solution
    is ``_solve_group``'s to the tolerance, in about as many matrix products.

    Until then, ``density`` is the bundle so far: only streamlines short enough for both of
    their halves to have been walked count yet.  Started from zero it grows to the solution,
    never past it; it only ever grows for a bundle without include ROIs on the single-value
    chain (inclusion-exclusion's alternating signs, and the moment chain's signed values, can
    make it dip a little on the way; it is clipped at 0).

    ``start``: a solved (or stopped) ``Interleaved`` with the same number of columns on the
    same chain (and the same kind of ``ops``), to start from its solution instead of from
    zero: far fewer steps when the bundle's ROIs changed a little.  The iteration converges
    from any start.  ``ops``: where the arrays live (default: the host, ``HostOps``).

        sol = Interleaved(chain, plans, launches)
        while not sol.step():
            show(sol.volume(0))             # the first bundle so far
        results = sol.results()
    """

    def __init__(self, chain, plans, L, tol=1e-6, max_iter=5000, source_every=4, on_trapped=None,
                 verbose=False, start=None, ops=None):
        self.chain, self.plans, self.L = chain, list(plans), L
        self.ops = ops = ops or HostOps(chain, L)
        self.tol, self.max_iter = tol, max_iter
        self.source_every = max(1, int(source_every))
        self.on_trapped, self.verbose = on_trapped, verbose
        self.names = [p.name for p in self.plans]
        self.clock, self.t0 = Clock(), time.time()
        if len(self.plans) > 1:
            log.info("solving %d bundles together (%d columns): %s", len(self.plans),
                     sum(p.C for p in self.plans), ", ".join(self.names))
        with self.clock("masks and absorption"):
            self.group = g = ops.group(self.plans)
        self.offs, self.mask, self.r = g.offs, g.mask, g.r
        self._scale_b = max(float(ops.col_mass(self.r).max()), 1e-300)
        warm = (start is not None and start.chain is chain and ops.same_place(start.ops)
                and tuple(start.beta.shape) == tuple(self.r.shape) and start.S is not None)
        if start is not None and not warm:
            log.info("the earlier solution has other columns; starting from zero")
        # step 0 of the series (or the earlier solution, on the states still allowed)
        self.beta = ops.masked(start.beta, self.mask) if warm else ops.copy(self.r)
        self._tmp_b = ops.empty_like(self.beta)
        with self.clock("seed weights (forward start)"):
            self.pi0, self.starts = g.forward_start(self.beta)
        self._set_scale_f()
        self.S = ops.masked(start.S, self.mask) if warm else ops.copy(self.pi0)
        self._tmp_f = ops.empty_like(self.S)
        self.warm = bool(warm)
        self.info_b = dict(iterations=0, residual=float("inf"), stalled=False)
        self.info_f = dict(iterations=0, residual=float("inf"), stalled=False)
        self.back_done = self.done = False
        self._ref_b = self._ref_f = None
        self._after = 0                 # forward steps since the backward solve finished
        self._t_b = self._t_f = 0.0
        ti = chain.tindex[L.seed_lin]
        self._pt = ti[L.pair_seed]
        self._on = self._pt >= 0

    # -------------------------------------------------------------- stepping
    def _set_scale_f(self):
        self._scale_f = max(float(self.ops.col_mass(self.pi0).max()), 1e-300)

    def _refresh_source(self):
        with self.clock("seed weights (forward start)"):
            _, self.starts = self.group.forward_start(self.beta, self.pi0)
        self._set_scale_f()

    def _stop(self, what, info, res, ref, steps):
        """Whether this solve ends at this step (converged, stalled, or out of steps); the
        mass it could not finish goes to ``on_trapped``.  Returns (stop, ref)."""
        if not np.isfinite(res) or res > 1e6:
            raise RuntimeError(f"{what} solve diverges (remaining mass {res:.1e} after {steps} steps)")
        if res < self.tol:
            return True, ref
        if steps % STALL_WINDOW == 0:
            if ref is not None and res > STALL_RATIO * ref:
                info["stalled"] = True
                log.warning("%s solve stalled at step %d: remaining mass %.2e circulates in states "
                            "that (almost) never get absorbed; stopping here", what, steps, res)
                return True, ref
            ref = res
        return steps >= self.max_iter, ref

    def _unfinished(self, what, info, new, old, scale):
        if info["residual"] >= self.tol:     # stalled or out of steps: where the mass still moves
            info["trapped"] = self.ops.row_mass(new, old) / scale
            with self.clock("trapped maps"):
                _trapped(self.chain, info, what, self.names, self.on_trapped)

    def _step(self):
        ops = self.ops
        if not self.back_done:
            t0 = time.perf_counter()
            ops.mul("backward", self.beta, self._tmp_b)
            res = float(ops.step(self._tmp_b, self.mask, self.r, self.beta).max()) / self._scale_b
            self.beta, self._tmp_b = self._tmp_b, self.beta
            self._t_b += time.perf_counter() - t0
            ib = self.info_b
            ib.update(iterations=ib["iterations"] + 1, residual=res)
            stop, self._ref_b = self._stop("backward", ib, res, self._ref_b, ib["iterations"])
            if stop:
                self.back_done = True
                self._unfinished("backward", ib, self.beta, self._tmp_b, self._scale_b)
                self._tmp_b = None
                ib.update(seconds=round(self._t_b, 1), ms_per_step=round(1e3 * self._t_b / ib["iterations"], 2))
            if stop or ib["iterations"] % self.source_every == 0:
                self._refresh_source()
        t0 = time.perf_counter()
        ops.mul("forward", self.S, self._tmp_f)
        res = float(ops.step(self._tmp_f, self.mask, self.pi0, self.S).max()) / self._scale_f
        self.S, self._tmp_f = self._tmp_f, self.S
        self._t_f += time.perf_counter() - t0
        i_f = self.info_f
        i_f.update(iterations=i_f["iterations"] + 1, residual=res)
        if self.back_done:      # the start no longer changes: the forward solve can finish
            self._after += 1
            stop, self._ref_f = self._stop("forward", i_f, res, self._ref_f, self._after)
            if stop:
                self.done = True
                self._unfinished("forward", i_f, self.S, self._tmp_f, self._scale_f)
                self._tmp_f = None
                i_f.update(seconds=round(self._t_f, 1), ms_per_step=round(1e3 * self._t_f / i_f["iterations"], 2))
        n = i_f["iterations"]
        if n % 100 == 0 and not self.done:
            log.info("  step %d: remaining mass backward %.2e, forward %.2e", n,
                     0.0 if self.back_done else self.info_b["residual"], res)
            if self.verbose:
                log.info("  [timing] %.1f ms/step so far (backward %.1f, forward %.1f); %d rows x %d columns",
                         1e3 * (self._t_b + self._t_f) / n, 1e3 * self._t_b / max(self.info_b["iterations"], 1),
                         1e3 * self._t_f / n, self.S.shape[0], self.S.shape[1])

    def step(self, n=1):
        """Advance up to ``n`` steps (a backward and a forward matrix product each, until the
        backward solve has converged; then forward ones).  Returns True once both have converged."""
        for _ in range(n):
            if self.done:
                break
            self._step()
        return self.done

    # -------------------------------------------------------------- the solution so far
    @property
    def steps(self):
        return self.info_f["iterations"]

    @property
    def remaining(self):
        """The larger of the two solves' remaining mass (relative; the solves stop below ``tol``)."""
        rb = 0.0 if self.back_done else self.info_b["residual"]
        return max(rb, self.info_f["residual"])

    @property
    def progress(self):
        """0..1: how far the remaining mass has fallen from 1 to ``tol``, on a log scale."""
        r = self.remaining
        if self.done:
            return 1.0
        if not np.isfinite(r) or r >= 1:
            return 0.0
        return float(np.log(r) / np.log(self.tol))

    def expected(self, i):
        """Expected number of streamlines of plan ``i``'s bundle, so far."""
        return float(self.starts[i][1].sum())

    def density(self, i):
        """Plan ``i``'s bundle density so far, per white-matter voxel (>= 0)."""
        ch = self.chain
        d = np.asarray(self.group.density(self.S, self.beta, i), np.float64)
        pb = self.starts[i][2]          # step 0, at the seed
        d += np.bincount(self._pt[self._on], weights=(self.L.pair_w * pb)[self._on], minlength=ch.t_lin.size)
        return np.clip(d, 0, None, out=d)

    def volume(self, i):
        """``density(i)`` as a volume on the chain's grid."""
        return self.chain.to_volume(self.density(i))

    def results(self, on_result=None, clock=None, keep_fields=False, arc_length=False, zooms=None):
        """{name: BundleResult}, as ``segment_bundles`` returns (the solves run on to the end
        first if they have not converged).  ``keep_fields``, ``arc_length``, ``zooms``: as for
        ``segment_bundles`` (the fields' launch: the forward start from the converged backward
        solve)."""
        while not self.done:
            self._step()
        ib, i_f = self.info_b, self.info_f
        for i in (ib, i_f):
            i["interleaved"] = True
        out = _bundle_results(self.chain, self.plans, self.L, self.group, self.S, self.beta, self.starts, self.tol,
                              ib, i_f, on_result, self.clock, launch=self.group.keep(self.pi0) if keep_fields else None,
                              arc_length=arc_length, zooms=zooms)
        self.clock.add("backward solve", self._t_b)
        self.clock.add("forward solve", self._t_f)
        _group_done(self.clock, clock, self.t0, ib, i_f)
        return out


def segment_bundles(chain, specs, launches, tol=1e-6, max_mem=8e9, max_iter=5000,
                    valid_classes=VALID_ENDS, on_result=None, on_trapped=None, verbose=False,
                    clock=None, keep_fields=False, accel=False, arc_length=False, zooms=None, on_step=None,
                    on_progress=None, source_every=4, ops=None, backend=None, device=None):
    """Segment bundles, sharing solves.  ``specs``: dicts with ``name`` and optional
    ``include`` (list of masks), ``exclude``, ``start``, ``end`` (boolean volumes on the chain
    grid).  Bundles are packed in order into groups whose columns fit in ``max_mem`` bytes;
    each group costs one backward and one forward series.
    ``on_trapped(direction, names, volume)``: called when a solve stalls or runs out of steps,
    with the mass it could not finish per voxel (relative to the largest column's start).
    ``on_progress(solve)``: advance each group's two solves together (``Interleaved``) and
    call this after every step, and once more when the group has converged, with the solve:
    ``solve.volume(i)`` is bundle ``solve.names[i]`` so far, ``solve.remaining`` the mass
    left.  The results are the same to the tolerance.  ``source_every``: see ``Interleaved``.
    ``backend``: where the solves run, "numba" (default: the host) or "warp" (a GPU, on
    ``device``; mpat.gpu), or ``ops``: the ops themselves (``make_ops``).  On a GPU, groups are
    sized by the device's free memory.
    ``verbose``: timing inside the solves.  ``clock``: a Clock to add the time per stage to
    (default: a new one, reported at the end).
    ``keep_fields``: also keep each bundle's solution columns (occupancy, completion and
    launch mass per state) as ``result.fields`` (mpat.fields.BundleFields), for edge flux,
    committors and conservation checks.  Three more float64 arrays of n_states x columns per
    bundle; the plain chain only (not with spatial moments).
    ``accel``: False (default), True / "mr" (Periodic Minimal Residual), or "aitken".
    ``arc_length``: weight class visits by physical intra-voxel chord lengths (in mm if ``zooms``
    is provided, else in voxels).
    ``on_step(step)``: called with a :class:`SolveStep` after every backward and forward
    iteration (e.g. to stream the solve as it converges).
    Returns {name: BundleResult}."""
    if keep_fields and chain.ncomp != 1:
        raise ValueError("keep_fields needs the plain chain (ncomp == 1), not the moment chain")
    if keep_fields:
        max_mem = max_mem / 2    # the launch copy during the forward solve
    live = on_progress is not None
    if live and on_step is not None:
        raise ValueError("on_step follows the series of the two solves one after the other; "
                         "with on_progress they advance together: give one of them")
    if live and accel:
        log.warning("accel: the interleaved solves (on_progress) are not extrapolated")
    own = clock is None
    clock = clock or Clock()
    t_all = time.perf_counter()
    if ops is None:
        ops = make_ops(chain, launches, backend, device)
    else:
        ops.L = launches        # this call's launches (e.g. SIFT2's, which replace the seeds' own)
    cap = ops.max_columns(max_mem, live)
    log.info("column groups of up to %d columns (%s)", cap, ops.budget_note(max_mem) if hasattr(ops, "budget_note")
             else f"{max_mem / 1e9:.1f} GB budget")

    def solve(group):
        if not live:
            return _solve_group(chain, group, launches, tol, max_iter, on_result, on_trapped, verbose, clock, ops,
                                keep_fields=keep_fields, accel=accel, arc_length=arc_length, zooms=zooms,
                                on_step=on_step)
        sol = Interleaved(chain, group, launches, tol, max_iter, source_every, on_trapped, verbose, ops=ops)
        on_progress(sol)
        while not sol.step():
            on_progress(sol)
        on_progress(sol)
        return sol.results(on_result, clock, keep_fields=keep_fields, arc_length=arc_length, zooms=zooms)

    out, group, used = {}, [], 0
    it = iter(specs)
    while True:
        with clock("bundle ROIs"):          # specs may warp each bundle's ROIs as it is asked for
            spec = next(it, None)
        if spec is None:
            break
        with clock("bundle plans"):
            p = _Plan(chain, spec, valid_classes)
        if p.C > cap:
            log.warning("%s needs %d columns, more than fit in max_mem (%d); solving it alone",
                        p.name, p.C, cap)
        if group and used + p.C > cap:
            out.update(solve(group))
            group, used = [], 0
        group.append(p)
        used += p.C
    if group:
        out.update(solve(group))
    if own:
        clock.report("bundle segmentation time", time.perf_counter() - t_all)
    return out


def solve_bundle_cascade(chain, spec, launches, tol=1e-6, max_iter=5000, valid_classes=VALID_ENDS,
                         accel=False):
    """Non-negative Strong-Markov hitting cascade for a bundle with 1 include ROI (or 0 include
    ROIs, delegated to segment_bundles).  Computes the exact bundle density, seed probabilities
    and endpoints with zero signed inclusion-exclusion subtraction (every term >= 0)."""
    inc_list = list(spec.get("include", ()))
    if len(inc_list) == 0:
        return segment_bundles(chain, [spec], launches, tol=tol, max_iter=max_iter,
                               valid_classes=valid_classes, accel=accel)[spec["name"]]
    if len(inc_list) > 1:
        raise ValueError("solve_bundle_cascade currently supports 0 or 1 unordered include ROI")
    t0 = time.time()
    Ns, nb = chain.n_states, chain.nb
    pad = np.zeros(chain.pshape, bool)

    def padded(m):
        p = pad.copy()
        p[1:-1, 1:-1, 1:-1] = m
        return p.ravel()

    inc1 = padded(inc_list[0])
    exc = padded(spec["exclude"]) if spec.get("exclude") is not None else pad.ravel()
    st_vox = np.repeat(chain.t_lin, nb)
    base_ok = ~exc[st_vox]
    if getattr(chain, "kill", None) is not None:
        base_ok &= ~chain.kill

    in_I1 = base_ok & inc1[st_vox]
    mask_0 = base_ok.astype(np.uint8)
    mask_avoid1 = (base_ok & ~inc1[st_vox]).astype(np.uint8)

    a_valid = np.isin(chain.a_class, np.asarray(valid_classes))
    ends = [padded(spec[k])[chain.a_lin] if spec.get(k) is not None else None for k in ("start", "end")]
    kinds, M_end, end_rule = _endpoint_kinds(a_valid, *ends)
    nK = len(kinds)

    absw_0 = np.stack([kinds[k] & ~exc[chain.a_lin] for k in range(nK)], 1).astype(np.float64)
    absw_avoid1 = np.stack([kinds[k] & ~exc[chain.a_lin] & ~inc1[chain.a_lin] for k in range(nK)], 1).astype(np.float64)
    absw_in1 = absw_0 - absw_avoid1

    m_post = np.repeat(mask_0[:, None], nK, axis=1)
    m_av1 = np.repeat(mask_avoid1[:, None], nK, axis=1)

    r_post = chain.absorb_rhs(absw_0) * m_post
    beta_post, info_b1 = solve_series(chain.mul_backward, r_post, m_post, tol, max_iter, "b_post", 1, accel=accel)

    hit_on_roi = beta_post * in_I1[:, None]
    q_into_roi = np.empty_like(beta_post)
    chain.mul_backward(np.ascontiguousarray(hit_on_roi), q_into_roi)
    r_into_roi = chain.absorb_rhs(absw_in1)
    rhs_hit = (q_into_roi + r_into_roi) * m_av1
    rhs_avoid = chain.absorb_rhs(absw_avoid1) * m_av1

    rhs_both = np.concatenate([rhs_avoid, rhs_hit], axis=1)
    m_both = np.concatenate([m_av1, m_av1], axis=1)
    sol_both, info_b2 = solve_series(chain.mul_backward, rhs_both, m_both, tol, max_iter, "b_av_hit", 1, accel=accel)
    beta_avoid = sol_both[:, :nK]
    beta_hit = hit_on_roi + sol_both[:, nK:]

    P = launches.pair_seed.size
    H = launches.hseed.size
    seed_in_exc = exc[launches.hseed]
    seed_in_I1 = inc1[launches.hseed] & ~seed_in_exc
    seed_outside = ~seed_in_exc & ~inc1[launches.hseed]

    beta_all = np.ascontiguousarray(np.concatenate([beta_post, beta_avoid, beta_hit], axis=1))
    absw_all = np.concatenate([absw_0, absw_avoid1, absw_in1], axis=1)
    seed_ok_all = np.ones((H, 3 * nK))
    vals_all = np.empty((H, 3 * nK))
    K.half_values(launches.hseed, launches.lptr, launches.lkey, launches.lval,
                  chain.label, chain.noff, chain.tindex, chain.aindex, nb,
                  beta_all, absw_all, seed_ok_all, vals_all)
    V_post = vals_all[:, :nK]
    V_av = vals_all[:, nK:2 * nK]
    V_ht = vals_all[:, 2 * nK:]

    Vh_hit = np.where(seed_in_I1[:, None], V_post, np.where(seed_outside[:, None], V_ht, 0.0)).reshape(P, 2, nK)
    Vh_av = np.where(seed_outside[:, None], V_av, 0.0).reshape(P, 2, nK)

    pb = (
        np.einsum("pk,kl,pl->p", Vh_hit[:, 0], M_end, Vh_hit[:, 1])
        + np.einsum("pk,kl,pl->p", Vh_hit[:, 0], M_end, Vh_av[:, 1])
        + np.einsum("pk,kl,pl->p", Vh_av[:, 0], M_end, Vh_hit[:, 1])
    )
    seed_prob = np.bincount(launches.pair_seed, weights=launches.pair_w * pb, minlength=len(launches.seeds))

    V_part_any = Vh_hit[:, ::-1] + Vh_av[:, ::-1]
    V_part_hit = Vh_hit[:, ::-1]
    coef_must_hit = (np.einsum("phl,kl->phk", V_part_any, M_end) * launches.pair_w[:, None, None]).reshape(H, nK)
    coef_may_avoid = (np.einsum("phl,kl->phk", V_part_hit, M_end) * launches.pair_w[:, None, None]).reshape(H, nK)

    tab_exc = exc[:, None]
    tab_av1 = (exc | inc1)[:, None]
    xcol_0 = np.zeros(nK, np.int64)
    pi_must_hit = np.zeros((Ns, nK))
    pi_already_hit = np.zeros((Ns, nK))
    pi_may_avoid = np.zeros((Ns, nK))
    a0_must_in1 = np.zeros((chain.a_lin.size, nK))
    a0_already = np.zeros((chain.a_lin.size, nK))
    a0_avoid = np.zeros((chain.a_lin.size, nK))

    K.scatter_init(launches.hseed, launches.lptr, launches.lkey, launches.lval,
                   coef_must_hit * seed_outside[:, None], chain.label, chain.noff, chain.tindex, chain.aindex,
                   nb, tab_exc, xcol_0, pi_must_hit, a0_must_in1)
    K.scatter_init(launches.hseed, launches.lptr, launches.lkey, launches.lval,
                   coef_must_hit * seed_in_I1[:, None], chain.label, chain.noff, chain.tindex, chain.aindex,
                   nb, tab_exc, xcol_0, pi_already_hit, a0_already)
    K.scatter_init(launches.hseed, launches.lptr, launches.lkey, launches.lval,
                   coef_may_avoid * seed_outside[:, None], chain.label, chain.noff, chain.tindex, chain.aindex,
                   nb, tab_av1, xcol_0, pi_may_avoid, a0_avoid)

    pi_stage0 = np.concatenate([pi_must_hit * m_av1, pi_may_avoid * m_av1], axis=1)
    S_stage0, info_f1 = solve_series(chain.mul_forward, pi_stage0, np.concatenate([m_av1, m_av1], axis=1),
                                     tol, max_iter, "f_stage0", 1, accel=accel)
    S_pre = S_stage0[:, :nK]
    S_av = S_stage0[:, nK:]

    q_from_pre = np.empty_like(S_pre)
    chain.mul_forward(np.ascontiguousarray(S_pre), q_from_pre)
    flux_I1 = (pi_must_hit + q_from_pre) * in_I1[:, None]

    pi_stage1 = (pi_already_hit + flux_I1) * m_post
    S_post, info_f2 = solve_series(chain.mul_forward, pi_stage1, m_post, tol, max_iter, "f_stage1", 1, accel=accel)

    dcls = (S_pre * beta_hit + S_post * beta_post + S_av * beta_avoid).sum(1).reshape(chain.t_lin.size, nb)
    ti = chain.tindex[launches.seed_lin]
    pt = ti[launches.pair_seed]
    on = pt >= 0
    w0 = 0.5 * (launches.pair_w * pb)[on]
    for k in (0, 1):
        np.add.at(dcls, (pt[on], chain.geo.vbin[launches.pair_d[on, k]]), w0)
    density = dcls.sum(1)
    endpoints = (
        ((a0_already + chain.absorbed_from(S_post)) * absw_0)
        + ((a0_must_in1 + chain.absorbed_from(S_pre)) * absw_in1)
        + ((a0_avoid + chain.absorbed_from(S_av)) * absw_avoid1)
    ).sum(1)
    info = dict(
        n_seeds=int(len(launches.seeds)), n_streamlines=int(P),
        expected_bundle_streamlines=float(seed_prob.sum()), endpoint_rule=end_rule,
        endpoint_mass_check=float(endpoints.sum() / max(2 * seed_prob.sum(), 1e-300)),
        n_columns=int(3 * nK), tolerance=tol, backward=info_b2, forward=info_f1,
        negative_mass_before_clip=dict(density=float(-density[density < 0].sum()),
                                       seed_prob=float(-seed_prob[seed_prob < 0].sum())),
        seconds=round(time.time() - t0, 1),
    )
    return BundleResult(spec["name"], launches.seeds, seed_prob, density, endpoints, info, dcls.astype(np.float32))

