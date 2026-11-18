"""The numba-cuda GPU solves (mpat.gpu, backend "numba-cuda") against the host's, in numba's
CUDA simulator (no GPU needed; slow, so on a small phantom).  MPAT_TEST_DEVICE=cuda runs them
on the GPU instead.  The kernels are also compiled to PTX for real GPUs (without one)."""
import importlib.util
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from phantoms import crossing_phantom
from scipy import ndimage

if importlib.util.find_spec("numba_cuda") is None:
    pytest.skip("numba-cuda is not installed", allow_module_level=True)

import numba  # noqa: E402

REAL = os.environ.get("MPAT_TEST_DEVICE", "").startswith("cuda")
if not REAL:
    if "numba.cuda" in sys.modules and not numba.config.ENABLE_CUDASIM:
        pytest.skip("numba.cuda was imported without the simulator", allow_module_level=True)
    numba.config.ENABLE_CUDASIM = 1

from mpat.chain import VALID_ENDS, Chain, ChainConfig  # noqa: E402
from mpat.solve import HostOps, Interleaved, _Plan, overlength_states, prepare_launches, segment_bundles  # noqa: E402


def _small(moments=False):
    sh, wm, cls, reg = crossing_phantom(n=10, width=2)
    seeds = np.argwhere(ndimage.binary_dilation(reg["capx"] | reg["capy"], np.ones((3, 3, 3))) & wm)
    c = reg["center"][0]
    A = np.zeros_like(wm)
    A[c - 2] = True
    A &= wm
    B = np.zeros_like(wm)
    B[c + 2] = True
    B &= wm
    end = np.zeros_like(wm)
    end[-2:] = True
    end &= reg["capx"]
    ch = Chain.build(sh, (1.0, 1.0, 1.0), wm, cls, ChainConfig(moments=moments))
    return ch, prepare_launches(ch, seeds), A, B, end


@pytest.fixture(scope="module")
def small():
    return _small()


@pytest.fixture(scope="module")
def small_m():
    return _small(moments=True)


def _ops(ch, L, **kw):
    from mpat.gpu import NumbaCudaOps

    return NumbaCudaOps(ch, L, device=os.environ.get("MPAT_TEST_DEVICE") if REAL else None, **kw)


def _close(a, b, tol=1e-8):
    return np.abs(a - b).sum() <= tol * max(np.abs(b).sum(), 1e-300)


def _same(got, ref, tol=1e-8):
    for k, r in ref.items():
        g = got[k]
        assert _close(g.density_t, r.density_t, tol) and _close(g.seed_prob, r.seed_prob, tol)
        assert _close(g.endpoints_a, r.endpoints_a, tol) and _close(g.density_class, r.density_class, max(tol, 1e-7))
        assert (g.flux26 is None) == (r.flux26 is None)
        if r.flux26 is not None:                  # the edge flux, from the bundle's columns on the host
            assert _close(g.flux26, r.flux26, max(tol, 1e-7))


@pytest.mark.parametrize("moments", [False, True])
@pytest.mark.parametrize("segment", [2 ** 30, 20_000])          # one segment, or several
def test_products_match_the_host(small, small_m, moments, segment):
    ch, L, *_ = small_m if moments else small
    ops = _ops(ch, L, segment=segment)
    assert (len(ops.dc.q) > 1) == (segment < ch.n_transitions)
    rng = np.random.default_rng(0)
    R, C = ch.n_states * ch.ncomp, 2
    x = rng.random((R, C))
    x[rng.random(R) < 0.3] = 0.0
    for what in ("forward", "backward"):
        ref = np.empty_like(x)
        (ch.mul_forward if what == "forward" else ch.mul_backward)(x.copy(), ref)
        out = ops.zeros((R, C))
        ops.mul(what, ops.from_host(x), out)
        assert np.abs(ops.to_host(out) - ref).max() <= 1e-12 * np.abs(ref).max()
    a = rng.random((ch.a_lin.size, C))
    out = ops.zeros((R, C))
    ops.dc.absorb_rhs(ops.from_host(a), out)
    assert np.allclose(ops.to_host(out), ch.absorb_rhs(a), rtol=1e-12, atol=0)
    ab = ops.to_host(ops.dc.absorbed_from(ops.from_host(x), ops.zeros((ch.a_lin.size, C))))
    assert np.allclose(ab, ch.absorbed_from(x), rtol=1e-12, atol=1e-300)


def test_reductions_and_steps_match_the_host(small_m):
    ch, L, *_ = small_m
    ops, host = _ops(ch, L), HostOps(ch, L, ncomp=ch.ncomp)
    ops.nblk = 7
    rng = np.random.default_rng(1)
    R, C = ch.n_states * ch.ncomp, 3
    nxt, total, src, cur = (rng.standard_normal((R, C)) for _ in range(4))
    mask = (rng.random((R, C)) > 0.2).astype(np.uint8)
    d = {k: ops.from_host(v) for k, v in dict(nxt=nxt, total=total, src=src, cur=cur, mask=mask).items()}
    n2, t2 = nxt.copy(), total.copy()
    assert np.allclose(ops.accumulate(d["nxt"], d["mask"], d["total"]), host.accumulate(n2, mask, t2), rtol=1e-12)
    assert np.allclose(ops.to_host(d["nxt"]), n2) and np.allclose(ops.to_host(d["total"]), t2)
    d2, n3 = ops.from_host(nxt), nxt.copy()
    assert np.allclose(ops.step(d2, d["mask"], d["src"], d["cur"]), host.step(n3, mask, src, cur), rtol=1e-12)
    assert np.allclose(ops.to_host(d2), n3)
    assert np.allclose(ops.col_mass(d["src"]), host.col_mass(src), rtol=1e-12)
    assert np.allclose(ops.row_mass(d["src"], d["cur"]), host.row_mass(src, cur), rtol=1e-6)


def test_bundles_match_the_host(small):
    ch, L, A, B, end = small
    specs = [dict(name="x", include=[A, B], end=end), dict(name="a", include=[A])]
    ref = segment_bundles(ch, specs, L)
    got = segment_bundles(ch, specs, L, backend="numba-cuda",
                          device=os.environ.get("MPAT_TEST_DEVICE") if REAL else None)
    _same(got, ref)
    for k in ref:
        assert got[k].info["backward"]["iterations"] == ref[k].info["backward"]["iterations"]


def test_live_solve_on_the_gpu(small):
    ch, L, A, B, end = small
    spec = dict(name="x", include=[A, B], end=end)
    frames = []
    got = segment_bundles(ch, [spec], L, ops=_ops(ch, L), on_progress=lambda s: frames.append(s.density(0).sum()))
    _same(got, segment_bundles(ch, [spec], L), 1e-6)
    assert len(frames) > 2 and frames[-1] > frames[0]


def test_moments_and_overlength(small_m):
    ch, L, A, B, end = small_m
    ops = _ops(ch, L)
    k1, s1, T1 = overlength_states(ch, 12.0, (1.0, 1.0, 1.0))
    k2, s2, T2 = overlength_states(ch, 12.0, (1.0, 1.0, 1.0), ops=ops)
    assert T1 == T2 and (k1 == k2).all() and np.allclose(s1, s2, rtol=1e-12, atol=1e-300)
    specs = [dict(name="a", include=[A], end=end)]
    _same(segment_bundles(ch, specs, L, ops=ops), segment_bundles(ch, specs, L))


def test_the_same_place(small):
    ch, L, *_ = small
    ops = _ops(ch, L)
    plan = _Plan(ch, dict(name="a", end=small[4]), VALID_ENDS)
    host = Interleaved(ch, [plan], L)
    assert ops.same_place(_ops(ch, L)) and not ops.same_place(host.ops) and not host.ops.same_place(ops)


# every kernel, as MPAT launches it: its argument types
_I, _I2, _A, _V, _U2 = "int32[::1]", "int32[:, ::1]", "float64[:, ::1]", "float64[::1]", "uint8[:, ::1]"
_SIGS = [("bsr_backward", [_I, _I, "float32[:, ::1]", "int64", _A, _A]),
         ("bsr_forward", [_I, _I, "float32[:, ::1]", "int64", _A, _A]),
         ("csr_gather", [_I, _I, _V, "int64", _A, _A]), ("csr_scatter", [_I, _I, _V, "int64", _A, _A]),
         ("rblk_gather", [_I, _I, "float32[:, ::1]", "int64", _A, _A]),
         ("rblk_scatter", [_I, _I, "float32[:, ::1]", "int64", _A, _A]),
         ("mul_mask", [_A, _U2]), ("masked_copy", [_A, _U2, _A]), ("zero_rows", [_A, "uint8[::1]", "int64"]),
         ("accumulate", [_A, _U2, _A, "int64", "int64", _A]),
         ("step_source", [_A, _U2, _A, _A, "int64", "int64", _A]), ("col_mass", [_A, "int64", "int64", _A]),
         ("half_values", [_I, _I, _I, _A, "int64", _A, _A, _I, _U2, _I, _I, _A]),
         ("pair_terms", [_A, _A, _V, _V, _I, _U2, "int64", _I, "int64", "int64", _A, _A]),
         ("scatter_init", [_I, _I, _I, _I, _A, "int64", _A, _U2, _I, _I, _A, _A])]
_SIGS = [(k, a + ["int64", "int64"]) for k, a in _SIGS]           # 2-D launches: n0, n1
_SIGS += [(k, a + ["int64"]) for k, a in [
    ("fill_zero", [_V]), ("sum_parts", [_A, _V]), ("row_mass", [_A, _A, "int64", "int64", _V]),
    ("row_sums", [_A, _V]), ("rows_density", [_A, _A, _V, "int64", "int64", _V])]]
_SIGS += [(k, [_I, f"{s}[::1]", f"{w}[::1]", _I2, _I2, _I2, "int64", "int64", _A, _A, "int64", "int64"])
          for k in ("stencil_backward", "stencil_forward") for s in ("uint8", "uint16") for w in ("float32", "float64")]


def test_kernels_compile_for_cuda():
    # a process of its own: the simulator (this one) cannot make PTX; NVVM needs no GPU
    script = textwrap.dedent(f"""
        import re, numpy as np
        if not hasattr(np, "row_stack"):
            np.row_stack = np.vstack
        from numba import cuda
        from mpat import _numba_cuda_kernels as K
        atomics = 0
        for arch in ((7, 5), (9, 0)):
            for name, args in {_SIGS!r}:
                ptx, _ = cuda.compile_ptx(getattr(K, name).py_func, "void(" + ", ".join(args) + ")", cc=arch)
                atomics += len(re.findall(r"(?:atom|red)(?:\\.global)?\\.add\\.f64", ptx))
                assert "atom.global.cas.b64" not in ptx, name
        print("atomics", atomics)
    """)
    env = {k: v for k, v in os.environ.items() if k != "NUMBA_ENABLE_CUDASIM"}
    env["PYTHONPATH"] = os.pathsep.join([str(p) for p in sys.path if p])
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=1800)
    if r.returncode and "nvvm" in (r.stderr or "").lower() and "not found" in r.stderr.lower():
        pytest.skip("no NVVM (pip install 'numba-cuda[cu13]')")
    assert r.returncode == 0, r.stderr[-3000:]
    assert int(r.stdout.split()[-1]) > 0                      # native float64 atomics
