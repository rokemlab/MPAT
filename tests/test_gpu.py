"""The GPU solves (mpat.gpu, NVIDIA Warp) against the host's, on Warp's CPU device: the same
kernels as on a CUDA device, so no GPU is needed."""
import numpy as np
import pytest
from phantoms import crossing_phantom
from scipy import ndimage

wp = pytest.importorskip("warp")

from mpat.chain import VALID_ENDS, Chain, ChainConfig  # noqa: E402
from mpat.solve import HostOps, Interleaved, _Plan, overlength_states, prepare_launches, segment_bundles  # noqa: E402

DEV = "cpu"


@pytest.fixture(autouse=True)
def like_cuda(monkeypatch):
    """Warp's CPU device behaving as a CUDA one does where the difference could hide a bug:
    numpy cannot read a device array (only ``.numpy()``, which copies), and new arrays from
    ``wp.empty`` hold garbage (NaN here) rather than the zeroed pages a CPU allocation gets."""
    A = wp.array
    iface, numpy = A.__array_interface__, A.numpy
    allowed = [False]

    def interface(self):
        if not allowed[0]:
            raise AttributeError("no host access to a device array")
        return iface.fget(self)

    def reject(self):
        def fail(dtype=None, copy=None):
            raise TypeError("numpy used a device array directly (this fails on a GPU): use .numpy()")
        return fail

    def copy(self, **kw):
        allowed[0] = True
        try:
            return np.array(numpy(self, **kw), copy=True)
        finally:
            allowed[0] = False

    def garbage(make):
        def f(*a, **kw):
            arr = make(*a, **kw)
            arr.fill_(float("nan") if arr.dtype in (wp.float64, wp.float32) else 0xA5)
            return arr
        return f

    monkeypatch.setattr(A, "__array_interface__", property(interface))
    monkeypatch.setattr(A, "__array__", property(reject))
    monkeypatch.setattr(A, "numpy", copy)
    monkeypatch.setattr(wp, "empty", garbage(wp.empty))
    monkeypatch.setattr(wp, "empty_like", garbage(wp.empty_like))


def test_the_cuda_emulation():
    a = wp.zeros(3, dtype=wp.float64, device=DEV)
    with pytest.raises(TypeError, match="device array"):
        np.asarray(a)
    h = a.numpy()
    h[0] = 1.0
    assert a.numpy()[0] == 0.0
    assert np.isnan(wp.empty_like(a).numpy()).all()


def _crossing(moments=False):
    sh, wm, cls, reg = crossing_phantom(n=25, width=5)
    caps = reg["capx"] | reg["capy"]
    seeds = np.argwhere(ndimage.binary_dilation(caps, np.ones((3, 3, 3))) & wm)
    c = reg["center"][0]
    A = np.zeros_like(wm)
    A[c - 4] = True
    A &= wm
    B = np.zeros_like(wm)
    B[c + 4] = True
    B &= wm
    end = np.zeros_like(wm)
    end[-2:] = True
    end &= reg["capx"]
    ch = Chain.build(sh, (1.0, 1.0, 1.0), wm, cls, ChainConfig(moments=moments))
    return ch, prepare_launches(ch, seeds), A, B, end


@pytest.fixture(scope="module")
def crossing():
    return _crossing()


@pytest.fixture(scope="module")
def crossing_m():
    return _crossing(moments=True)


def _ops(ch, L, **kw):
    from mpat.gpu import WarpOps

    return WarpOps(ch, L, device=DEV, **kw)


def _close(a, b, tol=1e-8):
    return np.abs(a - b).sum() <= tol * max(np.abs(b).sum(), 1e-300)


def _same(got, ref, tol=1e-8):
    for k, r in ref.items():
        g = got[k]
        assert _close(g.density_t, r.density_t, tol) and _close(g.seed_prob, r.seed_prob, tol)
        assert _close(g.endpoints_a, r.endpoints_a, tol) and _close(g.density_class, r.density_class, max(tol, 1e-7))
        assert abs(g.info["endpoint_mass_check"] - r.info["endpoint_mass_check"]) < max(tol, 1e-9)
        assert (g.flux26 is None) == (r.flux26 is None)
        if r.flux26 is not None:                  # the edge flux, from the bundle's columns on the host
            assert _close(g.flux26, r.flux26, max(tol, 1e-7))


# ------------------------------------------------------------------ the products
@pytest.mark.parametrize("moments", [False, True])
@pytest.mark.parametrize("segment", [2 ** 30, 50_000])          # one segment, or many
def test_products_match_the_host(crossing, crossing_m, moments, segment):
    ch, L, *_ = crossing_m if moments else crossing
    ops = _ops(ch, L, segment=segment)
    assert (len(ops.dc.q) > 1) == (segment < ch.n_transitions)
    rng = np.random.default_rng(0)
    R, C = ch.n_states * ch.ncomp, 3
    x = rng.random((R, C))
    x[rng.random(R) < 0.3] = 0.0                                  # rows the forward product skips
    for what in ("forward", "backward"):
        ref = np.empty_like(x)
        (ch.mul_forward if what == "forward" else ch.mul_backward)(x.copy(), ref)
        out = ops.zeros((R, C))
        ops.mul(what, ops.from_host(x), out)
        assert np.abs(out.numpy() - ref).max() <= 1e-12 * np.abs(ref).max()
    a = rng.random((ch.a_lin.size, C))
    out = ops.zeros((R, C))
    ops.dc.absorb_rhs(ops.from_host(a), out)
    assert np.allclose(out.numpy(), ch.absorb_rhs(a), rtol=1e-12, atol=0)
    ab = ops.dc.absorbed_from(ops.from_host(x), ops.zeros((ch.a_lin.size, C))).numpy()
    assert np.allclose(ab, ch.absorbed_from(x), rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("moments", [False, True])
@pytest.mark.parametrize("nblk", [7, 64, 8192])       # 8192: a GPU's row blocks (more than these rows)
def test_reductions_and_steps_match_the_host(crossing, crossing_m, moments, nblk):
    ch, L, *_ = crossing_m if moments else crossing
    ops, host = _ops(ch, L), HostOps(ch, L, ncomp=ch.ncomp)
    ops.nblk = nblk
    rng = np.random.default_rng(1)
    R, C = ch.n_states * ch.ncomp, 4
    nxt, total, src, cur = (rng.standard_normal((R, C)) for _ in range(4))
    mask = (rng.random((R, C)) > 0.2).astype(np.uint8)
    d = {k: ops.from_host(v) for k, v in dict(nxt=nxt, total=total, src=src, cur=cur, mask=mask).items()}
    res = ops.accumulate(d["nxt"], d["mask"], d["total"])
    n2, t2 = nxt.copy(), total.copy()
    assert np.allclose(res, host.accumulate(n2, mask, t2), rtol=1e-12)
    assert np.allclose(d["nxt"].numpy(), n2) and np.allclose(d["total"].numpy(), t2)
    d2 = ops.from_host(nxt)
    res = ops.step(d2, d["mask"], d["src"], d["cur"])
    n3 = nxt.copy()
    assert np.allclose(res, host.step(n3, mask, src, cur), rtol=1e-12) and np.allclose(d2.numpy(), n3)
    assert np.allclose(ops.col_mass(d["src"]), host.col_mass(src), rtol=1e-12)
    assert np.allclose(ops.row_mass(d["src"], d["cur"]), host.row_mass(src, cur), rtol=1e-6)
    assert np.allclose(ops.row_mass(d["src"]), host.row_mass(src), rtol=1e-6)


# ------------------------------------------------------------------ bundles
def test_bundles_match_the_host(crossing):
    ch, L, A, B, end = crossing
    specs = [dict(name="x", include=[A, B], end=end), dict(name="e", end=end), dict(name="a", include=[A])]
    ref = segment_bundles(ch, specs, L)
    got = segment_bundles(ch, specs, L, backend="warp", device=DEV)
    _same(got, ref)
    for k in ref:
        for w in ("backward", "forward"):
            assert got[k].info[w]["iterations"] == ref[k].info[w]["iterations"]
    # one column at a time, the chain in several segments
    ops = _ops(ch, L, segment=50_000)
    _same(segment_bundles(ch, specs, L, ops=ops, max_mem=1), ref)


def test_fields_arc_length_and_on_step_on_the_gpu(crossing):
    """keep_fields (mpat.fields), arc_length and on_step: on the GPU what they are on the host."""
    ch, L, A, B, end = crossing

    def specs():
        return [dict(name="Left Tube", end=end), dict(name="x", include=[A, B], end=end)]

    zooms = (1.0, 1.0, 1.0)
    seen = {"host": [], "gpu": []}

    def watch(where):
        def on_step(st):
            assert isinstance(st.total, np.ndarray) and isinstance(st.term, np.ndarray)
            assert st.phase == "backward" or isinstance(st.beta, np.ndarray)
            seen[where].append((st.phase, st.it, round(st.res, 12), float(np.abs(st.term).sum())))
        return on_step

    ref = segment_bundles(ch, specs(), L, keep_fields=True, arc_length=True, zooms=zooms, on_step=watch("host"))
    got = segment_bundles(ch, specs(), L, keep_fields=True, arc_length=True, zooms=zooms, on_step=watch("gpu"),
                          backend="warp", device=DEV)
    _same(got, ref)
    assert len(seen["gpu"]) == len(seen["host"]) > 0
    for g, h in zip(seen["gpu"], seen["host"]):
        assert g[:3] == h[:3] and abs(g[3] - h[3]) <= 1e-9 * max(h[3], 1.0)
    for k, r in ref.items():
        f, e = got[k].fields, r.fields
        for a in ("S", "beta", "launch", "a0"):
            assert isinstance(getattr(f, a), np.ndarray) and _close(getattr(f, a), getattr(e, a), 1e-8), a
        assert _close(f.edge_flux(), e.edge_flux(), 1e-8)


def test_live_solves_and_warm_start_on_the_gpu(crossing):
    ch, L, A, B, end = crossing
    specs = [dict(name="x", include=[A, B], end=end), dict(name="e", end=end)]
    ref = segment_bundles(ch, specs, L)
    frames = []

    def progress(sol):
        frames.append(float(sol.density(1).sum()))
        assert sol.volume(0).shape == ch.shape

    ops = _ops(ch, L)
    got = segment_bundles(ch, specs, L, ops=ops, on_progress=progress)
    _same(got, ref, 1e-6)
    m = np.array(frames)
    assert m[0] < 0.05 * m[-1] and (np.diff(m) >= -1e-12 * m[-1]).all()      # the bundle only grows
    # warm start after an edit (exclude part of the cross-section), on the device
    ex = np.zeros_like(A)
    ex[:, : A.shape[1] // 2 - 1] = True
    ex &= ~A & ~B
    edited = dict(name="x", include=[A, B], end=end, exclude=ex)
    first = Interleaved(ch, [_Plan(ch, specs[0], VALID_ENDS)], L, ops=ops)
    first.results()
    warm = Interleaved(ch, [_Plan(ch, edited, VALID_ENDS)], L, start=first, ops=ops)
    assert warm.warm
    _same(warm.results(), segment_bundles(ch, [edited], L), 1e-5)
    # a host solution cannot start a device one (or the reverse)
    host = Interleaved(ch, [_Plan(ch, specs[0], VALID_ENDS)], L)
    assert not Interleaved(ch, [_Plan(ch, specs[0], VALID_ENDS)], L, start=host, ops=ops).warm


def test_forward_starts_are_read_lazily(crossing, monkeypatch):
    # a live solve refreshes its forward start every few steps: only the start used crosses
    from mpat import gpu as G

    ch, L, A, B, end = crossing
    spec = dict(name="x", include=[A, B], end=end)
    reads, read = [], G.DeviceGroup._start
    monkeypatch.setattr(G.DeviceGroup, "_start", lambda self, i: reads.append(i) or read(self, i))
    ops = _ops(ch, L)
    sol = Interleaved(ch, [_Plan(ch, spec, VALID_ENDS)], L, ops=ops)
    _same(sol.results(), segment_bundles(ch, [spec], L), 1e-6)
    assert reads == [0] and sol.info_f["iterations"] > 4 * sol.source_every
    g = ops.group([_Plan(ch, spec, VALID_ENDS)])
    _, s1 = g.forward_start(g.r)
    _, s2 = g.forward_start(g.r)
    assert np.isfinite(s2[0][1]).all()
    with pytest.raises(RuntimeError, match="replaced"):
        s1[0]


def test_moments_on_the_gpu(crossing_m):
    ch, L, A, B, end = crossing_m
    specs = [dict(name="a", include=[A], end=end), dict(name="b", include=[A, B])]
    _same(segment_bundles(ch, specs, L, backend="warp", device=DEV), segment_bundles(ch, specs, L))


def test_float32_weights(crossing):
    ch, L, A, B, end = crossing
    specs = [dict(name="x", include=[A, B], end=end)]
    ops = _ops(ch, L, weight_dtype="float32")
    assert ops.dc.q[0][4].dtype == wp.float32
    ref = segment_bundles(ch, specs, L)
    got = segment_bundles(ch, specs, L, ops=ops)
    _same(got, ref, 1e-5)
    assert not _close(got["x"].density_t, ref["x"].density_t, 1e-12)          # it is float32


def test_overlength_on_the_gpu(crossing):
    ch, L, *_ = crossing
    k1, s1, T1 = overlength_states(ch, 40.0, (1.0, 1.0, 1.0))
    k2, s2, T2 = overlength_states(ch, 40.0, (1.0, 1.0, 1.0), ops=_ops(ch, L))
    assert T1 == T2 and (k1 == k2).all() and np.allclose(s1, s2, rtol=1e-12, atol=1e-300)
    # states past the soft maximum length: walkers end there on the device too
    kill = np.zeros(ch.n_states, bool)
    kill[np.random.default_rng(2).random(ch.n_states) < 0.02] = True
    ch.kill = kill
    try:
        specs = [dict(name="e", end=crossing[4])]
        _same(segment_bundles(ch, specs, L, backend="warp", device=DEV), segment_bundles(ch, specs, L))
    finally:
        ch.kill = None


@pytest.mark.parametrize("arch", [75, 90])          # Turing (the oldest Warp builds for), Hopper
def test_kernels_compile_for_cuda(arch, tmp_path):
    # the CUDA build of the kernels, with NVRTC (in the warp-lang wheel): no GPU needed
    rt = wp._src.context.runtime
    if rt is None or getattr(rt, "toolkit_version", None) is None:
        pytest.skip("this Warp has no CUDA compiler (e.g. macOS)")
    from mpat.gpu import kernels

    wp.compile_aot_module(wp.get_module(kernels().__name__), device=None, arch=arch, module_dir=str(tmp_path),
                          use_ptx=True)
    ptx = "".join(p.read_text() for p in tmp_path.rglob("*.ptx"))
    assert ptx.count(".entry") >= 27
    assert "atom.add.f64" in ptx and "atom.global.cas.b64" not in ptx         # native float64 atomics


def test_no_cuda_device_is_a_clear_error():
    from mpat.gpu import get_device

    if wp.is_cuda_available():
        pytest.skip("this machine has a CUDA device")
    with pytest.raises(RuntimeError, match="no CUDA device"):
        get_device("cuda")


def test_run_on_the_gpu(tmp_path):
    pytest.importorskip("AFQ")
    from synthetic_subject import make_afq, make_subject

    import mpat

    f = make_subject(tmp_path / "raw")
    ref = mpat.run(make_afq(f, tmp_path / "afq"), n_proc=2, max_mem="2GB", viz=False, cores=False)
    got = mpat.run(make_afq(f, tmp_path / "afq"), n_proc=2, max_mem="2GB", viz=False, cores=False, gpu=DEV,
                   out_dir=tmp_path / "gpu")
    _same(got, ref)
    # the maps from the whole-brain reference and the edge flux (mpat.quantify) too
    import nibabel as nib

    host = tmp_path / "afq" / "mpat"
    for b in ("A", "B"):
        for desc in ("afd", "tdimm", "qharmonic", "coherence", "keff"):
            g, r = (np.asarray(nib.load(next(d.glob(f"*_bundle-{b}_desc-{desc}_map.nii.gz"))).dataobj)
                    for d in (tmp_path / "gpu", host))
            assert _close(g, r, 1e-6), (b, desc)
