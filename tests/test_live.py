"""The two solves advanced together (mpat.solve.Interleaved) against the sequential solver, and
the live page's server (mpat.live)."""
import gzip
import json
import urllib.request

import nibabel as nib
import numpy as np
import pytest
from phantoms import crossing_phantom
from scipy import ndimage

from mpat.chain import VALID_ENDS, Chain, ChainConfig
from mpat.solve import Interleaved, _Plan, prepare_launches, segment_bundles


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


def _close(a, b, tol=1e-5):
    return np.abs(a - b).sum() <= tol * max(np.abs(b).sum(), 1e-300)


def _same(live, seq):
    for k, s in seq.items():
        v = live[k]
        assert _close(v.density_t, s.density_t) and _close(v.seed_prob, s.seed_prob)
        assert _close(v.endpoints_a, s.endpoints_a) and _close(v.density_class, s.density_class)
        assert abs(v.info["endpoint_mass_check"] - 1) < 1e-6
        assert v.info["backward"]["interleaved"] and "interleaved" not in s.info["backward"]


# ------------------------------------------------------------------ the solver
def test_interleaved_matches_sequential_and_grows(crossing):
    ch, L, A, B, end = crossing
    specs = [dict(name="x", include=[A, B], end=end), dict(name="e", end=end), dict(name="a", include=[A])]
    seq = segment_bundles(ch, specs, L)
    frames = {s["name"]: [] for s in specs}
    seen = []

    def progress(sol):
        seen.append((sol.steps, sol.remaining, sol.progress, sol.done))
        for i, name in enumerate(sol.names):
            frames[name].append(float(sol.density(i).sum()))

    live = segment_bundles(ch, specs, L, on_progress=progress)
    _same(live, seq)
    # about as many matrix products: the forward solve needs at most a few steps more
    for k in seq:
        nb, nf = (seq[k].info[w]["iterations"] for w in ("backward", "forward"))
        lb, lf = (live[k].info[w]["iterations"] for w in ("backward", "forward"))
        assert lb == nb and nf <= lf <= nf + 8
    # called before the first step, after every step and once more at the end
    assert seen[0][0] == 0 and seen[0][2] == 0.0 and not seen[0][3]
    assert seen[-1][3] and seen[-1][2] == 1.0 and seen[-1][1] < 1e-6
    assert [s[0] for s in seen[1:-1]] == list(range(1, seen[-1][0]))
    # the bundle so far only grows (no include ROIs: no alternating signs), to the solution
    m = np.array(frames["e"])
    assert m[0] < 0.05 * m[-1] and (np.diff(m) >= -1e-12 * m[-1]).all()
    for k in seq:
        assert abs(frames[k][-1] / seq[k].density_t.sum() - 1) < 1e-5


def test_interleaved_groups_columns_and_moments():
    ch, L, A, B, end = _crossing(moments=True)
    specs = [dict(name="a", include=[A], end=end), dict(name="b", include=[A, B])]
    seq = segment_bundles(ch, specs, L)
    _same(segment_bundles(ch, specs, L, on_progress=lambda sol: None), seq)
    # one column at a time (each bundle alone), and a source refreshed less often
    _same(segment_bundles(ch, specs, L, max_mem=1, on_progress=lambda sol: None, source_every=7), seq)


def test_interleaved_starts_from_an_earlier_solution(crossing):
    ch, L, A, B, end = crossing
    ex = np.zeros_like(A)
    ex[:, : A.shape[1] // 2 - 1] = True          # exclude one side of the tube's cross-section
    ex &= ~A & ~B
    first = Interleaved(ch, [_Plan(ch, dict(name="x", include=[A, B], end=end), VALID_ENDS)], L)
    r1 = first.results()["x"]
    edited = dict(name="x", include=[A, B], end=end, exclude=ex)
    cold = segment_bundles(ch, [edited], L)["x"]
    assert cold.density_t.sum() < 0.9 * r1.density_t.sum()          # the edit matters
    warm = Interleaved(ch, [_Plan(ch, edited, VALID_ENDS)], L, start=first)
    assert warm.warm
    assert _close(warm.volume(0), ch.to_volume(r1.density_t) * ~ex, 0.5)     # starts near the old bundle
    r2 = warm.results()["x"]
    assert _close(r2.density_t, cold.density_t) and _close(r2.seed_prob, cold.seed_prob)
    assert _close(r2.endpoints_a, cold.endpoints_a)
    # other columns (one include ROI fewer): starts from zero
    other = Interleaved(ch, [_Plan(ch, dict(name="x", include=[A], end=end), VALID_ENDS)], L, start=first)
    assert not other.warm
    assert _close(other.results()["x"].density_t,
                  segment_bundles(ch, [dict(name="x", include=[A], end=end)], L)["x"].density_t)


# ------------------------------------------------------------------ the live page's server
def _get(url, raw=False):
    with urllib.request.urlopen(url, timeout=10) as r:
        body = r.read()
        if not raw and r.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        return body


def test_live_colormap():
    from mpat.live import NODES, colormap

    c = colormap("magma", alpha=0.8, gamma=4.0)
    assert all(len(c[k]) == NODES for k in "RGBAI") and c["I"][0] == 0 and c["I"][-1] == 255
    assert c["A"][0] == 0 and c["A"][-1] == round(255 * 0.8) and (np.diff(c["A"]) >= 0).all()
    assert c["A"][NODES // 2] < 0.1 * c["A"][-1]            # steep: the faint levels are nearly clear
    assert (c["R"][-1], c["G"][-1], c["B"][-1]) == (252, 253, 191)        # magma's top


def test_live_view_serves_the_bundles_so_far(crossing):
    from mpat.live import ALL, LiveView
    from mpat.viewer import log_density_u8

    ch, L, A, B, end = crossing
    specs = [dict(name="Left Tube", end=end), dict(name="x", include=[A, B], end=end)]
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    seen = []
    with LiveView(ch.shape, aff, port=0, interval=0.0, share=float("inf"), title="sub-01") as view:
        def progress(sol):
            view.update(sol)
            s = json.loads(_get(view.url + "state"))
            seen.append((s["step"], s["version"], int(np.count_nonzero(view.frame("Left Tube"))), s["done"]))
            assert s["step"] == sol.steps and s["names"] == sol.names

        res = segment_bundles(ch, specs, L, on_progress=progress)
        s = json.loads(_get(view.url + "state"))
        assert [b["name"] for b in s["bundles"]] == ["Left Tube", "x"] and all(b["done"] for b in s["bundles"])
        assert not s["finished"] and s["title"] == "sub-01" and s["shape"] == list(ch.shape) and s["back_done"]
        for b in s["bundles"]:
            r = res[b["name"]]
            assert abs(b["expected"] / r.info["expected_bundle_streamlines"] - 1) < 1e-5
            assert abs(b["peak"] / r.density_t.max() - 1) < 1e-5
        # one redraw per step (interval 0): the version counts them; the frame is relative to
        # the peak so far, so it starts empty and ends full (the density itself only grows:
        # test_interleaved_matches_sequential_and_grows)
        steps, versions, filled, done = map(np.array, zip(*seen))
        assert (np.diff(versions) == 1).all() and filled[0] == 0 and filled[-1] == filled.max() > 0
        assert done[-1] and not done[:-1].any()
        # the frame: the log-scale bytes NiiVue draws, in the file's (Fortran) order
        ref = log_density_u8(ch.to_volume(res["Left Tube"].density_t), 1e-3)
        u8 = np.frombuffer(_get(view.url + "frame?name=Left%20Tube"), np.uint8).reshape(ch.shape, order="F")
        assert np.abs(u8.astype(int) - ref.astype(int)).max() <= 1 and u8.max() == 255
        img = nib.Nifti1Image.from_bytes(gzip.decompress(_get(view.url + "vol/Left%20Tube/density.nii.gz")))
        assert img.shape == ch.shape and np.allclose(img.affine, aff) and (np.asarray(img.dataobj) == u8).all()
        both = np.frombuffer(_get(view.url + "frame"), np.uint8).reshape(ch.shape, order="F")
        assert (both == np.maximum(u8, view.frame("x"))).all() and (view.frame(ALL) == both).all()
        assert not view.frame("no such bundle").any()
        page = _get(view.url).decode()
        assert "niivue" in page and '"cmap"' in page and "__CONFIG__" not in page and "sub-01" in page
        with pytest.raises(urllib.error.HTTPError):
            _get(view.url + "nothing-here")
        view.finish()
        assert json.loads(_get(view.url + "state"))["finished"]
        url = view.url
    with pytest.raises(OSError):            # closed with the block
        _get(url + "state")


def test_run_with_live_view(tmp_path):
    pytest.importorskip("AFQ")
    import socket
    import threading
    import time

    from synthetic_subject import make_afq, make_subject

    import mpat

    with socket.socket() as sock:           # a free port
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    states, stop = [], threading.Event()

    def watch():
        while not stop.is_set():
            try:
                states.append(json.loads(_get(f"http://localhost:{port}/state")))
            except OSError:
                pass
            time.sleep(0.05)

    t = threading.Thread(target=watch, daemon=True)
    t.start()
    f = make_subject(tmp_path / "raw")
    try:
        live = mpat.run(make_afq(f, tmp_path / "afq"), n_proc=2, max_mem="2GB", viz=False, live=port)
    finally:
        stop.set()
        t.join()
    assert states and states[-1]["finished"] and max(s["version"] for s in states) > 2
    assert {b["name"] for b in states[-1]["bundles"]} == set(live) and all(b["done"] for b in states[-1]["bundles"])
    with pytest.raises(OSError):            # the run closed its server
        _get(f"http://localhost:{port}/state")
    # the same bundles as a run without the live view
    plain = mpat.run(make_afq(f, tmp_path / "afq"), n_proc=2, max_mem="2GB", viz=False, out_dir=tmp_path / "plain")
    for k in plain:
        assert live[k].info["backward"]["interleaved"] and _close(live[k].density_t, plain[k].density_t)
