"""Fixed-step walkers by Lagrange-Galerkin (mpat.galerkin), on Warp's CPU device: the same
kernels as on a CUDA device, so no GPU is needed."""
import itertools
import os
from math import factorial

import numpy as np
import pytest
from phantoms import arc_phantom, tube_phantom

wp = pytest.importorskip("warp")

from mpat.galerkin import (  # noqa: E402
    GM_BARY,
    GM_W,
    KUHN,
    Bundles,
    Galerkin,
    GalerkinConfig,
    _tet_vol,
    basis,
    pieces,
)

DEV = os.environ.get("MPAT_TEST_DEVICE", "cpu")      # MPAT_TEST_DEVICE=cuda: on the GPU
CFG = GalerkinConfig(sphere="repulsion100", tol=1e-6, step_unit="vox", until="walkers", prune=0.0)   # exact


def _tube():
    sh, wm, cls = tube_phantom(shape=(20, 9, 9), radius=2.5)
    start = np.zeros(wm.shape, bool)
    start[:3] = True
    end = np.zeros(wm.shape, bool)
    end[-3:] = True
    absorb = np.where(wm, 0, cls).astype(np.int8)
    return sh, wm, absorb, dict(name="tube", include=[], exclude=None, start=start, end=end)


def test_basis_and_quadrature():
    x = GM_BARY[:, 1:]
    for p, q, r in itertools.product(range(6), repeat=3):       # degree 5 exactly on a tetrahedron
        if p + q + r <= 5:
            val = (GM_W / 6.0) @ (x[:, 0] ** p * x[:, 1] ** q * x[:, 2] ** r)
            assert abs(val - factorial(p) * factorial(q) * factorial(r) / factorial(p + q + r + 3)) < 1e-14
    X = np.einsum("qv,tvx->tqx", GM_BARY, KUHN).reshape(-1, 3)
    w = (np.abs(_tet_vol(KUHN.astype(float)))[:, None] * GM_W[None, :]).ravel()
    assert np.allclose((basis(X) * w[:, None]).T @ basis(X), np.eye(10), atol=1e-13)


@pytest.mark.parametrize("d", [(0.31, -0.22, 0.4), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (-0.7, 0.45, -0.1),
                               (1.4, -0.9, 0.35), (0.0, -2.3, 1.1)])
def test_pieces(d):
    ps = pieces(np.array(d), 0.6)          # raises unless every walker is in one piece, path lengths add up
    d = np.array(d)
    if np.all(np.abs(d) < 1):
        assert len(ps) <= 16               # at most one crossing per axis: 1 + 3 + 6 + 6 orders
    for p in ps:
        c = p["cells"]
        assert np.array_equal(c[0], [0, 0, 0]) and np.array_equal(c[-1], p["land"])
        assert np.all(np.abs(np.diff(c, axis=0)).sum(1) == 1)          # face neighbours, one by one
    if np.allclose(d, np.round(d)):                                    # a whole-bin shift: the identity
        assert len(ps) == 1 and np.allclose(ps[0]["A"], np.eye(10), atol=1e-12)


def test_tube():
    sh, wm, absorb, spec = _tube()
    r = Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV).bundle(spec)
    i = r.info
    assert all(s["forward"]["converged"] for s in i["solves"])
    assert i["n_solves"] == 4              # per end kind, a forward and a backward solve
    # every walker counted at its launch (backward) and where it ends (forward) alike
    assert i["transport_identity"] < 1e-5
    assert abs(i["endpoint_mass_check"] - 1) < 1e-5
    assert 0 < i["expected_bundle_streamlines"] < i["launched_walkers"]
    ep = r.endpoints
    assert ep[3:-3].sum() == 0 and np.isclose(ep[:3].sum(), ep[-3:].sum(), rtol=1e-4)
    assert (r.density[~wm] == 0).all() and (r.density[3:-3][wm[3:-3]] > 0).all()
    c = r.density[:, 4, 4]
    assert np.allclose(c[4:-4], c[4:-4].mean(), rtol=0.05)               # steady along the tube
    x = r.density[10, :, 4]
    assert x[4] > 1.2 * x[2] and np.isclose(x[2], x[6], rtol=0.02)      # less at its wall; symmetric (to the sphere)
    assert r.seed_prob.max() <= 1 and r.seed_prob[spec["start"]].max() > 0
    assert len(r.odf_ijk) and np.allclose(r.odf_coef[:, 0] * 2 * np.sqrt(np.pi),
                                          r.density[tuple(r.odf_ijk.T)], rtol=1e-4)


def test_graphs_match():
    """The steps replayed as graphs (as on a GPU): the same as launched one by one."""
    sh, wm, absorb, spec = _tube()
    a = Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV, graphs=False).bundle(spec)
    try:
        g = Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV, graphs=True).bundle(spec)
    except Exception as err:  # noqa: BLE001 -- older Warp captures graphs on CUDA devices only
        pytest.skip(f"no graph capture on Warp's CPU device here: {err}")
    assert np.allclose(a.density, g.density, rtol=1e-6, atol=0)


def test_monte_carlo():
    """The walkers sampled one by one (galerkin_mc) agree with the scheme: in the arc, where
    they turn, to a percent or two for the bundle's walkers and its density."""
    pytest.importorskip("numba")
    from galerkin_mc import monte_carlo

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    S, E = reg["cap_start"], reg["cap_end"]
    m = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, CFG, device=DEV)
    r = m.bundle(dict(name="arc", include=[], exclude=None, start=S, end=E))
    good, launched, dens, _ = monte_carlo(m, sh, [(S, E), (E, S)], nper=400)
    assert np.isclose(launched, r.info["launched_walkers"], rtol=1e-6)
    assert abs(r.info["expected_bundle_streamlines"] / good - 1) < 0.04
    dens = dens.reshape(wm.shape)
    assert np.abs(r.density - dens).sum() / dens.sum() < 0.08


def test_include_and_exclude():
    """An include ROI every bundle walker crosses changes nothing (inclusion-exclusion: the
    walkers avoiding it are none); one beside the tube leaves no bundle; an exclude ROI takes
    its voxels out of the white matter."""
    sh, wm, absorb, spec = _tube()
    m = Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV)
    base = m.bundle(spec)
    slab = np.zeros(wm.shape, bool)
    slab[10] = True
    r = m.bundle({**spec, "include": [slab]})
    assert r.info["n_solves"] == 8 and r.info["transport_identity"] < 1e-5
    assert np.isclose(r.info["expected_bundle_streamlines"], base.info["expected_bundle_streamlines"], rtol=1e-4)
    side = np.zeros(wm.shape, bool)
    side[10, 0] = True
    assert m.bundle({**spec, "include": [side]}).info["expected_bundle_streamlines"] < 1e-6
    half = np.zeros(wm.shape, bool)
    half[:, 5:] = True
    x = m.bundle({**spec, "exclude": half})
    assert x.density[half].max() == 0
    assert 0 < x.info["expected_bundle_streamlines"] < 0.8 * base.info["expected_bundle_streamlines"]
    assert x.info["transport_identity"] < 1e-5


def test_bins():
    """Finer bins: a voxel solve, then 2^3 bins within its bundle's region; every walker still
    accounted for, and the bundle about the voxel solve's (each voxel's own fODF in its bins: the
    default, interpolated from every voxel, blends the tube's near-isotropic surroundings into
    its wall bins and loses most of this thin bundle -- test_interp)."""
    sh, wm, absorb, spec = _tube()
    cfg = GalerkinConfig(sphere="repulsion100", tol=1e-6, step_unit="vox", bins=2, interp="")
    model = Bundles(sh, (2.0, 2.0, 2.0), wm, absorb, cfg, device=DEV)
    r = model.bundle(spec)
    i = r.info
    assert i["bins_per_voxel_side"] == 2 and 0 < i["region_voxels"] <= wm.sum()
    assert i["transport_identity"] < 1e-5 and abs(i["endpoint_mass_check"] - 1) < 1e-5
    v = i["voxel_solve"]["expected_bundle_streamlines"]
    assert abs(i["expected_bundle_streamlines"] / v - 1) < 0.05
    assert (r.density[~wm] == 0).all()


def test_lgn_eighth():
    """--ends-only: an optic radiation's thalamic end is cut to its lateral-posterior-inferior
    eighth (world axes; lateral: away from the midline), other bundles kept."""
    from mpat.afq import lgn_eighth, lgn_end

    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = (-20.0, -20.0, -20.0)          # voxel 20 is x = 0: voxels 4..11 are the left side
    thal = np.zeros((40, 40, 40), bool)
    thal[4:12, 10:18, 10:18] = True                 # a left "thalamus", 8^3 voxels
    cut = lgn_eighth(thal, affine)
    assert cut.sum() == 4 ** 3 and cut[4:8, 10:14, 10:14].all()      # lateral (x low), posterior, inferior
    right = np.zeros_like(thal)
    right[28:36, 10:18, 10:18] = True
    assert lgn_eighth(right, affine)[32:36, 10:14, 10:14].all()      # on the right, lateral is x high
    v1 = np.zeros_like(thal)
    v1[4:12, 0:4, 10:18] = True
    spec = dict(name="Left Optic Radiation", include=[], exclude=None, start=thal, end=v1)
    out = lgn_end(spec, {"left": thal, "right": right}, affine)
    assert out["start"].sum() == 64 and out["end"] is v1
    assert lgn_end({**spec, "name": "Left Arcuate"}, None, affine)["start"] is thal


def test_bundle_domain():
    """A bundle solved on the white matter its exclusions leave it (Bundles) gives what the
    whole white matter gives: walkers entering an excluded voxel are lost either way."""
    sh, wm, absorb, spec = _tube()
    half = np.zeros(wm.shape, bool)
    half[:, 5:] = True
    spec = {**spec, "exclude": half}
    whole = Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV).bundle(spec)
    own = Bundles(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV).bundle(spec)
    assert own.info["white_matter_voxels"] < whole.info["white_matter_voxels"]
    for a in ("density", "endpoints", "seed_prob"):
        assert np.allclose(getattr(own, a), getattr(whole, a), rtol=1e-5, atol=1e-6)
    assert np.isclose(own.info["expected_bundle_streamlines"], whole.info["expected_bundle_streamlines"], rtol=1e-6)


def test_solve_movie(tmp_path):
    """--galerkin-movie: each solve recorded (forward: walkers; backward: their chance of
    ending well) and written as an .mp4."""
    pytest.importorskip("imageio_ffmpeg")
    from mpat.solvemovie import SolveMovie

    sh, wm, absorb, spec = _tube()
    m = Bundles(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV)
    path = tmp_path / "tube_desc-solves.mp4"
    m.movie = SolveMovie("tube", str(path), np.eye(4), wm, frames=6)
    r = m.bundle(spec)
    kinds = [k for k, _, _ in m.movie.done]
    assert kinds.count("forward") <= 2 * 6 and kinds.count("backward") <= 2 * 6
    assert "forward" in kinds and "backward" in kinds
    m.movie.density(r.density)
    assert m.movie.write() == str(path) and path.stat().st_size > 1000


def test_bin_fodf():
    """--galerkin-interp: a bin centered on its voxel's center (the middle one of an odd kb) has the
    voxel's fODF; one between two voxels has the fODF between theirs."""
    from mpat.galerkin import bin_fodf, raw_sf
    from mpat.galerkin import Directions, fodf_on

    sh, wm, cls, _ = arc_phantom()
    dirs = Directions("repulsion100", 35.0)
    ijk = np.argwhere(wm)
    sf, rows = raw_sf(sh, ijk, wm.shape, dirs)
    mid = bin_fodf(sf, rows, ijk, 3, 13, dirs)
    assert np.allclose(mid, fodf_on(sh[tuple(ijk.T)], dirs), rtol=1e-4, atol=1e-6)
    assert np.allclose(mid @ dirs.w, 1.0, atol=1e-5)
    corner = bin_fodf(sf, rows, ijk, 2, 0, dirs)
    assert np.allclose(corner @ dirs.w, 1.0, atol=1e-5) and not np.allclose(corner, mid, atol=1e-3)


def test_interp():
    """--galerkin-interp: with the same fODF everywhere, interpolating it changes nothing.  In the
    arc, where it turns, every walker is still accounted for; from the white matter alone ("wm") the
    bundle is nearly the voxel-fODF one, while "all" blends the near-isotropic fODF around the thin
    arc into the bins along its walls, where walkers then turn out of it."""
    sh, wm, absorb, spec = _tube()
    same = np.broadcast_to(sh[tuple(np.argwhere(wm)[0])], sh.shape).copy()
    cfg = GalerkinConfig(sphere="repulsion100", tol=1e-6, step_unit="vox", bins=2, interp="")
    assert GalerkinConfig().interp == "all"                                # the default
    a = Galerkin(same, (2.0, 2.0, 2.0), wm, absorb, cfg, device=DEV).bundle(spec)
    for how in ("wm", "all"):
        b = Galerkin(same, (2.0, 2.0, 2.0), wm, absorb, GalerkinConfig(**{**cfg.key(), "interp": how}),
                     device=DEV).bundle(spec)
        assert np.allclose(a.density, b.density, rtol=1e-5, atol=1e-7)
        assert np.isclose(a.info["expected_bundle_streamlines"], b.info["expected_bundle_streamlines"], rtol=1e-5)

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    arc = dict(name="arc", include=[], exclude=None, start=reg["cap_start"], end=reg["cap_end"])
    r, m = {}, {}
    for how in ("", "wm", "all"):
        m[how] = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, GalerkinConfig(**{**cfg.key(), "interp": how}),
                          device=DEV)
        r[how] = m[how].bundle(arc)
        i = r[how].info
        assert i["transport_identity"] < 1e-5 and abs(i["endpoint_mass_check"] - 1) < 1e-5
        assert all(s["forward"]["converged"] for s in i["solves"])
    assert m[""].d_efw.shape == (m[""].P, 1) and m["wm"].d_efw.shape == (m["wm"].P, 8)
    assert "interp" not in r[""].info["config"] and r["wm"].info["config"]["interp"] == "wm"
    e = {k: v.info["expected_bundle_streamlines"] for k, v in r.items()}
    assert 0 < abs(e["wm"] / e[""] - 1) < 0.02
    assert 1e-4 < np.abs(r["wm"].density - r[""].density).sum() / r[""].density.sum() < 0.05
    assert e["all"] < 0.5 * e[""]
    with pytest.raises(ValueError):
        Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, GalerkinConfig(**{**cfg.key(), "interp": "grey"}), device=DEV)


def test_points():
    """--galerkin-points: with the same fODF everywhere, turning by it at Gauss points changes
    nothing (the points integrate products of the quadratics exactly); fewer than 3 are refused.
    In the arc, the forward and backward solves still agree (the turn's transpose is exact), and
    seeded walkers match walkers sampled one by one that turn by the fODF at their own point
    (galerkin_mc.monte_carlo_seeds_points)."""
    pytest.importorskip("numba")
    from galerkin_mc import monte_carlo_seeds_points

    sh, wm, absorb, spec = _tube()
    same = np.broadcast_to(sh[tuple(np.argwhere(wm)[0])], sh.shape).copy()
    for bins in (1, 2):
        cfg = GalerkinConfig(sphere="repulsion100", tol=1e-6, step_unit="vox", bins=bins, interp="")
        a = Galerkin(same, (2.0, 2.0, 2.0), wm, absorb, cfg, device=DEV).bundle(spec)
        b = Galerkin(same, (2.0, 2.0, 2.0), wm, absorb, GalerkinConfig(**{**cfg.key(), "points": 3}),
                     device=DEV).bundle(spec)
        assert np.allclose(a.density, b.density, rtol=1e-4, atol=1e-7)
        assert np.isclose(a.info["expected_bundle_streamlines"], b.info["expected_bundle_streamlines"], rtol=1e-5)
        assert b.info["config"]["points"] == 3 and "points" not in a.info["config"]
    with pytest.raises(ValueError):
        Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, GalerkinConfig(**{**cfg.key(), "points": 2}), device=DEV)

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    S, E = reg["cap_start"], reg["cap_end"]
    arc = dict(name="arc", include=[], exclude=None, start=S, end=E)
    m = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, GalerkinConfig(**{**CFG.key(), "bins": 2, "points": 3}), device=DEV)
    i = m.bundle(arc).info
    assert i["transport_identity"] < 1e-5 and abs(i["endpoint_mass_check"] - 1) < 1e-5
    ijk = np.argwhere(wm)
    mid = ijk[np.abs(np.argsort(np.argsort(ijk[:, 0])) - len(ijk) // 2) < 6]
    rng = np.random.default_rng(2)
    seeds = (mid[:, None, :] + rng.uniform(-0.45, 0.45, (len(mid), 3, 3))).reshape(-1, 3)
    r = m.seeded(seeds, [S, E])
    assert r["check"].max() < 1e-5 and r["info"]["seed_mass_without_state"] < 1e-3 * len(seeds)
    ends, pairs = monte_carlo_seeds_points(m, sh, seeds, [S, E], nper=4000)
    assert np.all(np.abs(r["ends"] / ends - 1) < 0.02)
    assert abs(r["pairs"][0, 1] / pairs[0, 1] - 1) < 0.02 and abs(r["pairs"][1, 0] / pairs[1, 0] - 1) < 0.02


def test_seeded():
    """Galerkin.seeded against walkers sampled one by one from the same seeds (galerkin_mc): in the
    arc, seeds in its middle, both halves (a bidirectional streamline) and one-sided starts, in
    2^3 bins (1^3 bins is about 1.5% high there, as test_monte_carlo's bundle); the backward
    solves agree with the forward's ends."""
    pytest.importorskip("numba")
    from galerkin_mc import monte_carlo_seeds

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    S, E = reg["cap_start"], reg["cap_end"]
    m = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, GalerkinConfig(**{**CFG.key(), "bins": 2, "interp": ""}),
                 device=DEV)                 # the sampled walkers turn by each voxel's own fODF
    ijk = np.argwhere(wm)
    mid = ijk[np.abs(np.argsort(np.argsort(ijk[:, 0])) - len(ijk) // 2) < 6]
    rng = np.random.default_rng(1)
    seeds = (mid[:, None, :] + rng.uniform(-0.45, 0.45, (len(mid), 3, 3))).reshape(-1, 3)
    n, N = len(seeds), 4000
    r = m.seeded(seeds, [S, E])
    assert r["check"].max() < 1e-5 and r["seeds_used"] == n
    assert r["seed_ends"].shape == (n, 2) and np.allclose(r["seed_ends"].sum(0), r["ends"])
    assert np.isclose(r["pairs"][0, 1], r["pairs"][1, 0]) and r["pairs"][0, 1] > 0.3 * n
    ends, pairs = monte_carlo_seeds(m, sh, seeds, [S, E], nper=N)
    assert np.all(np.abs(r["ends"] / ends - 1) < 0.02)                   # MC: about 0.5% at this count
    assert abs(r["pairs"][0, 1] / pairs[0, 1] - 1) < 0.02 and abs(r["pairs"][1, 0] / pairs[1, 0] - 1) < 0.02
    assert r["pairs"][0, 0] < 1e-3 * n and pairs[0, 0] == 0
    rd = m.seeded(seeds, [S, E], density=True)                           # both halves at once
    assert np.allclose(rd["ends_both"], rd["ends"] + rd["ends_reverse"], rtol=1e-5)
    v = rd["visits"]
    assert v.shape == wm.shape and v[~wm].sum() == 0 and v.sum() > 2 * n     # two halves, many steps each
    assert np.allclose(rd["ends"], r["ends"]) and np.allclose(rd["pairs"], r["pairs"])
    allowed = np.tile(m.dirs.v[:, 0] > 0, (n, 1))                        # one-sided: toward larger x only
    r1 = m.seeded(seeds, [S, E], allowed=allowed, both=False)
    e1, _ = monte_carlo_seeds(m, sh, seeds, [S, E], allowed=allowed, nper=N, seed=3)
    assert "pairs" not in r1 and abs(r1["ends"][0] / e1[0] - 1) < 0.02 and r1["ends"][1] < 1e-2


def test_inward():
    """inward: from a seed at a slab's face the allowed directions point into the slab."""
    from mpat.galerkin import inward
    from mpat.galerkin import Directions

    d = Directions("repulsion100", 35.0)
    inside = np.zeros((12, 12, 12), bool)
    inside[6:] = True
    a = inward([[6.0, 6.0, 6.0], [9.0, 6.0, 6.0]], inside, d)
    assert a.shape == (2, d.M)
    assert np.all(d.v[a[0], 0] > 0) and a[0].sum() < d.M          # into the slab: +x
    assert np.all(d.v[a[1], 0] > 0) or a[1].all()


def test_step_units():
    """The step: mm by default (as pyAFQ's), or a fraction of the smallest voxel side ("vox")."""
    from mpat.galerkin import parse_step

    assert parse_step(0.5) == (0.5, "mm") and parse_step("0.5") == (0.5, "mm") and parse_step("0.5mm") == (0.5, "mm")
    assert parse_step("0.4vox") == (0.4, "vox") and parse_step(" 1e-1 voxels") == (0.1, "vox")
    for bad in ("-1", "0", "abc", "0.5 cm"):
        with pytest.raises(ValueError):
            parse_step(bad)
    assert GalerkinConfig().step_unit == "mm" and GalerkinConfig().key()["step_unit"] == "mm"
    sh, wm, absorb, _ = _tube()
    mm = Galerkin(sh, (2.0, 2.0, 2.5), wm, absorb, GalerkinConfig(sphere="repulsion100", step=0.5), device=DEV)
    vx = Galerkin(sh, (2.0, 2.0, 2.5), wm, absorb, GalerkinConfig(sphere="repulsion100", step=0.5, step_unit="vox"),
                  device=DEV)
    assert np.isclose(mm.smm, 0.5) and np.isclose(mm.step_vox, 0.25)
    assert np.isclose(vx.smm, 1.0) and np.isclose(vx.step_vox, 0.5)
    with pytest.raises(ValueError):
        Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, GalerkinConfig(sphere="repulsion100", step_unit="cm"), device=DEV)


def test_seeded_bundle():
    """Bundles from streamlines seeded as pyAFQ's GPU tracker seeds them: one per fODF peak at each
    seed, its halves within the cone of the peak and of its reverse, drawn independently.  The
    expected streamlines in the bundle are the products of each half's chance to end in start / end
    (Galerkin.seeded, one half at a time); every half is counted where it ends and at its seed alike;
    an include ROI holding all of the white matter changes nothing (its avoid set kills every seed)."""
    from mpat.galerkin import seed_peaks, seeds_from_mask

    mask = np.zeros((6, 5, 4), bool)
    mask[1:4, 2, 1:3] = True
    s = seeds_from_mask(mask, 20, rng=3)
    vox = np.floor(s + 0.5).astype(int)
    assert s.shape == (20, 3) and mask[tuple(vox.T)].all()
    assert np.all(np.bincount(np.ravel_multi_index(tuple(vox.T), mask.shape), minlength=mask.size)[mask.ravel()] >= 3)

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    S, E = reg["cap_start"], reg["cap_end"]
    arc = dict(name="arc", include=[], exclude=None, start=S, end=E)
    m = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, CFG, device=DEV)
    ijk = np.argwhere(wm)
    mid = np.zeros(wm.shape, bool)
    mid[tuple(ijk[np.abs(np.argsort(np.argsort(ijk[:, 0])) - len(ijk) // 2) < 6].T)] = True
    seeds = seeds_from_mask(mid, 3 * int(mid.sum()), rng=1)
    sid, pk = seed_peaks(sh, seeds, "repulsion100")
    assert len(sid) >= len(seeds) and np.array_equal(np.unique(sid), np.arange(len(seeds)))  # a peak at each seed
    r = m.seeded_bundle(arc, seeds, (sid, pk))
    i = r.info
    assert i["transport_identity"] < 1e-5 and abs(i["endpoint_mass_check"] - 1) < 1e-5
    assert i["launched_walkers"] == len(sid) and 0.3 * len(sid) < i["expected_bundle_streamlines"] < len(sid)
    d = m.dirs
    cone = np.zeros((d.M, d.M), bool)
    for k in range(d.M):
        cone[k, d.cidx[d.cptr[k]:d.cptr[k + 1]]] = True
    va = m.seeded(seeds[sid], [S, E], allowed=cone[pk], both=False)["seed_ends"]
    vb = m.seeded(seeds[sid], [S, E], allowed=cone[d.rev[pk]], both=False)["seed_ends"]
    want = float((va[:, 0] * vb[:, 1] + va[:, 1] * vb[:, 0]).sum())
    assert np.isclose(i["expected_bundle_streamlines"], want, rtol=1e-5)
    neg = i["negative_mass_before_clip"]["endpoints"]                   # the maps are clipped at 0
    assert neg < 0.02 * want
    for roi in (S, E):
        assert want - 1e-4 * want <= r.endpoints[roi].sum() <= want + neg + 1e-4 * want
    assert (r.density[~wm] == 0).all() and r.density[mid].min() > 0
    assert r.seed_prob[mid].max() <= 1 and r.seed_prob[mid].mean() > 0.3 and r.seed_prob[~mid].sum() == 0
    ri = m.seeded_bundle(dict(arc, include=[wm]), seeds, (sid, pk))
    assert np.isclose(ri.info["expected_bundle_streamlines"], want, rtol=1e-5)
    assert np.allclose(ri.density, r.density, rtol=1e-4, atol=1e-6 * r.density.max())
    b = Bundles(sh, (1.0, 1.0, 1.0), wm, absorb, CFG, device=DEV)
    b.seeding = (seeds, (sid, pk))
    assert np.isclose(b.bundle(arc).info["expected_bundle_streamlines"], want, rtol=1e-6)


def test_seeded_bundle_points():
    """As test_seeded_bundle, turning at points (each half's first heading from the fODF at its seed)."""
    from mpat.galerkin import seed_peaks, seeds_from_mask

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    S, E = reg["cap_start"], reg["cap_end"]
    m = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, GalerkinConfig(**{**CFG.key(), "points": 3}), device=DEV)
    ijk = np.argwhere(wm)
    mid = np.zeros(wm.shape, bool)
    mid[tuple(ijk[np.abs(np.argsort(np.argsort(ijk[:, 0])) - len(ijk) // 2) < 6].T)] = True
    seeds = seeds_from_mask(mid, 2 * int(mid.sum()), rng=2)
    sid, pk = seed_peaks(sh, seeds, "repulsion100")
    r = m.seeded_bundle(dict(name="arc", include=[], exclude=None, start=S, end=E), seeds, (sid, pk))
    i = r.info
    assert i["transport_identity"] < 1e-5 and abs(i["endpoint_mass_check"] - 1) < 1e-5
    d = m.dirs
    cone = np.zeros((d.M, d.M), bool)
    for k in range(d.M):
        cone[k, d.cidx[d.cptr[k]:d.cptr[k + 1]]] = True
    va = m.seeded(seeds[sid], [S, E], allowed=cone[pk], both=False)["seed_ends"]
    vb = m.seeded(seeds[sid], [S, E], allowed=cone[d.rev[pk]], both=False)["seed_ends"]
    want = float((va[:, 0] * vb[:, 1] + va[:, 1] * vb[:, 0]).sum())
    assert np.isclose(i["expected_bundle_streamlines"], want, rtol=1e-5)
    assert i["seeding"]["seed_mass_without_state"] < 1e-3 * len(sid)


def test_interp_domain_and_stop_flags():
    """interp_domain: the voxels where the trilinear interpolant exceeds the threshold somewhere
    (checked against a dense sampling); trilinear_or_nan: NaN off the grid."""
    from mpat.galerkin import interp_domain, trilinear_or_nan

    rng = np.random.default_rng(0)
    vol = rng.random((6, 5, 4))
    D = interp_domain(vol, 0.7)
    u = np.linspace(-0.5, 0.5, 9)
    off = np.stack(np.meshgrid(u, u, u, indexing="ij"), -1).reshape(-1, 3)
    for ijk in np.ndindex(vol.shape):
        val = trilinear_or_nan(vol, np.asarray(ijk) + off)
        assert bool(np.nanmax(val, initial=-np.inf) > 0.7) == D[ijk] or np.isnan(val).any()
    assert np.isnan(trilinear_or_nan(vol, [[-0.2, 1, 1]]))[0] and np.isclose(trilinear_or_nan(vol, [[1, 2, 3]])[0],
                                                                               vol[1, 2, 3])


def test_seeded_bundle_interp_stop():
    """Stopping as pyAFQ's GPU tracker (stop "interp"): a walker stops where the stop map, trilinear
    at its landing point, is <= the threshold, ending in that voxel.  With a map that never stops a
    walker it is the voxel stopping on the same domain; with a blurred white matter, the forward and
    backward solves still agree (the stops' transpose is exact) and every bundle walker ends once."""
    from scipy import ndimage as ndi

    from mpat.galerkin import interp_domain, seed_peaks, seeds_from_mask

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    S, E = reg["cap_start"], reg["cap_end"]
    arc = dict(name="arc", include=[], exclude=None, start=S, end=E)
    cfg = GalerkinConfig(**{**CFG.key(), "points": 3})
    ijk = np.argwhere(wm)
    mid = np.zeros(wm.shape, bool)
    mid[tuple(ijk[np.abs(np.argsort(np.argsort(ijk[:, 0])) - len(ijk) // 2) < 6].T)] = True
    seeds = seeds_from_mask(mid, int(mid.sum()), rng=1)
    sid, pk = seed_peaks(sh, seeds, "repulsion100")
    base = Galerkin(sh, (1.0,) * 3, wm, absorb, cfg, device=DEV).seeded_bundle(arc, seeds, (sid, pk))
    scfg = GalerkinConfig(**{**cfg.key(), "stop": "interp"})
    assert scfg.key()["stop"] == "interp" and "stop" not in cfg.key()
    never = Galerkin(sh, (1.0,) * 3, wm, absorb, scfg, device=DEV, stop_map=np.ones(wm.shape))
    same = never.seeded_bundle(dict(arc, start=S & base_valid(wm, absorb), end=E & base_valid(wm, absorb)),
                               seeds, (sid, pk))
    assert np.isclose(same.info["expected_bundle_streamlines"], base.info["expected_bundle_streamlines"], rtol=1e-5)
    assert np.allclose(same.density, base.density, rtol=1e-4, atol=1e-6 * base.density.max())
    smap = ndi.gaussian_filter(wm.astype(float), 0.8)
    D = interp_domain(smap, 0.4)
    assert (D & wm).sum() == wm.sum() and D.sum() > wm.sum()
    m = Galerkin(sh, (1.0,) * 3, D, absorb, scfg, device=DEV, stop_map=smap)
    r = m.seeded_bundle(arc, seeds, (sid, pk))
    i = r.info
    assert i["transport_identity"] < 1e-5 and abs(i["endpoint_mass_check"] - 1) < 1e-5
    assert 0.2 * len(sid) < i["expected_bundle_streamlines"] < len(sid)
    assert r.endpoints[D & ~S & ~E].sum() == 0 and r.endpoints[S & D].sum() > 0     # some end at stops inside D
    with pytest.raises(ValueError):
        Galerkin(sh, (1.0,) * 3, D, absorb, GalerkinConfig(**{**CFG.key(), "stop": "interp"}), device=DEV,
                 stop_map=smap)                             # no points


def base_valid(wm, absorb):
    from mpat.galerkin import VALID_ENDS

    return np.isin(absorb, VALID_ENDS)


def test_bundle_interp_stop():
    """Launches with stop "interp": from the domain's faces (ROI voxels outside it), stopping at the
    points inside it.  A map that never stops a walker gives the voxel stopping's bundle; with a
    blurred white matter the forward and backward solves agree and every end is counted once."""
    from scipy import ndimage as ndi

    from mpat.galerkin import interp_domain

    sh, wm, cls, reg = arc_phantom()
    absorb = np.where(wm, 0, cls).astype(np.int8)
    v = base_valid(wm, absorb)
    arc = dict(name="arc", include=[], exclude=None, start=reg["cap_start"] & v, end=reg["cap_end"] & v)
    cfg = GalerkinConfig(**{**CFG.key(), "points": 3})
    scfg = GalerkinConfig(**{**cfg.key(), "stop": "interp"})
    lb = Galerkin(sh, (1.0,) * 3, wm, absorb, cfg, device=DEV).bundle(arc)
    ln = Galerkin(sh, (1.0,) * 3, wm, absorb, scfg, device=DEV, stop_map=np.ones(wm.shape)).bundle(arc)
    assert np.isclose(ln.info["expected_bundle_streamlines"], lb.info["expected_bundle_streamlines"], rtol=1e-5)
    smap = ndi.gaussian_filter(wm.astype(float), 0.8)
    m = Galerkin(sh, (1.0,) * 3, interp_domain(smap, 0.4), absorb, scfg, device=DEV, stop_map=smap)
    r = m.bundle(dict(arc, start=reg["cap_start"], end=reg["cap_end"]))
    assert r.info["transport_identity"] < 1e-5 and abs(r.info["endpoint_mass_check"] - 1) < 1e-5
    assert r.info["expected_bundle_streamlines"] > 0


def test_stop_map_own_grid():
    """A stop map on its own grid (volume, A: the fODF's voxel coordinates to its): the same map
    with its axes permuted and flipped gives the same values; point_domain is the voxels with a
    Gauss point above the threshold."""
    from mpat.galerkin import gauss_points, point_domain, stop_value

    rng = np.random.default_rng(1)
    vol = rng.random((7, 6, 5))
    pos = rng.random((200, 3)) * (np.array(vol.shape) - 1)
    vol2 = np.flip(vol.transpose(2, 0, 1), axis=0).copy()          # (z, x, y), z flipped
    A = np.zeros((4, 4))
    A[0, 2], A[0, 3] = -1.0, vol.shape[2] - 1                         # z' = (Z - 1) - z
    A[1, 0] = A[2, 1] = A[3, 3] = 1.0
    assert np.allclose(stop_value((vol2, A), pos), stop_value(vol, pos))
    D = point_domain((vol, np.eye(4)), vol.shape, 0.75, 1, 3)
    qu, _ = gauss_points(3)
    want = np.zeros(vol.shape, bool)
    for ijk in np.ndindex(vol.shape):
        val = stop_value(vol, np.asarray(ijk) + qu - 0.5)
        want[ijk] = bool((np.nan_to_num(val, nan=-np.inf) > 0.75).any())
    assert np.array_equal(D, want) and 0 < D.sum() < D.size


def test_point_turns_match_host():
    """The turns at points against the sum they compute, in float64 on the host:
        xn(p)_j = sum_q wq(q) fq(p, q) phi_j(u_q) sum_{l in the cone of p's direction} zq(l, q) (y(l) . phi(u_q))
    (fq: f w at the point, point_fodf's fODF; zq: 1 / the landing's cone's f w there), with about
    half the voxels given nothing to turn (the active set); and the backward turn its transpose,
    <g, F y> = <B g, y>.  In 1^3 and 2^3 bins."""
    from mpat.galerkin import point_fodf, raw_sf

    sh, wm, absorb, _ = _tube()
    for bins in (1, 2):
        cfg = GalerkinConfig(**{**CFG.key(), "bins": bins, "points": 3, "interp": ""})
        m = Galerkin(sh, (2.0, 2.0, 2.0), wm, absorb, cfg, device=DEV)
        kb, M, Q = m.kb, m.dirs.M, m.Q
        rng = np.random.default_rng(bins)
        y = rng.random((m.Lt, m.k3, 10))
        g = rng.random((m.P, m.k3, 10))
        dead = rng.random(m.Nv) < 0.5
        lidx, eidxT = m.d_lidx.numpy(), m.d_eidxT.numpy()
        y[lidx[dead][lidx[dead] >= 0]] = 0.0
        g[eidxT[dead][eidxT[dead] >= 0]] = 0.0
        m.ybuf = m._vec10(m.Lt)
        wp.copy(m.ybuf, wp.array(y.astype(np.float32), dtype=m.K.vec10, device=DEV))
        xn = wp.zeros((m.P, m.k3), dtype=m.K.vec10, device=DEV)
        m._fwd_turn(xn)
        got = xn.numpy().astype(np.float64)
        m._bwd_turn(wp.array(g.astype(np.float32), dtype=m.K.vec10, device=DEV),
                    wp.array(np.ones(m.Nv, np.uint8), dtype=wp.uint8, device=DEV))
        h = m.ybuf.numpy().astype(np.float64)
        sf, rows = raw_sf(sh, m.ijk, wm.shape, m.dirs)
        ev, ed = m.d_evox.numpy(), m.d_edir.numpy()
        lv, ld = m.d_lvox.numpy(), m.d_ldir.numpy()
        cone = np.zeros((M, M))
        for d in range(M):
            cone[d, m.dirs.cidx[m.dirs.cptr[d]:m.dirs.cptr[d + 1]]] = 1.0
        phi = basis(m.qu)
        want = np.zeros_like(got)
        for s in range(m.k3):
            sub = np.array([s // (kb * kb), (s // kb) % kb, s % kb])
            for q in range(Q):
                fw = point_fodf(sf, rows, m.ijk + (sub + m.qu[q]) / kb - 0.5, m.dirs).astype(np.float64) * m.dirs.w
                Z = (fw @ cone.T)[lv, ld]
                a = np.zeros((m.Nv, M))
                a[lv, ld] = np.divide(1.0, Z, out=np.zeros_like(Z), where=Z > 0) * (y[:, s] @ phi[q])
                want[:, s] += (m.qw[q] * fw[ev, ed] * (a @ cone.T)[ev, ed])[:, None] * phi[q][None, :]
        assert np.abs(want).max() > 0
        assert np.allclose(got, want, rtol=1e-4, atol=1e-5 * np.abs(want).max())
        assert np.isclose((g * got).sum(), (h * y).sum(), rtol=1e-4)


def test_arrivals_logged(caplog):
    """Verbose solves log, on each progress line, the walkers that have ended so far in their
    targets, and at the end the steps by which 99/99.9/99.99% of them had: a share reached at a
    logged step, in order, the last logged step's arrivals the final ones (seeded and not)."""
    import logging

    sh, wm, absorb, tube = _tube()
    m = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, CFG, device=DEV, verbose=True, log_every=0.0)
    ijk = np.argwhere(wm)
    seeds = ijk[np.abs(ijk[:, 0] - 10) <= 1].astype(float)
    from mpat.galerkin import seed_peaks

    sid, pk = seed_peaks(sh, seeds, "repulsion100")
    with caplog.at_level(logging.INFO, logger="mpat"):
        for r in (m.bundle(tube), m.seeded_bundle(tube, seeds, (sid, pk))):
            rows = [(s["kind"], a) for s in r.info["solves"] if s["forward"] for a in s["forward"].get("arrivals", ())]
            assert rows
            for _, a in rows:
                by = list(a["steps_by_share"].values())
                assert by == sorted(by) and by[-1] <= a["logged_at"][-1] and a["arrived"] > 0
    assert "arrived" in caplog.text and "subnormal:" in caplog.text and "99.99% of them by step" in caplog.text
    seeded = [a for s in r.info["solves"] if s["forward"] for a in s["forward"].get("arrivals", ())]
    assert {a["target"] for a in seeded} == {"end kind 1", "end kind 2", "their own ends"}
    own = sum(a["arrived"] for a in seeded if a["target"] == "their own ends")
    assert np.isclose(own, r.endpoints.sum(), rtol=1e-4)            # each half counted where it ends


def test_seeded_importance_steering():
    """Seeded bundles' forwards weighted by the partner, steered by their importance (the backward,
    known first): their expected arrivals at the start sum to the ends' arrivals (the transport
    identity); until "ends" stops when those still to come are below tol, the maps about as
    without; prune drops bins' walkers and tallies what they were worth, the ends losing at most
    that, the density about that share; prune also drops the backwards' importance where it times
    the all-halves forward's visits is small, the expected streamlines (from the backwards) moving
    by at most what that is bounded by.  Pruning and until "ends" are the defaults."""
    from mpat.galerkin import seed_peaks

    sh, wm, absorb, tube = _tube()
    ijk = np.argwhere(wm)
    seeds = ijk[np.abs(ijk[:, 0] - 10) <= 1].astype(float)
    sid, pk = seed_peaks(sh, seeds, "repulsion100")
    ref = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, CFG, device=DEV).seeded_bundle(tube, seeds, (sid, pk))
    tot, want = ref.endpoints.sum(), ref.info["expected_bundle_streamlines"]
    assert (GalerkinConfig().until, GalerkinConfig().prune) == ("ends", 1e-9)
    with pytest.raises(ValueError):
        Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, GalerkinConfig(sphere="repulsion100", until="nope"), device=DEV)
    for until, prune in (("ends", 0.0), ("walkers", 1e-7), ("ends", 1e-6)):
        cfg = GalerkinConfig(sphere="repulsion100", tol=1e-6, step_unit="vox", until=until, prune=prune)
        r = Galerkin(sh, (1.0, 1.0, 1.0), wm, absorb, cfg, device=DEV).seeded_bundle(tube, seeds, (sid, pk))
        imps = [s["forward"]["importance"] for s in r.info["solves"] if s["forward"] and "importance" in s["forward"]]
        assert len(imps) == 2 and r.info["config"].get("until", "walkers") == until
        bw = [s["backward"]["prune"] for s in r.info["solves"] if s["backward"] and "prune" in s["backward"]]
        assert len(bw) == (2 if prune else 0)
        bound = max([b["dropped_bound"] / b["expected_arrivals"] for b in bw] or [0.0])
        assert abs(r.info["expected_bundle_streamlines"] - want) <= (2 * bound + 1e-6) * want
        assert np.isclose(sum(i["expected_arrivals"] for i in imps), tot, rtol=1e-4)
        dropped = sum(i["dropped_abs"] for i in imps)
        share = dropped / tot
        assert (dropped > 0) == (prune > 0) and share < 1e-3
        if until == "ends":
            assert all(i["to_come_abs"] < cfg.tol * abs(i["expected_arrivals"]) for i in imps)
        assert abs(tot - r.endpoints.sum()) <= dropped + 1e-4 * tot
        d = np.abs(r.density - ref.density).sum() / ref.density.sum()
        assert d < 1e-5 + 10 * share, (d, share)                       # the density's loss: about that share


def test_soft_max_len():
    """A soft maximum length keeps a bundle's walkers within that distance of both its ROIs: long
    enough to hold the whole tube, the same bundle; shorter than the gap between the ROIs, no white
    matter to walk in (an error, as is a bundle with no end voxel)."""
    sh, wm, absorb, tube = _tube()
    b = Bundles(sh, (1.0, 1.0, 1.0), wm, absorb, CFG, device=DEV)
    ref = b.bundle(tube)
    r = b.bundle(dict(tube, soft_max_len=40.0))
    assert r.info["soft_max_len"] == 40.0
    assert np.isclose(r.info["expected_bundle_streamlines"], ref.info["expected_bundle_streamlines"], rtol=1e-6)
    near = b._near_ends(dict(tube, soft_max_len=7.0))
    assert near is not None and not (near & wm).any()       # the ROIs are 15 voxels apart: none is 7 from both
    with pytest.raises(ValueError, match="no white matter"):
        b.bundle(dict(tube, soft_max_len=7.0))
    with pytest.raises(ValueError, match="end ROI has no voxels"):
        b.bundle(dict(tube, end=np.zeros_like(tube["end"]), soft_max_len=40.0))
    mid = b._near_ends(dict(tube, soft_max_len=10.0)) & wm                # a slab between them, part of the tube
    assert 0 < mid.sum() < wm.sum() and set(np.unique(np.argwhere(mid)[:, 0])) == set(range(7, 13))
    assert b.bundle(dict(tube, soft_max_len=10.0)).info["expected_bundle_streamlines"] <= \
        ref.info["expected_bundle_streamlines"] * (1 + 1e-6)
