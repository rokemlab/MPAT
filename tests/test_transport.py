"""Bundles as linear transport (mpat.transport), on Warp's CPU device: the same kernels as on a
CUDA device, so no GPU is needed."""
import numpy as np
import pytest
from phantoms import crossing_phantom, tube_phantom

wp = pytest.importorskip("warp")

from mpat.transport import Directions, Transport, TransportConfig  # noqa: E402

DEV = "cpu"
CFG = TransportConfig(sphere="repulsion100", mfp=2.0)


def _tube():
    sh, wm, cls = tube_phantom(shape=(30, 11, 11), radius=3.0)
    start = np.zeros(wm.shape, bool)
    start[:3] = True
    end = np.zeros(wm.shape, bool)
    end[-3:] = True
    return sh, wm, np.where(wm, 0, cls).astype(np.int8), start, end


def test_directions():
    d = Directions("repulsion200", 35.0)
    assert np.isclose(d.w.sum(), 4 * np.pi)
    assert np.allclose(d.v[d.rev], -d.v)
    cone = np.zeros((d.M, d.M), bool)
    for m in range(d.M):
        cone[m, d.cidx[d.cptr[m]:d.cptr[m + 1]]] = True
    assert cone.diagonal().all() and (cone == cone.T).all()
    assert (cone[d.rev][:, d.rev] == cone).all()           # reversing both directions keeps the angle
    for o in range(8):                                     # each octant's directions have its signs
        dirs = d.odirs[o][d.odirs[o] >= 0]
        for k in range(3):
            assert ((d.v[dirs, k] < 0) == bool((o >> k) & 1)).all()


def test_tube():
    sh, wm, absorb, start, end = _tube()
    m = Transport(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV)
    r = m.bundle(dict(name="tube", include=[], exclude=None, start=start, end=end))
    i = r.info
    assert all(s["adjoint"]["converged"] and all(f["converged"] for f in s["forward"]) for s in i["solves"])
    assert i["n_solves"] == 3              # an adjoint solve (both end kinds), a forward solve per end kind
    # every walker counted at its launch (adjoint) and where it ends (forward) alike
    assert i["transport_identity"] < 1e-5
    assert abs(i["endpoint_mass_check"] - 1) < 1e-5
    assert 0 < i["expected_bundle_streamlines"] < i["launched_walkers"]
    # its ends are in the caps, half in each; its density fills the tube between them
    ep = r.endpoints
    assert ep[3:-3].sum() == 0 and np.isclose(ep[:3].sum(), ep[-3:].sum(), rtol=1e-4)
    assert (r.density[wm] > 0).all() and (r.density[~wm] == 0).all()
    c = r.density[:, 5, 5]
    assert c[15] > c[4] and c[15] > c[-5]                  # more paths through the middle
    assert r.seed_prob.max() <= 1 and r.seed_prob[start].max() > 0
    assert len(r.odf_ijk) == wm.sum() and np.allclose(r.odf_coef[:, 0] * 2 * np.sqrt(np.pi),
                                                      r.density[tuple(r.odf_ijk.T)], rtol=1e-4)
    # the walkers' directions there: along the tube (i)
    from dipy.core.sphere import Sphere
    from dipy.reconst.shm import sh_to_sf_matrix
    B = sh_to_sf_matrix(Sphere(xyz=np.eye(3)), sh_order_max=8, return_inv=False)
    sf = r.odf_coef @ B
    assert (sf[:, 0] > 3 * sf[:, 1]).mean() > 0.95


def test_graphs_match_launches():
    """The sweeps replayed as graphs (as on a GPU) give the same as launched one by one."""
    sh, wm, absorb, start, end = _tube()
    spec = dict(name="tube", include=[], exclude=None, start=start, end=end)
    try:
        g = Transport(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV, graphs=True).bundle(spec)
    except Exception as err:  # noqa: BLE001 -- older Warp captures graphs on CUDA devices only
        pytest.skip(f"no graph capture on Warp's CPU device here: {err}")
    p = Transport(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV, graphs=False).bundle(spec)
    assert np.array_equal(g.density, p.density)


def test_includes_excludes_crossing():
    sh, wm, cls, reg = crossing_phantom(n=31, width=5)
    absorb = np.where(wm, 0, cls).astype(np.int8)
    n = wm.shape[0]
    c = (n - 1) // 2
    ii, jj = np.indices(wm.shape)[:2]
    start, end = reg["capx"] & (ii < 2), reg["capx"] & (ii >= n - 2)
    m = Transport(sh, (2.0, 2.0, 2.0), wm, absorb, CFG, device=DEV)
    a = m.bundle(dict(name="A", include=[], exclude=None, start=start, end=end))
    d = a.density / a.density.max()
    z = 4
    assert d[c, 2:c - 3, z].max() == 0 and d[c, c + 4:n - 2, z].max() == 0      # nothing in the other arm
    assert d[2:n - 2, c, z].min() > 0.5
    # include ROIs across its own arm, either side of the crossing: the same bundle
    arm = lambda i: (ii == i) & (np.abs(jj - c) <= 3)  # noqa: E731
    b = m.bundle(dict(name="A", include=[arm(6), arm(n - 7)], exclude=None, start=start, end=end))
    assert b.info["n_solves"] == 12        # 4 avoid sets x (an adjoint, a forward per end kind)
    assert np.allclose(b.density, a.density, atol=1e-5 * a.density.max())
    assert abs(b.info["endpoint_mass_check"] - 1) < 1e-5
    # an include on the other bundle's arm (a 90 degree turn away), or the arm excluded: nothing
    other = (jj == 6) & (np.abs(ii - c) <= 3)
    assert m.bundle(dict(name="A", include=[other], exclude=None, start=start, end=end)).density.max() < \
        1e-6 * a.density.max()
    assert m.bundle(dict(name="A", include=[], exclude=arm(6), start=start, end=end)).density.max() == 0


def test_run_synthetic_subject(tmp_path):
    pytest.importorskip("AFQ")
    import json

    from synthetic_subject import make_afq, make_subject

    import mpat.transport as T

    f = make_subject(tmp_path / "raw")
    res = T.run(make_afq(f, tmp_path / "afq"), device=DEV, sphere="repulsion100", viz=False, cores=True)
    assert set(res) == {"A", "B"}
    out = tmp_path / "afq" / "mpat-transport"
    for b, band in (("A", f["A"]), ("B", f["B"])):
        r = res[b]
        assert r.info["n_solves"] == 12 and r.info["transport_identity"] < 1e-5
        assert abs(r.info["endpoint_mass_check"] - 1) < 1e-4
        d = r.density / r.density.max()
        assert d[band].mean() > 10 * max(d[~band].mean(), 1e-12)
        for suffix in ("density_map", "endpoints_map", "seedprob_probmap", "visitodf_dwimap", "core_path"):
            assert list(out.glob(f"*_bundle-{b}_desc-{suffix}*")), suffix
        info = json.loads(next(out.glob(f"*_bundle-{b}_desc-summary.json")).read_text())
        assert info["result"]["expected_bundle_streamlines"] > 0
    assert list(out.glob("*_desc-mpat_table.tsv"))


def test_upsample_sh():
    """The SH image on a finer grid: the same field of view, the mask's extent, values interpolated."""
    import nibabel as nib

    from mpat.transport import upsample_sh

    sh = np.zeros((6, 5, 4, 3), np.float32)
    sh[1:5, 1:4, 1:3] = [1.0, 2.0, -0.5]                  # a block of one fODF
    sh[1:5, 1:4, 1:3, 1] += np.arange(4)[:, None, None]     # ... with a ramp along x
    aff = np.array([[2.0, 0, 0, -10], [0, 2.0, 0, 5], [0, 0, 3.0, 1], [0, 0, 0, 1]])
    up = upsample_sh(nib.Nifti1Image(sh, aff), 1.0)
    assert up.shape == (12, 10, 12, 3) and np.allclose(up.header.get_zooms()[:3], 1.0)
    fine = up.get_fdata()
    # the same field of view: the volumes' outer corners coincide
    corner = lambda a, n: nib.affines.apply_affine(a, np.asarray(n) - 0.5)  # noqa: E731
    assert np.allclose(corner(aff, [0, 0, 0]), corner(up.affine, [0, 0, 0]))
    assert np.allclose(corner(aff, sh.shape[:3]), corner(up.affine, up.shape[:3]))
    # the mask keeps its extent (nearest neighbour): fine voxels inside coarse block voxels
    mask = np.any(fine != 0, -1)
    assert mask.sum() == 12 * np.any(sh != 0, -1).sum()      # 2 x 2 x 3 fine voxels per coarse one
    # inside, the constant coefficients stay and the ramp is linear between coarse centres
    assert np.allclose(fine[mask][:, 0], 1.0) and np.allclose(fine[mask][:, 2], -0.5)
    centre = lambda i: (i + 0.5) * 2.0 - 0.5  # noqa: E731   coarse voxel i's centre on the fine grid (x)
    x = np.arange(12)
    inner = (x >= centre(1)) & (x <= centre(4))
    ramp = fine[x, 4, 4, 1]
    assert np.allclose(np.diff(ramp[inner]), 0.5)


def test_run_upsampled(tmp_path):
    pytest.importorskip("AFQ")
    import nibabel as nib
    from scipy import ndimage

    from synthetic_subject import make_afq, make_subject

    import mpat.transport as T

    f = make_subject(tmp_path / "raw")
    kw = dict(device=DEV, sphere="repulsion100", viz=False, cores=False, bundles=["A"])
    afq = make_afq(f, tmp_path / "afq", space="template")      # ROIs warped by pyAFQ, as on real data
    base = T.run(afq, **kw)["A"]
    up = T.run(afq, upsample=1.0, out_dir=tmp_path / "up", **kw)["A"]
    assert up.density.shape == tuple(2 * n for n in base.density.shape)
    img = nib.load(next((tmp_path / "up").glob("*_bundle-A_desc-density_map.nii.gz")))
    assert np.allclose(img.header.get_zooms()[:3], 1.0)
    # the ROIs: made on the fODF's grid (where pyAFQ warps them), each fODF voxel's 2 x 2 x 2
    # voxels here -- the edge of the field included, where these caps lie
    for desc in ("start", "end", "include"):
        c = nib.load(next((tmp_path / "afq" / "mpat-transport").glob(f"*_bundle-A_desc-{desc}_mask.nii.gz")))
        u = nib.load(next((tmp_path / "up").glob(f"*_bundle-A_desc-{desc}_mask.nii.gz")))
        assert np.allclose(u.affine, img.affine)
        cd, ud = np.asarray(c.dataobj), np.asarray(u.dataobj)
        assert (ud == cd.repeat(2, 0).repeat(2, 1).repeat(2, 2)).all(), desc
        cen = lambda im, m: nib.affines.apply_affine(im.affine, np.argwhere(m).mean(0))  # noqa: E731
        assert np.allclose(cen(u, ud > 0), cen(c, cd > 0)), desc
    assert up.info["transport_identity"] < 1e-5 and abs(up.info["endpoint_mass_check"] - 1) < 1e-4
    # the same bundle: summed back onto the fODF's grid, it matches the solve there
    back = ndimage.zoom(up.density, 0.5, order=1)
    m = (back > 1e-3 * back.max()) | (base.density > 1e-3 * base.density.max())
    assert np.corrcoef(back[m], base.density[m])[0, 1] > 0.9
    band = f["A"]
    d = up.density / up.density.max()
    big = ndimage.zoom(band.astype(np.uint8), 2, order=0).astype(bool)
    assert d[big].mean() > 10 * max(d[~big].mean(), 1e-12)


def test_cli_dispatch(monkeypatch, tmp_path):
    """mpat ... --transport: the transport run gets the command line's options."""
    pytest.importorskip("AFQ")
    import AFQ.api.participant

    import mpat.bundles
    import mpat.transport
    from mpat.cli import main

    class Fake:
        def __init__(self, *args, **kwargs):
            self.args, self.kwargs = args, kwargs

    got = {}
    monkeypatch.setattr(AFQ.api.participant, "ParticipantAFQ", Fake)
    monkeypatch.setattr(mpat.bundles, "mpat_bd", lambda **kw: "bundles")
    monkeypatch.setattr(mpat.transport, "run", lambda afq, **kw: got.update(afq=afq, **kw))
    main(["d.nii.gz", "d.bval", "d.bvec", "t1.nii.gz", str(tmp_path), "--transport", "--transport-mfp", "3",
          "--transport-sphere", "repulsion200", "--angle", "30", "--tol", "1e-5", "--gpu", "cuda:1",
          "--bundles", "A", "B", "--no-viz", "--transport-upsample", "1.25", "--transport-max-iter", "800"])
    assert got["afq"].kwargs["bundle_info"] == "bundles"
    assert (got["mfp"], got["sphere"], got["angle"], got["tol"], got["device"]) == (3.0, "repulsion200", 30.0, 1e-5,
                                                                                  "cuda:1")
    assert got["bundles"] == ["A", "B"] and got["viz"] is False
    assert got["upsample"] == 1.25 and got["max_iter"] == 800
