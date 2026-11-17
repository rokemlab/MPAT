"""MPAT's bundle rules on SynthSeg labels: the lobes, the cortico-ponto-cerebellar and SynthSeg
corticospinal definitions, the field of view's lower edge as an end, and the sides rule (a
bundle crossing the midline only in the brainstem), on a toy brain."""
import nibabel as nib
import numpy as np
import pytest
from phantoms import lobes_to_sh

pytest.importorskip("AFQ")

from AFQ.nn.synthseg import SynthSegLabels as S  # noqa: E402

from mpat import afq as A  # noqa: E402
from mpat import bundles as B  # noqa: E402


def toy_brain():
    """Labels, tissue and fODFs of a small brain (1 mm voxels, x < 0 left, z up) with every way
    across the midline: two cerebral blocks joined by a callosal bridge and by white matter
    SynthSeg left unlabeled; the brainstem below them; two cerebellar blocks joined to it by
    peduncles and to each other by a vermis behind it; and the cord, unlabeled, leaving the
    field of view at the bottom."""
    shape = (23, 9, 26)
    seg = np.zeros(shape, np.int32)
    I, J, K = np.indices(shape)
    left, right = I <= 10, I >= 11                          # world x = I - 11 + 0.5 > 0 from I = 11
    box = lambda i, j, k: ((I >= i[0]) & (I <= i[1]) & (J >= j[0]) & (J <= j[1])  # noqa: E731
                           & (K >= k[0]) & (K <= k[1]))
    cer_l, cer_r = box((2, 9), (1, 4), (17, 23)), box((13, 20), (1, 4), (17, 23))
    callosum = box((10, 12), (1, 4), (21, 23))
    unlabeled = box((10, 12), (1, 4), (18, 19))
    stem = box((9, 13), (1, 4), (4, 16))
    cord = box((10, 12), (2, 3), (0, 3))
    cbl_l, cbl_r = box((1, 6), (1, 7), (2, 10)), box((16, 21), (1, 7), (2, 10))
    mcp_l, mcp_r = box((7, 8), (1, 4), (6, 8)), box((14, 15), (1, 4), (6, 8))
    vermis = box((7, 15), (6, 7), (4, 6))
    seg[cer_l | (callosum & left)] = S.LEFT_CEREBRAL_WHITE_MATTER
    seg[cer_r | (callosum & right)] = S.RIGHT_CEREBRAL_WHITE_MATTER
    seg[stem] = S.BRAIN_STEM
    seg[cbl_l | mcp_l | (vermis & left)] = S.LEFT_CEREBELLUM_WHITE_MATTER
    seg[cbl_r | mcp_r | (vermis & right)] = S.RIGHT_CEREBELLUM_WHITE_MATTER
    wm = cer_l | cer_r | callosum | unlabeled | stem | cord | cbl_l | cbl_r | mcp_l | mcp_r | vermis
    ctx_l, ctx_r = box((2, 9), (1, 4), (24, 24)), box((13, 20), (1, 4), (24, 24))
    seg[ctx_l] = S.CTX_LH_SUPERIORFRONTAL
    seg[ctx_r] = S.CTX_RH_SUPERIORFRONTAL
    cbc_l, cbc_r = box((0, 0), (1, 7), (2, 10)), box((22, 22), (1, 7), (2, 10))
    seg[cbc_l] = S.LEFT_CEREBELLUM_CORTEX
    seg[cbc_r] = S.RIGHT_CEREBELLUM_CORTEX
    bottom = cord & (K == 0)                                 # where the cord leaves the field of view
    transient = wm & ~bottom
    absorb = np.full(shape, 2, np.int8)                      # CSF around everything
    absorb[ctx_l | ctx_r | cbc_l | cbc_r] = 1
    absorb[bottom] = 3
    absorb[transient] = 0
    sh_wm, sh_gm, sh_csf = lobes_to_sh([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]] * 2 + [[(1, 0, 0), (0, 1, 0), (0, 0, 1)]],
                                       2.0)
    sh = np.zeros(shape + (sh_wm.size,), np.float32)
    sh[...] = lobes_to_sh([[(1, 0, 0), (0, 1, 0), (0, 0, 1)]], 0.5)[0]
    sh[wm] = sh_wm
    sh[absorb == 1] = sh_gm
    aff = np.eye(4)
    aff[0, 3] = -11.0
    return dict(seg=seg, transient=transient, absorb=absorb, sh=sh, affine=aff, callosum=callosum,
                unlabeled=unlabeled, stem=stem, cord=cord, vermis=vermis, ctx_l=ctx_l, cbc_r=cbc_r,
                bottom=bottom, left=left, right=right, cbl_l=cbl_l, mcp_l=mcp_l)


@pytest.fixture(scope="module")
def toy():
    return toy_brain()


MNI = nib.Nifti1Image(np.zeros((2, 2, 2), np.uint8), np.eye(4))   # BundleDict's template, not fetched
MCP = {s: nib.Nifti1Image(np.full((2, 2, 2), v, np.float32), np.eye(4)) for s, v in (("left", 0.25), ("right", 0.75))}


# ------------------------------------------------------------------ the definitions
def test_lobes_cover_the_parcellation_once():
    parcels = [p for ps in B.LOBES.values() for p in ps]
    assert len(parcels) == len(set(parcels)) == 34
    assert {f"CTX_LH_{p}" for p in parcels} == {m.name for m in S if m.name.startswith("CTX_LH_")}


def test_pontocerebellar_bd():
    bd = B.pontocerebellar_bd(resample_to=MNI, mcp=MCP)
    assert len(list(bd)) == 10 and "Left Frontopontocerebellar" in bd and "Right Insulopontocerebellar" in bd
    r = B.rules(bd, "Left Occipitopontocerebellar")
    assert r["start_synthseg"] == B.lobe_labels("Left", "Occipital")
    assert r["end_synthseg"] == ["RIGHT_CEREBELLUM_CORTEX"] and r["synthseg_tol_mm"] == 0.0
    assert r["sides"] == dict(cerebrum="left", cerebellum="right")
    assert bd._dict["Left Occipitopontocerebellar"]["include"] == [MCP["right"]]   # the cerebellum's side
    assert bd._dict["Right Frontopontocerebellar"]["include"] == [MCP["left"]]
    assert B.synthseg_values(r["start_synthseg"])           # real SynthSeg labels
    assert not list(B.pyafq_bd(bd))                          # MPAT only: nothing for pyAFQ's recognition


def test_mpat_bd_adds_them(monkeypatch):
    # pyAFQ's own bundles stand in small (their templates are downloaded on first use)
    import AFQ.api.bundle_dict as abd
    import AFQ.data.fetch as afd

    monkeypatch.setattr(afd, "read_mni_template", lambda *a, **k: MNI)
    monkeypatch.setattr(B, "mcp_rois", lambda: MCP)
    one = lambda name: (lambda *a, **k: abd.BundleDict({name: dict(cross_midline=False, include=[MNI])}))  # noqa: E731
    for f, name in (("default_bd", "Left Corticospinal"), ("slf_bd", "Left SLF I"),
                    ("callosal_bd", "Callosum Motor")):
        monkeypatch.setattr(abd, f, one(name))
    monkeypatch.setattr(B, "hcc_bd", one("Left Cingulum Hippocampus"))
    monkeypatch.setattr(abd, "cerebellar_bd", one("Left Middle Cerebellar Peduncle"))
    plain, full = B.mpat_bd(), B.mpat_bd(pontocerebellar=True, synthseg_cst=True, cerebellar=True)
    pcb = set(B.pontocerebellar_bd(resample_to=MNI, mcp=MCP))
    assert not pcb & set(plain) and pcb <= set(full)
    assert "Left Middle Cerebellar Peduncle" in full and "Left Middle Cerebellar Peduncle" not in plain
    assert B.rules(full, "Left Corticospinal")["end_edge"] == "inferior"


def test_synthseg_cst_ends_at_the_bottom_outside_the_cerebellum():
    r = B.rules(B.synthseg_cst_bd(resample_to=MNI), "Left Corticospinal")
    assert r["end_synthseg"] == ["BRAIN_STEM"] and r["end_edge"] == "inferior"
    assert set(r["exclude_synthseg"]) == set(B.CEREBELLUM) and r["midline_tol"] == 0.0   # a strict cut


# ------------------------------------------------------------------ the field of view's lower edge
@pytest.mark.parametrize("orient", ["ras", "z_down", "z_on_axis_1"])
def test_inferior_edge_under_the_brainstem(toy, orient):
    stem, absorb, transient, bottom = toy["stem"], toy["absorb"], toy["transient"], toy["bottom"]
    aff = toy["affine"].copy()
    if orient == "z_down":                                   # the voxel axis runs top to bottom
        flip = lambda a: a[:, :, ::-1]  # noqa: E731
        stem, absorb, transient, bottom = map(flip, (stem, absorb, transient, bottom))
        aff[2, 2] = -1.0
    elif orient == "z_on_axis_1":                            # world z along the second voxel axis
        swap = lambda a: np.swapaxes(a, 1, 2)  # noqa: E731
        stem, absorb, transient, bottom = map(swap, (stem, absorb, transient, bottom))
        aff[:3, :3] = aff[:3, :3][:, [0, 2, 1]]
    # the cord below the brainstem's label: the end is where it leaves the field, not in the label
    assert not (bottom & stem).any()
    edge = A.inferior_edge_under(stem, absorb, transient, aff)
    assert (edge == bottom).all()
    # an edge elsewhere (the side of the field) is not under the brainstem
    absorb2 = absorb.copy()
    absorb2[(absorb2 == 2) & (np.indices(absorb.shape)[0] == 0)] = 3
    assert (A.inferior_edge_under(stem, absorb2, transient, aff) == bottom).all()
    assert not A.inferior_edge_under(np.zeros_like(stem), absorb, transient, aff).any()


# ------------------------------------------------------------------ the sides rule
def test_side_exclusion(toy):
    t = toy
    exc = A.side_exclusion(t["seg"], dict(cerebrum="left", cerebellum="right"), t["affine"])
    assert not (exc & t["stem"]).any() and not (exc & t["cord"]).any()        # no side
    assert (exc[t["callosum"] & t["right"]]).all() and not (exc[t["callosum"] & t["left"]]).any()
    # unlabeled white matter takes the nearer side (the middle column is a tie, either side)
    I = np.indices(t["seg"].shape)[0]
    assert (exc[t["unlabeled"] & (I >= 12)]).all() and not (exc[t["unlabeled"] & (I <= 10)]).any()
    assert (exc[t["vermis"] & t["left"]]).all() and not (exc[t["vermis"] & t["right"]]).any()
    assert not (exc & t["ctx_l"]).any() and not (exc & t["cbc_r"]).any()       # its own ends
    both = A.side_exclusion(t["seg"], dict(cerebrum="both"), t["affine"])
    assert not both.any()
    none = A.side_exclusion(t["seg"], dict(cerebellum="none"), t["affine"])
    assert none[t["cbl_l"]].all() and none[t["vermis"]].all() and not (none & t["stem"]).any()
    with pytest.raises(ValueError):
        A.side_exclusion(t["seg"], dict(cerebrum="up"), t["affine"])


def test_crossing_only_in_the_brainstem(toy):
    from mpat.chain import Chain, ChainConfig
    from mpat.solve import prepare_launches, segment_bundles

    t = toy
    ch = Chain.build(t["sh"], (1.0, 1.0, 1.0), t["transient"], t["absorb"], ChainConfig())
    from scipy import ndimage

    seeds = np.argwhere(ndimage.binary_dilation(t["ctx_l"], np.ones((3, 3, 3))) & t["transient"])
    L = prepare_launches(ch, seeds)
    spec = dict(start=t["ctx_l"], end=t["cbc_r"])
    sides = A.side_exclusion(t["seg"], dict(cerebrum="left", cerebellum="right"), t["affine"])
    got = segment_bundles(ch, [dict(spec, name="sides", exclude=sides), dict(spec, name="free")], L)
    d = {k: ch.to_volume(v.density_t) for k, v in got.items()}
    assert got["sides"].info["expected_bundle_streamlines"] > 0
    # with the rule: only the brainstem joins the two (the callosum, the white matter SynthSeg
    # left unlabeled and the vermis carry nothing past the midline); without it, they do
    I = np.indices(t["seg"].shape)[0]
    for path in ("callosum", "unlabeled"):                 # (the middle column is a tie: a dead end)
        far = t[path] & (I >= 12)
        assert d["sides"][far].sum() == 0 and d["free"][far].sum() > 0
    assert d["sides"][t["vermis"] & (I <= 10)].sum() == 0 and d["free"][t["vermis"]].sum() > 0
    assert d["sides"][t["stem"]].sum() > 0 and d["sides"][t["mcp_l"]].sum() == 0


# ------------------------------------------------------------------ the rules through bundle_specs
class _BD(dict):
    def transform_rois(self, name, mapping, ref_img):
        return {}


class _AFQ:
    def __init__(self, bd, seg_img):
        self._x = dict(bundle_dict=bd, mapping=None, segmentation_params={}, synthseg_model=seg_img)

    def export(self, k):
        return self._x[k]


def test_bundle_specs_apply_the_rules(toy):
    t = toy
    img = nib.Nifti1Image(t["seg"].astype(np.int16), t["affine"])
    bd = _BD({"Left Frontopontocerebellar":
              B.pontocerebellar_bd(lobes=("Frontal",), resample_to=MNI, mcp=MCP)._dict["Left Frontopontocerebellar"],
              "Left Corticospinal": B.synthseg_cst_bd(resample_to=MNI)._dict["Left Corticospinal"]})
    ref = nib.Nifti1Image(np.zeros(t["seg"].shape, np.uint8), t["affine"])
    specs = {s["name"]: s for s in A.bundle_specs(_AFQ(bd, img), ref, t["transient"], absorb=t["absorb"])}
    p = specs["Left Frontopontocerebellar"]
    assert (p["start"] == t["ctx_l"]).all() and (p["end"] == t["cbc_r"]).all()
    assert p["exclude"][t["callosum"] & t["right"]].all() and not (p["exclude"] & t["stem"]).any()
    c = specs["Left Corticospinal"]
    assert (c["end"] == t["bottom"]).all()                                     # where the cord leaves
    assert c["exclude"][t["cbl_l"]].all() and c["exclude_rois"][t["cbl_l"]].all()
    assert not c["exclude"][t["stem"] & (np.indices(t["seg"].shape)[0] <= 11)].any()
    with pytest.raises(ValueError, match="absorbing"):
        A.bundle_specs(_AFQ(bd, img), ref, t["transient"])


def test_mcp_rois_union_each_side(monkeypatch):
    import AFQ.data.fetch as afd

    aff = np.eye(4)
    def img(i):
        d = np.zeros((4, 4, 4), np.float32)
        d[i, 0, 0] = 0.5
        return nib.Nifti1Image(d, aff)
    cp = {"MCP_L_inferior_prob": img(0), "MCP_L_superior_prob": img(1),
          "MCP_R_inferior_prob": img(2), "MCP_R_superior_prob": img(3)}
    monkeypatch.setattr(afd, "read_cp_templates", lambda *a, **k: cp)
    m = B.mcp_rois()
    assert (np.flatnonzero(m["left"].get_fdata()[:, 0, 0]) == [0, 1]).all()
    assert (np.flatnonzero(m["right"].get_fdata()[:, 0, 0]) == [2, 3]).all()


def test_cerebellar_bd_crosses_only_in_the_brainstem(monkeypatch):
    import AFQ.api.bundle_dict as abd

    names = [f"{s} {p} Cerebellar Peduncle" for p in ("Inferior", "Middle", "Superior") for s in ("Left", "Right")]
    monkeypatch.setattr(abd, "cerebellar_bd", lambda *a, **k: abd.BundleDict(
        {n: dict(include=[MNI], cross_midline="Inferior" not in n) for n in names}, resample_to=MNI))
    bd = B.cerebellar_bd()
    assert B.rules(bd, "Left Middle Cerebellar Peduncle")["sides"] == dict(cerebellum="left", cerebrum="right")
    assert B.rules(bd, "Right Superior Cerebellar Peduncle")["sides"] == dict(cerebellum="right", cerebrum="left")
    assert B.rules(bd, "Left Inferior Cerebellar Peduncle")["sides"] is None
    assert bd._dict["Left Inferior Cerebellar Peduncle"]["cross_midline"] is False
    assert "mpat" not in B.pyafq_bd(bd)._dict["Left Middle Cerebellar Peduncle"]   # pyAFQ: its own
