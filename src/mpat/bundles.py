"""MPAT's bundles: pyAFQ bundle dictionaries plus MPAT's own rules.

A bundle is defined by a pyAFQ ``BundleDict`` entry (include / exclude / start / end ROIs,
``cross_midline``, ``space``), which pyAFQ warps into the subject for MPAT.  What MPAT does
beyond that is a rule, kept as data in one place:

* ``MPAT_RULES`` -- rules for pyAFQ's own bundles, by name, applied to whichever bundle
  dictionary a run uses (so a plain pyAFQ dictionary gets them too);
* an entry's ``"mpat"`` sub-dictionary -- rules carried by the bundle itself, which override
  ``MPAT_RULES``.  Newer pyAFQ rejects unknown criteria, so pyafq_bd(bd) strips this key
  before pyAFQ's own recognition.

Rules (``RULE_DEFAULTS`` hold for every bundle that does not set them):

    midline_tol        mm past the midline (world x = 0) a cross_midline=False bundle may go
                       before the other hemisphere is excluded (0: cut at x = 0)
    exclude_mni_boxes  extra exclusion ROIs, MNI boxes {x: (lo, hi), y: ..., z: ...} in mm,
                       warped with pyAFQ's mapping
    include_mni_boxes  the same, as inclusion ROIs
    start_synthseg     start / end ROIs as SynthSeg labels of the subject (AFQ.nn.synthseg
    end_synthseg       SynthSegLabels names or values); they replace any template start / end
    synthseg_tol_mm    how far start / end ROIs given as SynthSeg labels are grown, mm (None: pyAFQ's
                       dist_to_atlas, 4 mm by default, as for template ROIs; 0: the labels as they are)
    start_edge         "inferior": the start / end is where the field of view's lower edge lies
    end_edge           under the start / end ROI (its footprint, grown 5 mm, below its center): where
                       a tract leaves the scan at the bottom, e.g. the corticospinal tract in the
                       medulla or the spinal cord below it (mpat.afq.inferior_edge_under)
    exclude_synthseg   exclusion ROIs as SynthSeg labels of the subject (names or values)
    sides              {"cerebrum": side, "cerebellum": side}, side "left", "right", "both" or
                       "none": the bundle stays on these sides of the cerebrum and the cerebellum.
                       Every voxel takes the side of the nearest SynthSeg label that has one; the
                       brainstem (and the cord under it) has none and is never excluded, so a bundle
                       from one side of the cerebrum to the other side of the cerebellum can cross
                       the midline only in the brainstem (mpat.afq.side_exclusion)
    thalamus           "synthseg": a start / end ROI that is the template's thalamus becomes the
                       subject's SynthSeg thalamus (when pyAFQ's tissue comes from SynthSeg);
                       "template": keep it
    mpat_only          True: pyafq_bd drops the bundle even if it has ROIs pyAFQ reads (its
                       include alone, without MPAT's ends and sides, would not define it)

``mpat_bd()`` is MPAT's default dictionary: pyAFQ's defaults (default + SLF + callosal) and
the hippocampal cingulum and the fornix (MPAT only: its ends are SynthSeg labels, which
pyAFQ's segmentation does not read; ``fornix=False`` leaves it out; pyafq_bd(bd) gives the
dictionary pyAFQ's own recognition accepts).  ``mpat_bd(synthseg_cst=True)`` (``--synthseg-cst``)
swaps pyAFQ's corticospinal tracts for ones defined by SynthSeg endpoints only
(synthseg_cst_bd).  ``mpat_bd(pontocerebellar=True)`` (``--pontocerebellar``) adds the
cortico-ponto-cerebellar tracts, from each lobe to the other side's cerebellar cortex
(pontocerebellar_bd; MPAT only), and ``mpat_bd(cerebellar=True)`` (``--cerebellar``)
pyAFQ's cerebellar peduncles, the crossing ones crossing only in the brainstem (cerebellar_bd).
"""
from __future__ import annotations

import re

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

# The fornix and septum pellucidum on the midline, under the corpus callosum and over the
# thalami, from the fornix's columns (in front of the thalami) back along its body (MNI mm).
FORNIX_BOX_MNI = dict(x=(-3.0, 3.0), y=(-28.0, 15.0), z=(-5.0, 22.0))
# The fornix's body alone: on the midline over the thalami, behind the columns.  The fornix's
# waypoint.  FORNIX_BOX_MNI reaches down to the roof of the hypothalamus, so a walker going
# straight across from the hippocampal head to the hypothalamus (the short way, under the
# thalamus) touched it; the body is only reached the long way, up the crus behind the
# thalamus and forward over it, as the fornix runs.
FORNIX_BODY_MNI = dict(x=(-4.0, 4.0), y=(-25.0, -5.0), z=(8.0, 22.0))

RULE_DEFAULTS = dict(midline_tol=0.0, exclude_mni_boxes=(), include_mni_boxes=(), start_synthseg=None,
                     end_synthseg=None, synthseg_tol_mm=None, thalamus="synthseg", start_edge=None, end_edge=None,
                     exclude_synthseg=(), sides=None, mpat_only=False)

MPAT_RULES = {
    # walkers otherwise reach the thalamus along the fornix on the midline; the real ATR runs
    # lateral to the ventricles
    "Left Anterior Thalamic": dict(exclude_mni_boxes=[FORNIX_BOX_MNI]),
    "Right Anterior Thalamic": dict(exclude_mni_boxes=[FORNIX_BOX_MNI]),
}


def rules(bd, name):
    """The MPAT rules of bundle ``name`` in ``bd``: RULE_DEFAULTS, then MPAT_RULES, then the
    entry's own "mpat" sub-dictionary."""
    out = dict(RULE_DEFAULTS)
    out.update(MPAT_RULES.get(name, {}))
    try:
        own = bd[name].get("mpat") or bd[name].get("mcat") or {}   # "mcat": the old name, still read
    except (KeyError, AttributeError):
        own = {}
    out.update(own)
    return out


def synthseg_values(labels):
    """SynthSeg label values from names ("LEFT_HIPPOCAMPUS") or numbers."""
    from AFQ.nn.synthseg import SynthSegLabels

    return [int(SynthSegLabels[v]) if isinstance(v, str) else int(v) for v in labels]


def hcc_bd():
    """The hippocampal cingulum (Cingulum Hippocampus): MATLAB AFQ's two waypoints from pyAFQ's
    templates (which have no endpoint ROIs), and for MPAT one end in SynthSeg's hippocampus
    (the other anywhere valid).  Not among pyAFQ's defaults (hard to delineate at low SNR /
    resolution)."""
    import AFQ.api.bundle_dict as abd
    import AFQ.data.fetch as afd

    t = afd.read_templates()
    return abd.BundleDict({
        f"{side} Cingulum Hippocampus": dict(
            include=[t[f"HCC_roi1_{s}"], t[f"HCC_roi2_{s}"]], prob_map=t[f"HCC_{s}_prob_map"],
            cross_midline=False, mpat=dict(end_synthseg=[f"{side.upper()}_HIPPOCAMPUS"]))
        for side, s in (("Left", "L"), ("Right", "R"))})


def fornix_bd(resample_to=None):
    """The fornix, from the hippocampus (fimbria) along the body on the midline to the
    hypothalamus (columns, mammillary bodies): SynthSeg's hippocampus and hypothalamus as its
    ends, its body (FORNIX_BODY_MNI) as a waypoint, each side kept to its own hemisphere but for 3 mm.
    MPAT only.  pyAFQ's SynthSeg run has no hypothalamus, so MPAT reruns SynthSeg with its
    hypothalamus model for these ROIs when it is missing (<stem>_model-synthseg2hypo_dseg.nii.gz,
    reused afterwards; mpat.afq._synthseg_on_grid).  pyAFQ's tissue is not changed: there the
    hypothalamus is ventral diencephalon (mixed), whose rim is gray matter, and the end ROI,
    grown by dist_to_atlas as every end ROI is, reaches it."""
    import AFQ.api.bundle_dict as abd

    out = {}
    for side, s in (("Left", "LEFT"), ("Right", "RIGHT")):
        out[f"{side} Fornix"] = dict(
            cross_midline=False,
            mpat=dict(start_synthseg=[f"{s}_HIPPOCAMPUS"], end_synthseg=[f"{s}_HYPOTHALAMUS"],
                      include_mni_boxes=[FORNIX_BODY_MNI], midline_tol=3.0))
    kw = {} if resample_to is None else dict(resample_to=resample_to)
    return abd.BundleDict(out, **kw)


PYAFQ_ROI_KEYS = ("include", "start", "end", "prob_map", "recobundles")


def pyafq_bd(bd):
    """``bd`` as pyAFQ's recognition accepts it: each bundle's "mpat" rules removed (newer
    pyAFQ rejects unknown criteria), and bundles left with no ROI pyAFQ uses (the fornix,
    defined only by MPAT's SynthSeg ends) or marked mpat_only dropped."""
    out = bd.copy()
    for name in list(out._dict):
        entry = {k: v for k, v in out._dict[name].items() if k not in ("mpat", "mcat")}
        if not any(k in entry for k in PYAFQ_ROI_KEYS) or rules(out, name)["mpat_only"]:
            del out[name]
            continue
        out._dict[name] = entry
    return out


SYNTHSEG_CST_CORTEX = ("PRECENTRAL", "PARACENTRAL", "POSTCENTRAL")   # the central (sensorimotor) cortex
CEREBELLUM = ("LEFT_CEREBELLUM_WHITE_MATTER", "LEFT_CEREBELLUM_CORTEX", "RIGHT_CEREBELLUM_WHITE_MATTER",
              "RIGHT_CEREBELLUM_CORTEX")


def synthseg_cst_labels(side):
    """(start, end) SynthSeg labels of the SynthSeg corticospinal tract on ``side``: the
    precentral, paracentral and postcentral cortex (SYNTHSEG_CST_CORTEX, SynthSeg's cortical
    parcellation), and the brainstem."""
    h = {"Left": "LH", "Right": "RH"}[side]
    return [f"CTX_{h}_{p}" for p in SYNTHSEG_CST_CORTEX], ["BRAIN_STEM"]


def synthseg_cst_bd(resample_to=None):
    """The corticospinal tracts from SynthSeg endpoints only (no waypoints): each side's
    precentral, paracentral and postcentral cortex to where the brainstem leaves the field of
    view at the bottom (synthseg_cst_labels, end_edge "inferior": the lower edge of the field
    under the brainstem, in the medulla or the cord below it), kept out of the cerebellum and
    to its own hemisphere as pyAFQ's CST is (MPAT_RULES' midline tolerance).  The labels are
    used as they are (synthseg_tol_mm 0), not grown by pyAFQ's 4 mm dist_to_atlas: they are the
    subject's own, with no registration error to allow for.  Named as pyAFQ's CSTs, so they
    replace them in a dictionary.  The cortical parcels come from rerunning SynthSeg with its
    parcellation (mpat.afq.synthseg_parcellated); pyAFQ's own recognition reads them as
    subject-space masks, ending in the brainstem (mpat.afq.synthseg_cst_for_pyafq)."""
    import AFQ.api.bundle_dict as abd

    out = {}
    for side in ("Left", "Right"):
        start, end = synthseg_cst_labels(side)
        out[f"{side} Corticospinal"] = dict(cross_midline=False, mpat=dict(
            start_synthseg=start, end_synthseg=end, synthseg_tol_mm=0.0, end_edge="inferior",
            exclude_synthseg=list(CEREBELLUM)))
    kw = {} if resample_to is None else dict(resample_to=resample_to)
    return abd.BundleDict(out, **kw)


# Desikan-Killiany parcels (SynthSeg's cortical parcellation) of each lobe: FreeSurfer's lobe
# grouping (mri_annotation2label --lobesStrict), the cingulate split between the frontal and
# parietal lobes, and the insula a lobe of its own
LOBES = {
    "Frontal": ("SUPERIORFRONTAL", "ROSTRALMIDDLEFRONTAL", "CAUDALMIDDLEFRONTAL", "PARSOPERCULARIS",
                "PARSTRIANGULARIS", "PARSORBITALIS", "LATERALORBITOFRONTAL", "MEDIALORBITOFRONTAL", "PRECENTRAL",
                "PARACENTRAL", "FRONTALPOLE", "ROSTRALANTERIORCINGULATE", "CAUDALANTERIORCINGULATE"),
    "Parietal": ("SUPERIORPARIETAL", "INFERIORPARIETAL", "SUPRAMARGINAL", "POSTCENTRAL", "PRECUNEUS",
                 "POSTERIORCINGULATE", "ISTHMUSCINGULATE"),
    "Temporal": ("SUPERIORTEMPORAL", "MIDDLETEMPORAL", "INFERIORTEMPORAL", "BANKSSTS", "TRANSVERSETEMPORAL",
                 "FUSIFORM", "PARAHIPPOCAMPAL", "ENTORHINAL", "TEMPORALPOLE"),
    "Occipital": ("LATERALOCCIPITAL", "LINGUAL", "CUNEUS", "PERICALCARINE"),
    "Insular": ("INSULA",),
}
PONTOCEREBELLAR = {"Frontal": "Frontopontocerebellar", "Parietal": "Parietopontocerebellar",
                   "Temporal": "Temporopontocerebellar", "Occipital": "Occipitopontocerebellar",
                   "Insular": "Insulopontocerebellar"}


def lobe_labels(side, lobe):
    """SynthSeg's cortical parcels of ``lobe`` (LOBES) on ``side`` ("Left" / "Right")."""
    h = {"Left": "LH", "Right": "RH"}[side]
    return [f"CTX_{h}_{p}" for p in LOBES[lobe]]


def mcp_rois():
    """{"left" / "right": that side's middle cerebellar peduncle}, from pyAFQ's cerebellar
    peduncle templates (Jossinger et al. 2022; downloaded on first use): the union of the
    side's two MCP waypoints (``MCP_<L/R>_inferior_prob`` and ``MCP_<L/R>_superior_prob``),
    probabilistic, in MNI space."""
    import AFQ.data.fetch as afd

    cp = afd.read_cp_templates()
    out = {}
    for side, s in (("left", "L"), ("right", "R")):
        a, b = cp[f"MCP_{s}_inferior_prob"], cp[f"MCP_{s}_superior_prob"]
        if a.shape != b.shape or not np.allclose(a.affine, b.affine):
            b = resample_from_to(b, a, order=1)
        out[side] = nib.Nifti1Image(np.maximum(a.get_fdata(), b.get_fdata()).astype(np.float32), a.affine)
    return out


def pontocerebellar_bd(lobes=tuple(LOBES), resample_to=None, mcp=None):
    """The cortico-ponto-cerebellar tracts: from each lobe's cortex (SynthSeg's parcels, LOBES)
    to the other side's cerebellar cortex (SynthSeg), named "<side> <Lobe>pontocerebellar"
    (PONTOCEREBELLAR) by the side of the cortex, through the middle cerebellar peduncle of the
    cerebellum's side.  Corticopontine fibers end on the pontine nuclei and the pontocerebellar
    fibers start there, cross, and enter the cerebellum through that peduncle: the walk stands
    for the two in a row.

    The sides rule keeps the walk in its own side of the cerebrum and the other side of the
    cerebellum, so it can cross the midline only in the brainstem (not in the corpus callosum
    or the vermis).  The middle peduncle (an include: pyAFQ's template waypoints, mcp_rois,
    warped to the subject) rules out the other way into that cerebellum, the superior
    peduncle: cortex, thalamus, the decussation in the midbrain.  The ends are the subject's
    own SynthSeg labels, used as they are (synthseg_tol_mm 0).  ``mcp``: mcp_rois()'s result
    (default: fetched).  MPAT only (mpat_only: pyAFQ's recognition would read the peduncle but
    not the ends, so pyafq_bd drops them)."""
    import AFQ.api.bundle_dict as abd

    mcp = mcp_rois() if mcp is None else mcp
    out = {}
    for side, other in (("Left", "Right"), ("Right", "Left")):
        for lobe in lobes:
            out[f"{side} {PONTOCEREBELLAR[lobe]}"] = dict(cross_midline=True, include=[mcp[other.lower()]], mpat=dict(
                start_synthseg=lobe_labels(side, lobe), end_synthseg=[f"{other.upper()}_CEREBELLUM_CORTEX"],
                synthseg_tol_mm=0.0, sides=dict(cerebrum=side.lower(), cerebellum=other.lower()), mpat_only=True))
    kw = {} if resample_to is None else dict(resample_to=resample_to)
    return abd.BundleDict(out, **kw)


def cerebellar_bd():
    """pyAFQ's cerebellar peduncles (AFQ.api.bundle_dict.cerebellar_bd, Jossinger et al. 2022:
    template waypoints), with the two that cross allowed to cross only in the brainstem.

    The middle peduncles (pontocerebellar fibers, crossing in the pons) and the superior ones
    (crossing in the midbrain, their decussation) are cross_midline in pyAFQ, so MPAT would let
    them cross anywhere, the vermis or the corpus callosum included.  Each gets the sides rule
    instead: "<Side> Middle / Superior Cerebellar Peduncle" stays in that side of the
    cerebellum (pyAFQ's waypoints: the inferior ones on that side, the superior one on the
    other) and the other side of the cerebrum, so its only way across is the brainstem, which
    has no side.  The inferior peduncles don't cross (cross_midline False: the strict cut at
    the midline).  The ROIs are pyAFQ's unchanged, so pyAFQ (--pyafq) gets the same bundles."""
    import AFQ.api.bundle_dict as abd

    bd = abd.cerebellar_bd()
    for name in list(bd._dict):
        m = re.fullmatch(r"(Left|Right) (?:Middle|Superior) Cerebellar Peduncle", name)
        if m:
            side = m.group(1).lower()
            e = bd._dict[name]
            bd._dict[name] = dict(e, mpat=dict(e.get("mpat", {}), sides=dict(
                cerebellum=side, cerebrum="right" if side == "left" else "left")))
    return bd


def mpat_bd(fornix=True, synthseg_cst=False, pontocerebellar=False, cerebellar=False):
    """MPAT's default bundle dictionary (see the module docstring); ``synthseg_cst``: the
    corticospinal tracts from SynthSeg endpoints (synthseg_cst_bd) instead of pyAFQ's;
    ``pontocerebellar``: with the cortico-ponto-cerebellar tracts (pontocerebellar_bd);
    ``cerebellar``: with pyAFQ's cerebellar peduncles (cerebellar_bd: the inferior, middle and
    superior peduncles of each side, from template waypoints, Jossinger et al. 2022; the middle
    and superior ones cross only in the brainstem).  Without start / end ROIs, MPAT ends them
    in any gray matter or at the edge of the field of view: a middle peduncle reaches the
    cortex too, through the pons and the cerebral peduncle, where pyAFQ's streamlines are cut
    by its other criteria."""
    import AFQ.api.bundle_dict as abd

    bd = abd.default_bd() + abd.slf_bd() + abd.callosal_bd() + hcc_bd()
    bd = bd + fornix_bd() if fornix else bd
    if synthseg_cst:
        cst = synthseg_cst_bd()
        bd = bd[[n for n in bd if n not in set(cst)]] + cst
    if pontocerebellar:
        bd = bd + pontocerebellar_bd()
    if cerebellar:
        bd = bd + cerebellar_bd()
    return bd
