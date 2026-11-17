---
orphan: true
---

(brainstem)=
# Corticospinal and cortico-ponto-cerebellar tracts

Three bundle sets reach into the brainstem and cerebellum. They are off by default; each is
a flag:

```bash
mpat sub-01_dwi.nii.gz sub-01_dwi.bval sub-01_dwi.bvec sub-01_T1w.nii.gz out/ --n-proc 8 \
    --synthseg-cst --pontocerebellar --cerebellar
```

From Python, build the bundle dictionary with `mpat.bundles.mpat_bd(synthseg_cst=True,
pontocerebellar=True, cerebellar=True)` and pass it to the `ParticipantAFQ` as `bundle_info`.

The first two are defined by the subject's own SynthSeg labels: their ends and the sides of
the brain they may use. The corticospinal tracts use nothing registered from a template, so
they hold up where registration does not, as next to a tumor. The cortico-ponto-cerebellar
tracts add one template ROI, the middle cerebellar peduncle, so they depend on registration
around the pons and cerebellum.

## The corticospinal tract (`--synthseg-cst`)

Each side's tract starts in its precentral, paracentral and postcentral cortex and ends where
the brainstem leaves the field of view at the bottom. It stays out of the cerebellum and on
its own side, with a strict cut at the midline (x = 0), as for every other one-sided bundle.
These bundles replace pyAFQ's corticospinal tracts.

**The end** is the field of view's lower edge under the brainstem: the edge voxels below
SynthSeg's brainstem, within 5 mm of its footprint. This is the medulla if the scan stops
there, or the spinal cord below it, which SynthSeg does not label. Walkers that stop
elsewhere in the brainstem, for example in gray matter at its surface, do not count.

## The cortico-ponto-cerebellar tracts (`--pontocerebellar`)

Ten bundles, one per lobe and side. Each runs from one lobe's cortex to the cerebellar cortex
on the other side, through that side's middle cerebellar peduncle. Corticopontine fibers end
on the pontine nuclei, and pontocerebellar fibers start there, cross the midline and enter
the cerebellum through the middle cerebellar peduncle. A walk stands for the two in a row.

| Bundle | Start: SynthSeg's Desikan–Killiany parcels |
|---|---|
| `Left Frontopontocerebellar` | superior frontal, rostral and caudal middle frontal, pars opercularis, triangularis and orbitalis, lateral and medial orbitofrontal, precentral, paracentral, frontal pole, rostral and caudal anterior cingulate |
| `Left Parietopontocerebellar` | superior and inferior parietal, supramarginal, postcentral, precuneus, posterior and isthmus cingulate |
| `Left Temporopontocerebellar` | superior, middle and inferior temporal, banks of the STS, transverse temporal, fusiform, parahippocampal, entorhinal, temporal pole |
| `Left Occipitopontocerebellar` | lateral occipital, lingual, cuneus, pericalcarine |
| `Left Insulopontocerebellar` | insula |

The lobes follow FreeSurfer's lobe grouping, with the insula as its own lobe; the `Right`
bundles mirror these. Every bundle ends in the other side's cerebellar cortex.

**Where they cross.** Ends on opposite sides force a crossing but don't say where. Without
a rule, walks could cross through the corpus callosum or the vermis. A **sides** rule keeps
each bundle in its own side of the cerebrum and the other side of the cerebellum. Every voxel
takes the side of the nearest SynthSeg label that has one, including white matter SynthSeg
leaves unlabeled or calls CSF. The brainstem, and the cord under it, has no side. So the
brainstem is the only place these bundles can cross. No midline plane is involved, so a
tilted head doesn't matter.

**Which way into the cerebellum.** The ends and the sides rule allow a second route: from
cortex to the thalamus, across in the midbrain, and into the cerebellum through the superior
cerebellar peduncle. That is the cerebello-thalamo-cortical pathway, which walks can't tell
from this one by direction. So each bundle must also pass through the middle cerebellar
peduncle on its cerebellum's side: pyAFQ's waypoint templates for that peduncle (Jossinger
et al. 2022, the same ones `--cerebellar` uses; both of that side's waypoints, as one ROI),
warped to the subject. `mpat.bundles.mcp_rois()` builds them; they are downloaded on first
use.

**What to expect:**
- **Sizes:** frontal and parietal projections are much the largest; temporal and occipital
  ones are smaller. The insula's projection to the pons is sparse. To reach the cerebral
  peduncle, a walk from it has to go around the putamen, so treat the insular bundles as
  exploratory.
- **Where they cross in the brainstem:** in the pons, as they should. The middle peduncle
  rules out the midbrain. Through the medulla, walks would come up the pyramids and still
  have to reach the middle peduncle. To check, look at a bundle's density on the brainstem's
  midline, slice by slice: it should peak in the pons.
- **Overlap with `--cerebellar`:** a bundle shares the middle peduncle and the pons with
  pyAFQ's middle cerebellar peduncle bundle, as the fibers do.
- **Motor cortex** projects to both the spinal cord and the pons. Frontal walks from the
  precentral gyrus are part of the frontal bundle, not leakage from the corticospinal tract.

## pyAFQ's cerebellar peduncles (`--cerebellar`)

pyAFQ's own definitions of the inferior, middle and superior cerebellar peduncles
(`AFQ.api.bundle_dict.cerebellar_bd`, Jossinger et al. 2022): waypoint ROIs in MNI space,
warped to the subject. pyAFQ gets them too with `--pyafq`. They have no start or end ROIs,
so MPAT ends them in any gray matter or at the edge of the field of view. As a result, a
middle peduncle also reaches the cortex through the pons and the cerebral peduncle, where
pyAFQ's streamline criteria would cut it. Unlike the bundles above, these depend on
registration.

**Crossing only in the brainstem.** The middle peduncles (crossing in the pons) and the
superior ones (crossing in the midbrain, their decussation) are `cross_midline` in pyAFQ.
MPAT gives them the sides rule (`mpat.bundles.cerebellar_bd`): a "Left" middle or superior
peduncle stays in the left cerebellum and the right cerebrum, and the reverse for "Right",
so it can cross only in the brainstem, not through the vermis or the corpus callosum. The
side named is the cerebellum's: pyAFQ's waypoints put the peduncle's inferior waypoints on
that side and its superior one on the other. The inferior peduncles don't cross: they keep
the strict cut at the midline. pyAFQ (`--pyafq`) gets its own definitions unchanged.

## The rules

The rules these bundles use are available to any bundle, as entries of its `"mpat"`
dictionary (`mpat.bundles`):

| Rule | |
|---|---|
| `start_edge`, `end_edge` | `"inferior"`: the start / end is where the field of view's lower edge lies under the start / end ROI |
| `exclude_synthseg` | exclusion ROIs as SynthSeg labels |
| `sides` | `{"cerebrum": side, "cerebellum": side}`, each `"left"`, `"right"`, `"both"` or `"none"` |

For tumors, see also the tissue: white matter with edema can fall below the white-matter
threshold of the tissue labels, where walkers stop. These rules can't help with that.
