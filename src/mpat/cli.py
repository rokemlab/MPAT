"""``mpat``: run pyAFQ (for its derivatives) and MPAT on one participant."""
import argparse
import json
import logging


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dwi"), p.add_argument("bval"), p.add_argument("bvec")
    p.add_argument("t1", help="T1w image, registered to the DWI (pyAFQ's t1_file)")
    p.add_argument("output_dir")
    p.add_argument("--n-proc", type=int, required=True, help="threads")
    p.add_argument("--max-mem", default="auto",
                   help="memory budget, e.g. 16GB, or auto (default): min(80%%, limit - 10 GB) of the job's "
                        "memory limit (cgroup / SLURM / physical), less what is already in use")
    p.add_argument("--moments", action="store_true", help="spatial first moments (narrower bundles)")
    p.add_argument("--tol", type=float, default=1e-6, help="remaining walker mass at which solves stop (default 1e-6)")
    p.add_argument("--angle", type=float, default=35.0, help="maximum turn per one-voxel hop (deg)")
    p.add_argument("--single-odf", action="store_true", help="first-order hops (pyAFQ-like), not doubleODF")
    p.add_argument("--bundles", nargs="+", help="pyAFQ bundle names (default: all)")
    p.add_argument("--max-length", type=float, default=250.0,
                   help="soft maximum streamline length, mm: states where walkers would mostly outlast "
                        "it (traps) end them, for every bundle (0: off)")
    p.add_argument("--no-krylov", action="store_true", help=argparse.SUPPRESS)   # no-op (kept for old scripts)
    p.add_argument("--out-dir", help="write MPAT's outputs here (default: <output_dir>/mpat); "
                   "output_dir is then only read for pyAFQ's derivatives")
    p.add_argument("--verbose", action="store_true",
                   help="log timing inside the solves (ms per power step)")
    p.add_argument("--mni-thalamus", action="store_true",
                   help="keep the template's thalamic start / end ROIs (default: SynthSeg's thalami)")
    p.add_argument("--no-viz", action="store_true",
                   help="skip the collage and viewer pages (written by default)")
    p.add_argument("--no-midline", nargs="+", default=(), metavar="BUNDLE",
                   help="ignore cross_midline=False for these bundles (no other-hemisphere exclusion)")
    p.add_argument("--no-fornix-exclude", action="store_true",
                   help="the anterior thalamic radiations as pyAFQ defines them, without the default exclusion "
                        "of the fornix / septum on the midline (MNI box x -3..3, y -28..15, z -5..22 mm)")
    p.add_argument("--synthseg-cst", action="store_true",
                   help="corticospinal tracts from SynthSeg endpoints only: precentral, paracentral and postcentral "
                        "cortex to where the brainstem leaves the field of view at the bottom, outside the "
                        "cerebellum, no waypoints, the same midline rule (mpat.bundles.synthseg_cst_bd); runs "
                        "SynthSeg's cortical parcellation and hypothalamus models once per subject if not done yet "
                        "(<stem>_model-synthseg2parc_dseg.nii.gz); with --pyafq, pyAFQ gets the same start and "
                        "the brainstem as its end")
    p.add_argument("--pontocerebellar", action="store_true",
                   help="add the cortico-ponto-cerebellar tracts: from each lobe's cortex (frontal, parietal, "
                        "temporal, occipital, insular; SynthSeg's parcellation) to the other side's cerebellar "
                        "cortex through its middle cerebellar peduncle (pyAFQ's template waypoints), crossing "
                        "the midline only in the brainstem (mpat.bundles.pontocerebellar_bd; 10 bundles, MPAT "
                        "only)")
    p.add_argument("--cerebellar", action="store_true",
                   help="add pyAFQ's cerebellar peduncles (AFQ.api.bundle_dict.cerebellar_bd: the inferior, middle "
                        "and superior peduncles of each side, from template waypoints, the middle and superior "
                        "crossing only in the brainstem; 6 bundles, also given to "
                        "pyAFQ with --pyafq)")
    p.add_argument("--schrodinger", action="store_true",
                   help="launch streamlines from gray matter instead of white-matter seeds, weighted so that "
                        "every gray-matter end voxel launches half and receives half of the gray-matter volume "
                        "behind it: one streamline end per mm^3 of gray matter, exactly (a Schrödinger bridge; "
                        "paths keep their transition probabilities; mpat.solve.end_bridge, cached per subject; "
                        "not with --moments)")
    p.add_argument("--pyafq", action="store_true",
                   help="also run pyAFQ's own pipeline (tractography, segmentation, profiles; export_all) "
                        "on the same ParticipantAFQ, for comparison: same inputs, kwargs and derivatives")
    p.add_argument("--afq-kwargs", default="{}",
                   help="extra ParticipantAFQ arguments: JSON, or a Python expression giving a dict, "
                        "with afd (AFQ.data.fetch), abd (AFQ.api.bundle_dict), mb (mpat.bundles) and the "
                        "AFQ.definitions image / mapping classes in scope, e.g. "
                        "'dict(bundle_info=abd.baby_bd(), brain_mask_definition=BabyBrainMask())'")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from AFQ.api.participant import ParticipantAFQ

    from .afq import run

    kwargs = afq_kwargs(a.afq_kwargs)
    if "bundle_info" not in kwargs:              # MPAT's bundles: pyAFQ's defaults + the HCC + the fornix
        from .bundles import mpat_bd

        kwargs["bundle_info"] = mpat_bd(synthseg_cst=a.synthseg_cst, pontocerebellar=a.pontocerebellar,
                                        cerebellar=a.cerebellar)
    elif a.synthseg_cst:
        p.error("--synthseg-cst replaces the CSTs of MPAT's default bundles; with your own bundle_info, "
                "add mb.synthseg_cst_bd() to it instead")
    elif a.pontocerebellar or a.cerebellar:
        p.error("--pontocerebellar / --cerebellar add to MPAT's default bundles; with your own bundle_info, "
                "add mb.pontocerebellar_bd() / abd.cerebellar_bd() to it instead")
    if "tracking_params" in kwargs:
        if "odf_model" not in kwargs["tracking_params"]:
            kwargs["tracking_params"]["odf_model"] = "CSD"
    else:
        kwargs["tracking_params"] = {"odf_model": "CSD"}

    afq = ParticipantAFQ(a.dwi, a.bval, a.bvec, a.t1, a.output_dir, **kwargs)
    run(afq, a.n_proc, a.max_mem, a.bundles, a.moments, a.tol, a.angle, not a.single_odf, viz=not a.no_viz,
        synthseg_thalamus=not a.mni_thalamus,
        max_length=a.max_length, no_midline=a.no_midline,
        out_dir=a.out_dir, verbose=a.verbose,
        fornix=not a.no_fornix_exclude, schrodinger=a.schrodinger)
    if a.pyafq:   # pyAFQ's recognition: MPAT's rules removed, MPAT-only bundles (fornix) dropped
        from .bundles import pyafq_bd

        bd = pyafq_bd(kwargs["bundle_info"])
        if a.synthseg_cst:     # pyAFQ reads the SynthSeg CST's ends as subject-space masks
            from .afq import synthseg_cst_for_pyafq

            bd = synthseg_cst_for_pyafq(afq, bd, a.moments, a.out_dir)
        logging.getLogger("mpat").info("pyAFQ's own pipeline on the same inputs (%d bundles) -> %s",
                                       len(bd), a.output_dir)
        ParticipantAFQ(a.dwi, a.bval, a.bvec, a.t1, a.output_dir,
                       **dict(kwargs, bundle_info=bd)).export_all(viz=not a.no_viz)


def afq_kwargs(text):
    """ParticipantAFQ keyword arguments from the command line: JSON, or else a Python
    expression evaluated with pyAFQ's usual helpers in scope (as in pyAFQ's examples):
    ``afd`` (AFQ.data.fetch), ``abd`` (AFQ.api.bundle_dict), every class in
    AFQ.definitions.image and AFQ.definitions.mapping, ``np`` and ``nib``."""
    try:
        kwargs = json.loads(text)
    except json.JSONDecodeError:
        import AFQ.api.bundle_dict as abd
        import AFQ.data.fetch as afd
        import AFQ.definitions.image as afm
        import AFQ.definitions.mapping as amp
        import nibabel as nib
        import numpy as np

        import mpat.bundles as mb

        scope = dict(afd=afd, abd=abd, mb=mb, np=np, nib=nib)
        scope.update({k: getattr(afm, k) for k in afm.__all__})
        scope.update({k: getattr(amp, k) for k in amp.__all__})
        try:
            kwargs = eval(text, scope)  # noqa: S307 -- the user's own command line
        except Exception as err:
            raise SystemExit(f"--afq-kwargs is neither JSON nor a valid Python expression: {err}") from err
    if not isinstance(kwargs, dict):
        raise SystemExit(f"--afq-kwargs must give a dict, got {type(kwargs).__name__}")
    return kwargs


if __name__ == "__main__":
    main()
