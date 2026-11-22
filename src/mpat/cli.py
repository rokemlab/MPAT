"""``mpat``: run pyAFQ (for its derivatives) and MPAT on one participant."""
import argparse
import json
import logging
import os


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dwi"), p.add_argument("bval"), p.add_argument("bvec")
    p.add_argument("t1", help="T1w image, registered to the DWI (pyAFQ's t1_file)")
    p.add_argument("output_dir")
    p.add_argument("--n-proc", type=int, help="threads (required unless --rois-only / --viz-only)")
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
    p.add_argument("--out-dir", help="write MPAT's outputs here (default: <output_dir>/mpat); "
                   "output_dir is then only read for pyAFQ's derivatives")
    p.add_argument("--verbose", action="store_true",
                   help="log timing inside the solves (ms per power step; with --galerkin, progress every 5 s and "
                        "each kernel's time in a step)")
    p.add_argument("--live", nargs="?", type=int, const=8765, default=None, metavar="PORT",
                   help="show the bundles in 3-D as they are solved, on a web page served from this "
                        "process (default port 8765, or the next free one): open the address it logs; "
                        "from another machine, forward the port first (ssh -L PORT:localhost:PORT host). "
                        "The backward and forward solves then advance together; same results")
    p.add_argument("--gpu", nargs="?", const="cuda", default=None, metavar="DEVICE",
                   help="--galerkin: the CUDA device (default the first; e.g. cuda:1). MPAT's chain "
                        "solves run on the CPU")
    p.add_argument("--galerkin", action="store_true",
                   help="solve the bundles with fixed-step walkers (probabilistic tracking's: a step, then a new "
                        "direction from the fODF within --angle), all at once and deterministically: quadratic "
                        "densities in bins, stepped exactly (Lagrange-Galerkin; mpat.galerkin); on a GPU with "
                        "NVIDIA Warp (--gpu DEVICE picks one); outputs in <output_dir>/mpat-galerkin (or "
                        "--out-dir); --tol (walkers left, of those launched; 1e-4 is plenty), --angle and "
                        "--colormap apply; with --viz-only / --rois-only, the figures / ROIs of a galerkin run")
    p.add_argument("--galerkin-bins", type=int, default=1, metavar="K",
                   help="--galerkin: K x K x K bins per voxel (1, 3, 5; default 1): finer keeps a beam's position "
                        "to 1/K voxel, at K^3 the memory and work; K > 1 solves each bundle in whole voxels first, "
                        "then in the bins only where that density reaches --galerkin-floor of its peak")
    p.add_argument("--galerkin-step", default="0.5", metavar="MM|Xvox",
                   help="--galerkin: step length in mm (default 0.5, as pyAFQ's step_size), or with 'vox' a "
                        "fraction of the smallest voxel side (e.g. 0.5vox, the step before mm); the log gives both; "
                        "outside 0.15-1 voxel it warns")
    p.add_argument("--galerkin-sphere", default="symmetric362",
                   help="--galerkin: the walkers' directions, a DIPY sphere (default symmetric362; also "
                        "repulsion200, symmetric642, repulsion724)")
    p.add_argument("--galerkin-max-steps", type=int, default=5000,
                   help="--galerkin: steps per solve at most (default 5000)")
    p.add_argument("--galerkin-floor", type=float, default=1e-3, metavar="FRAC",
                   help="--galerkin-bins > 1: solve in bins where the whole-voxel density reaches FRAC of its "
                        "peak (default 1e-3), grown by --galerkin-dilate voxels; walkers leaving it are lost")
    p.add_argument("--galerkin-dilate", type=int, default=2, metavar="VOXELS",
                   help="--galerkin-bins > 1: voxels the region of the fine solve is grown by (default 2)")
    p.add_argument("--galerkin-interp", nargs="?", const="all", default="all", choices=("all", "wm", "off"),
                   help="--galerkin-bins > 1: each bin's fODF, interpolated trilinearly between voxel centers: all "
                        "(default): from every voxel, grey matter too, as DIPY's trackers; wm: from the white "
                        "matter's voxels; off: every bin of a voxel has the voxel's fODF.  One bin per voxel: its "
                        "fODF either way")
    p.add_argument("--galerkin-points", type=int, default=0, metavar="N",
                   help="--galerkin: each bin's walkers turn by the fODF interpolated at N^3 Gauss points of the bin "
                        "(as DIPY's tracker, where each walker is) rather than one fODF per bin; N >= 3 (0: off).  "
                        "Fine fODF detail without finer bins; the fODF at the points is computed in the turns "
                        "(about the memory of no points), for spheres of up to 1024 directions")
    p.add_argument("--galerkin-seed-mask", default=None, metavar="NIFTI",
                   help="--galerkin: bundles from streamlines seeded in this mask (on the fODF's grid) as pyAFQ's "
                        "GPU tracker seeds them -- random seeds, one streamline per fODF peak at each, its halves "
                        "within the cone of the peak and of its reverse -- not from launches at the white "
                        "matter's edge.  To match pyAFQ's maps (pass the seed mask it wrote); one forward solve more, "
                        "and per end kind a backward and a forward")
    p.add_argument("--galerkin-seed-count", type=int, default=1_000_000, metavar="N",
                   help="--galerkin-seed-mask: seeds (default 1000000; pyAFQ's n_seeds with random_seeds, as DIPY's "
                        "random_seeds_from_mask).  The solves cost the same for any N: more seeds, less sampling "
                        "noise; the maps are for N seeds")
    p.add_argument("--galerkin-seed-threshold", type=float, default=0.0, metavar="T",
                   help="--galerkin-seed-mask: seed where the mask exceeds T (default 0)")
    p.add_argument("--galerkin-seed-rng", type=int, default=0, metavar="SEED",
                   help="--galerkin-seed-mask: the random generator's seed for the seeds (default 0)")
    p.add_argument("--galerkin-stop", default="interp", choices=("interp", "voxel"),
                   help="--galerkin: where walkers stop.  interp (default): as pyAFQ's GPU tracker, a walker stops "
                        "where the WM PVE, trilinear at its landing point, is <= --galerkin-stop-threshold, and "
                        "ends in that voxel whatever its tissue; the start / end ROIs decide.  Needs "
                        "--galerkin-points.  voxel: walkers move in the tissue labels' white-matter voxels and end "
                        "stepping into grey matter or the edge (CSF: lost)")
    p.add_argument("--galerkin-stop-grid", default="fodf", choices=("fodf", "native"),
                   help="--galerkin-stop interp: interpolate the WM PVE on the fODF's grid (default; resampled "
                        "first, as pyAFQ's tracker) or on its own grid (native: the T1's resolution when pyAFQ's "
                        "PVE is there)")
    p.add_argument("--galerkin-until", default="ends", choices=("walkers", "ends"),
                   help="--galerkin-seed-mask: when the forwards weighted by each half's partner stop (their "
                        "backward is solved first).  ends (default): when their expected arrivals still to come "
                        "(the walkers times their importance) are below --tol x those expected; walkers: when the "
                        "walkers left are below --tol x those launched, as every other solve")
    p.add_argument("--galerkin-prune", type=float, default=1e-9, metavar="FRAC",
                   help="--galerkin-seed-mask: drop a bin's walkers in the forwards weighted by the partner when "
                        "their expected arrivals are below FRAC x the solve's expected, and a bin's importance in "
                        "the backwards when it times the all-halves forward's visits there is below FRAC x that end "
                        "kind's arrivals (default 1e-9; 0: never).  What is dropped is tallied and logged: a bound "
                        "on the arrivals (streamlines) lost")
    p.add_argument("--galerkin-soft-max-len", type=float, default=0.0, metavar="MM",
                   help="--galerkin: a soft maximum streamline length (default 0: none): each bundle's walkers move "
                        "only within MM of both its start and its end ROI, where every streamline no longer than MM "
                        "lies (longer paths within still count).  A bundle without start or end voxels is an error")
    p.add_argument("--galerkin-stop-threshold", type=float, default=0.4, metavar="PVE",
                   help="--galerkin-stop interp: stop where the WM PVE is at most this (default 0.4, pyAFQ's)")
    p.add_argument("--galerkin-log-every", type=float, default=None, metavar="SECONDS",
                   help="--galerkin: seconds between a solve's progress lines (step, walkers left, time per step, "
                        "time to go; default 30, or 5 with --verbose)")
    p.add_argument("--movie", "--galerkin-movie", dest="movie", nargs="?", type=int, const=40, default=None,
                   metavar="FRAMES",
                   help="also write each bundle's solves as a video (<bundle>_desc-solves.mp4): the walkers "
                        "spreading from the ends (forward), the chance of reaching an end spreading back "
                        "(backward), then the bundle; at most FRAMES frames per solve (default 40).  The chain "
                        "(not with --live) and --galerkin")
    p.add_argument("--ends-only", "--galerkin-ends-only", dest="ends_only", action="store_true",
                   help="drop the bundles' include ROIs (waypoints): start / end ROIs, exclusions and the "
                        "hemisphere still apply (with --galerkin: one forward + backward pair per "
                        "end kind instead of one per end kind and subset of include ROIs); the optic radiations' "
                        "thalamic end is then cut to its lateral-posterior-inferior eighth (the LGN's).  For "
                        "every solver, and for --rois-only")
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
                        "cortex to the brainstem, "
                        "no waypoints, the same midline rule (mpat.bundles.synthseg_cst_bd); runs SynthSeg's "
                        "cortical parcellation and hypothalamus models once per subject if not done yet "
                        "(<stem>_model-synthseg2parc_dseg.nii.gz); with --pyafq, pyAFQ gets the same ends")
    p.add_argument("--pyafq", action="store_true",
                   help="also run pyAFQ's own pipeline (tractography, segmentation, profiles; export_all) "
                        "on the same ParticipantAFQ, for comparison: same inputs, kwargs and derivatives")
    p.add_argument("--rois-only", action="store_true",
                   help="solve nothing: write a finished run's ROI masks (start / end / include / exclude) "
                        "and a viewer page showing them, <stem>_desc-mpat_rois.html")
    p.add_argument("--viz-only", action="store_true",
                   help="solve nothing: rewrite a finished run's collage and viewer pages from its outputs "
                        "(e.g. after updating mpat); give the same arguments as the run.  The collage and the all-bundles "
                        "page always show every bundle; --bundles limits which per-bundle pages are redone")
    p.add_argument("--colormap", metavar="NAME",
                   help="NiiVue colour map of the density on every viewer page (default actc), "
                        "e.g. jet, hot, cubehelix; 'bundle': one hue per bundle")
    p.add_argument("--viz-rois", action="store_true",
                   help="viewer pages also show each bundle's include (green) / exclude (red) ROIs")
    p.add_argument("--flux", action="store_true", help="also write edge-flux maps per bundle (mpat.fields)")
    p.add_argument("--fixels", action="store_true",
                   help="also write an MRtrix fixel directory with each bundle's expected streamlines per fixel "
                        "and its share of all traffic (mpat.fixels; one extra whole-brain solve)")
    p.add_argument("--ptt", action="store_true",
                   help="use Parallel Transport Tractography (PTT) defaults (fod_thresh=0.08, fod_power=1.5, "
                        "curv_sigma=18.0, probe_radius=0.5, stop_mass=1e-4)")
    p.add_argument("--step", type=float, default=1.0, help="hop length in voxels (default 1.0)")
    p.add_argument("--stop-mass", type=float, default=None, help="MITTENS stopping regularizer tau")
    p.add_argument("--fod-thresh", type=float, default=None, help="relative fODF floor")
    p.add_argument("--fod-power", type=float, default=None, help="fODF sharpening exponent p")
    p.add_argument("--curv-sigma", type=float, default=None, help="PTT curvature & co-circularity sigma (deg)")
    p.add_argument("--probe-radius", type=float, default=None, help="PTT cylindrical probe radius (voxels)")
    p.add_argument("--sift2", action="store_true",
                   help="calibrate seed launch weights globally with path-consistent SIFT2 before segmentation")
    p.add_argument("--arbor", action="store_true",
                   help="also extract and write 3-D multi-branch cross-section ridge arbors (.trx, .json, .tsv)")
    p.add_argument("--arbor-branches", type=int, default=5,
                   help="maximum cortical endpoint branches per side for arbor extraction (default 5)")
    p.add_argument("--save-transitions", choices=("odx", "nifti", "both"), default=None,
                   help="persist transition probabilities as ODX (.odx), 4-D NIfTI (.nii.gz + codi/coasy), or both")
    p.add_argument("--afq-kwargs", default="{}",
                   help="extra ParticipantAFQ arguments: JSON, or a Python expression giving a dict, "
                        "with afd (AFQ.data.fetch), abd (AFQ.api.bundle_dict), mb (mpat.bundles) and the "
                        "AFQ.definitions image / mapping classes in scope, e.g. "
                        "'dict(bundle_info=abd.baby_bd(), brain_mask_definition=BabyBrainMask())'")
    a = p.parse_args(argv)
    if a.galerkin:
        from .galerkin import parse_step

        try:
            parse_step(a.galerkin_step)
        except ValueError as e:
            p.error(f"--galerkin-step: {e}")
        if a.galerkin_bins < 1:
            p.error("--galerkin-bins must be a positive integer")
        if a.galerkin_seed_count < 1:
            p.error("--galerkin-seed-count must be positive")
        if a.galerkin_stop == "interp" and not a.galerkin_points:
            p.error("--galerkin-stop interp (the default) needs --galerkin-points N (N >= 3); or pass "
                    "--galerkin-stop voxel")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    from AFQ.api.participant import ParticipantAFQ

    from .afq import run, write_rois, write_viz

    kwargs = afq_kwargs(a.afq_kwargs)
    if "bundle_info" not in kwargs:              # MPAT's bundles: pyAFQ's defaults + the HCC + the fornix
        from .bundles import mpat_bd

        kwargs["bundle_info"] = mpat_bd(synthseg_cst=a.synthseg_cst)
    elif a.synthseg_cst:
        p.error("--synthseg-cst replaces the CSTs of MPAT's default bundles; with your own bundle_info, "
                "add mb.synthseg_cst_bd() to it instead")
    if "tracking_params" in kwargs:
        if "odf_model" not in kwargs["tracking_params"]:
            kwargs["tracking_params"]["odf_model"] = "CSD"
    else:
        kwargs["tracking_params"] = {"odf_model": "CSD"}

    if a.gpu and not a.galerkin:
        p.error("--gpu picks --galerkin's device; MPAT's chain solves run on the CPU")
    if a.galerkin and not (a.rois_only or a.viz_only):     # fixed-step walkers on a GPU, not the chain
        from .galerkin import run_cli as galerkin_cli

        galerkin_cli(ParticipantAFQ(a.dwi, a.bval, a.bvec, a.t1, a.output_dir, **kwargs), a)
        return
    if a.galerkin and not a.out_dir:           # --rois-only / --viz-only of a galerkin run
        a.out_dir = os.path.join(a.output_dir, "mpat-galerkin")
    afq = ParticipantAFQ(a.dwi, a.bval, a.bvec, a.t1, a.output_dir, **kwargs)
    if a.rois_only:
        page = write_rois(afq, a.bundles, a.moments, synthseg_thalamus=not a.mni_thalamus, no_midline=a.no_midline,
                          out_dir=a.out_dir,
                          fornix=not a.no_fornix_exclude, ends_only=a.ends_only)
        logging.getLogger("mpat").info("ROI page: %s", page)
        return
    if a.viz_only:
        out = write_viz(afq, a.bundles, a.moments, out_dir=a.out_dir, rois=a.viz_rois, colormap=a.colormap)
        logging.getLogger("mpat").info("figures rewritten in %s", out)
        return
    if a.n_proc is None:
        p.error("--n-proc is required (except with --rois-only or --viz-only)")
    run(afq, a.n_proc, a.max_mem, a.bundles, a.moments, a.tol, a.angle, not a.single_odf, viz=not a.no_viz,
        synthseg_thalamus=not a.mni_thalamus,
        max_length=a.max_length, viz_rois=a.viz_rois, no_midline=a.no_midline,
        out_dir=a.out_dir, verbose=a.verbose,
        fornix=not a.no_fornix_exclude, flux=a.flux, fixels=a.fixels, viz_colormap=a.colormap,
        ptt=a.ptt, step=a.step, stop_mass=a.stop_mass, fod_thresh=a.fod_thresh,
        fod_power=a.fod_power, curv_sigma=a.curv_sigma, probe_radius=a.probe_radius,
        sift2=a.sift2, arbor=a.arbor, arbor_branches=a.arbor_branches,
        save_transitions=a.save_transitions or False,
        live=a.live, ends_only=a.ends_only,
        movie=a.movie)
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
        import nibabel as nib
        import numpy as np

        import mpat.bundles as mb

        scope = dict(mb=mb, np=np, nib=nib)
        try:
            import AFQ.api.bundle_dict as abd
            import AFQ.data.fetch as afd
            import AFQ.definitions.image as afm
            import AFQ.definitions.mapping as amp

            scope.update(afd=afd, abd=abd)
            scope.update({k: getattr(afm, k) for k in afm.__all__})
            scope.update({k: getattr(amp, k) for k in amp.__all__})
        except ImportError:
            pass
        try:
            kwargs = eval(text, scope)  # noqa: S307 -- the user's own command line
        except Exception as err:
            raise SystemExit(f"--afq-kwargs is neither JSON nor a valid Python expression: {err}") from err
    if not isinstance(kwargs, dict):
        raise SystemExit(f"--afq-kwargs must give a dict, got {type(kwargs).__name__}")
    return kwargs


if __name__ == "__main__":
    main()
