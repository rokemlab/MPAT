"""Time MPAT's GPU backends (NVIDIA Warp, numba-cuda) against each other and the CPU.

Both backends run the same kernels with the same thread layout, so this compares their
compilers.  On a synthetic chain (two crossing bundles, tests/phantoms.py) of a chosen size:

- the kernels' first-use compile time,
- each repeated operation (both chain products, a step's mask and sums) at several column
  counts, in ms per call (median), and as a multiple of the CPU's time (numba, --n-proc threads),
- whole solves of a group of bundles (the bundles repeated to fill ``--bundles`` columns),
  checking that every backend's results match the CPU's.

    python examples/bench_gpu.py                                   # both backends, cuda:0
    python examples/bench_gpu.py --size 161 --width 61 --columns 1 16 64 --device cuda:1
    python examples/bench_gpu.py --backends numba-cuda --no-cpu

Real data: time a whole participant with each backend (the chain is cached after the first
run, so the runs after it time the solves):

    python examples/hbn_participant.py --n-proc 8 --gpu --gpu-backend warp
    python examples/hbn_participant.py --n-proc 8 --gpu --gpu-backend numba-cuda
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))


def build(size, width, moments):
    from phantoms import crossing_phantom
    from scipy import ndimage

    from mpat.chain import Chain, ChainConfig
    from mpat.solve import prepare_launches

    sh, wm, cls, reg = crossing_phantom(n=size, width=width)
    seeds = np.argwhere(ndimage.binary_dilation(reg["capx"] | reg["capy"], np.ones((3, 3, 3))) & wm)
    c = reg["center"][0]
    rois = {}
    for name, off in (("A", -size // 6), ("B", size // 6)):
        r = np.zeros_like(wm)
        r[c + off] = True
        rois[name] = r & wm
    end = np.zeros_like(wm)
    end[-2:] = True
    rois["end"] = end & reg["capx"]
    t0 = time.perf_counter()
    ch = Chain.build(sh, (1.0, 1.0, 1.0), wm, cls, ChainConfig(moments=moments))
    L = prepare_launches(ch, seeds)
    print(f"chain: {ch.n_states:,} states, {ch.n_transitions:,} transitions, {len(seeds):,} seeds "
          f"(built in {time.perf_counter() - t0:.0f} s)")
    return ch, L, rois


def specs(rois, columns):
    base = [dict(include=[rois["A"], rois["B"]], end=rois["end"]), dict(end=rois["end"]),
            dict(include=[rois["A"]], end=rois["end"])]
    return [dict(base[i % 3], name=f"b{i}") for i in range(columns)]


def ops_for(name, ch, L, device, weights):
    from mpat.gpu import DeviceOps
    from mpat.solve import HostOps

    if name == "cpu":
        return HostOps(ch, L)
    return DeviceOps(ch, L, device=device, backend=name, weight_dtype=weights)


def sync(ops):
    if hasattr(ops, "bk"):
        ops.bk.synchronize()


def timed(f, ops, reps):
    f()
    sync(ops)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        sync(ops)
        ts.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(ts))


def kernels(name, ops, ch, columns, reps):
    """ms per call: backward, forward, mask + sums."""
    rows = {}
    rng = np.random.default_rng(0)
    for C in columns:
        x = rng.random((ch.n_states * ch.ncomp, C))
        mask = (rng.random(x.shape) > 0.1).astype(np.uint8)
        xd, out, tot = ops.from_host(x), ops.empty_like(ops.from_host(x)), ops.from_host(x)
        md = ops.from_host(mask)
        rows[C] = (timed(lambda: ops.mul("backward", xd, out), ops, reps),
                   timed(lambda: ops.mul("forward", xd, out), ops, reps),
                   timed(lambda: ops.accumulate(out, md, tot), ops, reps))
        xd = out = tot = md = None              # free them before the next size
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--size", type=int, default=121, help="phantom width in voxels (default 121)")
    p.add_argument("--width", type=int, default=41, help="bundle thickness in voxels (default 41)")
    p.add_argument("--moments", action="store_true", help="the moment chain (4 values per state)")
    p.add_argument("--columns", type=int, nargs="+", default=[4, 32], help="columns for the kernel timings")
    p.add_argument("--bundles", type=int, default=24, help="columns in the whole solves (default 24)")
    p.add_argument("--backends", nargs="+", default=["warp", "numba-cuda"], choices=("warp", "numba-cuda"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--weights", default="float64", choices=("float64", "float32"))
    p.add_argument("--n-proc", type=int, default=os.cpu_count(), help="CPU threads for the CPU's times")
    p.add_argument("--no-cpu", action="store_true", help="skip the CPU (the CPU solve can take long)")
    p.add_argument("--reps", type=int, default=10)
    a = p.parse_args(argv)

    import numba

    from mpat.solve import segment_bundles

    numba.set_num_threads(min(a.n_proc, numba.config.NUMBA_NUM_THREADS))
    ch, L, rois = build(a.size, a.width, a.moments)
    names = ([] if a.no_cpu else ["cpu"]) + a.backends
    table, solves, results = {}, {}, {}
    for name in names:
        t0 = time.perf_counter()
        ops = ops_for(name, ch, L, a.device, a.weights)
        x = ops.from_host(np.ones((ch.n_states * ch.ncomp, 1)))
        ops.mul("forward", x, ops.empty_like(x))               # compiles (or loads) the kernels
        sync(ops)
        setup = time.perf_counter() - t0
        table[name] = kernels(name, ops, ch, a.columns, a.reps)
        t0 = time.perf_counter()
        results[name] = segment_bundles(ch, specs(rois, a.bundles), L, ops=ops, max_mem=64e9)
        solves[name] = (time.perf_counter() - t0, setup)
        if hasattr(ops, "release"):
            ops.release()
        print(f"{name}: set up {setup:.1f} s (chain to the device, kernels compiled or loaded), "
              f"{a.bundles} bundles solved in {solves[name][0]:.1f} s")

    ref = names[0]
    print(f"\nms per call (median of {a.reps}); in brackets, the speedup over {ref}")
    print(f"{'backend':12s} {'columns':>7s} {'backward':>16s} {'forward':>16s} {'mask+sums':>16s}")
    for name in names:
        for C, row in table[name].items():
            cells = "".join(f"{t:9.2f} ({table[ref][C][i] / t:4.1f}x)" for i, t in enumerate(row))
            print(f"{name:12s} {C:7d} {cells}")
    print(f"\nwhole solve, {a.bundles} bundles:")
    for name in names:
        r = results[name]
        it = max(v.info["backward"]["iterations"] + v.info["forward"]["iterations"] for v in r.values())
        err = max(np.abs(r[k].density_t - results[ref][k].density_t).sum() / max(np.abs(results[ref][k].density_t).sum(), 1e-300)
                  for k in r)
        print(f"  {name:12s} {solves[name][0]:8.1f} s ({solves[ref][0] / solves[name][0]:4.1f}x)  "
              f"steps {it}, densities vs {ref}: {err:.1e}")


if __name__ == "__main__":
    main()
