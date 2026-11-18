---
orphan: true
---

(gpu)=
# Running the solves on a GPU

`mpat --gpu` runs MPAT's solves on an NVIDIA GPU. The results are the same as on the CPU, to
the solves' tolerance.

```bash
pip install warp-lang            # or: pip install "mpat-tractography[gpu] @ git+https://github.com/36000/MPAT.git"
mpat sub-01_dwi.nii.gz sub-01_dwi.bval sub-01_dwi.bvec sub-01_T1w.nii.gz out/ --n-proc 8 --gpu
```

From Python:

```python
mpat.run(afq, n_proc=8, gpu=True)          # the first CUDA device; or gpu="cuda:1"
```

## Two backends

The GPU code comes in two versions with the same kernels (the same operations, arguments and
threads), compiled by two different tools; `--gpu-backend` picks one:

| `--gpu-backend` | install | compiled by |
|---|---|---|
| `warp` (default) | `pip install warp-lang` | [NVIDIA Warp](https://github.com/NVIDIA/warp) |
| `numba` | `pip install "numba-cuda[cu13]"` (`[cu12]` for CUDA 12 drivers) | [numba-cuda](https://nvidia.github.io/numba-cuda/) |

Either needs only an NVIDIA driver besides: it compiles MPAT's kernels for your GPU the first
time they run and keeps them in its cache (Warp's in `~/.cache/warp`; numba's next to MPAT's
code when it can write there, otherwise in its own cache folder). If it finds no CUDA device, `--gpu` stops with an error that says so.
Recent Warp wheels need a Turing (RTX 20xx) or newer GPU; for older GPUs, such as the V100,
use numba-cuda with `[cu12]`, or Warp's CUDA 12 wheel from its GitHub releases.
numba-cuda needs a `cuda-core` that matches its own version. An `ImportError` that names
`cuda.core` (for example "cannot import name 'GraphBuilder' from 'cuda.core.cu13'") means they
do not match: numba-cuda 0.29 to 0.30.1 cannot import `cuda-core` 1.0 or newer, which another
package may have brought in. `pip install -U "numba-cuda[cu13]>=0.30.2"` fixes it.
Its linker, nvJitLink, should also come from the same CUDA release as its compiler (NVVM);
`numba-cuda[cu13]` does not install it, so the `gpu-numba` extra adds
`cuda-toolkit[nvjitlink]`, rather than leaving numba-cuda to find whichever copy conda or
another package left behind.

To see which is faster on your GPU, `examples/bench_gpu.py` times both (and the CPU) on a
synthetic chain, kernel by kernel and on whole solves; on real data, run a participant with
each (`examples/hbn_participant.py --gpu --gpu-backend ...`).

## What runs where

The chain is built on the CPU, as without `--gpu` (multithreaded, `--n-proc`; it is cached
next to the outputs), and so are the seeds' first steps. Then the chain and the seeds move to
the GPU, and everything the solves repeat runs there: both products with the transition
matrix, each step's masking and sums, the forward start, and each bundle's maps (density per
direction, endpoints). Only what the outputs need comes back. The soft maximum length
(`max_length` of {func}`mpat.run`) is solved on the GPU too. The live view ({doc}`live`) works with `--gpu`.

The solves read the whole transition matrix at every step, so they are limited by the GPU's
memory bandwidth more than by its arithmetic. Every product is in float64. GeForce cards do
float64 arithmetic far more slowly than data-center cards (A100, H100), which matters less
here than it would for dense arithmetic.

## Memory

On the GPU:

| | bytes |
|---|---|
| the chain | 9 per transition (5 with `--gpu-weights float32`; 68 with `--moments`) |
| a solution column | 33 per state (49 with `--live`; 4× with `--moments`), and 24 per seed half-walk and per absorbing voxel |

Bundles are packed into groups of columns that share every product with the chain, as on
the CPU (see `max_mem` in {doc}`options`). With `--gpu` the group size comes from the GPU's
free memory once the chain and seeds are there; `--max-mem` still sets the budget for the
work that stays on the CPU. The log gives both:

```text
chain on cuda:0: <states> states, <transitions> transitions, <GB> GB (float64 weights) in 1 segment(s), <s>s
column groups of up to <n> columns (<GB> GB free on cuda:0 for the solves; chain <GB> GB, launches <GB> GB there)
```

A GPU with less memory solves the same bundles in more groups, each a full solve. To leave
room for other jobs on the same GPU, cap what the solves use from Python:
`mpat.gpu.DeviceOps(chain, launches, max_gpu_mem=20e9)` passed as `ops=` to
{func}`mpat.segment_bundles`.

## Precision

The solves are float64 on the GPU as on the CPU. Their sums run in another order (the
forward products add with atomic operations), so the densities agree with a CPU run to the
tolerance, not bit for bit.

`--gpu-weights float32` stores the transition probabilities in float32, which nearly halves
the chain on the GPU. The walkers' mass then drifts by about 1e-7 per step, and densities
agree with float64 weights to about 1e-5. The moment chain's blocks are float32 either way.

## Without a GPU

`gpu="cpu"` runs the Warp kernels on Warp's CPU device, and `NUMBA_ENABLE_CUDASIM=1` runs
the numba kernels in numba's CUDA simulator. Both are slow and exist for testing: the test
suite checks both backends this way against the CPU solves, and builds their kernels for
CUDA without a GPU (`tests/test_gpu.py`, `tests/test_gpu_numba.py`). On a machine with a GPU,
`MPAT_TEST_DEVICE=cuda pytest tests/test_gpu.py tests/test_gpu_numba.py` runs the same tests
on it.
