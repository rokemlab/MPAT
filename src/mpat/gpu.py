"""MPAT's solves on a GPU, with NVIDIA Warp or numba-cuda.

    mpat dwi.nii.gz dwi.bval dwi.bvec t1.nii.gz out/ --n-proc 8 --gpu      # the first CUDA device
    mpat ... --gpu --gpu-backend numba                                      # numba-cuda instead of Warp
    mpat.run(afq, n_proc=8, gpu=True)                                       # or gpu="cuda:1"

Two backends run the same kernels (``mpat._warp_kernels``, ``mpat._numba_cuda_kernels``: one
per operation, the same arguments and thread layout), so they differ only in their compilers:

- ``"warp"``: ``pip install warp-lang`` (or ``mpat-tractography[gpu]``).  It also runs its
  kernels on the CPU (``device="cpu"``: slow, one thread; the tests use it).
- ``"numba-cuda"``: ``pip install "numba-cuda[cu13]"`` (``[cu12]`` for CUDA 12 drivers and
  pre-Turing GPUs).  NUMBA_ENABLE_CUDASIM=1 runs it in numba's CUDA simulator (very slow; the
  tests use it on small inputs).

Both need only an NVIDIA driver: they compile their kernels on first use and cache them.

What runs where.  The chain is built on the host (numba, multithreaded; it is cached), and so
are the seeds' first steps.  Then the chain's transitions, the launches and every solution
column move to the device, and everything the solves repeat runs there: the products with the
chain (both directions), each step's masking and sums, the forward start from the backward
solution, and the bundles' maps (densities, endpoints).  Only what the results need comes
back: a number per column each step, the densities and endpoints at the end.  The soft maximum
length's products run on the device too.

The algorithm is the host's (mpat.solve, through ``DeviceOps``), so the results agree with it
to the solves' tolerance (sums in a different order: forward products add with atomics).

Memory.  The chain needs ``n_transitions`` x 9 bytes (5 with ``weight_dtype="float32"``; 68
with moments), and each solution column 33 bytes per state (49 with the live view; 4x that
with moments).  Bundles are packed into groups of columns that fit in the device's free
memory (``max_gpu_mem`` to cap it).
float32 weights round each transition probability to 7 digits; the walkers' mass then drifts
by about that much per step, so densities agree with float64 weights to about 1e-5.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from . import kernel as K
from .solve import FLOAT_BLOCKS_INTERLEAVED, FLOAT_BLOCKS_PER_COLUMN, max_columns

log = logging.getLogger("mpat")

SEGMENT = 2 ** 30          # entries per device segment of a sparse matrix (kernels index with int32)
REDUCE_BLOCKS = 8192       # row blocks per column in the reductions on a GPU (64 on a CPU or simulator)
GPU_BLOCKS = 4             # float64 arrays per column on the device: total, term, next, and beta
GPU_BLOCKS_LIVE = 6        # r, beta, pi0, S and a work array per solve
WEIGHT_DTYPES = ("float64", "float32")
BACKENDS = ("warp", "numba-cuda")

_wp = None
_cuda = None


# ------------------------------------------------------------------ Warp
def warp():
    """The ``warp`` module (initialised), or an ImportError saying how to get it."""
    global _wp
    if _wp is None:
        try:
            import warp as wp
        except ImportError as err:
            raise ImportError("MPAT's GPU solves with Warp need NVIDIA Warp: pip install warp-lang "
                              "(or pip install 'mpat-tractography[gpu]')") from err
        wp.init()
        _wp = wp
    return _wp


def kernels():
    from . import _warp_kernels

    return _warp_kernels


def get_device(name=None):
    """A Warp device: ``None``, "cuda" or "gpu" for the first CUDA device, "cuda:1" for the
    second, "cpu" for Warp's CPU device (for testing)."""
    wp = warp()
    name = "cuda" if name in (None, "", "gpu", True) else str(name)
    if name.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError(
            "Warp sees no CUDA device (no NVIDIA GPU, or its driver is not loaded: "
            "`python -c 'import warp; warp.init()'` lists the devices Warp finds). Run without the "
            "GPU, or with device='cpu' to try the GPU code on the CPU (slow).")
    return wp.get_device(name)


# ------------------------------------------------------------------ numba-cuda
def numba_cuda():
    """``numba.cuda`` from numba-cuda, or an ImportError saying how to get it."""
    global _cuda
    if _cuda is None:
        if not hasattr(np, "row_stack"):          # numba-cuda <= 0.30.4 overloads np.row_stack,
            np.row_stack = np.vstack              # which NumPy 2.5 removed
        import importlib.util

        if importlib.util.find_spec("numba_cuda") is None:     # not numba's old built-in CUDA target
            raise ImportError('MPAT\'s GPU solves with numba need numba-cuda: pip install "numba-cuda[cu13]" '
                              '(or [cu12] for CUDA 12 drivers)')
        try:
            from numba import cuda
        except ImportError as err:                            # numba-cuda and cuda-core that do not match
            raise ImportError(f"numba-cuda could not be imported ({_cuda_versions()}): {err}.  numba-cuda 0.29 to "
                              "0.30.1 cannot import cuda-core 1.0 or newer; install versions that match: "
                              'pip install -U "numba-cuda[cu13]>=0.30.2" (or [cu12])') from err
        _cuda = cuda
    return _cuda


def _cuda_versions():
    from importlib.metadata import PackageNotFoundError, version

    out = []
    for p in ("numba-cuda", "cuda-core", "cuda-bindings"):
        try:
            out.append(f"{p} {version(p)}")
        except PackageNotFoundError:
            out.append(f"no {p}")
    return ", ".join(out)


def numba_kernels():
    numba_cuda()
    from . import _numba_cuda_kernels

    return _numba_cuda_kernels


def available(backend="warp"):
    """Whether ``backend`` ("warp" or "numba-cuda") is installed and sees a CUDA device."""
    try:
        if backend == "warp":
            return bool(warp().is_cuda_available())
        return bool(numba_cuda().is_available())
    except ImportError:
        return False


# ------------------------------------------------------------------ the two backends
class _WarpBackend:
    """Arrays and launches on a Warp device."""

    name = "warp"

    def __init__(self, device=None):
        self.wp = warp()
        self.K = kernels()
        self.device = device if hasattr(device, "is_cuda") else get_device(device)
        self.is_gpu = bool(self.device.is_cuda)
        self.key = f"warp:{self.device}"
        wp = self.wp
        self._dt = {np.dtype(np.float64): wp.float64, np.dtype(np.float32): wp.float32,
                    np.dtype(np.int32): wp.int32, np.dtype(np.uint8): wp.uint8, np.dtype(np.uint16): wp.uint16}

    def describe(self):
        return f"{self.device} (Warp)" if self.is_gpu else "Warp on the CPU"

    def array(self, a, dtype=None):
        a = np.ascontiguousarray(a, dtype=dtype)
        return self.wp.array(a, dtype=self._dt[a.dtype], device=self.device)

    def zeros(self, shape, dtype=np.float64):
        return self.wp.zeros(shape, dtype=self._dt[np.dtype(dtype)], device=self.device)

    def empty_like(self, a):
        return self.wp.empty_like(a)

    def clone(self, a):
        return self.wp.clone(a)

    def zero(self, a):
        a.zero_()
        return a

    def to_host(self, a):
        """A host copy (``numpy()`` copies from a GPU but is a view on Warp's CPU device)."""
        x = a.numpy()
        return x if self.is_gpu else x.copy()

    def nbytes(self, a):
        return a.capacity

    def launch(self, name, dim, *args):
        self.wp.launch(getattr(self.K, name), dim=dim, inputs=list(args), device=self.device)

    def synchronize(self):
        self.wp.synchronize_device(self.device)

    def free_memory(self):
        self.synchronize()                        # what earlier solves freed leaves the memory pool
        return float(self.device.free_memory)


class _NumbaCudaBackend:
    """Arrays and launches on a CUDA device with numba-cuda (or its simulator)."""

    name = "numba-cuda"
    THREADS = 256

    def __init__(self, device=None):
        cuda = self.cuda = numba_cuda()
        self.K = numba_kernels()
        from numba import config

        self.simulator = bool(getattr(config, "ENABLE_CUDASIM", False))
        name = "cuda" if device in (None, "", "gpu", True) else str(device)
        if not name.startswith("cuda"):
            raise ValueError(f"numba-cuda runs on CUDA devices ('cuda', 'cuda:1', ...), not {device!r}")
        if not cuda.is_available():
            raise RuntimeError("numba-cuda sees no CUDA device (no NVIDIA GPU, or its driver is not loaded: "
                               "`python -c 'from numba import cuda; cuda.detect()'` lists the devices). "
                               "Run without the GPU.")
        if ":" in name:
            cuda.select_device(int(name.split(":", 1)[1]))
        self.dev = cuda.get_current_device()
        self.is_gpu = True
        self.key = f"numba-cuda:{getattr(self.dev, 'id', 0)}"
        # the simulator: one thread, looping (its atomics serialise simulated threads on a lock)
        self.threads, self.max_blocks = (1, 1) if self.simulator else (self.THREADS, 2 ** 20)

    def describe(self):
        return "numba's CUDA simulator" if self.simulator else f"cuda:{getattr(self.dev, 'id', 0)} (numba-cuda)"

    def array(self, a, dtype=None):
        return self.cuda.to_device(np.ascontiguousarray(a, dtype=dtype))

    def zeros(self, shape, dtype=np.float64):
        return self.zero(self.cuda.device_array(shape, dtype=dtype))

    def empty_like(self, a):
        return self.cuda.device_array_like(a)

    def clone(self, a):
        b = self.cuda.device_array_like(a)
        b.copy_to_device(a)
        return b

    def zero(self, a):
        if a.size:
            self.launch("fill_zero", a.size, a.reshape(a.size))
        return a

    def to_host(self, a):
        return a.copy_to_host()

    def nbytes(self, a):
        return int(a.size) * a.dtype.itemsize

    def launch(self, name, dim, *args):
        dims = (int(dim),) if np.isscalar(dim) else tuple(int(d) for d in dim)
        total = math.prod(dims)
        if total == 0:
            return
        blocks = max(1, min(-(-total // self.threads), self.max_blocks))
        getattr(self.K, name)[blocks, self.threads](*args, *dims)

    def synchronize(self):
        self.cuda.synchronize()

    def free_memory(self):
        self.synchronize()
        ctx = self.cuda.current_context()
        if hasattr(ctx, "deallocations"):         # numba frees device arrays lazily (not the simulator)
            ctx.deallocations.clear()
        return float(ctx.get_memory_info()[0])


def _backend(backend, device):
    backend = {"numba": "numba-cuda", "cuda-numba": "numba-cuda"}.get(backend, backend)
    if backend == "warp":
        return _WarpBackend(device)
    if backend == "numba-cuda":
        return _NumbaCudaBackend(device)
    raise ValueError(f"unknown GPU backend {backend!r}: one of {BACKENDS}")


def _segments(ptr, cap):
    """State ranges [s0, s1) whose entries ptr[s1] - ptr[s0] number at most ``cap``."""
    n = ptr.size - 1
    out, s0 = [], 0
    while s0 < n:
        s1 = int(np.searchsorted(ptr, ptr[s0] + cap, side="right")) - 1
        s1 = min(max(s1, s0 + 1), n)
        out.append((s0, s1))
        s0 = s1
    return out


def _gb(n):
    return n / 1e9


# ------------------------------------------------------------------ the chain on a device
class DeviceChain:
    """A chain's transitions and absorption on a device, in segments of states whose entries
    fit 32-bit indices.  Kept on the chain (``DeviceChain.get``), so that every solve on that
    chain and device uses one copy; ``release`` frees it."""

    @classmethod
    def get(cls, chain, bk, weight_dtype="float64", segment=SEGMENT):
        cache = chain.__dict__.setdefault("_device_chains", {})
        key = (bk.key, weight_dtype, int(segment))
        if key not in cache:
            cache[key] = cls(chain, bk, weight_dtype, segment)
        return cache[key]

    @staticmethod
    def release(chain):
        chain.__dict__.pop("_device_chains", None)

    def __init__(self, chain, bk, weight_dtype="float64", segment=SEGMENT):
        if weight_dtype not in WEIGHT_DTYPES:
            raise ValueError(f"weight_dtype must be one of {WEIGHT_DTYPES}, not {weight_dtype!r}")
        t0 = time.time()
        self.bk, self.weight_dtype = bk, weight_dtype
        self.chain = chain
        self.moments = chain.ncomp > 1
        self.nb, self.nc, self.Ns, self.Na = chain.nb, chain.ncomp, chain.n_states, chain.a_lin.size
        self.nbytes = 0

        def up(a, dtype=None):
            arr = bk.array(a, dtype)
            self.nbytes += bk.nbytes(arr)
            return arr

        def segs(ptr, a1, a2):
            out = []
            for s0, s1 in _segments(ptr, segment):
                a, b = int(ptr[s0]), int(ptr[s1])
                out.append((s0, s1 - s0, up(ptr[s0:s1 + 1] - a, np.int32), up(a1(a, b)), up(a2(a, b))))
            return out

        if self.moments:
            self.q = segs(chain.Q_ptr, lambda a, b: chain.Q_ind[a:b].astype(np.int32),
                          lambda a, b: chain.Q_blk[a:b].astype(np.float32))
            self.r = segs(chain.R_ptr, lambda a, b: chain.R_ind[a:b].astype(np.int32),
                          lambda a, b: chain.R_blk[a:b].astype(np.float32))
        else:
            np_w = np.float64 if weight_dtype == "float64" else np.float32
            self.tn = up(chain.tmpl_n, np.int32)
            self.tb = up(chain.tmpl_b, np.int32)
            self.nbr = up(chain.nbr_t, np.int32)
            self.q = segs(chain.Q_ptr, lambda a, b: chain.Q_slot[a:b], lambda a, b: chain.Q_w[a:b].astype(np_w))
            self.r = segs(chain.R_ptr, lambda a, b: chain.R_ind[a:b].astype(np.int32),
                          lambda a, b: chain.R_val[a:b].astype(np.float64))
        self._kill, self._kill_src = None, None
        log.info("chain on %s: %d states, %d transitions, %.2f GB (%s weights) in %d segment(s), %.1fs",
                 bk.describe(), self.Ns, chain.n_transitions, _gb(self.nbytes),
                 "float32" if self.moments else weight_dtype, len(self.q), time.time() - t0)

    def kill(self):
        """The states past the soft maximum length (uint8 per state on the device), or None."""
        k = getattr(self.chain, "kill", None)
        if k is None:
            return None
        if self._kill_src is not k:
            self._kill = self.bk.array(k, np.uint8)
            self._kill_src = k
        return self._kill

    def backward(self, x, out):
        """out = Q x."""
        C = x.shape[1]
        for s0, n, ptr, a1, a2 in self.q:
            if self.moments:
                self.bk.launch("bsr_backward", (n, C), ptr, a1, a2, s0, x, out)
            else:
                self.bk.launch("stencil_backward", (n, C), ptr, a1, a2, self.tn, self.tb, self.nbr, self.nb, s0, x, out)
        return out

    def forward(self, x, out):
        """out = Q^T x."""
        self.bk.zero(out)
        C = x.shape[1]
        for s0, n, ptr, a1, a2 in self.q:
            if self.moments:
                self.bk.launch("bsr_forward", (n, C), ptr, a1, a2, s0, x, out)
            else:
                self.bk.launch("stencil_forward", (n, C), ptr, a1, a2, self.tn, self.tb, self.nbr, self.nb, s0, x, out)
        return out

    def absorb_rhs(self, absw, out):
        """out (rows x C) = the probability of being absorbed next step into voxels weighted by
        absw (absorbing voxels x C)."""
        C = absw.shape[1]
        for s0, n, ptr, ind, val in self.r:
            self.bk.launch("rblk_gather" if self.moments else "csr_gather", (n, C), ptr, ind, val, s0, absw, out)
        return out

    def absorbed_from(self, S, out):
        """out (absorbing voxels x C) = the mass absorbed from summed occupancy S."""
        self.bk.zero(out)
        C = S.shape[1]
        for s0, n, ptr, ind, val in self.r:
            self.bk.launch("rblk_scatter" if self.moments else "csr_scatter", (n, C), ptr, ind, val, s0, S, out)
        return out


class DeviceLaunches:
    """The seeds' half-walk launches on a device, one entry per (half-walk, first exit): the
    transient state it enters (-1: an absorbing voxel), that voxel, and the values."""

    @classmethod
    def get(cls, L, chain, bk):
        cache = L.__dict__.setdefault("_device_launches", {})
        if bk.key not in cache:
            cache[bk.key] = cls(L, chain, bk)
        return cache[bk.key]

    def __init__(self, L, chain, bk):
        nb = chain.nb
        H = L.hseed.size
        if L.lptr[-1] >= 2 ** 31:
            raise ValueError(f"{int(L.lptr[-1])} launch entries: more than the GPU code indexes (2^31)")
        h_of = np.repeat(np.arange(H), np.diff(L.lptr))
        key = L.lkey.astype(np.int64)
        vox = L.hseed[h_of] + chain.noff[key // nb]
        trans = chain.label[vox] == K.TRANSIENT
        state = np.where(trans, chain.tindex[vox] * nb + key % nb, -1)
        absorb = np.where(trans, -1, chain.aindex[vox])
        self.nbytes = 0

        def up(a, dtype):
            arr = bk.array(a, dtype)
            self.nbytes += bk.nbytes(arr)
            return arr

        self.H, self.P = H, L.pair_seed.size
        self.lptr = up(L.lptr, np.int32)
        self.ent_state = up(state, np.int32)
        self.ent_abs = up(absorb, np.int32)
        self.ent_vox = up(vox, np.int32)
        self.lval = up(L.lval, np.float64)
        self.hseed = up(L.hseed, np.int32)
        self.pair_w = up(L.pair_w, np.float64)


# ------------------------------------------------------------------ the solves' ops on a device
class DeviceOps:
    """``mpat.solve.HostOps`` on a device: the solves' arrays live there, and every repeated
    operation is a kernel of the backend (``WarpOps``, ``NumbaCudaOps``).

    chain, launches  as for HostOps (``launches`` can be set later, as ``.L``)
    device           "cuda" (default: the first CUDA device), "cuda:1", ... (Warp: also "cpu")
    weight_dtype     "float64" (default) or "float32": the transition probabilities on the
                     device (float32 halves the chain's largest array; see the module notes)
    max_gpu_mem      bytes the solves may use on the device (default: 90% of what is free
                     once the chain and launches are there)
    segment          most entries per device array of a sparse matrix (the kernels index with int32)
    backend          "warp" or "numba-cuda"
    """

    name = "device"

    def __init__(self, chain, launches=None, device=None, weight_dtype="float64", max_gpu_mem=None,
                 segment=SEGMENT, backend="warp"):
        self.bk = _backend(backend, device)
        self.name = self.bk.name
        self.chain, self.L = chain, launches
        self.nc = chain.ncomp
        self.max_gpu_mem = max_gpu_mem
        self.dc = DeviceChain.get(chain, self.bk, weight_dtype, segment)
        self.nblk = REDUCE_BLOCKS if self.bk.is_gpu and not getattr(self.bk, "simulator", False) else 64
        self._parts = {}

    @property
    def device(self):
        return getattr(self.bk, "device", None) or self.bk.key

    def launches(self):
        if self.L is None:
            raise ValueError("the GPU ops need the launches (ops.L = prepare_launches(...)) to solve bundles")
        return DeviceLaunches.get(self.L, self.chain, self.bk)

    # arrays
    def empty_like(self, a):
        return self.bk.empty_like(a)

    def copy(self, a):
        return self.bk.clone(a)

    def contiguous(self, a):
        return a

    def zeros(self, shape, dtype=np.float64):
        return self.bk.zeros(shape, dtype)

    def from_host(self, a):
        a = np.ascontiguousarray(a)
        if a.dtype not in (np.float64, np.uint8, np.int32):
            raise TypeError(f"no {a.dtype} arrays on the device")
        return self.bk.array(a)

    def to_host(self, a):
        return self.bk.to_host(a)

    def masked(self, a, mask):
        out = self.bk.empty_like(a)
        self.bk.launch("masked_copy", a.shape, a, mask, out)
        return out

    # the solves
    def mul(self, what, x, out):
        return self.dc.forward(x, out) if what == "forward" else self.dc.backward(x, out)

    def _reduce(self, kernel, x, *args):
        nrows, C = x.shape
        nblk = max(1, min(self.nblk, nrows))
        key = (nblk, C)
        if key not in self._parts:
            self._parts[key] = (self.zeros((nblk, C)), self.zeros(C))
        part, out = self._parts[key]
        self.bk.launch(kernel, (nblk, C), *args, nblk, part)
        self.bk.launch("sum_parts", C, part, out)
        return self.to_host(out)                  # out is reused: never hand out a view of it

    def accumulate(self, nxt, mask, total):
        return self._reduce("accumulate", nxt, nxt, mask, total, self.nc)

    def step(self, nxt, mask, src, cur):
        return self._reduce("step_source", nxt, nxt, mask, src, cur, self.nc)

    def col_mass(self, x):
        return self._reduce("col_mass", x, x, self.nc)

    def row_mass(self, x, minus=None):
        n = x.shape[0] // self.nc
        out = self.zeros(n)
        self.bk.launch("row_mass", n, x, x if minus is None else minus, int(minus is not None), self.nc, out)
        return self.to_host(out).astype(np.float32)

    # column groups
    def _per_column(self, live):
        """Device bytes per solution column (its share of the group's arrays): the solution
        blocks and the mask; per absorbing voxel the weights, the launch step's absorption and
        the endpoints; per half-walk its values and weights, per pair its share of P(bundle);
        a plan's ROI bits (at most one plan per column)."""
        L = self.launches()
        R = self.chain.n_states * self.nc
        blocks = GPU_BLOCKS_LIVE if live else GPU_BLOCKS
        return R * (8 * blocks + 1) + 8 * (3 * self.chain.a_lin.size + 3 * L.H) + int(np.prod(self.chain.pshape))

    def _budget(self):
        self.launches()                       # on the device before its free memory is read
        if self.max_gpu_mem is not None:
            return float(self.max_gpu_mem)
        return 0.9 * self.bk.free_memory() - 256e6

    def release(self):
        """Free the chain's and the launches' copies on the device (no solves after this)."""
        DeviceChain.release(self.chain)
        if self.L is not None:
            self.L.__dict__.pop("_device_launches", None)
        self.dc, self._parts = None, {}

    def max_columns(self, max_mem, live=False):
        if not self.bk.is_gpu:                # Warp's CPU device: host memory
            return max_columns(self.chain, self.L, max_mem,
                               FLOAT_BLOCKS_INTERLEAVED if live else FLOAT_BLOCKS_PER_COLUMN)
        budget = self._budget()
        if not math.isfinite(budget):         # the simulator
            return 256
        return int(max(1, min(256, budget // self._per_column(live))))

    def budget_note(self, max_mem):
        if not self.bk.is_gpu:
            return f"{max_mem / 1e9:.1f} GB budget, {self.bk.describe()}"
        return (f"{_gb(self._budget()):.1f} GB free on {self.bk.describe()} for the solves; chain "
                f"{_gb(self.dc.nbytes):.2f} GB, launches {_gb(self.launches().nbytes):.2f} GB there")

    def group(self, plans):
        return DeviceGroup(self, plans)

    def same_place(self, other):
        return isinstance(other, DeviceOps) and other.bk.key == self.bk.key and other.chain is self.chain


class WarpOps(DeviceOps):
    """``DeviceOps`` with NVIDIA Warp."""

    def __init__(self, chain, launches=None, device=None, **kwargs):
        super().__init__(chain, launches, device, backend="warp", **kwargs)


class NumbaCudaOps(DeviceOps):
    """``DeviceOps`` with numba-cuda."""

    def __init__(self, chain, launches=None, device=None, **kwargs):
        super().__init__(chain, launches, device, backend="numba-cuda", **kwargs)


class _Starts:
    """A device group's forward start per plan, (None, seed_prob, pb) as on the host, read from
    the device when first used.  A live solve refreshes its forward start every few steps;
    only the last one is read (at a redraw, and for the results), so the others never cross."""

    def __init__(self, group):
        self.group, self.version, self.got = group, group._version, {}

    def __len__(self):
        return len(self.group.plans)

    def __getitem__(self, i):
        if i not in self.got:
            if self.version != self.group._version:
                raise RuntimeError("a forward start read after a later one replaced it on the device")
            self.got[i] = self.group._start(i)
        return self.got[i]

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class DeviceGroup:
    """``mpat.solve.HostGroup`` on a device."""

    def __init__(self, ops, plans):
        bk = ops.bk
        self.ops, self.plans = ops, list(plans)
        ch = ops.chain
        self.dl = ops.launches()
        self.offs = np.cumsum([0] + [p.C for p in self.plans])
        C = int(self.offs[-1])
        R = ch.n_states * ch.ncomp
        mask = np.concatenate([p.mask for p in self.plans], axis=1)
        for p in self.plans:
            p.mask = None
        self.mask = ops.from_host(mask.astype(np.uint8, copy=False))
        del mask
        self.absw = ops.from_host(np.concatenate([p.absw for p in self.plans], axis=1).astype(np.float64))
        self.r = ops.zeros((R, C))
        ops.dc.absorb_rhs(self.absw, self.r)
        bk.launch("mul_mask", (R, C), self.r, self.mask)
        # each plan's avoid sets as bits per padded voxel (bit 0 exclude, 1 + i include ROI i)
        bits = np.zeros((len(self.plans), self.plans[0].taboo.shape[0]), np.uint8)
        col_plan = np.zeros(C, np.int32)
        col_mask = np.zeros(C, np.int32)
        self.small = []
        for i, (p, o0, o1) in enumerate(zip(self.plans, self.offs[:-1], self.offs[1:])):
            m = p.nX.bit_length() - 1
            if m > 7:
                raise ValueError(f"{p.name}: {m} include ROIs; the GPU code takes at most 7")
            b = p.taboo[:, 0].astype(np.uint8)
            for j in range(m):
                b |= p.taboo[:, 1 + j].astype(np.uint8) << np.uint8(j + 1)
            bits[i] = b
            xmask = np.array([1 + sum(2 << j for j in T) for T in p.subsets], np.int32)
            col_plan[o0:o1] = i
            col_mask[o0:o1] = xmask[p.xcol]
            self.small.append(dict(M=ops.from_host(np.ascontiguousarray(p.M, np.float64)),
                                   signs=ops.from_host(p.signs.astype(np.float64)),
                                   xmask=ops.from_host(xmask), colsign=ops.from_host(p.signs[p.xcol].astype(np.float64)),
                                   pbx=ops.zeros((self.dl.P, p.nX))))
        self.bits = ops.from_host(bits)
        self.col_plan, self.col_mask = ops.from_host(col_plan), ops.from_host(col_mask)
        self.vals = ops.zeros((self.dl.H, C))
        self.coef = ops.zeros((self.dl.H, C))
        self.a0 = ops.zeros((ch.a_lin.size, C))
        self._version = 0

    def forward_start(self, beta, pi0=None):
        """As HostGroup's; the starts are read from the device only when used (``_Starts``)."""
        ops, bk, dl = self.ops, self.ops.bk, self.dl
        ch = ops.chain
        C = int(self.offs[-1])
        if pi0 is None:
            pi0 = ops.zeros((ch.n_states * ch.ncomp, C))
        else:
            bk.zero(pi0)
        bk.launch("half_values", (dl.H, C), dl.lptr, dl.ent_state, dl.ent_abs, dl.lval, ch.ncomp, beta, self.absw,
                  dl.hseed, self.bits, self.col_plan, self.col_mask, self.vals)
        for i, (p, o0) in enumerate(zip(self.plans, self.offs[:-1])):
            sm = self.small[i]
            bk.launch("pair_terms", (dl.P, p.nX), self.vals, sm["M"], sm["signs"], dl.pair_w, dl.hseed, self.bits, i,
                      sm["xmask"], int(o0), p.nK, self.coef, sm["pbx"])
        bk.zero(self.a0)
        bk.launch("scatter_init", (dl.H, C), dl.lptr, dl.ent_state, dl.ent_abs, dl.ent_vox, dl.lval, ch.ncomp,
                  self.coef, self.bits, self.col_plan, self.col_mask, pi0, self.a0)
        kill = ops.dc.kill()
        if kill is not None:
            bk.launch("zero_rows", pi0.shape, pi0, kill, ch.ncomp)
        self._version += 1
        return pi0, _Starts(self)

    def _start(self, i):
        """Plan ``i``'s (None, seed_prob, pb) from the last forward start: P(each pair in the
        bundle), and per seed (a0, the launch step's absorption, stays on the device)."""
        ops, L = self.ops, self.ops.L
        pbx = self.small[i]["pbx"]
        pb = ops.zeros(pbx.shape[0])
        ops.bk.launch("row_sums", pbx.shape[0], pbx, pb)
        pb = ops.to_host(pb)
        return None, np.bincount(L.pair_seed, weights=L.pair_w * pb, minlength=len(L.seeds)), pb

    def _rows_density(self, S, beta, i, rows_per, n):
        out = self.ops.zeros(n)
        self.ops.bk.launch("rows_density", n, S, beta, self.small[i]["colsign"], int(self.offs[i]), rows_per, out)
        return self.ops.to_host(out)

    def density(self, S, beta, i):
        ch = self.ops.chain
        return self._rows_density(S, beta, i, ch.nb * ch.ncomp, ch.t_lin.size)

    def columns(self, X, i):
        """Plan ``i``'s columns of a group array (on the device, or a host copy from ``keep``),
        as a host array: only those columns cross."""
        o0, o1 = int(self.offs[i]), int(self.offs[i + 1])
        if isinstance(X, np.ndarray):
            return np.ascontiguousarray(X[:, o0:o1])
        out = self.ops.zeros((X.shape[0], o1 - o0))
        self.ops.bk.launch("copy_cols", out.shape, X, o0, out)
        return self.ops.to_host(out)

    def keep(self, x):
        """A copy of a group array that outlives the solve consuming it: on the host, so the
        device holds no extra array (``mpat.fields``' launch, with keep_fields)."""
        return self.ops.to_host(x)

    def launch_absorbed(self, a0, i):
        """Plan ``i``'s mass absorbed on the launch step, from the last forward start (on the
        device; ``a0`` is the host's, None here)."""
        return self.columns(self.a0, i)

    def maps(self, S, beta, starts):
        ops, ch = self.ops, self.ops.chain
        C = int(self.offs[-1])
        absorbed = ops.to_host(ops.dc.absorbed_from(S, ops.zeros((ch.a_lin.size, C))))
        absorbed += ops.to_host(self.a0)
        for i, (p, o0, o1) in enumerate(zip(self.plans, self.offs[:-1], self.offs[1:])):
            dcls = self._rows_density(S, beta, i, ch.ncomp, ch.n_states).reshape(ch.t_lin.size, ch.nb)
            yield dcls, (absorbed[:, o0:o1] * p.absw) @ p.signs[p.xcol]
