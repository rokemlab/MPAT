"""A bundle's solves as a video (``mpat --movie``, the chain or --galerkin): per frame, maximum-intensity
projections (sagittal, coronal, axial; RAS) of

- forward: the walkers in each voxel (all directions), log10 of the frame's peak, 4 decades:
  where the walkers launched from one end are, step by step;
- backward: the probability that a walker in each voxel (heading along its fODF) ends in the
  other end, log10, 1e-4 .. 1: the importance spreading back from that end (the chain: that it
  ends in one of the bundle's end ROIs, averaged over the voxel's direction classes);

The chain (ChainMovies) solves a group of bundles together, a backward solve (every end kind
at once) and then a forward one: each bundle's video is cut from the group's columns (those
avoiding no include ROI), the forward frames showing its walkers' wavefront.

each solve thinned to at most ``frames`` frames, evenly spaced, and the bundle's density last
(log10 of its peak, 3 decades), held for a moment.  Opaque colours on the white matter's
outline (grey).  Written as pyAFQ writes its videos: an .mp4 (H.264, yuv420p, faststart) with
imageio's ffmpeg.
"""
from __future__ import annotations

import logging

import nibabel as nib
import numpy as np

log = logging.getLogger("mpat")

KINDS = {   # colour map, log10 range below the reference (None: absolute), what it shows
    "forward": ("magma", 4.0, "walkers per voxel (log10, of this frame's peak)"),
    "backward": ("viridis", None, "P(a walker here ends in the target), log10, 1e-4 .. 1"),
    "density": ("inferno", 3.0, "bundle density (log10, of its peak)"),
}
FPS = 12            # frames per second
HOLD = 18           # frames the final density is shown for (1.5 s)
CRF = 23            # H.264 quality (lower: better, bigger)


class SolveMovie:
    """Collects a bundle's frames; ``write`` renders them to ``path`` (.mp4)."""

    def __init__(self, name, path, affine, white_matter, frames=40):
        self.name, self.path = name, path
        self.frames = max(2, int(frames))
        self.ornt = nib.orientations.io_orientation(affine)
        self.bg = [p > 0 for p in self._mips(np.asarray(white_matter, np.float32))]
        self.done = []          # (kind, title, projections)
        self._cur = None

    def _mips(self, vol):
        v = nib.orientations.apply_orientation(np.asarray(vol, np.float32), self.ornt)
        return [v.max(0), v.max(1), v.max(2)]      # sagittal (y, z), coronal (x, z), axial (x, y)

    # -------------------------------------------------------------- recording
    def begin(self, kind, title, total=None):
        """A new solve; ``total``: its steps if known (backward), for an even stride."""
        stride = 1 if total is None else max(1, int(np.ceil(total / self.frames)))
        self._cur = dict(kind=kind, title=title, stride=stride, items=[])

    def want(self, step):
        return self._cur is not None and step % self._cur["stride"] == 0

    def add(self, step, vol, note=""):
        c = self._cur
        c["items"].append((step, note, self._mips(vol)))
        if len(c["items"]) > 2 * self.frames:    # forward: too many, keep every other, halve the rate
            c["items"] = c["items"][1::2]
            c["stride"] *= 2

    def end(self):
        c, self._cur = self._cur, None
        if c is None:
            return
        items = c["items"]
        if len(items) > self.frames:
            keep = np.unique(np.linspace(0, len(items) - 1, self.frames).round().astype(int))
            items = [items[i] for i in keep]
        for step, note, p in items:
            self.done.append((c["kind"], f"{c['title']}, step {step}{note}", p))

    def density(self, vol):
        p = self._mips(vol)
        for _ in range(HOLD):
            self.done.append(("density", "the bundle", p))

    # -------------------------------------------------------------- rendering
    def write(self):
        if not self.done:
            return None
        import imageio
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = [b.shape for b in self.bg]
        width = sum(s[0] for s in sizes)
        h = max(s[1] for s in sizes)
        fig = plt.figure(figsize=(9.6, 9.6 * (h + 0.35 * h) / width), dpi=100, facecolor="black")
        axes, x0 = [], 0.0
        for s in sizes:
            ax = fig.add_axes([x0 / width, 0.0, s[0] / width, h / (1.35 * h)])
            ax.set_axis_off()
            axes.append(ax)
            x0 += s[0]
        ims = [ax.imshow(np.zeros(s[::-1] + (3,)), origin="lower", interpolation="nearest") for ax, s in zip(axes, sizes)]
        top = fig.text(0.01, 0.97, "", color="white", fontsize=9, va="top", family="sans-serif")
        sub = fig.text(0.01, 0.86, "", color="0.75", fontsize=8, va="top", family="sans-serif")
        cmaps = {k: plt.get_cmap(v[0]) for k, v in KINDS.items()}
        frames = []
        for kind, title, proj in self.done:
            _, decades, what = KINDS[kind]
            ref = None if decades is None else max(float(p.max()) for p in proj)
            for im, p, bg in zip(ims, proj, self.bg):
                rgb = np.where(bg[..., None], 0.22, 0.0) * np.ones(3)
                if ref is None:                 # absolute: probabilities
                    lo, val = -4.0, np.log10(np.maximum(p, 1e-30))
                else:
                    lo = -decades
                    val = np.log10(np.maximum(p, 1e-30) / ref) if ref > 0 else np.full(p.shape, -np.inf)
                on = val > lo
                t = np.clip((val - lo) / -lo, 0, 1) if ref is None else np.clip((val - lo) / decades, 0, 1)
                rgb = np.where(on[..., None], cmaps[kind](t)[..., :3], rgb)
                im.set_data(np.transpose(rgb, (1, 0, 2)))
            top.set_text(f"{self.name}: {title}")
            sub.set_text(what)
            fig.canvas.draw()
            img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames.append(img[:img.shape[0] // 2 * 2, :img.shape[1] // 2 * 2].copy())   # even sides (yuv420p)
        plt.close(fig)
        with imageio.get_writer(self.path, fps=FPS, format="ffmpeg", codec="libx264", pixelformat="yuv420p",
                                macro_block_size=1,
                                output_params=["-crf", str(CRF), "-preset", "veryfast", "-movflags", "+faststart"]) \
                as writer:
            for f in frames:
                writer.append_data(f)
        log.info("%s: the solves as a video (%d frames, %.0f s): %s", self.name, len(frames), len(frames) / FPS,
                 self.path)
        return self.path


class ChainMovies:
    """The chain's videos (mpat.afq.run): a ``segment_bundles`` on_step callback that records each
    bundle of a group from the group's columns, and ``finish(name, density, path)`` once the
    bundle's maps are written."""

    def __init__(self, chain, affine, white_matter, frames=40):
        self.chain, self.affine, self.wm, self.frames = chain, affine, white_matter, frames
        self.movies = {}
        self.phase, self.last = None, None
        nc = chain.ncomp
        self.nc = nc

    def want(self, it):
        m = next(iter(self.movies.values()), None)
        return m is None or it == 1 or m.want(it)

    def _volume(self, x, cols, mean_classes):
        ch = self.chain
        mass = np.asarray(x[::self.nc, cols], np.float64).sum(1).reshape(ch.t_lin.size, ch.nb)
        v = mass.mean(1) if mean_classes else mass.sum(1)
        return np.maximum(ch.to_volume(v), 0)

    def __call__(self, step):
        if step.names is None:
            return
        new = step.phase != self.phase or (self.last is not None and step.it <= self.last)
        self.phase, self.last = step.phase, step.it
        for i, name in enumerate(step.names):
            m = self.movies.get(name)
            if m is None:
                m = self.movies[name] = SolveMovie(name, None, self.affine, self.wm, self.frames)
            if new:
                m.end()
                m.begin("forward" if step.phase == "forward" else "backward",
                        f"{step.phase} ({'the walkers' if step.phase == 'forward' else 'every end kind'})")
            if step.it == 1 or m.want(step.it):
                cols = slice(int(step.offs[i]), int(step.offs[i]) + int(step.nks[i]))
                if step.phase == "forward":
                    m.add(step.it, self._volume(step.term, cols, False))
                else:
                    m.add(step.it, self._volume(step.total, cols, True))

    def finish(self, name, density, path):
        m = self.movies.pop(name, None)
        if m is None:
            return None
        m.end()
        m.density(density)
        m.path = path
        return m.write()
