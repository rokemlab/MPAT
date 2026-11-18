"""A live view of a run: the bundles in 3-D as they are solved, in a web browser.

    mpat dwi.nii.gz dwi.bval dwi.bvec t1.nii.gz out --n-proc 8 --live      # then open the URL it logs
    mpat.run(afq, n_proc=8, live=True)

``LiveView`` is a small web server (standard library only) with one page (``live.html``,
NiiVue).  ``mpat.run(live=...)`` hands its ``update`` to ``segment_bundles`` as
``on_progress``, which advances each group's backward and forward solves together
(``mpat.solve.Interleaved``): the page then draws each bundle's density so far and redraws it
as the solves advance, so a bundle is seen filling in from its shortest streamlines to its
longest.

The density is drawn as a volume: colour on a log scale from ``floor`` x the bundle's peak
(so far) to its peak, in the colour map of the MIP collage, and opacity rising steeply along
the same scale, so the bundle's core shows, solid, through the faint haze around it.

On a cluster, forward the port and open the page locally::

    ssh -L 8765:localhost:8765 <login node>        # (add a hop to the compute node if needed)

The server listens on this machine only (``host="127.0.0.1"``) unless told otherwise; it has
no password, so think before binding it to a public address.
"""
from __future__ import annotations

import gzip
import json
import logging
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

import nibabel as nib
import numpy as np

from .viewer import NIIVUE_CDN, log_density_u8

log = logging.getLogger("mpat")

ALL = "*"            # the name of the composite of every bundle
NODES = 17           # colour-map nodes NiiVue interpolates between
HAZE = 3             # of 255: the least opacity of a drawn level (NiiVue skips opacity < 0.01) ...
HAZE_FROM = (0.1, 0.25)   # ... reached over these levels: below them the colours are near black


def colormap(name="magma", alpha=1.0, gamma=6.0):
    """A NiiVue colour map (R, G, B, A, I node lists): matplotlib's ``name``, with the opacity
    of each voxel a ray crosses rising from 0 at the bottom of the scale to ``alpha`` at the
    top as level ** ``gamma``.  A steep rise (large ``gamma``) keeps the wide, faint outskirts
    of a bundle from hiding its core; they stay visible as a haze (``Amin``: the least opacity
    of each level, 3 / 255 from a quarter of the scale up, none at the near-black bottom).
    Level 0 -- below the floor -- is fully transparent."""
    import matplotlib

    t = np.linspace(0.0, 1.0, NODES)
    rgb = np.rint(255 * np.asarray(matplotlib.colormaps[name](t))[:, :3]).astype(int)
    amin = np.rint(HAZE * np.clip((t - HAZE_FROM[0]) / (HAZE_FROM[1] - HAZE_FROM[0]), 0, 1)).astype(int)
    a = np.maximum(np.rint(255 * alpha * t ** gamma).astype(int), amin)
    a[0] = 0
    return dict(R=rgb[:, 0].tolist(), G=rgb[:, 1].tolist(), B=rgb[:, 2].tolist(), A=a.tolist(),
                I=np.rint(255 * t).astype(int).tolist(), Amin=amin.tolist())


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):     # a page that went away is not an error
        log.debug("live view: request from %s failed", client_address, exc_info=True)


class LiveView:
    """Serve the bundles being solved to a browser page.

    shape, affine  the chain's grid (``chain.shape``, the fODF image's affine)
    port           first port tried (the next 20 are tried if it is taken); 0: any free port
    host           address to listen on (default: this machine only)
    floor          faintest density drawn, as a fraction of each bundle's peak so far
    interval       seconds between redraws, at least
    share          at most this fraction of the run's time goes to redraws (they are spaced
                   out when computing the densities of many bundles takes long)
    cmap, alpha, gamma  see ``colormap``
    niivue_js      a local ``niivue.umd.js`` to serve instead of loading NiiVue from its CDN
                   (for machines whose browser has no internet)

    ``update`` is ``segment_bundles``' ``on_progress``; ``finish`` marks the run as over;
    ``close`` stops the server.  ``url`` is the page's address.
    """

    def __init__(self, shape, affine, port=8765, host="127.0.0.1", floor=1e-3, interval=0.5, share=0.2,
                 title="", cmap="magma", alpha=1.0, gamma=6.0, niivue_js=None):
        self.shape, self.affine = tuple(int(s) for s in shape), np.asarray(affine, float)
        self.floor, self.interval, self.share = float(floor), float(interval), float(share)
        self.title, self.niivue_js = title, niivue_js
        self.cmap = colormap(cmap, alpha, gamma)
        self.cmap_name, self.alpha, self.gamma = cmap, float(alpha), float(gamma)
        self._lock = threading.Lock()
        self._order = []            # every bundle seen, in order
        self._meta = {}             # name -> dict(peak, expected, done)
        self._live = {}             # name -> uint8 volume (the group being solved)
        self._done = {}             # name -> zlib-compressed uint8 volume (finished bundles)
        self._all_done = np.zeros(self.shape, np.uint8)     # composite of the finished bundles
        self._group = dict(step=0, remaining=None, progress=0.0, back_done=False, done=False, names=[])
        self._version, self._finished = 0, False
        self._sol, self._next, self._t0 = None, 0.0, time.time()
        self._server = self._serve(host, int(port))
        self.port = self._server.server_address[1]
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0', '') else host}:{self.port}/"
        self._thread = threading.Thread(target=self._server.serve_forever, name="mpat-live", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ the solver's side
    def update(self, sol):
        """``on_progress``: publish the group's bundles so far (at most every ``interval`` s)."""
        new = sol is not self._sol
        if not (new or sol.done or time.monotonic() >= self._next):
            return
        t0 = time.perf_counter()
        frames = {}
        for i, name in enumerate(sol.names):
            d = sol.volume(i)
            frames[name] = (log_density_u8(d, self.floor), float(d.max()), sol.expected(i))
        with self._lock:
            if new:
                self._sol = sol
                self._order += [n for n in sol.names if n not in self._order]
            for name, (u8, peak, expected) in frames.items():
                self._meta[name] = dict(peak=peak, expected=expected, done=bool(sol.done))
                if sol.done:
                    self._live.pop(name, None)
                    self._done[name] = zlib.compress(u8.tobytes(order="F"), 1)
                    np.maximum(self._all_done, u8, out=self._all_done)
                else:
                    self._live[name] = u8
            r = sol.remaining
            self._group = dict(step=int(sol.steps), remaining=float(r) if np.isfinite(r) else None,
                               progress=float(sol.progress), back_done=bool(sol.back_done),
                               done=bool(sol.done), names=list(sol.names))
            self._version += 1
        cost = time.perf_counter() - t0
        self._next = time.monotonic() + max(self.interval, cost / max(self.share, 1e-3))

    def finish(self):
        """The run is over: the page says so and stops asking."""
        with self._lock:
            self._finished = True
            self._version += 1

    def close(self, linger=1.5):
        """Stop the server, ``linger`` seconds after the last update so open pages get it."""
        if self._server is None:
            return
        if linger:
            time.sleep(linger)
        self._server.shutdown()
        self._server.server_close()
        self._server = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()
        self.close(0)

    # ------------------------------------------------------------------ what the page asks for
    def state(self):
        with self._lock:
            g = self._group
            return dict(version=self._version, finished=self._finished, title=self.title, floor=self.floor,
                        shape=list(self.shape), elapsed=round(time.time() - self._t0, 1), **g,
                        bundles=[dict(name=n, live=n in self._live, **self._meta[n]) for n in self._order])

    def frame(self, name=ALL):
        """The uint8 volume drawn for ``name`` (a bundle, or ``ALL``: the brightest of every
        bundle in each voxel): 0 below the floor, 1..255 the log scale up to the peak."""
        with self._lock:
            if name == ALL:
                out = self._all_done.copy()
                for u8 in self._live.values():
                    np.maximum(out, u8, out=out)
                return out
            if name in self._live:
                return self._live[name]
            if name in self._done:
                return np.frombuffer(zlib.decompress(self._done[name]), np.uint8).reshape(self.shape, order="F")
        return np.zeros(self.shape, np.uint8)

    def nifti(self, name=ALL):
        """``frame`` as a gzipped NIfTI file on the run's grid."""
        img = nib.Nifti1Image(self.frame(name), self.affine)
        img.header.set_xyzt_units("mm")
        img.header["cal_min"], img.header["cal_max"] = 0, 255
        return gzip.compress(img.to_bytes(), 1)

    def page(self):
        from importlib.resources import files as resources

        lib = "niivue.umd.js" if self.niivue_js else NIIVUE_CDN
        page = (resources("mpat") / "live.html").read_text()
        return (page.replace("__TITLE__", (self.title or "MPAT live").replace("<", "&lt;"))
                .replace("__NIIVUE__", f'<script src="{lib}"></script>')
                .replace("__CONFIG__", json.dumps(dict(
                    cmap=self.cmap, cmap_name=self.cmap_name, alpha=self.alpha, gamma=self.gamma,
                    all=ALL, floor=self.floor)).replace("</", "<\\/")))

    # ------------------------------------------------------------------ the server
    def _serve(self, host, port):
        view = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):       # quiet: the run's log is the terminal's
                pass

            def _send(self, body, ctype, encoding=None, status=200):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if encoding:
                    self.send_header("Content-Encoding", encoding)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                u = urlparse(self.path)
                parts = [unquote(p) for p in u.path.split("/") if p]
                route = parts[-1] if parts else ""
                name = parse_qs(u.query).get("name", [ALL])[0]
                if route == "density.nii.gz" and len(parts) >= 3 and parts[-3] == "vol":
                    name = parts[-2]            # .../vol/<bundle>/density.nii.gz
                try:
                    if route in ("", "index.html"):
                        self._send(view.page().encode(), "text/html; charset=utf-8")
                    elif route == "state":
                        self._send(json.dumps(view.state()).encode(), "application/json")
                    elif route == "frame":
                        self._send(gzip.compress(view.frame(name).tobytes(order="F"), 1),
                                   "application/octet-stream", "gzip")
                    elif route == "density.nii.gz":
                        self._send(view.nifti(name), "application/gzip")
                    elif route == "niivue.umd.js" and view.niivue_js:
                        with open(view.niivue_js, "rb") as fh:
                            self._send(fh.read(), "text/javascript; charset=utf-8")
                    else:
                        self._send(b"not found", "text/plain", status=404)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        last = None
        for p in ([0] if port == 0 else range(port, port + 21)):
            try:
                server = _Server((host, p), Handler)
            except OSError as err:
                last = err
                continue
            if p != port:
                log.info("live view: port %d is taken, using %d", port, p)
            return server
        raise OSError(f"live view: no free port in {port}..{port + 20}: {last}")
