---
orphan: true
---

(live)=
# Watching a run: the live view

`mpat --live` serves a web page that draws each bundle in 3-D while it is being solved. A
bundle fills in from its shortest streamlines to its longest, until its solves converge. The
run writes the same files as without it.

## Starting it

```bash
mpat sub-01_dwi.nii.gz sub-01_dwi.bval sub-01_dwi.bvec sub-01_T1w.nii.gz out/ --n-proc 8 --live
```

The log gives the page's address, `http://localhost:8765/` unless that port is taken (the
next free one is used) or you give another (`--live 9000`). Open it in a browser any time
during the run. From Python:

```python
mpat.run(afq, n_proc=8, live=True)          # or live=9000, or a mpat.live.LiveView
```

**On a cluster**, the page is served by the MPAT process, on the machine it runs on, and
only to that machine. Forward the port to yours, then open `http://localhost:8765/` locally:

```bash
ssh -L 8765:localhost:8765 -J login.cluster.edu node123      # through the login node to the compute node
```

The page loads NiiVue from its CDN. If the browser has no internet, start the view yourself
with a local copy of `niivue.umd.js` (npm package `@niivue/niivue`, `dist/`):

```python
from mpat.live import LiveView

view = LiveView(shape, affine, niivue_js="niivue.umd.js")     # the fODF image's grid
mpat.run(afq, n_proc=8, live=view)
```

The server has no password. It listens on `127.0.0.1` unless you pass `host=`; think before
binding it to a public address.

## What the page shows

The density so far is drawn as a volume, with no background image:

- **Color** follows the density on a log scale, from `1e-3` of the bundle's peak so far to
  its peak, in the color map of the MIP collage (magma).
- **Opacity** rises along the same scale as level$^\gamma$. With a steep rise the bundle's
  core is solid and its wide, faint outskirts are a haze that does not hide it. **Core over
  haze** sets $\gamma$ (6 at the start) and **Opacity** the opacity at the peak.

Each frame is relative to the bundle's own peak so far, so an early frame is not dimmer than
a late one, only smaller.

**Bundle** chooses what is drawn: one bundle, or all of them (in each voxel, the brightest).
A check mark means the bundle is solved, an ellipsis that it is being solved now. Bundles
appear as their group's turn comes ({ref}`columns`).

**Solve** reports the step, the remaining walker mass (the larger of the two solves'; they
stop below `tol`) and how far that mass has fallen from 1 to `tol` on a log scale. Where
every walker needs about the same number of steps to be absorbed, the mass stays near 1 for
a while and then falls quickly, so the bar is not a clock.

The page asks for a new frame about twice a second. Redraws are spaced out further when
computing the densities takes long (many bundles in a group), so that they take at most a
fifth of the run's time.

## How it works: the two solves advanced together

MPAT normally finishes the backward solve, starts the forward one from its result, and only
then forms the density ({doc}`algorithm`, sections 7 to 9). Nothing could be drawn for the
first half of that. With the live view, each group's two solves take turns instead
(`mpat.solve.Interleaved`):

$$\beta \leftarrow M r + M Q \beta, \qquad S \leftarrow \pi_0(\beta) + M Q^\top S .$$

- Started from $\beta = M r$ and $S = \pi_0$, each step adds one more term of the power
  series the sequential solver sums, so the backward solve takes exactly the same steps.
- The forward solve's start $\pi_0$ depends on $\beta$: every half-walk's launch is weighted
  by its partner's probability of completing the bundle. It is recomputed from the current
  $\beta$ every 4 steps (`source_every`) and once more when the backward solve has converged.
  The forward solve then runs on alone to the tolerance.
- The density so far, $S \circ \beta$, counts only the streamlines short enough for both of
  their halves to have been walked. It grows to the solution and never passes it. (It only
  ever grows for a bundle without include ROIs on the single-value chain; the alternating
  signs of inclusion–exclusion and the signed values of the moment chain can make it dip a
  little on the way, and it is clipped at 0.)

The results are the sequential solver's to the tolerance, in about as many products with the
transition matrix: on the tests' phantoms, the same number of backward steps and one more
forward step. Each column needs one more array (7 instead of 6), so slightly fewer bundles
share a group for the same `max_mem`. A bundle's summary marks its solves with
`"interleaved": true`.

## From your own code

`segment_bundles` takes the callback directly. It is called before the first step, after
every step, and once more when the group has converged:

```python
from mpat import segment_bundles
from mpat.live import LiveView

with LiveView(chain.shape, affine) as view:              # serves until the block ends
    results = segment_bundles(chain, specs, launches, on_progress=view.update)
```

Your own callback gets the solve: `sol.names` (the group's bundles), `sol.volume(i)` (bundle
`i` so far, on the chain's grid), `sol.expected(i)` (its expected number of streamlines so
far), `sol.steps`, `sol.remaining`, `sol.progress` and `sol.done`.

```{eval-rst}
.. autoclass:: mpat.live.LiveView
   :members: update, finish, close, state, frame

.. autofunction:: mpat.live.colormap

.. autoclass:: mpat.solve.Interleaved
   :members: step, density, volume, expected, results
```
