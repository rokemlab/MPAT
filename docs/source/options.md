# Options

| `mpat.run(...)` | CLI | default | |
|---|---|---|---|
| `n_proc` | `--n-proc` | required | threads |
| `max_mem` | `--max-mem` | required | memory budget, bytes or e.g. `"16GB"`. Bundles are packed into groups that share every pass over the transition matrix; the budget sets how many fit. |
| `moments` | `--moments` | off | spatial first moments: narrower, more accurate bundles; about 5× slower |
| `tol` | `--tol` | `1e-10` | remaining walker mass at which the solves stop |
| `angle` | `--angle` | `35` | maximum turn per one-voxel hop (degrees) |
| `double_odf` | `--single-odf` | doubleODF | `--single-odf` gives first-order hops (see [MITTENS](https://github.com/mattcieslak/MITTENS)) |
| `bundles` | `--bundles` | all in pyAFQ's bundle dictionary | pyAFQ names, e.g. `"Left Arcuate"`; output files drop the spaces (`bundle-LeftArcuate`) |
| `cores` | – | `True` | also write each bundle's core |
| `live` | `--live` | off | draw each bundle in 3-D in a browser while it is solved ({doc}`live`) |

## Choosing them

**`max_mem`.** Each bundle needs one *column* per pair (avoid set, end kind) (see
{ref}`columns` and {ref}`rois`). A column costs about
$12 \times 8$ bytes per state, or 4× that with moments, and a state is a white-matter voxel
times 26 direction classes. Columns that fit together share every product with the
transition matrix, so a larger budget means fewer passes. Packing does not change any
result.

**`tol`.** A typical bundle loses about 10% of its walker mass per step, so $10^{-10}$ takes
300–500 steps.

**`double_odf`.** doubleODF weights each hop by how well the next voxel's fODF supports it
(similar to iFOD2). singleODF follows the current voxel's fODF only, like pyAFQ's first-order
probabilistic tracker. {numref}`fig-transition` shows the difference.

**`moments`.** Without moments, every voxel restarts its paths from a uniform distribution of
positions, which widens bundles that run obliquely to the grid ({numref}`fig-moments`).
