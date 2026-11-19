# Installation

MPAT needs Python 3.10 or newer. Install it from GitHub:

```bash
pip install "mpat-tractography[viewer] @ git+https://github.com/rokemlab/MPAT.git"
```

It depends on [MITTENS](https://github.com/mattcieslak/MITTENS) v2 (`mittens-dmri`), which pip
installs from PyPI as a prebuilt wheel for Linux (x86_64, aarch64), macOS (Intel, Apple Silicon)
and Windows, so no compiler is needed. On other platforms pip builds it from source, which needs a
Rust toolchain ([rustup.rs](https://rustup.rs)).

Extras:

| extra | for |
|---|---|
| `viewer` | `mpat.view` in notebooks ([ipyniivue](https://github.com/niivue/ipyniivue)) |
| `test` | the test suite (`pytest`) |
| `docs` | building this documentation |

The other dependencies are numpy, scipy, nibabel, numba, DIPY, scikit-image and pyAFQ (≥ 3.4).

## Building the documentation

```bash
pip install -e ".[docs,test]"
make -C docs html            # = sphinx-build -b html docs/source docs/build/html
```

The `docs/` folder keeps sources and outputs apart:

| folder | |
|---|---|
| `docs/source/` | the pages (MyST Markdown), `conf.py` and the bibliography |
| `docs/static/` | hand-written assets (CSS) |
| `docs/animations/` | the scripts that render the figures and animations, and the phantom they share |
| `docs/_ext/` | the Sphinx extension that runs those scripts and embeds their output |
| `docs/build/` | everything generated: `html/`, the rendered media (`generated/`) and the phantom's cached chains (`cache/`). Delete it to start over (`make -C docs clean`) |

The figures and animations on {doc}`algorithm` are rendered at build time by the scripts in
`docs/animations/`, which run MPAT on a small phantom. That takes a few minutes the first time.
The results are cached and re-rendered only when a script or MPAT's source changes. Two
environment variables control this:

- `MPAT_DOCS_ANIMATIONS=skip` skips rendering for quick edits to the text.
- `MPAT_DOCS_ANIMATIONS=gif` writes GIFs instead of MP4s.
