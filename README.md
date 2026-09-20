# BDH Visualizer

Animation-first visual analytics for **Baby Dragon Hatchling (BDH)**, a neural
pathfinding model. An 11-tab dashboard that replays inference step by step:
which neurons fire, how attention moves across layers, and how Hebbian weights
evolve as the model solves a grid.

**[Open the live demo](https://insaneado.github.io/Kriti26-AIML/)** ·
[Demo video](https://youtu.be/-gRUDOUw85g)

Built for **Kriti 2026**, the inter-hostel technology competition at IIT
Guwahati (AI/ML problem statement, Path A). This is a team project - see
[Team](#team) below.

---

## What it shows

The Python side pre-computes **12 real BDH inference runs** - one per start/end
pair - plus the Hebbian weight history, and exports them as static JSON. The
browser replays whichever run you pick, so there is no server, no inference at
runtime and no hosting cost.

`setup.html` offers exactly the runs that exist: cells with a pre-computed run
are selectable, everything else is greyed out. Picking a pair loads that run's
file (~1.5 MB) and nothing else.

> **On the board size.** The checkpoint in `model/boardpath.pt` is trained on a
> **10x10** board, so that is what the dashboard shows. An earlier build shipped
> 16x16 data produced by a model that was never committed here, which meant the
> repository could not regenerate its own data. Everything now comes from the
> checkpoint in this repository.

| Tab | What it shows |
|---|---|
| Graph | Force-directed neural activation graph over the top-20 neurons |
| Live Analysis | Per-frame telemetry, top-20 neurons, activation histogram |
| 3D Walkthrough | Three.js neural field with orbit controls |
| Sparse Brain | 10x10 activation heatmap with the solved path overlaid |
| Topology | Community detection and hub analysis over the neuron graph |
| Attention Atlas | Layer-wise attention heatmaps, animated across steps |
| Concept Probe | Functional phase assignment per neuron |
| Hebbian | Multi-board weight evolution animation |
| Memory Formation | Dual-stream x/y activation history charts |
| Scaling Lab | Interactive O(T) vs O(T^2) cost comparison |
| Inference | Step-by-step replay of the BDH compute cycle |

---

## Repository layout

```
core/            BDH model and analysis code
├── bdh.py       the model itself
├── analysis.py  activation / topology analysis
└── runtime.py
model/
└── boardpath.pt trained checkpoint
scripts/
├── build_runs.py           all 12 runs    -> web/data/runs/*.json + index.json
├── export_web_data.py      one run        -> a single JSON
└── export_hebbian_data.py  weight history -> web/data/hebbian_data.json
pages/           Streamlit multipage app (the exploratory version)
app.py           Streamlit entry point
web/             the static dashboard that GitHub Pages serves
├── setup.html   start/end coordinate picker - the entry point
├── index.html   the 11-tab dashboard
├── css/ js/     styling and panel logic
└── data/
    ├── runs/    one JSON per pre-computed run, plus index.json (the manifest)
    └── hebbian_data.json
index.html       root redirect into web/setup.html
```

---

## Run locally

Serve `web/` over HTTP - opening the files directly with `file://` breaks the
`fetch()` calls that load the JSON.

```bash
git clone https://github.com/insaneado/Kriti26-AIML.git
cd Kriti26-AIML/web
python -m http.server 8080
```

Then open <http://localhost:8080/setup.html>.

To regenerate the data from the checkpoint instead of using the committed JSON:

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/build_runs.py           # all 12 runs + the manifest
PYTHONPATH=. python scripts/export_hebbian_data.py  # weight history
```

`PYTHONPATH=.` is required - the scripts import `core`, and running them as
plain paths leaves the repository root off `sys.path`.

Edit `PAIRS` in `scripts/build_runs.py` to change which start/end pairs are
generated; the manifest and the picker follow automatically.

The Streamlit version of the same analysis:

```bash
streamlit run app.py
```

---

## Deployment

The dashboard is fully static, so GitHub Pages serves it directly from the
default branch with no build step:

> Settings -> Pages -> Source: *Deploy from a branch* -> Branch `master`, folder `/ (root)`

The root `index.html` redirects to `web/setup.html`, so the bare Pages URL is
the demo link.

**Note on load time.** Each run is ~1.5 MB and only the selected one is
fetched, so the dashboard becomes interactive quickly. An earlier build shipped
a single 40 MB dataset; splitting it per run and rounding the attention tensors
to four decimals brought the per-load cost down by roughly 27x.

---

## Team

Kriti 2026 team project. This repository is a fork of
[RsbhThakur/Kriti26-AIML](https://github.com/RsbhThakur/Kriti26-AIML), the
team's submission repository, which also hosts the original demo at
<https://rsbhthakur.github.io/Kriti26-AIML/web/setup.html>.

---

## References

- BDH paper: <https://arxiv.org/abs/2509.26507>
- Transformer Explainer: <https://poloclub.github.io/transformer-explainer/>
- Pathway: <https://pathway.com/>
