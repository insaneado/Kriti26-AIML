"""
Pre-compute a set of real BDH inference runs, one per start/end pair.

The web dashboard is static, so every run it can display has to be computed
ahead of time. This writes one JSON per run plus a manifest the setup page
reads to build its picker, so the picker offers exactly the runs that exist.

    PYTHONPATH=. python scripts/build_runs.py

Floats are rounded on the way out: the attention tensors dominate the payload
and full float64 precision is meaningless for a heatmap.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.export_web_data import export_data  # noqa: E402

# Start/end pairs chosen to spread across the board: corners, edges, centre.
PAIRS = [
    ((0, 0), (9, 9)),
    ((0, 9), (9, 0)),
    ((9, 9), (0, 0)),
    ((9, 0), (0, 9)),
    ((0, 0), (0, 9)),
    ((0, 0), (9, 0)),
    ((4, 4), (9, 9)),
    ((0, 0), (4, 4)),
    ((2, 7), (7, 2)),
    ((5, 0), (5, 9)),
    ((0, 5), (9, 5)),
    ((3, 3), (6, 6)),
]


def round_floats(obj, places=4):
    """Recursively round floats. Cuts the attention payload several-fold."""
    if isinstance(obj, float):
        return round(obj, places)
    if isinstance(obj, list):
        return [round_floats(v, places) for v in obj]
    if isinstance(obj, dict):
        return {k: round_floats(v, places) for k, v in obj.items()}
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="web/data/runs")
    ap.add_argument("--neurons", type=int, default=1000)
    ap.add_argument("--max-attempts", type=int, default=25)
    ap.add_argument("--places", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "_tmp.json"

    manifest = []
    for i, (start, end) in enumerate(PAIRS):
        run_id = f"r{start[0]}{start[1]}-{end[0]}{end[1]}"
        print(f"[{i + 1}/{len(PAIRS)}] {run_id}: start={start} end={end}", flush=True)

        export_data(
            output_path=str(tmp),
            m_neurons=args.neurons,
            start_pos=start,
            end_pos=end,
            max_attempts=args.max_attempts,
        )

        data = json.loads(tmp.read_text(encoding="utf-8"))
        # The absolute path of whoever ran the export is not useful to a reader.
        data["config"].pop("model_path", None)
        data["config"]["start"] = list(start)
        data["config"]["end"] = list(end)

        target = out_dir / f"{run_id}.json"
        target.write_text(
            json.dumps(round_floats(data, args.places), separators=(",", ":")),
            encoding="utf-8",
        )
        size_mb = target.stat().st_size / 1e6

        manifest.append({
            "id": run_id,
            "file": f"runs/{run_id}.json",
            "start": list(start),
            "end": list(end),
            "board_size": data["config"]["board_size"],
            "frames": len(data["frames"]),
            "accuracy_pct": data["config"].get("best_accuracy_pct"),
            "size_mb": round(size_mb, 2),
        })
        print(f"      -> {target.name}  {size_mb:.2f} MB  "
              f"{len(data['frames'])} frames  acc={data['config'].get('best_accuracy_pct')}%",
              flush=True)

    tmp.unlink(missing_ok=True)
    index = {
        "board_size": manifest[0]["board_size"] if manifest else None,
        "runs": manifest,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    total = sum(m["size_mb"] for m in manifest)
    print(f"\nWrote {len(manifest)} runs ({total:.1f} MB total) + index.json")


if __name__ == "__main__":
    main()
