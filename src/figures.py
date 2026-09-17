"""Build the article figures from the raw measurement files."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, RESULTS, json_load  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 8,
        "font.family": "DejaVu Sans",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 200,
    }
)

LABELS = {"resnet50": "ResNet-50", "convnext_tiny": "ConvNeXt-T", "vit_b_16": "ViT-B/16"}
MARKERS = {"resnet50": "o", "convnext_tiny": "s", "vit_b_16": "^"}


def _rows(path: Path) -> list[dict]:
    return json_load(path)["rows"]


def _median(values):
    return statistics.median(values) if values else float("nan")


def figure_throughput(rows: list[dict], out: Path) -> None:
    gpu = defaultdict(list)
    cpu = defaultdict(list)
    for row in rows:
        if row["mode"] != "inference_only" or row["precision"] != "fp16" and row["device"] == "cuda":
            continue
        key = (row["backbone"], row["batch_size"])
        if row["device"] == "cuda" and row["precision"] == "fp16":
            gpu[key].append(row["throughput_img_s"])
        elif row["device"] == "cpu":
            cpu[key].append(row["throughput_img_s"])

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, data, title in ((axes[0], gpu, "GPU GB10 (FP16)"), (axes[1], cpu, "CPU ARM (FP32, 20 threads)")):
        for backbone in sorted({k[0] for k in data}):
            xs = sorted({k[1] for k in data if k[0] == backbone})
            ys = [_median(data[(backbone, bs)]) for bs in xs]
            ax.plot(xs, ys, marker=MARKERS[backbone], label=LABELS[backbone], linewidth=1.4)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks([1, 8, 32])
        ax.set_xticklabels(["1", "8", "32"])
        ax.set_xlabel("размер батча")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("пропускная способность, изображений/с")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_energy(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for backbone in sorted({r["backbone"] for r in rows}):
        for mode, style in (("inference_only", "-"), ("end_to_end", "--")):
            pts = [
                r
                for r in rows
                if r["backbone"] == backbone
                and r["device"] == "cuda"
                and r["precision"] == "fp16"
                and r["mode"] == mode
                and r["energy_j_per_image"] is not None
            ]
            if not pts:
                continue
            grouped = defaultdict(list)
            for r in pts:
                grouped[r["batch_size"]].append(r)
            xs, ys = [], []
            for bs in sorted(grouped):
                xs.append(_median([r["throughput_img_s"] for r in grouped[bs]]))
                ys.append(_median([r["energy_j_per_image"] for r in grouped[bs]]))
            ax.plot(xs, ys, marker=MARKERS[backbone], linestyle=style, linewidth=1.3,
                    label=f"{LABELS[backbone]} ({'e2e' if mode == 'end_to_end' else 'только вывод'})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("пропускная способность, изображений/с")
    ax.set_ylabel("энергия на изображение, Дж")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def figure_probe(probe_results: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.7, 2.35))
    for backbone, payload in probe_results.items():
        xs = [p["per_class"] for p in payload["points"]]
        ys = [p["top1"] * 100 for p in payload["points"]]
        ax.plot(xs, ys, marker=MARKERS.get(backbone, "o"), linewidth=1.4, label=LABELS.get(backbone, backbone))
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 25, 50])
    ax.set_xticklabels(["5", "10", "25", "50"])
    ax.set_xlabel("изображений на класс в обучающей выборке")
    ax.set_ylabel("top-1, %")
    ax.set_ylim(20, 66)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default=str(RESULTS / "bench_raw.json"))
    parser.add_argument("--probe-dir", default=str(RESULTS / "probe"))
    args = parser.parse_args()

    rows = _rows(Path(args.bench))
    probe_results = {}
    for path in sorted(Path(args.probe_dir).glob("*_probe.json")):
        probe_results[path.stem.replace("_probe", "")] = json.loads(path.read_text(encoding="utf-8"))

    figure_throughput(rows, FIGURES / "fig_throughput.png")
    figure_energy(rows, FIGURES / "fig_energy.png")
    if probe_results:
        figure_probe(probe_results, FIGURES / "fig_probe.png")
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    main()
