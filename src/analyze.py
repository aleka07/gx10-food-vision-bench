"""Aggregate raw measurement files into the tables used by the paper.

Inputs:  results/bench_raw.json, results/probe_seeds/*.json
Outputs: results/summary.csv, results/summary.md, results/summary.json

Each configuration is measured `repeats` times; this script reports the median
across repeats together with the spread, so the article never quotes a single
lucky run.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from common import RESULTS, json_dump

GROUP_KEYS = ("device", "backbone", "precision", "batch_size", "mode")
METRICS = (
    "throughput_img_s",
    "latency_ms_p50",
    "latency_ms_p95",
    "power_mean_w",
    "energy_j_per_image",
    "energy_j_per_image_above_idle",
)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in GROUP_KEYS)].append(row)
    out = []
    for key, group in sorted(groups.items()):
        entry = dict(zip(GROUP_KEYS, key))
        entry["repeats"] = len(group)
        for metric in METRICS:
            values = [g[metric] for g in group if g.get(metric) is not None]
            if not values:
                continue
            entry[metric] = round(statistics.median(values), 4)
            entry[f"{metric}_min"] = round(min(values), 4)
            entry[f"{metric}_max"] = round(max(values), 4)
            entry[f"{metric}_spread_pct"] = (
                round(100.0 * (max(values) - min(values)) / statistics.median(values), 2)
                if statistics.median(values)
                else None
            )
        out.append(entry)
    return out


def probe_table(probe_dir: Path) -> dict:
    per_model: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(probe_dir.glob("*_s*.json")):
        backbone = path.stem.rsplit("_s", 1)[0]
        payload = json.loads(path.read_text(encoding="utf-8"))
        for point in payload["points"]:
            per_model[backbone][point["per_class"]].append(point["top1"])
    table = {}
    for backbone, points in per_model.items():
        table[backbone] = {
            str(per_class): {
                "n_seeds": len(values),
                "top1_mean": round(statistics.fmean(values), 4),
                "top1_std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            }
            for per_class, values in sorted(points.items())
        }
    return table


def to_csv(rows: list[dict], path: Path) -> None:
    import csv

    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown(rows: list[dict], probes: dict, idle: dict) -> str:
    lines = [f"Idle module power: {idle['mean_w']} W (n={idle['samples']} samples)", ""]
    lines.append("## GPU (FP16), inference only")
    lines.append("")
    lines.append("| model | batch | img/s | p50 ms | p95 ms | W | J/img |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in rows:
        if row["device"] == "cuda" and row["precision"] == "fp16" and row["mode"] == "inference_only":
            lines.append(
                f"| {row['backbone']} | {row['batch_size']} | {row['throughput_img_s']:.0f} | "
                f"{row['latency_ms_p50']:.2f} | {row['latency_ms_p95']:.2f} | {row['power_mean_w']:.1f} | "
                f"{row['energy_j_per_image']:.4f} |"
            )
    lines += ["", "## GPU (FP16), end to end", "", "| model | batch | img/s | p50 ms | p95 ms | W | J/img |",
              "|---|---|---|---|---|---|---|"]
    for row in rows:
        if row["device"] == "cuda" and row["precision"] == "fp16" and row["mode"] == "end_to_end":
            lines.append(
                f"| {row['backbone']} | {row['batch_size']} | {row['throughput_img_s']:.0f} | "
                f"{row['latency_ms_p50']:.2f} | {row['latency_ms_p95']:.2f} | {row['power_mean_w']:.1f} | "
                f"{row['energy_j_per_image']:.4f} |"
            )
    lines += ["", "## Precision comparison (batch 32, end to end)", "",
              "| model | precision | img/s | p50 ms | W | J/img |", "|---|---|---|---|---|---|"]
    for row in rows:
        if row["device"] == "cuda" and row["batch_size"] == 32 and row["mode"] == "end_to_end":
            lines.append(
                f"| {row['backbone']} | {row['precision']} | {row['throughput_img_s']:.0f} | "
                f"{row['latency_ms_p50']:.2f} | {row['power_mean_w']:.1f} | {row['energy_j_per_image']:.4f} |"
            )
    lines += ["", "## CPU ARM (FP32, 20 threads), inference only", "",
              "| model | batch | img/s | p50 ms | W | J/img |", "|---|---|---|---|---|---|"]
    for row in rows:
        if row["device"] == "cpu" and row["mode"] == "inference_only":
            lines.append(
                f"| {row['backbone']} | {row['batch_size']} | {row['throughput_img_s']:.1f} | "
                f"{row['latency_ms_p50']:.1f} | {row['power_mean_w']:.1f} | {row['energy_j_per_image']:.4f} |"
            )
    lines += ["", "## Linear probe (mean over 3 seeds)", "",
              "| model | imgs/class | top-1 % | std |", "|---|---|---|---|"]
    for backbone, points in probes.items():
        for per_class, payload in points.items():
            lines.append(
                f"| {backbone} | {per_class} | {payload['top1_mean'] * 100:.1f} | "
                f"{payload['top1_std'] * 100:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", default=str(RESULTS / "bench_raw.json"))
    parser.add_argument("--probe-dir", default=str(RESULTS / "probe_seeds"))
    args = parser.parse_args()

    payload = json.loads(Path(args.bench).read_text(encoding="utf-8"))
    rows = aggregate(payload["rows"])
    probes = probe_table(Path(args.probe_dir))
    summary = {"idle_power": payload["idle_power"], "rows": rows, "probe": probes}
    json_dump(summary, RESULTS / "summary.json")
    to_csv(rows, RESULTS / "summary.csv")
    (RESULTS / "summary.md").write_text(markdown(rows, probes, payload["idle_power"]), encoding="utf-8")
    print(markdown(rows, probes, payload["idle_power"]))


if __name__ == "__main__":
    main()
