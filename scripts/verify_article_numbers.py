#!/usr/bin/env python3
"""Recompute every measurement-derived number in the paper from the raw data.

The script reads article_text.py itself, so it stays valid after edits: table
cells are recomputed from results/bench_raw.json, probe accuracy and the
25→50 gains from results/probe_seeds/*.json, and the derived prose claims
(speed-up range, energy ratios, Wh per million images, camera streams) from the
same raw file. It exits nonzero on the first mismatch and writes a full audit to
results/article_number_audit.json.

This exists because the first draft carried hand-typed values that were wrong:
497 and 1371 img/s where the medians are 496,47 and 1370,45, 12,6 Вт·ч where the
best configuration gives 12,33, a "9–17 раз" energy range whose true minimum is
8,7, and a "5–7 п.п." data-scaling claim that matched none of the three models.
"""
from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results/bench_raw.json"
STAND = ROOT / "results/stand_info.json"
PROBES = ROOT / "results/probe_seeds"
OUT = ROOT / "results/article_number_audit.json"
MODELS = {"ConvNeXt-Tiny": "convnext_tiny", "ResNet-50": "resnet50", "ViT-B/16": "vit_b_16"}

raw = json.loads(RAW.read_text(encoding="utf-8"))
stand = json.loads(STAND.read_text(encoding="utf-8"))
rows = raw["rows"]
groups: dict[tuple, list[dict]] = defaultdict(list)
for row in rows:
    groups[(row["device"], row["backbone"], row["precision"], row["batch_size"], row["mode"])].append(row)

checks: list[dict] = []
failures: list[dict] = []


def add(claim: str, recomputed: str, ok: bool, comment: str = "") -> None:
    entry = {
        "article_claim": claim,
        "recomputed": recomputed,
        "verdict": "совпадает" if ok else "не совпадает",
        "comment": comment,
    }
    checks.append(entry)
    if not ok:
        failures.append(entry)


def med(device: str, model: str, precision: str, batch: int, mode: str, field: str) -> float:
    rs = groups[(device, model, precision, batch, mode)]
    assert len(rs) == 3, f"expected 3 repeats, got {len(rs)} for {(device, model, precision, batch, mode)}"
    return statistics.median(r[field] for r in rs)


def load_article():
    path = ROOT / "article/article_text.py"
    spec = importlib.util.spec_from_file_location("article_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def comma(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


article = load_article()
text = " ".join(block["text"] for block in article.blocks if block["kind"] == "text")
table_rows = [row for block in article.blocks if block["kind"] == "table" for row in block["rows"]]

# --- protocol facts encoded in the raw file ------------------------------------
add("каждая конфигурация повторялась трижды", str(sorted({len(v) for v in groups.values()})), {len(v) for v in groups.values()} == {3})
add("батчи 1, 8 и 32", str(sorted({r["batch_size"] for r in rows})), {r["batch_size"] for r in rows} == {1, 8, 32})
add("потребление в простое — 4,9 Вт", f"mean={raw['idle_power']['mean_w']:.3f} Вт", round(raw["idle_power"]["mean_w"], 1) == 4.9)

# --- table cells --------------------------------------------------------------
for row in table_rows:
    label, batch = row[0], int(row[1])
    model = MODELS[label]
    inference = med("cuda", model, "fp16", batch, "inference_only", "throughput_img_s")
    e2e_thr = med("cuda", model, "fp16", batch, "end_to_end", "throughput_img_s")
    e2e_p95 = med("cuda", model, "fp16", batch, "end_to_end", "latency_ms_p95")
    e2e_energy = med("cuda", model, "fp16", batch, "end_to_end", "energy_j_per_image")
    for column, expected, actual in (
        ("только вывод, изобр/с", str(half_up(inference)), row[2]),
        ("сквозной режим, изобр/с", str(half_up(e2e_thr)), row[3]),
        ("p95 сквозной, мс", comma(e2e_p95, 1), row[4]),
        ("Дж/изобр", comma(e2e_energy, 3), row[5]),
    ):
        add(f"Таблица, {label}, батч {batch}, {column} = {actual}", f"{expected} ({expected.replace(',', '.')})", expected == actual)

# --- prose claims -------------------------------------------------------------
for label, model, batch in (("ConvNeXt-Tiny", "convnext_tiny", 32), ("ResNet-50", "resnet50", 32), ("ViT-B/16", "vit_b_16", 32)):
    value = half_up(med("cuda", model, "fp16", batch, "inference_only", "throughput_img_s"))
    add(f"«{label}» на батче 32 в режиме «только вывод» = {value} изобр/с", str(value), str(value) in text)

for label, model in (("ConvNeXt-Tiny", "convnext_tiny"), ("ViT-B/16", "vit_b_16")):
    thr = half_up(med("cuda", model, "fp16", 8, "end_to_end", "throughput_img_s"))
    energy = comma(med("cuda", model, "fp16", 8, "end_to_end", "energy_j_per_image"), 3)
    add(f"«{label}» на батче 8 сквозным режимом = {thr} изобр/с и {energy} Дж", f"{thr} изобр/с, {energy} Дж", str(thr) in text and energy in text)

overheads = [100 * (med("cuda", m, "fp16", b, "inference_only", "throughput_img_s") - med("cuda", m, "fp16", b, "end_to_end", "throughput_img_s")) / med("cuda", m, "fp16", b, "inference_only", "throughput_img_s") for m in MODELS.values() for b in (1, 8, 32)]
add(f"расхождение режимов достигает {half_up(max(overheads))} %", f"max={max(overheads):.2f} %", f"достигает {half_up(max(overheads))} %" in text)

for label, model, batch in (("ConvNeXt-Tiny", "convnext_tiny", 32), ("ViT-B/16", "vit_b_16", 32), ("ConvNeXt-Tiny", "convnext_tiny", 1), ("ViT-B/16", "vit_b_16", 1)):
    value = comma(med("cpu", model, "fp32", batch, "inference_only", "throughput_img_s"), 1)
    add(f"CPU, {label}, батч {batch} = {value} изобр/с", value, value in text)

speedups, energy_ratios = [], []
for model in MODELS.values():
    for batch in (1, 8, 32):
        speedups.append(med("cuda", model, "fp16", batch, "inference_only", "throughput_img_s") / med("cpu", model, "fp32", batch, "inference_only", "throughput_img_s"))
        energy_ratios.append(med("cpu", model, "fp32", batch, "inference_only", "energy_j_per_image") / med("cuda", model, "fp16", batch, "inference_only", "energy_j_per_image"))
speedup_claim = f"от {half_up(min(speedups))} до {half_up(max(speedups))} раз"
energy_claim = f"в {comma(min(energy_ratios), 1)}–{comma(max(energy_ratios), 1)} раза"
add(f"ускорение на GPU {speedup_claim}", f"{min(speedups):.2f}–{max(speedups):.2f}×", speedup_claim in text)
add(f"энергия ниже {energy_claim}", f"{min(energy_ratios):.2f}–{max(energy_ratios):.2f}×", energy_claim in text)

fp_speed, fp_energy = [], []
for model in MODELS.values():
    fp_speed.append(med("cuda", model, "fp16", 32, "end_to_end", "throughput_img_s") / med("cuda", model, "fp32", 32, "end_to_end", "throughput_img_s"))
    fp_energy.append(med("cuda", model, "fp32", 32, "end_to_end", "energy_j_per_image") / med("cuda", model, "fp16", 32, "end_to_end", "energy_j_per_image"))
fp_speed_claim = f"в {comma(min(fp_speed), 1)}–{comma(max(fp_speed), 1)} раза"
fp_energy_claim = f"в {comma(min(fp_energy), 1)}–{comma(max(fp_energy), 1)} раза"
add(f"FP16 против FP32 по скорости: {fp_speed_claim}", f"{min(fp_speed):.2f}–{max(fp_speed):.2f}×", fp_speed_claim in text and "ускоряет вывод" in text)
add(f"FP16 против FP32 по энергии: {fp_energy_claim}", f"{min(fp_energy):.2f}–{max(fp_energy):.2f}×", fp_energy_claim in text)

best_gpu = min(med("cuda", m, "fp16", b, "end_to_end", "energy_j_per_image") for m in MODELS.values() for b in (1, 8, 32))
cpu_best = med("cpu", "convnext_tiny", "fp32", 32, "inference_only", "energy_j_per_image")
wh_gpu = comma(best_gpu * 1e6 / 3600, 1)
wh_cpu = f"{cpu_best * 1e6 / 3600:.0f}"
add(f"лучшая конфигурация — около {wh_gpu} Вт·ч на миллион изображений", f"{best_gpu * 1e6 / 3600:.2f} Вт·ч", f"около {wh_gpu} Вт·ч" in text)
add(f"процессорная конфигурация — около {wh_cpu} Вт·ч на миллион изображений", f"{cpu_best * 1e6 / 3600:.2f} Вт·ч", f"около {wh_cpu} Вт·ч" in text)
ratio = cpu_best / best_gpu
add(f"«почти в {half_up(ratio)} раз экономичнее»", f"{ratio:.2f}×", f"почти в {half_up(ratio)} раз" in text)

streams = half_up(med("cuda", "convnext_tiny", "fp16", 8, "end_to_end", "throughput_img_s") / 30)
add(f"эквивалентны примерно {streams} потокам по 30 кадров/с", f"{med('cuda', 'convnext_tiny', 'fp16', 8, 'end_to_end', 'throughput_img_s') / 30:.2f}", f"примерно {streams} потокам" in text)

# --- probe --------------------------------------------------------------------
probe_means: dict[tuple[str, int], float] = {}
for model in MODELS.values():
    docs = [json.loads((PROBES / f"{model}_s{seed}.json").read_text(encoding="utf-8")) for seed in (42, 43, 44)]
    for per_class in (5, 10, 25, 50):
        probe_means[(model, per_class)] = statistics.fmean(next(p["top1"] for p in doc["points"] if p["per_class"] == per_class) for doc in docs) * 100
for label, model, per_class in (("ViT-B/16", "vit_b_16", 50), ("ConvNeXt-Tiny", "convnext_tiny", 50), ("ResNet-50", "resnet50", 50), ("ViT-B/16", "vit_b_16", 25), ("ConvNeXt-Tiny", "convnext_tiny", 25), ("ResNet-50", "resnet50", 25)):
    value = comma(probe_means[(model, per_class)], 1)
    add(f"зонд, {label}, {per_class} изображений/класс = {value} %", value, value in text)
for label, model in MODELS.items():
    gain = comma((probe_means[(model, 50)] - probe_means[(model, 25)]), 1)
    add(f"{label}: удвоение выборки добавляет {gain} п.п.", f"{probe_means[(model, 50)] - probe_means[(model, 25)]:.2f} п.п.", f"{gain} п.п." in text)
resnet = [probe_means[("resnet50", n)] for n in (5, 10)]
add("на 5–10 изображениях ResNet-50 растёт немонотонно", f"{resnet[0]:.1f} % → {resnet[1]:.1f} %", resnet[1] < resnet[0] and "немонотонно" in text)

# --- stand facts present in stand_info.json -----------------------------------
for claim, actual, expected in (
    ("20 ядер", stand["cpu_count"], 20),
    ("121,6 ГБ памяти", stand["gpu_total_mem_gib"], 121.6),
    ("CUDA 13.0", stand["cuda"], "13.0"),
    ("PyTorch 2.14.0", stand["torch"].split("+")[0], "2.14.0"),
    ("torchvision 0.29.0", stand["torchvision"].split("+")[0], "0.29.0"),
):
    add(claim, str(actual), actual == expected, "подтверждено stand_info.json")

unverifiable = [
    "10× Cortex-X925 и 10× Cortex-A725: модели ядер отсутствуют в stand_info.json, источник — документация NVIDIA/ASUS.",
    "6144 CUDA-ядра: нет в stand_info.json и bench_raw.json, источник — документация производителя.",
    "до 1 ПФЛОП FP4 и TDP 140 Вт: паспортные характеристики, в измерениях не проверялись.",
    "Ubuntu 24.04.4: stand_info.json хранит строку платформы, но не версию дистрибутива (проверено отдельно: /etc/os-release на стенде).",
    "Food-101: 101 000 фотографий и лицензия CC BY 4.0 — внешние свойства набора данных.",
    "Разрешение 224×224, прогрев ≥ 1 с, окно ≥ 2 с, интервал NVML 20 мс: заданы кодом (src/features.py, src/bench.py, src/common.py), а не записаны в JSON.",
    "«BF16 практически повторяет FP16»: без числового допуска строгая проверка невозможна.",
    "Библиографические данные (годы, тома, DOI, даты обращения) проверялись по Crossref/arXiv/документации, а не по файлам измерений.",
]

report = {
    "files_checked": [str(ROOT / "article/article_text.py"), str(RAW), str(STAND)] + [str(PROBES / f"{m}_s{s}.json") for m in MODELS.values() for s in (42, 43, 44)],
    "checks": checks,
    "verified_true": sum(c["verdict"] == "совпадает" for c in checks),
    "mismatches": failures,
    "unverifiable": unverifiable,
}
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"checks: {len(checks)}, совпадает: {report['verified_true']}, не совпадает: {len(failures)}")
for item in failures:
    print(f"  MISMATCH {item['article_claim']} -> {item['recomputed']}")
sys.exit(1 if failures else 0)
