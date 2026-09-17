"""Inference benchmark for the GB10 stand.

Two measurement modes, because they answer different questions:

* ``inference_only`` - the input batch already lives on the target device. This
  is the GPU-bound number usually quoted in papers.
* ``end_to_end`` - JPEG decode and resize run on the CPU, tensors move over PCIe
  to the GPU, then the forward pass runs. This is what a production conveyor
  camera actually costs.

Every configuration is measured ``--repeats`` times; module power is sampled
with NVML during the measured window, so each run carries its own energy figure.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from common import RESULTS, PowerSampler, RunRecord, json_dump, json_load, set_seed
from data_food101 import DATA, load_split
from features import ImageList, build_backbone

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _prepared_batch(backbone: str, size: int, batch_size: int, device: str, dtype, pool: Path) -> torch.Tensor:
    """Build one batch tensor (no augmentation) from real Food-101 images."""
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize(int(size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    from PIL import Image

    files = sorted(p for p in pool.rglob("*.jpg"))[:batch_size]
    tensors = []
    for path in files:
        with Image.open(path) as im:
            tensors.append(tf(im.convert("RGB")))
    batch = torch.stack(tensors)
    if device == "cuda":
        batch = batch.to(device)
        if dtype != torch.float32:
            batch = batch.to(dtype)
    return batch


def run_config(
    backbone: str,
    device: str,
    precision: str,
    batch_size: int,
    mode: str,
    threads: int,
    repeats: int,
    warmup: int,
    min_images: int,
    sampler: PowerSampler,
    test_root: Path,
    idle_power_w: float | None = None,
    min_duration_s: float = 1.5,
) -> list[dict]:
    """Measure one configuration.

    The power sampler runs across the whole configuration, but energy is
    integrated only over each measured window. A load warm-up of at least one
    second precedes the window, otherwise the first NVML samples land on an idle
    module and the energy figure is biased low for short runs.
    """
    rows: list[dict] = []
    dtype = DTYPES[precision]
    if device == "cpu":
        torch.set_num_threads(threads)
    sampler.start()
    for repeat in range(repeats):
        set_seed(repeat)
        model, _, size = build_backbone(backbone)
        model = model.to(device)
        if device == "cuda":
            model = model.to(dtype)
        model.eval()

        batch = None
        loader = None
        if mode == "inference_only":
            batch = _prepared_batch(backbone, size, batch_size, device, dtype, test_root)
        else:

            def make_loader():
                paths, labels, _ = load_split("test", DATA / "food101")
                return torch.utils.data.DataLoader(
                    ImageList(paths, labels, size),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=min(8, max(1, threads // 3)),
                    pin_memory=(device == "cuda"),
                    drop_last=True,
                )

            def cycle(data_loader):
                while True:
                    for item in data_loader:
                        yield item

            loader = cycle(make_loader())

        def one_step():
            if mode == "inference_only":
                with torch.inference_mode():
                    out = model(batch)
                return batch.shape[0], out
            images, _ = next(loader)
            if device == "cuda":
                images = images.to(device, non_blocking=True)
                if dtype != torch.float32:
                    images = images.to(dtype)
            with torch.inference_mode():
                out = model(images)
            return images.shape[0], out

        for _ in range(warmup):
            one_step()
        if device == "cuda":
            torch.cuda.synchronize()
        load_start = time.perf_counter()
        while time.perf_counter() - load_start < 1.0:
            one_step()
        if device == "cuda":
            torch.cuda.synchronize()

        latencies: list[float] = []
        images_done = 0
        start = time.perf_counter()
        while images_done < min_images or (time.perf_counter() - start) < min_duration_s:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            n, _ = one_step()
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)
            images_done += n
        end = time.perf_counter()
        power = sampler.window(start, end)

        record = RunRecord(
            device=device,
            backbone=backbone,
            precision=precision,
            batch_size=batch_size,
            latencies_ms=latencies,
            throughput_ips=images_done / (end - start),
            images=images_done,
            duration_s=end - start,
            power=power,
            notes=f"mode={mode};repeat={repeat};threads={threads}",
        )
        row = record.as_dict(idle_power_w)
        row["mode"] = mode
        row["repeat"] = repeat
        row["threads"] = threads
        row["device_name"] = (
            torch.cuda.get_device_name(0) if device == "cuda" else f"CPU x{threads}"
        )
        rows.append(row)
        print(
            f"  {backbone:14s} {device:4s} {precision:5s} bs={batch_size:2d} {mode:14s} "
            f"thr={row['throughput_img_s']:9.2f} img/s  p50={row['latency_ms_p50']:8.2f} ms  "
            f"{row['power_mean_w']} W  {row['energy_j_per_image']} J/img",
            flush=True,
        )
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    sampler.stop()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbones", nargs="+", default=["resnet50", "convnext_tiny", "vit_b_16"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--gpu-precisions", nargs="+", default=["fp32", "fp16", "bf16"])
    parser.add_argument("--modes", nargs="+", default=["inference_only", "end_to_end"])
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--min-images", type=int, default=128)
    parser.add_argument("--min-duration", type=float, default=1.5, help="seconds per measured window")
    parser.add_argument("--idle-seconds", type=float, default=8.0)
    parser.add_argument("--out", default=str(RESULTS / "bench_raw.json"))
    args = parser.parse_args()

    test_root = DATA / "food101" / "test"
    sampler = PowerSampler()
    if not sampler.available():
        raise SystemExit("NVML power readings are not available on this host")
    idle = sampler.read_idle(args.idle_seconds)
    print(f"idle module power: {idle}", flush=True)

    rows: list[dict] = []
    for mode in args.modes:
        for backbone in args.backbones:
            for batch_size in args.batch_sizes:
                for precision in args.gpu_precisions:
                    rows += run_config(
                        backbone, "cuda", precision, batch_size, mode, args.threads,
                        args.repeats, args.warmup, args.min_images, sampler, test_root,
                        idle["mean_w"],
                        args.min_duration,
                    )
    for backbone in args.backbones:
        for batch_size in args.batch_sizes:
            rows += run_config(
                backbone, "cpu", "fp32", batch_size, "inference_only", args.threads,
                args.repeats, args.warmup, args.min_images, sampler, test_root,
                idle["mean_w"],
                args.min_duration,
            )
    for backbone in args.backbones:
        rows += run_config(
            backbone, "cpu", "fp32", 1, "end_to_end", args.threads,
            args.repeats, args.warmup, args.min_images, sampler, test_root,
            idle["mean_w"],
            args.min_duration,
        )

    json_dump({"idle_power": idle, "rows": rows}, args.out)
    print(f"wrote {args.out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
