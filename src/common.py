"""Shared helpers: paths, seeds, JSON I/O, GPU power sampling for the GB10 stand.

Measurement provenance
----------------------
Power is read with NVML (`nvidia-ml-py`), which returns the same value as
`nvidia-smi --query-gpu=power.draw` on this platform. On a GB10 (Grace-Blackwell
superchip with unified LPDDR5X) the sensor covers the whole module, not only the
GPU complex, so we report it as *module power* and treat it as a proxy for whole
platform power. See README.md for the limitations section.
"""
from __future__ import annotations

import json
import os
import platform
import random
import socket
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for _p in (DATA, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

BACKBONES = ("resnet50", "convnext_tiny", "vit_b_16")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_dump(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def json_load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q))


def host_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "kernel": platform.release(),
        "os": platform.platform(),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception:  # pragma: no cover - diagnostic only
        info["torchvision"] = None
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        info["gpu_name"] = prop.name
        info["gpu_capability"] = f"{prop.major}.{prop.minor}"
        info["gpu_total_mem_gib"] = round(prop.total_memory / 1024**3, 1)
        info["cuda"] = torch.version.cuda
    return info


def _read_first_line(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def dram_total_gib() -> float | None:
    value = _read_first_line("/proc/meminfo")
    if not value:
        return None
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return round(int(line.split()[1]) / 1024**2, 1)
    return None


class PowerSampler:
    """Poll NVML module power in a background thread and integrate energy.

    NVML reads cost microseconds, so the sampler does not measurably perturb the
    CPU-side benchmarks. Verified against `nvidia-smi --query-gpu=power.draw`.
    """

    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._handle = None

    @staticmethod
    def available() -> bool:
        try:
            import pynvml  # noqa: F401

            return subprocess.run(
                ["nvidia-smi", "-q", "-d", "POWER"],
                capture_output=True,
                text=True,
                timeout=15,
            ).stdout.lower().find("power draw") >= 0
        except Exception:
            return False

    def start(self) -> "PowerSampler":
        import pynvml

        pynvml.nvmlInit()
        self._nvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                mw = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
                self._samples.append((time.perf_counter(), mw / 1000.0))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            if self._nvml:
                self._nvml.nvmlShutdown()
        except Exception:
            pass

    def window(self, t0: float, t1: float) -> dict:
        pts = [(t, w) for t, w in self._samples if t0 <= t <= t1]
        if len(pts) < 2:
            return {"samples": len(pts), "mean_w": None, "max_w": None, "energy_j": None}
        watts = [w for _, w in pts]
        span = pts[-1][0] - pts[0][0]
        # trapezoidal integration of the sampled power over the measured window
        energy = 0.0
        for (ta, wa), (tb, wb) in zip(pts, pts[1:]):
            energy += 0.5 * (wa + wb) * (tb - ta)
        return {
            "samples": len(pts),
            "mean_w": round(statistics.fmean(watts), 3),
            "median_w": round(statistics.median(watts), 3),
            "max_w": round(max(watts), 3),
            "min_w": round(min(watts), 3),
            "span_s": round(span, 4),
            "energy_j": round(energy, 3),
        }

    def read_idle(self, seconds: float = 5.0) -> dict:
        """Sample while the GPU is idle; used as the power baseline."""
        self.start()
        t0 = time.perf_counter()
        time.sleep(seconds)
        t1 = time.perf_counter()
        summary = self.window(t0, t1)
        self.stop()
        return summary


@dataclass
class RunRecord:
    """A single measured (device, backbone, precision, batch) configuration."""

    device: str
    backbone: str
    precision: str
    batch_size: int
    latencies_ms: list[float] = field(default_factory=list)
    throughput_ips: float = 0.0
    images: int = 0
    duration_s: float = 0.0
    power: dict = field(default_factory=dict)
    accuracy_top1: float | None = None
    notes: str = ""

    def as_dict(self, idle_power_w: float | None = None) -> dict:
        lat = self.latencies_ms
        d = {
            "device": self.device,
            "backbone": self.backbone,
            "precision": self.precision,
            "batch_size": self.batch_size,
            "images": self.images,
            "duration_s": round(self.duration_s, 4),
            "throughput_img_s": round(self.throughput_ips, 2),
            "latency_ms_p50": round(percentile(lat, 50), 3) if lat else None,
            "latency_ms_p95": round(percentile(lat, 95), 3) if lat else None,
            "latency_ms_mean": round(statistics.fmean(lat), 3) if lat else None,
            "latency_ms_min": round(min(lat), 3) if lat else None,
            "power_mean_w": self.power.get("mean_w"),
            "energy_j_total": self.power.get("energy_j"),
            "energy_j_per_image": (
                round(self.power["energy_j"] / self.images, 4)
                if self.power.get("energy_j") and self.images
                else None
            ),
            "energy_j_per_image_above_idle": (
                round(
                    max(0.0, (self.power["mean_w"] - idle_power_w)) * self.duration_s / self.images,
                    4,
                )
                if self.power.get("mean_w") is not None and idle_power_w is not None and self.images
                else None
            ),
            "accuracy_top1": self.accuracy_top1,
        }
        return d
