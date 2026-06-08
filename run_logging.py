"""
run_logging.py — Eğitim loss & verimlilik kaydı (rapor için)
=============================================================
Tezin "resource-efficient" iddiasının sayısal kanıtı buradan gelir:
  • Loss eğrisi      → <output_dir>/metrics.csv  (step, loss[, ...])
  • Verimlilik özeti → ./efficiency_summary.csv  (her run bir satır)
      run, method, image_size, steps, effective_batch,
      trainable_params_M, total_params_M, trainable_pct,
      peak_vram_gb, wall_clock_min
"""

import csv
import os
import time

import torch


def peak_vram_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0


def reset_peak_vram() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def count_params(*modules) -> tuple:
    """(trainable, total) parametre sayısı — verilen modüller üzerinden (tekrarsız)."""
    seen, total, trainable = set(), 0, 0
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
    return trainable, total


class LossLogger:
    """Adım-başına loss CSV kaydı."""

    def __init__(self, path: str, header=("step", "loss")):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._f)
        self._w.writerow(list(header))
        self.path = path

    def log(self, *values):
        self._w.writerow(list(values))
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass


def append_efficiency(record: dict, csv_path: str = "./efficiency_summary.csv") -> None:
    """Verimlilik satırını ./efficiency_summary.csv'ye ekler (başlığı birleştirir)."""
    existing, fieldnames = [], list(record.keys())
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            for k in (reader.fieldnames or []):
                if k not in fieldnames:
                    fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in existing:
            writer.writerow(row)
        writer.writerow(record)


def log_efficiency(output_dir, method, image_size, steps, effective_batch,
                   trainable, total, start_time, csv_path="./efficiency_summary.csv"):
    """Tek çağrıda verimlilik özetini hesaplayıp kaydeder."""
    record = {
        "run": os.path.basename(os.path.normpath(output_dir)),
        "method": method,
        "image_size": image_size,
        "steps": steps,
        "effective_batch": effective_batch,
        "trainable_params_M": round(trainable / 1e6, 3),
        "total_params_M": round(total / 1e6, 1),
        "trainable_pct": round(100.0 * trainable / max(total, 1), 3),
        "peak_vram_gb": round(peak_vram_gb(), 2),
        "wall_clock_min": round((time.time() - start_time) / 60.0, 2),
    }
    append_efficiency(record, csv_path)
    return record
