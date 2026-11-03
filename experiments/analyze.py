"""Parse OFA megakernel logs and choose a conservative winner (stdlib only)."""
from __future__ import annotations

import csv
import json
import math
import re
import statistics
from pathlib import Path


def parse_log(text: str) -> dict:
    """Missing metrics stay None. No synthetic/default performance values."""
    def num(pattern: str, converter=float):
        match = re.search(pattern, text, re.MULTILINE)
        return converter(match.group(1).replace(',', '')) if match else None
    return {
        "gpu": next((line.strip() for line in text.splitlines() if line.startswith("GPU ")), None),
        "train_tokens_s": num(r"^steady state:.*?->\s*([\d,]+)\s+tok/s;"),
        "train_ms_per_step": num(r"^steady state:\s*([\d.]+)\s+ms/step"),
        "decode_tokens_s": num(r"^decode:\s*\d+\s+tokens generated.*?->\s*([\d,]+)\s+tok/s"),
        "sampled_tokens_s": num(r"^decode answers:.*?->\s*([\d,]+)\s+tok/s sampled"),
        "doom_fps": num(r"^doom:.*?;\s*([\d.]+)\s+fps while training"),
        "doom_frames": num(r"^doom:\s*(\d+)\s+frames", int),
        "kernel_seconds": num(r"^kernel time\s+([\d.]+)\s+s \(event\)"),
        "sm_partition": next((line for line in text.splitlines() if line.startswith("SM partition")), None),
    }


def loss_file(path: Path) -> list[float]:
    if not path.is_file():
        raise ValueError(f"missing loss file: {path}")
    values: list[float] = []
    for expected_step, line in enumerate(path.read_text().splitlines()):
        parts = line.strip().split()
        if len(parts) != 2 or int(parts[0]) != expected_step:
            raise ValueError(f"nonconsecutive loss at {path}:{expected_step + 1}")
        value = float(parts[1])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite loss at step {expected_step}")
        values.append(value)
    if not values:
        raise ValueError(f"empty loss file: {path}")
    return values


def validate_loss(reference: list[float], candidate: list[float], atol: float = 0.001) -> tuple[bool, float]:
    if len(reference) != len(candidate) or not reference:
        return False, math.inf
    error = max(abs(a - b) for a, b in zip(reference, candidate))
    return error <= atol, error


def read_tokens(path: Path) -> list[int]:
    if not path.is_file():
        raise ValueError(f"missing token file: {path}")
    return [int(t) for t in path.read_text().split()]


def validate_tokens(reference: list[int], candidate: list[int], compare_n: int = 128) -> bool:
    """Compare only shared token positions; each run may generate a different total count."""
    return len(reference) >= compare_n and len(candidate) >= compare_n and reference[:compare_n] == candidate[:compare_n]


def aggregate(records: list[dict], baseline: dict | None = None, max_train_slowdown: float = 0.03,
              min_doom_fps: float = 34.0, objective: str = "balanced") -> list[dict]:
    """Group successful repeats, reject incomplete runs, and rank valid candidates."""
    grouped: dict[str, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["config"], []).append(row)
    result = []
    for name, reps in sorted(grouped.items()):
        complete = [r for r in reps if r.get("ok") and all(
            isinstance(r.get(k), (int, float)) and math.isfinite(r[k]) and r[k] > 0
            for k in ("train_tokens_s", "sampled_tokens_s", "doom_fps"))]
        med = {k: statistics.median(r[k] for r in complete) if complete else None
               for k in ("train_tokens_s", "sampled_tokens_s", "doom_fps")}
        valid = len(complete) == len(reps) and len(reps) >= 2 and med["doom_fps"] is not None and med["doom_fps"] >= min_doom_fps
        if baseline and med["train_tokens_s"] is not None:
            valid &= med["train_tokens_s"] >= baseline["train_tokens_s"] * (1 - max_train_slowdown)
        score = None
        if valid and baseline:
            tr = med["train_tokens_s"] / baseline["train_tokens_s"]
            de = med["sampled_tokens_s"] / baseline["sampled_tokens_s"]
            score = {"balanced": math.sqrt(tr * de), "training": tr, "inference": de}[objective]
        result.append({"config": name, "repeats": len(reps), "valid": bool(valid), **med, "score": score})
    return sorted(result, key=lambda r: (not r["valid"], -(r["score"] or 0), r["config"]))


def write_results(directory: Path, records: list[dict], ranked: list[dict], comments: list[str]) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "raw.json").write_text(json.dumps(records, indent=2) + "\n")
    if records:
        fields = sorted({key for row in records for key in row})
        with (directory / "results.csv").open("w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    lines = ["# OFA AutoBench report", "", "Results below are *measured* only when an actual B200 run completed.", "",
             "| Config | Valid? | Training tok/s | Sampled tok/s | Doom FPS | Repeats |", "|---|---|---:|---:|---:|---:|"]
    for row in ranked:
        f = lambda x: f"{x:.1f}" if isinstance(x, (int, float)) else "N/A"
        lines.append(f"| {row['config']} | {'yes' if row['valid'] else 'no'} | {f(row['train_tokens_s'])} | {f(row['sampled_tokens_s'])} | {f(row['doom_fps'])} | {row['repeats']} |")
    lines += ["", "## Notes", ""] + [f"- {c}" for c in comments]
    report = "\n".join(lines) + "\n"
    (directory / "benchmark.md").write_text(report)
    return report


def write_charts(directory: Path, ranked: list[dict]) -> None:
    """Dependency-free static SVG comparisons; no fabricated values for invalid runs."""
    from xml.sax.saxutils import escape
    directory.mkdir(parents=True, exist_ok=True)
    rows = [r for r in ranked if r["valid"]]
    for field, label, name in (("train_tokens_s", "Training throughput (tokens/sec)", "training.svg"),
                               ("sampled_tokens_s", "Sampled-token throughput (tokens/sec)", "inference.svg")):
        width, height = 860, max(155, 90 + 55 * len(rows))
        largest = max((r[field] for r in rows), default=1) or 1
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
               '<rect width="100%" height="100%" fill="#fafafa"/>',
               f'<text x="20" y="31" font-family="Arial" font-size="19" fill="#1b2532">{escape(label)}</text>']
        if not rows:
            svg.append('<text x="20" y="80" font-family="Arial" font-size="14" fill="#555">No validated B200 measurements</text>')
        for i, r in enumerate(rows):
            y = 60 + i * 55
            svg.append(f'<text x="20" y="{y+17}" font-family="Arial" font-size="13" fill="#333">{escape(r["config"])}</text>')
            svg.append(f'<rect x="205" y="{y}" width="{round(540*r[field]/largest)}" height="23" rx="3" fill="#327bb4"/>')
            svg.append(f'<text x="{215+round(540*r[field]/largest)}" y="{y+17}" font-family="Arial" font-size="13" fill="#333">{r[field]:,.0f}</text>')
        svg.append('</svg>')
        (directory / name).write_text('\n'.join(svg) + '\n')
