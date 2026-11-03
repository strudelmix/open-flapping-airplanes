"""B200-only, sequential benchmark of minimal OFA tuning parameters.

One-time data setup (upstream): modal run b200-run/modal_app.py --step prep
Then: modal run experiments/autobench.py --suite quick
Inspect offline: python3 experiments/plan.py

IMPORTANT: --suite quick RENTS A GPU. Results are NOT asserted until they run.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "b200-run"))
import modal  # installed by `pip install modal`
import modal_app as upstream
from experiments.analyze import aggregate, loss_file, parse_log, read_tokens, validate_loss, validate_tokens, write_results, write_charts
from experiments.plan import plan

app = modal.App("ofa-autobench")
# Reuse Jerry's already defined CUDA+DOOM image, including all pinned dependencies.
# These extra builds happen in Modal image construction, NOT on a rented B200.
image = upstream.tc_img.run_commands(
    f"cd {upstream.MK} && MODEL=S bash build.sh sm_100a ofa_hr64 -DOFA_GEMM_BACKEND=1 -DGEMM100_STAGES=2 -DDECODE_HR=64",
    f"cd {upstream.MK} && MODEL=S bash build.sh sm_100a ofa_hr256 -DOFA_GEMM_BACKEND=1 -DGEMM100_STAGES=2 -DDECODE_HR=256",
)
BINARIES = {64: "ofa_hr64", 128: "ofa_tc2", 256: "ofa_hr256"}
SUITE_VERSION = "ofa001-v1"


def invoke(cmd: list[str], cwd: str, timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors="replace", timeout=timeout, check=False)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired as exc:
        data = exc.stdout or b""
        if isinstance(data, bytes):
            data = data.decode("utf-8", "replace")
        return 124, data + f"\nTIMEOUT after {timeout} s\n"


def build_command(hr: int, steps: int, doom: int, output: Path, *, greedy: bool = False) -> list[str]:
    """Fixed dataset, seed, prompt, optimizer settings; Doom allocation changes only one dimension."""
    cmd = [f"./{BINARIES[hr]}", "--steps", str(steps), "--data", f"{upstream.DATA}/tinystories_cl100k_train.u32",
           "--init", f"{upstream.DATA}/init_flat_S.f32", "--starts", f"{upstream.DATA}/starts_4000.u32",
           "--prompt", f"{upstream.DATA}/prompt_ids_fa.txt", "--answer-len", "256", "--seed", "1234",
           "--infer-sms", "24", "--doom-sms", str(doom), "--train-sms", str(124 - doom),
           "--loss-out", str(output / "loss.txt"), "--text-out", str(output / "tokens.txt")]
    if doom:
        cmd += ["--wad", "/opt/ofa/doom-gpu/doom1.wad"]
    else:
        cmd += ["--no-doom"]
    if greedy:
        cmd += ["--no-opt", "--temp", "0"]
    return cmd


@app.function(gpu="B200", image=image, volumes={upstream.V: upstream.vol},
              cpu=8.0, memory=32768, timeout=5400)
def run_suite(steps: int = 600, repeats: int = 2, confirm_steps: int = 4000,
              explore_one_sm: bool = False, run_id: str = SUITE_VERSION) -> dict:
    """One B200; sequential runs; resume by complete per-run records on the Volume."""
    import statistics
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", run_id):
        raise ValueError("run_id must be alphanumeric with optional -/_")
    if not 20 <= steps <= 4000 or not 2 <= repeats <= 5 or confirm_steps < steps or confirm_steps > 10000:
        raise ValueError("invalid steps, repeats or confirmation steps")
    upstream.vol.reload()
    required = [f"{upstream.DATA}/tinystories_cl100k_train.u32", f"{upstream.DATA}/init_flat_S.f32",
                f"{upstream.DATA}/starts_4000.u32", f"{upstream.DATA}/prompt_ids_fa.txt",
                "/opt/ofa/doom-gpu/doom1.wad"]
    missing = [x for x in required if not os.path.isfile(x)]
    if missing:
        raise RuntimeError("Missing benchmark inputs: " + ", ".join(missing) +
                           ". First run: modal run b200-run/modal_app.py --step prep")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "B200" not in gpu:
        raise RuntimeError("Benchmark expects B200, got " + gpu)
    root = Path(upstream.RES) / "ofa_autobench" / run_id
    root.mkdir(parents=True, exist_ok=True)
    binary_sha = {h: hashlib.sha256((Path(upstream.MK) / b).read_bytes()).hexdigest()
                  for h, b in BINARIES.items()}
    records: list[dict] = []
    comments = [f"GPU: {gpu}; suite {SUITE_VERSION}; steps={steps}, repeats={repeats}",
                "All variants built with tcgen05 GEMM100_STAGES=2; HR128 is Jerry's unchanged ofa_tc2 build.",
                "Fixed model S, data, weights, prompt, seed; runs sequentially on one B200.",
                "Quick runs may be noisy. Full confirmation compares baseline and finalist on this GPU."]

    def one(tag: str, *, hr: int, doom: int, nsteps: int, greedy: bool = False) -> dict:
        folder = root / tag
        folder.mkdir(parents=True, exist_ok=True)
        record_path = folder / "record.json"
        command = build_command(hr, nsteps, doom, folder, greedy=greedy)
        signature = {"command": command, "sha256": binary_sha[hr], "version": SUITE_VERSION}
        if record_path.exists():
            try:
                old = json.loads(record_path.read_text())
                if old.get("signature") == signature and (folder / "run.log").exists():
                    return old
            except (OSError, ValueError):
                pass
        start = time.monotonic()
        rc, log = invoke(command, upstream.MK, 240 if greedy else (900 if nsteps <= 4000 else 1800))
        (folder / "run.log").write_text(log)
        metrics = parse_log(log)
        # A completed run without metrics is a failure, NOT a zero-throughput measurement.
        basic = rc == 0 and metrics["train_tokens_s"] is not None and metrics["sampled_tokens_s"] is not None
        if not greedy:
            basic = basic and metrics["doom_fps"] is not None and metrics["doom_frames"] is not None
        try:
            losses = loss_file(folder / "loss.txt")
            basic = basic and len(losses) == nsteps
        except (OSError, ValueError) as e:
            basic = False
            losses = []
            metrics["loss_error"] = str(e)
        record = {"tag": tag, "config": f"HR{hr}/Doom{doom}", "hr": hr, "doom_sms": doom, "steps": nsteps,
                  "greedy": greedy, "returncode": rc, "elapsed_wall_seconds": round(time.monotonic() - start, 3),
                  "ok": bool(basic), "loss_final": losses[-1] if losses else None,
                  "signature": signature, **metrics}
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        upstream.vol.commit()
        return record

    # Frozen-weight greedy validation: compare only the first 128 common tokens.
    reference = one("check_hr128", hr=128, doom=0, nsteps=100, greedy=True)
    if not reference["ok"]:
        raise RuntimeError("Baseline correctness run failed; inspect " + str(root / "check_hr128"))
    ref_tokens = read_tokens(root / "check_hr128" / "tokens.txt")
    ref_loss = loss_file(root / "check_hr128" / "loss.txt")
    valid_hr = {128}
    for h in (64, 256):
        check = one(f"check_hr{h}", hr=h, doom=0, nsteps=100, greedy=True)
        ok_token = check["ok"] and validate_tokens(ref_tokens, read_tokens(root / f"check_hr{h}" / "tokens.txt"))
        ok_loss, loss_err = validate_loss(ref_loss, loss_file(root / f"check_hr{h}" / "loss.txt")) if check["ok"] else (False, math.inf)
        if ok_token and ok_loss:
            valid_hr.add(h)
        comments.append(f"HR{h} frozen correctness: tokens={'pass' if ok_token else 'FAIL'}, loss={'pass' if ok_loss else 'FAIL'} (max abs {loss_err:.6g})")
    records = []
    baseline_loss = None
    for rep in range(1, repeats + 1):
        for h in (128, 64, 256):
            if h not in valid_hr:
                continue
            r = one(f"quick_hr{h}_r{rep}", hr=h, doom=8, nsteps=steps)
            if r["ok"] and h == 128 and baseline_loss is None:
                baseline_loss = loss_file(root / f"quick_hr{h}_r{rep}" / "loss.txt")
            if r["ok"] and baseline_loss is not None:
                ok, error = validate_loss(baseline_loss, loss_file(root / f"quick_hr{h}_r{rep}" / "loss.txt"))
                r = {**r, "ok": bool(ok), "max_loss_error": error}
            records.append(r)
        upstream.vol.commit()
    base_runs = [r for r in records if r["config"] == "HR128/Doom8" and r["ok"]]
    if len(base_runs) != repeats:
        raise RuntimeError("Repeated baseline failed; inspect B200 logs in " + str(root))
    baseline = {"train_tokens_s": statistics.median(r["train_tokens_s"] for r in base_runs),
                "sampled_tokens_s": statistics.median(r["sampled_tokens_s"] for r in base_runs)}
    rank = aggregate(records, baseline)
    eligible = [r for r in rank if r["valid"]]
    best_hr = int(re.search(r"HR(\d+)", eligible[0]["config"]).group(1)) if eligible else 128
    comments.append(f"Best eligible decoder head after quick tests: HR{best_hr}. (No score is a guaranteed speedup.)")

    for doom in (4, 2) + ((1,) if explore_one_sm else ()):
        for rep in range(1, repeats + 1):
            r = one(f"sweep_hr{best_hr}_d{doom}_r{rep}", hr=best_hr, doom=doom, nsteps=steps)
            if r["ok"] and baseline_loss is not None:
                ok, error = validate_loss(baseline_loss, loss_file(root / f"sweep_hr{best_hr}_d{doom}_r{rep}" / "loss.txt"))
                r = {**r, "ok": bool(ok), "max_loss_error": error}
            records.append(r)
    rank = aggregate(records, baseline)
    candidates = [r for r in rank if r["valid"]]
    winner = candidates[0]["config"] if candidates else "HR128/Doom8"
    comments.append(f"Highest-scoring eligible quick-run configuration: {winner}")

    # Full-length confirmation: same GPU, same steps, compare baseline and finalist.
    confirm = [one("confirm_base", hr=128, doom=8, nsteps=confirm_steps)]
    if winner != "HR128/Doom8":
        h, d = map(int, re.fullmatch(r"HR(\d+)/Doom(\d+)", winner).groups())
        confirm.append(one("confirm_candidate", hr=h, doom=d, nsteps=confirm_steps))
        if confirm[0]["ok"] and confirm[1]["ok"]:
            a = loss_file(root / "confirm_base" / "loss.txt")
            b = loss_file(root / "confirm_candidate" / "loss.txt")
            same_loss, err = validate_loss(a, b)
            comments.append(f"Full-run loss comparison: {'pass' if same_loss else 'FAIL'}; max abs deviation {err:.6g}")
            if not same_loss:
                confirm[1] = {**confirm[1], "ok": False}
    comments.append("Full confirmations are single measurements, not evidence of statistical significance.")
    for r in confirm:
        records.append({**r, "config": "CONFIRM " + r["config"]})
    rank = aggregate([r for r in records if not r["config"].startswith("CONFIRM ")], baseline)
    report = write_results(root, records, rank, comments)
    write_charts(root, rank)
    upstream.vol.commit()
    return {"run_id": run_id, "gpu": gpu, "baseline": baseline, "quick_winner": winner,
            "confirm": [{k: r.get(k) for k in ("config", "ok", "train_tokens_s", "sampled_tokens_s", "doom_fps", "loss_final")} for r in confirm],
            "report": report, "volume_results": str(root)}


@app.local_entrypoint()
def main(suite: str = "quick", steps: int = 600, repeats: int = 2, confirm_steps: int = 4000,
         explore_one_sm: bool = False, run_id: str = SUITE_VERSION):
    if suite == "dry-run":
        for entry in plan(steps, repeats, confirm_steps, explore_one_sm):
            print(*entry, sep=" | ")
        print("No GPU job submitted.")
        return
    if suite != "quick":
        raise ValueError("--suite must be quick or dry-run")
    result = run_suite.remote(steps=steps, repeats=repeats, confirm_steps=confirm_steps,
                              explore_one_sm=explore_one_sm, run_id=run_id)
    dest = ROOT / "experiments" / "results" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "benchmark.md").write_text(result["report"])
    (dest / "summary.json").write_text(json.dumps({k: v for k, v in result.items() if k != "report"}, indent=2) + "\n")
    print(result["report"])
    print("Full raw logs and CSV remain in Modal Volume:", result["volume_results"])
    print("Local report:", dest / "benchmark.md")
