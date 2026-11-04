"""OFA 004: one-B200 decoder endurance validation (not the autobench quick suite).

Example, paid GPU: modal run experiments/endurance.py --steps 10000 --ring-cap 8192
Optional extra profiling overhead: --profile
Only run after Overlay 004 CPU checks and the existing Modal preflight pass.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "b200-run"))
DATA, MK, V, RES = "/vol/data", "/opt/ofa/llm/b200/megakernel", "/vol", "/vol/results"

app = modal.App("ofa-decoder-endurance")
vol = modal.Volume.from_name("ofa-b200-data", create_if_missing=True)
if modal.is_local():
    import modal_app as upstream

    image = upstream.tc_img.run_commands(
        f"cd {MK} && MODEL=S bash build.sh sm_100a ofa_hr256 -DOFA_GEMM_BACKEND=1 -DGEMM100_STAGES=2 -DDECODE_HR=256"
    )
else:
    image = modal.Image.debian_slim(python_version="3.12")


@app.function(gpu="B200", image=image, volumes={V: vol}, cpu=8.0, memory=32768, timeout=5400,
              max_containers=1, retries=0)
def run_endurance(steps: int, ring_cap: int, profile: bool, hr: int, run_id: str) -> dict:
    if not 1000 <= steps <= 20000 or ring_cap not in (8192, 32768, 1 << 20) or hr not in (128, 256):
        raise ValueError("Invalid endurance parameters")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,50}", run_id):
        raise ValueError("Invalid run ID")
    vol.reload()
    inputs = [f"{DATA}/tinystories_cl100k_train.u32", f"{DATA}/init_flat_S.f32",
              f"{DATA}/starts_4000.u32", f"{DATA}/prompt_ids_fa.txt",
              f"{DATA}/cl100k_vocab.bin", "/opt/ofa/doom-gpu/doom1.wad"]
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:
        raise RuntimeError("Missing prepared inputs: " + ", ".join(missing))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True).strip()
    if "B200" not in gpu:
        raise RuntimeError("Expected B200 GPU, found " + gpu)
    directory = Path(RES) / "ofa_decoder_endurance" / run_id
    directory.mkdir(parents=True, exist_ok=True)
    binary = "ofa_tc2" if hr == 128 else "ofa_hr256"
    command = [f"./{binary}", "--steps", str(steps), "--sched-steps", str(steps),
               "--split", "b200", "--wad", "/opt/ofa/doom-gpu/doom1.wad",
               "--vocab", f"{DATA}/cl100k_vocab.bin",
               "--prompt", f"{DATA}/prompt_ids_fa.txt", "--prompt-text", "What is a megakernel?",
               "--answer-len", "256", "--data", f"{DATA}/tinystories_cl100k_train.u32",
               "--init", f"{DATA}/init_flat_S.f32", "--starts", f"{DATA}/starts_4000.u32",
               "--decode-out-cap", str(ring_cap), "--decode-max-tokens", str(1 << 26),
               "--loss-out", str(directory / "loss.txt"), "--text-out", str(directory / "tokens.txt")]
    if profile:
        command += ["--decode-profile"]
    # One run with a hard timeout, never retry. This is paid B200 time.
    log_file = directory / "run.log"
    t0 = time.monotonic()
    result_code = None
    try:
        with log_file.open("w") as f:
            proc = subprocess.Popen(["timeout", "-k", "10", "1500", *command], cwd=MK,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, errors="replace", bufsize=1)
            for line in proc.stdout:
                f.write(line)
                if line.startswith(("decode progress:", "decode:", "decode answers:", "doom:", "decode safety:", "decode profile:", "CUDA error")):
                    print(line, end="", flush=True)
            result_code = proc.wait()
    finally:
        vol.commit()  # Preserve logs, even after failure
    log = log_file.read_text()
    progress = [(int(s), int(d), int(f)) for s, d, f in re.findall(
        r"decode progress: train_step=(\d+) decoder_tokens=(\d+) doom_frames=(\d+)", log)]
    m = re.search(r"decode: (\d+) tokens generated", log)
    doom = re.search(r"doom: (\d+) frames in the run, ([\d.]+) fps", log)
    doom_train = re.search(r"; ([\d.]+) fps while training steps (\d+)\.\.(\d+)", log)
    loss_lines = (directory / "loss.txt").read_text().strip().splitlines() if (directory / "loss.txt").exists() else []
    token_count = len((directory / "tokens.txt").read_text().split()) if (directory / "tokens.txt").exists() else 0
    # Check *both* decoder and Doom counters advanced late in the same persistent launch.
    late = [(d, f) for s, d, f in progress if s >= steps * 0.75]
    early = [(d, f) for s, d, f in progress if s <= steps * 0.50]
    valid = (result_code == 0 and m is not None and doom is not None and doom_train is not None
             and len(loss_lines) == steps and int(m.group(1)) == token_count and token_count > ring_cap
             and len(progress) >= 4 and bool(late) and bool(early)
             and min(d for d, f in late) > max(d for d, f in early)
             and min(f for d, f in late) > max(f for d, f in early)
             and float(doom_train.group(1)) >= 33.0 and int(doom_train.group(3)) == steps - 1)
    report = {"ok": bool(valid), "returncode": result_code, "gpu": gpu, "steps": steps,
              "hr": hr, "ring_cap": ring_cap, "profile": profile, "decode_tokens": int(m.group(1)) if m else None,
              "tokens_recorded": token_count, "doom_fps": float(doom.group(2)) if doom else None,
              "doom_fps_while_training": float(doom_train.group(1)) if doom_train else None,
              "progress": progress, "duration_seconds": round(time.monotonic() - t0, 2),
              "volume_directory": str(directory)}
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    vol.commit()
    if not valid:
        raise RuntimeError("Endurance check failed, see saved artifacts: " + str(directory) + "\n" + json.dumps(report))
    return report


@app.local_entrypoint()
def main(steps: int = 10000, ring_cap: int = 8192, profile: bool = False,
         hr: int = 256, run_id: str = "decoder004-test") -> None:
    print("PAID: launching a single B200 decoder endurance test. No automatic retries.", flush=True)
    result = run_endurance.remote(steps, ring_cap, profile, hr, run_id)
    print(json.dumps(result, indent=2))
