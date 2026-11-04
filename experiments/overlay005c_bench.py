"""Overlay 005 C: compile-only preflight and one-allocation B200 A/B/C benchmark.

Use after installing Candidate C in a fresh worktree based on Candidate A.
  modal run experiments/overlay005c_bench.py --mode build
  modal run experiments/overlay005c_bench.py --mode all --steps 4000 --run-id overlay005c-ab1

Paid GPU modes: verify, compare, all. No retries; all variants are run
sequentially in one GPU container. 'all' does frozen-weight deterministic
prefix verification BEFORE the 4,000-step shared training/decoder/Doom test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'b200-run'))
MK, DATA, RES, V = '/opt/ofa/llm/b200/megakernel', '/vol/data', '/vol/results', '/vol'
app = modal.App('ofa-overlay005c-head512')
vol = modal.Volume.from_name('ofa-b200-data', create_if_missing=True)

# Matched factorial design. A is the original fused-argmax 256-row kernel,
# compiled from the unchanged Candidate A fast path in decode_wide.cuh.
VARIANTS = {
    'a256': {'hr': 256, 'sample_only': 0, 'write_logits': 1},
    'tile512': {'hr': 512, 'sample_only': 0, 'write_logits': 1},
    'fast256': {'hr': 256, 'sample_only': 1, 'write_logits': 0},
    'fast512': {'hr': 512, 'sample_only': 1, 'write_logits': 0},
}
BUILD_ORDER = list(VARIANTS)

if modal.is_local():
    import modal_app as upstream
    commands = [
        (f'cd {MK} && MODEL=S bash build.sh sm_100a ofa_c_{name} '
         f'-DOFA_GEMM_BACKEND=1 -DGEMM100_STAGES=2 '
         f'-DDECODE_HR={args["hr"]} '
         f'-DDECODE_HEAD_SAMPLE_ONLY={args["sample_only"]} '
         f'-DDECODE_HEAD_WRITE_LOGITS={args["write_logits"]}')
        for name, args in VARIANTS.items()
    ]
    image = upstream.mk_img.run_commands(*commands)
else:
    image = modal.Image.debian_slim(python_version='3.12')


@app.function(image=image, cpu=1.0, memory=1024, timeout=120, retries=0)
def check_build():
    for name in BUILD_ORDER:
        f = Path(MK) / f'ofa_c_{name}'
        if not f.is_file() or f.stat().st_size < 100000:
            raise RuntimeError(f'Missing compiled CUDA binary {f}')
        print(f'CUDA BUILD PASS: {name}, {f.stat().st_size} bytes', flush=True)
    return {'cuda_binaries': BUILD_ORDER, 'gpu_used': False}


def run_variant(kind: str, name: str, steps: int, base: Path, snapshot_token: int) -> dict:
    """Run one binary once. Keep all artifacts after failure. No retries."""
    cfg = VARIANTS[name]
    directory = base / kind / name
    directory.mkdir(parents=True, exist_ok=True)
    binary = f'./ofa_c_{name}'
    cmd = [binary, '--steps', str(steps), '--sched-steps', str(steps),
           '--split', 'b200', '--wad', '/opt/ofa/doom-gpu/doom1.wad',
           '--vocab', f'{DATA}/cl100k_vocab.bin',
           '--prompt', f'{DATA}/prompt_ids_fa.txt', '--prompt-text', 'What is a megakernel?',
           '--answer-len', '256', '--data', f'{DATA}/tinystories_cl100k_train.u32',
           '--init', f'{DATA}/init_flat_S.f32', '--starts', f'{DATA}/starts_4000.u32',
           '--decode-out-cap', '8192', '--decode-max-tokens', str(1 << 26),
           '--decode-prof-token', str(snapshot_token),
           '--loss-out', str(directory / 'loss.txt'),
           '--text-out', str(directory / 'tokens.txt')]
    if kind == 'verify':
        cmd.append('--no-opt')  # fixed weights: tokens MUST match across variants
    log_path = directory / 'run.log'
    started = time.monotonic()
    print(f'>>> {kind}/{name}, {steps} steps, paid B200', flush=True)
    rc = None
    try:
        with log_path.open('w') as log_file:
            process = subprocess.Popen(
                ['timeout', '-k', '10', '750', *cmd], cwd=MK,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, errors='replace', bufsize=1)
            for line in process.stdout:
                log_file.write(line)
                if line.startswith(('steady state:', 'decode:', 'doom:', 'CUDA error', 'decode phase snapshot:')):
                    print(f'[{kind}/{name}] {line.rstrip()}', flush=True)
            rc = process.wait()
    finally:
        vol.commit()
    log = log_path.read_text()
    train = re.search(r'steady state:.*?->\s*([\d.]+) tok/s', log)
    decode = re.search(r'^decode:\s*(\d+) tokens generated.*?->\s*([\d.]+) tok/s', log, re.M)
    doom = re.search(r'^doom:.*?;\s*([\d.]+) fps while training steps 3\.\.(\d+)', log, re.M)
    snapshot = re.search(r'^decode phase snapshot: token=(\d+) whole_us=([\d.]+) spans_us=([\d.]+) gaps_us=([\d.]+)', log, re.M)
    phases = []
    for p, k, tasks, span, gap in re.findall(
        r'^decode phase wall: ph=(\d+) kind=(\w+) tasks=(\d+) span_us=([\d.]+) gap_us=([-\d.]+)', log, re.M):
        phases.append({'phase': int(p), 'kind': k, 'tasks': int(tasks),
                       'span_us': float(span), 'gap_us': float(gap)})
    token_ids = (directory / 'tokens.txt').read_text().split() if (directory / 'tokens.txt').exists() else []
    loss = []
    if (directory / 'loss.txt').exists():
        for line in (directory / 'loss.txt').read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                loss.append(float(fields[1]))
    progress = [(int(s), int(d), int(f)) for s,d,f in re.findall(
        r'decode progress: train_step=(\d+) decoder_tokens=(\d+) doom_frames=(\d+)', log)]
    progressed = (len(progress) >= 4 and progress[-1][0] == steps
                  and progress[-1][1] > progress[0][1]
                  and progress[-1][2] > progress[0][2])
    valid = bool(rc == 0 and train and decode and doom and snapshot
                 and len(phases) == 3 * 12 + 2
                 and int(doom.group(2)) == steps - 1
                 and float(doom.group(1)) >= 33.0
                 and len(loss) == steps
                 and len(token_ids) == int(decode.group(1))
                 and len(token_ids) > 8192
                 and progressed and all(math.isfinite(x) for x in loss))
    result = {
        'variant': name, 'kind': kind, 'hr': cfg['hr'],
        'sample_only': bool(cfg['sample_only']), 'write_logits': bool(cfg['write_logits']),
        'steps': steps, 'ok': valid, 'returncode': rc,
        'decoder_tok_s': float(decode.group(2)) if decode else None,
        'training_tok_s': float(train.group(1)) if train else None,
        'doom_fps_training': float(doom.group(1)) if doom else None,
        'decode_tokens': len(token_ids), 'loss_first': loss[0] if loss else None,
        'loss_last': loss[-1] if loss else None,
        'snapshot': {'token': int(snapshot.group(1)), 'whole_us': float(snapshot.group(2)),
                     'spans_us': float(snapshot.group(3)), 'gaps_us': float(snapshot.group(4))} if snapshot else None,
        'phase_wall': phases,
        'seconds': round(time.monotonic() - started, 2),
        'volume_directory': str(directory),
    }
    (directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    vol.commit()
    if not valid:
        raise RuntimeError(f'{kind}/{name} FAILED, artifacts at {directory}, result={result}')
    return result


def read_tokens(directory: Path):
    return (directory / 'tokens.txt').read_text().split()


def read_loss(directory: Path):
    return [float(line.split()[1]) for line in (directory / 'loss.txt').read_text().splitlines()]


@app.function(gpu='B200', image=image, volumes={V: vol}, cpu=8.0, memory=32768,
              timeout=3600, max_containers=1, retries=0)
def run_suite(mode: str, steps: int, run_id: str) -> dict:
    import os
    if mode not in ('verify', 'compare', 'all') or not 1000 <= steps <= 10000:
        raise ValueError('Invalid mode or steps; choose verify, compare, all; 1000..10000')
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,50}', run_id):
        raise ValueError('run_id must be alphanumeric, -, _, <=51 chars')
    vol.reload()
    required = [f'{DATA}/tinystories_cl100k_train.u32', f'{DATA}/init_flat_S.f32',
                f'{DATA}/starts_4000.u32', f'{DATA}/prompt_ids_fa.txt',
                f'{DATA}/cl100k_vocab.bin', '/opt/ofa/doom-gpu/doom1.wad']
    missing = [p for p in required if not os.path.isfile(p)]
    if missing:
        raise RuntimeError(f'Missing prepared inputs: {missing}')
    gpu = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
    if 'B200' not in gpu:
        raise RuntimeError(f'Expected B200, found {gpu}')
    base = Path(RES) / 'ofa_overlay005c' / run_id
    if base.exists() and list(base.glob('**/run.log')):
        raise RuntimeError(f'Results already exist: {base}; use a new run_id')
    base.mkdir(parents=True, exist_ok=True)
    results: dict = {'gpu': gpu, 'mode': mode, 'run_id': run_id, 'variants': VARIANTS,
                     'verify': [], 'compare': [], 'decision': None}
    try:
        if mode in ('verify', 'all'):
            # Fixed weights, deterministic tokenizer, same temperature and seed.
            for name in BUILD_ORDER:
                results['verify'].append(run_variant('verify', name, 1000, base, snapshot_token=2048))
            ref = read_tokens(base / 'verify' / 'a256')
            prefix = min(8192, len(ref), *(len(read_tokens(base / 'verify' / name)) for name in BUILD_ORDER[1:]))
            if prefix < 8192:
                raise RuntimeError(f'Not enough fixed-weight tokens for equality test: {prefix}')
            for name in BUILD_ORDER[1:]:
                actual = read_tokens(base / 'verify' / name)
                mismatch = next((i for i in range(prefix) if actual[i] != ref[i]), None)
                if mismatch is not None:
                    raise RuntimeError(f'Frozen-weight mismatch: {name} token {mismatch}: {actual[mismatch]} != {ref[mismatch]}')
            results['frozen_weight_prefix_equal'] = prefix
            print(f'PASS: {prefix} exact tokens matched against Candidate A for every variant', flush=True)

        if mode in ('compare', 'all'):
            for name in BUILD_ORDER:
                results['compare'].append(run_variant('perf', name, steps, base, snapshot_token=2048))
            baseline = results['compare'][0]
            losses_ref = read_loss(base / 'perf' / 'a256')
            eligible = []
            for r in results['compare']:
                series = read_loss(base / 'perf' / r['variant'])
                max_delta = max(abs(a-b) for a,b in zip(series, losses_ref))
                r['max_loss_delta_vs_a256'] = round(max_delta, 6)
                r['training_ok'] = r['training_tok_s'] >= baseline['training_tok_s'] * .99
                r['doom_ok'] = r['doom_fps_training'] >= 33.0
                r['loss_ok'] = max_delta <= .01
                if r['training_ok'] and r['doom_ok'] and r['loss_ok']:
                    eligible.append(r)
            best = max(eligible, key=lambda x: x['decoder_tok_s']) if eligible else None
            results['decision'] = {
                'best_eligible': best['variant'] if best else None,
                'best_decoder_tok_s': best['decoder_tok_s'] if best else None,
                'target_tok_s': 1946,
                'target_met': bool(best and best['decoder_tok_s'] >= 1946),
                'baseline_decoder_tok_s': baseline['decoder_tok_s'],
                'baseline_train_tok_s': baseline['training_tok_s'],
            }
            print('\nOVERLAY 005C BENCHMARK (matched steps, single B200 allocation)', flush=True)
            print('variant    decode_tok_s train_tok_s doom_fps max_loss_delta critical_us head_us', flush=True)
            for r in results['compare']:
                head = sum(x['span_us'] for x in r['phase_wall'] if x['kind'] == 'LM_HEAD')
                print(f"{r['variant']:10s} {r['decoder_tok_s']:12.0f} {r['training_tok_s']:11.0f} "
                      f"{r['doom_fps_training']:8.1f} {r['max_loss_delta_vs_a256']:14.6f} "
                      f"{r['snapshot']['whole_us']:11.1f} {head:7.1f}", flush=True)
            print('Decision:', results['decision'], flush=True)
    finally:
        (base / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        vol.commit()
    return results


@app.local_entrypoint()
def main(mode: str = 'build', steps: int = 4000, run_id: str = 'overlay005c-first'):
    if mode == 'build':
        print('CPU image-build-only. No B200 allocation.', flush=True)
        print(json.dumps(check_build.remote(), indent=2))
    elif mode in ('verify', 'compare', 'all'):
        print('PAID B200: exactly one GPU container, sequential variants, no retries.', flush=True)
        result = run_suite.remote(mode, steps, run_id)
        print('Summary:', json.dumps(result.get('decision') or {
            'frozen_weight_prefix_equal': result.get('frozen_weight_prefix_equal')}, indent=2))
    else:
        raise ValueError('mode: build | verify | compare | all')
