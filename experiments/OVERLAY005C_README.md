# Overlay 005 C — reproducible LM_HEAD optimization experiment

**Base:** Candidate A, `experiment/overlay005-head-fused-argmax-v1`, commit `8107269`.

## Goals

- Preserve A256 exact code as within-build baseline.
- Test four variants, compiled with the same CUDA 12.8.1, `sm_100a`, tcgen05 stages=2 flags:
  - `a256`: HR=256, fused double argmax, writes logits (Candidate A).
  - `tile512`: HR=512, fused double argmax, writes logits (tile effect alone).
  - `fast256`: HR=256, only active sampler argmax, no full-logit write.
  - `fast512`: HR=512, only active sampler argmax, no full-logit write.
- Add optional **single-token phase wall snapshots**: min start / max end per phase for token 2048, with transition gap. This instruments one token; unlike `--decode-profile`, it does not add an atomic to every decode task for every step. Any timing instrument still has some overhead; compare variants identically.
- Verify fixed-weight token equality (`--no-opt`) against A256, then compare concurrent **training + decoder + Doom**, all on a single B200 sequentially.

## Codespaces installation

Upload / unzip the package somewhere in the repository worktree, then:

```bash
cd /workspaces/open-flapping-airplanes
bash ./ofa_overlay005c/install.sh
cd /workspaces/open-flapping-airplanes-c
```

The installer **does not modify Candidate B**, even if B has uncommitted files: it uses `git worktree add` to create a sibling checkout based on the remote Candidate A branch, applies two strict-anchored edits, copies the benchmark script, checks CPU tests (including a new tile-routing simulation), and commits Candidate C locally. It does *not* push automatically.

## Execution

First (no GPU):

```bash
modal run experiments/overlay005c_bench.py --mode build
```

After build passes, this is **paid B200** (one GPU allocation, four fixed-weight correctness runs, then four performance runs, sequential; no automatic retries):

```bash
modal run experiments/overlay005c_bench.py --mode all --steps 4000 --run-id overlay005c-ab1
```

Additional modes `--mode verify` (frozen-weight only) and `--mode compare` (training performance only) are available but each allocates a new B200. `all` keeps all stages in one allocation and will stop before performance runs if equivalence fails. For short dry runs use `--steps 1000` but compare with **matched** workloads, not older 4,000-step results.

The command prints a summary. Artifacts including `summary.json`, `run.log`, `tokens.txt`, `loss.txt` for each run are persisted to the `ofa-b200-data` Modal volume under `/vol/results/ofa_overlay005c/<run_id>/`. For example:

```bash
modal volume get ofa-b200-data /results/ofa_overlay005c/overlay005c-ab1/summary.json ./overlay005c-summary.json
```

`--no-opt` leaves model weights fixed, so the first **8,192 token IDs**, including markers, must agree for all variants. The performance runs use live training, so their token sequences are **not** required to match. Each performance run checks training loss curve vs A256 (maximum difference <=0.01), at least 33 Doom FPS, a working token ring, and per-token snapshot completeness.

Promotion criterion: Candidate C must exceed ~**1,946 decoder tok/s**, preserve training throughput to within 1% of same-batch A256, preserve Doom at >=33 FPS, and pass frozen-weight equality. Treat small measured differences as run-to-run noise until repeated.

No claims of >5.6% have been established until measured. GPU resource costs apply to modes other than `build`.
