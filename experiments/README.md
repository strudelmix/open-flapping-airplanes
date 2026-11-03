# OFA AutoBench — Overlay 001

This is a *benchmarking scaffold*, not a demonstrated performance speedup. The B200
builds, numerical checks and GPU results require an actual Modal B200 execution.

## What's changed

- `llm/b200/megakernel/ofa.cu`: host-side bounds for 32-bit tickets and signed
  completion counters. Decoder now stops at the allocated output capacity at most.
  An oversized training job is rejected; normal `--steps 4000` is unaffected.
- `llm/b200/megakernel/decode_wide.cuh`: refuse `DECODE_HR > 256`, because
  the 256-thread head candidate reduction does not support wider tiles.
- `experiments/autobench.py`: builds HR64, HR128 (original), HR256 and runs
  two repeats per short experiment, then a Doom SM sweep and confirmation.
- CPU regression tests, offline run plan, metrics parser, CSV/JSON/Markdown
  reports and dependency-free SVG bar charts.

## Safe checks first: no GPU bill

On GitHub, the overlay installer runs `bash experiments/check_local.sh` on a free
GitHub-hosted CPU runner before committing the overlay. These tests **do not**
compile CUDA or execute a GPU kernel.

For a local experiment plan (Python 3):

```sh
python3 experiments/plan.py
```

## One-time Modal setup

1. Sign in to [Modal](https://modal.com), configure billing / GPU quota if needed,
   and authenticate its CLI (`pip install modal`; `modal setup`).
2. Using Jerry's original B200 prep path, run:

```sh
modal run b200-run/modal_app.py --step prep
```

This downloads and prepares TinyStories and initialization in the Modal Volume
`ofa-b200-data`. It is required **once** before the benchmark suite.

3. Inspect the planned commands without requesting a GPU:

```sh
python3 experiments/plan.py
# Or, if Modal is installed:
modal run experiments/autobench.py --suite dry-run
```

4. **Only when you approve renting a B200**, run:

```sh
modal run experiments/autobench.py --suite quick
```

A single B200 function runs experiments **sequentially** and commits raw results
to `ofa-b200-data:/results/ofa_autobench/ofa001-v1/` after each run. Repeated
invocations reuse completed runs with identical exact command and binary hash.
The local entry point saves `experiments/results/ofa001-v1/benchmark.md` and
`summary.json`. The Volume has all raw logs, complete loss and token dumps,
CSV, JSON and SVG charts. For large logs, use the Modal Volume browser or
`modal volume get` with the path shown by the function.

**Cloud costs are separate from any ChatGPT/Codex subscription.** The base
image builds CUDA/Doom during image construction and may take substantial time.
No CUDA build or GPU measurement was possible in the initial overlay package.

## Experimental fairness

- HR=128 uses the original `ofa_tc2` build; variants use the same tcgen05
  backend and 2 pipeline stages. Model S, dataset, init, starts, prompt,
  answer length and seed remain fixed.
- First, frozen weights (`--no-opt`) and greedy (`--temp 0`) compare the first
  128 common token IDs and all 100 training losses. Different total token
  counts are expected when variant throughputs differ.
- Only passing variants enter the performance comparison. Every quick-run
  configuration runs at least twice. GPU timeouts and missing metrics fail.
- The throughput score is the geometric mean of the ratios of training and
  sampled-token throughput, relative to the unchanged baseline. Require
  Doom >=34 FPS during training and no more than 3% training regression.
- The best HR is selected before the Doom SM sweep: 8 -> 4 -> 2, optionally 1.
  Inference remains 24 SMs, training gets the remainder.
- Full 4000-step confirmation runs compare the baseline against the chosen
  candidate on the same GPU. One full run is not a statistical guarantee.
- The report distinguishes recorded failures from measured performance.

## Important limits

This experiment does **not** implement a 64-bit scheduler, a new GEMM or dynamic
cross-workload scheduling. The counter change is a conservative host cap.
`DECODE_HR` values other than 64/128/256 are outside this experiment.

The suite deliberately keeps loss tolerances strict (`1e-3` maximum absolute
difference) and will treat material deviations as failed comparisons. A loss
mismatch may reflect actual numerical issues, differences in update timing or
nondeterminism; investigate rather than silently relaxing the tolerance.

The runner assumes the original Modal B200 image, Doom build, and Volume data
preparation continue to work. CUDA compilation and performance may need a
small corrective overlay after the first real execution.
