# OFA Overlay 004 — Decoder endurance + opt-in phase profiling

Applies **only** to `experiment/autobench`, based on Overlay 002. No Overlay 003 prerequisites or edits to the existing quick benchmark code.

## What this addresses

- A **decode-only 64-bit scheduler**: task tickets and per-phase completion counts no longer overflow after a few hundred thousand to ~1M tokens. The training scheduler stays 32-bit and preserves the existing validated safety limits.
- A pinned **rolling token-output ring**, read and written continuously by the CPU while the single persistent training+decoding+Doom CUDA kernel runs. It avoids stopping recording at 1,048,576 tokens. A slow consumer causes a visible **hard error**, never silent truncated output.
- **Endurance markers** printed while the kernel runs: training step and total generated decoder tokens, including at the end. Their purpose is to prove decoding did not die earlier.
- `--decode-profile`: optional busy-time totals for QKV, ATTWO, UPDN, LM_HEAD, SAMPLE. Default OFF; it changes device timings and is a directional diagnostic, *not* a latency decomposition.
- `--decode-out-cap N` (4,096–1,048,576, default 1,048,576) allows an 8,192-entry ring-wrap smoke test without renting a GPU for an hour.
- `--decode-max-tokens N` (from out-cap to 268,435,456, default 67,108,864) caps the decoder even on a very long training run.

## CPU checks (no GPU)

```bash
bash experiments/check_local.sh
modal run experiments/autobench.py --suite preflight
```

## Recommended first paid B200 test

```bash
modal run experiments/endurance.py --steps 10000 --ring-cap 8192
```

This launches **one B200 and one HR256/Doom8 run**. It keeps the training schedule matched to the number of steps, uses prepared TinyStories weights/data, and checks that decoding progresses after the 75% mark, all token IDs were recorded across ring wraps, the loss file has the required number of steps, and the Doom engine produced frames. Raw logs/token output/loss and summary are committed to Modal Volume `ofa-b200-data` under `/vol/results/ofa_decoder_endurance/decoder004-test/`.

Note: batch starts use upstream `starts_4000.u32`, which cycles on longer training runs. This *endurance verification* is not a rigorous 10k-step model-quality benchmark; it specifically tests the long-lived decoder and co-scheduling. It always starts from the initial weights and does not resume a checkpoint.

## Profile one run afterward, only if endurance passes

```bash
modal run experiments/endurance.py --steps 10000 --ring-cap 8192 --profile --run-id decoder004-profile
```

Then, for a same-allocation HR128 control (extra paid run):

```bash
modal run experiments/endurance.py --steps 10000 --ring-cap 8192 --hr 128 --run-id decoder004-hr128
```

These are **not** independent performance measurements when `--profile` differs, so compare timings with profiling off. Use phase busy shares to select the next optimization rather than guessing.

## Safety and validation status

- Existing CUDA source changes are strict exact-anchor substitutions from Overlay 001, applied by `experiments/decoder004_patch.py` only once. If the expected code has changed, the install aborts rather than silently patching different code.
- CPU unit tests, syntax checks and simulated GitHub install can pass without a GPU. They **do not** prove CUDA will compile or the decoder will be safe at runtime. Start with one capped endurance run.
- The output ring is synchronous within the GPU kernel; the host polls every 100 ms and fails if the producer overtakes the consumer by more than the ring capacity. If the host pauses for too long, the test exits with a clear error.
- The slow-drain scenario and CUDA cache coherence must be verified on B200 before trusting long-run results.
