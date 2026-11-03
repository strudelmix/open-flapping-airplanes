# Overlay 002: Modal remote-import fix and preflight

Overlay 001's `autobench.py` imported `modal_app` at module load. Modal imports
the function script again inside remote containers, where that module was not
mounted. This caused repeated failing containers before any useful GPU run.

This update:
- Imports upstream `b200-run/modal_app.py` only in the local Modal launcher
  and uses stable path/volume constants within remote containers.
- Explicitly mounts the pure-Python `experiments` package in the GPU image.
- Adds a separate, cheap CPU-only remote import check (`--suite preflight`).
- Automatically performs this check before `--suite quick` submits a B200 job.
- Changes the default run ID to `ofa001-v2` to distinguish results from the
  broken earlier launcher. The previous data preparation stays intact.

## Validate in Codespaces after applying ZIP via GitHub Actions

```bash
bash experiments/check_local.sh
modal run experiments/autobench.py --suite preflight
```

Only after preflight PASSES, and after verifying your Modal spending limit:

```bash
modal run experiments/autobench.py --suite quick
```

Do not re-run data preparation; it already succeeded. CUDA compilation and
actual B200 correctness/performance were not validated by this ZIP package.
The CPU preflight proves Python imports, not CUDA compilation or numeric
correctness. Benchmark runs are billable.
