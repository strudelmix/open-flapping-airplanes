#!/usr/bin/env bash
# Runs from anywhere inside the repository. Preserves your Candidate B worktree
# completely; creates Candidate C as a sibling git worktree from remote A.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git rev-parse --show-toplevel)"
PARENT="$(dirname "$ROOT")"
WORKTREE="$PARENT/open-flapping-airplanes-c"
BRANCH='experiment/overlay005c-head512-sampling-v1'
BASE='origin/experiment/overlay005-head-fused-argmax-v1'

printf 'Source repo: %s\n' "$ROOT"
printf 'Candidate C worktree: %s\n' "$WORKTREE"

git -C "$ROOT" fetch origin experiment/overlay005-head-fused-argmax-v1
if [[ ! -d "$WORKTREE" ]]; then
  git -C "$ROOT" worktree add -b "$BRANCH" "$WORKTREE" "$BASE"
else
  ACTUAL="$(git -C "$WORKTREE" branch --show-current)"
  [[ "$ACTUAL" == "$BRANCH" ]] || { echo "Refusing to touch unrelated worktree $WORKTREE ($ACTUAL)" >&2; exit 2; }
  echo 'Reusing existing Candidate C worktree'
fi

python3 "$HERE/apply_patch.py" --root "$WORKTREE"
cp "$HERE/overlay005c_bench.py" "$WORKTREE/experiments/overlay005c_bench.py"
cp "$HERE/test_tile_model.py" "$WORKTREE/experiments/test_overlay005c_tile.py"
cp "$HERE/README.md" "$WORKTREE/experiments/OVERLAY005C_README.md"

(cd "$WORKTREE" && \
 python3 -m py_compile experiments/overlay005c_bench.py && \
 python3 experiments/test_overlay005c_tile.py && \
 bash experiments/check_local.sh && \
 git diff --check)

if [[ -n "$(git -C "$WORKTREE" status --porcelain)" ]]; then
  git -C "$WORKTREE" add llm/b200/megakernel/decode_wide.cuh \
      llm/b200/megakernel/ofa.cu \
      experiments/overlay005c_bench.py experiments/test_overlay005c_tile.py experiments/OVERLAY005C_README.md
  git -C "$WORKTREE" commit -m 'Overlay 005 C: head 512, sampling fast path, matched B200 A/B harness'
fi

echo
echo 'Candidate C created, patched, CPU checked and committed.'
echo "Worktree: $WORKTREE"
echo "Branch: $BRANCH"
echo 'NEXT (CPU-only CUDA compile):'
echo "  cd $WORKTREE && modal run experiments/overlay005c_bench.py --mode build"
echo 'THEN (paid single B200 allocation, if compile passes):'
echo '  modal run experiments/overlay005c_bench.py --mode all --steps 4000 --run-id overlay005c-ab1'
echo 'Optional push:'
echo "  git push -u origin $BRANCH"
