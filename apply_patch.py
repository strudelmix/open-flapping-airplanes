#!/usr/bin/env python3
"""Apply Overlay005 Candidate C to a worktree based on Overlay005 Candidate A.

Only touches two source files, with strict anchors. Re-run safe via marker detection.
"""
from pathlib import Path
import argparse
import subprocess


def once(src: str, old: str, new: str, tag: str) -> str:
    found = src.count(old)
    if found != 1:
        raise RuntimeError(f"{tag}: expected one exact source anchor, got {found}")
    return src.replace(old, new, 1)


def patch_head(src: str) -> str:
    src = once(src,
        '#ifndef DECODE_HEAD_FUSED_ARGMAX\n#define DECODE_HEAD_FUSED_ARGMAX 1\n#endif\n',
        '#ifndef DECODE_HEAD_FUSED_ARGMAX\n#define DECODE_HEAD_FUSED_ARGMAX 1\n#endif\n'
        '// Overlay 005 C: compile-time independent switches.\n'
        '#ifndef DECODE_HEAD_SAMPLE_ONLY\n#define DECODE_HEAD_SAMPLE_ONLY 0\n#endif\n'
        '#ifndef DECODE_HEAD_WRITE_LOGITS\n#define DECODE_HEAD_WRITE_LOGITS 1\n#endif\n',
        'macro options')
    src = once(src,
        '  static_assert(HR_ <= NTHREADS, "head tile cannot exceed the 256-thread candidate reduction");',
        '  static_assert(HR_ <= 512, "head tile currently supports up to two rows per thread");',
        'head HR bound')

    old = '''  // chunk-local candidates
  float* sv = scratch;                  // 8 floats + 8 ints
  int* si = (int*)(scratch + 8);
  const float temp = P.temperature;
  const float invT = temp > 0.f ? 1.f / temp : 1.f;
  const uint32_t step = (uint32_t)__ldcg(&st->step);
  float v0 = -INFINITY, v1 = -INFINITY;
  int gi = t * C::HR + tid;
  if (tid < C::HR) {
    float l = lg[tid];
    st->logits[gi] = l;
    if (gi < P.valid_vocab) {
      v0 = l;
      v1 = temp > 0.f ? l * invT + gumbel(P.seed, step, gi) : l;
    }
  }
  float b0 = v0; int i0 = gi;
  float b1 = v1; int i1 = gi;
#if DECODE_HEAD_FUSED_ARGMAX
  float* svb = scratch + 16;
  int* sib = (int*)(scratch + 24);
  block_argmax_pair(b0, i0, b1, i1, sv, si, svb, sib);
#else
  block_argmax(b0, i0, sv, si);
  block_argmax(b1, i1, sv, si);
#endif
  if (tid == 0) {
    st->cand_v[0][t] = b0; st->cand_i[0][t] = i0;
    st->cand_v[1][t] = b1; st->cand_i[1][t] = i1;
  }
'''
    new = '''  // chunk-local candidates
  float* sv = scratch;                  // 8 floats + 8 ints
  int* si = (int*)(scratch + 8);
  const float temp = P.temperature;
  const float invT = temp > 0.f ? 1.f / temp : 1.f;
  const uint32_t step = (uint32_t)__ldcg(&st->step);
#if !DECODE_HEAD_SAMPLE_ONLY && DECODE_HEAD_WRITE_LOGITS && DECODE_HR <= 256
  // Exact Candidate A implementation retained as the within-build baseline.
  float v0 = -INFINITY, v1 = -INFINITY;
  int gi = t * C::HR + tid;
  if (tid < C::HR) {
    float l = lg[tid];
    st->logits[gi] = l;
    if (gi < P.valid_vocab) {
      v0 = l;
      v1 = temp > 0.f ? l * invT + gumbel(P.seed, step, gi) : l;
    }
  }
  float b0 = v0; int i0 = gi;
  float b1 = v1; int i1 = gi;
#if DECODE_HEAD_FUSED_ARGMAX
  float* svb = scratch + 16;
  int* sib = (int*)(scratch + 24);
  block_argmax_pair(b0, i0, b1, i1, sv, si, svb, sib);
#else
  block_argmax(b0, i0, sv, si);
  block_argmax(b1, i1, sv, si);
#endif
  if (tid == 0) {
    st->cand_v[0][t] = b0; st->cand_i[0][t] = i0;
    st->cand_v[1][t] = b1; st->cand_i[1][t] = i1;
  }
#else
  // HR=512 assigns rows tid and tid+256 to each thread; each thread
  // selects its own winner before the block-wide argmax. For the fast
  // path, only the sampling mode actually consumed by task_sample is stored.
  const int mode = temp > 0.f ? 1 : 0;
  float b0 = -INFINITY, b1 = -INFINITY;
  int i0 = 0x7fffffff, i1 = 0x7fffffff;
#pragma unroll
  for (int rep = 0; rep < (C::HR + NTHREADS - 1) / NTHREADS; ++rep) {
    const int loc = tid + rep * NTHREADS;
    if (loc < C::HR) {
      const int gi = t * C::HR + loc;
      const float l = lg[loc];
#if DECODE_HEAD_WRITE_LOGITS
      st->logits[gi] = l;
#endif
      if (gi < P.valid_vocab) {
#if DECODE_HEAD_SAMPLE_ONLY
        const float value = mode ? l * invT + gumbel(P.seed, step, gi) : l;
        if (better(value, gi, b0, i0)) { b0 = value; i0 = gi; }
#else
        const float g = temp > 0.f ? l * invT + gumbel(P.seed, step, gi) : l;
        if (better(l, gi, b0, i0)) { b0 = l; i0 = gi; }
        if (better(g, gi, b1, i1)) { b1 = g; i1 = gi; }
#endif
      }
    }
  }
#if DECODE_HEAD_SAMPLE_ONLY
  block_argmax(b0, i0, sv, si);
  if (tid == 0) {
    st->cand_v[mode][t] = b0; st->cand_i[mode][t] = i0;
  }
#else
#if DECODE_HEAD_FUSED_ARGMAX
  block_argmax_pair(b0, i0, b1, i1, sv, si, scratch + 16, (int*)(scratch + 24));
#else
  block_argmax(b0, i0, sv, si);
  block_argmax(b1, i1, sv, si);
#endif
  if (tid == 0) {
    st->cand_v[0][t] = b0; st->cand_i[0][t] = i0;
    st->cand_v[1][t] = b1; st->cand_i[1][t] = i1;
  }
#endif
#endif
'''
    return once(src, old, new, 'replace LM_HEAD candidate section')


def patch_profile(src: str) -> str:
    src = once(src,
       '  u64* decode_phase_busy; // optional, instrumentation perturbs throughput',
       '  u64* decode_phase_busy; // optional, instrumentation perturbs throughput\n'
       '  u64 *decode_phase_first, *decode_phase_last; int decode_prof_token; // one-token low-overhead snapshot',
       'context snapshot fields')
    src = once(src,
        '      if (job_id == 1 && C.decode_phase_busy) atomicAdd(C.decode_phase_busy + ph, t1 - t0);',
        '      if (job_id == 1 && C.decode_phase_busy) atomicAdd(C.decode_phase_busy + ph, t1 - t0);\n'
        '      if (job_id == 1 && step == C.decode_prof_token && C.decode_phase_first) {\n'
        '        atomicMin(C.decode_phase_first + ph, t0);\n'
        '        atomicMax(C.decode_phase_last + ph, t1);\n'
        '      }',
        'one token instrumentation')
    src = once(src,
        '  int decode_max_tokens = 1 << 26, decode_out_cap = 1 << 20;',
        '  int decode_max_tokens = 1 << 26, decode_out_cap = 1 << 20;\n'
        '  int decode_prof_token = -1; // no snapshot unless explicitly requested',
        'CLI default')
    src = once(src,
        '    else if (!strcmp(argv[i], "--decode-profile")) decode_profile = true;',
        '    else if (!strcmp(argv[i], "--decode-profile")) decode_profile = true;\n'
        '    else if (is("--decode-prof-token")) decode_prof_token = atoi(argv[++i]);',
        'CLI parse')
    src = once(src,
        '  C.decode_phase_busy = (use_decode && decode_profile) ? dalloc<u64>(DC::PHASES) : nullptr;',
        '  C.decode_phase_busy = (use_decode && decode_profile) ? dalloc<u64>(DC::PHASES) : nullptr;\n'
        '  C.decode_prof_token = decode_prof_token;\n'
        '  C.decode_phase_first = (use_decode && decode_prof_token >= 0) ? dalloc<u64>(DC::PHASES) : nullptr;\n'
        '  C.decode_phase_last = (use_decode && decode_prof_token >= 0) ? dalloc<u64>(DC::PHASES) : nullptr;\n'
        '  if (C.decode_phase_first) {\n'
        '    std::vector<u64> init(DC::PHASES, ~0ull);\n'
        '    CK(cudaMemcpy(C.decode_phase_first, init.data(), init.size() * sizeof(u64), cudaMemcpyHostToDevice));\n'
        '  }',
        'snapshot arrays init')
    anchor = '  if (use_decode && C.decode_phase_busy) {'
    block = '''  if (use_decode && C.decode_phase_first) {
    std::vector<u64> first(DC::PHASES), last(DC::PHASES);
    CK(cudaMemcpy(first.data(), C.decode_phase_first, DC::PHASES * sizeof(u64), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(last.data(), C.decode_phase_last, DC::PHASES * sizeof(u64), cudaMemcpyDeviceToHost));
    u64 previous_end = 0, first_begin = 0;
    double sum_us = 0.0, gap_us = 0.0;
    bool complete = true;
    for (int ph = 0; ph < DC::PHASES; ++ph) {
      if (first[ph] == ~0ull || last[ph] < first[ph]) { complete = false; break; }
      const char* kind = ph < 3 * L ?
          ((ph % 3 == 0) ? "QKV" : (ph % 3 == 1) ? "ATTWO" : "UPDN") :
          (ph == 3 * L ? "LM_HEAD" : "SAMPLE");
      const double span = (last[ph] - first[ph]) * 1e-3;
      const double gap = previous_end ? ((double)((long long)first[ph] - (long long)previous_end) * 1e-3) : 0.0;
      if (!ph) first_begin = first[ph];
      printf("decode phase wall: ph=%d kind=%s tasks=%d span_us=%.3f gap_us=%.3f\\n",
             ph, kind, decode::phase_tasks<DC>(ph), span, gap);
      sum_us += span; gap_us += gap;
      previous_end = last[ph];
    }
    if (complete)
      printf("decode phase snapshot: token=%d whole_us=%.3f spans_us=%.3f gaps_us=%.3f\\n",
             decode_prof_token, (last.back() - first_begin) * 1e-3, sum_us, gap_us);
    else printf("decode phase snapshot: INCOMPLETE token=%d\\n", decode_prof_token);
  }
'''
    src = once(src, anchor, block + anchor, 'snapshot reporting')
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='Candidate C worktree')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    h = root / 'llm/b200/megakernel/decode_wide.cuh'
    c = root / 'llm/b200/megakernel/ofa.cu'
    if not h.is_file() or not c.is_file():
        ap.error(f'Not an OFA repository: {root}')
    before_h, before_c = h.read_text(), c.read_text()
    if 'Overlay 005 C: compile-time independent switches.' in before_h:
        print('Candidate C patch already installed; skipping source changes')
        return
    # Calculate both edits entirely before touching either file.
    after_h = patch_head(before_h)
    after_c = patch_profile(before_c)
    h.write_text(after_h)
    c.write_text(after_c)
    print('PASS: Candidate C applied to decode_wide.cuh and ofa.cu')
    r = subprocess.run(['git', 'diff', '--check'], cwd=root)
    if r.returncode:
        raise SystemExit('git diff --check failed')

if __name__ == '__main__':
    main()
