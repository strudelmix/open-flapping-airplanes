#!/usr/bin/env python3
"""Strict, one-shot source transformation for OFA Overlay 004.

This is intentionally anchored to the Overlay 001/002 CUDA code. A mismatch
aborts the installer, rather than guessing and producing a broken kernel.
"""
from pathlib import Path
import sys


def replace_once(text: str, before: str, after: str, target: str) -> str:
    hits = text.count(before)
    if hits != 1:
        raise RuntimeError(f'{target}: expected exactly one matching anchor, got {hits}: {before[:95]!r}')
    return text.replace(before, after, 1)


def replace_n(text: str, before: str, after: str, count: int, target: str) -> str:
    hits = text.count(before)
    if hits != count:
        raise RuntimeError(f'{target}: expected {count} matching anchors, got {hits}: {before[:95]!r}')
    return text.replace(before, after)


def apply_ofa(src: str) -> str:
    t = 'ofa.cu'
    src_holder = [src]
    def one(before, after):
        src_holder[0] = replace_once(src_holder[0], before, after, t)
    def many(before, after, n):
        src_holder[0] = replace_n(src_holder[0], before, after, n, t)

    one('  int* done;\n  unsigned* ticket;\n',
        '  int* done;\n  unsigned* ticket;\n  u64* done64;           // decode only: avoid 32-bit per-phase wrap\n  u64* ticket64;         // decode only: avoid 32-bit task-ticket wrap\n')
    one('  u64 *phase_busy, *phase_first, *phase_last; int prof_step;\n',
        '  u64 *phase_busy, *phase_first, *phase_last; int prof_step;\n'
        '  u64* decode_phase_busy; // optional, instrumentation perturbs throughput\n')
    one('__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }',
        '__device__ __forceinline__ int ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }\n'
        '__device__ __forceinline__ u64 ld_acq(const u64* p) { u64 v; asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v; }')
    one('  const unsigned total = (unsigned)J.limit * (unsigned)J.tps;',
        '  const u64 total = (u64)J.limit * (u64)J.tps; // widened for decode; training remains 32-bit checked')
    one('        const unsigned g = atomicAdd(const_cast<unsigned*>(J.ticket), 1u);',
        '        const u64 g = job_id == 1 ? atomicAdd(J.ticket64, 1ull) : (u64)atomicAdd(const_cast<unsigned*>(J.ticket), 1u);')
    one('          step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);',
        '          step = (int)(g / (u64)J.tps); const int r = (int)(g - (u64)step * (u64)J.tps);')
    one('        const unsigned g = ld_vol(J.ticket);',
        '        const u64 g = job_id == 1 ? *reinterpret_cast<const volatile u64*>(J.ticket64) : (u64)ld_vol(J.ticket);')
    one('        step = (int)(g / (unsigned)J.tps); const int r = (int)(g - (unsigned)step * (unsigned)J.tps);',
        '        step = (int)(g / (u64)J.tps); const int r = (int)(g - (u64)step * (u64)J.tps);')
    one('        if (atomicCAS(const_cast<unsigned*>(J.ticket), g, g + 1) == g) { local = r - __ldg(J.prefix + ph); break; }',
        '        const bool claimed = job_id == 1 ? atomicCAS(J.ticket64, g, g + 1ull) == g\n'
        '                                         : (u64)atomicCAS(const_cast<unsigned*>(J.ticket), (unsigned)g, (unsigned)(g + 1)) == g;\n'
        '        if (claimed) { local = r - __ldg(J.prefix + ph); break; }')
    many('              const int need = __ldg(J.ntasks + e.x) * (step + 1 + e.y);\n'
         '              if (need > 0 && ld_acq(J.done + e.x) < need) ready = false;',
         '              const u64 need = (u64)__ldg(J.ntasks + e.x) * (u64)(step + 1 + e.y);\n'
         '              if (need > 0 && (job_id == 1 ? ld_acq(J.done64 + e.x) : (u64)ld_acq(J.done + e.x)) < need) ready = false;', 1)
    many('            const int need = __ldg(J.ntasks + e.x) * (step + 1 + e.y);\n'
         '            if (need > 0 && ld_acq(J.done + e.x) < need) ready = false;',
         '            const u64 need = (u64)__ldg(J.ntasks + e.x) * (u64)(step + 1 + e.y);\n'
         '            if (need > 0 && (job_id == 1 ? ld_acq(J.done64 + e.x) : (u64)ld_acq(J.done + e.x)) < need) ready = false;', 1)
    one('      busy += t1 - t0;\n      if (job_id == 0) {',
        '      busy += t1 - t0;\n'
        '      if (job_id == 1 && C.decode_phase_busy) atomicAdd(C.decode_phase_busy + ph, t1 - t0);\n'
        '      if (job_id == 0) {')
    one('      const int old = atomicAdd(J.done + ph, 1);\n'
        '      if (job_id == 0 && ph == J.last_ph && old == __ldg(J.ntasks + ph) * (step + 1) - 1) {',
        '      // SAMPLE is a one-task phase. Publish before completion so token N+1 cannot overtake N.\n'
        '      if (job_id == 1 && ph == J.last_ph && C.live) {\n'
        '        __threadfence_system(); C.live->decode_tokens = (u64)(step + 1); __threadfence_system();\n'
        '      }\n'
        '      const u64 old = job_id == 1 ? atomicAdd(J.done64 + ph, 1ull) : (u64)atomicAdd(J.done + ph, 1);\n'
        '      if (job_id == 0 && ph == J.last_ph && old == (u64)__ldg(J.ntasks + ph) * (u64)(step + 1) - 1ull) {')
    one('int port = 0; std::string wad_path, vocab_path, prompt_text = "Once upon a time", index_path = "index.html"; bool no_doom = false;',
        'int port = 0; std::string wad_path, vocab_path, prompt_text = "Once upon a time", index_path = "index.html"; bool no_doom = false, decode_profile = false;\n'
        '  int decode_max_tokens = 1 << 26, decode_out_cap = 1 << 20;')
    one('    else if (is("--answer-len")) answer_len = atoi(argv[++i]); else if (is("--start-delay")) start_delay = atof(argv[++i]);',
        '    else if (is("--answer-len")) answer_len = atoi(argv[++i]); else if (is("--start-delay")) start_delay = atof(argv[++i]);\n'
        '    else if (is("--decode-max-tokens")) decode_max_tokens = atoi(argv[++i]); else if (is("--decode-out-cap")) decode_out_cap = atoi(argv[++i]);\n'
        '    else if (!strcmp(argv[i], "--decode-profile")) decode_profile = true;')
    one('  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));',
        '  if (decode_out_cap < 4096 || decode_out_cap > (1 << 20) || decode_max_tokens < decode_out_cap || decode_max_tokens > (1 << 28)) {\n'
        '    fprintf(stderr, "invalid decoder limits: require 4096 <= --decode-out-cap <= 1048576 and out-cap <= --decode-max-tokens <= 268435456\\n");\n'
        '    return 2;\n'
        '  }\n'
        '  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));')
    one('  auto upload_job = [&](Builder& b, std::vector<int>& pre, int limit, int last, Job& J) {',
        '  auto upload_job = [&](Builder& b, std::vector<int>& pre, int limit, int last, Job& J, bool wide = false) {')
    one('    ddone = dalloc<int>(n); dticket = dalloc<unsigned>(1);',
        '    ddone = wide ? nullptr : dalloc<int>(n); dticket = wide ? nullptr : dalloc<unsigned>(1);\n'
        '    J.done64 = wide ? dalloc<u64>(n) : nullptr; J.ticket64 = wide ? dalloc<u64>(1) : nullptr;')
    one('  decode::State<DC>* dstate = nullptr; Builder Dd; std::vector<int> dprefix; int* h_out_tokens_pinned = nullptr; const int out_cap = 1 << 20;',
        '  decode::State<DC>* dstate = nullptr; Builder Dd; std::vector<int> dprefix; int* h_out_tokens_pinned = nullptr; const int out_cap = decode_out_cap;')
    one('    const int decode_limit = ofa_limits::max_safe_steps(dprefix[DC::PHASES], Dd.ntasks, out_cap);\n'
        '    if (decode_limit <= 0) { fprintf(stderr, "invalid decode scheduler bounds\\n"); return 2; }\n'
        '    if (decode_limit < (1 << 22)) {\n'
        '      printf("decode safety: cap %d tokens (was %d; 32-bit ticket, signed completion counters, output buffer)\\n", decode_limit, 1 << 22);\n'
        '    }\n'
        '    upload_job(Dd, dprefix, decode_limit, DC::PHASES - 1, C.dec);',
        '    const int decode_limit = decode_max_tokens;\n'
        '    upload_job(Dd, dprefix, decode_limit, DC::PHASES - 1, C.dec, true);\n'
        '    printf("decode safety: 64-bit ticket+completion; token budget=%d; rolling output cap=%d\\n", decode_limit, out_cap);')
    one('  C.dm.finished = C.finished;\n  CK(cudaDeviceSynchronize());',
        '  C.dm.finished = C.finished;\n'
        '  C.decode_phase_busy = (use_decode && decode_profile) ? dalloc<u64>(DC::PHASES) : nullptr;\n'
        '  CK(cudaDeviceSynchronize());')
    one('  std::vector<int> seen_tokens; size_t shown = 0; std::string textdump;\n'
        '  while (cudaEventQuery(e1) == cudaErrorNotReady) {\n'
        '    usleep(200000);\n'
        '    if (use_decode) { unsigned long long t = 0; (void)t; }\n'
        '  }\n'
        '  CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());',
        '  FILE* token_file = nullptr;\n'
        '  if (use_decode && !text_out.empty()) {\n'
        '    token_file = fopen(text_out.c_str(), "w");\n'
        '    if (!token_file) { fprintf(stderr, "cannot open token output %s\\n", text_out.c_str()); return 1; }\n'
        '  }\n'
        '  u64 tokens_drained = 0, sampled_count = 0, answer_count = 0;\n'
        '  std::vector<int> first_tokens; first_tokens.reserve(40);\n'
        '  auto drain_tokens = [&]() {\n'
        '    if (!use_decode || !live_host) return;\n'
        '    const u64 published = live_host->decode_tokens;\n'
        '    if (published < tokens_drained || published - tokens_drained > (u64)out_cap) {\n'
        '      fprintf(stderr, "decoder ring overrun / regressed progress: consumed=%llu published=%llu cap=%d\\n", tokens_drained, published, out_cap);\n'
        '      if (token_file) fclose(token_file);\n'
        '      exit(3); // never silently report a truncated decoder stream\n'
        '    }\n'
        '    for (u64 i = tokens_drained; i < published; i++) {\n'
        '      const int tok = h_out_tokens_pinned[(size_t)(i % (u64)out_cap)];\n'
        '      if (first_tokens.size() < 40) first_tokens.push_back(tok);\n'
        '      if (token_file) fprintf(token_file, "%d ", tok);\n'
        '      if (tok == decode::OUT_RESTART) answer_count++;\n'
        '      else if (tok >= 0 && tok < decode::OUT_RESTART) sampled_count++;\n'
        '    }\n'
        '    tokens_drained = published;\n'
        '  };\n'
        '  const int marks[] = {std::max(1, steps / 4), std::max(1, steps / 2), std::max(1, steps * 3 / 4), std::max(1, steps * 9 / 10)};\n'
        '  int next_mark = 0;\n'
        '  while (cudaEventQuery(e1) == cudaErrorNotReady) {\n'
        '    usleep(100000);\n'
        '    drain_tokens();\n'
        '    if (use_decode && live_host) {\n'
        '      while (next_mark < 4 && (int)live_host->steps >= marks[next_mark]) {\n'
        '        printf("decode progress: train_step=%u decoder_tokens=%llu doom_frames=%u\\n", live_host->steps, (u64)live_host->decode_tokens, live_host->frames);\n'
        '        next_mark++;\n'
        '      }\n'
        '    }\n'
        '  }\n'
        '  CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());\n'
        '  drain_tokens();\n'
        '  if (token_file) { if (fclose(token_file)) { fprintf(stderr, "token dump write failed\\n"); return 1; } }\n'
        '  if (use_decode && live_host) printf("decode progress: train_step=%u decoder_tokens=%llu doom_frames=%u (final)\\n", live_host->steps, tokens_drained, live_host->frames);')
    one('      long gen = 0, answers = 0;\n'
        '      for (int i = 0; i < std::min(dstep, out_cap); i++) { const int id = h_out_tokens_pinned[i]; if (id == decode::OUT_RESTART) answers++; else if (id >= 0 && id < decode::OUT_RESTART) gen++; }\n'
        '      printf("decode answers: %ld answers (%d prompt tokens + %d sampled each), %ld sampled tokens -> %.0f tok/s sampled (prompt steps excluded)\\n",\n'
        '             answers, (int)forced.size(), answer_len, gen, gen / ((double)(sd[steps - 1] - sd[0]) * 1e-9 + 1e-9));',
        '      printf("decode answers: %llu answers (%d prompt tokens + %d sampled each), %llu sampled tokens -> %.0f tok/s sampled (prompt steps excluded)\\n",\n'
        '             answer_count, (int)forced.size(), answer_len, sampled_count, sampled_count / ((double)(sd[steps - 1] - sd[0]) * 1e-9 + 1e-9));')
    one('    FILE* f = text_out.empty() ? nullptr : fopen(text_out.c_str(), "w"); int cnt = std::min(dstep, out_cap);\n'
        '    if (f) { for (int i = 0; i < cnt; i++) fprintf(f, "%d ", h_out_tokens_pinned[i]); fclose(f); }\n'
        '    printf("first sampled token ids:"); for (int i = 0; i < std::min(cnt, 40); i++) printf(" %d", h_out_tokens_pinned[i]); printf("\\n");',
        '    printf("first sampled token ids:"); for (int tok : first_tokens) printf(" %d", tok); printf("\\n");')
    one('  if (dm.on) {\n    const unsigned fr = live_host->frames;',
        '  if (use_decode && C.decode_phase_busy) {\n'
        '    std::vector<u64> phase(DC::PHASES);\n'
        '    CK(cudaMemcpy(phase.data(), C.decode_phase_busy, DC::PHASES * sizeof(u64), cudaMemcpyDeviceToHost));\n'
        '    double groups[5] = {};\n'
        '    for (int p = 0; p < DC::PHASES; p++) groups[p >= 3 * L ? (p == 3 * L ? 3 : 4) : p % 3] += (double)phase[p];\n'
        '    const char* names[] = {"QKV", "ATTWO", "UPDN", "LM_HEAD", "SAMPLE"};\n'
        '    double sum = 0; for (double v : groups) sum += v;\n'
        '    printf("decode profile: summed block-task busy time (NOT serial wall time; instrumentation ON)\\n");\n'
        '    for (int i = 0; i < 5; i++) printf("  decode phase %-8s %.3f block-s %5.1f%%\\n", names[i], groups[i] * 1e-9, sum ? 100.0 * groups[i] / sum : 0.0);\n'
        '  }\n'
        '  if (dm.on) {\n    const unsigned fr = live_host->frames;')
    return src_holder[0]


def apply_decode(src: str) -> str:
    # Both answer mode and story mode use the same pinned rolling ring.
    return replace_n(src, 'st->out_tokens && step < st->out_cap) st->out_tokens[step] =',
                     'st->out_tokens && st->out_cap > 0) st->out_tokens[step % st->out_cap] =', 2, 'decode_wide.cuh')


def apply_doom(src: str) -> str:
    return replace_once(src, 'struct Live { volatile unsigned frames, steps; volatile float loss; };',
                        'struct Live { volatile unsigned frames, steps; volatile float loss; volatile u64 decode_tokens; };', 'doom_job.cuh')


def main() -> None:
    base = Path.cwd() / 'llm/b200/megakernel'
    targets = {'ofa.cu': apply_ofa, 'decode_wide.cuh': apply_decode, 'doom_job.cuh': apply_doom}
    # Compute all edits first: mismatch cannot leave a partially written working tree.
    changes = {}
    for name, transform in targets.items():
        p = base / name
        if not p.is_file() or p.is_symlink():
            raise RuntimeError(f'missing or unsafe source: {p}')
        before = p.read_text()
        after = transform(before)
        if before == after:
            raise RuntimeError(f'no change to {p}')
        changes[p] = after
    for p, after in changes.items():
        p.write_text(after)
        print('Patched', p)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError) as exc:
        sys.exit('Overlay 004 cannot be applied: ' + str(exc))
