#!/usr/bin/env python3
"""OFA Overlay 005 candidate A: fuse both LM-head argmax reductions.

Apply only in a disposable experiment branch based on Overlay 004.
This is a source-only patch; it does not compile CUDA or launch Modal.
"""
from pathlib import Path
import sys

TARGET = Path('llm/b200/megakernel/decode_wide.cuh')


def replace_one(text: str, before: str, after: str, description: str) -> str:
    n = text.count(before)
    if n != 1:
        raise ValueError(f'{description}: expected exactly one anchor, got {n}; no files changed')
    return text.replace(before, after, 1)


def transform(text: str) -> str:
    text = replace_one(
        text,
        '#ifndef DECODE_HR\n#define DECODE_HR 128\n#endif\n',
        '#ifndef DECODE_HR\n#define DECODE_HR 128\n#endif\n'
        '// Overlay 005 candidate A: independent reductions, shared sync points.\n'
        '// Set to 0 to compile the unchanged Overlay 004 reductions.\n'
        '#ifndef DECODE_HEAD_FUSED_ARGMAX\n#define DECODE_HEAD_FUSED_ARGMAX 1\n#endif\n',
        'compile-time switch',
    )

    pair = '''// Overlay 005 candidate A: same comparisons and warp/reduction order for both
// streams, but the shared-memory exchanges and barriers are synchronized once.
__device__ __forceinline__ void block_argmax_pair(float& a, int& ai, float& b, int& bi,
                                                  float* sva, int* sia, float* svb, int* sib) {
  for (int o = 16; o; o >>= 1) {
    float oa = __shfl_xor_sync(0xffffffffu, a, o);
    int oai = __shfl_xor_sync(0xffffffffu, ai, o);
    float ob = __shfl_xor_sync(0xffffffffu, b, o);
    int obi = __shfl_xor_sync(0xffffffffu, bi, o);
    if (better(oa, oai, a, ai)) { a = oa; ai = oai; }
    if (better(ob, obi, b, bi)) { b = ob; bi = obi; }
  }
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { sva[w] = a; sia[w] = ai; svb[w] = b; sib[w] = bi; }
  __syncthreads();
  a = sva[0]; ai = sia[0]; b = svb[0]; bi = sib[0];
#pragma unroll
  for (int i = 1; i < 8; i++) {
    if (better(sva[i], sia[i], a, ai)) { a = sva[i]; ai = sia[i]; }
    if (better(svb[i], sib[i], b, bi)) { b = svb[i]; bi = sib[i]; }
  }
  __syncthreads();
}

'''
    text = replace_one(text, '__device__ __forceinline__ uint64_t splitmix64(',
                       pair + '__device__ __forceinline__ uint64_t splitmix64(', 'pair helper')
    text = replace_one(text, '  size_t hd = (C::D + 16 + C::HR) * 4;',
                       '  size_t hd = (C::D + 32 + C::HR) * 4; // two argmax scratch pairs',
                       'shared-memory allocation')
    text = replace_one(text, '  float* lg = scratch + 16;           // HR',
                       '  float* lg = scratch + 32;           // HR (two argmax scratch pairs)',
                       'LM_HEAD scratch layout')

    old = '''  float b0 = v0; int i0 = gi;
  block_argmax(b0, i0, sv, si);
  float b1 = v1; int i1 = gi;
  block_argmax(b1, i1, sv, si);'''
    new = '''  float b0 = v0; int i0 = gi;
  float b1 = v1; int i1 = gi;
#if DECODE_HEAD_FUSED_ARGMAX
  float* svb = scratch + 16;
  int* sib = (int*)(scratch + 24);
  block_argmax_pair(b0, i0, b1, i1, sv, si, svb, sib);
#else
  block_argmax(b0, i0, sv, si);
  block_argmax(b1, i1, sv, si);
#endif'''
    text = replace_one(text, old, new, 'LM_HEAD candidate reduction')
    return text


def main() -> None:
    if not TARGET.is_file():
        sys.exit(f'Cannot find {TARGET}. Run from repository root.')
    before = TARGET.read_text()
    try:
        after = transform(before)
    except ValueError as exc:
        sys.exit(str(exc))
    if '--check' in sys.argv:
        print('All 5 exact source anchors present; patch is applicable (dry run).')
        return
    TARGET.write_text(after)
    print(f'Patched {TARGET} (Overlay 005 candidate A; no GPU tests performed).')
    print('Only this file was changed. To compare against the original algorithm,')
    print('compile with -DDECODE_HEAD_FUSED_ARGMAX=0.')


if __name__ == '__main__':
    main()
