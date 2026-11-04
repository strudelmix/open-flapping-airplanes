// decode_wide.cuh (copy of llm/decode/decode.cuh with D<=2048 allowed and ModelF added) : batch-1 autoregressive decode of open-flapping-airplanes model "S" as megakernel tasks.
// Header-only. See README.md. Every task is a __device__ function called by ALL 256 threads of a block.
// No blockIdx/gridDim use inside tasks. Deterministic (fixed reduction orders, no float atomics).
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cstddef>

namespace decode {
#ifdef DECODE_TRACE
__device__ unsigned long long* g_trace;
#define DTRACE(t, k) do { if (threadIdx.x == 0) { unsigned long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); g_trace[(t) * 8 + (k)] = _t; } } while (0)
#else
#define DTRACE(t, k) do {} while (0)
#endif

typedef __nv_bfloat16 bf16;
constexpr int MAXL = 16;
constexpr int NTHREADS = 256;

// ---------------------------------------------------------------- configuration
#ifndef DECODE_POLL_MIN_NS
#define DECODE_POLL_MIN_NS 50
#endif
#ifndef DECODE_POLL_MAX_NS
#define DECODE_POLL_MAX_NS 500
#endif
#ifndef DECODE_HR
#define DECODE_HR 128
#endif
#ifndef DECODE_HEAD_FUSED_ARGMAX
#define DECODE_HEAD_FUSED_ARGMAX 1
#endif
// Overlay 005 C: compile-time independent switches.
#ifndef DECODE_HEAD_SAMPLE_ONLY
#define DECODE_HEAD_SAMPLE_ONLY 0
#endif
#ifndef DECODE_HEAD_WRITE_LOGITS
#define DECODE_HEAD_WRITE_LOGITS 1
#endif
template <int L_, int D_, int H_, int T_, int V_, int HR_ = DECODE_HR>
struct Cfg {
  static constexpr int L = L_, D = D_, H = H_, DH = 64, T = T_, V = V_, FF = 4 * D_;
  static constexpr int HR = HR_;                 // vocab rows per LM-head task
  static constexpr int NCB = D_ / 256;           // column blocks (256 cols) of the fused attention+Wo task
  static constexpr int NPD = FF / 64;            // fused up+down tasks == number of down partial sums
  static constexpr int NCHUNK = V_ / HR_;
  static constexpr int PHASES = 3 * L_ + 2;      // per layer: QKV, ATTWO, UPDN ; then HEAD, SAMPLE
  static_assert(DH == 64 && D_ == H_ * 64, "dh must be 64");
  static_assert(D_ % 256 == 0 && D_ <= 2048, "D % 256 == 0, D <= 2048");   // widened from 1024 for config F (L=4, d=2048)
  static_assert(V_ % HR_ == 0 && HR_ % 64 == 0, "V % HR");
  static_assert(HR_ <= 512, "head tile currently supports up to two rows per thread");
};
using ModelS = Cfg<12, 768, 12, 512, 100352>;
using ModelM = Cfg<16, 1024, 16, 512, 100352>;
using ModelF = Cfg<4, 2048, 32, 512, 100352>;     // config F

struct Weights {  // all bf16, SPEC layout (W[in,out] row-major; wte[V,d]; wpe[T,d]); 16 B aligned
  const bf16 *wte, *wpe, *gf;
  const bf16 *wqkv[MAXL], *wo[MAXL], *wup[MAXL], *wdn[MAXL], *g1[MAXL], *g2[MAXL];
};

template <class C>
struct State {  // lives in device memory (cudaMalloc); see alloc_state
  // Residual stream in FIXED POINT (int64, 2^-30 units, two's complement in unsigned long long). Tasks add their partial
  // products with 64-bit integer atomics: integer addition is associative, so the result is bitwise independent of task order.
  // xw: the master copy (read by QKV and head; updated by ATTWO and UPDN).  xr: copy read by UPDN (updated by ATTWO only).
  // QKV task 0 refreshes xr := xw each layer. The SAMPLE task writes the embedding of the next token into xw.
  unsigned long long xw[C::D], xr[C::D];
  bf16 q[C::D];                     // current query (bf16)
  float cand_v[2][C::NCHUNK];       // per-chunk best (0: greedy, 1: gumbel)
  int cand_i[2][C::NCHUNK];
  int token, pos, step, ctx_full;
  int astep;                        // step within the current answer (answer mode, Params::answer_len > 0)
  bf16* kv;                         // [L][2][H][T][64]
  float* logits;                    // [V]
  int* out_tokens;                  // [out_cap] sampled token for each step
  int out_cap;
};

template <class C>
struct Params {
  Weights w;
  State<C>* st;
  float temperature;                // <= 0 : greedy
  unsigned long long seed;
  int valid_vocab;                  // tokens >= valid_vocab are never sampled (e.g. 100277)
  const int* forced;                // teacher forcing: token at step s+1 := forced[s+1] while s+1 < nforced
  int nforced;
  int auto_reset;                   // if 1: when pos reaches T, restart the context at pos 0 with the sampled token
  int answer_len;                   // > 0: answer mode. After the prompt (forced) and answer_len sampled tokens, ask again: restart at
                                    // pos 0 with forced[0], from the current (live) weights. out_tokens then marks prompt steps
                                    // (OUT_FORCED) and the start of each answer (OUT_RESTART) instead of recording discarded samples.
};
constexpr int OUT_RESTART = 1 << 24;   // out_tokens marker: a new answer starts (prompt about to be fed)
constexpr int OUT_FORCED = OUT_RESTART + 1;   // out_tokens marker: a prompt token was fed (no sample kept)

// phase index within a step: ph = 3*layer + {0: QKV, 1: ATTWO, 2: UPDN}; ph = 3L : HEAD; ph = 3L+1 : SAMPLE
template <class C> __host__ __device__ inline int phase_tasks(int ph) {
  if (ph < 3 * C::L) {
    switch (ph % 3) {
      case 0: return 3 * C::D / 64;
      case 1: return C::H * C::NCB;
      default: return C::NPD;
    }
  }
  if (ph == 3 * C::L) return C::NCHUNK;
  return 1;
}
inline const char* phase_name(int kind) {
  static const char* n[] = {"QKV(+res,LN1)", "ATT+WO", "UP(+res,LN2)+DOWN"};
  return n[kind];
}

// ---------------------------------------------------------------- small helpers
// 16 B streaming load for weights / KV: weak load that does not allocate in L1 (so it can never hit a stale L1 line
// even if the optimizer refreshed the weights earlier in the same kernel; not .nc for the same reason).
__device__ __forceinline__ uint4 ldw(const void* p) {
  uint4 r;
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}
__device__ __forceinline__ float lo_f(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_f(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }
__device__ __forceinline__ float ldcgf(const float* p) { return __ldcg(p); }
__device__ __forceinline__ float ldcgb(const bf16* p) { return __bfloat162float(__ldcg(p)); }

__device__ __forceinline__ float block_sum(float v, float* scratch) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) scratch[w] = v;
  __syncthreads();
  float t = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) t += scratch[i];
  __syncthreads();
  return t;
}
__device__ __forceinline__ float block_max(float v, float* scratch) {
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) scratch[w] = v;
  __syncthreads();
  float t = scratch[0];
#pragma unroll
  for (int i = 1; i < 8; i++) t = fmaxf(t, scratch[i]);
  __syncthreads();
  return t;
}
// (value, index) argmax, ties -> smaller index; NaN loses.
__device__ __forceinline__ bool better(float a, int ai, float b, int bi) {
  if (a != a) return false;
  if (b != b) return true;
  return a > b || (a == b && ai < bi);
}
__device__ __forceinline__ void block_argmax(float& v, int& idx, float* sv, int* si) {
  for (int o = 16; o; o >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, v, o);
    int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (better(ov, oi, v, idx)) { v = ov; idx = oi; }
  }
  int w = threadIdx.x >> 5, l = threadIdx.x & 31;
  if (l == 0) { sv[w] = v; si[w] = idx; }
  __syncthreads();
  v = sv[0]; idx = si[0];
#pragma unroll
  for (int i = 1; i < 8; i++)
    if (better(sv[i], si[i], v, idx)) { v = sv[i]; idx = si[i]; }
  __syncthreads();
}


// Overlay 005: fuse two independent argmax reductions.
// Comparison order and tie-breaking are preserved.
__device__ __forceinline__ void block_argmax_pair(
    float& a, int& ai, float& b, int& bi,
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
  if (l == 0) {
    sva[w] = a; sia[w] = ai;
    svb[w] = b; sib[w] = bi;
  }
  __syncthreads();
  a = sva[0]; ai = sia[0];
  b = svb[0]; bi = sib[0];
#pragma unroll
  for (int i = 1; i < 8; i++) {
    if (better(sva[i], sia[i], a, ai)) { a = sva[i]; ai = sia[i]; }
    if (better(svb[i], sib[i], b, bi)) { b = svb[i]; bi = sib[i]; }
  }
  __syncthreads();
}

__device__ __forceinline__ uint64_t splitmix64(uint64_t z) {
  z += 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}
// counter-based Gumbel(0,1) noise for (seed, step, vocab index)
__device__ __forceinline__ float gumbel(uint64_t seed, uint32_t step, uint32_t v) {
  uint64_t r = splitmix64(splitmix64(seed ^ ((uint64_t)step << 32)) + v);
  float u = ((float)(uint32_t)(r >> 40) + 0.5f) * (1.0f / 16777216.0f);
  return -logf(-logf(u));
}

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }
constexpr float FX_SCALE = 1073741824.0f;             // 2^30
constexpr float FX_INV = 1.0f / 1073741824.0f;
__device__ __forceinline__ unsigned long long fx(float v) { return (unsigned long long)llrintf(v * FX_SCALE); }
__device__ __forceinline__ float unfx(unsigned long long u) { return (float)(long long)u * FX_INV; }

// ---------------------------------------------------------------- shared memory budget
template <class C> __host__ __device__ constexpr size_t smem_bytes() {
  size_t aw = (64 + C::T + 2048 + 32 + 64) * 4;                 // q, scores, reduce, scratch, o
  size_t ud = (C::D + 2048 + 64 + 16 + 64) * 4;                 // xs, reduce, res, scratch, gelu
  size_t hd = (C::D + 32 + C::HR) * 4;
  size_t m = aw > ud ? aw : ud;
  return m > hd ? m : hd;
}

// ---------------------------------------------------------------- GEMV core
// res[c] = sum_{k<nk} xs[k] * W[(k0+k)*ld + c0 + c],  c in [0,64).  xs: smem floats.
// thread = (rg = tid/8 : row group 0..31, cg = tid%8 : 8 columns = one 16 B load). Rows are streamed 32 per round; the first
// batch of UNR rows per thread is issued by gemv_issue() BEFORE the (weight independent) prologue so DRAM latency overlaps it.
// Reduction over the 32 row groups in a fixed order via smem. red: [32*64] floats, res: [64]. Ends with a sync (res valid).
template <int UNR>
__device__ __forceinline__ void gemv_issue(const bf16* __restrict__ W, int ld, int c0, int k0, int nk, uint4 (&w)[UNR]) {
  const int tid = threadIdx.x, rg = tid >> 3, cg = tid & 7;
  const bf16* base = W + (size_t)k0 * ld + c0 + cg * 8;
#pragma unroll
  for (int u = 0; u < UNR; u++) {
    int r = rg + 32 * u;
    if (r < nk) w[u] = ldw(base + (size_t)r * ld);
  }
}
// Request the remaining rows of the tile (one 128 B line per row) from DRAM into L2 right away, so the later batches hit L2.
template <int NK>
__device__ __forceinline__ void gemv_l2_prefetch(const bf16* __restrict__ W, int ld, int c0, int first_row) {
  for (int r = first_row + threadIdx.x; r < NK; r += NTHREADS) {
    const bf16* p = W + (size_t)r * ld + c0;
    asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
  }
}
template <int UNR>
__device__ __forceinline__ void gemv_consume(const bf16* __restrict__ W, int ld, int c0, int k0, int nk, uint4 (&w)[UNR],
                                             const float* xs, float* red, float* res) {
  const int tid = threadIdx.x, rg = tid >> 3, cg = tid & 7;
  float acc[8];
#pragma unroll
  for (int j = 0; j < 8; j++) acc[j] = 0.f;
  auto fma8 = [&](const uint4& v, float x) {
    acc[0] += x * lo_f(v.x); acc[1] += x * hi_f(v.x); acc[2] += x * lo_f(v.y); acc[3] += x * hi_f(v.y);
    acc[4] += x * lo_f(v.z); acc[5] += x * hi_f(v.z); acc[6] += x * lo_f(v.w); acc[7] += x * hi_f(v.w);
  };
  const bf16* base = W + (size_t)k0 * ld + c0 + cg * 8;
  // rows beyond the first batch (only when nk > 32*UNR)
  for (int r0 = rg + 32 * UNR; r0 < nk; r0 += 32 * UNR) {
    uint4 v[UNR];
#pragma unroll
    for (int u = 0; u < UNR; u++) { int r = r0 + 32 * u; if (r < nk) v[u] = ldw(base + (size_t)r * ld); }
#pragma unroll
    for (int u = 0; u < UNR; u++) { int r = r0 + 32 * u; if (r < nk) fma8(v[u], xs[r]); }
  }
#pragma unroll
  for (int u = 0; u < UNR; u++) { int r = rg + 32 * u; if (r < nk) fma8(w[u], xs[r]); }
#pragma unroll
  for (int j = 0; j < 8; j++) red[rg * 64 + cg * 8 + j] = acc[j];
  __syncthreads();
  if (tid < 64) {
    float s = 0.f;
#pragma unroll
    for (int g = 0; g < 32; g++) s += red[g * 64 + tid];
    res[tid] = s;
  }
  __syncthreads();
}

// ---------------------------------------------------------------- LayerNorm prologue
// xs[k] = bf16( LN(x)[k] * gamma[k] ), x = fixed-point residual `src` converted to fp32. If `copy_to` != null, the task also stores
// src -> copy_to (used by exactly one QKV task per layer to refresh xr). One block reduction (shifted sum / sum of squares).
template <class C>
__device__ __forceinline__ void prologue_ln(const unsigned long long* src, const bf16* gamma, float* xs, float* scratch,
                                            unsigned long long* copy_to) {
  constexpr int NE = C::D / NTHREADS;
  const int tid = threadIdx.x;
  unsigned long long raw[NE];
  float g[NE];
#pragma unroll
  for (int i = 0; i < NE; i++) { raw[i] = __ldcg(&src[tid + i * NTHREADS]); g[i] = ldcgb(gamma + tid + i * NTHREADS); }
  const float xref = unfx(__ldcg(&src[0]));
  float x[NE], s1 = 0.f, s2 = 0.f;
#pragma unroll
  for (int i = 0; i < NE; i++) {
    x[i] = unfx(raw[i]);
    float d = x[i] - xref;
    s1 += d; s2 += d * d;
    if (copy_to) copy_to[tid + i * NTHREADS] = raw[i];
  }
  for (int o = 16; o; o >>= 1) { s1 += __shfl_xor_sync(0xffffffffu, s1, o); s2 += __shfl_xor_sync(0xffffffffu, s2, o); }
  if ((tid & 31) == 0) { scratch[tid >> 5] = s1; scratch[8 + (tid >> 5)] = s2; }
  __syncthreads();
  float t1 = 0.f, t2 = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) { t1 += scratch[i]; t2 += scratch[8 + i]; }
  const float m1 = t1 * (1.0f / C::D);
  const float var = fmaxf(t2 * (1.0f / C::D) - m1 * m1, 0.f);
  const float mean = xref + m1;
  const float rstd = rsqrtf(var + 1e-5f);
#pragma unroll
  for (int i = 0; i < NE; i++) xs[tid + i * NTHREADS] = bfr((x[i] - mean) * rstd * g[i]);
  __syncthreads();
}

// ---------------------------------------------------------------- phase A of a layer: residual merge + LN1 + QKV GEMV (+ KV append)
// tasks: 3D/64 (36 for S). task t: columns [64t, 64t+64) of [q|k|v]. Needs: previous step's SAMPLE (layer 0) / previous layer's UPDN.
// Outputs: st->q (t < H) or the kv cache row `pos` for (layer, k|v, head).
template <class C>
__device__ __forceinline__ void task_qkv(const Params<C>& P, int layer, int t, char* smem) {
  State<C>* st = P.st;
  constexpr int GU = 8;
  float* xs = (float*)smem;
  float* red = xs + C::D;
  float* res = red + 2048;
  float* scratch = res + 64;
  uint4 w[GU];
  DTRACE(t, 0);
  gemv_issue<GU>(P.w.wqkv[layer], 3 * C::D, t * 64, 0, C::D, w);
  gemv_l2_prefetch<C::D>(P.w.wqkv[layer], 3 * C::D, t * 64, GU * 32);
  prologue_ln<C>(st->xw, P.w.g1[layer], xs, scratch, t == 0 ? st->xr : nullptr);
  DTRACE(t, 1);
  gemv_consume<GU>(P.w.wqkv[layer], 3 * C::D, t * 64, 0, C::D, w, xs, red, res);
  DTRACE(t, 2);
  if (threadIdx.x < 64) {
    int c = threadIdx.x;
    bf16 o = __float2bfloat16(res[c]);
    if (t < C::H) {
      st->q[t * 64 + c] = o;
    } else {
      int which = (t - C::H) / C::H, head = (t - C::H) % C::H;
      int pos = __ldcg(&st->pos);
      st->kv[((((size_t)layer * 2 + which) * C::H + head) * C::T + pos) * 64 + c] = o;
    }
  }
}

// ---------------------------------------------------------------- phase B: attention (one head) fused with its Wo slice
// tasks: H*NCB (36 for S): head = t / NCB, column block cb = t % NCB (256 output columns). The softmax attention of the head is
// recomputed by each of its NCB tasks (reads <= 2*T*128 B of KV, L2 resident); then o[64] . Wo[64*head .. +64, cb*256 .. +256]
// -> added (fixed point, integer atomics) to the residual xr and xw. Needs QKV phase done and UPDN of the previous layer.
template <class C>
__device__ __forceinline__ void task_attwo(const Params<C>& P, int layer, int t, char* smem) {
  State<C>* st = P.st;
  constexpr int UK = 8;  // keys per group per round -> 256 keys per round
  float* qs = (float*)smem;
  float* sc = qs + 64;
  float* red = sc + C::T;
  float* scratch = red + 2048;
  float* ob = scratch + 32;
  const int tid = threadIdx.x, g = tid >> 3, l8 = tid & 7;
  const int head = t / C::NCB, cb = t % C::NCB;
  const int rg8 = tid >> 5, cc = tid & 31;
  // Wo slice -> registers (independent of everything in this step)
  uint4 wo[8];
  {
    const bf16* base = P.w.wo[layer] + (size_t)(head * 64) * C::D + cb * 256 + cc * 8;
#pragma unroll
    for (int i = 0; i < 8; i++) wo[i] = ldw(base + (size_t)(rg8 + 8 * i) * C::D);
  }
  const int n = __ldcg(&st->pos) + 1;
  {  // V rows of this head into L2 early (K loads are issued right below)
    const bf16* Vp = st->kv + (((size_t)layer * 2 + 1) * C::H + head) * C::T * 64;
    for (int j = tid; j < n; j += NTHREADS) asm volatile("prefetch.global.L2 [%0];" ::"l"(Vp + (size_t)j * 64));
  }
  if (tid < 64) qs[tid] = ldcgb(&st->q[head * 64 + tid]) * 0.125f;
  __syncthreads();
  const bf16* Kc = st->kv + (((size_t)layer * 2 + 0) * C::H + head) * C::T * 64;
  const bf16* Vc = st->kv + (((size_t)layer * 2 + 1) * C::H + head) * C::T * 64;
  float qv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) qv[i] = qs[l8 * 8 + i];
  for (int jb = 0; jb < n; jb += 32 * UK) {
    uint4 kw[UK];
#pragma unroll
    for (int u = 0; u < UK; u++) { int j = jb + g + 32 * u; kw[u] = ldw(Kc + (size_t)(j < n ? j : 0) * 64 + l8 * 8); }
#pragma unroll
    for (int u = 0; u < UK; u++) {
      int j = jb + g + 32 * u;
      float d = qv[0] * lo_f(kw[u].x) + qv[1] * hi_f(kw[u].x) + qv[2] * lo_f(kw[u].y) + qv[3] * hi_f(kw[u].y) +
                qv[4] * lo_f(kw[u].z) + qv[5] * hi_f(kw[u].z) + qv[6] * lo_f(kw[u].w) + qv[7] * hi_f(kw[u].w);
      d += __shfl_xor_sync(0xffffffffu, d, 1);
      d += __shfl_xor_sync(0xffffffffu, d, 2);
      d += __shfl_xor_sync(0xffffffffu, d, 4);
      if (l8 == 0 && j < n) sc[j] = d;
    }
  }
  // first V batch in flight during the softmax reductions
  uint4 vb[UK];
#pragma unroll
  for (int u = 0; u < UK; u++) { int j = g + 32 * u; vb[u] = ldw(Vc + (size_t)(j < n ? j : 0) * 64 + l8 * 8); }
  __syncthreads();
  float mx = -INFINITY;
  for (int j = tid; j < n; j += NTHREADS) mx = fmaxf(mx, sc[j]);
  mx = block_max(mx, scratch);
  float sm = 0.f;
  for (int j = tid; j < n; j += NTHREADS) { float e = expf(sc[j] - mx); sc[j] = e; sm += e; }
  sm = block_sum(sm, scratch);  // also syncs
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; i++) acc[i] = 0.f;
  for (int jb = 0; jb < n; jb += 32 * UK) {
    if (jb > 0) {
#pragma unroll
      for (int u = 0; u < UK; u++) { int j = jb + g + 32 * u; vb[u] = ldw(Vc + (size_t)(j < n ? j : 0) * 64 + l8 * 8); }
    }
#pragma unroll
    for (int u = 0; u < UK; u++) {
      int j = jb + g + 32 * u;
      if (j < n) {
        float p = sc[j];
        acc[0] += p * lo_f(vb[u].x); acc[1] += p * hi_f(vb[u].x);
        acc[2] += p * lo_f(vb[u].y); acc[3] += p * hi_f(vb[u].y);
        acc[4] += p * lo_f(vb[u].z); acc[5] += p * hi_f(vb[u].z);
        acc[6] += p * lo_f(vb[u].w); acc[7] += p * hi_f(vb[u].w);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; i++) red[g * 64 + l8 * 8 + i] = acc[i];
  __syncthreads();
  if (tid < 64) {
    float o = 0.f;
#pragma unroll
    for (int gg = 0; gg < 32; gg++) o += red[gg * 64 + tid];
    ob[tid] = bfr(o / sm);
  }
  __syncthreads();
  // o[64] . Wo slice : thread (rg8 = row group of 8, cc = 8 columns)
  float a2[8];
#pragma unroll
  for (int j = 0; j < 8; j++) a2[j] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    float x = ob[rg8 + 8 * i];
    a2[0] += x * lo_f(wo[i].x); a2[1] += x * hi_f(wo[i].x); a2[2] += x * lo_f(wo[i].y); a2[3] += x * hi_f(wo[i].y);
    a2[4] += x * lo_f(wo[i].z); a2[5] += x * hi_f(wo[i].z); a2[6] += x * lo_f(wo[i].w); a2[7] += x * hi_f(wo[i].w);
  }
#pragma unroll
  for (int j = 0; j < 8; j++) red[rg8 * 256 + cc * 8 + j] = a2[j];
  __syncthreads();
  float r = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) r += red[i * 256 + tid];
  { unsigned long long u = fx(r); atomicAdd(&st->xr[cb * 256 + tid], u); atomicAdd(&st->xw[cb * 256 + tid], u); }
}

// ---------------------------------------------------------------- phase C: residual + LN2 + up GEMV + GELU + down slice
// tasks: 4D/64 (48 for S). task t: hidden units [64t, 64t+64): h = GELU(ln2 . Wup[:, 64t..]) (kept in smem, bf16-rounded), then
// h . Wdn[64t .. 64t+64, :] -> added to the residual xw (integer atomics). Needs ATTWO done.
template <class C>
__device__ __forceinline__ void task_updn(const Params<C>& P, int layer, int t, char* smem) {
  State<C>* st = P.st;
  constexpr int GU = 8;
  float* xs = (float*)smem;
  float* red = xs + C::D;
  float* res = red + 2048;
  float* scratch = res + 64;
  float* hb = scratch + 16;
  const int tid = threadIdx.x;
  uint4 w[GU];
  gemv_issue<GU>(P.w.wup[layer], C::FF, t * 64, 0, C::D, w);
  gemv_l2_prefetch<C::D>(P.w.wup[layer], C::FF, t * 64, GU * 32);
  {  // pull the down slice into L2 meanwhile: 64 rows x D*2 bytes
    constexpr int LPR = C::D * 2 / 128;  // 128 B lines per row
    for (int i = tid; i < 64 * LPR; i += NTHREADS) {
      const bf16* p = P.w.wdn[layer] + (size_t)(t * 64 + i / LPR) * C::D + (i % LPR) * 64;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
    }
  }
  prologue_ln<C>(st->xr, P.w.g2[layer], xs, scratch, nullptr);
  gemv_consume<GU>(P.w.wup[layer], C::FF, t * 64, 0, C::D, w, xs, red, res);
  if (tid < 64) {
    float x = res[tid];
    float gl = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    hb[tid] = bfr(gl);
  }
  __syncthreads();
  const int rg8 = tid >> 5, cc = tid & 31;
  constexpr int NJ = C::D / 256;
  uint4 dw[8], dn[8];
  auto ldd = [&](uint4 (&dst)[8], int jb) {
#pragma unroll
    for (int i = 0; i < 8; i++) dst[i] = ldw(P.w.wdn[layer] + (size_t)(t * 64 + rg8 + 8 * i) * C::D + jb * 256 + cc * 8);
  };
  ldd(dw, 0);
  float hv[8];
#pragma unroll
  for (int i = 0; i < 8; i++) hv[i] = hb[rg8 + 8 * i];
#pragma unroll
  for (int jb = 0; jb < NJ; jb++) {
    if (jb + 1 < NJ) ldd(dn, jb + 1);
    float a2[8];
#pragma unroll
    for (int j = 0; j < 8; j++) a2[j] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float x = hv[i];
      a2[0] += x * lo_f(dw[i].x); a2[1] += x * hi_f(dw[i].x); a2[2] += x * lo_f(dw[i].y); a2[3] += x * hi_f(dw[i].y);
      a2[4] += x * lo_f(dw[i].z); a2[5] += x * hi_f(dw[i].z); a2[6] += x * lo_f(dw[i].w); a2[7] += x * hi_f(dw[i].w);
    }
#pragma unroll
    for (int j = 0; j < 8; j++) red[rg8 * 256 + cc * 8 + j] = a2[j];
    __syncthreads();
    float r = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) r += red[i * 256 + tid];
    atomicAdd(&st->xw[jb * 256 + tid], fx(r));
    __syncthreads();
    if (jb + 1 < NJ) {
#pragma unroll
      for (int i = 0; i < 8; i++) dw[i] = dn[i];
    }
  }
}

// ---------------------------------------------------------------- LM head over tied wte[V,D]
// tasks: V/HR (784 for HR=128). Prologue: final residual merge + LN_f. Then logits[v] = ln . wte[v,:] (row v contiguous: a warp
// streams rows, lanes take 16 B chunks, shuffle reduction; next row batch is loaded while the current one is computed).
// Writes logits fp32 and chunk-local best (greedy, and Gumbel-max for temperature sampling). Needs the last UPDN done.
template <class C>
__device__ __forceinline__ void task_head(const Params<C>& P, int t, char* smem) {
  State<C>* st = P.st;
  constexpr int NCH = C::D / 8 / 32;  // 16 B chunks per lane per row (3 for S)
  constexpr int RB = 2;               // rows per batch per warp
  constexpr int RPW = C::HR / 8;      // rows per warp
  constexpr int NB = RPW / RB;
  static_assert(RPW % RB == 0, "HR");
  float* xs = (float*)smem;
  float* scratch = xs + C::D;         // 16
  float* lg = scratch + 32;           // HR
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int vbase = t * C::HR + w * RPW;
  uint4 cur[RB][NCH], nxt[RB][NCH];
  auto load = [&](uint4 (&dst)[RB][NCH], int b) {
#pragma unroll
    for (int r = 0; r < RB; r++)
#pragma unroll
      for (int i = 0; i < NCH; i++) dst[r][i] = ldw(P.w.wte + (size_t)(vbase + b * RB + r) * C::D + (lane + 32 * i) * 8);
  };
  load(cur, 0);
  prologue_ln<C>(st->xw, P.w.gf, xs, scratch, nullptr);
#pragma unroll
  for (int b = 0; b < NB; b++) {
    if (b + 1 < NB) load(nxt, b + 1);
    float s[RB];
#pragma unroll
    for (int r = 0; r < RB; r++) s[r] = 0.f;
#pragma unroll
    for (int i = 0; i < NCH; i++) {
      const float4 a0 = *reinterpret_cast<const float4*>(&xs[(lane + 32 * i) * 8]);
      const float4 a1 = *reinterpret_cast<const float4*>(&xs[(lane + 32 * i) * 8 + 4]);
#pragma unroll
      for (int r = 0; r < RB; r++)
        s[r] += a0.x * lo_f(cur[r][i].x) + a0.y * hi_f(cur[r][i].x) + a0.z * lo_f(cur[r][i].y) + a0.w * hi_f(cur[r][i].y) +
                a1.x * lo_f(cur[r][i].z) + a1.y * hi_f(cur[r][i].z) + a1.z * lo_f(cur[r][i].w) + a1.w * hi_f(cur[r][i].w);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1)
#pragma unroll
      for (int r = 0; r < RB; r++) s[r] += __shfl_xor_sync(0xffffffffu, s[r], o);
    if (lane == 0)
#pragma unroll
      for (int r = 0; r < RB; r++) lg[w * RPW + b * RB + r] = s[r];
    if (b + 1 < NB) {
#pragma unroll
      for (int r = 0; r < RB; r++)
#pragma unroll
        for (int i = 0; i < NCH; i++) cur[r][i] = nxt[r][i];
    }
  }
  __syncthreads();
  // chunk-local candidates
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
}

// xw := fixed point (wte[tok] + wpe[pos]); all 256 threads.
template <class C>
__device__ __forceinline__ void write_embed(const Params<C>& P, int tok, int pos) {
  for (int k = threadIdx.x; k < C::D; k += NTHREADS)
    P.st->xw[k] = fx(__bfloat162float(P.w.wte[(size_t)tok * C::D + k]) + __bfloat162float(P.w.wpe[(size_t)pos * C::D + k]));
}
template <class C>
__global__ void k_init_embed(Params<C> P) { write_embed<C>(P, P.st->token, P.st->pos); }

// ---------------------------------------------------------------- sampling: deterministic final reduction
// 1 task. Picks the next token (greedy or Gumbel-max), applies teacher forcing, advances pos/step, records out_tokens.
template <class C>
__device__ __forceinline__ void task_sample(const Params<C>& P, char* smem) {
  State<C>* st = P.st;
  float* sv = (float*)smem;
  int* si = (int*)(sv + 8);
  int* sn = si + 8;
  const int tid = threadIdx.x;
  const int mode = P.temperature > 0.f ? 1 : 0;
  float bv = -INFINITY; int bi = 0x7fffffff;
  for (int c = tid; c < C::NCHUNK; c += NTHREADS) {
    float v = __ldcg(&st->cand_v[mode][c]);
    int i = __ldcg(&st->cand_i[mode][c]);
    if (better(v, i, bv, bi)) { bv = v; bi = i; }
  }
  block_argmax(bv, bi, sv, si);
  if (tid == 0) {
    int step = st->step, pos = st->pos;
    int next = bi;
    if (P.answer_len > 0) {                      // answer mode: the prompt is fed again before every answer
      int astep = st->astep;
      const bool forcing = astep + 1 < P.nforced;
      if (forcing) next = P.forced[astep + 1];
      if (st->out_tokens && st->out_cap > 0) st->out_tokens[step % st->out_cap] = forcing ? (astep == 0 ? OUT_RESTART : OUT_FORCED) : bi;
      pos += 1; astep += 1;
      if (astep >= P.nforced - 1 + P.answer_len) { next = P.forced[0]; pos = 0; astep = 0; }   // answer done: ask again
      st->astep = astep;
    } else {
      if (step + 1 < P.nforced) next = P.forced[step + 1];
      if (st->out_tokens && st->out_cap > 0) st->out_tokens[step % st->out_cap] = bi;
      pos += 1;
    }
    st->ctx_full = (pos >= C::T);
    if (pos >= C::T && P.auto_reset) pos = 0;
    st->token = next; st->pos = pos; st->step = step + 1;
    sn[0] = next; sn[1] = pos;
  }
  __syncthreads();
  write_embed<C>(P, sn[0], sn[1]);
}

// ---------------------------------------------------------------- dispatcher
// Phase p+1 may start only when ALL tasks of phase p are complete (and across steps: phase 0 of step s+1 after SAMPLE of step s).
// `smem`: 128 B aligned, smem_bytes<C>() bytes. All 256 threads call; the caller must __syncthreads() before reusing smem.
template <class C>
__device__ __forceinline__ void run_task(const Params<C>& P, int ph, int task, char* smem) {
  if (ph < 3 * C::L) {
    int l = ph / 3;
    switch (ph % 3) {
      case 0: task_qkv<C>(P, l, task, smem); break;
      case 1: task_attwo<C>(P, l, task, smem); break;
      default: task_updn<C>(P, l, task, smem); break;
    }
  } else if (ph == 3 * C::L) {
    task_head<C>(P, task, smem);
  } else {
    task_sample<C>(P, smem);
  }
}

// ---------------------------------------------------------------- optional: L2 prefetch of a task's weights
// Weights do not depend on activations: a block can pull the weights of a task it will run later (next claimed task, or the
// next phase's tasks while waiting at the barrier) into L2 with prefetch.global.L2. Hint only. All 256 threads call.
template <class C>
__device__ __forceinline__ void prefetch_task(const Params<C>& P, int ph, int task) {
  const int tid = threadIdx.x;
  auto pf_rows = [&](const bf16* W, int ld, int c0, int k0, int nk) {  // one 128 B line per row
    for (int r = tid; r < nk; r += NTHREADS) {
      const bf16* p = W + (size_t)(k0 + r) * ld + c0;
      asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
    }
  };
  if (ph < 3 * C::L) {
    int l = ph / 3;
    switch (ph % 3) {
      case 0: pf_rows(P.w.wqkv[l], 3 * C::D, task * 64, 0, C::D); break;
      case 1: {
        int head = task / C::NCB, cb = task % C::NCB;  // 64 rows x 512 B
        for (int i = tid; i < 64 * 4; i += NTHREADS) {
          const bf16* p = P.w.wo[l] + (size_t)(head * 64 + i / 4) * C::D + cb * 256 + (i % 4) * 64;
          asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
        }
      } break;
      default: pf_rows(P.w.wup[l], C::FF, task * 64, 0, C::D); break;  // (down slice is prefetched inside the task)
    }
  } else if (ph == 3 * C::L) {
    const char* base = (const char*)(P.w.wte + (size_t)task * C::HR * C::D);
    for (int o = tid * 128; o < C::HR * C::D * 2; o += NTHREADS * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(base + o));
  }
}

// ---------------------------------------------------------------- reference kernels (plain grid) and persistent kernel
template <class C>
__global__ void __launch_bounds__(NTHREADS, 2) k_phase(Params<C> P, int ph) {
  extern __shared__ __align__(128) char smem_raw[];
  run_task<C>(P, ph, blockIdx.x, smem_raw);
}

// Persistent: blocks claim task indices of a phase with an atomic ticket (always one ticket ahead, so the next claim and the
// L2 prefetch of the next task overlap the current one), optionally scrambled (order independence test), and wait on a per-phase
// completion counter. `nsteps` decode steps run in ONE launch. counters next/done: [nsteps*PHASES], zeroed.
// Only blocks on SMs < sm_limit with ticket < per_sm participate (to emulate a partition of the GPU); others exit at once.
// prefetch bit0: L2-prefetch the claimed-ahead task; bit1: while waiting at the barrier, prefetch next phase's tasks (by rank).
template <class C>
__global__ void __launch_bounds__(NTHREADS, 2)
k_persistent(Params<C> P, int nsteps, unsigned* next, unsigned* done, int sm_limit, int per_sm, unsigned* sm_tickets,
             int scramble, int prefetch, int nactive, unsigned long long* ts = nullptr) {
  extern __shared__ __align__(128) char smem_raw[];
  __shared__ int s_a, s_b;
  const int tid = threadIdx.x;
  if (tid == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned tk = atomicAdd(&sm_tickets[smid], 1u);
    s_a = (smid < (unsigned)sm_limit && tk < (unsigned)per_sm) ? 1 : 0;
    if (s_a) s_b = (int)atomicAdd(&sm_tickets[1024], 1u);  // block rank among active blocks
  }
  __syncthreads();
  if (!s_a) return;
  const unsigned my = (unsigned)s_b;
  auto map_task = [&](unsigned i, unsigned nt, int gp) -> unsigned {
    if (!scramble) return i;
    unsigned mul = 2u * (unsigned)(gp * 7 + 3) + 1u;  // bijective scramble: multiply by a number coprime to nt (+ offset)
    while (true) { unsigned a = mul, b = nt, r; while (b) { r = a % b; a = b; b = r; } if (a == 1 || nt == 1) break; mul += 2; }
    return (unsigned)(((unsigned long long)i * mul + (unsigned)gp * 5u) % nt);
  };
  for (int step = 0; step < nsteps; step++) {
    for (int ph = 0; ph < C::PHASES; ph++) {
      const int gp = step * C::PHASES + ph;
      const unsigned nt = (unsigned)phase_tasks<C>(ph);
      unsigned my_done = 0;
      // First task = block rank (no atomic). Further tasks (only if the phase has more tasks than blocks) via ticket
      // nactive + atomicAdd; for big phases one ticket is claimed ahead so the claim latency and the L2 prefetch of the next
      // task overlap the current one (not done for small phases: a reserved task would sit behind a running one).
      const bool dyn = nt > (unsigned)nactive;
      const bool ahead = nt > 2u * (unsigned)nactive;
      unsigned cur = my < nt ? my : nt, nxt = nt, t_nxt = nt;
      if (ahead) {
        if (tid == 0) s_a = (int)((unsigned)nactive + atomicAdd(&next[gp], 1u));
        __syncthreads();
        nxt = (unsigned)s_a;
        __syncthreads();
      }
      while (cur < nt) {
        if (tid == 0 && ahead) t_nxt = (unsigned)nactive + atomicAdd(&next[gp], 1u);  // consumed after the task
        if ((prefetch & 1) && nxt < nt) prefetch_task<C>(P, ph, (int)map_task(nxt, nt, gp));
        run_task<C>(P, ph, (int)map_task(cur, nt, gp), smem_raw);
        my_done++;
        if (scramble && ((cur + my + gp) % 5u == 0u)) __nanosleep(2000 + 300 * (my & 7));
        if (!dyn) break;
        if (tid == 0) {
          if (ahead) { s_a = (int)nxt; s_b = (int)t_nxt; }
          else { s_a = (int)((unsigned)nactive + atomicAdd(&next[gp], 1u)); s_b = (int)nt; }
        }
        __syncthreads();
        cur = (unsigned)s_a; nxt = (unsigned)s_b;
        __syncthreads();
      }
      __syncthreads();  // all threads' task writes precede thread 0's release below
      if (tid == 0 && my_done) asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(&done[gp]), "r"(my_done) : "memory");
      // wait for phase completion; meanwhile pull the next phase's weights into L2 (static slice per block rank)
      if ((prefetch & 2) && ph + 1 < C::PHASES) {
        const unsigned nnt = (unsigned)phase_tasks<C>(ph + 1);
        for (unsigned j = my; j < nnt; j += (unsigned)nactive) prefetch_task<C>(P, ph + 1, (int)j);
      }
      if (tid == 0) {
        // poll with exponential backoff: hundreds of blocks hammering one L2 line would delay the very atomics they wait for
        unsigned sl = DECODE_POLL_MIN_NS;
        while (ld_acquire(&done[gp]) < nt) { __nanosleep(sl); sl = min(sl * 2u, (unsigned)DECODE_POLL_MAX_NS); }
        if (ts && my == 0) { unsigned long long tt; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(tt)); ts[gp] = tt; }
      }
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------- host helpers
template <class C>
inline State<C>* alloc_state(int out_cap) {
  State<C>* d;
  cudaMalloc(&d, sizeof(State<C>));
  cudaMemset(d, 0, sizeof(State<C>));
  bf16* kv; float* lg; int* ot;
  cudaMalloc(&kv, sizeof(bf16) * (size_t)C::L * 2 * C::H * C::T * 64);
  cudaMemset(kv, 0, sizeof(bf16) * (size_t)C::L * 2 * C::H * C::T * 64);
  cudaMalloc(&lg, sizeof(float) * C::V);
  cudaMalloc(&ot, sizeof(int) * out_cap);
  cudaMemset(ot, 0, sizeof(int) * out_cap);
  cudaMemcpy(&d->kv, &kv, sizeof(kv), cudaMemcpyHostToDevice);
  cudaMemcpy(&d->logits, &lg, sizeof(lg), cudaMemcpyHostToDevice);
  cudaMemcpy(&d->out_tokens, &ot, sizeof(ot), cudaMemcpyHostToDevice);
  cudaMemcpy(&d->out_cap, &out_cap, sizeof(int), cudaMemcpyHostToDevice);
  return d;
}
// (re)start the context: token = tok at position `pos` (default 0), step=0, and embed the token into the residual stream.
// KV cache needs no clearing: only rows <= pos are ever read. Synchronous; call between decode launches.
template <class C>
inline void begin_context(const Params<C>& P, int tok, int pos = 0) {
  State<C>* d = P.st;
  int z = 0;
  cudaMemcpy(&d->token, &tok, 4, cudaMemcpyHostToDevice);
  cudaMemcpy(&d->pos, &pos, 4, cudaMemcpyHostToDevice);
  cudaMemcpy(&d->step, &z, 4, cudaMemcpyHostToDevice);
  cudaMemcpy(&d->ctx_full, &z, 4, cudaMemcpyHostToDevice);
  cudaMemcpy(&d->astep, &z, 4, cudaMemcpyHostToDevice);
  k_init_embed<C><<<1, NTHREADS>>>(P);
  cudaDeviceSynchronize();
}
// one decode step as a plain-grid sequence of kernel launches (phase ph -> grid = tasks)
template <class C>
inline void launch_step_grid(const Params<C>& P, cudaStream_t s) {
  for (int ph = 0; ph < C::PHASES; ph++) k_phase<C><<<phase_tasks<C>(ph), NTHREADS, smem_bytes<C>(), s>>>(P, ph);
}

}  // namespace decode
