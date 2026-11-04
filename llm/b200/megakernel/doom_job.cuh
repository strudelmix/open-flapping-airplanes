// doom_job.cuh : the Doom job for the open-flapping-airplanes megakernel (device side + host setup). Header-only so it can be merged into any version of the kernel:
//   * include after the library headers, add `DoomCtx dm; Live* live;` to the kernel's Ctx,
//   * in the kernel entry, on the Doom partition's blocks:  if (C.dm.on) doom_loop(C.dm);
//   * host: doom_host_setup(...) fills DoomCtx and the pinned frame/counter buffers.
// Task 0 of frame f runs ONE Doom tic with all 256 threads of the block (thread 0 = the game, 255 workers draw queued columns/spans
// behind one named barrier); tasks 1..RT copy the 8-bit screen + palette into the pinned frame buffer. 35 Hz pacing starts after the
// engine booted. Doom is compiled for the GPU by doom-gpu/ (clang NVPTX) and linked as ~/doom-gpu/doom_dev.o (nvcc -rdc=true).
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

typedef unsigned long long u64;
extern "C" __device__ void gpu_doom_setup(const unsigned char*, long, volatile char*, long);
extern "C" __device__ void gpu_doom_create(unsigned long long);
extern "C" __device__ void gpu_doom_frame(unsigned long long);
extern "C" __device__ void gpu_doom_worker_always(int);
extern "C" __device__ void gpu_doom_enable_par(int);
extern "C" __device__ unsigned char* I_VideoBuffer;
extern "C" __device__ unsigned int colors[256];

constexpr int GW = 320, GH = 200, RT = 5, TPG = 1 + RT, FBSZ = GW * GH + 1024;     // frame = 64000 palette indices + 256 x RGBA palette
constexpr long long FRAME_NS = 28571429ll;                                          // Doom runs at 35 Hz
constexpr int DOOM_MAX_FRAMES = 1 << 17;

struct Live { volatile unsigned frames, steps; volatile float loss; volatile u64 decode_tokens; };               // pinned, host-visible counters

struct DoomCtx {
  int on; int* dctr; unsigned* ticket; u64* epoch; u64* frame_done; unsigned char* fb; Live* live; const int* finished;
  const unsigned char* wad; long wad_len; volatile char* dbg; long dbg_len;
};

__device__ __forceinline__ u64 dm_gtimer() { u64 t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
__device__ __forceinline__ int dm_ld_vol(const int* p) { return *reinterpret_cast<const volatile int*>(p); }
__device__ __forceinline__ unsigned dm_ld_vol(const unsigned* p) { return *reinterpret_cast<const volatile unsigned*>(p); }
__device__ __forceinline__ int dm_ld_acq(const int* p) { int v; asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory"); return v; }

__device__ inline void doom_tick_task(const DoomCtx& D, int f) {
  if (threadIdx.x != 0) { gpu_doom_worker_always(threadIdx.x); return; }
  if (f == 0) { gpu_doom_setup(D.wad, D.wad_len, D.dbg, D.dbg_len); gpu_doom_enable_par(1); gpu_doom_create(0ull); *reinterpret_cast<volatile u64*>(D.epoch) = dm_gtimer(); }
  gpu_doom_frame((unsigned long long)(f + 1) * 28571429ull);
}
__device__ inline void doom_copy_task(const DoomCtx& D, int rt) {
  const int rows = GH / RT, p0 = rt * rows * GW;
  const unsigned char* vb = I_VideoBuffer;
  for (int p = threadIdx.x; p < rows * GW; p += 256) D.fb[p0 + p] = __ldcg(vb + p0 + p);
  if (rt == 0) reinterpret_cast<unsigned int*>(D.fb + GW * GH)[threadIdx.x] = __ldcg(colors + threadIdx.x);     // 256 threads = 256 palette entries
}
// Runs on every block of the Doom partition until the training job finishes (D.finished).
__device__ inline void doom_loop(const DoomCtx& D) {
  __shared__ volatile int s_sel[3];
  const int tid = threadIdx.x;
  for (;;) {
    if (tid == 0) {
      int res = 0, f = 0, r = 0;
      for (;;) {
        if (dm_ld_vol(D.finished)) { res = -2; break; }
        const unsigned g = dm_ld_vol(D.ticket);
        f = (int)(g / TPG); r = (int)(g % TPG);
        bool ready;
        if (r == 0) {
          ready = (f == 0) || (dm_ld_acq(D.dctr + 1) >= RT * f);
          if (ready && f > 0) { const u64 ep = *reinterpret_cast<volatile u64*>(D.epoch); ready = ep != 0 && dm_gtimer() >= ep + (u64)f * (u64)FRAME_NS; }
        } else ready = dm_ld_acq(D.dctr + 0) >= f + 1;
        if (!ready) { __nanosleep(r == 0 ? 20000 : 3000); continue; }
        if (atomicCAS(const_cast<unsigned*>(D.ticket), g, g + 1) == g) break;
      }
      s_sel[0] = res; s_sel[1] = f; s_sel[2] = r;
    }
    __syncthreads();
    const int res = s_sel[0], f = s_sel[1], r = s_sel[2];
    __syncthreads();
    if (res < 0) break;
    if (r == 0) doom_tick_task(D, f); else doom_copy_task(D, r - 1);
    __threadfence();
    __syncthreads();
    if (tid == 0) {
      const int old = atomicAdd(D.dctr + (r == 0 ? 0 : 1), 1);
      if (r != 0 && old == RT * (f + 1) - 1) {
        if (f < DOOM_MAX_FRAMES) D.frame_done[f] = dm_gtimer();
        D.live->frames = (unsigned)(f + 1); __threadfence_system();
      }
    }
  }
}

// ---------------------------------------------------------------- host setup
// Allocates the pinned live counters (+ frame buffer) and, if enabled, the WAD copy, debug log and counters. Returns host pointers via refs.
inline bool doom_host_setup(DoomCtx& D, const std::string& wad_path, bool doom_on, unsigned char*& fb_host, Live*& live_host) {
  memset(&D, 0, sizeof D);
  Live* live_dev; unsigned char* fb_dev;
  if (cudaHostAlloc((void**)&live_host, sizeof(Live), cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer((void**)&live_dev, live_host, 0) != cudaSuccess) return false;
  memset((void*)live_host, 0, sizeof(Live));
  if (cudaHostAlloc((void**)&fb_host, FBSZ, cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer((void**)&fb_dev, fb_host, 0) != cudaSuccess) return false;
  memset(fb_host, 0, FBSZ);
  D.live = live_dev; D.fb = fb_dev;
  if (!doom_on) return true;
  FILE* wf = fopen(wad_path.c_str(), "rb"); if (!wf) { fprintf(stderr, "cannot open WAD %s\n", wad_path.c_str()); return false; }
  fseek(wf, 0, SEEK_END); long wn = ftell(wf); fseek(wf, 0, SEEK_SET); std::vector<unsigned char> wad(wn);
  if (fread(wad.data(), 1, wn, wf) != (size_t)wn) { fclose(wf); return false; } fclose(wf);
  unsigned char* dwad; cudaMalloc(&dwad, wn); cudaMemcpy(dwad, wad.data(), wn, cudaMemcpyHostToDevice);
  char *hdbg, *ddbg; cudaHostAlloc((void**)&hdbg, 1 << 20, cudaHostAllocMapped); cudaHostGetDevicePointer((void**)&ddbg, hdbg, 0); memset(hdbg, 0, 1 << 20);
  int* dctr; cudaMalloc(&dctr, 8); cudaMemset(dctr, 0, 8);
  unsigned* tk; cudaMalloc(&tk, 4); cudaMemset(tk, 0, 4);
  u64* ep; cudaMalloc(&ep, 8); cudaMemset(ep, 0, 8);
  u64* fd; cudaMalloc(&fd, (size_t)DOOM_MAX_FRAMES * 8); cudaMemset(fd, 0, (size_t)DOOM_MAX_FRAMES * 8);
  D.on = 1; D.wad = dwad; D.wad_len = wn; D.dbg = ddbg; D.dbg_len = 1 << 20; D.dctr = dctr; D.ticket = tk; D.epoch = ep; D.frame_done = fd;
  cudaDeviceSetLimit(cudaLimitStackSize, 24 * 1024);                 // Doom recurses (BSP walk); reserved per resident thread (~5 GB)
  printf("doom: WAD %s (%ld bytes), partition runs the real engine at 35 Hz\n", wad_path.c_str(), wn);
  return true;
}
