# How the AMD lane fits together

## 1. Lane selection is a build-time feature

Exactly the seam this document recommended before the work started, now in
`crates/sceneworks-worker/src/inference_runtime.rs`:

```rust
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
use runtime_cuda as platform_runtime;
#[cfg(all(not(target_os = "macos"), feature = "backend-rocm",
          not(feature = "backend-candle")))]
use runtime_rocm as platform_runtime;
#[cfg(target_os = "macos")]
use runtime_macos as platform_runtime;
```

The `not(feature = "backend-candle")` guard makes the two mutually exclusive, so
enabling both does not produce a double-import. One binary, one backend. A
runtime-switchable binary carrying both toolchains is a different project and is
NOT what this is.

Everywhere else the port only ever **widens** an existing gate:

```rust
- #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
+ #[cfg(all(not(target_os = "macos"), any(feature = "backend-candle",
+                                         feature = "backend-rocm")))]
```

## 2. The dependency graph is isolated, not patched in place

A Cargo `[patch]` is unconditional and single-valued, so patching candle in the
root manifest would drag the ROCm fork into every SceneWorks build. Instead the
AMD lane has its own workspace, `amd/`, whose shim crate
(`amd/worker-shim`, package `sceneworks-rocm-worker`) depends on the real crates
with `features = ["backend-rocm"]` and carries the patches locally:

```toml
# amd/Cargo.toml -- applies ONLY when cargo is invoked from this workspace
[patch."https://github.com/huggingface/candle"]
candle-core = { path = "../candle-rocm-test/candle/candle-core" }
[patch."https://github.com/SceneWorks/inference"]
candle-gen  = { path = "../inference-rocm/crates/media/candle-gen/candle-gen" }
```

So: `cargo build` at the root is unaffected and still resolves upstream candle
and upstream inference. `cargo build` from `amd/` gets the ROCm fork. Build the
AMD binary with:

```
cd amd
cargo build -p sceneworks-rocm-worker --bin sceneworks-api \
    --features embedded-api --release
```

(`embedded-api` already implies `sceneworks-rust-api/embed-web`, which bakes the
web bundle in through rust-embed -- so build `apps/web` FIRST or the binary
carries a stale UI.)

## 3. The processing chain, with the hot spots

Video upscale, in order. `[H]` marks a place where something non-obvious had to
change for AMD, or where the performance went.

```text
  source clip
      |
      v
  [1] ffmpeg decode, streaming                 worker/video_jobs/seedvr2.rs
      |
      v
  [2] worker window plan   64 frames, overlap 4                        [H]
      |
      v
  +---- per window ------------------------------------------------+
  |   [3] spatial tile plan   fit to VRAM budget              [H]  |
  |       |                                                        |
  |       v                                                        |
  |   [4] SeedVR2 DiT, one step   3B fp16                     [H]  |
  |       |                                                        |
  |       v                                                        |
  |   [5] 3D causal VAE decode   conv3d via conv2d taps       [H]  |
  |       |                                                        |
  |       v                                                        |
  |   [6] tile stitch + GroupNorm                             [H]  |
  +----------------------------------------------------------------+
      |
      v
  [7] cross-fade across the window seam
      |
      v
  [8] ffmpeg encode + source-audio passthrough
      |
      v
  [9] asset write
```

**[1] Decode.** Streaming and windowed rather than whole-clip, so peak host RAM
is bounded by one window instead of the whole video. `-fps_mode passthrough`.
Backend-neutral -- unchanged by this port.

**[2] Window plan. [H]** `SEEDVR2_WORKER_CHUNK_FRAMES = 64` frames per worker
window with a 4-frame overlap. Note this is NOT the model's attention window,
which is `(4,3,3)`; conflating the two is an easy mistake. The seam between
windows was verified numerically, not by eye: boundary z = +0.16 / -0.01.

**[3] Spatial tiling. [H] First of the three speedups.** The original tiler cut
square tiles; the frame is not square, so a 1440p frame needed 4 tiles of
1376x1376. A rectangular tiler fitted to the actual frame does the same frame in
**2 tiles of 1440x1312**. Constants: `VAE_SAFE_DECODE_EDGE_PX = 1536`,
`GB_PER_MPX_FRAME = 8.0`, `SPATIAL_ALIGN = 16`, `MIN_TILE_PX = 256`. Pure tile
arithmetic, no backend calls -- **this would speed up the CUDA lane too.**

**[4] DiT. [H]** The checkpoint is **fp16**. The engine mapped
`Precision::Bf16` to `DType::BF16` on GPU, and bf16 has 7 mantissa bits against
fp16's 10, so the weights were being silently truncated -- output was visibly
blurred mush, not a subtle regression. One-line dtype fix; everything about
AMD performance was unmeasurable until it landed.

**[5] conv3d. [H]** candle ships no `conv3d`, so `CausalConv3d` decomposes a
`kT x kH x kW` kernel into `kT` conv2d taps that are summed. **Second
speedup:** when `T == 1` the whole decomposition collapses to a single conv2d,
which is the common case in the VAE decoder.

**[6] GroupNorm. [H] Third speedup.** Replaced five strided f32 passes with one
fused call to `candle_nn::ops::layer_norm`. This is a real **numerical** change
-- variance as `E[x^2] - mu^2` instead of two-pass `mean((x-mu)^2)` -- so it was
gated rather than assumed: PSNR Y 48.35 dB, SSIM 0.9929 against the previous
build on the same clip and seed. `layer_norm` already has CUDA and Metal paths,
so this is portable, **but whether it wins on NVIDIA is unmeasured** -- it won
on RDNA3 because those strided passes were expensive there, which is an
observation about this card, not a law.

**[7] Cross-fade.** The engine's own blend, byte-identical to its internal
window seam. The worker does not implement its own.

**[8-9] Encode and asset write.** Backend-neutral.

### The one that will bite a first-time user: MIOpen

Not a stage, but it sits under [4] and [5]. `miopenFindConvolutionForwardAlgorithm`
registers an invoker as a side effect, and the invoker cache is bounded and
evicts -- so calling it per-forward, as the backend originally did, is both
slow and necessary to call at least once. Worse, the first time MIOpen sees a
new tile shape it runs an exhaustive algorithm search: **321.82 s measured on a
first tile**, cached afterwards per architecture under
`%LOCALAPPDATA%\candle-rocm\{arch}-{rocm_version}` and `%USERPROFILE%\.miopen`.

On a warm cache this is invisible. On a fresh machine the first job of the first
session stalls for minutes at near-zero visible progress, and every user who
has not been warned will kill it. Anything shipped to users needs either a
post-install pre-warm of the common resolutions or an explicit "compiling GPU
kernels, one time only" state in the UI.

## 4. The original plan, and what actually happened

| Surface | AMD action planned | Outcome |
|---|---|---|
| Worker feature | Add a distinct ROCm feature/runtime | **Done** -- `backend-rocm`, mutually exclusive with `backend-candle` |
| Inference bundle | Add `runtime-rocm` on the same `gen-core` contract | **Done** -- `inference-rocm`, contract unchanged |
| Candle | Move to a ROCm-capable revision | **Done** -- fork with a ROCm backend; isolated via `amd/` workspace `[patch]` |
| GPU discovery | HIP/AMD discovery path | **Done** -- `rocm_gpu()` sentinel beside `mlx_gpu()`; the ROCm build never calls `nvidia-smi` |
| Preflight | HIP/ROCm preflight | **Done** -- `rocm_preflight()` in `gpu_check()`, plus `rocm_startup_health` / `probe_rocm_health` with no-op fallbacks off-lane |
| Compute capability | AMD gfx-family capabilities | **Done** -- arch resolved at runtime (`CANDLE_ROCM_ARCH`, then `option_env!`, then `rocminfo`), so one binary is not arch-locked |
| ONNX Runtime | AMD provider strategy | **Not done, by decision** -- no HIP execution provider for `ort`, so Real-ESRGAN and `dataset_upscale` are refused on the ROCm lane only |
| Packaging | ROCm/HIP runtime packaging | **Open** -- no installer; see `README-AMD-TESTER.md` |
| Provider imports | Preserve API through re-export | **Done** -- `platform_runtime` alias, model and job code untouched |
