# Implementation notes

The long version. `README.md`, `PORT_MAP.md` and `README-AMD-TESTER.md` are the
short ones; this is what is behind them, including the things that went wrong
and the measurements that settled them.

---

## 1. The bug that made everything else unmeasurable

`candle-gen-seedvr2/src/lib.rs` mapped `Precision::Bf16` to `DType::BF16` on
GPU. The shipped checkpoint is `seedvr2_ema_3b_fp16.safetensors` -- **fp16**.

bf16 and fp16 are both 16 bits but spend them differently: bf16 has 8 exponent
bits and 7 explicit mantissa bits, fp16 has 5 and 10. Loading fp16 weights as
bf16 therefore throws away 3 bits of mantissa on every value. The result was
not a subtle quality regression:

| dtype | output size | look |
|---|---|---|
| bf16 | 416,110 B | blurred mush |
| f16 | larger, detailed | correct |

Everything downstream was being tuned against a broken baseline. This is worth
flagging first because it is the kind of defect that looks like a performance
or model problem and is neither.

A related one in the same family: `miopen_dtype` tested for `"f16"` **before**
`"bf16"`, so a bf16 tensor matched the f16 arm and was described to MIOpen as
fp16. The `MiopenBFloat16` arm was unreachable.

---

## 2. MIOpen: two separate findings

### 2.1 Find-and-forget is not how it works

`miopenFindConvolutionForwardAlgorithm` is not a pure query. It **registers an
invoker** as a side effect, and the invoker cache is bounded and evicts. The
backend originally called it on *every* forward pass, which is pathologically
slow; but simply hoisting it out is also wrong, because the registration is
required at least once per shape and an evicted entry has to be re-registered.

### 2.2 The 17x timing swing was a warm cache, not the code

Two runs of the same build differed by ~17x and the difference was entirely
whether MIOpen had already searched that convolution shape. The first time it
sees a shape it runs an exhaustive algorithm search -- **321.82 s on a first
tile**, measured -- then caches:

```
%LOCALAPPDATA%\candle-rocm\{arch}-{rocm_version}\     compiled .hip kernels
%USERPROFILE%\.miopen\gfx1100_48.HIP.*.ufdb.txt       MIOpen find-db
```

Both are **load-bearing performance state in the user's home directory, not in
the project.** Delete either and the next run pays the search again.

Two consequences:

1. **Any benchmark on this card is meaningless unless you state cache state.**
   Twice in this work a conclusion was drawn from a warm-cache run and had to
   be retracted. Treat "it ran fast" and "it ran at all" as different claims
   from "it is fast from cold".
2. A build shipped to users needs a pre-warm or an honest first-run UI state.
   See the MIOpen section of `PORT_MAP.md`.

### 2.3 A related trap

At odd shapes MIOpen can fall through to a naive-convolution path that pegs a
core and does not finish. A 320x180 test clip triggered this -- 180 is not a
multiple of the VAE scale. Test clips must use dimensions the pipeline actually
aligns to, and ideally a shape already in the cache.

---

## 3. The 4.0x, in three parts

All three live in `candle-gen-seedvr2`, not in candle, and none of them contain
a single ROCm-specific call. **They are portable, and would speed up the CUDA
lane too.** Files: `conv3d.rs`, `nn.rs`, `pipeline.rs`, `vae.rs`, `video.rs`.

| Stage | Change | Per frame at 1440p | Cumulative |
|---|---|---|---|
| baseline | 4 tiles of 1376x1376 | 61.4 s | -- |
| + rectangular tiler | 2 tiles of 1440x1312 | 29.8 s | 2.06x |
| + conv3d `T==1` collapse, fused GroupNorm | same tiling | **15.3 s** | **4.00x** |

**Rectangular spatial tiler** (`video.rs`). The tiler cut square tiles; frames
are not square. Fitting tiles to the actual frame halves the tile count on a
1440p frame. Pure arithmetic over the tile plan -- no backend calls, nothing to
re-gate.

**conv3d `T==1` collapse** (`conv3d.rs`). candle has no `conv3d`, so
`CausalConv3d` decomposes a `kT x kH x kW` kernel into `kT` conv2d taps and
sums them. When the temporal extent is 1 -- the common case in the VAE decoder
-- the whole sum collapses to one conv2d. Op composition at the candle level,
so it is backend-agnostic by construction.

**Fused GroupNorm** (`nn.rs`). Five strided f32 passes replaced by one
`candle_nn::ops::layer_norm`. Two honest caveats:

- It changes the arithmetic: variance as `E[x^2] - mu^2` rather than two-pass
  `mean((x-mu)^2)`. **This is a numerical change, not a reassociation**, and it
  was gated as one: PSNR Y 48.35 dB / SSIM 0.9929 against the previous build,
  same clip and seed, encoded size -1.0%. Re-gate on CUDA before trusting it
  there.
- Whether `layer_norm`'s CUDA kernel is a win on NVIDIA is **unmeasured**. It
  won here because those strided passes are expensive on RDNA3. That is an
  observation about this card.

---

## 4. Windowing and the seam

`SEEDVR2_WORKER_CHUNK_FRAMES = 64` frames per worker window, 4-frame overlap.
This is distinct from the DiT's own attention window of `(4,3,3)` -- they are
different mechanisms at different scales and conflating them leads to wrong
reasoning about both memory and quality.

The worker does not implement its own blend: the cross-fade at each worker
window boundary is the engine's own `DEFAULT_OVERLAP` blend, byte-identical to
its internal one. That was verified numerically rather than visually --
boundary z = **+0.16 / -0.01** -- because a seam that looks fine at one bitrate
can be obvious at another.

VRAM budget constants, for reference:
`VAE_SAFE_DECODE_EDGE_PX = 1536`, `GB_PER_MPX_FRAME = 8.0`,
`SPATIAL_ALIGN = 16`, `MIN_TILE_PX = 256`.

---

## 5. Cross-cutting fixes that are not AMD-specific

These were found while working on the AMD lane but are defects in shared code,
and they affect every platform.

**Multi-window video progress was wrong.** The progress fraction was
`0.25 + 0.30 * (step / 64)`, computed from the step counter *within the current
window* with no awareness of which window it was. On a 10-window job it
reported 41% while 1.1% of the work was done, then reset. Replaced with a
banded scheme: each window gets a slice of the overall bar
(`lo + span * step/total`), so progress is monotonic across the whole job.

**The ETA pipeline existed end to end and was always null.** `jobs_store`
accepted `etaSeconds`, the DTO carried it, the UI had a slot for it -- but the
`video_progress` helper hard-coded `eta_seconds: None`, so nothing ever reached
it. Added `video_progress_with_eta` beside it, and a `GenProgressWindow` that
carries the driver's estimate through the inner per-step posts (the first
attempt at this failed *because* the inner posts wrote `None` and erased what
the driver had set).

The estimator itself warms up: it ignores the first window, then projects from
the mean of completed windows. One known weakness -- it charges every remaining
window at full cost, so on a job whose last window is partial it over-estimates
the tail. Observed: predicted 2 h 3 m remaining, actual 1 h 2 m. It errs
pessimistic, which is the right direction to be wrong.

**The disk guard over-estimated by 4.1x** and refused jobs that would have
fitted. It assumed raw RGB output; the pipeline writes PNG. Measured 6.88
MB/frame against the guard's 28.34. Corrected with an explicit
`SEEDVR2_PNG_RAW_RATIO = 0.5`, and a zero dimension now short-circuits to
"unknown" rather than computing a nonsense budget.

**`formatSeconds` on a raw float** rendered an ETA as
`"325m 33.56638480000038s"`. Replaced with a `formatRemaining` that rounds and
degrades to "under a minute".

---

## 6. What you need to build it

Prerequisites are in `README.md`. The mechanics:

```powershell
# 1. Environment. Discovers ROCm / LLVM, or honours SCENEWORKS_* overrides.
#    Machine-specific paths belong in set-scene-env.local.ps1 (gitignored);
#    copy set-scene-env.local.example.ps1 as a starting point.
. .\set-scene-env.ps1

# 2. Web bundle FIRST -- embedded-api implies embed-web, which bakes
#    apps/web/dist into the binary via rust-embed.
cd apps\web ; npm run build ; cd ..\..

# 3. The ROCm binary, from the amd/ workspace (NOT the repo root).
cd amd
cargo build -p sceneworks-rocm-worker --bin sceneworks-api `
    --features embedded-api --release
```

Helpers in `amd/tools/`: `_check-rocm.ps1` (fast `cargo check`),
`_build-rocm.ps1` (release build), `launch-probe-stack.ps1` (isolated API +
worker), `_smoke-image-upscale.ps1` (end-to-end job).

### Gotchas that cost real time here

- **`build-headless-api.ps1` names a package that is not a workspace member.**
  `sceneworks-rocm-worker` lives in the `amd/` workspace; run cargo from there.
- **Build the web bundle before cargo**, or rust-embed silently bakes in a
  stale UI and you debug a frontend that is not the one you edited.
- **The linker will fail with `Access is denied` if a built binary is still
  running.** Renaming the running exe aside works (Windows permits it) and lets
  the link proceed.
- **`rocm-sdk-core` on public PyPI is an unrelated placeholder** at 0.1.0, and
  `rocm-sdk-libraries-custom` 404s there. The real wheels are on
  `repo.radeon.com`.
- **With the ROCm DLLs missing, the worker exits instantly with no stderr and
  no listener** -- Windows `STATUS_DLL_NOT_FOUND` prints nothing. It looks like
  "the binary does nothing" rather than a missing dependency. Any packaging
  work should fail loudly here instead.
- **`#[ignore]`-gated GPU tests**: plain `cargo test` reports "ok, 0 passed,
  2 ignored", which reads as a pass. They need `-- --ignored`.
- **vitest needs `--environment jsdom`** when invoking a single file directly;
  without it a component suite fails on "document is not defined".

---

## 7. Open items and known risks

**Runtime payload size is not settled.** The measured minimum to *start and run
a warm job* is 1.78 GB (`_rocm_sdk_core\bin` 0.17 GB +
`_rocm_sdk_libraries_custom\bin` 1.61 GB), with `lib\llvm\bin` (1.53 GB) and
`rocm-sdk-devel` (1.14 GB) excluded. But that A/B ran against a warm kernel
cache, and kernels are compiled **at runtime** by
`candle-rocm-kernels/src/compile.rs` -- there is no build script. If that
compilation needs the HIP compiler from `llvm\bin`, the real runtime payload is
~3.3 GB. The test that settles it: clear the kernel cache, drop `llvm\bin` from
`PATH`, run one image upscale.

**cuda/rocm feature exclusivity is handled for one binary, not one tree.** The
`not(feature = "backend-candle")` guard makes the runtime alias unambiguous, and
the `amd/` workspace keeps the graphs apart. Building *both* lanes from the root
workspace in one invocation has not been attempted and is the main unknown if
the goal becomes a single tree producing both binaries.

**No installer.** Tauri has never bundled this; the resource dirs
(`rocm-worker/`, `ffmpeg/`, `onnxruntime/`, `ui/`) are empty. A ROCm build is
currently a console binary plus a browser.

**The catalog gate** (`model_import`) is the blocker for the generation
studios, and is platform-neutral. See `README-AMD-TESTER.md`.

**Deferred, minor:** a hard kill leaks the SeedVR2 scratch directories under
`%TEMP%\sceneworks_seedvr2_{src,out}_*` (a graceful cancel does not), and
`over_vae_cap` is ignored in `generate_budgeted`.
