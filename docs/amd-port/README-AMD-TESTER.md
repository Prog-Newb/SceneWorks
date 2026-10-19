# Running it, and what it does

Target: Radeon RX 7900 XTX (gfx1100), Windows.

## Running it on localhost

Build the web bundle, then the ROCm binary, then launch an API and a worker
process against the same port. `amd/tools/launch-probe-stack.ps1` does the
launch; it uses port 8010 and **its own `SCENEWORKS_DATA_DIR`**, which matters:
two stacks sharing one data dir register the same worker id and corrupt each
other's job state.

```powershell
. .\set-scene-env.ps1                            # discovers ROCm/LLVM, or SCENEWORKS_* overrides
cd apps\web ; npm run build ; cd ..\..           # rust-embed bakes this in
powershell -File amd\tools\_build-rocm.ps1       # cargo build --release, backgrounded + logged
powershell -File amd\tools\launch-probe-stack.ps1
```

Then open `http://127.0.0.1:8010`, or drive it headlessly:

```powershell
powershell -File amd\tools\_smoke-image-upscale.ps1
```

That creates a project, uploads an image, submits an `image_upscale` job with
`requestedGpu=rocm`, and polls to completion. Two notes if you adapt it:
`create_job` requires a top-level `requestedGpu`, and Windows PowerShell 5.1 has
no `Invoke-RestMethod -Form`, so the multipart upload goes through `curl.exe`.

The worker's own registration is the quickest confirmation the lane is live:

```
GET /api/v1/workers
  id=rust-utility-worker  gpu=rocm  status=idle
  capabilities: candle, gpu, image_upscale, rocm, video_upscale
```

Both lane markers present, upscale advertised, and no `dataset_upscale` --
which is the ROCm restriction working rather than the feature being broken.

## Results

### The clock

| | Before | After | |
|---|---|---|---|
| Tiles per 1440p frame | 4 x 1376x1376 | **2 x 1440x1312** | rectangular tiler |
| Mean per tile | 15.3 s | **7.09 s** | conv3d collapse + fused GroupNorm |
| **Per frame, 1440p** | **61.4 s** | **15.3 s** | **4.00x** |

Measured end to end on the same 47-frame job, same clip, same seed: 790.9 s ->
720.8 s wall, 16.8 -> 15.3 s/frame.

Other points on the curve, same card:

| Job | Output | Rate |
|---|---|---|
| image upscale, 360x360 -> 720x720 | 720x720 | **15 s total** |
| video, 1280x720 -> 2560x1440 | 1440p | **15.3 s/frame** (2 tiles, 7.09 s/tile) |
| video, 1920x1080 -> 3840x2160, 567 frames | 4K | **6 h 27 m** (~41 s/frame) |

A 30-second 1440p clip is about 3.1 hours. That is worth saying out loud in the
UI rather than leaving a user to discover it.

### Quality, gated rather than eyeballed

- **Video smoke:** 6 frames, 128 -> 256, 164 s, `min corr_vs_bicubic = 0.7661`
  against 0.766 documented.
- **Fused GroupNorm** (the one real numerical change): PSNR Y **48.35 dB**,
  SSIM **0.9929** vs the previous build, same clip and seed. Encoded size
  -1.0%.
- **Window seam:** boundary z = **+0.16 / -0.01**, i.e. the cross-fade closes
  it within noise.

### Before / after

<!-- Attach a 1:1 crop pair here: source vs 2x upscaled, same region, no
     resampling in the viewer. The 760x760 crop used for the GroupNorm gate is
     a good candidate. -->

### Verification state of this branch

| Check | Result |
|---|---|
| `cargo check`, ROCm worker | 0 errors across all 5 crates |
| Web suite (vitest) | 244 files / 3649 tests passing |
| `cargo build --release` | clean |
| Worker capability advertisement | correct (see above) |
| Real GPU job, end to end | `completed` in 15 s, 360x360 -> 720x720 |

## Where the GUI stands

| Screen | State |
|---|---|
| **Image Editor** (where image upscale lives) | **works** |
| **Video upscale panel** | **works** |
| Video Editor, Assets, Queue, Stats, Logs | work |
| Image / Video / Document / Audio **generation** studios | **gated** |

The generation studios are gated for a reason unrelated to AMD:
`GET /api/v1/models` returns **91 models, 0 of which report `installed`**, and
every studio wraps its body in `ModelAvailabilityGate`, which is ready only
when an installed model with the right capability exists. SeedVR2 works because
the worker resolves its checkpoint from `SCENEWORKS_SEEDVR2_CHECKPOINT` and
bypasses the catalog entirely; the catalog's own `seedvr2_upscaler` entry has
empty `installed` AND empty `capabilities`, so it fails the gate twice over.

`ModelAvailabilityGate` gained an optional `aside` so `VideoUpscalePanel`
renders below the gate card while generation stays gated. That is a targeted
workaround for one screen, **not** a catalog fix.

**What it would take to tie in properly:** make the catalog report what is
actually on disk (`model_import`). That is the single change standing between
"an upscaler that works" and "the studios light up", and it is not
AMD-specific -- it affects every platform equally.

## Who owns this afterwards

Worth settling explicitly rather than discovering later: **this lane cannot be
tested by anyone without an AMD GPU.** Compilation does not need the device --
kernels are built at runtime, so a build machine needs only the ROCm SDK and CI
could produce AMD artifacts -- but nothing can *verify* them without a Radeon.
Either someone with the hardware stays on the hook for running the gate above
when the lane changes, or the lane will rot quietly while still compiling.
