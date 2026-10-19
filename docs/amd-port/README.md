# SceneWorks AMD / ROCm Port

AMD Radeon support for SceneWorks 0.8.7, as a third backend beside MLX and
Candle/CUDA. Validated on an RX 7900 XTX (gfx1100) under Windows.

## Scope

**Goal:** run SeedVR2 image and video upscaling on Radeon hardware without
touching the Mac or NVIDIA lanes.

**Status:** done and measured. SeedVR2 image upscale and video upscale both run
on ROCm, at 15.3 s/frame for 1440p video (4.0x faster than the first working
build). Numbers and method in `README-AMD-TESTER.md`.

**What this does NOT change.** The CUDA and MLX lanes are untouched. ROCm is
selected at BUILD time by a feature flag, and its dependency graph lives in a
separate Cargo workspace, so a normal `cargo build` of SceneWorks never sees
any of it. Against the 0.8.7 baseline this branch touches 35 upstream files for
+1018 / -161 lines, and **no file is net-subtractive** -- nothing upstream was
removed to make room for it.

The one deliberate narrowing is scoped to the new lane only: a ROCm worker
serves SeedVR2 and refuses Real-ESRGAN, because Real-ESRGAN runs through `ort`
and there is no HIP execution provider for it. That refusal lives in the router
(`worker_supports_job`, gated on `worker_is_rocm`), so every other lane keeps
every engine it had.

## Prerequisites

| Piece | Version / source | What it does in the AMD lane |
|---|---|---|
| **AMD GPU** | gfx1100 validated (RX 7900 XTX / XT) | The compute device. Other ROCm-supported gfx targets should work -- kernels are compiled at runtime per architecture -- but only gfx1100 has been run. |
| **ROCm runtime** | 7.2.1, `rocm-sdk-core` + `rocm-sdk-libraries-custom` wheels from `repo.radeon.com/rocm/windows/rocm-rel-7.2.1/` | HIP runtime, rocBLAS and MIOpen. Needed at run time. These are `py3-none-win_amd64` wheels, i.e. zip archives of DLLs, so they can be unpacked anywhere -- no venv and no Python required. |
| **ROCm devel SDK** | `rocm-sdk-devel` 7.2.1 | Headers and import libraries. **Build time only.** |
| **candle fork** | `candle-rocm-test/candle`, a ROCm-backend fork of candle 0.10.2 | Upstream candle has no ROCm backend at all. The fork adds one (`candle-core/src/rocm_backend`) plus `candle-rocm-kernels`, which carries the `.hip` kernels. See the attribution note below. |
| **SceneWorks inference fork** | `inference-rocm` | The pinned `SceneWorks/inference` bundle with a `runtime-rocm` sibling to `runtime-cuda`, implementing the same `gen-core` contract. |
| **LLVM / libclang** | any recent LLVM | `bindgen` only, at build time. The resource dir is resolved from the highest installed clang rather than pinned. |
| **ffmpeg** | any standard Windows build | Decode and encode for the **video** path only. Image upscale does not need it. |
| **SeedVR2 weights** | `seedvr2_ema_3b_fp16.safetensors` (6.47 GB) + `ema_vae_fp16.safetensors` (0.48 GB) | The 3B one-step diffusion DiT and its 3D causal VAE. Filenames are matched exactly by the loader. **fp16, not bf16** -- see the dtype note in `IMPLEMENTATION-NOTES.md`. |

Set the build environment with `set-scene-env.ps1` at the repo root. It
discovers every path above or takes `SCENEWORKS_*` overrides, and fails with the
missing piece named rather than silently.

## Attribution

The ROCm backend inside the candle fork is not ours. 17 of the 25 commits on
that fork are a third-party ROCm port of candle; 8 are ours (the performance
work and the dtype fix). candle is MIT / Apache-2.0. This matters for how the
lane should be maintained -- see "Who owns this afterwards" in
`README-AMD-TESTER.md`.

## The three documents

- **this file** -- scope and prerequisites
- **`PORT_MAP.md`** -- how it fits together, stage by stage, with the hot spots
- **`README-AMD-TESTER.md`** -- how to run it, and the measured results
- **`IMPLEMENTATION-NOTES.md`** -- the long version: what broke, why, and the numbers behind each fix
