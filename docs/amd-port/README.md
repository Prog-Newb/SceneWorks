# SceneWorks AMD / ROCm Port

AMD Radeon support for SceneWorks 0.8.7, as a third backend beside MLX and
Candle/CUDA. Validated on an RX 7900 XTX (gfx1100) under Windows.

> **Build from `amd-rocm-0.8.7`, not from `amd-on-087`.** Two branches carry
> this work and only one is sound. See "Which branch" below.

## Scope

**Goal:** run SeedVR2 image and video upscaling on Radeon hardware without
touching the Mac or NVIDIA lanes.

**Status:** done and measured **on this machine**. SeedVR2 image upscale and
video upscale both run on ROCm, at 15.3 s/frame for 1440p video (4.0x faster
than the first working build). Numbers and method in `README-AMD-TESTER.md`.

Read "Can you build this today?" before you trust that figure: most of the 4.0x
comes from code that is **not in this branch**.

## Which branch

| Branch | State |
|---|---|
| **`amd-rocm-0.8.7`** | **Use this one.** The AMD work on a clean `release/0.8.7` base -- the base it was built and validated against. Verified against upstream: no upstream file is net-subtractive, no `runtime-cuda` dependency line is added or removed in any manifest, and the `backend-candle` feature definition is unchanged. |
| `amd-on-087` | **Do not build from this.** Kept only because it is the branch attached to the review in PR #3094 and rewriting it would break that record. A merge of `main` into it on 2026-10-19 (`fe2caf064`) resolved badly -- details below. |

### What went wrong on `amd-on-087`

The 2026-10-19 merge of `main` kept the 0.8.7 side of hunks it should have
taken `main`'s side of. Three measured consequences, none of them intentional:

- **`runtime-cuda` pinned backward.** `crates/sceneworks-worker/Cargo.toml`
  reads `rev = "0e4adc6f"` where `main` has `d47ce76f`, roughly 3,400 commits
  back. This is what the reviewer caught, and it is real.
- **`candle-gen-ltx` dropped from `backend-candle`.** `main` has it; that branch
  does not. A removal from an upstream feature.
- **~800 lines of upstream tests lost.**
  `crates/sceneworks-core/src/jobs_store/tests/candle_routing_tests.rs` is 5595
  lines on `main` and 4859 there. The AMD commit itself only ever added 42 lines
  to that file and removed none, so the shortfall is merge loss, not an edit.

The manifest and the root lockfile also disagree on that branch: the merge took
`main`'s `d47ce76f` into `Cargo.lock` while leaving `0e4adc6f` in the manifest.

None of this is present on `amd-rocm-0.8.7`.

## What this does NOT change

The CUDA and MLX lanes are untouched on `amd-rocm-0.8.7`. ROCm is selected at
BUILD time by a feature flag, and the bulk of its dependency graph lives in a
separate Cargo workspace under `amd/`. Against the 0.8.7 baseline the branch
touches 35 upstream files for +1018 / -161 lines and **no upstream file is
net-subtractive** -- verified by `git diff --numstat upstream/release/0.8.7..HEAD`,
which returns no file whose removals exceed its additions.

In `crates/sceneworks-worker/Cargo.toml` specifically, the whole diff is four
added lines and nothing removed: the `backend-rocm` feature and the optional
`runtime-rocm` dependency. No `runtime-cuda` dependency line is added or removed
in any manifest in the tree, and the `backend-candle` feature definition is
unchanged.

**But "untouched" is not the same as "unedited."** 90 `cfg` gates in upstream
Rust source that read `feature = "backend-candle"` were widened to
`any(feature = "backend-candle", feature = "backend-rocm")`, so the ROCm lane
can reuse the code paths the CUDA lane already had. Those lines are modified
upstream lines. They are semantically transparent to a CUDA build -- with
`backend-rocm` off, `any(candle, rocm)` evaluates exactly as `candle` did -- but
anyone auditing the diff should expect to see them, and anyone rebasing will
have to carry them. `PORT_MAP.md` lists where they are.

**Correction to an earlier claim.** An earlier version of this document said a
normal `cargo build` at the repo root "never sees any of it." That was wrong.
`crates/sceneworks-worker/Cargo.toml` and `apps/rust-api/Cargo.toml` each gained
an *optional* path dependency on the shim, and Cargo reads a path dependency's
manifest during resolution whether or not its feature is enabled. So a clean
checkout can fail at resolution time before anyone turns a ROCm feature on. The
`amd/` workspace limits how far the ROCm graph spreads; it does not make the
root build indifferent to it. Fixing that properly means publishing the forks
and depending on them by git revision instead of by local path -- see below.

The one deliberate narrowing is scoped to the new lane only: a ROCm worker
serves SeedVR2 and refuses Real-ESRGAN, because Real-ESRGAN runs through `ort`
and there is no HIP execution provider for it. That refusal lives in the router
(`worker_supports_job`, gated on `worker_is_rocm`), so every other lane keeps
every engine it had.

## Can you build this today?

**No -- not from this branch alone, and not without the author's machine.** Be
clear-eyed about this before investing time.

**1. Two dependencies are unpublished forks living on local disk.**
`amd/Cargo.toml` carries `[patch]` entries pointing at `candle-rocm-test/candle`
and `inference-rocm` as filesystem paths. Neither fork is published anywhere.
Clone this branch and resolution fails immediately -- there is nothing for those
paths to resolve to. This is the single blocker, and nothing else matters until
it is fixed.

**2. The SeedVR2 performance work is in the inference fork, not here.**
The rectangular tiler, the `T == 1` collapse in conv3d, the fused GroupNorm and
the fp16 dtype fix all live in
`inference-rocm/crates/media/candle-gen/candle-gen-seedvr2`. That is where the
4.0x comes from. This branch contains the *plumbing* -- backend selection, the
worker shim, capability advertisement, the job router gate -- and the plumbing
alone gets you a ROCm lane that runs, not a fast one. Any figure in these docs
is a measurement of **branch + both forks together**, never of this branch by
itself.

**3. `amd/Cargo.lock` pins `SceneWorks/inference` at the 0.8.7 revision**
(`0e4adc6f`), because that is the release this branch was built on. Upstream
`main` has since moved to `d47ce76f`, roughly 3,400 commits ahead. The AMD
workspace has never been resolved against anything newer, so treat 0.8.7 as the
only version this lane is known to work with.

What it would take to make this buildable by a third party:

| Step | Why |
|---|---|
| Publish the candle fork, pin an immutable revision | Removes the first `[patch]` path; preserves the third-party ROCm attribution in history |
| Publish `inference-rocm`, pin an immutable revision | Removes the second, and brings the engine work with it |
| Replace both `[patch]` paths with `git` + `rev` | Makes a clean clone resolve |
| Rebase onto a current upstream release, resolving each conflict against that release | The 0.8.7 base is validated but old; this is the step `amd-on-087` got wrong |
| Re-resolve `amd/Cargo.lock` against a current inference | Closes the 3,400-commit gap |
| Re-validate on gfx1100 after all of the above | Nothing above is a no-op |

None of that is done. Until it is, this branch is a **record of a working port**
on specific hardware, not a distributable one.

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

## The documents

- **this file** -- scope, which branch, buildability, and prerequisites
- **`PORT_MAP.md`** -- how it fits together, stage by stage, with the hot spots
- **`README-AMD-TESTER.md`** -- how to run it, and the measured results
- **`IMPLEMENTATION-NOTES.md`** -- the long version: what broke, why, and the numbers behind each fix
