# Epic sc-25678 (Iris-3B) — terminal evidence campaign (sc-25690), 2026-10-12

Revisions: SceneWorks `feature/sc-25678-iris-3b` @437296bf8 + story branch
`story/sc-25690-epic-25678-iris-3b`; inference pin 25231d15b (inference `main`, CI gate run
[38258963637](https://github.com/SceneWorks/inference/actions/runs/38258963637)); inference probe
branch `claude/sc-25690-iris-memory-probe` (test + dispatch-only CI only, no runtime crate change).

Machines.
- **Mac2** = nax-macos-2, Apple M5 Max, 128 GiB unified memory. Runner name enforced before any build.
- **CUDA box** = two RTX PRO 6000 Blackwell cards, each `memory.total` 97,887 MiB = **95.59 GiB**
  (sold as "96 GB"). Card 0 was held by another tenant for the whole campaign (95,262 MiB used), so
  every run here used card 1 alone. The limit on multi-GPU evidence is **tenancy, not hardware**.
- The dev Mac GPU was not used (CPU-only compile and unit tests locally).

**Units.** Everything compared with a floor is **GiB** (2^30 bytes): the floors, the capacities the
gates read (`memory.total` MiB ÷ 1024; `hw.memsize` ÷ 2^30) and the guard's caps and peaks. Probe
summaries that print decimal GB are quoted as printed and labelled "GB (decimal)"; the floor tables
convert them (÷ 1.073741824) before adding a reserve and rounding up
(`crates/sceneworks-core/src/training/iris/floors.rs`).

## Epic acceptance — status of every test

Status is `closed` only where evidence of the kind the test asks for exists and is cited. `open`
names the exact blocker.

| AT | Asks for | Status | Evidence / blocker |
|---|---|---|---|
| AT1 | Packaged Apple Silicon build + NVIDIA worker: install each task via Model Manager; run generation, depth, restoration through the product/job path; depth/restoration with no Qwen installed; save/reopen jobs, settings, assets, raw-depth exports | **open** | Job-path half is evidenced on both backends (rows J1–J3, J10 below; depth/restoration ran against a hub that holds no Qwen repository). **Blocker:** nothing here ran on a *packaged* build, installed through Model Manager, with save/reopen — see "Installed packaged-build acceptance". |
| AT2 | Native generation conditioning/model/solver tensors and fixed-noise outputs vs the frozen upstream: square, portrait, landscape, prompt batches, negative conditioning, CFG intervals, solver orders, changed shift/steps; declared tolerances | **open (two cases)** | Closed at fixture level for the rest: inference @25231d15b `mlx-gen-iris` / `candle-gen-iris` `tests/{text,dit,solver,e2e,controls}_parity.rs` against upstream goldens (`tests/fixtures/iris_{text,dit,solver,e2e,controls}_golden.safetensors`, producers `crates/media/mlx-gen/tools/dump_iris_*.py`), run by the inference CI gate at the pin (job "MLX + Candle Metal packages (macOS)", run 38258963637). `controls_parity` covers solver order 1, shift 2, CFG interval (0.3, 0.8), a negative prompt, CFG off, a portrait canvas, a two-prompt batch and the `prediction: x` reading; declared tolerance 6e-2 of peak (FP32 backbone/solver, bf16 text tower). **Blocker:** the controls oracle's case list has no *landscape* canvas and no *changed-steps* case; each needs one more case in `dump_iris_controls.py` (upstream Python reference environment) and its golden. |
| AT3 | Depth and restoration outputs vs the reference across EXIF orientation, odd/small/non-square inputs, capped/native depth, restoration scales, budgeted/original inputs, colour correction on/off, tile boundaries; NPY round-trip, lineage, depth-control adaptation without raw-data loss; a saved depth asset usable after restart through a separate consumer | **open (two items)** | Closed for the numerical comparisons: inference @25231d15b `tests/depth_parity.rs` (odd / small / non-square sources, capped vs native max side, the size law, no-resize sources; `iris_depth_golden.safetensors`) and `tests/restoration_parity.rs` (torch bicubic at fractional / non-dyadic scales, antialiased resize, Gaussian fusion window and tile grid, wavelet colour fix at the boundary, PIL-Lanczos `fit_budget` bit-exact, the whole restorer in FP32 at several scales incl. a single tile, a portrait 3×, the budgeted input path and 1.3×, colour fix on and off), same CI gate run. NPY round-trip: `crates/sceneworks-core/src/depth_map.rs` `npy_round_trips_shape_dtype_byte_order_and_bits` (SceneWorks CI on the PR head). EXIF: a real-weight `image_depth` job on an EXIF-6 photo (row J2). **Blockers:** (1) EXIF orientation has no *reference-comparison* fixture — the evidence is a job-path run, not a numerical comparison; (2) "usable after restart through a separate consumer" has not been run on an installed build (part of packaged acceptance). |
| AT4 | Both backends: short real-weight runs of generation full fine-tuning, LoRA, LoKr, depth training, restoration training; cancellation/resume; export/registration; inference with the result; random-init generation and all curriculum stages on controlled fixtures | **open (SW job path, 2 rows)** | Closed through the SceneWorks job path for LoRA, LoKr (cancel → resume), depth full fine-tune, restoration LoRA and two-rank DDP (rows T1–T6). Curriculum stages / random init on fixtures: inference `tests/curricula_{parity,lifecycle}.rs`, `train_{parity,lifecycle,lineage}.rs` (CI gate run 38258963637). **Blockers:** generation FULL fine-tune and restoration FULL fine-tune have only trainer-level real-weight runs (rows T7, T8); neither has run through the SceneWorks job path, so the admission gate's full-fine-tune floor rows have never been read by a real job. The smokes exist (`iris_training_gpu_smoke_full_ft_generation_cancel_resume`, `iris_training_gpu_smoke_full_ft_restore`) and are registered in both lanes; rows T9, T10 are filled when they have run. The LoKr resume rows are re-run with the continuation assertion (rows T2, T3). |
| AT5 | Distributed replication and sharding, global-batch accounting, checkpoint save/resume, rank failure, with real multi-rank execution on supported MLX and CUDA topologies; missing hardware evidence leaves it open | **open** | Mechanism closed on one device: FSDP full / grad_op vs DDP, sharded checkpoint resume + reshard, rank exit / stall → typed failure, multi-process on ONE card (`share_device`) — inference run 38233658960 (CUDA); MLX ring/TCP multi-rank mechanism tests (inference CI) and a real-weight two-rank DDP run on one Mac (row T5). **Blockers:** (1) a real two-GPU CUDA run — both cards exist; card 0 was occupied by another tenant, and no dispatchable two-card Iris smoke exists yet (the worker smoke pins `CUDA_VISIBLE_DEVICES: "1"` and its two-rank test sets `share_device`); (2) multi-host MLX needs a second Mac free at the same time. |
| AT6 | Memory-cap admission, task switching, source unavailability, incomplete/wrong-task artifacts, invalid training targets, no-valid-depth pixels, canceled tiles/updates, failed exports; ordinary CI and regression behaviour for existing paths | **open (not fully enumerated)** | Located and green in CI on the PR head (SceneWorks `check-macos` / `check-linux` / `candle` / `parity`, PR 3050): memory-floor and cap refusals before load (`crates/sceneworks-worker/src/iris_training.rs` `tests` + `job_tests`: under-floor, unreadable host, no row), missing backbone / encoder / teacher / training resource refusals, cancel before and after the first step, task switching and eviction (`image_jobs/iris.rs` `run_iris_task_switch`, `an_iris_training_run_evicts_a_resident_generator_or_transform_before_loading`), wrong-task export refusal (`iris_restore/tests.rs` `resolution_refuses_a_wrong_task_export`; inference `restoration_parity::wrong_task_artifacts_are_refused`), canceled tiles (inference `cancel_between_tiles_is_typed_and_yields_no_image`). Existing image/upscale/control/training paths: the same required CI. **Blocker:** tests for *no-valid-depth pixels* and *failed exports* were not located in this pass; each must be cited by name or written. |
| AT7 | After inference lands and the single final pin is stable: one terminal capability/memory/calibration campaign over the new registered routes; final refs and installed-product outcomes; revisions, hashes, settings, devices, outputs, URLs | **open** | Capability dumps at the final pin list the Iris engines (preflight: `capabilities.mlx.json` vs a fresh dump PASS; `capabilities.candle.json` verified in run 38294903430). Memory: the training-floor measurements below. **Blockers:** (1) `measure-memory-catalog.mjs --list` (unfiltered, both lanes) has **zero Iris rows** — `iris_3b` has no memory-strategy contract / calibration fingerprint at the pin, so no plan anchor exists to measure (needs an inference memory contract and a pin; a product decision); (2) the training-floor measurements are being re-taken (cells marked PENDING RE-MEASUREMENT); (3) installed-product outcomes (below); (4) no 16 GB card exists here, and nothing emulates a small card's real allocation (only admission caps). |
| — | Installed packaged-build acceptance | **open** | Not run. **Blocker:** needs a packaged Apple Silicon build and an installed NVIDIA worker driven by hand: install the three tasks through Model Manager, run each, save/reopen jobs, settings, assets and raw-depth exports, restart and read a saved depth asset through a separate consumer. Every run in this ledger is a job-path test binary, a different evidence type (E9). |

## Runs

Smoke peaks in rows J1–T6 were read by the lane's first guard: `footprint -p` text, sampled about
once a second, summed over the process tree. That tool prints whole GiB above 1 GiB (the "GB" it
prints is GiB), so those peaks are **GiB at 1 GiB resolution** (MiB resolution below 1 GiB per
process). The guard now reads bytes (`scripts/ci/iris-mlx-real-weights/tree_guard.py`); rows run
with it report bytes and GiB.

| # | AT | What ran | Machine | Run | Result | Peak |
|---|---|---|---|---|---|---|
| J1 | AT1 | Worker `image_generate`, native prompt batch 2×768², 20 steps | Mac2 (MLX) | [38289099623](https://github.com/SceneWorks/SceneWorks/actions/runs/38289099623) | pass | 21 GiB (±1 GiB, ~1 Hz) |
| J2 | AT1, AT3 | Worker `image_depth`, EXIF-6 photo, raw NPY + preview; hub holds only speridlabs/iris-3b (no Qwen) | Mac2 (MLX) | 38289099623 | pass; range [-1.023, 0.996] | 9.13 GiB (~1 Hz) |
| J3 | AT1 | Worker `image_upscale` engine=iris, 4× (6 tiles) + 1×; no Qwen in hub | Mac2 (MLX) | 38289099623 | pass | 11 GiB (±1 GiB, ~1 Hz) |
| T1 | AT4 | Training job: generation LoRA (8 steps, 256²) → adapter → generate | Mac2 (MLX) | [38291366533](https://github.com/SceneWorks/SceneWorks/actions/runs/38291366533) | pass; adapter Δ 84.7 | 17 GiB (±1 GiB, ~1 Hz) |
| T2 | AT4 | Generation LoKr, `save_every` 2: first durable checkpoint at step 2; cancel requested once it existed and landed at step 4; resumed (`resumeFromJobId` translation) to 8 → generate | Mac2 (MLX) | 38291366533 | pass (without the continuation assertion); Δ 12.1 | 36 GiB (±1 GiB, ~1 Hz) |
| T2′ | AT4 | T2 with the continuation assertion (resumed job's first trained step = stop + 1, no step ≤ stop, executed = 8 − stop) | Mac2 (MLX) | PENDING RE-RUN | — | — |
| T3 | AT4 | Same LoKr cancel → resume → generate | CUDA card 1 | [38294903430](https://github.com/SceneWorks/SceneWorks/actions/runs/38294903430) | pass (without the continuation assertion); canceled at 4, resumed to 8; Δ 20.7 | not recorded |
| T3′ | AT4 | T3 with the continuation assertion | CUDA card 1 | PENDING RE-RUN | — | — |
| T4 | AT4 | Depth full fine-tune job (128², 4 steps) → registered export → depth job selects it; no Qwen in hub | Mac2 (MLX) | 38291366533 | pass | 59 GiB (±1 GiB, ~1 Hz) |
| T5 | AT4, AT5 | Two-rank DDP generation LoRA (`share_device`, one Mac) → generate | Mac2 (MLX) | 38291366533 | pass | 44 GiB (process tree, ±1 GiB per process, ~1 Hz) |
| T6 | AT4 | Restoration LoRA job (256², full recipe incl. critic / LPIPS) → export → restore job selects it; no Qwen | Mac2 (MLX) | 38291366533 | pass | 29 GiB (±1 GiB, ~1 Hz) |
| J10 | AT1, AT4 | NVIDIA worker smokes 8/8 (generation + controls + LoRA use, batch, depth, restore 4×/1×, LoRA training, depth full fine-tune, restoration LoRA, two-rank DDP on one card) | CUDA card 1 | [38273954419](https://github.com/SceneWorks/SceneWorks/actions/runs/38273954419) | pass | not recorded |
| T7 | AT4 | Generation FULL fine-tune, trainer level: official stage1_256 + REPA → stage2_512 new-phase resume, Muon | Mac2 / CUDA | inference 38179054198 (MLX; 68.16 GB decimal) / 38233658960 (CUDA) | pass | **trainer level only — not the SceneWorks job path** |
| T8 | AT4 | Restoration paper FULL fine-tune 256² (train → export → restore), trainer level | Mac2 / CUDA | inference 38145886212 (87.5 GB decimal) / 38149768393 (~93.7 GiB VRAM) | pass | **trainer level only — not the SceneWorks job path** |
| T9 | AT4 | Generation FULL fine-tune through the job path: official stage1_256 (REPA, Muon), admitted at the full floor row, canceled after the step-4 checkpoint, resumed with the continuation assertion, exported, the checkpoint rendered by a generation job | Mac2 (MLX) and CUDA card 1 | PENDING RUN | — | — |
| T10 | AT4 | Restoration FULL fine-tune through the job path (256² crop, the paper recipe) → export → restore job selects it | Mac2 (MLX) and CUDA card 1 | PENDING RUN | — | — |
| D1 | AT5 | FSDP full / grad_op vs DDP, sharded checkpoint resume + reshard, rank exit / stall → typed failure, multi-process on ONE card | CUDA card 1 | inference 38233658960 | pass | one device only |

## Training-floor measurements

Probe: `train_memory_probe` — full fine-tune, batch 1, two steps + export, whole-process peak. The
first campaign's numbers (inference runs
[38292137908](https://github.com/SceneWorks/inference/actions/runs/38292137908) on Mac2 and
[38291037726](https://github.com/SceneWorks/inference/actions/runs/38291037726) on CUDA card 1) are
partly unsound and are being re-taken: the MLX 1024² values were allocator-cache ceilings rather
than peaks, and the CUDA values were polled `nvidia-smi` readings.

| Shape | MLX (Mac2), GiB | Candle (CUDA card 1), GiB |
|---|---|---|
| Generation full 512² | PENDING RE-MEASUREMENT | PENDING RE-MEASUREMENT |
| Generation full 1024² | PENDING RE-MEASUREMENT | exceeded the 95.59 GiB card (out of memory) |
| Generation full 1024² + activation checkpointing | PENDING RE-MEASUREMENT | PENDING RE-MEASUREMENT |
| Depth full 512² | PENDING RE-MEASUREMENT | PENDING RE-MEASUREMENT |
| Depth full 1024² | PENDING RE-MEASUREMENT | exceeded the 95.59 GiB card (out of memory) |
| Restoration full 512² crop | PENDING RE-MEASUREMENT | PENDING RE-MEASUREMENT |
| Restoration full 1024² crop | lower bound only (guard stop): PENDING RE-MEASUREMENT | exceeded the 95.59 GiB card (out of memory) |

## Floors (`crates/sceneworks-core/src/training/iris{,/depth,/restore}.rs`)

- Floors are GiB and each row carries its own provenance — `measured`, `extrapolated`,
  `lower_bound` or `declared` — published per row (`memoryFloors.provenance`) and named in every
  refusal, the curriculum check and the Studio hint. There is no table-wide `measured` flag.
- A measured row is the probe's peak in the unit it printed (the `*_PEAK` constants), converted to
  GiB, plus a 2 GiB allocator reserve on Candle, rounded up; a Candle row never exceeds the card
  that held its run. Extrapolated rows are computed from the same constants. Re-measuring is an
  edit of those constants and a fixture regeneration.
- Row values: PENDING RE-MEASUREMENT (the table currently holds the first campaign's peaks run
  through the conversion; they are replaced before this PR is pushed).
- Activation checkpointing exists only for generation (the depth and restoration trainers refuse
  it), so only generation has `full_ac` rows. A full generation run refused at its plain row is told
  when the same run under activation checkpointing fits the machine.
- Inference-task manifest floors (iris_3b 32 GB MLX / 24 GB Candle, depth admission, restore
  16 / 14) are left declared: the smoke peaks above are at different geometries than the declared
  rows.
