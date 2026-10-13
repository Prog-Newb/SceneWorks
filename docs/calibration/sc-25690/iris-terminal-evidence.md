# Epic sc-25678 (Iris-3B) — terminal evidence campaign (sc-25690)

Updated 2026-10-13. Revisions: SceneWorks side branch `wip/sc-25690-phase-a` @28c764b2d (the
story PR 3050 has not been pushed since c07981738; it gets one push at the end). Inference pin
**c1f594203** (inference `main`, merge of #1242). One more inference PR (MLX activation
checkpointing for depth / restoration) is expected to move the pin once more; rows that depend on
it say so.

Machines.
- **Mac2** = nax-macos-2, Apple M5 Max, 128 GiB unified memory. Runner name enforced before any build.
- **CUDA box** = two RTX PRO 6000 Blackwell cards, each `memory.total` 97,887 MiB = **95.59 GiB**
  (102.64 GB decimal; sold as "96 GB"). Card 0 has been held by another tenant throughout
  (95,262 MiB used, last read 2026-10-13T15:47Z in run 38422258699); every run here
  used card 1 alone. The limit on two-card evidence is **tenancy, not hardware**.
- The dev Mac GPU was not used (CPU-only compile and unit tests locally).

**Units.** Everything compared with a floor is **GiB** (2^30 bytes): the floors, the capacities the
gates read (`memory.total` MiB ÷ 1024; `hw.memsize` ÷ 2^30) and the guard's caps and peaks. Probe
summaries print decimal GB; they are quoted as printed, labelled "GB", and converted
(÷ 1.073741824) before a reserve is added and the sum rounded up
(`crates/sceneworks-core/src/training/iris/floors.rs`).

**Evidence types** (E9) are kept apart below: *fixture parity* (native vs the frozen upstream on a
miniature), *unit / job-path tests* (the production job path with the trainer or estimator
stubbed), *real-weight job-path runs* (a test binary driving the production job path on real
weights), *trainer-level runs* (inference's own real-weight tests) and *installed-product
acceptance* (a packaged build driven by hand). Local test results on this branch are stated as
local; CI on the PR head will exist only after the final push.

## Epic acceptance — status of every test

`closed` only where evidence of the kind the test asks for exists and is cited. `open` names the
exact blocker.

| AT | Asks for | Status | Evidence / blocker |
|---|---|---|---|
| AT1 | Packaged Apple Silicon build + NVIDIA worker: install each task via Model Manager; run generation, depth, restoration through the product/job path; depth/restoration with no Qwen installed; save/reopen jobs, settings, assets, raw-depth exports | **open** | Real-weight job-path runs exist on both backends (rows J1–J3, J10; depth and restoration ran against a hub that holds no Qwen repository). **Blocker:** nothing ran on a *packaged* build, installed through Model Manager, with save/reopen — see "Installed packaged-build acceptance". |
| AT2 | Native generation conditioning/model/solver tensors and fixed-noise outputs vs the frozen upstream: square, portrait, landscape, prompt batches, negative conditioning, CFG intervals, solver orders, changed shift/steps; declared tolerances | **closed (fixture parity)** | Inference `mlx-gen-iris` / `candle-gen-iris` `tests/{text,dit,solver,e2e,controls}_parity.rs` against goldens produced from the frozen upstream (speridlabs/iris-3b @a8d15239; producers `crates/media/mlx-gen/tools/dump_iris_*.py`). The controls oracle covers solver order 1, shift 2, CFG interval (0.3, 0.8), a negative prompt, CFG off, a portrait canvas (12×8), a two-prompt batch, the `prediction: x` reading and — added by inference #1241 (afc88d24f, in the pin) — a **square** canvas (8×8), a second **landscape** canvas (8×16) and a **changed step count** (8 vs 5). `controls_parity::every_control_matches_upstream_on_the_miniature` pins the case list by name on both backends. Declared tolerances: MLX 6e-2 of peak (FP32 backbone / solver, bf16 text tower; observed 2.2e-2 – 4.0e-2 on the new cases), Candle 2e-4 (fp32 tower; observed ≤ 5.3e-5). Three mutations of the shared `gen-core` plan each red exactly the case they should (#1241). CI: inference `CI gate` on `main` at the pin. The comparison is tensor-level on a miniature snapshot, not on the 3B weights — that is what the test asks for ("controlled fixtures"). |
| AT3 | Depth and restoration outputs vs the reference across EXIF orientation, odd/small/non-square inputs, capped/native depth, restoration scales, budgeted/original inputs, colour correction on/off, tile boundaries; NPY round-trip, lineage, depth-control adaptation without raw-data loss; a saved depth asset usable after restart through a separate consumer | **open (one item)** | Fixture parity, in the pin: inference `tests/depth_parity.rs` (odd / small / non-square sources, capped vs native max side, the size law, no-resize sources) and `tests/restoration_parity.rs` (torch bicubic at fractional / non-dyadic scales, antialiased resize, Gaussian fusion window and tile grid, wavelet colour fix at the boundary, PIL-Lanczos `fit_budget` bit-exact, the whole restorer in FP32 at several scales incl. a single tile, a portrait 3×, the budgeted input path and 1.3×, colour fix on and off). **EXIF orientation** (SceneWorks cddae1d4f, on this branch; local run green, CI pending the push): `depth_jobs/tests.rs` `an_exif_rotated_source_yields_the_reference_depth_of_its_upright_pixels` — for every orientation 1..=8 the job's raw `.npy` is bit for bit the reference estimate of the upright pixels, the upright reference itself pinned to Pillow 12.3.0's `exif_transpose` (`the_upright_reference_is_pils_exif_transpose`). NPY round-trip: `sceneworks-core` `depth_map.rs` `npy_round_trips_shape_dtype_byte_order_and_bits`. Real weights: an `image_depth` job on an EXIF-6 photo (row J2). **Blocker:** "usable after restart through a separate consumer" has not been run on an installed build (part of packaged acceptance). |
| AT4 | Both backends: short real-weight runs of generation full fine-tuning, LoRA, LoKr, depth training, restoration training; cancellation/resume; export/registration; inference with the result; random-init generation and all curriculum stages on controlled fixtures | **closed (real-weight job-path runs, both backends, at pin c1f594203 for Candle)** | **MLX: closed** through the SceneWorks job path on real weights — LoRA, LoKr cancel → resume (continuation asserted), depth full fine-tune, restoration LoRA, two-rank DDP, generation FULL fine-tune (official stage1_256 + REPA; cancel → resume → export → render the checkpoint) and restoration FULL fine-tune (rows T1–T6, T9, T10). **Candle:** **closed** through the same job path on card 1 at pin c1f594203 — LoKr cancel → resume with the continuation assertion (T3), generation FULL fine-tune stage1_256 + REPA cancel → resume → export → render (T9c), restoration FULL fine-tune → export → restore job (T10c), depth full fine-tune → export → depth job (T4c), and depth and restoration again under gradient checkpointing (T4c, T10c); LoRA, restoration LoRA and one-card two-rank DDP from the earlier 8/8 run at pin 25231d15b (J10). Caveat, stated rather than hidden: the MLX rows T1, T2, T5, T6, T9 ran at pin 25231d15b; only T4 and T10 were re-run at c1f594203, and the pin will move once more — the final-pin re-run is F2 work. Curriculum stages / random init on fixtures: inference `tests/curricula_{parity,lifecycle}.rs`, `train_{parity,lifecycle,lineage}.rs` (CI gate at the pin). |
| AT5 | Distributed replication and sharding, global-batch accounting, checkpoint save/resume, rank failure, with real multi-rank execution on supported MLX and CUDA topologies; missing hardware evidence leaves it open | **open** | Mechanism on one device: FSDP full / grad_op vs DDP, sharded checkpoint resume + reshard, rank exit / stall → typed failure, multi-process on ONE card (`share_device`) — inference run 38233658960 (CUDA); MLX ring/TCP multi-rank mechanism tests (inference CI) and a real-weight two-rank DDP run on one Mac (row T5). **Blockers:** (1) a real two-GPU CUDA run — the smoke exists and is dispatchable (`iris_two_card_distributed_smoke`: DDP cancel → resume with the continuation assertion, a killed rank failing the job typed with no rank left holding a card, FSDP to completion, global batch 2, each rank's card read from the run itself). **Attempted 2026-10-13 and refused by its own preflight** (row D2): card 0 is still occupied, so nothing was built and no rank started. Blocker: card 0 free. (2) multi-host MLX needs a second Mac free at the same time. |
| AT6 | Memory-cap admission, task switching, source unavailability, incomplete/wrong-task artifacts, invalid training targets, no-valid-depth pixels, canceled tiles/updates, failed exports; ordinary CI and regression behaviour for existing paths | **closed (unit / job-path tests, local; CI pending the push)** | All on this branch; `npm run rust:check` green locally at 28c764b2d (macOS/MLX), with `rust:check:candle` and `rust:check:neither` typechecks; the PR's required CI runs at the final push. Memory admission: `iris_training.rs` `tests` + `job_tests` (under-floor, unreadable host, no row, per-row provenance, the checkpointing hint and per-lane gate). Missing backbone / encoder / teacher / training-resource refusals before load; cancel before and after the first step. Task switching and eviction: `image_jobs/iris.rs` `run_iris_task_switch`, `an_iris_training_run_evicts_a_resident_generator_or_transform_before_loading`. Wrong-task artifacts: `iris_restore/tests.rs` `resolution_refuses_a_wrong_task_export`; inference `restoration_parity::wrong_task_artifacts_are_refused`. Canceled tiles: inference `cancel_between_tiles_is_typed_and_yields_no_image`. **No valid depth pixels** (cddae1d4f): inference — `depth_jobs/tests.rs` `an_estimate_with_no_valid_depth_pixels_is_refused_and_publishes_nothing` (a product defect fixed in the same commit: an all-NaN estimate used to be published as an asset; it is now refused `[iris_depth_no_valid_pixels]` before anything is staged); training — `iris_training.rs` `a_depth_dataset_with_no_valid_depth_pixels_is_refused_and_publishes_nothing`. **Failed exports**, every objective: `a_failed_export_reports_the_failure_and_registers_nothing_for_every_objective` (worker), `a_failed_task_export_registers_no_model_and_leaves_its_sources_untouched` and `a_failed_generation_export_registers_nothing_and_leaves_its_sources_untouched` (API). Existing image / upscale / control / training paths: the same workspace test run. Not claimed: any of this on an installed build. |
| AT7 | After inference lands and the single final pin is stable: one terminal capability/memory/calibration campaign over the new registered routes; final refs and installed-product outcomes; revisions, hashes, settings, devices, outputs, URLs | **open** | Landed: the Iris memory-strategy contracts (inference #1239 MLX, #1240 Candle) and their SceneWorks absorption + catalog wiring on this branch — `measure-memory-catalog.mjs --list` now plans **six Iris cells** (generation, depth, restoration × MLX, Candle). MLX cells **rehearsed** on Mac2: depth [38396033589](https://github.com/SceneWorks/SceneWorks/actions/runs/38396033589) 7.33 GiB, restoration [38398047045](https://github.com/SceneWorks/SceneWorks/actions/runs/38398047045) 11.53 GiB, generation [38399806219](https://github.com/SceneWorks/SceneWorks/actions/runs/38399806219) 21.60 GiB. Training floors: MLX final (below), Candle final after the memory fix of inference #1242 (below). **Blockers:** (1) the pin is not final (one more inference PR), so the rehearsals are not the campaign — the real six-cell campaign and the capability dumps at the final pin are still to run (the dumps on this branch are labelled 0b1a5a57f; structurally complete, re-dump at the final pin); (2) installed-product outcomes (below); (3) no 16 GB card exists here and nothing emulates a small card's real allocation (only admission caps) — the adapter rows that would admit one are declared, not measured. |
| — | Installed packaged-build acceptance | **open** | Not run. **Blocker:** needs a packaged Apple Silicon build and an installed NVIDIA worker driven by hand: install the three tasks through Model Manager, run each, save/reopen jobs, settings, assets and raw-depth exports, restart and read a saved depth asset through a separate consumer. Every run in this ledger is a test binary on the job path — a different evidence type. |
| — | 16 GB card | **open** | No 16 GB card on any runner. **Blocker:** hardware. |

## Runs

**Peak (Mac2) = process-tree phys_footprint in GiB**, read in bytes from `proc_pid_rusage` by
`scripts/ci/iris-mlx-real-weights/tree_guard.py`: `kernel max` is the root process's
`ri_lifetime_max_phys_footprint` (exact, no sampling); `sampled` is the largest tree sum seen at a
0.25 s poll. Footprint includes MLX's buffer cache, so it can exceed the run's need; it is what the
process held, not a floor. CUDA smoke runs record no peak (the floors come from the probe runs).

Mac2 jobs ran one at a time, each far under the 60-minute ceiling: [38320809891](https://github.com/SceneWorks/SceneWorks/actions/runs/38320809891) 6 min 28 s (all
eight smokes, incl. a 244 s build), [38321534896](https://github.com/SceneWorks/SceneWorks/actions/runs/38321534896) 1 min 55 s, [38321764171](https://github.com/SceneWorks/SceneWorks/actions/runs/38321764171) 43 s (2026-10-12,
wip 415dc9308, pin 25231d15b); [38411944085](https://github.com/SceneWorks/SceneWorks/actions/runs/38411944085) 4 min 33 s and [38412625724](https://github.com/SceneWorks/SceneWorks/actions/runs/38412625724) 44 s (2026-10-13,
wip 80db2b133, pin c1f594203). Host before each: 128 GiB, 93–95 % free.

| # | AT | What ran | Machine | Run | Result | Peak (GiB) |
|---|---|---|---|---|---|---|
| J1 | AT1 | Worker `image_generate`, native prompt batch 2×768², 20 steps | Mac2 (MLX) | [38320809891](https://github.com/SceneWorks/SceneWorks/actions/runs/38320809891) | pass | 20.740 kernel max |
| J2 | AT1, AT3 | Worker `image_depth`, EXIF-6 photo 1024×768 → upright 768×1024, raw NPY + preview; hub holds only speridlabs/iris-3b (no Qwen) | Mac2 (MLX) | 38320809891 | pass; range [-1.023, 0.996] | 9.132 kernel max |
| J3 | AT1 | Worker `image_upscale` engine=iris, 4× (6 tiles, 2048×1536) + 1× (2 tiles); no Qwen in hub | Mac2 (MLX) | 38320809891 | pass | 10.987 kernel max |
| J10 | AT1, AT4 | NVIDIA worker smokes 8/8 (generation + controls + LoRA use, batch, depth, restore 4×/1×, LoRA training, depth full fine-tune, restoration LoRA, two-rank DDP on one card) | CUDA card 1 | [38273954419](https://github.com/SceneWorks/SceneWorks/actions/runs/38273954419) (pin 25231d15b) | pass | not recorded |
| T1 | AT4 | Training job: generation LoRA (8 steps, 256²) → adapter → generate | Mac2 (MLX) | 38320809891 | pass; adapter Δ 84.7 | 17.111 kernel max |
| T2 | AT4 | Generation LoKr, `save_every` 2: first durable checkpoint at step 2; cancel requested once it existed and landed at step 4; resumed (`resumeFromJobId` translation) → generate. Continuation asserted from the resumed job's own progress | Mac2 (MLX) | 38320809891 | pass; `canceled_at_step=4 first_checkpoint_step=2 resumed_first_step=5 resumed_steps_executed=4 resumed_to=8`; Δ 12.1 | 42.689 kernel max (39.020 sampled) |
| T3 | AT4 | Same LoKr cancel → resume → generate, with the continuation assertion, at pin c1f594203 | CUDA card 1 | [38406592757](https://github.com/SceneWorks/SceneWorks/actions/runs/38406592757) (wip 23f7ac1d8; smoke job 7 min 17 s) | pass; gate read the 24 GiB declared row against 95.59 GiB; `canceled_at_step=4 first_checkpoint_step=2 resumed_first_step=5 resumed_steps_executed=4 resumed_to=8`; Δ 20.7 | not recorded |
| T4 | AT4 | Depth full fine-tune job (128², 4 steps) → registered export → depth job selects it; no Qwen in hub. At pin c1f594203 also the gradient-checkpointing question: the MLX trainer refuses it and the job path refuses the toggle before load | Mac2 (MLX) | [38411944085](https://github.com/SceneWorks/SceneWorks/actions/runs/38411944085) | pass; `checkpointing_lane=mlx engine_accepts=false checkpointed_run=refused_before_load` | 58.917 kernel max |
| T4c | AT4 | T4 on Candle, where the trainer accepts gradient checkpointing: a second short job WITH the toggle on, admitted at the `full_ac` row | CUDA card 1 | [38417641518](https://github.com/SceneWorks/SceneWorks/actions/runs/38417641518) (wip 28c764b2d; smoke job 10 min 16 s) | pass; depth job selects the trained export; `checkpointing_lane=candle engine_accepts=true checkpointed_run=pass checkpointed_steps=2`, the checkpointed job admitted at the 68 GiB `full_ac` row (declared at 16 384 px) against 95.59 GiB | not recorded |
| T5 | AT4, AT5 | Two-rank DDP generation LoRA (`share_device`, one Mac) → generate | Mac2 (MLX) | 38320809891 | pass; Δ 10.5 | 43.944 sampled tree sum over 3 processes (upper bound 61.044 = sum of per-process kernel maxima) |
| T6 | AT4 | Restoration LoRA job (256², full recipe incl. critic / LPIPS) → export → restore job selects it; no Qwen | Mac2 (MLX) | 38320809891 | pass; gate read the 28 GiB declared row | 30.121 kernel max |
| T7 | AT4 | Generation FULL fine-tune, trainer level: official stage1_256 + REPA → stage2_512 new-phase resume, Muon | Mac2 / CUDA | inference 38179054198 (MLX) / 38233658960 (CUDA) | pass | trainer level only — superseded by T9 / T9c |
| T8 | AT4 | Restoration paper FULL fine-tune 256² (train → export → restore), trainer level | Mac2 / CUDA | inference 38145886212 / 38149768393 | pass | trainer level only — superseded by T10 / T10c |
| T9 | AT4 | **Generation FULL fine-tune through the job path:** the engine's official `stage1_256` preset (REPA against DINOv2-B/14, Muon), bounded to batch 1 / 8 steps (LR 1e-3, no warmup, EMA off). Admitted at the full row: 64 GiB + 4 GiB teacher = 68 GiB (extrapolated) against 128 GiB. Canceled after the step-4 checkpoint, resumed with the continuation assertion, exported, the exported checkpoint rendered by a generation job as a registered trained model | Mac2 (MLX) | [38321534896](https://github.com/SceneWorks/SceneWorks/actions/runs/38321534896) | pass; `canceled_at_step=5 first_checkpoint_step=4 resumed_first_step=6 resumed_steps_executed=3 resumed_to=8`; `irisPlan.stage = stage1_256` with REPA; render Δ 101.6 vs the release weights, std 62.5 | 61.049 kernel max (60.348 sampled) |
| T9c | AT4 | T9 on Candle | CUDA card 1 | [38412227879](https://github.com/SceneWorks/SceneWorks/actions/runs/38412227879) (wip 80db2b133; smoke job 14 min 51 s for T9c + T10c) | pass; gate read 79 GiB + 4 GiB teacher = 83 GiB (extrapolated) against 95.59 GiB; `canceled_at_step=6 first_checkpoint_step=4 resumed_first_step=7 resumed_steps_executed=2 resumed_to=8`; checkpoint render Δ 114.9 vs the release weights, std 25.3. The REPA teacher (facebook/dinov2-base @f9e44c814b77) was present in the box's hub | not recorded |
| T10 | AT4 | **Restoration FULL fine-tune through the job path** (256² center crop, the paper recipe incl. critic / LPIPS, 2 optimizer steps) → export → restore job selects it by id; no Qwen in hub. Admitted at the full row: 69 GiB (extrapolated, bounded by the measured 512² row) against 128 GiB. At pin c1f594203 also the gradient-checkpointing question, as T4 | Mac2 (MLX) | [38321764171](https://github.com/SceneWorks/SceneWorks/actions/runs/38321764171); re-run at the new pin [38412625724](https://github.com/SceneWorks/SceneWorks/actions/runs/38412625724) | pass both; at the new pin `checkpointing_lane=mlx engine_accepts=false checkpointed_run=refused_before_load` | 81.342 / 81.116 kernel max — above the 69 GiB floor: the footprint includes MLX's cache and the follow-up restore job in the same process; the floor is the cache-off active peak + 2 (see the emulated-limit evidence below) |
| T10c | AT4 | T10 on Candle, plus a second short job WITH gradient checkpointing on, admitted at the `full_ac` row | CUDA card 1 | 38412227879 | pass; gate read the 73 GiB measured row against 95.59 GiB; restore job selects the trained export; `checkpointing_lane=candle engine_accepts=true checkpointed_run=pass checkpointed_steps=2`, the checkpointed job admitted at the 73 GiB `full_ac` row (extrapolated) | not recorded |
| D1 | AT5 | FSDP full / grad_op vs DDP, sharded checkpoint resume + reshard, rank exit / stall → typed failure, multi-process on ONE card | CUDA card 1 | inference 38233658960 | pass | one device only |
| D2 | AT5 | Two-card distributed smoke through the job path (`iris_two_card_distributed_smoke`) | CUDA cards 0 + 1 | [38422258699](https://github.com/SceneWorks/SceneWorks/actions/runs/38422258699) (dispatched with `iris_smoke_cards=0,1`) | **not run — refused by the card preflight in 15 s, before any fetch or build:** "card 0 (NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition): 95262 MiB used of 97887 MiB — OCCUPIED by another tenant (over the 2048 MiB free-card threshold). The two-card smoke needs card 0 free; nothing was built and no rank was started." Card 1: 324 MiB used — free. Blocker: card 0 free | — |

Two Mac2 dispatches at wip 23f7ac1d8 ([38410831839](https://github.com/SceneWorks/SceneWorks/actions/runs/38410831839), [38411624596](https://github.com/SceneWorks/SceneWorks/actions/runs/38411624596)) failed in the new
checkpointing probe — a defect of the probe (it loaded a task trainer with the generation path's
load inputs), not of the product; the training and follow-up jobs in those runs had passed. Fixed
in bc6419be9 and re-run green (T4, T10). No CUDA smoke job failed; the two-card dispatch ended `failure` by design at its preflight (D2).

## Training-floor measurements

Probe: `train_memory_probe` — full fine-tune, batch 1, whole-process peak; summaries print decimal
GB, converted here to GiB (÷ 1.073741824). Floor = ⌈peak GiB + 2⌉ on both lanes.

**MLX (final).** Mac2, inference PR #1238 @b1c04bd0d, runs [38312190794](https://github.com/SceneWorks/inference/actions/runs/38312190794) (two steps + export)
and [38318086154](https://github.com/SceneWorks/inference/actions/runs/38318086154) (six steps). The quantity is the **active peak with MLX's buffer cache
disabled** — what the run needs. It is steady by optimizer step 2 and flat through step 6, and the
process footprint with the cache off agrees within 0.5 GB. Production runs keep MLX's cache, so
process footprint can rise to the allocator GC ceiling (0.95 × recommended working set: 109.68 GB
= 102.2 GiB on Mac2) without that being need; the first campaign's MLX numbers (run 38292137908:
109.96 / 110.08 GB at 1024²) were that ceiling, not peaks.

*Smaller-Mac evidence* (inference run [38322999659](https://github.com/SceneWorks/inference/actions/runs/38322999659), Mac2 with MLX's memory / cache / GC limits
set to what a 64 GiB and a 96 GiB Mac would get, default cache, six steps): every probed shape
completed with the footprint held at the emulated limit and the active peak unchanged — on the
64 GiB limits (48.96 / 54.84 GB) depth 256² active 49.12–49.35 GB, footprint ≤ 55.77 GB, and depth
512² active 52.78–53.60 GB, footprint ≤ 55.30 GB; on the 96 GiB limits (73.44 / 82.26 GB)
generation 512² active 65.57 GB, restoration 512² 71.57 GB, generation 1024² + AC 74.20–74.88 GB,
depth 1024² 91.59 GB (footprint 92.14 GB = 85.8 GiB). This is allocator-limit emulation on a
128 GiB machine, not a run on a smaller Mac: it shows the cache yields to the limit; it does not
show behaviour under real memory pressure from other processes.

**Candle (final).** RTX PRO 6000 card 1 (95.59 GiB), inference run [38393679427](https://github.com/SceneWorks/inference/actions/runs/38393679427) @1449045c8:
the **live pool peak** over six optimizer steps, steady from step 2 — after the training-tape
memory fix of inference #1242, which removed a 7–39 GB excess (live pool peaks before → after,
GB: generation 512² 95.40 → 82.41; generation 1024² + AC 89.54 → 82.58; depth 512² 83.38 → 70.41;
restoration 512² 126.28 → 87.36, which did not fit the card before and does now; "before" from
CUDA run 38312194527). Three shapes still need more than the card and completed only by spilling
to system memory (115–157 s/step for generation): their rows carry the measured need under the
provenance `measured_spill` and refuse the card; they are not capped to it. #1242 also added
activation checkpointing to the Candle depth and restoration trainers; their 1024² `full_ac` rows
are measured.

| Shape | MLX active peak, cache off | MLX floor (GiB) | Candle live pool peak | Candle floor (GiB) |
|---|---|---|---|---|
| Generation full 512² | 65.57 GB = 61.07 GiB | 64 measured | 82.41 GB = 76.75 GiB | 79 measured |
| Generation full 512² + AC | — | 64 extrapolated | 69.19 GB = 64.44 GiB | 67 measured |
| Generation full 1024² | 104.49 GB = 97.31 GiB | 100 measured | 128.75 GB = 119.91 GiB, spilling 29.2 GB | 122 measured_spill (does not fit the card) |
| Generation full 1024² + AC | 74.88 GB = 69.74 GiB | 72 measured | 82.58 GB = 76.91 GiB | 79 measured |
| Depth full 512² | 53.60 GB = 49.92 GiB | 52 measured | 70.41 GB = 65.57 GiB | 68 measured |
| Depth full 1024² | 92.58 GB = 86.22 GiB | 89 measured | 116.77 GB = 108.75 GiB, spilling 17.2 GB | 111 measured_spill (does not fit the card) |
| Depth full 1024² + AC | not implemented on MLX at this pin | — | 70.56 GB = 65.71 GiB | 68 measured (new capability) |
| Restoration full 256² crop | — | 69 extrapolated | 76.15 GB = 70.92 GiB | 73 measured |
| Restoration full 512² crop | 71.57 GB = 66.65 GiB | 69 measured | 87.36 GB = 81.36 GiB | 84 measured |
| Restoration full 1024² crop | lower bound only: footprint 116.47 GB = 108.47 GiB at the guard stop with the cache off (live need above 115 GB; the active peak had reached ≥ 87.45 GB by step 2 of 4) | 128 lower_bound | 142.96 GB = 133.14 GiB, spilling 42.5 GB | 136 measured_spill (does not fit the card) |
| Restoration full 1024² crop + AC | not implemented on MLX at this pin | — | 86.47 GB = 80.53 GiB | 83 measured (new capability) |

## Floors (`crates/sceneworks-core/src/training/iris{,/depth,/restore}.rs`)

- Floors are GiB and each row carries its own provenance — `measured`, `measured_spill`,
  `extrapolated`, `lower_bound` or `declared` — published per row (`memoryFloors.provenance`) and
  named in every refusal, the curriculum check and the Studio hint. There is no table-wide flag.
- A measured row is the probe's peak in the unit it printed (the `*_PEAK` constants), converted to
  GiB, plus a 2 GiB reserve, rounded up. A Candle row that ran without spilling never exceeds the
  card that held its run; a spilled measurement is never held to the card (that does not
  compile). The MLX reserve covers the ≤ 0.5 GB the footprint sat above the active peak plus
  allocator granularity; the macOS/app reserve (4 GiB) is separate. Extrapolated rows are computed
  from the same constants: a smaller run is bounded by the measured larger one; a checkpointed
  run by the lower of its plain row and the next measured `full_ac` row; depth between its two
  measured sizes by the lane's own chord.
- Full tables, GiB (m = measured, s = measured_spill, e = extrapolated, lb = lower_bound,
  d = declared):

  | Table | Rows | MLX | Candle |
  |---|---|---|---|
  | Generation `full` | 256 / 512 / 1024 | 64 e / 64 m / 100 m | 79 e / 79 m / 122 s |
  | Generation `full_ac` | 256 / 512 / 1024 | 64 e / 64 e / 72 m | 67 e / 67 m / 79 m |
  | Generation `adapter` | 256 / 512 / 1024 | 16 / 24 / 40 d | 24 / 32 / 48 d |
  | Depth | 16 384 / 262 144 / 307 200 / 589 824 / 1 048 576 px | 52 e / 52 m / 54 e / 68 e / 89 m | 68 d / 68 m / 71 e / 86 e / 111 s |
  | Depth `full_ac` | same sizes | — (not implemented) | 68 d / 68 e / 68 e / 68 e / 68 m |
  | Restoration `full` | 65 536 / 262 144 / 1 048 576 px | 69 e / 69 m / 128 lb | 73 m / 84 m / 136 s |
  | Restoration `full_ac` | same sizes | — (not implemented) | 73 e / 83 e / 83 m |
  | Restoration `adapter` | same sizes | 28 / 36 / 56 d | 32 / 40 / 60 d |

  Before the campaign (the PR's first commit, decimal-GB footprints compared against GiB
  capacity) the full rows were: generation MLX 80 / 85 / 110, Candle 80 / 100 / 125; depth MLX
  64 / 80 / 96 / 112, Candle 68 / 90 / 104 / 126; restoration MLX 92 / 102 / 128, Candle
  94 / 104 / 140. The earlier claim "Candle restoration full 512² measured 95" (a card-capped
  reading) is gone. No row is declared below a measured peak.
- **Gradient (activation) checkpointing for depth and restoration** is the shared
  `advanced.gradientCheckpointing` toggle, a per-lane capability. The engine publishes no
  declaration of it (the trainer descriptor has no such flag; a lane that lacks it refuses in
  `Trainer::validate`, which needs the weights), so `iris_gradient_checkpointing_lanes` is an
  explicit table — Candle at this pin — published on the target
  (`limits.gradientCheckpointingLanes`) and read by the Studio, the submit and the worker. The
  real-weight depth and restoration smokes ask the lane's own trainer and fail if its answer
  differs from the table, so the MLX lane starting to accept it cannot go unnoticed (rows T4,
  T10: MLX refuses at this pin). A lane without it refuses the toggle before load; a run refused
  at its plain row is told when the toggle admits it at the `full_ac` row. Generation keeps its
  richer `advanced.iris.activation_checkpointing` policy option.
- Inference-task manifest floors are owned by the memory-catalog work (AT7), not by this section.
