# Epic sc-25678 (Iris-3B) — terminal evidence campaign (sc-25690), 2026-10-12

Revisions: SceneWorks `feature/sc-25678-iris-3b` @437296bf8 + story branch `story/sc-25690-epic-25678-iris-3b`
(f1710f48f / 513c02b72 / 8c8f2b728), inference pin 25231d15b; inference probe branch
`claude/sc-25690-iris-memory-probe` (test + dispatch-only CI only, no runtime crate change).
Machines: Mac2 = nax-macos-2 (Apple M5 Max, 128 GiB, runner check enforced before any build);
CUDA = RTX PRO 6000 Blackwell 96 GB box — card 0 had 95.3 GB used by another tenant, every run
here used card 1 (96.6 GB free). The dev Mac GPU was not used (CPU-only compile/unit tests locally).
Mac2 wall-clock used: about 45 min over 4 serial jobs (11 + 9 + 6 + ~8 min), each well under 60 min.

| AT | What ran | Machine | Run URL | Result / peak | Status |
|---|---|---|---|---|---|
| AT1/AT4 | Worker job path: `image_generate` native prompt batch 2×768², 20 steps | Mac2 (MLX) | https://github.com/SceneWorks/SceneWorks/actions/runs/38289099623 | pass; peak 21.0 GB | closed (job path) |
| AT1/AT4 | Worker `image_depth` EXIF-6 photo, raw NPY + preview, hub holds ONLY speridlabs/iris-3b (no Qwen) | Mac2 (MLX) | 38289099623 | pass; 9.13 GB; range [-1.023, 0.996] | closed |
| AT1/AT4 | Worker `image_upscale` engine=iris 4× (6 tiles) + 1×, no Qwen in hub | Mac2 (MLX) | 38289099623 | pass; 11.0 GB | closed |
| AT4 | Training job path: gen LoRA (8 steps, 256²) → adapter → generate | Mac2 (MLX) | https://github.com/SceneWorks/SceneWorks/actions/runs/38291366533 | pass; 17.0 GB; adapter Δ 84.7 | closed |
| AT4 | **New** gen LoKr canceled via job API after first durable checkpoint (step 4) → resumed (resumeFromJobId translation) to 8 → generate | Mac2 (MLX) | 38291366533 | pass; 36.0 GB; Δ 12.1 | closed |
| AT4 | Same LoKr cancel → resume → generate | CUDA card 1 | https://github.com/SceneWorks/SceneWorks/actions/runs/38294903430 | pass; canceled at 4, resumed to 8; Δ 20.7 | closed |
| AT4 | Depth full FT job (128², 4 steps) → registered export → depth job selects it; no Qwen in hub | Mac2 (MLX) | 38291366533 | pass; 59.0 GB | closed |
| AT4 | Restoration LoRA job (256², full recipe incl. critic/LPIPS) → export → restore job selects it; no Qwen | Mac2 (MLX) | 38291366533 | pass; 29.0 GB | closed |
| AT4/AT5 | Two-rank DDP gen LoRA (share_device) → generate | Mac2 (MLX) | 38291366533 | pass; 44.0 GB (process tree) | closed (single host) |
| AT1/AT4 | NVIDIA worker smokes 8/8 (gen + controls + LoRA use, batch, depth, restore 4×/1×, LoRA, depth FT, restore LoRA, 2-rank DDP) — reused | CUDA card 1 | https://github.com/SceneWorks/SceneWorks/actions/runs/38273954419 | pass | closed (job path) |
| AT4 | Gen FULL FT on real weights (official stage1_256+REPA → stage2_512 new-phase resume, Muon) — story evidence | Mac2 / CUDA | inference 38179054198 (MLX, 68.16 GB) / 38233658960 (CUDA) | pass | closed at trainer level; not through the SW job path |
| AT4 | Restoration paper FULL FT 256² (train→export→restore) — story evidence | Mac2 / CUDA | inference 38145886212 (87.5 GB) / 38149768393 | pass | closed at trainer level |
| Memory | Full-FT probes (2 steps + export, batch 1): gen 1024² 109.96, 1024²+AC 101.52, 512² 84.64; depth 1024² 110.08, 512² 72.65; restore 512² 101.13, 1024² >115 (guard stop at 116.75 = lower bound) | Mac2 (MLX, phys_footprint, decimal GB) | https://github.com/SceneWorks/inference/actions/runs/38292137908 | measured | closed except restore 1024² (lower bound only) |
| Memory | Same probes: gen 1024²+AC 92.2, 512² 97.6; depth 512² 85.9; restore 512² 101.8; gen/depth/restore 1024² without AC → CUDA OOM on 96 GB card | CUDA card 1 (VRAM over idle baseline) | https://github.com/SceneWorks/inference/actions/runs/38291037726 | measured / exceeds card | 1024² no-AC CUDA rows: extrapolated (hardware) |
| AT5 | Mechanism: FSDP full/grad_op vs DDP, sharded checkpoint resume+reshard, rank exit/stall → typed failure, multi-process on ONE card (share_device) | CUDA | inference 38233658960 | pass | open: real 2-GPU / multi-host (hardware) |
| AT5 | MLX ring/TCP multi-rank mechanism tests (CPU) + real-weight 2-rank DDP on one Mac | CI / Mac2 | inference CI; 38291366533 | pass | open: multi-host MLX needs a second Mac free (hardware/policy) |
| AT7 | `measure-memory-catalog.mjs --list` (unfiltered, both lanes) | local (no GPU) | — | **zero iris rows**: iris_3b has no memory-strategy contract / calibration fingerprint in the inference crates, so no plan anchor or adapter arm is possible | open (product decision: inference memory contract + pin) |
| AT7 | Capability dumps at final pin (capabilities.{mlx,candle}.json list iris engines) | CI / preflight | preflight 8c8f2b728 (capabilities.mlx.json vs fresh dump PASS) | pass | partial |
| — | 16 GB card evidence | — | — | no 16 GB card; no codebase VRAM-cap emulation of real allocation (only admission caps) | open (hardware) |

## Floors written (sc-25690 commit 8c8f2b728, `crates/sceneworks-core/src/training/iris{,/depth,/restore}.rs`)
- Generation full: MLX 512 85 (was 88), 1024 110 (was 104 — under-declared), new `full_ac` rows MLX 1024 102 / Candle 1024 95
  (fits a 96 GB card); Candle 512 100 (was 88 — under-declared); Candle 1024 no-AC 125 EXTRAPOLATED (was 104).
- Depth: Candle 307 200 px 90 (was 84 — under-declared); Candle 589 824 / 1 048 576 px 104 / 126 EXTRAPOLATED (were 100 / 116);
  MLX rows unchanged (measured values inside them).
- Restoration full: 262 144 px MLX 102 (was 104), Candle 104 (was 110); 1 048 576 px kept at 128 / 140 (above measured lower bounds).
- Activation checkpointing exists only for generation (depth and restoration trainers refuse it), so no AC rows there.
- Inference-task manifest floors (iris_3b 32 GB MLX / 24 GB Candle, depth admission, restore 16/14) left declared: measured
  smoke peaks (21.0 / 9.13 / 11.0 GB MLX) were at different geometries than the declared rows.
