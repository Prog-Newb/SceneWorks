//! Worker-side seams of Iris-3B generation training (epic sc-25678; S7 sc-25685 MLX, S8 sc-25686
//! Candle/CUDA).
//!
//! The trainer itself is `load_trainer("iris_3b", …)` — reached through the generic native training
//! executor in `training_jobs`, exactly like every other family. This module carries only what that
//! executor cannot know about Iris:
//!
//! * the **unused-tail** report — upstream's single-device ranged sampler walks only the first
//!   `640 · ⌊n / 640⌋` items of an `n > 640` dataset that is not a multiple of 640 (641–1279 items
//!   train on 640). The trainer prints that on stderr only, so SceneWorks computes it from the SAME
//!   `gen_core::iris::train::DataWalk` the trainer walks and surfaces it before launch (the API's
//!   data-walk endpoint) and in the job (progress message, structured log event and result);
//! * the **declared memory floor** gate — a routed Iris training lane is never admitted unbounded:
//!   a host under `sceneworks_core::training::iris::iris_training_memory_floor_gb` is refused
//!   before the model loads (and so is a host whose capacity cannot be read).

use gen_core::iris::train::{DataWalk, CANONICAL_CHUNKS};
use serde::Serialize;

/// The pre-launch view of the trainer's data walk for a dataset of `items` at `batch_size`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IrisDataWalkReport {
    pub items: usize,
    pub covered: usize,
    pub unused_items: usize,
    pub batch_size: usize,
    pub chunks: usize,
    /// The trainer's own run-start warning text (`DataWalk::unused_tail_warning`), or `None` when
    /// every item is trained on.
    pub warning: Option<String>,
}

/// Compute the Iris data walk for `items` dataset items at `batch_size`. `None` for an empty
/// dataset or a zero batch (the trainer refuses both before walking).
pub fn iris_data_walk_report(items: usize, batch_size: usize) -> Option<IrisDataWalkReport> {
    let walk = DataWalk::new(items, batch_size).ok()?;
    Some(IrisDataWalkReport {
        items: walk.items,
        covered: walk.covered,
        unused_items: walk.unused_items(),
        batch_size: walk.batch_size,
        chunks: CANONICAL_CHUNKS,
        warning: walk.unused_tail_warning(),
    })
}

/// The memory lane this build trains on: `"mlx"` (Apple-Silicon unified memory) or `"candle"`
/// (NVIDIA VRAM).
pub(crate) fn training_lane() -> &'static str {
    if cfg!(target_os = "macos") {
        "mlx"
    } else {
        "candle"
    }
}

fn capacity_noun(lane: &str) -> &'static str {
    if lane == "mlx" {
        "unified memory"
    } else {
        "GPU memory (VRAM)"
    }
}

/// The admission decision for an Iris run against a host of `capacity_gb` on `lane`: `Some(message)`
/// to refuse, `None` to admit. Missing evidence REFUSES (the inverse of the generation fit gate's
/// fail-open): the floors are the only bound a training lane has, so an unreadable host or an
/// undeclared shape is never admitted unbounded.
pub(crate) fn iris_memory_refusal(
    lane: &str,
    network_type: &str,
    resolution: u32,
    capacity_gb: Option<f64>,
) -> Option<String> {
    let artifact = if network_type.trim().eq_ignore_ascii_case("full") {
        "a full fine-tune"
    } else {
        "an adapter (LoRA/LoKr) run"
    };
    let noun = capacity_noun(lane);
    let Some(floor) = sceneworks_core::training::iris::iris_training_memory_floor_gb(
        lane,
        network_type,
        resolution,
    ) else {
        return Some(format!(
            "Iris 3B training declares no memory floor for {artifact} at {resolution}px on the \
             {lane} lane, so it is refused rather than admitted without a bound."
        ));
    };
    let Some(capacity) = capacity_gb else {
        return Some(format!(
            "Iris 3B training needs at least {floor} GB of {noun} for {artifact} at \
             {resolution}px (a declared floor, not yet measured), but this machine's {noun} could \
             not be read, so the run is refused rather than admitted without a bound."
        ));
    };
    // Unified memory is shared with macOS and the other apps, so an MLX floor is judged against the
    // pool minus the same OS/app reserve the MLX generation gate holds out. (The Candle floors already
    // fold their allocator reserve into the declared row; dedicated VRAM has no OS share.)
    let reserve = if lane == "mlx" {
        crate::mlx_fit_gate::OS_APP_RESERVE_GB
    } else {
        0.0
    };
    let usable = capacity - reserve;
    (usable + 0.05 < f64::from(floor)).then(|| {
        let reserve_note = if reserve > 0.0 {
            format!(
                " (~{} GB once the {} GB macOS/app reserve is held out)",
                usable.max(0.0).round() as i64,
                reserve.round() as i64
            )
        } else {
            String::new()
        };
        format!(
            "Iris 3B training needs at least {floor} GB of {noun} for {artifact} at \
             {resolution}px (a declared floor, not yet measured), but this machine has ~{} GB\
             {reserve_note}. Lower the training resolution, train a LoRA/LoKr adapter instead of \
             a full fine-tune, or run on a machine with more memory.",
            capacity.round() as i64
        )
    })
}

/// [`iris_memory_refusal`] for any Iris objective: the plan's base model selects the objective's
/// declared floor table (generation: network type × resolution; depth: micro-batch pixels of the
/// training size; restoration: artifact × micro-batch crop pixels). Same refusal discipline: no
/// declared row or no capacity reading refuses; an MLX floor is judged against unified memory minus
/// the macOS/app reserve.
pub(crate) fn iris_plan_memory_refusal(
    lane: &str,
    base_model: &str,
    config: &sceneworks_core::training::TrainingConfig,
    capacity_gb: Option<f64>,
) -> Option<String> {
    use sceneworks_core::training::iris::{
        iris_objective_memory_floor_gb, iris_objective_shape, IrisObjective,
    };
    let objective = IrisObjective::from_base_model(base_model)?;
    if objective == IrisObjective::Generation {
        let network_type = config
            .advanced
            .get("networkType")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("lora");
        return iris_memory_refusal(lane, network_type, config.resolution, capacity_gb);
    }
    let shape = iris_objective_shape(objective, config);
    let noun = capacity_noun(lane);
    let Some(floor) = iris_objective_memory_floor_gb(objective, lane, config) else {
        return Some(format!(
            "Iris 3B training declares no memory floor for {shape} on the {lane} lane, so it is \
             refused rather than admitted without a bound. Lower the batch size or the training \
             size (raise gradient accumulation to keep the global batch), or train a LoRA/LoKr \
             where the objective offers one."
        ));
    };
    let Some(capacity) = capacity_gb else {
        return Some(format!(
            "Iris 3B training needs at least {floor} GB of {noun} for {shape} (a declared floor, not \
             yet measured), but this machine's {noun} could not be read, so the run is refused \
             rather than admitted without a bound."
        ));
    };
    let reserve = if lane == "mlx" {
        crate::mlx_fit_gate::OS_APP_RESERVE_GB
    } else {
        0.0
    };
    let usable = capacity - reserve;
    (usable + 0.05 < f64::from(floor)).then(|| {
        let reserve_note = if reserve > 0.0 {
            format!(
                " (~{} GB once the {} GB macOS/app reserve is held out)",
                usable.max(0.0).round() as i64,
                reserve.round() as i64
            )
        } else {
            String::new()
        };
        format!(
            "Iris 3B training needs at least {floor} GB of {noun} for {shape} (a declared floor, not \
             yet measured), but this machine has ~{} GB{reserve_note}. Lower the batch size or the \
             training size, train a LoRA/LoKr where the objective offers one, or run on a machine \
             with more memory.",
            capacity.round() as i64
        )
    })
}

/// The submit-time twin of [`iris_plan_memory_refusal`] for the co-located macOS API. `None` off
/// macOS (a Candle worker's card is not visible to the API; the worker gate decides there).
pub fn iris_plan_memory_error(
    base_model: &str,
    config: &sceneworks_core::training::TrainingConfig,
) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        iris_plan_memory_refusal("mlx", base_model, config, mlx_training_capacity_gb())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (base_model, config);
        None
    }
}

/// This Mac's unified-memory training budget (GiB), honoring the `SCENEWORKS_MLX_MEMORY_CAP_GB`
/// small-Mac emulation. `None` off macOS.
#[cfg(target_os = "macos")]
pub(crate) fn mlx_training_capacity_gb() -> Option<f64> {
    crate::mlx_fit_gate::resolve_budget(
        crate::mlx_fit_gate::probe_total_unified_memory_gib(),
        crate::mlx_fit_gate::mlx_memory_cap_gb(),
    )
    .map(|budget| budget.total_gb)
}

/// The selected NVIDIA card's VRAM (GB), honoring the `SCENEWORKS_CUDA_VRAM_CAP_GB` small-card
/// emulation.
#[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
pub(crate) async fn candle_training_capacity_gb(gpu_id: &str) -> Option<f64> {
    crate::vram_gate::apply_vram_cap(
        crate::gpu::nvidia_vram_budget_gb(gpu_id).await,
        crate::vram_gate::cuda_vram_cap_gb(),
    )
    .map(|budget| budget.total_gb)
}

/// The submit-time twin of the worker's gate for the co-located macOS API (the training analogue of
/// `full_finetune_memory_error`): `Some(message)` when this Mac is under the declared floor. `None`
/// off macOS — a Candle worker's card is not visible to the API, so the worker gate decides there.
pub fn iris_training_memory_error(network_type: &str, resolution: u32) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        iris_memory_refusal("mlx", network_type, resolution, mlx_training_capacity_gb())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (network_type, resolution);
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The upstream ranged walk: `n ≤ 640` uses everything; 641–1279 items train on 640; exact
    /// multiples use everything. These are the trainer's own numbers (`gen_core` `DataWalk`), not a
    /// SceneWorks re-derivation.
    #[test]
    fn the_unused_tail_matches_the_trainers_ranged_walk() {
        let full = iris_data_walk_report(640, 1).unwrap();
        assert_eq!((full.covered, full.unused_items), (640, 0));
        assert!(full.warning.is_none());

        let small = iris_data_walk_report(12, 4).unwrap();
        assert_eq!((small.covered, small.unused_items), (12, 0));
        assert!(small.warning.is_none());

        for (items, unused) in [(641, 1), (1000, 360), (1279, 639)] {
            let report = iris_data_walk_report(items, 1).unwrap();
            assert_eq!(report.covered, 640, "{items} items");
            assert_eq!(report.unused_items, unused, "{items} items");
            let warning = report.warning.expect("an unused tail warns");
            assert!(
                warning.contains(&format!("{unused} of {items}")),
                "the trainer's text names the tail: {warning}"
            );
        }
        let doubled = iris_data_walk_report(1280, 2).unwrap();
        assert_eq!((doubled.covered, doubled.unused_items), (1280, 0));

        assert!(iris_data_walk_report(0, 1).is_none());
        assert!(iris_data_walk_report(10, 0).is_none());
    }

    #[test]
    fn the_memory_gate_refuses_under_the_floor_and_without_evidence() {
        // MLX LoRA at 256: floor 16 GB, judged against unified memory minus the 4 GB OS/app
        // reserve — a 16 GB Mac (12 GB usable) is refused, a 24 GB Mac (20 GB usable) admitted.
        let sixteen = iris_memory_refusal("mlx", "lora", 256, Some(16.0))
            .expect("a 16 GB Mac is under the floor once the OS/app reserve is held out");
        assert!(sixteen.contains("reserve"), "{sixteen}");
        assert!(iris_memory_refusal("mlx", "lora", 256, Some(24.0)).is_none());
        assert!(iris_memory_refusal("mlx", "lora", 256, Some(64.0)).is_none());
        let under = iris_memory_refusal("mlx", "lokr", 256, Some(8.0)).expect("8 GB is under");
        assert!(
            under.contains("16 GB") && under.contains("unified memory"),
            "{under}"
        );
        // Candle names VRAM, and a full run's floor is far above an adapter's.
        let full = iris_memory_refusal("candle", "full", 256, Some(48.0)).expect("48 GB < full");
        assert!(
            full.contains("VRAM") && full.contains("full fine-tune"),
            "{full}"
        );
        assert!(iris_memory_refusal("candle", "lora", 256, Some(48.0)).is_none());
        // No reading, no declared row: refused, never admitted unbounded.
        assert!(iris_memory_refusal("mlx", "lora", 256, None).is_some());
        assert!(iris_memory_refusal("mlx", "lora", 4096, Some(512.0)).is_some());
    }

    /// The depth and restoration floors (sc-25687 / sc-25688): per objective and lane, with the MLX
    /// OS reserve, refusing beyond the declared table and without a capacity reading.
    #[test]
    fn task_objective_floors_gate_per_lane() {
        use sceneworks_core::training::iris::{depth, restore};
        let depth = depth::iris_depth_target().defaults;
        // 480×640, batch 1: MLX 80 (+4 reserve), CUDA 84.
        assert!(iris_plan_memory_refusal("mlx", "iris_3b_depth", &depth, Some(84.0)).is_none());
        let under = iris_plan_memory_refusal("mlx", "iris_3b_depth", &depth, Some(83.0))
            .expect("83 GB − 4 GB reserve < 80");
        assert!(
            under.contains("80 GB") && under.contains("reserve"),
            "{under}"
        );
        assert!(iris_plan_memory_refusal("candle", "iris_3b_depth", &depth, Some(84.0)).is_none());
        assert!(iris_plan_memory_refusal("candle", "iris_3b_depth", &depth, Some(83.0)).is_some());
        assert!(iris_plan_memory_refusal("mlx", "iris_3b_depth", &depth, None).is_some());
        let mut big = depth.clone();
        big.batch_size = 32;
        big.gradient_accumulation = 1;
        let refused = iris_plan_memory_refusal("mlx", "iris_3b_depth", &big, Some(512.0))
            .expect("no declared row for 32 × 480×640");
        assert!(refused.contains("declares no memory floor"), "{refused}");

        let mut restore = restore::iris_restore_target().defaults;
        restore.resolution = 256;
        // Full recipe at 256²: MLX 92 (+4), CUDA 94 (the measured peak on the 96 GB card).
        assert!(iris_plan_memory_refusal("mlx", "iris_3b_restore", &restore, Some(96.0)).is_none());
        assert!(iris_plan_memory_refusal("mlx", "iris_3b_restore", &restore, Some(95.0)).is_some());
        assert!(
            iris_plan_memory_refusal("candle", "iris_3b_restore", &restore, Some(95.6)).is_none()
        );
        restore
            .advanced
            .insert("networkType".into(), serde_json::json!("lora"));
        assert!(iris_plan_memory_refusal("mlx", "iris_3b_restore", &restore, Some(32.0)).is_none());
        let lora = iris_plan_memory_refusal("candle", "iris_3b_restore", &restore, Some(24.0))
            .expect("24 < 32");
        assert!(lora.contains("VRAM") && lora.contains("32 GB"), "{lora}");
        // Non-Iris base models are not this gate's.
        assert!(iris_plan_memory_refusal("mlx", "sdxl", &restore, Some(1.0)).is_none());
    }
}

/// The REAL training job path — `training_jobs::run_training_execution_using`: plan preflight, the
/// Iris memory floor + data walk, base/encoder resolution, the blocking drive, the progress/cancel
/// consumer and the completion/cancel result — with only the trainer injected. A plan built by the
/// real `sceneworks_core` target registry is what the job executes, so these pin that every
/// offered option reaches `Trainer::train`, and what the job API sees, per backend.
#[cfg(all(
    test,
    any(
        target_os = "macos",
        all(not(target_os = "macos"), feature = "backend-candle")
    )
))]
mod job_tests {
    use crate::api_client::ApiClient;
    use crate::error::{WorkerError, WorkerResult};
    use axum::extract::{Path as AxumPath, State};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use gen_core::{
        LoadSpec, Trainer, TrainerDescriptor, TrainingOutput, TrainingProgress, TrainingRequest,
        WeightsSource,
    };
    use sceneworks_core::contracts::JobSnapshot;
    use sceneworks_core::training::{
        build_training_plan, builtin_training_targets, BuildTrainingPlan, TrainingDataset,
        TrainingPlan,
    };
    use serde_json::{json, Value};
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const BACKBONE_REPO: &str = "speridlabs/iris-3b";
    const BACKBONE_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";
    const TE_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
    const TE_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Behavior {
        /// Run to completion: progress, a checkpoint, the export.
        Completes,
        /// Wait for the request's cancel flag, then stop the way the trainer does BEFORE its first
        /// optimizer step: the typed cancellation, nothing written.
        CancelBeforeFirstStep,
        /// Report a step, wait for the cancel flag, then publish the checkpoint + export and return
        /// `Ok` — the trainer's contract for a cancel after the first optimizer step.
        CancelAfterFirstStep,
    }

    #[derive(Default)]
    struct Seen {
        request: Option<TrainingRequest>,
        weights: Option<PathBuf>,
        text_encoder: Option<PathBuf>,
        /// Every staged component as `(id, path)` (depth / restoration task runs).
        components: Vec<(String, PathBuf)>,
        loads: u32,
    }

    struct FakeIrisTrainer {
        descriptor: TrainerDescriptor,
        behavior: Behavior,
        seen: Arc<Mutex<Seen>>,
        /// The job API's `cancelRequested`: the fake "user" clicks Cancel only once the trainer is
        /// mid-run, so the cancel reaches the training loop rather than the pre-start check.
        user_cancel: Arc<std::sync::atomic::AtomicBool>,
    }

    fn write_export(req: &TrainingRequest) -> PathBuf {
        std::fs::create_dir_all(&req.output_dir).unwrap();
        let checkpoint = req.output_dir.join(format!(
            "{}.checkpoints/step_00000001",
            Path::new(&req.file_name)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
        ));
        std::fs::create_dir_all(&checkpoint).unwrap();
        std::fs::write(checkpoint.join("state.json"), b"{}").unwrap();
        let adapter = req.output_dir.join(&req.file_name);
        std::fs::write(&adapter, b"adapter").unwrap();
        adapter
    }

    impl Trainer for FakeIrisTrainer {
        fn descriptor(&self) -> &TrainerDescriptor {
            &self.descriptor
        }
        fn validate(&self, _req: &TrainingRequest) -> gen_core::Result<()> {
            Ok(())
        }
        fn train(
            &mut self,
            req: &TrainingRequest,
            on_progress: &mut dyn FnMut(TrainingProgress),
        ) -> gen_core::Result<TrainingOutput> {
            self.seen.lock().unwrap().request = Some(req.clone());
            on_progress(TrainingProgress::Preparing);
            on_progress(TrainingProgress::LoadingModel);
            on_progress(TrainingProgress::Caching {
                current: 1,
                total: 1,
            });
            let user_cancel = Arc::clone(&self.user_cancel);
            let click_cancel = || user_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
            let wait_for_cancel = || {
                let deadline = std::time::Instant::now() + Duration::from_secs(30);
                while !req.cancel.is_cancelled() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                req.cancel.is_cancelled()
            };
            match self.behavior {
                Behavior::Completes => {
                    let accumulation = req.config.gradient_accumulation.max(1);
                    on_progress(TrainingProgress::Training {
                        step: accumulation,
                        total: req.config.steps,
                        loss: 0.25,
                    });
                    on_progress(TrainingProgress::Checkpoint { step: accumulation });
                    on_progress(TrainingProgress::Saving);
                    let path = write_export(req);
                    Ok(TrainingOutput {
                        adapter_path: path,
                        steps: req.config.steps,
                        final_loss: 0.25,
                    })
                }
                Behavior::CancelBeforeFirstStep => {
                    // Outlive the consumer's 2 s cancel-poll throttle with a Training event so the
                    // poll fires, but never complete an optimizer step.
                    std::thread::sleep(Duration::from_millis(2_300));
                    click_cancel();
                    on_progress(TrainingProgress::Training {
                        step: 0,
                        total: req.config.steps,
                        loss: f32::NAN,
                    });
                    assert!(wait_for_cancel(), "the worker tripped the cancel flag");
                    Err(gen_core::Error::Canceled)
                }
                Behavior::CancelAfterFirstStep => {
                    on_progress(TrainingProgress::Training {
                        step: 1,
                        total: req.config.steps,
                        loss: 0.5,
                    });
                    std::thread::sleep(Duration::from_millis(2_300));
                    click_cancel();
                    on_progress(TrainingProgress::Training {
                        step: 2,
                        total: req.config.steps,
                        loss: 0.4,
                    });
                    assert!(wait_for_cancel(), "the worker tripped the cancel flag");
                    let path = write_export(req);
                    Ok(TrainingOutput {
                        adapter_path: path,
                        steps: 2,
                        final_loss: 0.4,
                    })
                }
            }
        }
    }

    type Posts = Arc<Mutex<Vec<Value>>>;
    type CancelSwitch = Arc<std::sync::atomic::AtomicBool>;

    fn requested(cancel: &CancelSwitch) -> bool {
        cancel.load(std::sync::atomic::Ordering::SeqCst)
    }

    async fn spawn_job_api(cancel_requested: CancelSwitch) -> (String, Posts) {
        async fn job_route(
            State((_, cancel)): State<(Posts, CancelSwitch)>,
            AxumPath(job_id): AxumPath<String>,
        ) -> Response {
            Json(crate::tests::job_snapshot_json(&job_id, requested(&cancel))).into_response()
        }
        async fn progress_route(
            State((posts, cancel)): State<(Posts, CancelSwitch)>,
            AxumPath(job_id): AxumPath<String>,
            Json(body): Json<Value>,
        ) -> Response {
            posts.lock().unwrap().push(body);
            Json(crate::tests::job_snapshot_json(&job_id, requested(&cancel))).into_response()
        }
        async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Response {
            Json(json!({
                "id": worker_id, "gpuId": "cpu", "gpuName": null, "status": "busy",
                "currentJobId": "job-iris-train", "capabilities": [], "loadedModels": [],
                "registeredAt": "2026-07-01T00:00:00Z", "lastSeenAt": "2026-07-01T00:00:00Z"
            }))
            .into_response()
        }
        let posts: Posts = Arc::default();
        let app = Router::new()
            .route("/api/v1/jobs/:job_id", get(job_route))
            .route("/api/v1/jobs/:job_id/progress", post(progress_route))
            .route(
                "/api/v1/workers/:worker_id/heartbeat",
                post(heartbeat_route),
            )
            .with_state((posts.clone(), cancel_requested));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), posts)
    }

    fn stage(root: &Path, repo: &str, revision: &str, file: &str) -> PathBuf {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .unwrap()
            .join("snapshots")
            .join(revision);
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"x").unwrap();
        let refs = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .unwrap()
            .join("refs");
        std::fs::create_dir_all(&refs).unwrap();
        std::fs::write(refs.join("main"), revision).unwrap();
        snapshot
    }

    struct Install {
        backbone: PathBuf,
        text_encoder: PathBuf,
    }

    fn stage_install(data: &Path) -> Install {
        let mut backbone = PathBuf::new();
        for file in ["config.yaml", "model.safetensors"] {
            backbone = stage(data, BACKBONE_REPO, BACKBONE_REVISION, file);
        }
        let mut text_encoder = PathBuf::new();
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ] {
            text_encoder = stage(data, TE_REPO, TE_REVISION, file);
        }
        Install {
            backbone,
            text_encoder,
        }
    }

    /// A dataset of `items` images under the data dir, the first carrying named captions.
    fn dataset(data: &Path, items: usize) -> (TrainingDataset, PathBuf) {
        let root = data.join("projects/p1/training/datasets/ds1");
        std::fs::create_dir_all(root.join("images")).unwrap();
        let entries: Vec<Value> = (0..items)
            .map(|index| {
                let path = format!("images/item{index}.png");
                std::fs::write(root.join(&path), b"png").unwrap();
                let mut item = json!({
                    "id": format!("item{index}"),
                    "path": path,
                    "displayName": format!("item{index}"),
                    "caption": {"text": format!("a photo {index}"), "source": "manual", "triggerWords": []},
                    "addedAt": "2026-10-11T00:00:00Z",
                    "captions": {"short": format!("short {index}"), "tags": "red, square"}
                });
                if index > 0 {
                    item.as_object_mut().unwrap().remove("captions");
                    item["captions"] = json!({"short": format!("short {index}")});
                }
                item
            })
            .collect();
        let dataset: TrainingDataset = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds1", "version": 3, "projectId": "p1", "name": "ds",
            "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
            "updatedAt": "2026-10-11T00:00:00Z", "items": entries
        }))
        .unwrap();
        (dataset, root)
    }

    /// Build the plan the API would, from the REAL target registry, with `advanced` merged over the
    /// target defaults.
    fn plan(
        data: &Path,
        backbone: &Path,
        items: usize,
        configure: impl FnOnce(&mut sceneworks_core::training::TrainingConfig),
    ) -> TrainingPlan {
        let target = builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.kernel == "iris_train")
            .expect("the Iris target is registered");
        let mut config = target.defaults.clone();
        configure(&mut config);
        let (dataset, root) = dataset(data, items);
        let output = data.join("loras/lora_iris");
        build_training_plan(BuildTrainingPlan {
            job_id: "job-iris-train",
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: "lora_iris",
            base_model_path: backbone.display().to_string(),
            dataset_root: &root,
            output_dir: &output,
            file_name: "iris_style.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        })
        .expect("the plan builds")
    }

    struct Run {
        outcome: WorkerResult<()>,
        posts: Vec<Value>,
        seen: Seen,
    }

    /// Isolate the HF cache env and pin the emulated memory cap for the WHOLE test — staging the
    /// install resolves the cache path too, so the guard must be held before anything is staged.
    fn isolated_env(cap_gb: &str) -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
            #[cfg(target_os = "macos")]
            (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, cap_gb),
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            (crate::vram_gate::CUDA_VRAM_CAP_ENV, cap_gb),
        ])
    }

    fn run(behavior: Behavior, plan: TrainingPlan, data: &Path) -> Run {
        run_as("iris_3b", behavior, plan, data)
    }

    fn run_as(
        expected_engine: &'static str,
        behavior: Behavior,
        plan: TrainingPlan,
        data: &Path,
    ) -> Run {
        let seen = Arc::new(Mutex::new(Seen::default()));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (outcome, posts) = runtime.block_on(async {
            let user_cancel = CancelSwitch::default();
            let (base_url, posts) = spawn_job_api(Arc::clone(&user_cancel)).await;
            let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
            settings.api_url = base_url;
            settings.data_dir = data.to_path_buf();
            settings.heartbeat_seconds = 5;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json("job-iris-train", false))
                    .unwrap();
            let descriptor = crate::inference_runtime::trainer_descriptor(expected_engine)
                .expect("this backend links the Iris trainer");
            let loader_seen = Arc::clone(&seen);
            let user_cancel = Arc::clone(&user_cancel);
            let loader = move |engine: &str, spec: &LoadSpec| {
                assert_eq!(engine, expected_engine);
                let mut seen = loader_seen.lock().unwrap();
                seen.loads += 1;
                for (id, source) in &spec.components {
                    let path = match source {
                        WeightsSource::Dir(path) | WeightsSource::File(path) => path.clone(),
                    };
                    seen.components.push((id.clone(), path));
                }
                if let WeightsSource::Dir(dir) = &spec.weights {
                    seen.weights = Some(dir.clone());
                }
                if let Some(WeightsSource::Dir(dir)) = spec.components.get("text_encoder") {
                    seen.text_encoder = Some(dir.clone());
                }
                Ok(Box::new(FakeIrisTrainer {
                    descriptor,
                    behavior,
                    seen: Arc::clone(&loader_seen),
                    user_cancel,
                }) as Box<dyn Trainer>)
            };
            let outcome = tokio::time::timeout(
                Duration::from_secs(90),
                crate::training_jobs::run_training_execution_using(
                    &api, &settings, &job, &plan, loader,
                ),
            )
            .await
            .expect("the training job finished");
            let posts = posts.lock().unwrap().clone();
            (outcome, posts)
        });
        let seen = std::mem::take(&mut *seen.lock().unwrap());
        Run {
            outcome,
            posts,
            seen,
        }
    }

    fn terminal(posts: &[Value]) -> &Value {
        posts.last().expect("the job posted progress")
    }

    /// Every option the Studio offers reaches the trainer's request — the shared fields mapped onto
    /// `TrainingConfig`, the Iris bag verbatim as `model_options["iris"]` — and the trainer is loaded
    /// from the installed backbone with the generation install's text encoder staged.
    #[test]
    fn every_offered_option_reaches_the_trainer_and_the_export_completes() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let iris = json!({
            "init": "weights",
            "prediction": "v",
            "flow_shift": 3.0,
            "shift_law": "sd3",
            "shift_base_tokens": 256,
            "logit_mean": 0.0,
            "logit_std": 1.0,
            "x_pred_sigma_min": 0.05,
            "text_dropout": 0.15,
            "gradient_clip": 1.0,
            "betas": [0.9, 0.99],
            "muon_momentum": 0.9,
            "muon_nesterov": false,
            "muon_adjust_lr": "spectral_norm",
            "auto_lr": "sqrt",
            "base_batch_size": 64,
            "ema_enabled": true,
            "ema_decay": 0.999,
            "num_epochs": 3,
            "nan_loss_tolerance": 5,
            "caption_field": "caption",
            "caption_fields": ["short", "tags"],
            "on_caption_overflow": "error",
            "text_conditioning": "on_the_fly",
            "keep_last_checkpoints": 1,
            "milestone_steps": [10, 20],
            "preview_weights": "ema",
            "preview_negative_prompt": "blurry",
            "export_weights": "ema",
            "export_dtype": "bf16",
            "upstream_validation_prompts": false
        });
        let plan = plan(&data, &install.backbone, 3, |config| {
            config.optimizer = "muon".to_owned();
            config.gradient_accumulation = 2;
            config.steps = 40;
            config.save_every = 10;
            config.resolution = 512;
            config.batch_size = 2;
            config.rank = 8;
            config.alpha = 4;
            let advanced = &mut config.advanced;
            advanced.insert("networkType".into(), json!("lokr"));
            advanced.insert("decomposeFactor".into(), json!(4));
            advanced.insert("lrScheduler".into(), json!("cosine"));
            advanced.insert("lrWarmupSteps".into(), json!(6));
            advanced.insert("timestepType".into(), json!("uniform"));
            advanced.insert("mixedPrecision".into(), json!("f32"));
            advanced.insert("weightDecay".into(), json!(0.01));
            advanced.insert("sampleEvery".into(), json!(20));
            advanced.insert("sampleSteps".into(), json!(12));
            advanced.insert("sampleGuidanceScale".into(), json!(2.5));
            advanced.insert("sampleCount".into(), json!(2));
            advanced.insert(
                "samplePrompts".into(),
                json!(["a red fox", "a blue bird", "a third prompt"]),
            );
            advanced.insert("iris".into(), iris.clone());
        });
        let run = run(Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);

        assert_eq!(run.seen.loads, 1);
        assert_eq!(
            run.seen.weights.as_deref(),
            Some(install.backbone.as_path())
        );
        assert_eq!(
            run.seen.text_encoder.as_deref(),
            Some(install.text_encoder.as_path()),
            "the generation install's Qwen3-VL encoder is the trainer's text_encoder component"
        );
        let request = run.seen.request.expect("the trainer ran");
        let config = &request.config;
        assert_eq!(config.optimizer, "muon");
        assert_eq!(config.lr_scheduler, gen_core::LrSchedule::Cosine);
        assert_eq!(config.lr_warmup_steps, 6);
        assert_eq!(config.timestep_type, "uniform");
        assert_eq!(config.timestep_bias, "balanced");
        assert_eq!(config.loss_type, "mse");
        assert_eq!(config.train_dtype, "f32");
        assert_eq!(config.gradient_accumulation, 2);
        assert_eq!((config.steps, config.save_every), (40, 10));
        assert_eq!((config.resolution, config.batch_size), (512, 2));
        assert_eq!((config.rank, config.alpha), (8, 4.0));
        assert_eq!(config.network_type, gen_core::NetworkType::Lokr);
        assert_eq!(config.decompose_factor, 4);
        assert!(!config.full_finetune);
        assert!(!config.gradient_checkpointing);
        assert!((config.weight_decay - 0.01).abs() < 1e-6);
        assert_eq!(
            (config.sample_every, config.sample_steps),
            (20, 12),
            "preview cadence + steps"
        );
        assert!((config.sample_guidance_scale - 2.5).abs() < 1e-6);
        assert_eq!(
            config.sample_prompts,
            vec!["a red fox".to_owned(), "a blue bird".to_owned()],
            "the pool is capped at sampleCount"
        );
        assert!(!config.resume);
        assert_eq!(
            config.model_options.get("iris"),
            Some(&iris),
            "the Iris options bag reaches the trainer verbatim"
        );
        // Per-item named captions ride on the item.
        assert_eq!(request.items.len(), 3);
        assert_eq!(
            request.items[0].model_options.get("captions"),
            Some(&json!({"short": "short 0", "tags": "red, square"}))
        );
        assert_eq!(request.file_name, "iris_style.safetensors");
        assert!(request.output_dir.ends_with("loras/lora_iris"));

        let done = terminal(&run.posts);
        assert_eq!(done["status"], "completed", "{done:#}");
        let result = &done["result"];
        assert_eq!(result["kernel"], "iris_train");
        assert_eq!(result["networkType"], "lokr");
        assert_eq!(result["dataWalk"]["unusedItems"], 0);
        assert!(result.get("warnings").is_none(), "no tail ⇒ no warning");
        assert!(result.get("trainedArtifactExported").is_none());
    }

    /// The trainer's unused tail (641 items ⇒ 640 trained) reaches the job: a progress message, the
    /// result's `warnings` and `dataWalk`, before the trainer runs.
    #[test]
    fn the_unused_tail_is_reported_in_the_job() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 641, |_| {});
        let run = run(Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        let warned = run
            .posts
            .iter()
            .position(|post| {
                post["message"]
                    .as_str()
                    .is_some_and(|message| message.contains("1 of 641"))
            })
            .expect("the unused tail is posted as a job message");
        assert_eq!(run.posts[warned]["status"], "preparing");
        let result = &terminal(&run.posts)["result"];
        assert_eq!(result["dataWalk"]["covered"], 640);
        assert_eq!(result["dataWalk"]["unusedItems"], 1);
        assert!(result["warnings"][0]
            .as_str()
            .unwrap()
            .contains("never trained on"));
    }

    /// A host under the declared floor is refused BEFORE the loader runs; nothing is written.
    #[test]
    fn a_host_under_the_declared_floor_is_refused_before_load() {
        let _env = isolated_env("48");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, |config| {
            config.advanced.insert("networkType".into(), json!("full"));
        });
        let run = run(Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("full fine-tune"), "{message}")
            }
            other => panic!("a 48 GB host must be refused for a full run, got {other:?}"),
        }
        assert_eq!(run.seen.loads, 0, "refused before the loader ran");
        assert!(!data.join("loras/lora_iris/iris_style.safetensors").exists());
    }

    /// A cancel honoured before the first optimizer step ends Canceled with no result and nothing
    /// published.
    #[test]
    fn a_cancel_before_the_first_step_publishes_nothing() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, |_| {});
        let run = run(Behavior::CancelBeforeFirstStep, plan, &data);
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert!(last["result"].is_null(), "{last:#}");
        assert!(!data.join("loras/lora_iris/iris_style.safetensors").exists());
    }

    /// A cancel honoured after the first optimizer step: the trainer published the checkpoint and
    /// the export, so the terminal Canceled carries the result the API registers from.
    #[test]
    fn a_cancel_after_the_first_step_reports_the_published_export() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, |_| {});
        let run = run(Behavior::CancelAfterFirstStep, plan, &data);
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        let result = &last["result"];
        assert_eq!(result["trainedArtifactExported"], true, "{last:#}");
        assert_eq!(result["stepsCompleted"], 2);
        assert!(result["outputPath"]
            .as_str()
            .unwrap()
            .ends_with("iris_style.safetensors"));
        assert!(last["message"].as_str().unwrap().contains("step 2"));
        assert!(data
            .join("loras/lora_iris/iris_style.safetensors")
            .is_file());
    }

    /// Only Iris finishes a cancel durably. Another trainer that returns `Ok` after a cancel (SDXL
    /// saves a partial adapter once a step has run) still ends a plain, result-less Canceled, so the
    /// API registers nothing for it — the pre-Iris behaviour (E8). Drives the REAL event consumer.
    #[test]
    fn a_non_iris_trainer_returning_ok_after_cancel_publishes_no_result() {
        use crate::training_jobs::{consume_training_events, TrainEvent};
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let mut plan = plan(&data, &install.backbone, 2, |_| {});
        plan.target.kernel = "sdxl_lora".to_owned();
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (outcome, posts) = runtime.block_on(async {
            let user_cancel = CancelSwitch::default();
            let (base_url, posts) = spawn_job_api(Arc::clone(&user_cancel)).await;
            let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
            settings.api_url = base_url;
            settings.data_dir = data.clone();
            settings.heartbeat_seconds = 5;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json("job-iris-train", false))
                    .unwrap();
            let (tx, rx) = tokio::sync::mpsc::channel(16);
            let cancel = gen_core::CancelFlag::new();
            let flag = cancel.clone();
            let adapter = data.join("loras/lora_iris/partial.safetensors");
            let blocking = tokio::task::spawn_blocking(move || {
                let training = |step| {
                    TrainEvent::Progress(TrainingProgress::Training {
                        step,
                        total: 40,
                        loss: 0.5,
                    })
                };
                let _ = tx.blocking_send(training(1));
                // Outlive the consumer's 2 s cancel-poll throttle, then "click Cancel".
                std::thread::sleep(Duration::from_millis(2_300));
                user_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                let _ = tx.blocking_send(training(2));
                let deadline = std::time::Instant::now() + Duration::from_secs(30);
                while !flag.is_cancelled() && std::time::Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(10));
                }
                assert!(flag.is_cancelled(), "the worker tripped the cancel flag");
                let _ = tx.blocking_send(TrainEvent::Done(TrainingOutput {
                    adapter_path: adapter,
                    steps: 2,
                    final_loss: 0.4,
                }));
                Ok(())
            });
            let outcome = tokio::time::timeout(
                Duration::from_secs(90),
                consume_training_events(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    gen_core::TrainingConfig::default(),
                    "mlx",
                    40,
                    rx,
                    cancel,
                    blocking,
                    serde_json::Map::new(),
                ),
            )
            .await
            .expect("the consumer finished");
            let posts = posts.lock().unwrap().clone();
            (outcome, posts)
        });
        assert!(
            matches!(outcome, Err(WorkerError::Canceled(_))),
            "{outcome:?}"
        );
        let last = terminal(&posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert!(last["result"].is_null(), "{last:#}");
    }

    // ------------------------------------------------------------------------------------------
    // Iris DEPTH (S9 sc-25687) and RESTORATION (S10 sc-25688) task runs through the same executor.
    // ------------------------------------------------------------------------------------------

    /// Serializes the tests that substitute the pinned training-resource table.
    static RESOURCE_TABLE: Mutex<()> = Mutex::new(());

    fn stage_bytes(root: &Path, repo: &str, revision: &str, file: &str, bytes: &[u8]) -> PathBuf {
        let path = stage(root, repo, revision, file).join(file);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// The task export folder (`depth/` or `upscaler/`) of the pinned snapshot.
    fn stage_task_export(data: &Path, subdir: &str) -> PathBuf {
        let mut dir = PathBuf::new();
        for file in [
            "config.yaml",
            "model.safetensors",
            "empty_prompt.safetensors",
        ] {
            dir = stage(
                data,
                BACKBONE_REPO,
                BACKBONE_REVISION,
                &format!("{subdir}/{file}"),
            );
        }
        dir.join(subdir)
    }

    /// A task dataset of `items` images; each depth item has its depth map (+ a mask on item 0),
    /// each restoration item its aligned degraded input when `paired`.
    fn task_dataset(
        data: &Path,
        objective: &str,
        items: usize,
        paired: bool,
    ) -> (TrainingDataset, PathBuf) {
        let root = data.join("projects/p1/training/datasets/ds_task");
        std::fs::create_dir_all(root.join("images")).unwrap();
        std::fs::create_dir_all(root.join("sidecars")).unwrap();
        let entries: Vec<Value> = (0..items)
            .map(|index| {
                let path = format!("images/item{index}.png");
                std::fs::write(root.join(&path), b"png").unwrap();
                let mut item = json!({
                    "id": format!("item{index}"), "path": path,
                    "displayName": format!("item{index}"),
                    // A caption the task runs never read (and never send).
                    "caption": {"text": "ignored caption", "source": "manual", "triggerWords": []},
                    "addedAt": "2026-10-11T00:00:00Z"
                });
                if objective == "depth" {
                    let depth = format!("sidecars/item{index}.depth.npy");
                    std::fs::write(root.join(&depth), b"npy").unwrap();
                    item["irisDepth"] = json!({"path": depth, "scale": 0.001, "maxDepth": 80.0});
                    if index == 0 {
                        std::fs::write(root.join("sidecars/item0.mask.png"), b"png").unwrap();
                        item["irisDepth"]["maskPath"] = json!("sidecars/item0.mask.png");
                    }
                } else if paired {
                    let degraded = format!("sidecars/item{index}.degraded.png");
                    std::fs::write(root.join(&degraded), b"png").unwrap();
                    item["irisDegradedImage"] = json!(degraded);
                }
                item
            })
            .collect();
        let dataset: TrainingDataset = serde_json::from_value(json!({
            "schemaVersion": 1, "id": "ds_task", "version": 1, "projectId": "p1", "name": "ds",
            "modality": "image", "status": "ready", "createdAt": "2026-10-11T00:00:00Z",
            "updatedAt": "2026-10-11T00:00:00Z", "items": entries
        }))
        .unwrap();
        (dataset, root)
    }

    fn task_plan(
        data: &Path,
        base_model: &str,
        dataset: (TrainingDataset, PathBuf),
        configure: impl FnOnce(&mut sceneworks_core::training::TrainingConfig),
    ) -> TrainingPlan {
        let target = builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.base_model == base_model)
            .expect("the Iris task target is registered");
        let mut config = target.defaults.clone();
        configure(&mut config);
        let backbone =
            sceneworks_core::hf_home::huggingface_repo_cache_path(data, BACKBONE_REPO).unwrap();
        let output = data.join("models/finetunes/finetune_task");
        build_training_plan(BuildTrainingPlan {
            job_id: "job-iris-train",
            target: &target,
            dataset: &dataset.0,
            config,
            preset: None,
            lora_id: "finetune_task",
            base_model_path: backbone.display().to_string(),
            dataset_root: &dataset.1,
            output_dir: &output,
            file_name: "task_model.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        })
        .expect("the task plan builds")
    }

    /// Every depth option the Studio offers reaches the depth trainer: the shared fields (a FULL
    /// fine-tune, L1) and the `iris` bag verbatim as the ONLY `model_options` key; each item is its
    /// image + its depth target resolved under the dataset root, with no caption. The run loads the
    /// installed `depth/` export alone — no text encoder, no generation backbone.
    #[test]
    fn every_depth_option_reaches_the_depth_trainer_from_the_depth_export() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let export = stage_task_export(&data, "depth");
        let iris = json!({
            "depth_size": [352, 1216], "resize": "none", "crop": "random", "hflip_prob": 0.25,
            "gradient_weight": 4.0, "low_percentile": 1.0, "high_percentile": 99.0,
            "clip_target": true, "loss_reduction": "per_image", "invalid_examples": "skip",
            "min_valid_pixels": 16, "betas": [0.9, 0.99], "gradient_clip": 0.5,
            "muon_momentum": 0.9, "muon_nesterov": false, "muon_adjust_lr": "none",
            "auto_lr": "sqrt", "base_batch_size": 64, "ema_enabled": true, "ema_decay": 0.999,
            "num_epochs": 2, "nan_loss_tolerance": 3, "keep_last_checkpoints": 1,
            "milestone_steps": [5], "preview_items": 2, "preview_weights": "ema",
            "export_weights": "ema"
        });
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 2, false),
            |c| {
                c.optimizer = "muon".to_owned();
                c.gradient_accumulation = 4;
                c.steps = 40;
                c.save_every = 8;
                c.resolution = 1216;
                c.advanced.insert("sampleEvery".into(), json!(8));
                c.advanced.insert("lrWarmupSteps".into(), json!(4));
                c.advanced.insert("lrScheduler".into(), json!("cosine"));
                c.advanced.insert("mixedPrecision".into(), json!("f32"));
                c.advanced.insert("iris".into(), iris.clone());
            },
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(run.seen.weights.as_deref(), Some(export.as_path()));
        assert!(run.seen.components.is_empty(), "{:?}", run.seen.components);
        let request = run.seen.request.expect("the trainer ran");
        let config = &request.config;
        assert!(config.full_finetune);
        assert_eq!(config.loss_type, "l1");
        assert_eq!(config.optimizer, "muon");
        assert_eq!(
            (config.steps, config.save_every, config.sample_every),
            (40, 8, 8)
        );
        assert_eq!(
            (config.gradient_accumulation, config.lr_warmup_steps),
            (4, 4)
        );
        assert_eq!(config.train_dtype, "f32");
        assert_eq!(config.lr_scheduler, gen_core::LrSchedule::Cosine);
        assert!(config.sample_prompts.is_empty() && config.trigger_word.is_none());
        assert_eq!(
            Value::Object(config.model_options.clone()),
            json!({ "iris": iris }),
            "the depth bag is the only model option"
        );
        assert!(request.trigger_words.is_empty());
        let root = data.join("projects/p1/training/datasets/ds_task");
        let item = &request.items[0];
        assert_eq!(item.caption, "", "the depth task reads no caption");
        assert_eq!(
            Value::Object(item.model_options.clone()),
            json!({"depth": {
                "path": root.join("sidecars").join("item0.depth.npy"),
                "mask_path": root.join("sidecars").join("item0.mask.png"),
                "scale": 0.001, "max_depth": 80.0
            }})
        );
        assert!(request.items[1].model_options["depth"]
            .get("mask_path")
            .is_none());
        let done = terminal(&run.posts);
        assert_eq!(done["status"], "completed", "{done:#}");
        assert_eq!(done["result"]["dataWalk"]["items"], 2);
    }

    /// The paper's start (`depthInit: parent`): the generation backbone, conditioned on the depth
    /// export's SHIPPED empty prompt (no text encoder at all); without a depth install, the generation
    /// install's text encoder encodes it once.
    #[test]
    fn a_parent_depth_run_prefers_the_shipped_empty_prompt_over_the_encoder() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let export = stage_task_export(&data, "depth");
        let parent = |c: &mut sceneworks_core::training::TrainingConfig| {
            c.gradient_accumulation = 1;
            c.steps = 4;
            c.save_every = 2;
            c.advanced.insert("sampleEvery".into(), json!(2));
            c.advanced.insert("depthInit".into(), json!("parent"));
            c.advanced["iris"]["reducer_init"] = json!("channel_mean");
        };
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            parent,
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        assert_eq!(
            run.seen.weights.as_deref(),
            Some(install.backbone.as_path())
        );
        assert_eq!(
            run.seen.components,
            vec![(
                "empty_prompt".to_owned(),
                export.join("empty_prompt.safetensors")
            )]
        );

        // No depth install: the generation text encoder (its complete pinned closure), once.
        std::fs::remove_dir_all(&export).unwrap();
        let entry = crate::training_jobs::builtin_model_manifest_entry("iris_3b").unwrap();
        let encoder = entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .find(|download| download["componentId"] == "text_encoder")
            .unwrap();
        // The declared shard glob resolves against the shard actually present.
        for file in encoder["files"].as_array().unwrap() {
            let file = file.as_str().unwrap().replace('*', "00001-of-00002");
            stage(&data, TE_REPO, TE_REVISION, &file);
        }
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            parent,
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        assert_eq!(
            run.seen.components,
            vec![("text_encoder".to_owned(), install.text_encoder.clone())]
        );
    }

    /// Without the depth export an `export` run is refused BEFORE load, naming the install.
    #[test]
    fn a_depth_run_without_its_install_is_refused_before_load() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_install(&data);
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            |c| {
                c.gradient_accumulation = 1;
                c.steps = 4;
                c.save_every = 2;
                c.advanced.insert("sampleEvery".into(), json!(2));
            },
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(
                    message.contains("Iris 3B Depth is not installed"),
                    "{message}"
                )
            }
            other => panic!("refused before load, got {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);
    }

    /// The declared depth floor refuses an undersized host before load (MLX: unified memory minus
    /// the OS reserve; CUDA: VRAM).
    #[test]
    fn a_host_under_the_depth_floor_is_refused_before_load() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "depth");
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            |_| {},
        );
        let run = run_as("iris_3b_depth", Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => assert!(
                message.contains("depth full fine-tune at 480×640") && message.contains("GB"),
                "{message}"
            ),
            other => panic!("a 64 GB host must be refused for the 480×640 depth run: {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);
    }

    /// A depth cancel after the first optimizer step publishes the export like a generation run
    /// (Iris-only durable cancel): the terminal Canceled carries the result the API registers from.
    #[test]
    fn a_depth_cancel_after_the_first_step_reports_the_published_export() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "depth");
        let plan = task_plan(
            &data,
            "iris_3b_depth",
            task_dataset(&data, "depth", 1, false),
            |c| {
                c.gradient_accumulation = 1;
                c.steps = 4;
                c.save_every = 2;
                c.advanced.insert("sampleEvery".into(), json!(2));
            },
        );
        let run = run_as("iris_3b_depth", Behavior::CancelAfterFirstStep, plan, &data);
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert_eq!(last["result"]["trainedArtifactExported"], true, "{last:#}");
    }

    fn fixture_resource(
        data: &Path,
        component: &'static str,
        repo: &'static str,
        file: &'static str,
        bytes: &[u8],
    ) -> gen_core::iris::restoration_train::TrainingResource {
        use sha2::Digest;
        let revision = "0123456789abcdef0123456789abcdef01234567";
        stage_bytes(data, repo, revision, file, bytes);
        gen_core::iris::restoration_train::TrainingResource {
            component,
            name: component,
            url: "https://example.invalid/fixture",
            repo,
            revision,
            file,
            sha256: Box::leak(format!("{:x}", sha2::Sha256::digest(bytes)).into_boxed_str()),
            bytes: bytes.len() as u64,
            license: "fixture",
        }
    }

    /// Every restoration option reaches the trainer: HYPIR's LoRA arm (the recipe's full fine-tune
    /// is the default), the `iris_restore` bag as the ONLY model option, aligned pairs as each item's
    /// `degraded_image`, the three training-only resources staged as components after their pinned
    /// size + sha256 verify, and the generation backbone as `parent_backbone` under `init: parent`.
    #[test]
    fn every_restoration_option_reaches_the_trainer_with_verified_resources() {
        let _table = RESOURCE_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let export = stage_task_export(&data, "upscaler");
        let resources = vec![
            fixture_resource(
                &data,
                "lpips_vgg16",
                "fixture/vgg",
                "model.safetensors",
                b"vgg",
            ),
            fixture_resource(&data, "lpips_lin", "fixture/lin", "lin.safetensors", b"lin"),
            fixture_resource(
                &data,
                "critic_trunk",
                "fixture/trunk",
                "trunk.bin",
                b"trunk",
            ),
        ];
        *crate::iris_task_training::TEST_TRAINING_RESOURCES
            .lock()
            .unwrap() = Some(resources.clone());
        let bag = json!({
            "loss_l2": 1.0, "loss_lpips": 4.0, "loss_gan": 0.25, "gan_real_label": 0.9,
            "lr_critic": 2e-5, "betas": [0.9, 0.99], "critic_betas": [0.5, 0.99],
            "critic_weight_decay": 0.0, "gradient_clip": 0.5, "critic_gradient_clip": 2.0,
            "ema_enabled": true, "ema_decay": 0.99, "crop": "center",
            "degradation": {"use_hflip": false}, "critic_dtype": "fp32", "init": "parent",
            "num_epochs": 3, "nan_loss_tolerance": 5, "keep_last_checkpoints": 1,
            "milestone_steps": [2], "export_weights": "raw", "export_dtype": "bf16",
            "preview_weights": "ema", "lora_init": "kaiming"
        });
        let plan = task_plan(
            &data,
            "iris_3b_restore",
            task_dataset(&data, "restoration", 2, true),
            |c| {
                c.rank = 8;
                c.alpha = 4;
                c.gradient_accumulation = 2;
                c.steps = 40;
                c.save_every = 8;
                c.resolution = 512;
                c.advanced.insert("networkType".into(), json!("lokr"));
                c.advanced.insert("decomposeFactor".into(), json!(4));
                c.advanced.insert("sampleEvery".into(), json!(8));
                c.advanced.insert("iris_restore".into(), bag.clone());
            },
        );
        let run = run_as("iris_3b_restore", Behavior::Completes, plan, &data);
        *crate::iris_task_training::TEST_TRAINING_RESOURCES
            .lock()
            .unwrap() = None;
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(run.seen.weights.as_deref(), Some(export.as_path()));
        let mut components = run.seen.components.clone();
        components.sort();
        let resource_path = |resource: &gen_core::iris::restoration_train::TrainingResource| {
            crate::model_jobs::huggingface_pinned_snapshot_dir(
                &data,
                resource.repo,
                resource.revision,
            )
            .unwrap()
            .join(resource.file)
        };
        assert_eq!(
            components,
            vec![
                ("critic_trunk".to_owned(), resource_path(&resources[2])),
                ("lpips_lin".to_owned(), resource_path(&resources[1])),
                ("lpips_vgg16".to_owned(), resource_path(&resources[0])),
                ("parent_backbone".to_owned(), install.backbone.clone()),
            ]
        );
        let request = run.seen.request.expect("the trainer ran");
        let config = &request.config;
        assert!(!config.full_finetune);
        assert_eq!(config.network_type, gen_core::NetworkType::Lokr);
        assert_eq!(
            (config.rank, config.alpha, config.decompose_factor),
            (8, 4.0, 4)
        );
        assert_eq!(
            (config.steps, config.save_every, config.sample_every),
            (40, 8, 8)
        );
        assert_eq!(config.resolution, 512);
        assert_eq!(config.loss_type, "mse");
        assert!(config.sample_prompts.is_empty());
        assert_eq!(
            Value::Object(config.model_options.clone()),
            json!({ "iris_restore": bag })
        );
        let root = data.join("projects/p1/training/datasets/ds_task");
        assert_eq!(request.items[1].caption, "");
        assert_eq!(
            Value::Object(request.items[1].model_options.clone()),
            json!({"degraded_image": root.join("sidecars").join("item1.degraded.png")})
        );
    }

    /// A staged training resource that does not match its pin (here: the real pins against stand-in
    /// bytes) is refused BEFORE load, naming the Model Manager install to repair.
    #[test]
    fn a_restoration_resource_failing_its_pin_is_refused_before_load() {
        let _table = RESOURCE_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        stage_task_export(&data, "upscaler");
        for resource in gen_core::iris::restoration_train::TRAINING_RESOURCES {
            stage(&data, resource.repo, resource.revision, resource.file);
        }
        let plan = task_plan(
            &data,
            "iris_3b_restore",
            task_dataset(&data, "restoration", 1, false),
            |c| {
                c.resolution = 256;
                c.advanced.insert("networkType".into(), json!("lora"));
                c.steps = 4;
                c.save_every = 2;
                c.advanced.insert("sampleEvery".into(), json!(2));
            },
        );
        let run = run_as("iris_3b_restore", Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => assert!(
                message.contains("re-install") && message.contains("iris_restore_train_vgg16"),
                "{message}"
            ),
            other => panic!("a resource failing its pin must be refused: {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);

        // A missing resource is named before anything is hashed.
        let missing = sceneworks_core::hf_home::huggingface_repo_cache_path(
            &data,
            gen_core::iris::restoration_train::CRITIC_TRUNK_RESOURCE.repo,
        )
        .unwrap();
        std::fs::remove_dir_all(missing).unwrap();
        let refusal = crate::iris_task_training::iris_task_training_unavailable(
            &data,
            sceneworks_core::training::iris::IrisObjective::Restoration,
            &builtin_training_targets()
                .targets
                .into_iter()
                .find(|t| t.base_model == "iris_3b_restore")
                .unwrap()
                .defaults,
        )
        .expect("the trunk is missing");
        assert!(
            refusal.contains("iris_restore_train_critic_trunk"),
            "{refusal}"
        );
    }
}
