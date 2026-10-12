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

/// The accelerator a rank occupies on `lane` (refusal text).
pub(crate) fn device_noun(lane: &str) -> &'static str {
    if lane == "mlx" {
        "Mac GPU"
    } else {
        "GPU"
    }
}

/// The admission decision for a whole Iris run (sc-25689) on a worker with `devices` accelerators:
/// the declared floor of ONE rank (+ the REPA teacher) times the ranks sharing the busiest device,
/// against that device's capacity. Missing evidence refuses, exactly like [`iris_memory_refusal`].
pub(crate) fn iris_run_memory_refusal(
    lane: &str,
    config: &sceneworks_core::training::TrainingConfig,
    devices: u64,
    capacity_gb: Option<f64>,
) -> Option<String> {
    use sceneworks_core::training::iris::curricula;
    let world = curricula::iris_world_size(config);
    let ranks = curricula::ranks_per_device(world, devices);
    let network_type = config
        .advanced
        .get("networkType")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("lora");
    let floor = curricula::iris_run_memory_floor_gb(lane, config, devices);
    let repa = sceneworks_core::training::iris::iris_options(config)
        .map(|options| curricula::iris_repa_enabled(&options))
        .unwrap_or(false);
    let mut shape = run_shape(network_type, config.resolution);
    if repa {
        shape.push_str(" with REPA (+ the DINOv2 teacher)");
    }
    if ranks > 1 {
        shape.push_str(&format!(
            " × {ranks} ranks sharing one {}",
            device_noun(lane)
        ));
    }
    memory_refusal_for_floor(lane, floor, &shape, capacity_gb)
}

fn run_shape(network_type: &str, resolution: u32) -> String {
    let artifact = if network_type.trim().eq_ignore_ascii_case("full") {
        "a full fine-tune"
    } else {
        "an adapter (LoRA/LoKr) run"
    };
    format!("{artifact} at {resolution}px")
}

/// The submit-time twin of the worker's run gate for the co-located macOS API (sc-25689): the
/// declared floor of the whole run (REPA, ranks sharing the one Mac GPU). `None` off macOS.
pub fn iris_training_run_memory_error(
    config: &sceneworks_core::training::TrainingConfig,
) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        iris_run_memory_refusal("mlx", config, 1, mlx_training_capacity_gb())
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = config;
        None
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
    let floor = sceneworks_core::training::iris::iris_training_memory_floor_gb(
        lane,
        network_type,
        resolution,
    );
    memory_refusal_for_floor(
        lane,
        floor,
        &run_shape(network_type, resolution),
        capacity_gb,
    )
}

/// [`iris_memory_refusal`] for an already-resolved declared `floor` of a run described by `shape`.
fn memory_refusal_for_floor(
    lane: &str,
    floor: Option<u32>,
    shape: &str,
    capacity_gb: Option<f64>,
) -> Option<String> {
    let noun = capacity_noun(lane);
    let Some(floor) = floor else {
        return Some(format!(
            "Iris 3B training declares no memory floor for {shape} on the {lane} lane, so it is \
             refused rather than admitted without a bound."
        ));
    };
    let Some(capacity) = capacity_gb else {
        return Some(format!(
            "Iris 3B training needs at least {floor} GB of {noun} for {shape} (a declared floor, \
             not yet measured), but this machine's {noun} could not be read, so the run is \
             refused rather than admitted without a bound."
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
            "Iris 3B training needs at least {floor} GB of {noun} for {shape} (a declared floor, \
             not yet measured), but this machine has ~{} GB{reserve_note}. Lower the training \
             resolution, train a LoRA/LoKr adapter instead of a full fine-tune, run fewer ranks \
             on one device, or run on a machine with more memory.",
            capacity.round() as i64
        )
    })
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
        teacher: Option<PathBuf>,
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
        // The worker resolves the engine plan against the backbone's real architecture (sc-25689).
        std::fs::write(
            backbone.join("config.yaml"),
            crate::iris_test_weights::RELEASE_CONFIG_YAML,
        )
        .unwrap();
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
        plan_with_dataset(data, backbone, items, |_| {}, configure)
    }

    /// [`plan`] with the dataset adjusted before the plan is built (e.g. holdout items).
    fn plan_with_dataset(
        data: &Path,
        backbone: &Path,
        items: usize,
        adjust: impl FnOnce(&mut TrainingDataset),
        configure: impl FnOnce(&mut sceneworks_core::training::TrainingConfig),
    ) -> TrainingPlan {
        let target = builtin_training_targets()
            .targets
            .into_iter()
            .find(|target| target.kernel == "iris_train")
            .expect("the Iris target is registered");
        let mut config = target.defaults.clone();
        configure(&mut config);
        let (mut dataset, root) = dataset(data, items);
        adjust(&mut dataset);
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
            let descriptor = crate::inference_runtime::trainer_descriptor("iris_3b")
                .expect("this backend links the Iris trainer");
            let loader_seen = Arc::clone(&seen);
            let user_cancel = Arc::clone(&user_cancel);
            let loader = move |engine: &str, spec: &LoadSpec| {
                assert_eq!(engine, "iris_3b");
                let mut seen = loader_seen.lock().unwrap();
                seen.loads += 1;
                if let WeightsSource::Dir(dir) = &spec.weights {
                    seen.weights = Some(dir.clone());
                }
                if let Some(WeightsSource::Dir(dir)) = spec.components.get("text_encoder") {
                    seen.text_encoder = Some(dir.clone());
                }
                if let Some(WeightsSource::Dir(dir)) = spec.components.get("repa_teacher") {
                    seen.teacher = Some(dir.clone());
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

    // ---- S11 (sc-25689): curricula, REPA, holdout, fp16 and multi-rank execution -------------

    const TEACHER_BYTES: &[u8] = b"a stand-in dinov2-base model.safetensors";

    fn stage_teacher(data: &Path) -> PathBuf {
        let snapshot = stage(
            data,
            sceneworks_core::training::iris::curricula::IRIS_REPA_TEACHER_REPO,
            sceneworks_core::training::iris::curricula::IRIS_REPA_TEACHER_REVISION,
            "model.safetensors",
        );
        std::fs::write(snapshot.join("model.safetensors"), TEACHER_BYTES).unwrap();
        snapshot
    }

    /// Every S11 option the Studio offers reaches the trainer: the curriculum, shape, RoPE, REPA,
    /// activation-checkpointing, validation, logging, retention, global-batch and preview keys in
    /// the bag verbatim, fp16 as the engine dtype, the holdout flag on its item, and the REPA
    /// teacher staged as the `repa_teacher` component after its integrity check.
    #[test]
    fn every_offered_s11_option_reaches_the_trainer_with_the_teacher_staged() {
        let _env = isolated_env("64");
        let _pin = crate::iris_curricula::test_teacher_integrity(Some(TEACHER_BYTES));
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let teacher = stage_teacher(&data);
        let iris = json!({
            "stage": "stage1_256",
            "run_name": "iris s1",
            "tracking": {"project": "iris", "tags": ["s1"]},
            "shape_policy": "area",
            "shape_align": 32,
            "shape_max_ratio": 2.0,
            "rope_aspect": "square",
            "repa": {"weight": 0.5, "variant": "irepa", "spatial_norm_gamma": 0.8,
                     "proj_hidden_dim": 512, "teacher_image_size": 224,
                     "teacher_match_student": true, "layer": 10},
            "activation_checkpointing": "selective_layer",
            "ac_selective_every": 3,
            "val_every_steps": 5,
            "val_samples": 2,
            "log_every": 5,
            "log_block_grad_norms": true,
            "save_every_epochs": 1,
            "expected_global_batch": 4,
            "scale_warmup_by_world": false,
            "perf": {"lazy_sync": true},
            "dist": {"strategy": "ddp"},
            "preview_at_step_one": true,
            "preview_order": 1,
            "upstream_validation_prompts": ["a lighthouse at dusk"]
        });
        let plan = plan_with_dataset(
            &data,
            &install.backbone,
            3,
            |dataset| {
                dataset.items[2]
                    .extra
                    .insert("split".into(), json!("holdout"));
            },
            |config| {
                config.batch_size = 2;
                config.gradient_accumulation = 2;
                config.steps = 40;
                config.save_every = 10;
                config
                    .advanced
                    .insert("mixedPrecision".into(), json!("fp16"));
                config.advanced.insert("sampleEvery".into(), json!(10));
                config.advanced.insert("samplePrompts".into(), json!([]));
                config.advanced.insert("iris".into(), iris.clone());
            },
        );
        let run = run(Behavior::Completes, plan, &data);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(
            run.seen.teacher.as_deref(),
            Some(teacher.as_path()),
            "the verified teacher is the trainer's repa_teacher component"
        );
        let request = run.seen.request.expect("the trainer ran");
        assert_eq!(request.config.train_dtype, "fp16");
        assert_eq!(request.config.model_options.get("iris"), Some(&iris));
        assert_eq!(
            request.items[2].model_options.get("split"),
            Some(&json!("holdout"))
        );
        assert!(request.items[0].model_options.get("split").is_none());
        // The engine-resolved receipts ride on the result.
        let result = &terminal(&run.posts)["result"];
        let resolved = &result["irisPlan"];
        assert_eq!(resolved["stage"], json!("stage1_256"));
        assert_eq!(resolved["globalBatch"], json!(4));
        assert_eq!(
            resolved["activationCheckpointing"],
            json!("selective_layer")
        );
        assert_eq!(resolved["repa"]["variant"], json!("irepa"));
        assert_eq!(resolved["holdout"]["items"], json!(1));
        assert_eq!(resolved["mixedPrecision"], json!("fp16"));
        assert!(resolved["shapePolicy"]
            .as_str()
            .unwrap()
            .starts_with("area:"));
        assert!(resolved["dataIdentity"]
            .as_str()
            .unwrap()
            .contains("shape=area"));
    }

    /// REPA without the installed teacher is refused before the loader runs, naming the Model
    /// Manager install; a teacher whose bytes are not the pinned ones is refused the same way.
    #[test]
    fn repa_needs_the_installed_verified_teacher() {
        let _env = isolated_env("64");
        let _pin = crate::iris_curricula::test_teacher_integrity(Some(TEACHER_BYTES));
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let repa = |config: &mut sceneworks_core::training::TrainingConfig| {
            config
                .advanced
                .insert("iris".into(), json!({"repa": {"weight": 0.5}}));
        };
        let run_missing = run(
            Behavior::Completes,
            plan(&data, &install.backbone, 2, repa),
            &data,
        );
        match &run_missing.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("Model Manager"), "{message}")
            }
            other => panic!("REPA without a teacher must be refused, got {other:?}"),
        }
        assert_eq!(run_missing.seen.loads, 0);
        let teacher = stage_teacher(&data);
        let mut tampered = TEACHER_BYTES.to_vec();
        tampered[0] ^= 1;
        std::fs::write(teacher.join("model.safetensors"), &tampered).unwrap();
        let run_tampered = run(
            Behavior::Completes,
            plan(&data, &install.backbone, 2, repa),
            &data,
        );
        match &run_tampered.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("sha256"), "{message}")
            }
            other => panic!("a tampered teacher must be refused, got {other:?}"),
        }
        assert_eq!(run_tampered.seen.loads, 0);
    }

    /// The engine's own plan resolution refuses what only it knows (a REPA capture block outside
    /// the 24-block trunk) before anything loads.
    #[test]
    fn the_engine_plan_refusals_arrive_before_load() {
        let _env = isolated_env("64");
        let _pin = crate::iris_curricula::test_teacher_integrity(Some(TEACHER_BYTES));
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        stage_teacher(&data);
        let plan = plan(&data, &install.backbone, 2, |config| {
            config
                .advanced
                .insert("iris".into(), json!({"repa": {"weight": 0.5, "layer": 99}}));
        });
        let run = run(Behavior::Completes, plan, &data);
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("capture block"), "{message}")
            }
            other => panic!("an out-of-trunk REPA layer must be refused, got {other:?}"),
        }
        assert_eq!(run.seen.loads, 0);
    }

    fn fleet_records_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    /// Drive the REAL job path with fake rank processes. The in-process loader must never run.
    fn run_fleet(
        behavior: &'static str,
        plan: TrainingPlan,
        data: &Path,
        records: &Path,
        cancel_on_training: bool,
    ) -> Run {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        let (outcome, posts) = runtime.block_on(async {
            let user_cancel = CancelSwitch::default();
            let (base_url, posts) = spawn_job_api(Arc::clone(&user_cancel)).await;
            if cancel_on_training {
                let posts = Arc::clone(&posts);
                let user_cancel = Arc::clone(&user_cancel);
                tokio::spawn(async move {
                    loop {
                        let training = posts.lock().unwrap().iter().any(|post: &Value| {
                            post["message"]
                                .as_str()
                                .is_some_and(|message| message.starts_with("Training step"))
                        });
                        if training {
                            user_cancel.store(true, std::sync::atomic::Ordering::SeqCst);
                            return;
                        }
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    }
                });
            }
            let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
            settings.api_url = base_url;
            settings.data_dir = data.to_path_buf();
            settings.heartbeat_seconds = 5;
            let api = ApiClient::new(&settings);
            let job: JobSnapshot =
                serde_json::from_value(crate::tests::job_snapshot_json("job-iris-train", false))
                    .unwrap();
            let loader = |_: &str, _: &LoadSpec| -> gen_core::Result<Box<dyn Trainer>> {
                panic!("a multi-rank run never loads a trainer in the worker process")
            };
            let outcome = tokio::time::timeout(
                Duration::from_secs(120),
                crate::training_jobs::run_training_execution_drive(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    loader,
                    Arc::new(crate::iris_dist::test_support::FakeRankLauncher {
                        behavior,
                        record_dir: records.to_path_buf(),
                    }),
                ),
            )
            .await
            .expect("the training job finished");
            let posts = posts.lock().unwrap().clone();
            (outcome, posts)
        });
        for record in crate::iris_dist::test_support::records(records) {
            assert!(
                !crate::iris_dist::test_support::alive(record["pid"].as_u64().unwrap()),
                "rank {} outlived the job",
                record["rank"]
            );
        }
        Run {
            outcome,
            posts,
            seen: Seen::default(),
        }
    }

    fn two_ranks(
        share_device: bool,
    ) -> impl FnOnce(&mut sceneworks_core::training::TrainingConfig) {
        move |config| {
            config.advanced.insert(
                "iris".into(),
                json!({"dist": {"strategy": "ddp",
                                "world": {"world_size": 2, "share_device": share_device}}}),
            );
        }
    }

    /// A two-rank run: the worker coordinates two rank processes, rank 0's progress drives the job
    /// and its export completes it, with the topology recorded on the result.
    #[test]
    fn a_two_rank_run_is_coordinated_and_rank_zero_completes_the_job() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let plan = plan(&data, &install.backbone, 2, two_ranks(true));
        let run = run_fleet("complete", plan, &data, records.path(), false);
        assert!(run.outcome.is_ok(), "{:?}\n{:#?}", run.outcome, run.posts);
        assert_eq!(
            crate::iris_dist::test_support::records(records.path()).len(),
            2
        );
        assert!(run.posts.iter().any(|post| post["message"]
            .as_str()
            .is_some_and(|message| message.contains("Training step 4 of 4"))));
        assert!(
            !run.posts.iter().any(|post| post["message"]
                .as_str()
                .is_some_and(|message| message.contains("step 99"))),
            "rank 1's progress is not the job's"
        );
        let done = terminal(&run.posts);
        assert_eq!(done["status"], "completed", "{done:#}");
        let dist = &done["result"]["irisDistributed"];
        assert_eq!(dist["worldSize"], json!(2));
        assert_eq!(dist["shareDevice"], json!(true));
        assert_eq!(dist["topology"], json!("single_host"));
        assert_eq!(done["result"]["irisPlan"]["globalBatch"], json!(2));
        assert_eq!(done["result"]["irisPlan"]["dist"]["world_size"], json!(2));
    }

    /// More ranks than the worker's devices without the share-device opt-in is a typed refusal
    /// before any rank process starts; two ranks on one device need twice the declared floor.
    #[test]
    fn unsupported_topologies_are_refused_before_any_rank_starts() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let run = run_fleet(
            "complete",
            plan(&data, &install.backbone, 2, two_ranks(false)),
            &data,
            records.path(),
            false,
        );
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("share device"), "{message}")
            }
            other => panic!("2 ranks on 1 device without share_device: {other:?}"),
        }
        assert!(crate::iris_dist::test_support::records(records.path()).is_empty());
        drop(_env);
        // One adapter rank fits a 32 GB host; two on its one device do not.
        let _small = isolated_env("32");
        let run = run_fleet(
            "complete",
            plan(&data, &install.backbone, 2, two_ranks(true)),
            &data,
            records.path(),
            false,
        );
        match &run.outcome {
            Err(WorkerError::InvalidPayload(message)) => {
                assert!(message.contains("2 ranks"), "{message}")
            }
            other => panic!("two ranks need twice the floor: {other:?}"),
        }
        assert!(crate::iris_dist::test_support::records(records.path()).is_empty());
    }

    /// A rank that dies fails the job with a typed message naming it; the survivor is stopped.
    #[test]
    fn a_lost_rank_fails_the_job_typed() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let run = run_fleet(
            "rank1_dies",
            plan(&data, &install.backbone, 2, two_ranks(true)),
            &data,
            records.path(),
            false,
        );
        match &run.outcome {
            Err(WorkerError::Engine(message)) => {
                assert!(message.contains("[rank_lost] on rank 1 of 2"), "{message}")
            }
            other => panic!("a lost rank must fail the job: {other:?}"),
        }
    }

    /// A user cancel reaches every rank; rank 0 finishes durably, so the job ends Canceled with the
    /// export reported for registration.
    #[test]
    fn a_user_cancel_fans_out_to_every_rank() {
        let _env = isolated_env("128");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let records = fleet_records_dir();
        let run = run_fleet(
            "wait_cancel",
            plan(&data, &install.backbone, 2, two_ranks(true)),
            &data,
            records.path(),
            true,
        );
        assert!(
            matches!(run.outcome, Err(WorkerError::Canceled(_))),
            "{:?}",
            run.outcome
        );
        for rank in 0..2 {
            assert!(
                records.path().join(format!("cancel-{rank}")).is_file(),
                "rank {rank} saw the cancel"
            );
        }
        let last = terminal(&run.posts);
        assert_eq!(last["status"], "canceled", "{last:#}");
        assert_eq!(last["result"]["trainedArtifactExported"], json!(true));
    }

    /// The rank process itself: the SAME preflight and trainer inputs as the job, its world stamped
    /// onto the request, progress and the export written as protocol lines, and a typed distributed
    /// failure reported with its kind.
    #[test]
    fn the_rank_process_trains_with_its_world_stamped() {
        let _env = isolated_env("64");
        let data_holder = tempfile::tempdir().unwrap();
        let data = data_holder.path().canonicalize().unwrap();
        let install = stage_install(&data);
        let plan = plan(&data, &install.backbone, 2, two_ranks(true));
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.data_dir = data.clone();
        let spec = crate::iris_dist::RankSpec {
            job_id: "job-iris-train".to_owned(),
            plan,
        };
        let world = crate::iris_dist::RankWorld {
            rank: 1,
            world_size: 2,
            master_addr: "127.0.0.1".to_owned(),
            master_port: 41000,
        };
        let seen = Arc::new(Mutex::new(Seen::default()));
        let descriptor = crate::inference_runtime::trainer_descriptor("iris_3b").unwrap();
        let loader_seen = Arc::clone(&seen);
        let loader = move |_: &str, spec: &LoadSpec| {
            if let Some(WeightsSource::Dir(dir)) = spec.components.get("text_encoder") {
                loader_seen.lock().unwrap().text_encoder = Some(dir.clone());
            }
            Ok(Box::new(FakeIrisTrainer {
                descriptor,
                behavior: Behavior::Completes,
                seen: Arc::clone(&loader_seen),
                user_cancel: Arc::default(),
            }) as Box<dyn Trainer>)
        };
        let mut out = Vec::new();
        let code = crate::iris_dist::run_rank_process_using(
            &spec,
            &world,
            &settings,
            loader,
            std::io::empty(),
            &mut out,
        );
        assert_eq!(code, 0, "{}", String::from_utf8_lossy(&out));
        let seen = seen.lock().unwrap();
        let request = seen.request.as_ref().expect("the rank trained");
        let stamped = &request.config.model_options["iris"]["dist"]["world"];
        assert_eq!(stamped["rank"], json!(1));
        assert_eq!(stamped["master_port"], json!(41000));
        assert_eq!(stamped["share_device"], json!(true));
        assert_eq!(
            seen.text_encoder.as_deref(),
            Some(install.text_encoder.as_path())
        );
        let lines: Vec<_> = String::from_utf8(out)
            .unwrap()
            .lines()
            .filter_map(crate::iris_dist::parse_message)
            .collect();
        assert!(matches!(
            lines.last(),
            Some(crate::iris_dist::RankMessage::Done { steps: 1000, .. })
        ));
        assert!(lines.iter().any(|line| matches!(
            line,
            crate::iris_dist::RankMessage::Progress {
                progress: crate::iris_dist::WireProgress::Training { .. }
            }
        )));

        // A launcher world that disagrees with the plan is a split world, reported typed.
        let wrong = crate::iris_dist::RankWorld {
            world_size: 3,
            ..world
        };
        let mut out = Vec::new();
        let code = crate::iris_dist::run_rank_process_using(
            &spec,
            &wrong,
            &settings,
            |_: &str, _: &LoadSpec| -> gen_core::Result<Box<dyn Trainer>> {
                panic!("a split world never loads")
            },
            std::io::empty(),
            &mut out,
        );
        assert_eq!(code, 1);
        let text = String::from_utf8(out).unwrap();
        assert!(text.contains("split world"), "{text}");
    }
}
