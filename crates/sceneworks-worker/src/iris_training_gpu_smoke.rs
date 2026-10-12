//! Real-weight acceptance for Iris-3B generation TRAINING through the production job path (epic
//! sc-25678; S7 sc-25685 on MLX, S8 sc-25686 on Candle/CUDA). Backend-neutral: the same test runs
//! the MLX trainer on Apple Silicon and the Candle trainer on NVIDIA.
//!
//! 1. One bounded LoRA `lora_train` run through `training_jobs::run_training_execution` — the plan
//!    built by the REAL `sceneworks_core` Iris target, the Iris memory floor + data walk, cache-only
//!    resolution of the backbone and its `text_encoder` co-requisite, the trainer, a durable
//!    checkpoint, a preview, and the export. The adapter must carry the Iris identity stamps
//!    (`family = iris`, `baseModel = iris_3b`, `irisTask = generation`, raw weights).
//! 2. Two `image_generate` jobs through `run_image_generate_job` at the same seed — without and with
//!    the trained adapter (as the registered LoRA entry hands it over) — and the adapter must change
//!    the render: the trained artifact is loadable and applied by the generation engine.
//!
//! sc-25690 adds, on the same path: a LoKr run and the official `stage1_256` FULL fine-tune each
//! canceled after a durable checkpoint and RESUMED (the resumed job's own progress must show it
//! continued from the stop, [`resume_continuation`]), and the restoration paper's FULL fine-tune —
//! so the admission gate's full-fine-tune floor rows are read by real jobs, not only by the
//! trainers' own tests.
//!
//! Nothing downloads: `HF_HUB_CACHE` must already hold both pinned snapshots.
//!
//! ```text
//! HF_HUB_CACHE=~/.cache/huggingface/hub \
//!   cargo test -p sceneworks-worker --release iris_training_gpu_smoke -- --ignored --nocapture --test-threads=1
//! # Windows/CUDA: add --features backend-candle
//! ```

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::*;

const TRAIN_JOB: &str = "iris-train-smoke";
const BACKBONE_REPO: &str = "speridlabs/iris-3b";
const TEXT_ENCODER_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
const RESOLUTION: u32 = 256;
const TRAIN_STEPS: u32 = 8;
const RENDER_SIZE: u32 = 256;
const RENDER_STEPS: u32 = 20;

type ProgressPosts = Arc<Mutex<Vec<(String, Value)>>>;

fn stub_job_snapshot(job_id: &str, job_type: &str, project_id: &str) -> Value {
    json!({
        "id": job_id,
        "type": job_type,
        "status": "running",
        "projectId": project_id,
        "projectName": "Iris training smoke",
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": "0",
        "workerId": "iris-train-smoke-worker",
        "progress": 0.1,
        "stage": "running",
        "message": "running",
        "error": null,
        "etaSeconds": null,
        "elapsedSeconds": null,
        "attempts": 1,
        "sourceJobId": null,
        "duplicateOfJobId": null,
        "cancelRequested": false,
        "createdAt": "2026-10-11T00:00:00Z",
        "updatedAt": "2026-10-11T00:00:00Z",
        "startedAt": "2026-10-11T00:00:00Z",
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": "2026-10-11T00:00:00Z"
    })
}

async fn stub_job(AxumPath(job_id): AxumPath<String>) -> Json<Value> {
    Json(stub_job_snapshot(&job_id, "lora_train", "smoke"))
}

async fn stub_progress(
    State(posts): State<ProgressPosts>,
    AxumPath(job_id): AxumPath<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    posts
        .lock()
        .expect("progress posts lock")
        .push((job_id.clone(), payload));
    Json(stub_job_snapshot(&job_id, "lora_train", "smoke"))
}

async fn stub_heartbeat(AxumPath(worker_id): AxumPath<String>) -> Json<Value> {
    Json(json!({
        "id": worker_id, "gpuId": "0", "gpuName": null, "status": "busy",
        "currentJobId": TRAIN_JOB, "capabilities": [], "loadedModels": [],
        "registeredAt": "2026-10-11T00:00:00Z", "lastSeenAt": "2026-10-11T00:00:00Z"
    }))
}

async fn stub_metrics(Json(payload): Json<Value>) -> Json<Value> {
    Json(payload)
}

async fn spawn_worker_api_stub() -> (String, ProgressPosts) {
    let posts: ProgressPosts = Arc::default();
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(stub_job))
        .route("/api/v1/jobs/:job_id/progress", post(stub_progress))
        .route("/api/v1/jobs/:job_id/metrics", post(stub_metrics))
        .route("/api/v1/workers/:worker_id/heartbeat", post(stub_heartbeat))
        .with_state(posts.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("worker API stub listener binds");
    let address = listener.local_addr().expect("worker API stub has address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("worker API stub serves");
    });
    (format!("http://{address}"), posts)
}

fn shipped_iris_entry() -> Value {
    let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
        .iter()
        .find(|(name, _)| *name == "builtin.models.jsonc")
        .expect("builtin.models.jsonc embedded")
        .1;
    let manifest: Value =
        serde_json::from_str(&strip_jsonc_comments(text)).expect("parse builtin.models.jsonc");
    manifest["models"]
        .as_array()
        .and_then(|models| models.iter().find(|model| model["id"] == "iris_3b"))
        .cloned()
        .expect("iris_3b is in the builtin catalog")
}

/// A small, learnable dataset: four flat-colour squares with their colour as the caption.
fn write_dataset(root: &Path) -> sceneworks_core::training::TrainingDataset {
    std::fs::create_dir_all(root.join("images")).expect("dataset dir");
    let colours = [
        ("red", [220u8, 30, 30]),
        ("green", [30, 200, 60]),
        ("blue", [40, 60, 220]),
        ("yellow", [230, 210, 40]),
    ];
    let items: Vec<Value> = colours
        .iter()
        .enumerate()
        .map(|(index, (name, rgb))| {
            let path = format!("images/{name}.png");
            let image = image::RgbImage::from_fn(RESOLUTION, RESOLUTION, |x, y| {
                // A soft vignette so the target is not a single constant.
                let edge = ((x.min(RESOLUTION - x)).min(y.min(RESOLUTION - y)) as f32
                    / (RESOLUTION as f32 / 2.0))
                    .min(1.0);
                image::Rgb(rgb.map(|channel| (channel as f32 * (0.6 + 0.4 * edge)) as u8))
            });
            image.save(root.join(&path)).expect("write dataset image");
            json!({
                "id": format!("item{index}"),
                "path": path,
                "displayName": format!("{name}.png"),
                "caption": { "text": format!("a flat {name} square, smoke style"), "source": "manual", "triggerWords": [] },
                "width": RESOLUTION,
                "height": RESOLUTION,
                "addedAt": "2026-10-11T00:00:00Z"
            })
        })
        .collect();
    serde_json::from_value(json!({
        "schemaVersion": 1, "id": "ds_smoke", "version": 1, "projectId": "smoke",
        "name": "Iris smoke", "modality": "image", "status": "ready",
        "createdAt": "2026-10-11T00:00:00Z", "updatedAt": "2026-10-11T00:00:00Z",
        "items": items
    }))
    .expect("smoke dataset")
}

fn safetensors_metadata(path: &Path) -> serde_json::Map<String, Value> {
    let bytes = std::fs::read(path).expect("read adapter");
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + len]).expect("adapter header");
    header["__metadata__"]
        .as_object()
        .cloned()
        .unwrap_or_default()
}

async fn render(
    settings: &Settings,
    posts: &ProgressPosts,
    project: &sceneworks_core::contracts::JsonObject,
    job_id: &str,
    loras: Value,
) -> gen_core::Image {
    render_model(settings, posts, project, job_id, loras, None).await
}

/// [`render`], optionally with a TRAINED full checkpoint `(model id, export dir)` exactly as the
/// API hands one to the worker (`apps/rust-api` `alias_trained_checkpoint`): the payload names the
/// `iris_3b` engine, the engine's manifest entry carries `modelPath` = the export and the
/// `trainedCheckpoint` identity, and `checkpointModelId` records what was picked.
async fn render_model(
    settings: &Settings,
    posts: &ProgressPosts,
    project: &sceneworks_core::contracts::JsonObject,
    job_id: &str,
    loras: Value,
    checkpoint: Option<(&str, &Path)>,
) -> gen_core::Image {
    let mut entry = shipped_iris_entry();
    if let Some((model_id, export)) = checkpoint {
        entry["modelPath"] = json!(export.display().to_string());
        entry["trainedCheckpoint"] = json!({
            "engineModel": "iris_3b", "family": "iris", "task": "generation",
            "baseModel": "iris_3b", "layout": "iris_backbone_directory", "modelId": model_id,
        });
    }
    let mut snapshot = stub_job_snapshot(job_id, "image_generate", "smoke");
    snapshot["projectId"] = project["id"].clone();
    snapshot["payload"] = json!({
        "projectId": project["id"],
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": "a flat red square, smoke style",
        "width": RENDER_SIZE,
        "height": RENDER_SIZE,
        "count": 1,
        "seed": 11,
        "loras": loras,
        "advanced": { "steps": RENDER_STEPS, "guidanceScale": 3.0 },
        "modelManifestEntry": entry,
    });
    if let Some((model_id, _)) = checkpoint {
        snapshot["payload"]["checkpointModelId"] = json!(model_id);
    }
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("render job deserializes");
    let api = ApiClient::new(settings);
    crate::image_jobs::run_image_generate_job(&api, settings, &job)
        .await
        .unwrap_or_else(|error| panic!("the Iris render {job_id} completes: {error}"));
    let result = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == job_id && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the render posted a completed result");
    let media = result["assetWrites"][0]["mediaPath"]
        .as_str()
        .expect("the render wrote one asset")
        .to_owned();
    let decoded = image::open(Path::new(project["path"].as_str().unwrap()).join(media))
        .expect("open render")
        .to_rgb8();
    gen_core::Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris training smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and an Apple-Silicon GPU or a CUDA device"]
async fn iris_training_gpu_smoke() {
    run_iris_training_smoke(1).await;
}

/// sc-25689: the same bounded LoRA run as TWO ddp ranks sharing the worker's one device
/// (`dist.world.share_device`) — two real rank processes of this test binary re-entered through the
/// production [`crate::iris_dist::iris_rank_entry`], rendezvousing over loopback TCP with the per-job
/// token, rank 0's events driving the job — then the adapter is applied by a generation job exactly
/// as the single-rank smoke's is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight two-rank Iris training smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and an Apple-Silicon GPU or a CUDA device"]
async fn iris_training_gpu_smoke_two_rank_ddp() {
    run_iris_training_smoke(2).await;
}

async fn run_iris_training_smoke(ranks: u64) {
    let hub = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(
        hub.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hub.display()
    );

    let temp = tempfile::tempdir().expect("smoke tempdir");
    let data_dir = temp.path().join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let data_dir = data_dir.canonicalize().unwrap();
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(data_dir.clone());
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-train-smoke-worker".to_owned();
    settings.heartbeat_seconds = 30;
    if cfg!(target_os = "macos") {
        settings.gpu_id = "mlx".to_owned();
        settings.backend_mlx_enabled = true;
    } else {
        // The worker's gpu id is the nvidia-smi index the Iris memory floor is read from: the
        // (physical) card CUDA_VISIBLE_DEVICES selects, as the production candle worker is launched.
        settings.gpu_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|devices| devices.split(',').next().map(|id| id.trim().to_owned()))
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        settings.backend_candle_enabled = true;
    }
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE must already hold the Iris backbone (no download at training time)");
    assert!(
        crate::model_jobs::huggingface_snapshot_dir(&data_dir, TEXT_ENCODER_REPO).is_some(),
        "HF_HUB_CACHE must already hold the Qwen3-VL text encoder"
    );

    let project_store = ProjectStore::new(data_dir.clone(), "iris-train-smoke");
    let project = project_store
        .create_project("Iris training smoke")
        .expect("smoke project");
    let project_value = serde_json::to_value(&project).unwrap();
    let project_object = project_value.as_object().unwrap().clone();
    let dataset_root = PathBuf::from(&project.path).join("training/datasets/ds_smoke");
    let dataset = write_dataset(&dataset_root);
    let output_dir = PathBuf::from(&project.path).join("loras/lora_iris_smoke");

    let target = sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.kernel == "iris_train")
        .expect("the Iris target is registered");
    let mut config = target.defaults.clone();
    config.rank = 4;
    config.alpha = 4;
    config.steps = TRAIN_STEPS;
    config.save_every = TRAIN_STEPS / 2;
    config.resolution = RESOLUTION;
    config.learning_rate =
        sceneworks_core::contracts::ContractNumber::from_f64(1e-3).expect("finite");
    config.trigger_word = Some("smoke style".to_owned());
    config
        .advanced
        .insert("sampleEvery".into(), json!(TRAIN_STEPS));
    config.advanced.insert("sampleSteps".into(), json!(10));
    config.advanced.insert("sampleCount".into(), json!(1));
    config.advanced.insert(
        "samplePrompts".into(),
        json!(["a flat red square, smoke style"]),
    );
    if ranks > 1 {
        config.advanced.insert(
            "iris".into(),
            json!({
                "dist": {"strategy": "ddp", "world": {"world_size": ranks, "share_device": true}},
                "expected_global_batch": ranks,
            }),
        );
    }
    let plan = sceneworks_core::training::build_training_plan(
        sceneworks_core::training::BuildTrainingPlan {
            job_id: TRAIN_JOB,
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: "lora_iris_smoke",
            base_model_path: backbone.display().to_string(),
            dataset_root: &dataset_root,
            output_dir: &output_dir,
            file_name: "iris_smoke.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        },
    )
    .expect("the smoke plan builds from the real target");

    let mut snapshot = stub_job_snapshot(TRAIN_JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("train job deserializes");
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    if ranks > 1 {
        crate::training_jobs::run_training_execution_drive(
            &api,
            &settings,
            &job,
            &plan,
            |_: &str, _: &gen_core::LoadSpec| -> gen_core::Result<Box<dyn gen_core::Trainer>> {
                panic!("a multi-rank run never loads a trainer in the worker process")
            },
            Arc::new(crate::iris_dist::test_support::TestBinaryRankLauncher),
        )
        .await
        .expect("the two-rank Iris LoRA training job completes");
    } else {
        crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
            .await
            .expect("the Iris LoRA training job completes");
    }
    let train_seconds = started.elapsed().as_secs_f64();

    let result = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == TRAIN_JOB && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the training job posted a completed result");
    assert_eq!(result["stepsCompleted"], TRAIN_STEPS, "{result:#}");
    if ranks > 1 {
        let dist = &result["irisDistributed"];
        assert_eq!(dist["worldSize"], json!(ranks), "{result:#}");
        assert_eq!(dist["shareDevice"], json!(true), "{result:#}");
        assert_eq!(
            result["irisPlan"]["globalBatch"],
            json!(ranks),
            "{result:#}"
        );
    }
    let final_loss = result["finalLoss"].as_f64().expect("final loss");
    assert!(final_loss.is_finite(), "final loss {final_loss}");
    assert!(
        !result["checkpoints"].as_array().unwrap().is_empty(),
        "a durable checkpoint was published: {result:#}"
    );
    assert!(
        !result["trainingSamples"].as_array().unwrap().is_empty(),
        "a preview rendered: {result:#}"
    );
    let adapter = output_dir.join("iris_smoke.safetensors");
    let metadata = safetensors_metadata(&adapter);
    for (key, expected) in [
        ("family", "iris"),
        ("baseModel", "iris_3b"),
        ("irisTask", "generation"),
        ("irisArtifact", "adapter"),
        ("networkType", "lora"),
        ("irisWeights", "raw"),
    ] {
        assert_eq!(
            metadata.get(key).and_then(Value::as_str),
            Some(expected),
            "adapter metadata {key}: {metadata:#?}"
        );
    }
    assert!(
        output_dir
            .join("iris_smoke.checkpoints")
            .join("latest")
            .is_file(),
        "the checkpoint root carries its latest pointer"
    );

    // The registered adapter, as the LoRA registry hands it to a generation job.
    let base = render(
        &settings,
        &posts,
        &project_object,
        "iris-render-base",
        json!([]),
    )
    .await;
    let tuned = render(
        &settings,
        &posts,
        &project_object,
        "iris-render-lora",
        json!([{
            "id": "lora_iris_smoke",
            "name": "Iris smoke",
            "family": "iris",
            "networkType": "lora",
            "installedPath": adapter.display().to_string(),
            "weight": 1.0
        }]),
    )
    .await;
    assert_eq!((tuned.width, tuned.height), (RENDER_SIZE, RENDER_SIZE));
    let std = crate::smoke_support::image_std(&tuned);
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the adapter render looks degenerate (RGB std {std:.2})"
    );
    let mean_abs_diff = base
        .pixels
        .iter()
        .zip(&tuned.pixels)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / base.pixels.len() as f64;
    assert!(
        mean_abs_diff > 0.5,
        "the trained adapter must change the render at a fixed seed (mean |Δ| {mean_abs_diff:.3})"
    );
    eprintln!(
        "RESULT status=pass trainer=iris_3b ranks={ranks} backend={} steps={TRAIN_STEPS} resolution={RESOLUTION} \
         final_loss={final_loss:.4} train_s={train_seconds:.1} adapter={} render_std={std:.2} \
         adapter_effect_mean_abs_diff={mean_abs_diff:.3}",
        if cfg!(target_os = "macos") { "mlx" } else { "candle" },
        adapter.display()
    );
}

// =============================================================================================
// Iris-3B DEPTH (S9 sc-25687) and RESTORATION (S10 sc-25688) training — the same production path:
// a bounded run through `training_jobs::run_training_execution` from the REAL target, its export
// written where the API registers it (`<data>/models/finetunes/<id>/<stem>/`), then the trained
// model SELECTED by id (`irisModel`) in the depth / restoration job — no file surgery.
// =============================================================================================

fn smoke_settings(temp: &Path, posts_url: String, worker: &str) -> Settings {
    let data_dir = temp.join("data");
    std::fs::create_dir_all(&data_dir).unwrap();
    let mut settings = Settings::for_test(data_dir.canonicalize().unwrap());
    settings.api_url = posts_url;
    settings.config_dir = temp.join("config");
    settings.worker_id = worker.to_owned();
    settings.heartbeat_seconds = 30;
    if cfg!(target_os = "macos") {
        settings.gpu_id = "mlx".to_owned();
        settings.backend_mlx_enabled = true;
    } else {
        settings.gpu_id = std::env::var("CUDA_VISIBLE_DEVICES")
            .ok()
            .and_then(|devices| devices.split(',').next().map(|id| id.trim().to_owned()))
            .filter(|id| !id.is_empty())
            .unwrap_or_else(|| "0".to_owned());
        settings.backend_candle_enabled = true;
    }
    settings
}

fn write_npy_f32(path: &Path, h: u32, w: u32, f: impl Fn(u32, u32) -> f32) {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({h}, {w}), }}").into_bytes();
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(b' ');
    }
    header.push(b'\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&header);
    for y in 0..h {
        for x in 0..w {
            bytes.extend_from_slice(&f(y, x).to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

fn task_target(base_model: &str) -> sceneworks_core::training::TrainingTarget {
    sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.base_model == base_model)
        .expect("the Iris task target is registered")
}

fn completed_result(posts: &ProgressPosts, job_id: &str) -> Value {
    posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == job_id && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .unwrap_or_else(|| panic!("{job_id} posted a completed result"))
}

fn import_source(
    store: &ProjectStore,
    project_id: &str,
    temp: &Path,
    image: image::RgbImage,
) -> String {
    let source_file = temp.join("source.png");
    image.save(&source_file).unwrap();
    let asset = store
        .import_asset(
            project_id,
            sceneworks_core::project_store::UploadAsset {
                filename: "source.png".to_owned(),
                content_type: Some("image/png".to_owned()),
                source_path: source_file,
                source_asset_id: None,
                provenance: None,
            },
        )
        .expect("source imports");
    asset["id"].as_str().unwrap().to_owned()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris DEPTH training smoke; needs the pinned speridlabs/iris-3b depth/ export under HF_HUB_CACHE and a GPU (~65 GB)"]
async fn iris_training_gpu_smoke_depth() {
    const JOB: &str = "iris-depth-train-smoke";
    const SIZE: u32 = 128;
    const STEPS: u32 = 4;
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_worker_api_stub().await;
    let settings = smoke_settings(temp.path(), api_url, "iris-depth-train-smoke");
    let data_dir = settings.data_dir.clone();
    let store = ProjectStore::new(data_dir.clone(), "iris-depth-train-smoke");
    let project = store.create_project("Iris depth training smoke").unwrap();

    // Two aligned examples: an RGB gradient + a depth ramp (near at the bottom), one with a mask.
    let root = PathBuf::from(&project.path).join("training/datasets/ds_depth");
    std::fs::create_dir_all(root.join("images")).unwrap();
    std::fs::create_dir_all(root.join("sidecars")).unwrap();
    let items: Vec<Value> = (0..2u32)
        .map(|index| {
            image::RgbImage::from_fn(SIZE, SIZE, |x, y| {
                image::Rgb([(x * 2) as u8, (255 - y * 2) as u8, (60 * index + 40) as u8])
            })
            .save(root.join(format!("images/i{index}.png")))
            .unwrap();
            write_npy_f32(
                &root.join(format!("sidecars/i{index}.depth.npy")),
                SIZE,
                SIZE,
                |y, _| 1.0 + 9.0 * (1.0 - y as f32 / (SIZE - 1) as f32),
            );
            let mut depth =
                json!({ "path": format!("sidecars/i{index}.depth.npy"), "maxDepth": 80.0 });
            if index == 1 {
                image::GrayImage::from_fn(SIZE, SIZE, |x, _| image::Luma([u8::from(x > 4) * 255]))
                    .save(root.join("sidecars/i1.mask.png"))
                    .unwrap();
                depth["maskPath"] = json!("sidecars/i1.mask.png");
            }
            json!({
                "id": format!("i{index}"), "path": format!("images/i{index}.png"),
                "displayName": format!("i{index}.png"),
                "caption": { "text": "", "source": "manual", "triggerWords": [] },
                "addedAt": "2026-10-11T00:00:00Z", "irisDepth": depth
            })
        })
        .collect();
    let dataset: sceneworks_core::training::TrainingDataset = serde_json::from_value(json!({
        "schemaVersion": 1, "id": "ds_depth", "version": 1, "projectId": project.id,
        "name": "depth", "modality": "image", "status": "ready",
        "createdAt": "2026-10-11T00:00:00Z", "updatedAt": "2026-10-11T00:00:00Z", "items": items
    }))
    .unwrap();

    let target = task_target("iris_3b_depth");
    let mut config = target.defaults.clone();
    config.steps = STEPS;
    config.save_every = STEPS / 2;
    config.gradient_accumulation = 1;
    config.resolution = SIZE;
    config.learning_rate = sceneworks_core::contracts::ContractNumber::from_f64(1e-4).unwrap();
    config.advanced.insert("sampleEvery".into(), json!(STEPS));
    config.advanced["iris"]["depth_size"] = json!([SIZE, SIZE]);
    config.advanced["iris"]["preview_items"] = json!(1);
    config.advanced["iris"]["keep_last_checkpoints"] = json!(1);
    let model_id = "finetune_depth_smoke";
    let output_dir = data_dir.join("models/finetunes").join(model_id);
    let plan = sceneworks_core::training::build_training_plan(
        sceneworks_core::training::BuildTrainingPlan {
            job_id: JOB,
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: model_id,
            base_model_path: crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
                .expect("HF_HUB_CACHE holds the pinned speridlabs/iris-3b snapshot")
                .display()
                .to_string(),
            dataset_root: &root,
            output_dir: &output_dir,
            file_name: "depth_smoke.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        },
    )
    .expect("the depth plan builds from the real target");
    let mut snapshot = stub_job_snapshot(JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .expect("the Iris depth training job completes");
    let train_seconds = started.elapsed().as_secs_f64();
    let result = completed_result(&posts, JOB);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    assert!(
        !result["trainingSamples"].as_array().unwrap().is_empty(),
        "a depth preview rendered"
    );
    let export = output_dir.join("depth_smoke");
    for file in [
        "config.yaml",
        "model.safetensors",
        "empty_prompt.safetensors",
    ] {
        assert!(export.join(file).is_file(), "the depth export has {file}");
    }
    let metadata = safetensors_metadata(&export.join("model.safetensors"));
    assert_eq!(
        metadata.get("irisTask").and_then(Value::as_str),
        Some("depth")
    );

    // The trained model, selected by id in a depth job.
    let source = import_source(
        &store,
        &project.id,
        temp.path(),
        image::RgbImage::from_fn(256, 192, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 90])
        }),
    );
    let depth_job_id = "iris-depth-trained-job";
    let mut snapshot = stub_job_snapshot(depth_job_id, "image_depth", &project.id);
    let mut trained_entry =
        crate::training_jobs::builtin_model_manifest_entry("iris_3b_depth").unwrap();
    trained_entry["id"] = json!(model_id);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "sourceAssetId": source,
        "resolution": { "mode": "capped", "maxSide": 256 },
        "irisModel": model_id,
        // The API attaches the trained model's own entry (its engine's depth declaration
        // overlaid) in place of the release entry.
        "modelManifestEntries": [trained_entry],
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    crate::depth_jobs::run_image_depth_job(&api, &settings, &job)
        .await
        .expect("the depth job runs on the trained model");
    let fact = completed_result(&posts, depth_job_id)["assetWrites"][0].clone();
    assert_eq!(fact["model"], model_id, "{fact:#}");
    assert_eq!(fact["depth"]["model"]["trainedModelId"], model_id);
    let npy = std::fs::read(PathBuf::from(&project.path).join(fact["mediaPath"].as_str().unwrap()))
        .unwrap();
    let map = sceneworks_core::depth_map::decode_npy_f32(&npy).unwrap();
    assert_eq!((map.width, map.height), (256, 192));
    assert!(map.values.iter().all(|v| v.is_finite()));
    eprintln!(
        "RESULT status=pass trainer=iris_3b_depth backend={} steps={STEPS} size={SIZE} \
         final_loss={} train_s={train_seconds:.1} export={} depth_job_model={model_id}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
        result["finalLoss"],
        export.display()
    );
}

/// The bounded restoration config of a smoke: the real target's defaults (the paper's recipe) at a
/// 256² center crop, two optimizer steps (= generator, critic, generator, critic); `full` keeps the
/// paper's FULL fine-tune, otherwise the LoRA arm.
pub(crate) fn restore_smoke_config(full: bool) -> sceneworks_core::training::TrainingConfig {
    let mut config = task_target("iris_3b_restore").defaults;
    config.steps = RESTORE_SMOKE_STEPS;
    config.save_every = RESTORE_SMOKE_STEPS;
    config.resolution = RESTORE_SMOKE_CROP;
    if !full {
        config.rank = 8;
        config.alpha = 8;
        config.advanced.insert("networkType".into(), json!("lora"));
    }
    config
        .advanced
        .insert("sampleEvery".into(), json!(RESTORE_SMOKE_STEPS));
    config.advanced["iris_restore"]["crop"] = json!("center");
    config.advanced["iris_restore"]["keep_last_checkpoints"] = json!(1);
    config
}

const RESTORE_SMOKE_CROP: u32 = 256;
// Two steps = generator, critic, generator, critic.
const RESTORE_SMOKE_STEPS: u32 = 4;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris RESTORATION training smoke (LoRA arm); needs the pinned upscaler/ export and the three training resources under HF_HUB_CACHE and a GPU (~30 GB)"]
async fn iris_training_gpu_smoke_restore() {
    run_restore_training_smoke(false).await;
}

/// sc-25690: the paper's FULL fine-tune through the job path at the smallest admitted shape (a
/// 256² crop, batch 1) — the run the restoration `full` floor rows gate — then its export selected
/// by id in a restore job.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris RESTORATION FULL fine-tune smoke; needs the pinned upscaler/ export and the three training resources under HF_HUB_CACHE and ~95 GiB of unified memory or VRAM"]
async fn iris_training_gpu_smoke_full_ft_restore() {
    run_restore_training_smoke(true).await;
}

async fn run_restore_training_smoke(full: bool) {
    const SIZE: u32 = RESTORE_SMOKE_CROP;
    const STEPS: u32 = RESTORE_SMOKE_STEPS;
    let arm = if full { "full" } else { "lora" };
    let job_name = format!("iris-restore-{arm}-train-smoke");
    let job_id = job_name.as_str();
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_worker_api_stub().await;
    let settings = smoke_settings(temp.path(), api_url, job_id);
    let data_dir = settings.data_dir.clone();
    let store = ProjectStore::new(data_dir.clone(), job_id);
    let project = store
        .create_project("Iris restoration training smoke")
        .unwrap();

    // Clean images: synthetic (seeded) second-order degradation.
    let root = PathBuf::from(&project.path).join("training/datasets/ds_restore");
    std::fs::create_dir_all(root.join("images")).unwrap();
    let items: Vec<Value> = (0..2u32)
        .map(|index| {
            image::RgbImage::from_fn(SIZE, SIZE, |x, y| {
                image::Rgb([
                    ((x / 8 + y / 8 + index) % 2 * 200) as u8,
                    (x % 256) as u8,
                    (y % 256) as u8,
                ])
            })
            .save(root.join(format!("images/c{index}.png")))
            .unwrap();
            json!({
                "id": format!("c{index}"), "path": format!("images/c{index}.png"),
                "displayName": format!("c{index}.png"),
                "caption": { "text": "", "source": "manual", "triggerWords": [] },
                "addedAt": "2026-10-11T00:00:00Z"
            })
        })
        .collect();
    let dataset: sceneworks_core::training::TrainingDataset = serde_json::from_value(json!({
        "schemaVersion": 1, "id": "ds_restore", "version": 1, "projectId": project.id,
        "name": "restore", "modality": "image", "status": "ready",
        "createdAt": "2026-10-11T00:00:00Z", "updatedAt": "2026-10-11T00:00:00Z", "items": items
    }))
    .unwrap();

    let target = task_target("iris_3b_restore");
    let config = restore_smoke_config(full);
    let gate = admitted_floor(&settings, "iris_3b_restore", &config).await;
    let model_name = format!("finetune_restore_{arm}_smoke");
    let model_id = model_name.as_str();
    let output_dir = data_dir.join("models/finetunes").join(model_id);
    let plan = sceneworks_core::training::build_training_plan(
        sceneworks_core::training::BuildTrainingPlan {
            job_id,
            target: &target,
            dataset: &dataset,
            config,
            preset: None,
            lora_id: model_id,
            base_model_path: crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
                .expect("HF_HUB_CACHE holds the pinned speridlabs/iris-3b snapshot")
                .display()
                .to_string(),
            dataset_root: &root,
            output_dir: &output_dir,
            file_name: "restore_smoke.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        },
    )
    .expect("the restoration plan builds from the real target");
    let mut snapshot = stub_job_snapshot(job_id, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .expect("the Iris restoration training job completes (training resources verified)");
    let train_seconds = started.elapsed().as_secs_f64();
    let result = completed_result(&posts, job_id);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    assert!(
        !result["trainingSamples"].as_array().unwrap().is_empty(),
        "a restoration preview"
    );
    let export = output_dir.join("restore_smoke");
    for file in [
        "config.yaml",
        "model.safetensors",
        "empty_prompt.safetensors",
    ] {
        assert!(
            export.join(file).is_file(),
            "the upscaler export has {file}"
        );
    }
    let metadata = safetensors_metadata(&export.join("model.safetensors"));
    assert_eq!(
        metadata.get("irisTask").and_then(Value::as_str),
        Some("restoration")
    );

    // The trained model, selected by id in an Iris restore job (a 1× restore).
    let source = import_source(
        &store,
        &project.id,
        temp.path(),
        image::RgbImage::from_fn(128, 96, |x, y| {
            image::Rgb([
                (x * 2) as u8,
                (y * 2) as u8,
                ((x / 8 + y / 8) % 2 * 200) as u8,
            ])
        }),
    );
    let restore_job_id = "iris-restore-trained-job";
    let mut snapshot = stub_job_snapshot(restore_job_id, "image_upscale", &project.id);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "sourceAssetId": source,
        "engine": "iris",
        "scale": 1.0,
        "irisModel": model_id,
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    crate::upscale_jobs::run_image_upscale_job(&api, &settings, &job)
        .await
        .expect("the restore job runs on the trained model");
    let asset = completed_result(&posts, restore_job_id)["assetWrites"][0].clone();
    let restored =
        image::open(PathBuf::from(&project.path).join(asset["mediaPath"].as_str().unwrap()))
            .unwrap()
            .to_rgb8();
    assert_eq!(restored.dimensions(), (128, 96));
    eprintln!(
        "RESULT status=pass trainer=iris_3b_restore arm={arm} backend={} steps={STEPS} crop={SIZE} \
         {gate} final_loss={} train_s={train_seconds:.1} export={} restore_job_model={model_id}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
        result["finalLoss"],
        export.display()
    );
}

// =============================================================================================
// sc-25690 (epic 25678 terminal evidence): a LoKr generation run CANCELED through the job API after
// its first durable checkpoint, then RESUMED from that checkpoint to completion — the production
// path end to end (`run_training_execution`, the in-loop cancel poller, the trainer's durable
// cancel, the API's `resumeFromJobId` → `iris.resume_from` translation reproduced exactly as
// `apps/rust-api` `resolve_iris_run_sources` writes it), then the LoKr adapter applied by a
// generation job.
// =============================================================================================

type CancelSwitch = Arc<std::sync::atomic::AtomicBool>;
type CancellableState = (ProgressPosts, CancelSwitch);

fn cancellable_snapshot(job_id: &str, cancel: &CancelSwitch) -> Value {
    let mut job = stub_job_snapshot(job_id, "lora_train", "smoke");
    job["cancelRequested"] = json!(cancel.load(std::sync::atomic::Ordering::SeqCst));
    job
}

async fn cancellable_job(
    State((_, cancel)): State<CancellableState>,
    AxumPath(job_id): AxumPath<String>,
) -> Json<Value> {
    Json(cancellable_snapshot(&job_id, &cancel))
}

async fn cancellable_progress(
    State((posts, cancel)): State<CancellableState>,
    AxumPath(job_id): AxumPath<String>,
    Json(payload): Json<Value>,
) -> Json<Value> {
    posts.lock().unwrap().push((job_id.clone(), payload));
    Json(cancellable_snapshot(&job_id, &cancel))
}

/// The worker API stub with a user cancel: every job snapshot (the cancel poller's GET and each
/// progress-post reply) reports `cancelRequested` from `cancel`.
async fn spawn_cancellable_api_stub(cancel: CancelSwitch) -> (String, ProgressPosts) {
    let posts: ProgressPosts = Arc::default();
    let app = Router::new()
        .route("/api/v1/jobs/:job_id", get(cancellable_job))
        .route("/api/v1/jobs/:job_id/progress", post(cancellable_progress))
        .route("/api/v1/jobs/:job_id/metrics", post(stub_metrics))
        .route("/api/v1/workers/:worker_id/heartbeat", post(stub_heartbeat))
        .with_state((posts.clone(), cancel));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), posts)
}

/// The floor row the admission gate reads for `config` on this lane, and the capacity it is judged
/// against — printed in a smoke's RESULT line so the evidence names the row a real job ran under.
/// Panics when the gate would refuse: the job below would be refused before load.
async fn admitted_floor(
    settings: &Settings,
    base_model: &str,
    config: &sceneworks_core::training::TrainingConfig,
) -> String {
    use sceneworks_core::training::iris::{curricula, iris_objective_memory_floor, IrisObjective};
    let lane = crate::iris_training::training_lane();
    #[cfg(target_os = "macos")]
    let capacity = {
        let _ = settings;
        crate::iris_training::mlx_training_capacity_gb()
    };
    #[cfg(not(target_os = "macos"))]
    let capacity = crate::iris_training::candle_training_capacity_gb(&settings.gpu_id)
        .await
        .ok();
    let objective = IrisObjective::from_base_model(base_model).expect("an Iris base model");
    let floor = match objective {
        IrisObjective::Generation => curricula::iris_run_memory_floor(lane, config, 1),
        _ => iris_objective_memory_floor(objective, lane, config),
    }
    .expect("a floor row covers the smoke's shape");
    let refusal =
        crate::iris_training::iris_plan_memory_refusal(lane, base_model, config, 1, capacity);
    assert_eq!(refusal, None, "the gate admits the smoke on this host");
    format!(
        "gate_floor_gib={} gate_floor_provenance={} gate_capacity_gib={}",
        floor.gib,
        floor.provenance.as_str(),
        capacity.map_or_else(|| "unread".to_owned(), |gib| format!("{gib:.2}"))
    )
}

/// The micro-steps a job's Training progress posts reported, in post order. The worker posts one
/// update per trainer `Training` event (`map_training_progress`: "Training step N of T (loss …)."),
/// and the Iris trainers emit that event only after an optimizer step completes.
fn posted_training_steps(posts: &[(String, Value)], job_id: &str) -> Vec<u64> {
    posts
        .iter()
        .filter(|(id, _)| id == job_id)
        .filter_map(|(_, body)| {
            body["message"]
                .as_str()?
                .strip_prefix("Training step ")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
        .collect()
}

/// sc-25690: what a RESUMED run's own progress must show to have CONTINUED from the canceled run's
/// durable stop at `stopped_at` rather than restarted. `stepsCompleted == total` alone cannot tell
/// the two apart — a run that ignored `iris.resume_from` and trained 1..=total also ends at
/// `total`. A continuation trains exactly `stopped_at + 1 ..= total`: its first step is
/// `stopped_at + 1`, it never reports a step at or under the stop, and it executes
/// `total − stopped_at` steps. Returns the executed step count.
fn resume_continuation(resumed_steps: &[u64], stopped_at: u64, total: u64) -> Result<u64, String> {
    let Some(first) = resumed_steps.first() else {
        return Err("the resumed run reported no training step".to_owned());
    };
    if let Some(repeated) = resumed_steps.iter().find(|step| **step <= stopped_at) {
        return Err(format!(
            "the resumed run trained step {repeated}, at or under the canceled run's stop at \
             {stopped_at}: it restarted instead of continuing (steps {resumed_steps:?})"
        ));
    }
    let expected: Vec<u64> = (stopped_at + 1..=total).collect();
    if resumed_steps != expected {
        return Err(format!(
            "the resumed run must train exactly steps {}..={total} (first {first}); it reported \
             {resumed_steps:?}",
            stopped_at + 1
        ));
    }
    Ok(expected.len() as u64)
}

/// The two generation runs the cancel → resume smoke drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CancelResumeArm {
    /// A LoKr adapter (256², AdamW).
    LoKr,
    /// The official `stage1_256` FULL fine-tune (REPA against the DINOv2 teacher, Muon), bounded.
    FullStage1,
}

const CANCEL_RESUME_STEPS: u32 = 8;

impl CancelResumeArm {
    fn label(self) -> &'static str {
        match self {
            Self::LoKr => "lokr",
            Self::FullStage1 => "full",
        }
    }

    /// The first durable checkpoint lands here; the cancel is requested once it has.
    fn save_every(self) -> u32 {
        match self {
            Self::LoKr => 2,
            // A full-model checkpoint is tens of GB: one before the cancel, one at the end.
            Self::FullStage1 => 4,
        }
    }

    /// The run's config, from the REAL target: the LoKr arm from its defaults; the full arm from
    /// the engine's own `stage1_256` preset as the Studio offers it
    /// (`iris_curricula::iris_stage_presets`), edited only for the bound — one device, batch 1 (so
    /// no multi-node mesh and no 1024 global batch), no EMA copy, eight steps with no warmup, one
    /// retained checkpoint, the smoke's preview prompt. The stage identity, Muon, the flow shift,
    /// the RoPE law and REPA are the preset's.
    pub(crate) fn config(self) -> sceneworks_core::training::TrainingConfig {
        let mut config = match self {
            Self::LoKr => {
                let mut config = sceneworks_core::training::iris::iris_generation_target().defaults;
                config.rank = 4;
                config.alpha = 4;
                config.advanced.insert("networkType".into(), json!("lokr"));
                config.advanced.insert("decomposeFactor".into(), json!(4));
                config
            }
            Self::FullStage1 => {
                let preset = crate::iris_curricula::iris_stage_presets()
                    .expect("the engine's stage presets translate")
                    .into_iter()
                    .find(|preset| preset["id"] == "stage1_256")
                    .expect("stage1_256 is an official stage");
                let mut config: sceneworks_core::training::TrainingConfig =
                    serde_json::from_value(preset["config"].clone())
                        .expect("a stage preset is a training config");
                config.batch_size = 1;
                config.gradient_accumulation = 1;
                config.advanced.insert("lrWarmupSteps".into(), json!(0));
                let options = config.advanced["iris"]
                    .as_object_mut()
                    .expect("a stage preset carries the iris options");
                for key in ["dist", "expected_global_batch"] {
                    options.remove(key);
                }
                options.insert("ema_enabled".into(), json!(false));
                options.insert("keep_last_checkpoints".into(), json!(1));
                options.insert("upstream_validation_prompts".into(), json!(false));
                config
            }
        };
        config.steps = CANCEL_RESUME_STEPS;
        config.save_every = self.save_every();
        config.resolution = RESOLUTION;
        config.learning_rate =
            sceneworks_core::contracts::ContractNumber::from_f64(1e-3).expect("finite");
        config.trigger_word = Some("smoke style".to_owned());
        config
            .advanced
            .insert("sampleEvery".into(), json!(CANCEL_RESUME_STEPS));
        config.advanced.insert("sampleSteps".into(), json!(10));
        config.advanced.insert("sampleCount".into(), json!(1));
        config.advanced.insert(
            "samplePrompts".into(),
            json!(["a flat red square, smoke style"]),
        );
        config
    }
}

/// The resumed run's plan config, exactly as `apps/rust-api` `resolve_iris_run_sources` writes the
/// `resumeFromJobId` translation: `advanced.iris.resume_from` = the previous run's checkpoint root
/// and `advanced.resume` = true.
pub(crate) fn apply_resume_translation(
    config: &mut sceneworks_core::training::TrainingConfig,
    checkpoint_root: &Path,
) {
    let mut options = match config.advanced.remove("iris") {
        Some(Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    options.insert(
        "resume_from".to_owned(),
        json!(checkpoint_root.display().to_string()),
    );
    config
        .advanced
        .insert("iris".to_owned(), Value::Object(options));
    config.advanced.insert("resume".to_owned(), json!(true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris LoKr cancel/resume smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and an Apple-Silicon GPU or a CUDA device"]
async fn iris_training_gpu_smoke_lokr_cancel_resume() {
    run_cancel_resume_smoke(CancelResumeArm::LoKr).await;
}

/// sc-25690: the generation FULL fine-tune through the job path — the official `stage1_256`
/// (REPA, Muon), admitted by the gate at its full-fine-tune floor row, canceled after its first
/// durable checkpoint, resumed to the end, exported, and the exported checkpoint rendered by a
/// generation job exactly as a registered trained model is.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris generation FULL fine-tune cancel/resume smoke (stage1_256 + REPA); needs the pinned speridlabs/iris-3b, Qwen3-VL-4B-Instruct and facebook/dinov2-base snapshots under HF_HUB_CACHE and ~90 GiB of unified memory or VRAM"]
async fn iris_training_gpu_smoke_full_ft_generation_cancel_resume() {
    run_cancel_resume_smoke(CancelResumeArm::FullStage1).await;
}

async fn run_cancel_resume_smoke(arm: CancelResumeArm) {
    const STEPS: u32 = CANCEL_RESUME_STEPS;
    let label = arm.label();
    let full = arm == CancelResumeArm::FullStage1;
    let first_job = format!("iris-train-{label}");
    let resume_job = format!("iris-train-{label}-resume");
    let file = format!("iris_{label}.safetensors");
    let output_id = if full {
        "finetune_iris_full_smoke".to_owned()
    } else {
        "lora_iris_lokr".to_owned()
    };
    let cancel: CancelSwitch = Arc::default();
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_cancellable_api_stub(cancel.clone()).await;
    let settings = smoke_settings(temp.path(), api_url, &format!("iris-{label}-smoke-worker"));
    let data_dir = settings.data_dir.clone();
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE must already hold the Iris backbone (no download at training time)");
    let store = ProjectStore::new(data_dir.clone(), &format!("iris-{label}-smoke"));
    let project = store.create_project("Iris cancel/resume smoke").unwrap();
    let project_object = serde_json::to_value(&project)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    let dataset_root = PathBuf::from(&project.path).join("training/datasets/ds_smoke");
    let dataset = write_dataset(&dataset_root);
    // An adapter is written beside its project; a full run where the API registers a trained
    // model from (`<data>/models/finetunes/<id>`).
    let output_dir = if full {
        data_dir.join("models/finetunes").join(&output_id)
    } else {
        PathBuf::from(&project.path).join("loras").join(&output_id)
    };

    let target = sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.kernel == "iris_train")
        .expect("the Iris target is registered");
    let config = arm.config();
    let gate = admitted_floor(&settings, "iris_3b", &config).await;
    let build = |job_id: &str, config| {
        sceneworks_core::training::build_training_plan(
            sceneworks_core::training::BuildTrainingPlan {
                job_id,
                target: &target,
                dataset: &dataset,
                config,
                preset: None,
                lora_id: &output_id,
                base_model_path: backbone.display().to_string(),
                dataset_root: &dataset_root,
                output_dir: &output_dir,
                file_name: file.clone(),
                created_at: "2026-10-12T00:00:00Z".to_owned(),
            },
        )
        .expect("the plan builds from the real target")
    };
    let api = ApiClient::new(&settings);

    // 1. Run, and request a cancel the moment the first durable checkpoint lands.
    let checkpoint_root = crate::iris_checkpoint_root(&output_dir, &file);
    let watcher = {
        let latest = checkpoint_root.join("latest");
        let cancel = cancel.clone();
        tokio::spawn(async move {
            while !latest.is_file() {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
            cancel.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    let plan = build(&first_job, config.clone());
    let mut snapshot = stub_job_snapshot(&first_job, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let started = std::time::Instant::now();
    let outcome = crate::training_jobs::run_training_execution(&api, &settings, &job, &plan).await;
    let first_seconds = started.elapsed().as_secs_f64();
    watcher.abort();
    assert!(
        matches!(outcome, Err(WorkerError::Canceled(_))),
        "the canceled {label} run ends Canceled: {outcome:?}"
    );
    let canceled = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| *id == first_job && body["status"] == "canceled")
        .map(|(_, body)| body.clone())
        .expect("the run posted a terminal canceled update");
    let stopped_at = canceled["result"]["stepsCompleted"]
        .as_u64()
        .unwrap_or_else(|| panic!("a durable cancel reports its step: {canceled:#}"));
    assert!(
        (u64::from(arm.save_every())..u64::from(STEPS)).contains(&stopped_at),
        "canceled after the first checkpoint and before the end: {canceled:#}"
    );
    assert!(
        checkpoint_root.join("latest").is_file(),
        "the canceled run kept its durable checkpoint"
    );

    // 2. Resume from that run's checkpoint (the API's `resumeFromJobId` translation) to the end.
    cancel.store(false, std::sync::atomic::Ordering::SeqCst);
    let mut plan = build(&resume_job, config);
    apply_resume_translation(&mut plan.config, &checkpoint_root);
    let mut snapshot = stub_job_snapshot(&resume_job, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .unwrap_or_else(|error| panic!("the resumed {label} run completes: {error}"));
    let resume_seconds = started.elapsed().as_secs_f64();
    let result = completed_result(&posts, &resume_job);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    // The resumed job CONTINUED: its own progress starts one past the stop and never repeats a
    // step the canceled run already trained. (`stepsCompleted` alone is also what a restart from
    // step 0 reports.)
    let resumed_steps = posted_training_steps(&posts.lock().unwrap(), &resume_job);
    let resumed_executed = resume_continuation(&resumed_steps, stopped_at, u64::from(STEPS))
        .unwrap_or_else(|error| panic!("{error}"));
    let resumed_first = resumed_steps[0];
    for checkpoint in result["checkpoints"].as_array().unwrap() {
        assert!(
            checkpoint["step"].as_u64() > Some(stopped_at),
            "a resumed checkpoint is past the stop at {stopped_at}: {result:#}"
        );
    }
    let final_loss = result["finalLoss"].as_f64().expect("final loss");
    assert!(final_loss.is_finite(), "final loss {final_loss}");

    // 3. The resumed artifact is applied by a generation job.
    let base = render(
        &settings,
        &posts,
        &project_object,
        &format!("iris-{label}-base"),
        json!([]),
    )
    .await;
    let tuned = if full {
        // The full export, where the API registers it from, selected as a trained checkpoint.
        assert_eq!(
            result["irisPlan"]["stage"],
            json!("stage1_256"),
            "{result:#}"
        );
        assert!(
            result["irisPlan"]["repa"].is_object(),
            "the stage's REPA alignment ran: {result:#}"
        );
        let export = output_dir.join(format!("iris_{label}"));
        for name in ["config.yaml", "model.safetensors"] {
            assert!(
                export.join(name).is_file(),
                "the full export has {name}: {}",
                export.display()
            );
        }
        let metadata = safetensors_metadata(&export.join("model.safetensors"));
        assert_eq!(
            metadata.get("irisTask").and_then(Value::as_str),
            Some("generation"),
            "export metadata: {metadata:#?}"
        );
        render_model(
            &settings,
            &posts,
            &project_object,
            "iris-full-tuned",
            json!([]),
            Some((&output_id, &export)),
        )
        .await
    } else {
        let adapter = output_dir.join(&file);
        let metadata = safetensors_metadata(&adapter);
        for (key, expected) in [
            ("family", "iris"),
            ("baseModel", "iris_3b"),
            ("irisTask", "generation"),
            ("networkType", "lokr"),
        ] {
            assert_eq!(
                metadata.get(key).and_then(Value::as_str),
                Some(expected),
                "adapter metadata {key}: {metadata:#?}"
            );
        }
        render(
            &settings,
            &posts,
            &project_object,
            "iris-lokr-tuned",
            json!([{
                "id": output_id,
                "name": "Iris LoKr smoke",
                "family": "iris",
                "networkType": "lokr",
                "installedPath": adapter.display().to_string(),
                "weight": 1.0
            }]),
        )
        .await
    };
    let std = crate::smoke_support::image_std(&tuned);
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the {label} render looks degenerate (RGB std {std:.2})"
    );
    let mean_abs_diff = base
        .pixels
        .iter()
        .zip(&tuned.pixels)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / base.pixels.len() as f64;
    assert!(
        mean_abs_diff > 0.5,
        "the resumed {label} artifact must change the render at a fixed seed (mean |Δ| {mean_abs_diff:.3})"
    );
    eprintln!(
        "RESULT status=pass trainer=iris_3b network={label} backend={} {gate} \
         canceled_at_step={stopped_at} first_checkpoint_step={} resumed_first_step={resumed_first} \
         resumed_steps_executed={resumed_executed} resumed_to={STEPS} first_s={first_seconds:.1} \
         resume_s={resume_seconds:.1} final_loss={final_loss:.4} render_std={std:.2} \
         artifact_effect_mean_abs_diff={mean_abs_diff:.3}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
        arm.save_every(),
    );
}

/// What needs no GPU: the continuation rule the resume smokes assert, and that the full-fine-tune
/// smokes' own configs are valid plans the admission gate admits on the hosts they will run on —
/// checked here so a real-weight dispatch does not spend its hour discovering a refused config.
#[cfg(test)]
mod unit {
    use super::*;

    fn training(job: &str, step: u64) -> (String, Value) {
        (
            job.to_owned(),
            json!({ "status": "running", "stage": "training",
                    "message": format!("Training step {step} of 8 (loss 0.1234).") }),
        )
    }

    #[test]
    fn a_resumed_run_must_continue_one_past_the_stop() {
        // Canceled at 4 of 8: the continuation trains 5, 6, 7, 8.
        assert_eq!(resume_continuation(&[5, 6, 7, 8], 4, 8), Ok(4));
        assert_eq!(resume_continuation(&[8], 7, 8), Ok(1));
        // A run that ignored `iris.resume_from` restarts at 1 — and still ends at step 8.
        let restarted = resume_continuation(&[1, 2, 3, 4, 5, 6, 7, 8], 4, 8).unwrap_err();
        assert!(
            restarted.contains("trained step 1") && restarted.contains("restarted"),
            "{restarted}"
        );
        // Re-training the stop step itself is a repeat too.
        assert!(resume_continuation(&[4, 5, 6, 7, 8], 4, 8)
            .unwrap_err()
            .contains("trained step 4"));
        // Skipping ahead, stopping short, and reporting nothing are not continuations either.
        assert!(resume_continuation(&[6, 7, 8], 4, 8).is_err());
        assert!(resume_continuation(&[5, 6, 7], 4, 8).is_err());
        assert!(resume_continuation(&[5, 7, 8], 4, 8).is_err());
        assert!(resume_continuation(&[], 4, 8)
            .unwrap_err()
            .contains("no training step"));
    }

    #[test]
    fn training_steps_are_read_from_the_named_jobs_progress_posts_only() {
        let posts = vec![
            training("first", 1),
            training("first", 2),
            (
                "resumed".to_owned(),
                json!({ "status": "running", "message": "Loading base model." }),
            ),
            training("resumed", 3),
            (
                "resumed".to_owned(),
                json!({ "status": "running", "message": "Saved checkpoint at step 4." }),
            ),
            training("resumed", 4),
            (
                "resumed".to_owned(),
                json!({ "status": "completed", "result": { "stepsCompleted": 4 } }),
            ),
        ];
        assert_eq!(posted_training_steps(&posts, "resumed"), vec![3, 4]);
        assert_eq!(posted_training_steps(&posts, "first"), vec![1, 2]);
        assert!(posted_training_steps(&posts, "absent").is_empty());
        // The message is the worker's own: this is what `map_training_progress` posts.
        let (_, _, _, message) = crate::training_jobs::map_training_progress(
            gen_core::TrainingProgress::Training {
                step: 5,
                total: 8,
                loss: 0.25,
            },
            8,
        );
        assert_eq!(
            posted_training_steps(&[("j".to_owned(), json!({ "message": message }))], "j"),
            vec![5]
        );
    }

    /// The full-fine-tune smokes read the FULL floor rows, and the hosts they are dispatched to —
    /// nax-macos-2 (128 GiB unified) and the campaign RTX PRO 6000 (95.59 GiB) — are admitted.
    #[test]
    fn the_full_fine_tune_smokes_are_admitted_on_the_hosts_they_run_on() {
        use sceneworks_core::training::iris::floors::CAMPAIGN_CUDA_CARD;
        use sceneworks_core::training::iris::{
            curricula, iris_generation_memory_floor, restore, validate_iris_training_config,
        };
        let hosts = [("mlx", 128.0), ("candle", CAMPAIGN_CUDA_CARD.gib())];

        let generation = CancelResumeArm::FullStage1.config();
        validate_iris_training_config(
            &sceneworks_core::training::iris::iris_generation_target(),
            &generation,
        )
        .expect("the bounded stage1_256 config is a valid Iris request");
        assert_eq!(generation.advanced["networkType"], json!("full"));
        assert_eq!(generation.advanced["iris"]["stage"], json!("stage1_256"));
        assert_eq!(generation.optimizer, "muon");
        let options = sceneworks_core::training::iris::iris_options(&generation).unwrap();
        assert!(
            curricula::iris_repa_enabled(&options),
            "stage 1 aligns to REPA"
        );
        assert_eq!(curricula::iris_world_size(&generation), 1);
        for (lane, capacity) in hosts {
            // One rank's full 256² row plus the REPA teacher.
            let row = iris_generation_memory_floor(lane, "full", RESOLUTION, false).unwrap();
            assert_eq!(
                curricula::iris_run_memory_floor(lane, &generation, 1).map(|floor| floor.gib),
                Some(row.gib + curricula::REPA_EXTRA_FLOOR_GIB)
            );
            assert_eq!(
                crate::iris_training::iris_plan_memory_refusal(
                    lane,
                    "iris_3b",
                    &generation,
                    1,
                    Some(capacity)
                ),
                None,
                "{lane}: stage1_256 full fine-tune"
            );
        }
        // The resumed run is the same request plus the API-owned resume keys.
        let mut resumed = generation.clone();
        apply_resume_translation(&mut resumed, Path::new("/data/run.checkpoints"));
        assert_eq!(
            resumed.advanced["iris"]["resume_from"],
            json!("/data/run.checkpoints")
        );
        assert_eq!(resumed.advanced["resume"], json!(true));
        assert_eq!(resumed.advanced["iris"]["stage"], json!("stage1_256"));

        let restoration = restore_smoke_config(true);
        restore::validate_restore_config(&restoration)
            .expect("the bounded paper recipe is a valid restoration request");
        assert_eq!(restoration.advanced["networkType"], json!("full"));
        let lora = restore_smoke_config(false);
        for (lane, capacity) in hosts {
            let full = restore::restore_memory_floor(lane, &restoration).unwrap();
            assert!(
                full.gib > restore::restore_memory_floor(lane, &lora).unwrap().gib,
                "{lane}: the full smoke reads the full row"
            );
            assert_eq!(
                crate::iris_training::iris_plan_memory_refusal(
                    lane,
                    "iris_3b_restore",
                    &restoration,
                    1,
                    Some(capacity)
                ),
                None,
                "{lane}: restoration full fine-tune at 256²"
            );
        }
    }
}
