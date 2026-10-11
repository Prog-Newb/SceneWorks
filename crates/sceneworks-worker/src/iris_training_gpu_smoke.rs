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
        "modelManifestEntry": shipped_iris_entry(),
    });
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
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .expect("the Iris LoRA training job completes");
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
        "RESULT status=pass trainer=iris_3b backend={} steps={TRAIN_STEPS} resolution={RESOLUTION} \
         final_loss={final_loss:.4} train_s={train_seconds:.1} adapter={} render_std={std:.2} \
         adapter_effect_mean_abs_diff={mean_abs_diff:.3}",
        if cfg!(target_os = "macos") { "mlx" } else { "candle" },
        adapter.display()
    );
}
