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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris RESTORATION training smoke (LoRA arm); needs the pinned upscaler/ export and the three training resources under HF_HUB_CACHE and a GPU (~30 GB)"]
async fn iris_training_gpu_smoke_restore() {
    const JOB: &str = "iris-restore-train-smoke";
    const SIZE: u32 = 256;
    // Two steps = generator, critic, generator, critic.
    const STEPS: u32 = 4;
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_worker_api_stub().await;
    let settings = smoke_settings(temp.path(), api_url, "iris-restore-train-smoke");
    let data_dir = settings.data_dir.clone();
    let store = ProjectStore::new(data_dir.clone(), "iris-restore-train-smoke");
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
    let mut config = target.defaults.clone();
    config.rank = 8;
    config.alpha = 8;
    config.steps = STEPS;
    config.save_every = STEPS;
    config.resolution = SIZE;
    config.advanced.insert("networkType".into(), json!("lora"));
    config.advanced.insert("sampleEvery".into(), json!(STEPS));
    config.advanced["iris_restore"]["crop"] = json!("center");
    config.advanced["iris_restore"]["keep_last_checkpoints"] = json!(1);
    let model_id = "finetune_restore_smoke";
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
            file_name: "restore_smoke.safetensors".to_owned(),
            created_at: "2026-10-11T00:00:00Z".to_owned(),
        },
    )
    .expect("the restoration plan builds from the real target");
    let mut snapshot = stub_job_snapshot(JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .expect("the Iris restoration training job completes (training resources verified)");
    let train_seconds = started.elapsed().as_secs_f64();
    let result = completed_result(&posts, JOB);
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
        "RESULT status=pass trainer=iris_3b_restore arm=lora backend={} steps={STEPS} crop={SIZE} \
         final_loss={} train_s={train_seconds:.1} export={} restore_job_model={model_id}",
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight Iris LoKr cancel/resume smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and an Apple-Silicon GPU or a CUDA device"]
async fn iris_training_gpu_smoke_lokr_cancel_resume() {
    const FIRST_JOB: &str = "iris-train-lokr";
    const RESUME_JOB: &str = "iris-train-lokr-resume";
    const STEPS: u32 = 8;
    const FILE: &str = "iris_lokr.safetensors";
    let cancel: CancelSwitch = Arc::default();
    let temp = tempfile::tempdir().unwrap();
    let (api_url, posts) = spawn_cancellable_api_stub(cancel.clone()).await;
    let settings = smoke_settings(temp.path(), api_url, "iris-lokr-smoke-worker");
    let data_dir = settings.data_dir.clone();
    let backbone = crate::model_jobs::huggingface_snapshot_dir(&data_dir, BACKBONE_REPO)
        .expect("HF_HUB_CACHE must already hold the Iris backbone (no download at training time)");
    let store = ProjectStore::new(data_dir.clone(), "iris-lokr-smoke");
    let project = store
        .create_project("Iris LoKr cancel/resume smoke")
        .unwrap();
    let project_object = serde_json::to_value(&project)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    let dataset_root = PathBuf::from(&project.path).join("training/datasets/ds_smoke");
    let dataset = write_dataset(&dataset_root);
    let output_dir = PathBuf::from(&project.path).join("loras/lora_iris_lokr");

    let target = sceneworks_core::training::builtin_training_targets()
        .targets
        .into_iter()
        .find(|target| target.kernel == "iris_train")
        .expect("the Iris target is registered");
    let mut config = target.defaults.clone();
    config.rank = 4;
    config.alpha = 4;
    config.steps = STEPS;
    config.save_every = 2;
    config.resolution = RESOLUTION;
    config.learning_rate =
        sceneworks_core::contracts::ContractNumber::from_f64(1e-3).expect("finite");
    config.trigger_word = Some("smoke style".to_owned());
    config.advanced.insert("networkType".into(), json!("lokr"));
    config.advanced.insert("decomposeFactor".into(), json!(4));
    config.advanced.insert("sampleEvery".into(), json!(STEPS));
    config.advanced.insert("sampleSteps".into(), json!(10));
    config.advanced.insert("sampleCount".into(), json!(1));
    config.advanced.insert(
        "samplePrompts".into(),
        json!(["a flat red square, smoke style"]),
    );
    let build = |job_id: &str, config| {
        sceneworks_core::training::build_training_plan(
            sceneworks_core::training::BuildTrainingPlan {
                job_id,
                target: &target,
                dataset: &dataset,
                config,
                preset: None,
                lora_id: "lora_iris_lokr",
                base_model_path: backbone.display().to_string(),
                dataset_root: &dataset_root,
                output_dir: &output_dir,
                file_name: FILE.to_owned(),
                created_at: "2026-10-12T00:00:00Z".to_owned(),
            },
        )
        .expect("the LoKr plan builds from the real target")
    };
    let api = ApiClient::new(&settings);

    // 1. Run, and request a cancel the moment the first durable checkpoint lands.
    let checkpoint_root = crate::iris_checkpoint_root(&output_dir, FILE);
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
    let plan = build(FIRST_JOB, config.clone());
    let mut snapshot = stub_job_snapshot(FIRST_JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let started = std::time::Instant::now();
    let outcome = crate::training_jobs::run_training_execution(&api, &settings, &job, &plan).await;
    let first_seconds = started.elapsed().as_secs_f64();
    watcher.abort();
    assert!(
        matches!(outcome, Err(WorkerError::Canceled(_))),
        "the canceled LoKr run ends Canceled: {outcome:?}"
    );
    let canceled = posts
        .lock()
        .unwrap()
        .iter()
        .rev()
        .find(|(id, body)| id == FIRST_JOB && body["status"] == "canceled")
        .map(|(_, body)| body.clone())
        .expect("the run posted a terminal canceled update");
    let stopped_at = canceled["result"]["stepsCompleted"]
        .as_u64()
        .unwrap_or_else(|| panic!("a durable cancel reports its step: {canceled:#}"));
    assert!(
        (2..u64::from(STEPS)).contains(&stopped_at),
        "canceled after the first checkpoint and before the end: {canceled:#}"
    );
    assert!(
        checkpoint_root.join("latest").is_file(),
        "the canceled run kept its durable checkpoint"
    );

    // 2. Resume from that run's checkpoint (the API's `resumeFromJobId` translation) to the end.
    cancel.store(false, std::sync::atomic::Ordering::SeqCst);
    let mut plan = build(RESUME_JOB, config);
    let mut options = match plan.config.advanced.remove("iris") {
        Some(Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    options.insert(
        "resume_from".to_owned(),
        json!(checkpoint_root.display().to_string()),
    );
    plan.config
        .advanced
        .insert("iris".to_owned(), Value::Object(options));
    plan.config
        .advanced
        .insert("resume".to_owned(), json!(true));
    let mut snapshot = stub_job_snapshot(RESUME_JOB, "lora_train", &project.id);
    snapshot["payload"] = json!({ "dryRun": false, "plan": plan });
    let job: JobSnapshot = serde_json::from_value(snapshot).unwrap();
    let started = std::time::Instant::now();
    crate::training_jobs::run_training_execution(&api, &settings, &job, &plan)
        .await
        .expect("the resumed LoKr run completes");
    let resume_seconds = started.elapsed().as_secs_f64();
    let result = completed_result(&posts, RESUME_JOB);
    assert_eq!(result["stepsCompleted"], STEPS, "{result:#}");
    let final_loss = result["finalLoss"].as_f64().expect("final loss");
    assert!(final_loss.is_finite(), "final loss {final_loss}");
    let adapter = output_dir.join(FILE);
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

    // 3. The resumed adapter is applied by a generation job.
    let base = render(
        &settings,
        &posts,
        &project_object,
        "iris-lokr-base",
        json!([]),
    )
    .await;
    let tuned = render(
        &settings,
        &posts,
        &project_object,
        "iris-lokr-tuned",
        json!([{
            "id": "lora_iris_lokr",
            "name": "Iris LoKr smoke",
            "family": "iris",
            "networkType": "lokr",
            "installedPath": adapter.display().to_string(),
            "weight": 1.0
        }]),
    )
    .await;
    let mean_abs_diff = base
        .pixels
        .iter()
        .zip(&tuned.pixels)
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .sum::<f64>()
        / base.pixels.len() as f64;
    assert!(
        mean_abs_diff > 0.5,
        "the resumed LoKr adapter must change the render at a fixed seed (mean |Δ| {mean_abs_diff:.3})"
    );
    eprintln!(
        "RESULT status=pass trainer=iris_3b network=lokr backend={} canceled_at_step={stopped_at} \
         resumed_to={STEPS} first_s={first_seconds:.1} resume_s={resume_seconds:.1} \
         final_loss={final_loss:.4} adapter_effect_mean_abs_diff={mean_abs_diff:.3}",
        if cfg!(target_os = "macos") {
            "mlx"
        } else {
            "candle"
        },
    );
}
