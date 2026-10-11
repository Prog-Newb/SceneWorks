//! Real-weight worker acceptance for Iris-3B on Candle/CUDA (sc-25680, epic 25678).
//!
//! Drives ONE production `image_generate` job through `run_image_generate_job` — the shipped
//! `iris_3b` catalog entry, cache-only resolution of the backbone AND its `text_encoder`
//! co-requisite through the generic component seam, the cached `candle-gen-iris` load, the per-item
//! drive and `consume_gen_events` — and asserts the job completes with one non-degenerate image
//! asset at the requested geometry. Nothing is downloaded: `HF_HUB_CACHE` must already hold both
//! pinned snapshots (the inference `candle-iris` real-weight profile materializes them into the same
//! `E:\huggingface\hub` on the real-weights host). An absent snapshot is a failure, never a skip.
//!
//! ```text
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! cargo test -p sceneworks-worker --features backend-candle --release iris_worker_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use super::*;

const JOB_ID: &str = "iris-cuda-image-generate";
const BACKBONE_REPO: &str = "speridlabs/iris-3b";
const TEXT_ENCODER_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
const SIZE: u32 = 512;
const STEPS: u32 = 30;

type ProgressPosts = Arc<Mutex<Vec<(String, Value)>>>;

fn stub_job_snapshot(job_id: &str) -> Value {
    json!({
        "id": job_id,
        "type": "image_generate",
        "status": "running",
        "projectId": null,
        "projectName": null,
        "payload": {},
        "result": {},
        "requestedGpu": "auto",
        "assignedGpu": "0",
        "workerId": "iris-smoke-worker",
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
        "createdAt": "2026-10-10T00:00:00Z",
        "updatedAt": "2026-10-10T00:00:00Z",
        "startedAt": "2026-10-10T00:00:00Z",
        "completedAt": null,
        "canceledAt": null,
        "lastHeartbeatAt": "2026-10-10T00:00:00Z"
    })
}

async fn stub_job(AxumPath(job_id): AxumPath<String>) -> Json<Value> {
    Json(stub_job_snapshot(&job_id))
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
    Json(stub_job_snapshot(&job_id))
}

async fn stub_heartbeat(AxumPath(worker_id): AxumPath<String>) -> Json<Value> {
    Json(json!({
        "id": worker_id, "gpuId": "0", "gpuName": null, "status": "busy",
        "currentJobId": JOB_ID, "capabilities": [], "loadedModels": [],
        "registeredAt": "2026-10-10T00:00:00Z", "lastSeenAt": "2026-10-10T00:00:00Z"
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

/// The shipped `iris_3b` catalog entry (the real `coRequisite` declaration, not a test copy).
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots under HF_HUB_CACHE and a CUDA device"]
async fn iris_worker_gpu_smoke() {
    let hf_hub_cache = PathBuf::from(
        std::env::var("HF_HUB_CACHE").expect("set $HF_HUB_CACHE to the real-weights hub"),
    );
    assert!(
        hf_hub_cache.is_dir(),
        "HF_HUB_CACHE is not a directory: {}",
        hf_hub_cache.display()
    );

    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-smoke-worker".to_owned();
    settings.gpu_id = "cuda".to_owned();
    settings.backend_candle_enabled = true;
    settings.heartbeat_seconds = 30;

    for repo in [BACKBONE_REPO, TEXT_ENCODER_REPO] {
        assert!(
            crate::model_jobs::huggingface_snapshot_dir(&settings.data_dir, repo).is_some(),
            "HF_HUB_CACHE must already hold {repo} (no download at generation time)"
        );
    }

    let project = ProjectStore::new(settings.data_dir.clone(), "iris-smoke")
        .create_project("Iris-3B CUDA worker smoke")
        .expect("smoke project creates");
    let mut snapshot = stub_job_snapshot(JOB_ID);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": "a lighthouse on a rocky coast at dusk, warm window light, detailed",
        "negativePrompt": "blurry",
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "seed": 7,
        "advanced": { "steps": STEPS, "guidanceScale": 3.0 },
        "modelManifestEntry": shipped_iris_entry(),
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("smoke job deserializes");

    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::image_jobs::run_image_generate_job(&api, &settings, &job)
        .await
        .expect("the Iris-3B CUDA worker job completes");
    let elapsed = started.elapsed();

    let posts = posts.lock().expect("progress posts lock").clone();
    let step_posts = posts
        .iter()
        .filter(|(_, body)| {
            body.get("message")
                .and_then(Value::as_str)
                .is_some_and(|message| message.contains(&format!("/{STEPS}")))
        })
        .count();
    assert!(
        step_posts > 0,
        "the engine's step progress must reach the job API: {posts:#?}"
    );
    let result = posts
        .iter()
        .rev()
        .find(|(id, body)| id == JOB_ID && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the job posted a completed result");
    let asset = result["assetWrites"]
        .as_array()
        .and_then(|assets| assets.first())
        .expect("the completed result carries one image asset");
    let media_path = asset["mediaPath"]
        .as_str()
        .expect("the asset carries its media path");
    let rendered_path = PathBuf::from(&project.path).join(media_path);
    let decoded = image::open(&rendered_path)
        .unwrap_or_else(|error| panic!("open rendered asset {}: {error}", rendered_path.display()))
        .to_rgb8();
    let rendered = gen_core::Image {
        width: decoded.width(),
        height: decoded.height(),
        pixels: decoded.into_raw(),
    };
    assert_eq!((rendered.width, rendered.height), (SIZE, SIZE));
    let std = crate::smoke_support::image_std(&rendered);
    assert!(
        std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
        "the worker render looks degenerate (RGB std {std:.2})"
    );
    eprintln!(
        "RESULT status=pass model=iris_3b backend=candle geometry={}x{} steps={STEPS} \
         elapsed_s={:.1} rgb_std={std:.3} output={}",
        rendered.width,
        rendered.height,
        elapsed.as_secs_f64(),
        rendered_path.display(),
    );
}
