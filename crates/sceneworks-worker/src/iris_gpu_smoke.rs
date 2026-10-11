//! Real-weight worker acceptance for Iris-3B on Candle/CUDA (sc-25680, epic 25678).
//!
//! Drives ONE production `image_generate` job through `run_image_generate_job` — the shipped
//! `iris_3b` catalog entry, cache-only resolution of the backbone AND its `text_encoder`
//! co-requisite through the generic component seam, the cached `candle-gen-iris` load, the per-item
//! drive and `consume_gen_events` — and asserts the job completes with one non-degenerate image
//! asset at the requested geometry. sc-25681 runs it with NON-default controls (Euler / order-1
//! solver, a schedule shift, a CFG interval, the `warn` caption-overflow policy on a prompt longer
//! than the caption budget) and a tiny random-init generation-task LoRA stamped `family=iris`,
//! `irisTask=generation`, `baseModel=iris_3b` (written here, PEFT layout on three real backbone
//! projections), and asserts the live preview, the `caption_truncated` warning and the adapter's
//! applied report reach the job and the asset. Nothing is downloaded: `HF_HUB_CACHE` must already hold both
//! pinned snapshots (the inference `candle-iris` real-weight profile materializes them into the same
//! `E:\huggingface\hub` on the real-weights host). An absent snapshot is a failure, never a skip.
//!
//! ```text
//! $env:HF_HUB_CACHE = "E:\huggingface\hub"
//! cargo test -p sceneworks-worker --features backend-candle --release iris_worker_gpu_smoke -- --ignored --nocapture --test-threads=1
//! ```

// The single-image + LoRA smoke is CUDA-only; on macOS only the native-batch smoke compiles.
#![cfg_attr(target_os = "macos", allow(dead_code))]

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
const LORA_RANK: usize = 4;

/// Write a tiny random-init Iris generation-task LoRA (PEFT `lora_A`/`lora_B`, bare keys) on three
/// real backbone projections, with the identity stamps the engine requires. Small magnitudes: it
/// probes the adapter path on real weights, it is not a style.
#[cfg(not(target_os = "macos"))]
fn write_stamped_smoke_lora(path: &std::path::Path) {
    let targets: [(&str, usize, usize); 3] = [
        ("blocks.23.attn_proj", 2560, 2560),
        ("blocks.23.q_proj", 2560, 2560),
        ("blocks.22.mlp.w2", 2560, 6826),
    ];
    let mut state: u64 = 25_681;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.02
    };
    let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (stem, out, inp) in targets {
        for (suffix, shape) in [("lora_A", [LORA_RANK, inp]), ("lora_B", [out, LORA_RANK])] {
            let data: Vec<u8> = (0..shape[0] * shape[1])
                .flat_map(|_| next().to_le_bytes())
                .collect();
            tensors.push((format!("{stem}.{suffix}.weight"), shape.to_vec(), data));
        }
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    header.insert(
        "__metadata__".to_owned(),
        json!({
            "family": "iris", "irisTask": "generation", "baseModel": "iris_3b",
            "networkType": "lora", "rank": LORA_RANK.to_string(), "alpha": LORA_RANK.to_string(),
        }),
    );
    let mut offset = 0usize;
    for (name, shape, data) in &tensors {
        header.insert(
            name.clone(),
            json!({ "dtype": "F32", "shape": shape, "data_offsets": [offset, offset + data.len()] }),
        );
        offset += data.len();
    }
    let mut raw = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    while !raw.len().is_multiple_of(8) {
        raw.push(b' ');
    }
    let mut bytes = (raw.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&raw);
    for (_, _, data) in &tensors {
        bytes.extend_from_slice(data);
    }
    std::fs::create_dir_all(path.parent().expect("lora dir")).expect("lora dir creates");
    std::fs::write(path, bytes).expect("smoke LoRA writes");
}

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

#[cfg(not(target_os = "macos"))]
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
    let lora_path = settings
        .data_dir
        .join("loras")
        .join("iris-smoke-style")
        .join("adapter.safetensors");
    write_stamped_smoke_lora(&lora_path);
    // A prompt well past the 300-token caption budget, so the `warn` policy truncates and reports.
    let prompt = std::iter::repeat_n(
        "a lighthouse on a rocky coast at dusk, warm window light, crashing waves, gulls, weathered stone, glowing lantern, dramatic clouds, detailed",
        24,
    )
    .collect::<Vec<_>>()
    .join(", ");
    let mut snapshot = stub_job_snapshot(JOB_ID);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": prompt,
        "negativePrompt": "blurry",
        "width": SIZE,
        "height": SIZE,
        "count": 1,
        "seed": 7,
        "loras": [{ "id": "iris-smoke-style", "path": lora_path, "weight": 0.8 }],
        "advanced": {
            "steps": STEPS,
            "guidanceScale": 4.0,
            "sampler": "euler",
            "schedulerShift": 3.0,
            "cfgInterval": [0.05, 0.95],
            "captionOverflow": "warn",
        },
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
    assert!(
        posts
            .iter()
            .any(|(_, body)| body["result"]["previewFrame"]["dataUrl"]
                .as_str()
                .is_some_and(|url| url.starts_with("data:image/"))),
        "the engine's per-step x0 preview must reach the job API"
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
    let settings_block = &asset["rawAdapterSettings"];
    assert!(
        settings_block["generationWarnings"]
            .as_array()
            .is_some_and(|warnings| warnings
                .iter()
                .any(|warning| warning["code"] == "caption_truncated")),
        "the over-long prompt's caption_truncated warning is on the asset: {settings_block}"
    );
    let applied = &settings_block["adapterApply"][0];
    assert!(
        applied["applied"].as_u64().is_some_and(|count| count > 0)
            && applied["sha256"].as_str().map(str::len) == Some(64),
        "the stamped LoRA's apply report + identity are on the asset: {settings_block}"
    );
    for (key, value) in [
        ("sampler", json!("euler")),
        ("cfgInterval", json!([0.05, 0.95])),
        ("captionOverflow", json!("warn")),
    ] {
        assert_eq!(settings_block[key], value, "{key} recorded with the image");
    }
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
         sampler=euler shift=3.0 cfg_interval=0.05..0.95 lora_applied={} \
         elapsed_s={:.1} rgb_std={std:.3} output={}",
        rendered.width,
        rendered.height,
        applied["applied"],
        elapsed.as_secs_f64(),
        rendered_path.display(),
    );
}

/// sc-25681: the NATIVE prompt batch on real weights — a 2-prompt × count-1 job is one
/// `candle-gen-iris` call (`prompt_batch`), published as two non-degenerate assets in prompt order
/// on seeds `seed` and `seed + 1`, each recording its own prompt and the batch. Runs in the same
/// `iris_worker_gpu_smoke` filter as the single-image smoke above on CUDA, and as the bounded MLX
/// real-weight check on a Mac (768², 20 steps).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "real-weight worker GPU smoke; needs the pinned speridlabs/iris-3b and Qwen3-VL-4B-Instruct snapshots in the HF cache and a CUDA device or Apple GPU"]
async fn iris_worker_gpu_smoke_native_prompt_batch() {
    const BATCH_JOB: &str = "iris-cuda-native-batch";
    const BATCH_STEPS: u32 = 20;
    const BATCH_SIZE: u32 = 768;
    let prompts = [
        "a red fox in fresh snow, morning light",
        "a blue whale breaching at sunset",
    ];
    let temp = tempfile::tempdir().expect("smoke tempdir creates");
    let (api_url, posts) = spawn_worker_api_stub().await;
    let mut settings = Settings::for_test(temp.path().join("data"));
    settings.api_url = api_url;
    settings.config_dir = temp.path().join("config");
    settings.worker_id = "iris-smoke-worker".to_owned();
    #[cfg(not(target_os = "macos"))]
    {
        settings.gpu_id = "cuda".to_owned();
        settings.backend_candle_enabled = true;
    }
    settings.heartbeat_seconds = 30;
    let project = ProjectStore::new(settings.data_dir.clone(), "iris-smoke")
        .create_project("Iris-3B native batch smoke")
        .expect("smoke project creates");
    let mut snapshot = stub_job_snapshot(BATCH_JOB);
    snapshot["payload"] = json!({
        "projectId": project.id,
        "model": "iris_3b",
        "mode": "text_to_image",
        "prompt": prompts[0],
        "prompts": prompts,
        "width": BATCH_SIZE,
        "height": BATCH_SIZE,
        "count": 1,
        "seed": 11,
        "advanced": { "steps": BATCH_STEPS },
        "modelManifestEntry": shipped_iris_entry(),
    });
    let job: JobSnapshot = serde_json::from_value(snapshot).expect("smoke job deserializes");
    let api = ApiClient::new(&settings);
    let started = std::time::Instant::now();
    crate::image_jobs::run_image_generate_job(&api, &settings, &job)
        .await
        .expect("the Iris-3B native batch job completes");
    let elapsed = started.elapsed();
    let posts = posts.lock().expect("progress posts lock").clone();
    let result = posts
        .iter()
        .rev()
        .find(|(id, body)| id == BATCH_JOB && body["status"] == "completed")
        .map(|(_, body)| body["result"].clone())
        .expect("the job posted a completed result");
    let assets = result["assetWrites"].as_array().expect("asset writes");
    assert_eq!(assets.len(), 2, "one asset per prompt: {result}");
    let mut stds = Vec::new();
    for (index, asset) in assets.iter().enumerate() {
        assert_eq!(asset["prompt"], prompts[index]);
        assert_eq!(asset["seed"], 11 + index as i64);
        assert_eq!(asset["rawAdapterSettings"]["promptBatch"], json!(prompts));
        let path = PathBuf::from(&project.path).join(asset["mediaPath"].as_str().expect("media"));
        let decoded = image::open(&path).expect("open rendered asset").to_rgb8();
        assert_eq!(
            (decoded.width(), decoded.height()),
            (BATCH_SIZE, BATCH_SIZE)
        );
        let std = crate::smoke_support::image_std(&gen_core::Image {
            width: decoded.width(),
            height: decoded.height(),
            pixels: decoded.into_raw(),
        });
        assert!(
            std > crate::smoke_support::DEGENERATE_STD_FLOOR_DEFAULT,
            "batch image {index} looks degenerate (RGB std {std:.2})"
        );
        stds.push(std);
    }
    eprintln!(
        "RESULT status=pass model=iris_3b native_batch=2x1 geometry={BATCH_SIZE}x{BATCH_SIZE} \
         steps={BATCH_STEPS} elapsed_s={:.1} rgb_std={stds:?}",
        elapsed.as_secs_f64(),
    );
}
