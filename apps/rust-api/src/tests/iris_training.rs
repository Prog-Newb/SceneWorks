//! Iris-3B generation training through the API (epic sc-25678; S7 sc-25685 MLX, S8 sc-25686
//! Candle): the target's submit-time contract, the pre-launch data walk, resume / load-from
//! resolution, registration of the adapter and of the full-model export (and of an export a
//! cancel published), and generation of a registered full checkpoint through the `iris_3b` engine.
use super::support::*;

const IRIS_TARGET: &str = "iris_3b_generation";

/// Holds the HF-env lock (via [`isolate_hf_cache`]) and pins the MLX unified-memory emulation cap,
/// so the macOS submit-time memory floor judges a deterministic budget on any host.
struct IrisEnv {
    _hf: HfCacheEnvGuard,
    previous_cap: Option<String>,
}

const MLX_CAP_ENV: &str = "SCENEWORKS_MLX_MEMORY_CAP_GB";

fn iris_env(cap_gb: &str) -> IrisEnv {
    let hf = isolate_hf_cache();
    let previous_cap = std::env::var(MLX_CAP_ENV).ok();
    std::env::set_var(MLX_CAP_ENV, cap_gb);
    IrisEnv {
        _hf: hf,
        previous_cap,
    }
}

impl Drop for IrisEnv {
    fn drop(&mut self) {
        match &self.previous_cap {
            Some(previous) => std::env::set_var(MLX_CAP_ENV, previous),
            None => std::env::remove_var(MLX_CAP_ENV),
        }
    }
}

/// The adapter registry accepts the `iris` family once the generation install advertises it (S3,
/// sc-25681, owns that declaration); stand that in so the training guardrail is satisfied.
fn seed_iris_install(settings: &Settings) {
    seed_installed_training_base(&settings.data_dir, "iris_3b");
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir");
    std::fs::write(
        manifests.join("user.models.jsonc"),
        serde_json::to_vec_pretty(&json!({
            "models": [{
                "id": "iris_adapter_host",
                "name": "Iris adapter host",
                "type": "image",
                "family": "iris",
                "loraCompatibility": { "families": ["iris"], "types": ["lora", "lokr"] }
            }]
        }))
        .unwrap(),
    )
    .expect("user models manifest");
}

async fn project(app: &axum::Router) -> String {
    let (_, project) = request(
        app.clone(),
        "POST",
        "/api/v1/projects",
        json!({ "name": "Iris Project" }),
    )
    .await;
    project["id"].as_str().expect("project id").to_owned()
}

async fn dataset(app: &axum::Router, project_id: &str, items: Value) -> String {
    let mut built = Vec::new();
    for (index, item) in items
        .as_array()
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .enumerate()
    {
        let (_, asset) = request_multipart_upload(
            app.clone(),
            &format!("/api/v1/projects/{project_id}/assets"),
            &format!("Item{index}.png"),
            "image/png",
            b"png-bytes",
        )
        .await;
        let mut item = item;
        item["assetId"] = asset["id"].clone();
        built.push(item);
    }
    let (status, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Iris set", "items": built }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{dataset}");
    dataset["id"].as_str().expect("dataset id").to_owned()
}

async fn iris_target(app: &axum::Router) -> Value {
    let (_, registry) = request(app.clone(), "GET", "/api/v1/training/targets", Value::Null).await;
    registry["targets"]
        .as_array()
        .expect("targets")
        .iter()
        .find(|target| target["id"] == IRIS_TARGET)
        .expect("the Iris target is served")
        .clone()
}

async fn submit(
    app: &axum::Router,
    project_id: &str,
    dataset_id: &str,
    config: Value,
    dry_run: bool,
) -> (StatusCode, Value) {
    request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/jobs"),
        json!({
            "targetId": IRIS_TARGET,
            "datasetId": dataset_id,
            "config": config,
            "outputName": "Iris Style",
            "dryRun": dry_run
        }),
    )
    .await
}

fn one_item() -> Value {
    json!([{ "caption": { "text": "a red square" } }])
}

/// Every refusal the trainer would raise after load is a 400 at submit — typed for the advertised
/// limits (the micro-step rule names the field and the multiple).
#[tokio::test]
async fn iris_submit_refuses_what_the_trainer_refuses() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let defaults = target["defaults"].clone();

    // Defaults are runnable as served.
    let (status, job) = submit(&app, &project_id, &dataset_id, defaults.clone(), true).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(job["payload"]["plan"]["target"]["kernel"], "iris_train");
    assert_eq!(
        job["payload"]["plan"]["config"]["advanced"]["iris"]["text_conditioning"],
        "cached"
    );

    // Micro-steps: steps 1000 at accumulation 3 is not a whole number of windows.
    let mut config = defaults.clone();
    config["gradientAccumulation"] = json!(3);
    config["steps"] = json!(999);
    config["saveEvery"] = json!(250);
    let (status, error) = submit(&app, &project_id, &dataset_id, config, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert_eq!(error["code"], "training_target_limit", "{error}");
    assert_eq!(error["context"]["kind"], "not_multiple_of", "{error}");
    assert_eq!(error["context"]["field"], "saveEvery", "{error}");
    assert_eq!(error["context"]["multiple"], 3);

    for (mutate, needle) in [
        (
            json!({"advanced": {"lrWarmupSteps": 1}, "gradientAccumulation": 2, "steps": 1000, "saveEvery": 250}),
            "lrWarmupSteps",
        ),
        (json!({"optimizer": "adamw8bit"}), "optimizer"),
        (
            json!({"advanced": {"lrScheduler": "linear"}}),
            "lrScheduler",
        ),
        (
            json!({"advanced": {"timestepBias": "high_noise"}}),
            "timestepBias",
        ),
        (json!({"advanced": {"lossType": "mae"}}), "lossType"),
        (
            json!({"advanced": {"timestepType": "sigmoid_shifted"}}),
            "timestepType",
        ),
        (
            json!({"advanced": {"mixedPrecision": "fp16"}}),
            "mixedPrecision",
        ),
        (
            json!({"advanced": {"gradientCheckpointing": true}}),
            "gradient checkpointing",
        ),
        (json!({"advanced": {"sampleCount": 5}}), "sampleCount"),
        (
            json!({"advanced": {"iris": {"not_an_option": 1}}}),
            "not_an_option",
        ),
        (
            json!({"advanced": {"iris": {"resume_from": "/tmp/x"}}}),
            "resumeFromJobId",
        ),
        (
            json!({"advanced": {"iris": {"init": "random"}}}),
            "Random initialization",
        ),
        (
            json!({"advanced": {"iris": {"ema_enabled": false, "export_weights": "ema"}}}),
            "ema",
        ),
        (
            json!({"advanced": {"iris": {"text_dropout": 2.0}}}),
            "text_dropout",
        ),
        (
            json!({"advanced": {"iris": {"resume_data_policy": "new_phase"}}}),
            "resuming",
        ),
        (
            json!({"advanced": {"loadFromJobId": "job_x"}}),
            "full fine-tune",
        ),
        (
            json!({"advanced": {"iris": {"upstream_validation_prompts": true}, "samplePrompts": ["mine"]}}),
            "upstream",
        ),
    ] {
        let mut config = defaults.clone();
        for (key, value) in mutate.as_object().unwrap() {
            if key == "advanced" {
                for (inner, value) in value.as_object().unwrap() {
                    if inner == "iris" {
                        for (option, value) in value.as_object().unwrap() {
                            config["advanced"]["iris"][option] = value.clone();
                        }
                    } else {
                        config["advanced"][inner] = value.clone();
                    }
                }
            } else {
                config[key] = value.clone();
            }
        }
        let (status, error) = submit(&app, &project_id, &dataset_id, config, true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{mutate} → {error}");
        assert!(
            error.to_string().contains(needle),
            "{mutate}: the refusal names {needle:?}: {error}"
        );
    }

    // A caption field no item carries is refused before launch.
    let mut config = defaults.clone();
    config["advanced"]["iris"]["caption_field"] = json!("short");
    let (status, error) = submit(&app, &project_id, &dataset_id, config, true).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(error.to_string().contains("short"), "{error}");
}

/// The trainer's unused tail is reported before launch, from the trainer's own walk.
#[tokio::test]
async fn data_walk_reports_the_unused_tail_before_launch() {
    let temp_dir = tempfile::tempdir().unwrap();
    let app = create_app(test_settings(&temp_dir)).unwrap();
    let (status, walk) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/training/data-walk?targetId={IRIS_TARGET}&items=1000&batchSize=4"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(walk["applies"], true);
    assert_eq!(walk["covered"], 640);
    assert_eq!(walk["unusedItems"], 360);
    assert!(walk["warning"]
        .as_str()
        .is_some_and(|text| text.contains("360 of 1000")));

    let (_, whole) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/training/data-walk?targetId={IRIS_TARGET}&items=640"),
        Value::Null,
    )
    .await;
    assert_eq!(whole["unusedItems"], 0);
    assert!(whole["warning"].is_null());

    let (_, other) = request(
        app,
        "GET",
        "/api/v1/training/data-walk?targetId=z_image_turbo_lora&items=1000",
        Value::Null,
    )
    .await;
    assert_eq!(other["applies"], false);
}

async fn complete(app: &axum::Router, job_id: &str, status: &str, result: Value) -> Value {
    let (code, body) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{job_id}/progress"),
        json!({
            "status": status,
            "stage": status,
            "progress": 1,
            "message": "done",
            "workerId": TEST_TRAINING_WORKER_ID,
            "result": result
        }),
    )
    .await;
    assert_eq!(code, StatusCode::OK, "{body}");
    body
}

/// A LoRA run's adapter registers as an Iris generation adapter with its task identity; per-item
/// named captions reach the plan; a later run resumes the first run's newest checkpoint.
#[tokio::test]
async fn iris_adapter_registers_with_task_identity_and_resumes() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(
        &app,
        &project_id,
        json!([{ "caption": { "text": "a red square" }, "captions": { "short": "red" } }]),
    )
    .await;
    let target = iris_target(&app).await;
    let mut config = target["defaults"].clone();
    config["triggerWord"] = json!("irisstyle");
    config["advanced"]["iris"]["caption_fields"] = json!(["short"]);
    let (status, job) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let plan = &job["payload"]["plan"];
    assert_eq!(
        plan["dataset"]["items"][0]["captions"],
        json!({ "short": "red" }),
        "named captions ride on the plan item"
    );
    assert_eq!(job["payload"]["manifestEntry"]["irisTask"], "generation");
    let job_id = job["id"].as_str().unwrap().to_owned();
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();

    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&output_dir).unwrap();
    write_test_safetensors(&output_dir.join(&file_name));
    let stem = file_name.trim_end_matches(".safetensors");
    let checkpoint = output_dir.join(format!("{stem}.checkpoints/step_00000250"));
    std::fs::create_dir_all(&checkpoint).unwrap();
    std::fs::write(checkpoint.join("state.json"), b"{}").unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(completed["result"]["loraRegistered"], true, "{completed}");

    let (_, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    let entry = loras
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "Iris Style")
        .expect("the adapter is registered")
        .clone();
    assert_eq!(entry["family"], "iris");
    assert_eq!(entry["baseModel"], "iris_3b");
    assert_eq!(entry["irisTask"], "generation");
    assert_eq!(entry["networkType"], "lora");
    assert_eq!(entry["installState"], "installed");

    // Resume: the second run names the first; SceneWorks hands the trainer its checkpoint root.
    let mut resume = config.clone();
    resume["advanced"]["resumeFromJobId"] = json!(job_id);
    resume["advanced"]["iris"]["resume_data_policy"] = json!("new_phase");
    let (status, resumed) = submit(&app, &project_id, &dataset_id, resume, false).await;
    assert_eq!(status, StatusCode::CREATED, "{resumed}");
    let advanced = &resumed["payload"]["plan"]["config"]["advanced"];
    assert_eq!(advanced["resume"], true);
    assert_eq!(
        advanced["iris"]["resume_from"],
        json!(output_dir
            .join(format!("{stem}.checkpoints"))
            .display()
            .to_string())
    );
    assert_eq!(advanced["iris"]["resume_data_policy"], "new_phase");

    // A run with no checkpoint cannot be resumed; a different artifact kind cannot either.
    std::fs::remove_dir_all(output_dir.join(format!("{stem}.checkpoints"))).unwrap();
    let mut resume = config.clone();
    resume["advanced"]["resumeFromJobId"] = json!(job_id);
    let (status, error) = submit(&app, &project_id, &dataset_id, resume, false).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
    assert!(
        error.to_string().contains("no complete checkpoint"),
        "{error}"
    );
}

/// A full run's export registers as a selectable user image model with its Iris task identity, and
/// generating it runs the `iris_3b` engine on that export with the generation install's encoder.
#[tokio::test]
async fn iris_full_export_registers_and_generates_through_the_iris_engine() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let mut config = target["defaults"].clone();
    config["triggerWord"] = json!("irisstyle");
    config["advanced"]["networkType"] = json!("full");
    config["advanced"]["iris"]["init"] = json!("random");
    let (status, job) = submit(&app, &project_id, &dataset_id, config, false).await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let plan = &job["payload"]["plan"];
    assert_eq!(plan["target"]["outputKind"], "base_checkpoint");
    let job_id = job["id"].as_str().unwrap().to_owned();
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    assert!(output_dir.starts_with(settings.data_dir.join("models").join("finetunes")));
    let stem = plan["output"]["fileName"]
        .as_str()
        .unwrap()
        .trim_end_matches(".safetensors")
        .to_owned();
    let export = output_dir.join(&stem);

    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&export).unwrap();
    std::fs::write(export.join("config.yaml"), b"model: {}\n").unwrap();
    write_test_safetensors(&export.join("model.safetensors"));
    // The trainer's resumable checkpoints sit beside the export.
    let checkpoints = output_dir.join(format!("{stem}.checkpoints"));
    std::fs::create_dir_all(checkpoints.join("step_00000001")).unwrap();
    std::fs::write(checkpoints.join("step_00000001/state.json"), b"{}").unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(
        completed["result"]["baseCheckpointRegistered"], true,
        "{completed}"
    );
    assert!(completed["result"].get("loraRegistered").is_none());
    let model_id = completed["result"]["baseCheckpointModelId"]
        .as_str()
        .unwrap()
        .to_owned();

    let (_, models) = request(app.clone(), "GET", "/api/v1/models", Value::Null).await;
    let entry = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the export is a catalog model")
        .clone();
    assert_eq!(entry["type"], "image");
    assert_eq!(entry["family"], "iris");
    assert_eq!(entry["irisTask"], "generation");
    assert_eq!(entry["trainedCheckpoint"]["engineModel"], "iris_3b");
    assert_eq!(entry["trainedCheckpoint"]["task"], "generation");
    assert_eq!(entry["installState"], "installed");
    assert_eq!(entry["adapter"], "mlx_iris");
    assert!(entry["capabilities"]
        .as_array()
        .unwrap()
        .contains(&json!("text_to_image")));
    assert_eq!(entry["paths"]["model"], json!(export.display().to_string()));
    #[cfg(target_os = "macos")]
    assert_eq!(
        entry["macSupport"]["supported"], true,
        "the export renders through iris_3b's MLX lane: {}",
        entry["macSupport"]
    );

    // The stored entry keeps the identity only; the generation surface is the engine's, read at
    // catalog time — so a later change to the `iris_3b` entry reaches an already-registered export.
    let manifests = settings.config_dir.join("manifests");
    let stored: Value = serde_json::from_str(&crate::strip_jsonc_comments(
        &std::fs::read_to_string(manifests.join("user.models.jsonc")).unwrap(),
    ))
    .unwrap();
    let stored = stored["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the export is stored")
        .clone();
    assert_eq!(stored["trainedCheckpoint"]["engineModel"], "iris_3b");
    for key in [
        "loraCompatibility",
        "limits",
        "defaults",
        "capabilities",
        "adapter",
    ] {
        assert!(
            stored.get(key).is_none(),
            "{key} is not snapshotted: {stored:#}"
        );
    }
    let mut builtin: Value = serde_json::from_str(&crate::strip_jsonc_comments(include_str!(
        "../../../../config/manifests/builtin.models.jsonc"
    )))
    .unwrap();
    let engine = builtin["models"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|model| model["id"] == "iris_3b")
        .expect("the builtin iris_3b entry");
    engine["loraCompatibility"] =
        json!({ "families": ["iris", "later_family"], "types": ["lora"] });
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        serde_json::to_vec_pretty(&builtin).unwrap(),
    )
    .unwrap();
    let later_app = create_app(settings.clone()).unwrap();
    let (_, models) = request(later_app, "GET", "/api/v1/models", Value::Null).await;
    let later = models
        .as_array()
        .unwrap()
        .iter()
        .find(|model| model["id"] == json!(model_id))
        .expect("the export is still a catalog model")
        .clone();
    assert_eq!(
        later["loraCompatibility"]["families"],
        json!(["iris", "later_family"]),
        "{later:#}"
    );

    // Generate it: the job runs `iris_3b` on the export, with the generation install's entry (its
    // text-encoder co-requisite) and the user's pick kept for provenance.
    let (status, generation) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": model_id,
            "mode": "text_to_image",
            "prompt": "a lighthouse",
            "width": 512,
            "height": 512,
            "count": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{generation}");
    let payload = &generation["payload"];
    assert_eq!(payload["model"], "iris_3b");
    assert_eq!(payload["checkpointModelId"], json!(model_id));
    let worker_entry = &payload["modelManifestEntry"];
    assert_eq!(worker_entry["id"], "iris_3b");
    assert_eq!(
        worker_entry["modelPath"],
        json!(export.canonicalize().unwrap().display().to_string())
    );
    assert_eq!(
        worker_entry["trainedCheckpoint"]["modelId"],
        json!(model_id)
    );
    assert!(
        worker_entry["downloads"]
            .as_array()
            .unwrap()
            .iter()
            .any(|download| download["componentId"] == "text_encoder"
                && download["coRequisite"] == true),
        "the generation install's text_encoder co-requisite still applies: {worker_entry}"
    );

    // A duplicate that picks ANOTHER model is not re-aliased onto the recorded checkpoint.
    let generation_id = generation["id"].as_str().unwrap().to_owned();
    let (status, duplicate) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{generation_id}/duplicate"),
        json!({ "payloadChanges": { "model": "sdxl" } }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{duplicate}");
    let payload = &duplicate["payload"];
    assert_eq!(payload["model"], "sdxl", "{payload:#}");
    assert!(payload["checkpointModelId"].is_null(), "{payload:#}");
    assert_eq!(payload["modelManifestEntry"]["id"], "sdxl", "{payload:#}");
    // A plain duplicate of the aliased job still re-derives the checkpoint alias.
    let (status, replay) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/jobs/{generation_id}/duplicate"),
        json!({ "payloadChanges": {} }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{replay}");
    assert_eq!(replay["payload"]["checkpointModelId"], json!(model_id));
    assert_eq!(
        replay["payload"]["modelManifestEntry"]["trainedCheckpoint"]["modelId"],
        json!(model_id)
    );

    // A client-authored `checkpointModelId` naming something that is not a trained checkpoint of
    // `model` is dropped: `model`'s own manifest is what resolves.
    let (status, spoofed) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": "iris_3b",
            "checkpointModelId": "iris_adapter_host",
            "mode": "text_to_image",
            "prompt": "a lighthouse",
            "width": 512,
            "height": 512,
            "count": 1
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{spoofed}");
    let payload = &spoofed["payload"];
    assert_eq!(payload["model"], "iris_3b");
    assert!(payload.get("checkpointModelId").is_none(), "{payload:#}");
    assert_eq!(
        payload["modelManifestEntry"]["id"], "iris_3b",
        "{payload:#}"
    );
    assert!(
        payload["modelManifestEntry"]
            .get("trainedCheckpoint")
            .is_none(),
        "{payload:#}"
    );

    // Model Manager delete removes the whole run directory — the export AND its checkpoints.
    let (status, deleted) = request(
        app.clone(),
        "DELETE",
        &format!("/api/v1/models/{model_id}?permanent=true"),
        Value::Null,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{deleted}");
    assert!(
        !checkpoints.exists(),
        "the run's checkpoints are not orphaned: {}",
        checkpoints.display()
    );
    assert!(!output_dir.exists(), "{}", output_dir.display());
}

/// A cancel after the first optimizer step published an export: it registers; a cancel that
/// published nothing registers nothing.
#[tokio::test]
async fn a_canceled_run_registers_only_what_the_trainer_exported() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    seed_iris_install(&settings);
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let mut config = target["defaults"].clone();
    config["triggerWord"] = json!("irisstyle");

    // Nothing exported.
    let (_, job) = submit(&app, &project_id, &dataset_id, config.clone(), false).await;
    let job_id = job["id"].as_str().unwrap().to_owned();
    claim_training_job_as_mlx_worker(&app, &job_id).await;
    let canceled = complete(&app, &job_id, "canceled", Value::Null).await;
    assert!(
        canceled["result"].get("loraRegistered").is_none(),
        "{canceled}"
    );

    // Exported on cancel.
    let (_, job) = submit(&app, &project_id, &dataset_id, config, false).await;
    let job_id = job["id"].as_str().unwrap().to_owned();
    let plan = &job["payload"]["plan"];
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();
    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&output_dir).unwrap();
    write_test_safetensors(&output_dir.join(&file_name));
    let canceled = complete(
        &app,
        &job_id,
        "canceled",
        json!({ "trainedArtifactExported": true, "stepsCompleted": 2 }),
    )
    .await;
    assert_eq!(canceled["status"], "canceled");
    assert_eq!(canceled["result"]["loraRegistered"], true, "{canceled}");
}

/// HF-style Parquet imports can bring named caption columns; malformed names are refused.
#[tokio::test]
async fn parquet_import_forwards_caption_field_columns() {
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    let (_, dataset) = request(
        app.clone(),
        "POST",
        &format!("/api/v1/projects/{project_id}/training/datasets"),
        json!({ "name": "Empty", "items": [] }),
    )
    .await;
    let dataset_id = dataset["id"].as_str().unwrap().to_owned();
    let source = temp_dir.path().join("laion.parquet");
    std::fs::write(&source, b"PAR1").unwrap();
    let path =
        format!("/api/v1/projects/{project_id}/training/datasets/{dataset_id}/parquet-import-jobs");
    let (status, job) = request(
        app.clone(),
        "POST",
        &path,
        json!({
            "sourcePath": source.display().to_string(),
            "captionFieldColumns": ["short_caption", "tags"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    assert_eq!(
        job["payload"]["captionFieldColumns"],
        json!(["short_caption", "tags"])
    );
    let (status, error) = request(
        app,
        "POST",
        &path,
        json!({
            "sourcePath": source.display().to_string(),
            "captionFieldColumns": [" padded"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
}

/// The S7 adapter (Training Studio) meets the S3 adapter gate (sc-25681): a registered Iris LoRA
/// whose file carries the engine's stamp (`family=iris`, `irisTask=generation`,
/// `baseModel=iris_3b`) is accepted on an `iris_3b` image job — the shipped entry's
/// `loraCompatibility` (families `iris`, `irisTask` generation) admits it end to end.
#[tokio::test]
async fn a_trained_iris_adapter_is_accepted_by_the_iris_image_job_gate() {
    let _env = iris_env("256");
    let temp_dir = tempfile::tempdir().unwrap();
    let settings = test_settings(&temp_dir);
    // The catalog is the SHIPPED `iris_3b` row alone — no stand-in adapter host: its
    // `loraCompatibility` must satisfy both the training family guardrail and the image-job gate.
    seed_installed_training_base(&settings.data_dir, "iris_3b");
    let manifests = settings.config_dir.join("manifests");
    std::fs::create_dir_all(&manifests).expect("manifest dir");
    let shipped: Value = serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(
        sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1,
    ))
    .expect("builtin.models.jsonc parses");
    let iris = shipped["models"]
        .as_array()
        .expect("models")
        .iter()
        .find(|model| model["id"] == "iris_3b")
        .expect("iris_3b is shipped")
        .clone();
    std::fs::write(
        manifests.join("builtin.models.jsonc"),
        json!({ "schemaVersion": 1, "models": [iris] }).to_string(),
    )
    .expect("builtin models writes");
    let app = create_app(settings.clone()).unwrap();
    let project_id = project(&app).await;
    let dataset_id = dataset(&app, &project_id, one_item()).await;
    let target = iris_target(&app).await;
    let (status, job) = submit(
        &app,
        &project_id,
        &dataset_id,
        target["defaults"].clone(),
        false,
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{job}");
    let job_id = job["id"].as_str().unwrap().to_owned();
    let plan = &job["payload"]["plan"];
    let output_dir = std::path::PathBuf::from(plan["output"]["outputDir"].as_str().unwrap());
    let file_name = plan["output"]["fileName"].as_str().unwrap().to_owned();

    claim_training_job_as_mlx_worker(&app, &job_id).await;
    std::fs::create_dir_all(&output_dir).unwrap();
    // What the Iris trainer writes: an adapter whose safetensors metadata names its task identity.
    let header = json!({
        "__metadata__": { "family": "iris", "irisTask": "generation", "baseModel": "iris_3b" },
        "blocks.0.attn.qkv.lora_down.weight": { "dtype": "F32", "shape": [1], "data_offsets": [0, 4] }
    })
    .to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&[0; 4]);
    std::fs::write(output_dir.join(&file_name), bytes).unwrap();
    let completed = complete(&app, &job_id, "completed", json!({})).await;
    assert_eq!(completed["result"]["loraRegistered"], true, "{completed}");

    let (_, loras) = request(
        app.clone(),
        "GET",
        &format!("/api/v1/loras?projectId={project_id}"),
        Value::Null,
    )
    .await;
    let lora_id = loras
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "Iris Style")
        .expect("the adapter is registered")["id"]
        .as_str()
        .unwrap()
        .to_owned();

    let (status, generation) = request(
        app.clone(),
        "POST",
        "/api/v1/image/jobs",
        json!({
            "projectId": project_id,
            "model": "iris_3b",
            "mode": "text_to_image",
            "prompt": "a lighthouse",
            "width": 512,
            "height": 512,
            "count": 1,
            "loras": [{ "id": lora_id, "weight": 0.8 }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{generation}");
    assert!(
        generation["payload"]["loras"]
            .as_array()
            .is_some_and(|loras| loras.iter().any(|lora| lora["id"] == json!(lora_id))),
        "{generation:#}"
    );
}
