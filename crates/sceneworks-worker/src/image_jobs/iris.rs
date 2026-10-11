//! Iris-3B generation-task wiring (sc-25679, epic 25678).
//!
//! The `iris_3b` engine loads the GENERATION task from two resources: `LoadSpec::weights` = the
//! `speridlabs/iris-3b` backbone root (`config.yaml` + `model.safetensors`) and
//! `LoadSpec::components["text_encoder"]` = the pinned `Qwen/Qwen3-VL-4B-Instruct` snapshot. The
//! engine's descriptor declares `required_components: [TEXT_ENCODER_COMPONENT]`, so the generic
//! `attach_required_components` seam stages the encoder cache-only through the catalog's
//! all-or-nothing `coRequisite` (`componentId: "text_encoder"`): a job never downloads at
//! generation time, and a missing or torn encoder is a typed failure before load (pinned by the
//! tests below against the shipped catalog entry).
//!
//! The Iris request controls (sampler = solver order, schedule shift, CFG interval, caption-overflow
//! policy, CFG-bound negative prompt, adapters, previews; sc-25681) ride the GENERIC image lane:
//! [`super::lane_controls`] resolves and refuses them for every engine against its linked
//! descriptor, so nothing here is Iris-only production code. This module pins the Iris job arm end
//! to end on both lanes with a recording stub engine.

/// The Iris text-encoder directory a resolved spec carries, for diagnostics/tests.
#[cfg(test)]
pub(crate) fn staged_text_encoder_dir(spec: &gen_core::LoadSpec) -> Option<&std::path::Path> {
    match spec.components.get(gen_core::iris::TEXT_ENCODER_COMPONENT) {
        Some(gen_core::WeightsSource::Dir(dir)) => Some(dir.as_path()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::WorkerResult;
    use crate::settings::Settings;
    use gen_core::iris::{IrisTask, TEXT_ENCODER_COMPONENT};
    use gen_core::{LoadSpec, WeightsSource};
    use sceneworks_core::contracts::JsonObject;
    use serde_json::json;
    use serde_json::Value;
    use std::path::Path;

    const TE_REPO: &str = "Qwen/Qwen3-VL-4B-Instruct";
    const TE_REVISION: &str = "ebb281ec70b05090aa6165b016eac8ec08e71b17";

    fn settings_at(data_dir: &Path) -> Settings {
        let mut settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        settings.data_dir = data_dir.to_path_buf();
        settings
    }

    /// Neutralize the ambient HF cache env so the cache resolves under the temp data dir.
    fn isolate_hf_cache() -> crate::test_env::EnvVars {
        crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
        ])
    }

    /// The shipped catalog entry, so the test exercises the real `coRequisite` declaration.
    fn shipped_iris_entry() -> JsonObject {
        let text = sceneworks_core::builtin_manifests::BUILTIN_MANIFESTS
            .iter()
            .find(|(name, _)| *name == "builtin.models.jsonc")
            .expect("builtin.models.jsonc embedded")
            .1;
        let manifest: Value =
            serde_json::from_str(&sceneworks_core::jsonc::strip_jsonc_comments(text))
                .expect("parse builtin.models.jsonc");
        manifest
            .get("models")
            .and_then(Value::as_array)
            .and_then(|models| {
                models
                    .iter()
                    .find(|model| model.get("id").and_then(Value::as_str) == Some("iris_3b"))
            })
            .and_then(Value::as_object)
            .cloned()
            .expect("iris_3b is in the builtin catalog")
    }

    fn stage(root: &Path, repo: &str, revision: &str, file: &str) {
        let snapshot = sceneworks_core::hf_home::huggingface_repo_cache_path(root, repo)
            .expect("repo cache path resolves")
            .join("snapshots")
            .join(revision);
        let path = snapshot.join(file);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"x").unwrap();
    }

    fn stage_text_encoder(root: &Path) {
        for file in [
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
        ] {
            stage(root, TE_REPO, TE_REVISION, file);
        }
    }

    #[test]
    fn the_contract_names_only_the_text_encoder_for_generation() {
        let descriptor = crate::inference_runtime::media_descriptor("iris_3b")
            .expect("iris_3b is linked on macOS");
        assert_eq!(descriptor.required_components, &[TEXT_ENCODER_COMPONENT]);
        assert_eq!(TEXT_ENCODER_COMPONENT, "text_encoder");
        assert!(IrisTask::Generation.uses_text_encoder());
        assert!(!IrisTask::Depth.uses_text_encoder());
        assert!(!IrisTask::Restoration.uses_text_encoder());
    }

    #[test]
    fn generation_stages_the_installed_text_encoder_snapshot() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        stage_text_encoder(data.path());
        let settings = settings_at(data.path());
        let backbone = data.path().join("backbone");
        let spec = super::super::attach_required_components(
            LoadSpec::new(WeightsSource::Dir(backbone.clone())),
            "iris_3b",
            &shipped_iris_entry(),
            &settings,
        )
        .expect("an installed encoder stages");
        let staged = staged_text_encoder_dir(&spec).expect("text_encoder staged as a Dir");
        assert!(
            staged.ends_with(format!("snapshots/{TE_REVISION}")),
            "{staged:?}"
        );
        assert_eq!(
            spec.components.len(),
            1,
            "only the generation task's encoder is staged"
        );
        assert_eq!(spec.weights, WeightsSource::Dir(backbone));
    }

    #[test]
    fn a_missing_text_encoder_is_a_typed_failure_before_load() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        let settings = settings_at(data.path());
        let error = super::super::attach_required_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
            "iris_3b",
            &shipped_iris_entry(),
            &settings,
        )
        .expect_err("no encoder installed");
        let message = error.to_string();
        assert!(
            message.contains("text_encoder") && message.contains("not installed"),
            "{message}"
        );
    }

    #[test]
    fn a_torn_text_encoder_is_a_typed_failure_before_load() {
        let _env = isolate_hf_cache();
        let data = tempfile::tempdir().unwrap();
        // Everything but the shards.
        for file in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            stage(data.path(), TE_REPO, TE_REVISION, file);
        }
        let error = super::super::attach_required_components(
            LoadSpec::new(WeightsSource::Dir(data.path().join("backbone"))),
            "iris_3b",
            &shipped_iris_entry(),
            &settings_at(data.path()),
        )
        .expect_err("torn encoder");
        assert!(error.to_string().contains("missing"), "{error}");
    }

    /// Run `body` under an INFO subscriber whose output is captured, so a test can read the worker
    /// events it emitted. Local (not `ladder_e2e_sc18101`'s macOS-only twin) because these tests run
    /// on both lanes.
    fn with_captured_tracing<T>(body: impl FnOnce() -> T) -> (T, String) {
        #[derive(Clone, Default)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        crate::test_env::install_tracing_interest_floor();
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_target(false)
            .without_time()
            .finish();
        let value = tracing::subscriber::with_default(subscriber, body);
        let text = String::from_utf8_lossy(&capture.0.lock().unwrap()).into_owned();
        (value, text)
    }

    /// The catalog row's honored-controls surface is EXACTLY the linked engine's (sc-25681): the
    /// sampler menu is the descriptor's solver menu, there is no scheduler menu but a declared shift
    /// axis, the CFG interval / caption-overflow declarations mirror the descriptor's capability
    /// flags, adapters are offered iff the engine applies them (generation-task stamps only), and
    /// there is no quant tier because the engine refuses quantization.
    ///
    /// *Mutation that reds this:* drop `euler` from `limits.samplers`, or `supportsCfgInterval`.
    #[test]
    fn the_catalog_row_offers_exactly_the_linked_engines_controls() {
        let model = crate::engines::mlx_model("iris_3b").expect("iris_3b is linked");
        assert_eq!(model.engine_id(), "iris_3b");
        assert!(!model.supports_quant());
        assert!(model.supports_guidance() && model.supports_negative_prompt());
        let caps = &model.descriptor.capabilities;
        assert_eq!(model.default_steps(), gen_core::iris::DEFAULT_STEPS);
        assert_eq!(model.default_guidance(), gen_core::iris::DEFAULT_CFG_SCALE);

        let entry = shipped_iris_entry();
        let limits = entry.get("limits").and_then(Value::as_object).unwrap();
        let samplers: Vec<&str> = limits["samplers"]
            .as_array()
            .expect("iris declares its solver menu")
            .iter()
            .filter_map(Value::as_str)
            .collect();
        assert_eq!(samplers, caps.samplers, "the menu IS the engine's");
        assert_eq!(samplers, gen_core::iris::SAMPLERS);
        assert_eq!(
            entry["defaults"]["sampler"].as_str(),
            Some(gen_core::iris::SOLVER_ORDER_2_SAMPLER)
        );
        assert!(caps.schedulers.is_empty());
        for absent in ["schedulers", "guidanceMethods"] {
            assert!(!limits.contains_key(absent), "iris must not offer {absent}");
        }
        assert!(
            sceneworks_core::image_sampling_controls::supports_scheduler_shift(&entry),
            "the shift axis is declared without a scheduler menu"
        );
        assert!(entry["defaults"]["schedulerShift"].as_f64().unwrap() > 0.0);
        assert_eq!(
            sceneworks_core::image_sampling_controls::supports_cfg_interval(&entry),
            caps.supports_cfg_interval
        );
        assert_eq!(
            !sceneworks_core::image_sampling_controls::caption_overflow_policies(&entry).is_empty(),
            caps.supports_caption_overflow_policy
        );
        assert!(
            sceneworks_core::image_sampling_controls::negative_prompt_requires_guidance(&entry)
        );
        assert!(caps.supports_preview, "previews are wired generically");
        assert!(caps.supports_lora && caps.supports_lokr);
        assert_eq!(entry["loraCompatibility"]["families"], json!(["iris"]));
        assert_eq!(
            entry["loraCompatibility"]["irisTask"].as_str(),
            Some(IrisTask::Generation.name())
        );
        assert!(entry
            .get("mlx")
            .and_then(|mlx| mlx.get("quantize"))
            .is_none());
        assert_eq!(
            limits["requiresDimensionsMultipleOf"].as_u64(),
            Some(u64::from(gen_core::iris::SIZE_MULTIPLE))
        );
        assert_eq!(
            limits["maxDimension"].as_u64(),
            Some(u64::from(caps.max_size))
        );
        assert_eq!(
            limits["minDimension"].as_u64(),
            Some(u64::from(caps.min_size))
        );
        assert_eq!(
            entry["defaults"]["steps"].as_u64(),
            Some(u64::from(gen_core::iris::DEFAULT_STEPS))
        );
        assert_eq!(
            entry["defaults"]["resolution"].as_str(),
            Some(format!("{0}x{0}", gen_core::iris::DEFAULT_SIZE).as_str()),
            "the default stays 1024-class"
        );
        assert_eq!(
            entry["image"]["patchGridVisibleBelow"].as_u64(),
            Some(u64::from(gen_core::iris::DEFAULT_SIZE))
        );
    }

    /// Iris's release default (100 steps) is above the generic lane's historical 80-step ceiling;
    /// the ceiling rises to the model's own default so it is never clamped away.
    #[test]
    fn the_steps_ceiling_admits_the_models_own_default() {
        assert_eq!(
            super::super::max_requested_steps(gen_core::iris::DEFAULT_STEPS),
            gen_core::iris::DEFAULT_STEPS
        );
        assert_eq!(super::super::max_requested_steps(40), 80);
    }

    // ---------------------------------------------------------------------------------------
    // sc-25679 — the Iris job arm end to end with a stub engine.
    //
    // `generate_stream_using` (MLX) / `generate_candle_stream_using` (Candle, sc-25680) is the
    // production arm (the plain names pass the real loader):
    // weights/encoder resolution, the tier/admission preflight, the cached load, the per-item drive
    // and `consume_gen_events` all run for real against a staged (weightless) cache. Only the engine
    // is a stub, so these pin what the worker does with the engine's Step progress, a user cancel
    // and an engine failure — not a re-implementation of the arm.
    // ---------------------------------------------------------------------------------------

    use crate::api_client::ApiClient;
    use axum::extract::{Path as AxumPath, State};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use gen_core::{GenerationOutput, GenerationRequest, Generator, Progress};
    use sceneworks_core::contracts::JobSnapshot;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    const BACKBONE_REPO: &str = "speridlabs/iris-3b";
    const BACKBONE_REVISION: &str = "7445443349bc9abe3c96f01ff793e2098ca012b3";

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum ArmBehavior {
        /// Report a step, outlive the consumer's 2-second cancel-poll throttle, report another, then
        /// stop only when the request's own cancel flag trips.
        CancelAware,
        /// Report a step, then fail the way an engine refusal does.
        Fails,
        /// Report a step, emit one (W/16)×(H/16) x0 preview frame the way the Iris engine does every
        /// step, report a second step (the post the preview rides), then complete with an image and
        /// a `caption_truncated` warning through `generate_with_report`.
        Completes,
    }

    /// What the engine was handed: every request field the Iris controls map onto.
    #[derive(Clone, Debug, PartialEq)]
    struct Seen {
        prompt: String,
        negative_prompt: Option<String>,
        width: u32,
        height: u32,
        seed: Option<u64>,
        steps: Option<u32>,
        guidance: Option<f32>,
        sampler: Option<String>,
        scheduler: Option<String>,
        scheduler_shift: Option<f32>,
        cfg_interval: Option<(f32, f32)>,
        caption_overflow: Option<gen_core::CaptionOverflowPolicy>,
    }

    impl Seen {
        fn of(req: &GenerationRequest) -> Self {
            Self {
                prompt: req.prompt.clone(),
                negative_prompt: req.negative_prompt.clone(),
                width: req.width,
                height: req.height,
                seed: req.seed,
                steps: req.steps,
                guidance: req.guidance,
                sampler: req.sampler.clone(),
                scheduler: req.scheduler.clone(),
                scheduler_shift: req.scheduler_shift,
                cfg_interval: req.cfg_interval,
                caption_overflow: req.caption_overflow,
            }
        }
    }

    const FIXTURE_WARNING: &str = "iris: caption overflow — 7 caption tokens dropped (fixture)";

    struct IrisArmProbe {
        descriptor: gen_core::ModelDescriptor,
        behavior: ArmBehavior,
        generate_calls: Arc<Mutex<u32>>,
        /// Every request the engine was handed, in order.
        seen: Arc<Mutex<Vec<Seen>>>,
        /// Set when the engine stopped because its OWN request cancel flag tripped.
        stopped_on_cancel: Arc<Mutex<bool>>,
        /// The adapters the loader handed this engine (`LoadSpec::adapters`).
        adapters: Vec<gen_core::AdapterSpec>,
    }

    impl Generator for IrisArmProbe {
        fn descriptor(&self) -> &gen_core::ModelDescriptor {
            &self.descriptor
        }

        /// One report per loaded adapter, the way the Iris engine publishes its install outcome:
        /// everything applied, one stem skipped.
        fn adapter_apply_reports(&self) -> Vec<gen_core::AdapterApplyReport> {
            self.adapters
                .iter()
                .map(|adapter| gen_core::AdapterApplyReport {
                    adapter_path: adapter.path.clone(),
                    applied: 24,
                    skipped: vec!["pixel.blocks.0.attn".to_owned()],
                })
                .collect()
        }

        fn memory_strategy_safety_check(
            &self,
            _context: &gen_core::MemoryRunContext,
        ) -> gen_core::MemorySafetyDecision {
            gen_core::MemorySafetyDecision::Accept
        }

        fn begin_memory_strategy_request(
            &self,
            _context: &gen_core::MemoryRunContext,
        ) -> gen_core::Result<Option<Box<dyn gen_core::MemoryRequestScope + '_>>> {
            Ok(None)
        }

        fn validate(&self, _req: &GenerationRequest) -> gen_core::Result<()> {
            Ok(())
        }

        fn generate(
            &self,
            req: &GenerationRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<GenerationOutput> {
            *self.generate_calls.lock().unwrap() += 1;
            self.seen.lock().unwrap().push(Seen::of(req));
            on_progress(Progress::Step {
                current: 1,
                total: 100,
            });
            let image = || {
                GenerationOutput::Images(vec![gen_core::Image {
                    width: req.width,
                    height: req.height,
                    pixels: vec![0; (req.width * req.height * 3) as usize],
                }])
            };
            match self.behavior {
                ArmBehavior::Fails => Err(gen_core::Error::Unsupported(
                    "iris fixture engine failure".to_owned(),
                )),
                ArmBehavior::Completes => {
                    req.preview.emit(gen_core::PreviewFrame {
                        current: 1,
                        total: 100,
                        image: gen_core::Image {
                            width: req.width / 16,
                            height: req.height / 16,
                            pixels: vec![128; ((req.width / 16) * (req.height / 16) * 3) as usize],
                        },
                    });
                    on_progress(Progress::Step {
                        current: 2,
                        total: 100,
                    });
                    Ok(image())
                }
                ArmBehavior::CancelAware => {
                    std::thread::sleep(Duration::from_millis(2_300));
                    on_progress(Progress::Step {
                        current: 2,
                        total: 100,
                    });
                    let deadline = std::time::Instant::now() + Duration::from_secs(20);
                    while !req.cancel.is_cancelled() && std::time::Instant::now() < deadline {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    if req.cancel.is_cancelled() {
                        *self.stopped_on_cancel.lock().unwrap() = true;
                        return Err(gen_core::Error::Canceled);
                    }
                    Ok(image())
                }
            }
        }

        /// The Iris engine's primary path: the output plus its `caption_truncated` warnings.
        fn generate_with_report(
            &self,
            req: &GenerationRequest,
            on_progress: &mut dyn FnMut(Progress),
        ) -> gen_core::Result<gen_core::GenerationReport> {
            let mut report =
                gen_core::GenerationReport::from_output(self.generate(req, on_progress)?);
            if self.behavior == ArmBehavior::Completes {
                report.warnings.push(gen_core::GenerationWarning {
                    code: "caption_truncated".to_owned(),
                    message: FIXTURE_WARNING.to_owned(),
                });
            }
            Ok(report)
        }
    }

    type Posts = Arc<Mutex<Vec<Value>>>;

    /// A job API stub: the job GET reports `cancel_requested`, every progress POST is recorded.
    async fn spawn_job_api(cancel_requested: bool) -> (String, Posts) {
        async fn job_route(
            State((_, cancel)): State<(Posts, bool)>,
            AxumPath(job_id): AxumPath<String>,
        ) -> Response {
            Json(crate::tests::job_snapshot_json(&job_id, cancel)).into_response()
        }
        async fn progress_route(
            State((posts, cancel)): State<(Posts, bool)>,
            AxumPath(job_id): AxumPath<String>,
            Json(body): Json<Value>,
        ) -> Response {
            posts.lock().unwrap().push(body);
            Json(crate::tests::job_snapshot_json(&job_id, cancel)).into_response()
        }
        async fn heartbeat_route(AxumPath(worker_id): AxumPath<String>) -> Response {
            Json(json!({
                "id": worker_id, "gpuId": "cpu", "gpuName": null, "status": "busy",
                "currentJobId": "job-iris", "capabilities": [], "loadedModels": [],
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

    /// What the arm produced: its outcome, the progress posts, the asset writes, whether the
    /// stub engine ran, and any file written under the project.
    struct ArmRun {
        outcome: WorkerResult<()>,
        /// How many times the arm called the loader (0 ⇒ refused before load).
        load_calls: u32,
        posts: Vec<Value>,
        asset_writes: Vec<Value>,
        generate_calls: u32,
        seen: Vec<Seen>,
        /// The adapter stack the loader was handed (`LoadSpec::adapters`).
        loaded_adapters: Vec<gen_core::AdapterSpec>,
        stopped_on_cancel: bool,
        project_files: Vec<std::path::PathBuf>,
    }

    fn files_under(root: &Path) -> Vec<std::path::PathBuf> {
        let mut found = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    found.push(path);
                }
            }
        }
        found
    }

    /// Drive `generate_stream_using` for one iris text-to-image job; `extra` is merged into the
    /// job payload.
    fn run_iris_arm(behavior: ArmBehavior, cancel_requested: bool, extra: Value) -> ArmRun {
        run_iris_arm_with_memory_cap(behavior, cancel_requested, extra, "64")
    }

    /// [`run_iris_arm`] against an emulated `memory_cap_gb` budget: the MLX unified-memory cap on
    /// macOS, the `SCENEWORKS_CUDA_VRAM_CAP_GB` card cap on the Candle lane.
    fn run_iris_arm_with_memory_cap(
        behavior: ArmBehavior,
        cancel_requested: bool,
        extra: Value,
        memory_cap_gb: &str,
    ) -> ArmRun {
        run_iris_arm_with(behavior, cancel_requested, extra, memory_cap_gb, &[])
    }

    /// One adapter file the arm installs under the data dir's `loras/` before the job runs: its
    /// catalog id, its safetensors `__metadata__`, and the job weight.
    struct TestAdapter {
        id: &'static str,
        metadata: Value,
        weight: f64,
    }

    /// Write a minimal, header-valid safetensors adapter (one f32 tensor) with `metadata`.
    fn write_adapter(path: &Path, metadata: &Value) {
        let header = json!({
            "__metadata__": metadata,
            "blocks.0.attn.qkv.lora_down.weight": {
                "dtype": "F32", "shape": [1], "data_offsets": [0, 4]
            },
        })
        .to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0, 0, 0, 0]);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// [`run_iris_arm_with_memory_cap`] with `adapters` installed and selected on the job.
    fn run_iris_arm_with(
        behavior: ArmBehavior,
        cancel_requested: bool,
        extra: Value,
        memory_cap_gb: &str,
        adapters: &[TestAdapter],
    ) -> ArmRun {
        // The arm admits against the host's memory before loading; pin an emulated budget so the
        // outcome does not depend on the host (a 7 GiB hosted runner otherwise refuses the request
        // before the arm under test is reached, and a GPU-less runner yields no budget at all).
        let _env = crate::test_env::EnvVars::set(&[
            ("HF_HUB_CACHE", ""),
            ("HUGGINGFACE_HUB_CACHE", ""),
            ("HF_HOME", ""),
            #[cfg(target_os = "macos")]
            (crate::mlx_fit_gate::MLX_MEMORY_CAP_ENV, memory_cap_gb),
            #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
            (crate::vram_gate::CUDA_VRAM_CAP_ENV, memory_cap_gb),
        ]);
        let data = tempfile::tempdir().unwrap();
        let project = tempfile::tempdir().unwrap();
        // A weightless but complete install: the pinned backbone and its text encoder.
        for file in ["config.yaml", "model.safetensors"] {
            stage(data.path(), BACKBONE_REPO, BACKBONE_REVISION, file);
        }
        stage_text_encoder(data.path());
        for (repo, revision) in [(BACKBONE_REPO, BACKBONE_REVISION), (TE_REPO, TE_REVISION)] {
            let refs = sceneworks_core::hf_home::huggingface_repo_cache_path(data.path(), repo)
                .unwrap()
                .join("refs");
            std::fs::create_dir_all(&refs).unwrap();
            std::fs::write(refs.join("main"), revision).unwrap();
        }
        let loras: Vec<Value> = adapters
            .iter()
            .map(|adapter| {
                let path = data
                    .path()
                    .join("loras")
                    .join(adapter.id)
                    .join("adapter.safetensors");
                write_adapter(&path, &adapter.metadata);
                json!({ "id": adapter.id, "path": path, "weight": adapter.weight })
            })
            .collect();
        let calls = Arc::new(Mutex::new(0_u32));
        let loads = Arc::new(Mutex::new(0_u32));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let loaded_adapters = Arc::new(Mutex::new(Vec::new()));
        let stopped_on_cancel = Arc::new(Mutex::new(false));
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        // `block_on` polls the arm on THIS thread, so the thread-local capture sees its events.
        let ((outcome, posts, asset_writes), _logged) = with_captured_tracing(|| {
            runtime.block_on(async {
                let (base_url, posts) = spawn_job_api(cancel_requested).await;
                let mut settings = settings_at(data.path());
                settings.api_url = base_url;
                settings.heartbeat_seconds = 5;
                let api = ApiClient::new(&settings);
                let job: JobSnapshot =
                    serde_json::from_value(crate::tests::job_snapshot_json("job-iris", false))
                        .unwrap();
                let mut payload = json!({
                    "projectId": "project-iris",
                    "model": "iris_3b",
                    "mode": "text_to_image",
                    "prompt": "a lighthouse at dusk",
                    "width": 64,
                    "height": 64,
                    "count": 1,
                    "loras": loras,
                    "modelManifestEntry": Value::Object(shipped_iris_entry()),
                });
                for (key, value) in extra.as_object().cloned().unwrap_or_default() {
                    payload[key] = value;
                }
                let request = sceneworks_core::image_request::ImageRequest::from_payload(
                    payload.as_object().unwrap(),
                );
                let plan = super::super::ImagePlan::with_count(&request, request.count, None);
                let descriptor = crate::engines::mlx_model("iris_3b")
                    .expect("iris_3b is linked")
                    .descriptor;
                let loader_calls = Arc::clone(&calls);
                let loader_loads = Arc::clone(&loads);
                let loader_seen = Arc::clone(&seen);
                let loader_stopped = Arc::clone(&stopped_on_cancel);
                let loader_adapters = Arc::clone(&loaded_adapters);
                let loader = move |_engine: &str, spec: &LoadSpec| {
                    *loader_loads.lock().unwrap() += 1;
                    *loader_adapters.lock().unwrap() = spec.adapters.clone();
                    Ok(Box::new(IrisArmProbe {
                        descriptor,
                        behavior,
                        generate_calls: loader_calls,
                        seen: loader_seen,
                        stopped_on_cancel: loader_stopped,
                        adapters: spec.adapters.clone(),
                    }) as Box<dyn Generator>)
                };
                let mut asset_writes = Vec::new();
                // The SAME arm body on both lanes: the MLX `generate_stream_using` on macOS, the
                // Candle `generate_candle_stream_using` off-Mac (sc-25680) — so progress, cancel,
                // failure and knob shaping are pinned per backend by one set of assertions.
                #[cfg(target_os = "macos")]
                let arm = super::super::generate_stream_using(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    project.path(),
                    "mlx",
                    &mut asset_writes,
                    loader,
                );
                #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
                let arm = super::super::generate_candle_stream_using(
                    &api,
                    &settings,
                    &job,
                    &plan,
                    project.path(),
                    "cuda",
                    &mut asset_writes,
                    loader,
                );
                let outcome = tokio::time::timeout(Duration::from_secs(60), arm)
                    .await
                    .expect("the iris arm finished");
                let posts = posts.lock().unwrap().clone();
                (outcome, posts, asset_writes)
            })
        });
        let generate_calls = *calls.lock().unwrap();
        let seen = seen.lock().unwrap().clone();
        let loaded_adapters = loaded_adapters.lock().unwrap().clone();
        let stopped_on_cancel = *stopped_on_cancel.lock().unwrap();
        let load_calls = *loads.lock().unwrap();
        ArmRun {
            outcome,
            load_calls,
            posts,
            asset_writes,
            generate_calls,
            seen,
            loaded_adapters,
            stopped_on_cancel,
            project_files: files_under(project.path()),
        }
    }

    fn message(post: &Value) -> &str {
        post["message"].as_str().unwrap_or_default()
    }

    /// The first post reporting the engine's step 1/100, asserted NON-terminal `generating`.
    fn forwarded_step(posts: &[Value]) -> usize {
        let index = posts
            .iter()
            .position(|post| message(post).contains("step 1/100"))
            .unwrap_or_else(|| panic!("the engine's Step must reach the job API: {posts:#?}"));
        assert_eq!(
            (
                posts[index]["status"].as_str(),
                posts[index]["stage"].as_str()
            ),
            (Some("running"), Some("generating")),
            "{posts:#?}"
        );
        index
    }

    #[test]
    fn the_iris_arm_forwards_steps_and_a_user_cancel_ends_canceled_with_no_asset() {
        let run = run_iris_arm(ArmBehavior::CancelAware, true, json!({}));
        assert!(
            matches!(run.outcome, Err(crate::error::WorkerError::Canceled(_))),
            "a user-cancelled iris job ends Canceled, got {:?}",
            run.outcome
        );
        assert_eq!(run.load_calls, 1, "an admitted job loads exactly once");
        assert_eq!(run.generate_calls, 1, "the stub engine ran exactly once");
        assert!(
            run.stopped_on_cancel,
            "the user cancel must reach the engine's request flag, so the denoise stops"
        );
        let step = forwarded_step(&run.posts);
        let acknowledged = run
            .posts
            .iter()
            .position(|post| message(post).contains("Cancelling"))
            .unwrap_or_else(|| panic!("the cancel must be acknowledged: {:#?}", run.posts));
        assert!(acknowledged > step, "{:#?}", run.posts);
        assert_eq!(
            run.posts.last().unwrap()["status"],
            "canceled",
            "{:#?}",
            run.posts
        );
        assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        assert!(run.project_files.is_empty(), "{:?}", run.project_files);
    }

    #[test]
    fn an_iris_engine_failure_is_a_typed_failure_with_no_asset() {
        let run = run_iris_arm(ArmBehavior::Fails, false, json!({}));
        match &run.outcome {
            Err(crate::error::WorkerError::Engine(message)) => assert!(
                message.contains("iris fixture engine failure"),
                "the engine's reason survives: {message}"
            ),
            other => panic!("an engine failure must be a typed Engine error, got {other:?}"),
        }
        assert_eq!(run.generate_calls, 1);
        forwarded_step(&run.posts);
        assert!(
            run.posts.iter().all(|post| post["status"] != "completed"),
            "{:#?}",
            run.posts
        );
        assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        assert!(run.project_files.is_empty(), "{:?}", run.project_files);
    }

    // ---------------------------------------------------------------------------------------
    // sc-25681 — every Iris control through the REAL arm (MLX on macOS, Candle off-Mac).
    // ---------------------------------------------------------------------------------------

    /// What the engine sees with nothing but a prompt: the catalog defaults the arm resolves (steps
    /// 100, guidance 3.0) and the engine's own for everything else.
    fn default_seen(seed: Option<u64>) -> Seen {
        Seen {
            prompt: "a lighthouse at dusk".to_owned(),
            negative_prompt: None,
            width: 64,
            height: 64,
            seed,
            steps: Some(gen_core::iris::DEFAULT_STEPS),
            guidance: Some(gen_core::iris::DEFAULT_CFG_SCALE),
            sampler: None,
            scheduler: None,
            scheduler_shift: None,
            cfg_interval: None,
            caption_overflow: None,
        }
    }

    /// Table: a job payload → the exact request the engine is handed. Every control the catalog
    /// row offers reaches the engine verbatim (nothing accepted-then-ignored), and its absence
    /// means the engine's own default.
    ///
    /// *Mutation that reds this:* drop `cfg_interval` / `caption_overflow` from the request
    /// `generate_one_on_surface` builds, or re-introduce the shift withholding.
    #[test]
    fn every_iris_control_reaches_the_engine_through_the_real_arm() {
        use gen_core::CaptionOverflowPolicy as Policy;
        let cases: Vec<(Value, Seen)> = vec![
            (json!({ "seed": 11 }), default_seen(Some(11))),
            (
                json!({
                    "seed": 7,
                    "width": 512,
                    "height": 768,
                    "negativePrompt": "blurry, watermark",
                    "advanced": {
                        "steps": 30,
                        "guidanceScale": 4.5,
                        "sampler": "euler",
                        "schedulerShift": 2.5,
                        "cfgInterval": [0.1, 0.9],
                        "captionOverflow": "error",
                    },
                }),
                Seen {
                    negative_prompt: Some("blurry, watermark".to_owned()),
                    width: 512,
                    height: 768,
                    seed: Some(7),
                    steps: Some(30),
                    guidance: Some(4.5),
                    sampler: Some("euler".to_owned()),
                    scheduler_shift: Some(2.5),
                    cfg_interval: Some((0.1, 0.9)),
                    caption_overflow: Some(Policy::Error),
                    ..default_seen(None)
                },
            ),
            (
                json!({
                    "seed": 3,
                    "advanced": {
                        "sampler": "dpmpp_2m",
                        "schedulerShift": "6",
                        "captionOverflow": "silent",
                        "cfgInterval": [0, 0.5],
                    },
                }),
                Seen {
                    seed: Some(3),
                    sampler: Some("dpmpp_2m".to_owned()),
                    scheduler_shift: Some(6.0),
                    caption_overflow: Some(Policy::Silent),
                    cfg_interval: Some((0.0, 0.5)),
                    ..default_seen(None)
                },
            ),
            (
                // CFG off with nothing CFG-bound: honoured, guidance 1.0 reaches the engine.
                json!({ "seed": 5, "advanced": { "guidanceScale": 1.0, "captionOverflow": "warn" } }),
                Seen {
                    seed: Some(5),
                    guidance: Some(1.0),
                    caption_overflow: Some(Policy::Warn),
                    ..default_seen(None)
                },
            ),
        ];
        for (extra, want) in cases {
            let run = run_iris_arm(ArmBehavior::Fails, false, extra.clone());
            assert_eq!(run.seen, vec![want], "{extra}");
        }
    }

    /// A count-N job renders N engine requests on per-image seeds (seed, seed+1, …), each
    /// carrying the same controls.
    #[test]
    fn a_batch_renders_one_request_per_image_on_consecutive_seeds() {
        let run = run_iris_arm(
            ArmBehavior::Completes,
            false,
            json!({ "seed": 40, "count": 2, "advanced": { "sampler": "euler" } }),
        );
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        let seeds: Vec<_> = run.seen.iter().map(|seen| seen.seed).collect();
        assert_eq!(seeds, vec![Some(40), Some(41)]);
        assert!(run
            .seen
            .iter()
            .all(|seen| seen.sampler.as_deref() == Some("euler")));
        assert_eq!(run.asset_writes.len(), 2);
    }

    /// CFG off makes the bound negative prompt and the interval inert, and the engine refuses
    /// both: the arm refuses them too, typed and BEFORE load (no load, no render, no asset) —
    /// sc-25679 silently dropped the negative prompt; sc-25681 refuses it like the API does.
    /// Malformed values are typed refusals before load as well.
    #[test]
    fn inert_or_malformed_controls_are_refused_before_load() {
        let cases = [
            (
                json!({ "negativePrompt": "blurry", "advanced": { "guidanceScale": 1.0 } }),
                "negative prompt",
            ),
            (
                json!({ "advanced": { "guidanceScale": 1.0, "cfgInterval": [0.2, 0.8] } }),
                "guidance interval",
            ),
            (
                json!({ "advanced": { "cfgInterval": [0.9, 0.1] } }),
                "0 <= start",
            ),
            (
                json!({ "advanced": { "cfgInterval": "early" } }),
                "pair of numbers",
            ),
            (
                json!({ "advanced": { "captionOverflow": "loud" } }),
                "caption-overflow",
            ),
            (json!({ "advanced": { "schedulerShift": 0 } }), "> 0"),
        ];
        for (extra, fragment) in cases {
            let run = run_iris_arm(ArmBehavior::Completes, false, extra.clone());
            match &run.outcome {
                Err(crate::error::WorkerError::InvalidPayload(message)) => assert!(
                    message.contains("iris_3b") && message.contains(fragment),
                    "{extra}: {message}"
                ),
                other => panic!("{extra}: expected a typed refusal, got {other:?}"),
            }
            assert_eq!(run.load_calls, 0, "{extra}: refused before load");
            assert!(
                run.asset_writes.is_empty() && run.project_files.is_empty(),
                "{extra}"
            );
        }
    }

    fn generation_adapter(
        id: &'static str,
        network_type: Option<&str>,
        weight: f64,
    ) -> TestAdapter {
        let mut metadata = json!({
            "family": "iris",
            "irisTask": "generation",
            "baseModel": "iris_3b",
        });
        if let Some(network_type) = network_type {
            metadata["networkType"] = json!(network_type);
        }
        TestAdapter {
            id,
            metadata,
            weight,
        }
    }

    /// A completed render through the real arm: the engine's x0 step preview reaches the job API
    /// on the preview path, its `caption_truncated` warning and the applied LoRA + LoKr stack
    /// (identity joined with the engine's own apply report) land in the asset's provenance, and the
    /// controls ride the recorded settings so the recipe reopens with them.
    ///
    /// *Mutation that reds this:* skip `send_image_facts`, or drop the `ImageFacts` merge in
    /// `consume_gen_events`.
    #[test]
    fn a_completed_render_reports_previews_warnings_and_adapter_provenance() {
        let run = run_iris_arm_with(
            ArmBehavior::Completes,
            false,
            json!({
                "seed": 9,
                "width": 256,
                "height": 256,
                "advanced": {
                    "sampler": "euler",
                    "schedulerShift": 3.0,
                    "cfgInterval": [0.1, 0.9],
                    "captionOverflow": "warn",
                },
            }),
            "64",
            &[
                generation_adapter("iris-style", None, 0.8),
                generation_adapter("iris-lokr", Some("lokr"), 1.2),
            ],
        );
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        // Both adapters reached the load spec, in order, with their kind and job weight.
        let loaded: Vec<_> = run
            .loaded_adapters
            .iter()
            .map(|adapter| (adapter.kind, adapter.scale))
            .collect();
        assert_eq!(
            loaded,
            vec![
                (gen_core::AdapterKind::Lora, 0.8),
                (gen_core::AdapterKind::Lokr, 1.2)
            ]
        );
        // The preview rode a non-terminal progress post at the engine's (W/16)x(H/16) frame.
        let preview = run
            .posts
            .iter()
            .find_map(|post| post["result"]["previewFrame"].as_object())
            .unwrap_or_else(|| panic!("a preview frame must reach the job API: {:#?}", run.posts));
        assert_eq!(preview["current"], 1);
        assert!(preview["dataUrl"]
            .as_str()
            .is_some_and(|url| url.starts_with("data:image/")));
        // Provenance on the written asset.
        assert_eq!(run.asset_writes.len(), 1);
        let settings = &run.asset_writes[0]["rawAdapterSettings"];
        assert_eq!(
            settings["generationWarnings"],
            json!([{ "code": "caption_truncated", "message": FIXTURE_WARNING }])
        );
        let applied = settings["adapterApply"]
            .as_array()
            .expect("adapter provenance");
        assert_eq!(applied.len(), 2);
        for (entry, (id, kind, adapter)) in applied.iter().zip([
            ("iris-style", "lora", &run.loaded_adapters[0]),
            ("iris-lokr", "lokr", &run.loaded_adapters[1]),
        ]) {
            assert_eq!(entry["id"], id);
            assert_eq!(entry["kind"], kind);
            assert_eq!(entry["path"], json!(adapter.path.display().to_string()));
            assert_eq!(
                entry["scale"].as_f64().map(|v| v as f32),
                Some(adapter.scale)
            );
            assert_eq!(
                entry["sha256"].as_str().map(str::len),
                Some(64),
                "the loaded file's digest: {entry}"
            );
            assert_eq!(entry["applied"], 24);
            assert_eq!(entry["skipped"], json!(["pixel.blocks.0.attn"]));
        }
        // The controls are recorded with the image, so reopening the recipe restores them.
        for (key, value) in [
            ("sampler", json!("euler")),
            ("schedulerShift", json!(3.0)),
            ("cfgInterval", json!([0.1, 0.9])),
            ("captionOverflow", json!("warn")),
        ] {
            assert_eq!(settings[key], value, "{key}: {settings}");
        }
        assert_eq!(
            run.asset_writes[0]["loras"].as_array().map(Vec::len),
            Some(2)
        );
    }

    /// A replayed `advanced` block cannot forge the worker-owned provenance keys: a render with
    /// nothing to report records neither.
    #[test]
    fn replayed_provenance_keys_are_scrubbed() {
        let run = run_iris_arm(
            ArmBehavior::Completes,
            false,
            json!({
                "width": 256,
                "height": 256,
                "advanced": {
                    "adapterApply": [{ "id": "forged" }],
                    "generationWarnings": [{ "code": "forged", "message": "forged" }],
                },
            }),
        );
        assert!(run.outcome.is_ok(), "{:?}", run.outcome);
        let settings = &run.asset_writes[0]["rawAdapterSettings"];
        // The stub always reports its fixture warning, so the key is present with ONLY that.
        assert_eq!(
            settings["generationWarnings"],
            json!([{ "code": "caption_truncated", "message": FIXTURE_WARNING }])
        );
        assert!(settings.get("adapterApply").is_none(), "{settings}");
    }

    // ---------------------------------------------------------------------------------------
    // sc-25680 review — the Candle lane's declared VRAM floor (E8). Candle-only: `vram_gate` and
    // the Candle arm compile only off-Mac with `backend-candle`, so these run on the
    // windows-candle lane, never on a Mac.
    // ---------------------------------------------------------------------------------------

    /// Every tier key the Candle gate can name for the single bf16 artifact resolves a resident
    /// AND a staged prediction, so `fit_decision` is never `Unknown` (admit-anything) for iris.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn the_candle_gate_has_a_resident_and_a_staged_prediction_for_every_iris_tier_key() {
        let iris = shipped_iris_entry();
        for advanced in [
            json!({}),
            json!({ "mlxQuantize": 0 }),
            json!({ "mlxQuantize": 4 }),
            json!({ "mlxQuantize": 8 }),
        ] {
            let tier =
                crate::vram_gate::requested_tier_key(advanced.as_object().unwrap(), &iris, false);
            assert_eq!(
                crate::vram_gate::predicted_peak_gb(&iris, tier),
                Some(24.0),
                "{tier}: the padded resident floor"
            );
            assert_eq!(
                crate::vram_gate::predicted_sequential_peak_gb(&iris, tier),
                Some(21.4 + crate::vram_gate::HEADROOM_GB),
                "{tier}: the staged bound (recorded resident peak + reserve)"
            );
        }
    }

    /// A card under the floor gets a typed refusal BEFORE the loader runs — not a load that OOMs.
    #[cfg(all(not(target_os = "macos"), feature = "backend-candle"))]
    #[test]
    fn a_card_under_the_iris_candle_floor_is_refused_before_load() {
        let run = run_iris_arm_with_memory_cap(ArmBehavior::Fails, false, json!({}), "16");
        match &run.outcome {
            Err(crate::error::WorkerError::InvalidPayload(message)) => assert!(
                message.contains("iris_3b") && message.contains("VRAM"),
                "the refusal names the model and its VRAM need: {message}"
            ),
            other => panic!("a 16 GB card must be refused before load, got {other:?}"),
        }
        assert_eq!(run.load_calls, 0, "refused before the loader ran");
        assert_eq!(run.generate_calls, 0);
        assert!(run.asset_writes.is_empty(), "{:?}", run.asset_writes);
        assert!(run.project_files.is_empty(), "{:?}", run.project_files);
    }
}
