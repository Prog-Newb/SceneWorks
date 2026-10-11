//! The generic image lane's request controls beyond the positional sampling knobs (sc-25681, epic
//! 25678): the CFG interval, the caption-overflow policy, the CFG-bound negative prompt, plus the
//! per-image facts a render reports back (non-fatal engine warnings, adapter apply reports).
//!
//! Shared by the MLX `generate_stream` arm and the Candle `generate_candle_stream` arm, so both
//! lanes resolve, refuse and record exactly the same thing. Everything is decided against the
//! LINKED engine descriptor (what the engine honours) and the manifest declarations the API's
//! enqueue gate reads ([`sceneworks_core::image_sampling_controls`]); a control the engine does not
//! honour is a typed refusal before load, never accepted and dropped (E3).

use gen_core::{AdapterApplyReport, AdapterKind, AdapterSpec, Capabilities, CaptionOverflowPolicy};
use sceneworks_core::contracts::JsonObject;
use sceneworks_core::image_request::ImageRequest;
use sceneworks_core::image_sampling_controls::{
    self as controls, CaptionOverflow, CAPTION_OVERFLOW_KEY, CFG_INTERVAL_KEY,
};
use serde_json::{json, Value};

use crate::error::{WorkerError, WorkerResult};

/// `rawAdapterSettings` key of the non-fatal engine warnings of one image (e.g. Iris's
/// `caption_truncated`).
pub(crate) const GENERATION_WARNINGS_KEY: &str = "generationWarnings";
/// `rawAdapterSettings` key of the applied adapter stack of one image: identity (path, sha256,
/// scale, kind) joined with the engine's own apply report (applied / skipped).
pub(crate) const ADAPTER_APPLY_KEY: &str = "adapterApply";

/// The resolved request controls one generic-lane render carries onto its `GenerationRequest`.
/// `Default` is "the engine's own" for every field — what every model that declares none sends.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LaneRequestControls {
    pub(crate) cfg_interval: Option<(f32, f32)>,
    pub(crate) caption_overflow: Option<CaptionOverflowPolicy>,
}

fn engine_policy(policy: CaptionOverflow) -> CaptionOverflowPolicy {
    match policy {
        CaptionOverflow::Warn => CaptionOverflowPolicy::Warn,
        CaptionOverflow::Error => CaptionOverflowPolicy::Error,
        CaptionOverflow::Silent => CaptionOverflowPolicy::Silent,
    }
}

/// Resolve the request controls against the linked engine's `caps`, the resolved `guidance` and
/// the resolved `negative_prompt`, refusing (typed `InvalidPayload`, before any load) a malformed
/// value, a control the engine does not honour, and a control CFG-off would make inert:
///
/// * `advanced.cfgInterval` — needs `supports_cfg_interval`, and guidance != 1.0;
/// * `advanced.captionOverflow` — needs `supports_caption_overflow_policy`;
/// * `advanced.schedulerShift` — validated (> 0) for a model that declares the shift axis;
/// * a non-empty negative prompt on a model declaring `image.negativePromptRequiresGuidance` at
///   guidance 1.0 (the engine refuses it; sc-25679 dropped it, sc-25681 refuses it like the API).
pub(crate) fn resolve_lane_request_controls(
    request: &ImageRequest,
    caps: &Capabilities,
    guidance: Option<f32>,
    negative_prompt: Option<&str>,
) -> WorkerResult<LaneRequestControls> {
    let refuse =
        |message: String| WorkerError::InvalidPayload(format!("{}: {message}", request.model));
    let cfg_interval = controls::requested_cfg_interval(&request.advanced).map_err(refuse)?;
    let caption_overflow =
        controls::requested_caption_overflow(&request.advanced).map_err(refuse)?;
    if controls::supports_scheduler_shift(&request.model_manifest_entry) {
        controls::requested_scheduler_shift(&request.advanced).map_err(refuse)?;
    }
    if cfg_interval.is_some() && !caps.supports_cfg_interval {
        return Err(refuse(format!(
            "the engine does not honour a guidance interval (advanced.{CFG_INTERVAL_KEY})"
        )));
    }
    if caption_overflow.is_some() && !caps.supports_caption_overflow_policy {
        return Err(refuse(format!(
            "the engine does not honour a caption-overflow policy (advanced.{CAPTION_OVERFLOW_KEY})"
        )));
    }
    let cfg_off = guidance == Some(1.0);
    if cfg_off && cfg_interval.is_some() {
        return Err(refuse(
            "a guidance interval has no effect at guidance 1.0 (classifier-free guidance is off)"
                .to_owned(),
        ));
    }
    if cfg_off
        && controls::negative_prompt_requires_guidance(&request.model_manifest_entry)
        && negative_prompt.is_some_and(|prompt| !prompt.trim().is_empty())
    {
        return Err(refuse(
            "the negative prompt has no effect at guidance 1.0 (classifier-free guidance is off)"
                .to_owned(),
        ));
    }
    Ok(LaneRequestControls {
        cfg_interval,
        caption_overflow: caption_overflow.map(engine_policy),
    })
}

/// A native prompt batch resolved for one job (sc-25681): the prompts ONE engine call renders
/// together (`GenerationRequest::prompt_batch`) and the `count` repetitions of the whole list. The
/// engine returns `prompts × count` images in prompt-minor order, image `k` on seed `seed + k`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct NativePromptBatch {
    pub(crate) prompts: Vec<String>,
    pub(crate) count: u32,
}

impl NativePromptBatch {
    /// The images the one engine call returns.
    pub(crate) fn images(&self) -> u32 {
        u32::try_from(self.prompts.len())
            .unwrap_or(u32::MAX)
            .saturating_mul(self.count)
    }
}

/// Resolve the request's native prompt batch against the LINKED engine (`caps`), refusing (typed
/// `InvalidPayload`, before any load) a batch the engine cannot render — no `supports_prompt_batch`,
/// more images than its `max_count`, or a shape the API would refuse (the same core check) — and a
/// batch combined with a lane feature that renders per image (`per_image`: name → in use), which the
/// one-call batch would otherwise silently skip. `None` for an ordinary request.
pub(crate) fn resolve_native_prompt_batch(
    request: &ImageRequest,
    caps: &Capabilities,
    per_image: &[(&str, bool)],
) -> WorkerResult<Option<NativePromptBatch>> {
    if request.prompts.is_empty() {
        return Ok(None);
    }
    let refuse =
        |message: String| WorkerError::InvalidPayload(format!("{}: {message}", request.model));
    if !caps.supports_prompt_batch {
        return Err(refuse(
            "the engine does not render a prompt batch in one pass; send one job per prompt"
                .to_owned(),
        ));
    }
    if let Some(message) = controls::prompt_batch_error(
        &request.model,
        &request.prompt,
        &request.prompts,
        request.count,
        &request.model_manifest_entry,
    ) {
        return Err(WorkerError::InvalidPayload(message));
    }
    let batch = NativePromptBatch {
        prompts: request.prompts.clone(),
        count: request.count,
    };
    if batch.images() > caps.max_count {
        return Err(refuse(format!(
            "a prompt batch of {} images exceeds the {} the engine renders per request",
            batch.images(),
            caps.max_count
        )));
    }
    if let Some((feature, _)) = per_image.iter().find(|(_, in_use)| *in_use) {
        return Err(refuse(format!(
            "{feature} renders per image and cannot ride a prompt batch; send one job per prompt"
        )));
    }
    Ok(Some(batch))
}

/// The DECLARED extra working set of a native prompt batch on the Candle lane, in bytes (sc-25681):
/// every image beyond the first adds its pixel-space activations,
/// `(images − 1) × candle.promptBatchImageGb × max(1, width·height / 1024²)`. The single-image
/// floors (`candle.minMemoryGb` / `sequentialPeakGb`) already carry one image; the area term never
/// shrinks below 1024² (the floors were declared there). `0` for an ordinary request. A batch on an
/// entry that declares no per-image term is refused before load rather than admitted on a floor
/// sized for one image. S6 (sc-25684) owns the measured value.
#[cfg_attr(target_os = "macos", allow(dead_code))]
pub(crate) fn native_batch_overlay_bytes(
    request: &ImageRequest,
    batch: Option<&NativePromptBatch>,
    width: u32,
    height: u32,
) -> WorkerResult<u64> {
    let Some(batch) = batch else {
        return Ok(0);
    };
    let per_image_gb = request
        .model_manifest_entry
        .get("candle")
        .and_then(|candle| candle.get("promptBatchImageGb"))
        .and_then(Value::as_f64)
        .filter(|gb| gb.is_finite() && *gb > 0.0)
        .ok_or_else(|| {
            WorkerError::InvalidPayload(format!(
                "{}: no declared per-image memory for a prompt batch (candle.promptBatchImageGb); \
                 send one job per prompt",
                request.model
            ))
        })?;
    let megapixels = (f64::from(width) * f64::from(height) / (1024.0 * 1024.0)).max(1.0);
    let extra_images = f64::from(batch.images().saturating_sub(1));
    Ok((extra_images * per_image_gb * megapixels * crate::fit_gate::BYTES_PER_GIB).round() as u64)
}

/// One adapter's identity, resolved once per job before load: what was requested, from where,
/// at what strength, and the SHA-256 of the exact file the engine loads.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct AdapterIdentity {
    pub(crate) id: Option<String>,
    pub(crate) path: std::path::PathBuf,
    pub(crate) sha256: Option<String>,
    pub(crate) scale: f32,
    pub(crate) kind: AdapterKind,
}

/// The identities of `adapters` (the resolved load-order stack), labelled with the request LoRA
/// ids they came from (`request.loras` is the same order), resolved ONCE per job before load.
///
/// The digest comes from [`super::trusted_lora_hash`] — the same path workflow attribution uses:
/// the install marker's cached `loraFileSha256` when it still describes this exact file (no
/// re-hash), otherwise a SHA-256 on a blocking task with the worker heartbeat kept alive (never on
/// the async executor thread), cached back into the marker. A hashing failure leaves `sha256`
/// `None` — attribution, not generation, fails.
pub(crate) async fn adapter_identities(
    api: &crate::api_client::ApiClient,
    settings: &crate::settings::Settings,
    job: &sceneworks_core::contracts::JobSnapshot,
    request: &ImageRequest,
    adapters: &[AdapterSpec],
) -> Vec<AdapterIdentity> {
    let mut identities = Vec::with_capacity(adapters.len());
    for (index, adapter) in adapters.iter().enumerate() {
        let lora = request.loras.get(index).cloned().unwrap_or(Value::Null);
        let sha256 = super::trusted_lora_hash(api, settings, job, &lora, &adapter.path).await;
        identities.push(AdapterIdentity {
            id: lora
                .as_str()
                .or_else(|| lora.get("id").and_then(Value::as_str))
                .or_else(|| lora.get("loraId").and_then(Value::as_str))
                .map(str::to_owned),
            path: adapter.path.clone(),
            sha256,
            scale: adapter.scale,
            kind: adapter.kind,
        });
    }
    identities
}

/// The per-image facts block a render reports into its asset's `rawAdapterSettings`: the engine's
/// non-fatal warnings and, when adapters were loaded, each one's identity joined (by path) with the
/// engine's apply report. `None` when there is nothing to record, so a plain render's settings are
/// unchanged.
pub(crate) fn image_facts(
    warnings: &[gen_core::GenerationWarning],
    identities: &[AdapterIdentity],
    reports: &[AdapterApplyReport],
) -> Option<JsonObject> {
    let mut facts = JsonObject::new();
    if !warnings.is_empty() {
        facts.insert(
            GENERATION_WARNINGS_KEY.to_owned(),
            Value::Array(
                warnings
                    .iter()
                    .map(|warning| json!({ "code": warning.code, "message": warning.message }))
                    .collect(),
            ),
        );
    }
    if !identities.is_empty() {
        facts.insert(
            ADAPTER_APPLY_KEY.to_owned(),
            Value::Array(
                identities
                    .iter()
                    .map(|identity| {
                        let report = reports
                            .iter()
                            .find(|report| report.adapter_path == identity.path);
                        json!({
                            "id": identity.id,
                            "path": identity.path.display().to_string(),
                            "sha256": identity.sha256,
                            "scale": identity.scale,
                            "kind": match identity.kind {
                                AdapterKind::Lora => "lora",
                                AdapterKind::Lokr => "lokr",
                            },
                            // `null` when the engine publishes no report for this file (providers
                            // without a partial-install surface) — never a guessed count.
                            "applied": report.map(|report| report.applied),
                            "skipped": report.map(|report| report.skipped.clone()),
                        })
                    })
                    .collect(),
            ),
        );
    }
    (!facts.is_empty()).then_some(facts)
}

/// The job's distinct warning messages across its written assets, for the completion message.
pub(crate) fn job_warning_messages(asset_writes: &[Value]) -> Vec<String> {
    let mut messages: Vec<String> = Vec::new();
    for write in asset_writes {
        let warnings = write
            .get("rawAdapterSettings")
            .and_then(|settings| settings.get(GENERATION_WARNINGS_KEY))
            .and_then(Value::as_array);
        for warning in warnings.into_iter().flatten() {
            if let Some(message) = warning.get("message").and_then(Value::as_str) {
                if !messages.iter().any(|seen| seen == message) {
                    messages.push(message.to_owned());
                }
            }
        }
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(advanced: Value, negative: &str, image: Value) -> ImageRequest {
        let payload = json!({
            "projectId": "p",
            "model": "m",
            "prompt": "x",
            "negativePrompt": negative,
            "advanced": advanced,
            "modelManifestEntry": { "id": "m", "image": image },
        });
        ImageRequest::from_payload(payload.as_object().unwrap())
    }

    fn honouring() -> Capabilities {
        Capabilities {
            supports_cfg_interval: true,
            supports_caption_overflow_policy: true,
            ..Default::default()
        }
    }

    #[test]
    fn controls_reach_the_request_when_the_engine_honours_them() {
        let request = request(
            json!({ "cfgInterval": [0.25, 0.75], "captionOverflow": "error" }),
            "",
            json!({}),
        );
        let resolved =
            resolve_lane_request_controls(&request, &honouring(), Some(3.0), None).unwrap();
        assert_eq!(
            resolved,
            LaneRequestControls {
                cfg_interval: Some((0.25, 0.75)),
                caption_overflow: Some(CaptionOverflowPolicy::Error),
            }
        );
        let none = resolve_lane_request_controls(
            &self::request(json!({}), "", json!({})),
            &Capabilities::default(),
            Some(3.0),
            None,
        )
        .unwrap();
        assert_eq!(none, LaneRequestControls::default());
    }

    #[test]
    fn an_unhonoured_or_inert_control_is_refused_not_dropped() {
        let cases = [
            // (advanced, negative, image decl, caps honour, guidance, message fragment)
            (
                json!({ "cfgInterval": [0.2, 0.8] }),
                "",
                json!({}),
                false,
                Some(3.0),
                "guidance interval",
            ),
            (
                json!({ "captionOverflow": "warn" }),
                "",
                json!({}),
                false,
                Some(3.0),
                "caption-overflow",
            ),
            (
                json!({ "cfgInterval": [0.2, 0.8] }),
                "",
                json!({}),
                true,
                Some(1.0),
                "guidance 1.0",
            ),
            (
                json!({}),
                "blurry",
                json!({ "negativePromptRequiresGuidance": true }),
                true,
                Some(1.0),
                "negative prompt",
            ),
            (
                json!({ "cfgInterval": [0.8, 0.2] }),
                "",
                json!({}),
                true,
                Some(3.0),
                "0 <= start",
            ),
            (
                json!({ "captionOverflow": "shout" }),
                "",
                json!({}),
                true,
                Some(3.0),
                "not a caption",
            ),
            (
                json!({ "schedulerShift": 0 }),
                "",
                json!({ "supportsSchedulerShift": true }),
                true,
                Some(3.0),
                "> 0",
            ),
        ];
        for (advanced, negative, image, honour, guidance, fragment) in cases {
            let request = request(advanced.clone(), negative, image);
            let caps = if honour {
                honouring()
            } else {
                Capabilities::default()
            };
            let negative = (!negative.is_empty()).then_some(negative);
            match resolve_lane_request_controls(&request, &caps, guidance, negative) {
                Err(WorkerError::InvalidPayload(message)) => {
                    assert!(message.contains(fragment), "{advanced}: {message}")
                }
                other => panic!("{advanced}: expected a typed refusal, got {other:?}"),
            }
        }
        // The undeclared negative-prompt binding never refuses.
        let request = request(json!({}), "blurry", json!({}));
        assert!(
            resolve_lane_request_controls(&request, &honouring(), Some(1.0), Some("blurry"))
                .is_ok()
        );
    }

    #[test]
    fn image_facts_join_identity_with_the_engine_report_by_path() {
        let identities = vec![
            AdapterIdentity {
                id: Some("style".into()),
                path: "/a/style.safetensors".into(),
                sha256: Some("ab".into()),
                scale: 0.8,
                kind: AdapterKind::Lora,
            },
            AdapterIdentity {
                id: Some("lokr".into()),
                path: "/a/lokr.safetensors".into(),
                sha256: None,
                scale: 1.0,
                kind: AdapterKind::Lokr,
            },
        ];
        let reports = vec![AdapterApplyReport {
            adapter_path: "/a/style.safetensors".into(),
            applied: 12,
            skipped: vec!["blocks.3.attn".into()],
        }];
        let warnings = vec![gen_core::GenerationWarning {
            code: "caption_truncated".into(),
            message: "dropped 4 tokens".into(),
        }];
        let facts = image_facts(&warnings, &identities, &reports).unwrap();
        assert_eq!(
            Value::Object(facts.clone()),
            json!({
                "generationWarnings": [{ "code": "caption_truncated", "message": "dropped 4 tokens" }],
                "adapterApply": [
                    { "id": "style", "path": "/a/style.safetensors", "sha256": "ab", "scale": 0.800000011920929,
                      "kind": "lora", "applied": 12, "skipped": ["blocks.3.attn"] },
                    { "id": "lokr", "path": "/a/lokr.safetensors", "sha256": null, "scale": 1.0,
                      "kind": "lokr", "applied": null, "skipped": null },
                ],
            })
        );
        assert_eq!(image_facts(&[], &[], &reports), None);
        let writes = vec![
            json!({ "rawAdapterSettings": Value::Object(facts.clone()) }),
            json!({ "rawAdapterSettings": Value::Object(facts) }),
            json!({ "rawAdapterSettings": {} }),
        ];
        assert_eq!(
            job_warning_messages(&writes),
            vec!["dropped 4 tokens".to_owned()]
        );
    }

    async fn identities_for(loras: Vec<Value>, adapters: &[AdapterSpec]) -> Vec<AdapterIdentity> {
        let settings = crate::tests::test_settings("http://127.0.0.1:0".to_owned(), None);
        let api = crate::api_client::ApiClient::new(&settings);
        let job: sceneworks_core::contracts::JobSnapshot =
            serde_json::from_value(crate::tests::job_snapshot_json("job-hash", false)).unwrap();
        let mut request = request(json!({}), "", json!({}));
        request.loras = loras;
        adapter_identities(&api, &settings, &job, &request, adapters).await
    }

    /// No install marker: the loaded file is hashed (off the executor) and labelled with its
    /// request id.
    #[tokio::test]
    async fn adapter_identities_hash_the_loaded_file_and_keep_the_request_id() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("a.safetensors");
        std::fs::write(&file, b"abc").unwrap();
        let identities = identities_for(
            vec![json!({ "id": "style-a", "weight": 0.7 })],
            &[AdapterSpec::new(file.clone(), 0.7, AdapterKind::Lora)],
        )
        .await;
        assert_eq!(identities.len(), 1);
        assert_eq!(identities[0].id.as_deref(), Some("style-a"));
        assert_eq!(
            identities[0].sha256.as_deref(),
            Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        );
        assert_eq!(identities[0].path, file);
    }

    /// An install marker that still describes the exact file supplies its cached digest — the file
    /// is NOT re-hashed (the marker's value is deliberately not the bytes' real digest).
    ///
    /// *Mutation that reds this:* hash the file unconditionally (bypass `trusted_lora_hash`).
    #[tokio::test]
    async fn adapter_identities_reuse_the_install_markers_cached_digest() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("cached.safetensors");
        std::fs::write(&file, b"abc").unwrap();
        let identity = crate::imports::model_file_identity(&file).unwrap();
        let cached = "f".repeat(64);
        std::fs::write(
            dir.path().join(crate::INSTALL_MARKER),
            json!({
                "loraFileName": identity.name,
                "loraFileBytes": identity.bytes,
                "loraFileModifiedNanos": identity.modified_nanos,
                "loraFileSha256": cached,
            })
            .to_string(),
        )
        .unwrap();
        let identities = identities_for(
            vec![json!({ "id": "cached" })],
            &[AdapterSpec::new(file, 1.0, AdapterKind::Lora)],
        )
        .await;
        assert_eq!(identities[0].sha256.as_deref(), Some(cached.as_str()));
    }
}
