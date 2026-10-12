//! Iris-3B generation training — the SceneWorks product surface of the native MLX (sc-25685) and
//! Candle/CUDA (sc-25686) trainers registered under `iris_3b` (epic sc-25678).
//!
//! The engine contract is `gen_core::iris::train`: the shared `TrainingConfig` plus the
//! `model_options["iris"]` options bag, whose unknown keys the trainer REFUSES. This module owns
//! the SceneWorks half of that contract so the Training Studio offers exactly what the trainer
//! accepts and nothing it refuses (E3):
//!
//! * [`iris_generation_target`] — the one training target. Its `limits` enumerate every choice the
//!   trainer honors (optimizer `adamw | muon`, LR scheduler `constant | cosine`, timestep sampler
//!   `logit_normal | uniform`, train dtype `bf16 | f32`, network `lora | lokr | full`) and pin the
//!   refused ones closed (timestep bias `balanced`, loss `mse`, no gradient checkpointing).
//! * [`validate_iris_training_config`] — the submit-time twin of `IrisTrainPlan::resolve`'s
//!   request-only refusals, so a bad request is a typed 400 rather than a failure after the
//!   weights load. Notably the MICRO-STEP rule: the shared contract counts `steps`, `saveEvery`,
//!   `sampleEvery` and `lrWarmupSteps` in micro-steps, and the trainer lands checkpoints, previews
//!   and the warmup on optimizer-step boundaries, so each must be a whole number of
//!   `gradientAccumulation` windows ([`TrainingTargetLimitError::NotMultipleOf`]).
//! * [`validate_iris_dataset_captions`] / [`validate_item_captions`] — per-item named captions
//!   (`TrainingItem.model_options["captions"] = {field: text}`, upstream's per-sample `info`
//!   dict) and the `caption_field(s)` selection that reads them.
//! * [`iris_training_memory_floor_gb`] — the DECLARED (measured: false) per-lane floors, so a routed
//!   Iris training lane is never admitted unbounded. S6 owns the measured values.
//!
//! Out of scope here (S11 sc-25689, and the trainer refuses them typed): official stage presets,
//! bucket/area shapes, REPA/iREPA, activation-checkpointing policies, holdout validation,
//! distributed training. None of them is offered.
//!
//! The same `iris_train` kernel serves the two downstream TASK trainers too, keyed on the target's
//! base model ([`IrisObjective`]): depth ([`depth`], S9 sc-25687) and restoration ([`restore`], S10
//! sc-25688). Everything kernel-wide (durable cancel, resume / new phase, the memory-floor gate, the
//! data walk) applies to all three; the request surface, the dataset contract and the floors are per
//! objective.

pub mod depth;
pub mod restore;

use serde_json::{json, Map, Value};

use super::{
    object, ContractNumber, ExtraFields, TrainingConfig, TrainingDataset, TrainingModality,
    TrainingOutputKind, TrainingPlanError, TrainingTarget, TrainingTargetLimitError,
};

/// The SceneWorks training kernel id the router and worker key the Iris trainer on.
pub const IRIS_TRAINING_KERNEL: &str = "iris_train";
/// The one Iris training target id.
pub const IRIS_TRAINING_TARGET_ID: &str = "iris_3b_generation";
/// The catalog model (and engine trainer id) the target trains: the generation task install.
pub const IRIS_BASE_MODEL: &str = "iris_3b";
/// The upstream backbone repo the generation install pulls.
pub const IRIS_BASE_MODEL_REPO: &str = "speridlabs/iris-3b";
/// The SceneWorks LoRA family an Iris adapter registers under (the engine stamps `family = "iris"`).
pub const IRIS_FAMILY: &str = "iris";
/// The task an Iris generation artifact belongs to (the engine's `irisTask` stamp).
pub const IRIS_GENERATION_TASK: &str = "generation";
/// `TrainingConfig.advanced` key carrying the engine's `model_options["iris"]` bag verbatim.
pub const IRIS_OPTIONS_KEY: &str = "iris";
/// `TrainingDatasetItem.extra` key carrying an item's named captions — passed through to the
/// engine `TrainingItem.model_options["captions"]` (`gen_core::iris::train::ITEM_CAPTIONS_KEY`).
pub const ITEM_CAPTIONS_KEY: &str = "captions";
/// Request key naming a previous Iris training job to RESUME (its newest durable checkpoint). The
/// API resolves it to the engine's `resume_from` path; a client never sends a path.
pub const RESUME_FROM_JOB_KEY: &str = "resumeFromJobId";
/// Request key naming a previous FULL Iris training job whose weights seed a full run
/// (`init: load_from`). The API resolves it to the engine's `load_from` path.
pub const LOAD_FROM_JOB_KEY: &str = "loadFromJobId";
/// Upstream's most preview prompts per cadence (`gen_core::iris::train::PREVIEW_PROMPT_CAP`).
pub const IRIS_PREVIEW_PROMPT_CAP: u32 = 4;

/// The micro-step fields that must be whole `gradientAccumulation` windows. `(request field, where it
/// lives)`; `true` = a top-level `TrainingConfig` field, `false` = an `advanced` key.
pub const IRIS_MICRO_STEP_FIELDS: [(&str, bool); 4] = [
    ("steps", true),
    ("saveEvery", true),
    ("sampleEvery", false),
    ("lrWarmupSteps", false),
];

/// Every engine option a client may set. The engine's full allow-list
/// (`gen_core::iris::train::IrisTrainOptions`) minus the two PATH options the API resolves from a
/// trusted job id ([`IRIS_API_OWNED_OPTION_KEYS`]).
pub const IRIS_CLIENT_OPTION_KEYS: [&str; 33] = [
    "init",
    "prediction",
    "flow_shift",
    "shift_law",
    "shift_base_tokens",
    "logit_mean",
    "logit_std",
    "x_pred_sigma_min",
    "text_dropout",
    "gradient_clip",
    "betas",
    "muon_momentum",
    "muon_nesterov",
    "muon_adjust_lr",
    "auto_lr",
    "base_batch_size",
    "ema_enabled",
    "ema_decay",
    "num_epochs",
    "nan_loss_tolerance",
    "caption_field",
    "caption_fields",
    "on_caption_overflow",
    "text_conditioning",
    "keep_last_checkpoints",
    "milestone_steps",
    "resume_data_policy",
    "override_lr_on_resume",
    "preview_weights",
    "preview_negative_prompt",
    "export_weights",
    "export_dtype",
    "upstream_validation_prompts",
];

/// Engine options that carry an on-disk path. Never client-authored: the API derives them from a
/// previous job ([`RESUME_FROM_JOB_KEY`] / [`LOAD_FROM_JOB_KEY`]) after validating it.
pub const IRIS_API_OWNED_OPTION_KEYS: [&str; 2] = ["resume_from", "load_from"];

/// Whether `kernel` is an Iris trainer (generation, depth or restoration — see [`IrisObjective`]).
pub fn is_iris_training_kernel(kernel: &str) -> bool {
    kernel == IRIS_TRAINING_KERNEL
}

/// What an Iris training run trains, keyed on its target's base model (the task install).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IrisObjective {
    /// `iris_3b` — LoRA / LoKr adapters and full fine-tunes of the generation backbone.
    Generation,
    /// `iris_3b_depth` — the supervised depth fine-tune (a `depth/` export).
    Depth,
    /// `iris_3b_restore` — the adversarial restoration fine-tune (an `upscaler/` export).
    Restoration,
}

impl IrisObjective {
    /// The objective of an Iris target / plan base model, `None` for any other model.
    pub fn from_base_model(base_model: &str) -> Option<Self> {
        match base_model {
            IRIS_BASE_MODEL => Some(Self::Generation),
            depth::IRIS_DEPTH_BASE_MODEL => Some(Self::Depth),
            restore::IRIS_RESTORE_BASE_MODEL => Some(Self::Restoration),
            _ => None,
        }
    }

    /// The task identity its artifacts carry (`irisTask`).
    pub fn task(self) -> &'static str {
        match self {
            Self::Generation => IRIS_GENERATION_TASK,
            Self::Depth => depth::IRIS_DEPTH_TASK,
            Self::Restoration => restore::IRIS_RESTORATION_TASK,
        }
    }

    /// The catalog model (engine trainer id) it trains.
    pub fn base_model(self) -> &'static str {
        match self {
            Self::Generation => IRIS_BASE_MODEL,
            Self::Depth => depth::IRIS_DEPTH_BASE_MODEL,
            Self::Restoration => restore::IRIS_RESTORE_BASE_MODEL,
        }
    }

    /// The `TrainingConfig.advanced` key of its engine options bag.
    pub fn options_key(self) -> &'static str {
        match self {
            Self::Generation | Self::Depth => IRIS_OPTIONS_KEY,
            Self::Restoration => restore::RESTORE_OPTIONS_KEY,
        }
    }
}

/// The objective of an Iris training target, `None` for a non-Iris target.
pub fn iris_objective(target: &TrainingTarget) -> Option<IrisObjective> {
    if !is_iris_training_kernel(&target.kernel) {
        return None;
    }
    IrisObjective::from_base_model(&target.base_model)
}

/// The declared floor (GB) of an Iris run of `objective` with `config` on `lane`, or `None` when no
/// floor is declared for that shape (callers refuse rather than admit unbounded).
pub fn iris_objective_memory_floor_gb(
    objective: IrisObjective,
    lane: &str,
    config: &TrainingConfig,
) -> Option<u32> {
    match objective {
        IrisObjective::Generation => iris_training_memory_floor_gb(
            lane,
            advanced_string(config, "networkType").unwrap_or("lora"),
            config.resolution,
        ),
        IrisObjective::Depth => depth::depth_memory_floor_gb(lane, config),
        IrisObjective::Restoration => restore::restore_memory_floor_gb(lane, config),
    }
}

/// A short description of the run shape a floor was declared for (refusal messages).
pub fn iris_objective_shape(objective: IrisObjective, config: &TrainingConfig) -> String {
    let network = advanced_string(config, "networkType").unwrap_or(match objective {
        IrisObjective::Generation => "lora",
        _ => "full",
    });
    let artifact = if network.eq_ignore_ascii_case("full") {
        "a full fine-tune"
    } else {
        "an adapter (LoRA/LoKr) run"
    };
    match objective {
        IrisObjective::Generation => format!("{artifact} at {}px", config.resolution),
        IrisObjective::Depth => {
            let (height, width) = depth::depth_training_size(config);
            format!(
                "a depth full fine-tune at {height}×{width}, batch {}",
                config.batch_size.max(1)
            )
        }
        IrisObjective::Restoration => format!(
            "a restoration {} at a {}² crop, batch {}",
            artifact.trim_start_matches("a ").trim_start_matches("an "),
            config.resolution,
            config.batch_size.max(1)
        ),
    }
}

/// Declared (NOT measured) memory floors, GB, per `(lane, artifact, resolution)`.
///
/// Evidence (inference real-weight runs on the 3B backbone, 256², batch 1, bf16): MLX LoRA peak
/// 15.3 GB; Candle LoRA peak 21.8 GB. Each 256 floor is that peak rounded up (Candle adds the 2 GB
/// allocator reserve every padded Candle floor folds in). Nothing larger has been run, so the 512 /
/// 1024 rows scale the activation term conservatively, and a FULL run holds the f32 masters, the
/// AdamW moments, the gradients and the EMA of every 3B weight (~60 GB of state before activations)
/// — declared, honestly, far above any adapter run. S6 (sc-25684) replaces these with measured
/// values; until then a machine under the floor is refused before load, never admitted unbounded.
const IRIS_MEMORY_FLOORS_GB: [(&str, &str, u32, u32); 12] = [
    ("mlx", "adapter", 256, 16),
    ("mlx", "adapter", 512, 24),
    ("mlx", "adapter", 1024, 40),
    ("mlx", "full", 256, 80),
    ("mlx", "full", 512, 88),
    ("mlx", "full", 1024, 104),
    ("candle", "adapter", 256, 24),
    ("candle", "adapter", 512, 32),
    ("candle", "adapter", 1024, 48),
    ("candle", "full", 256, 80),
    ("candle", "full", 512, 88),
    ("candle", "full", 1024, 104),
];

/// The declared memory floor (GB) an Iris run of `network_type` at `resolution` needs on `lane`
/// (`"mlx"` unified memory, `"candle"` device VRAM). A resolution between two declared rows takes
/// the next row up (conservative); `None` above the largest declared row or for an unknown lane —
/// which callers treat as a refusal, never as "unbounded".
pub fn iris_training_memory_floor_gb(
    lane: &str,
    network_type: &str,
    resolution: u32,
) -> Option<u32> {
    let artifact = if network_type.trim().eq_ignore_ascii_case("full") {
        "full"
    } else {
        "adapter"
    };
    IRIS_MEMORY_FLOORS_GB
        .iter()
        .filter(|(row_lane, row_artifact, row_resolution, _)| {
            *row_lane == lane && *row_artifact == artifact && *row_resolution >= resolution
        })
        .min_by_key(|(_, _, row_resolution, _)| *row_resolution)
        .map(|(_, _, _, floor)| *floor)
}

/// The floor table as published on the target (`memoryFloors`), so the Studio can say what a run
/// needs before launch.
fn memory_floors_json() -> Value {
    let mut lanes = Map::new();
    for (lane, artifact, resolution, floor) in IRIS_MEMORY_FLOORS_GB {
        let lane_entry = lanes
            .entry(lane.to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        let artifacts = lane_entry.as_object_mut().expect("lane entry is an object");
        let rows = artifacts
            .entry(artifact.to_owned())
            .or_insert_with(|| Value::Object(Map::new()));
        rows.as_object_mut()
            .expect("artifact rows are an object")
            .insert(resolution.to_string(), json!(floor));
    }
    json!({
        "measured": false,
        "unit": "GB",
        "basis": "Declared floors (no measurement campaign): MLX LoRA 256px peak 15.3 GB and Candle \
                  LoRA 256px peak 21.8 GB on the real 3B backbone; larger resolutions and full \
                  fine-tuning are conservative declarations. An MLX floor is judged against unified \
                  memory minus the macOS/app reserve the MLX generation gate holds out. A machine \
                  under its floor is refused before the model loads.",
        "lanes": Value::Object(lanes),
    })
}

/// The Iris-3B generation training target: random-init / weights-init FULL training and LoRA /
/// LoKr adapters over the frozen upstream objective, on the MLX and Candle trainers.
pub fn iris_generation_target() -> TrainingTarget {
    let mut extra = ExtraFields::new();
    extra.insert("memoryFloors".to_owned(), memory_floors_json());
    TrainingTarget {
        id: IRIS_TRAINING_TARGET_ID.to_owned(),
        name: "Iris 3B".to_owned(),
        modality: TrainingModality::Image,
        // `networkType: "full"` turns a run into a base checkpoint (`training_output_kind`).
        output_kind: TrainingOutputKind::Lora,
        family: IRIS_FAMILY.to_owned(),
        base_model: IRIS_BASE_MODEL.to_owned(),
        base_model_repo: Some(IRIS_BASE_MODEL_REPO.to_owned()),
        kernel: IRIS_TRAINING_KERNEL.to_owned(),
        defaults: TrainingConfig {
            rank: 16,
            alpha: 16,
            learning_rate: ContractNumber::from_f64(0.0001).expect("0.0001 is finite"),
            // MICRO-steps (the shared unit); all four cadences are multiples of the default
            // accumulation of 1.
            steps: 1000,
            batch_size: 1,
            gradient_accumulation: 1,
            // Upstream's fixed-square `data.image_size`; 256 is the release's first stage and the
            // resolution every real-weight training run so far used.
            resolution: 256,
            save_every: 250,
            seed: 42,
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                // bf16 autocast over f32 masters (upstream `mixed_precision: bf16`); `f32` is `no`.
                "mixedPrecision": "bf16",
                "networkType": "lora",
                "decomposeFactor": -1,
                // The backbone's own `flow.timestep_sampler` (logit-normal).
                "timestepType": "logit_normal",
                // The only bias/loss the objective has; pinned so the generic controls cannot
                // offer a refused value.
                "timestepBias": "balanced",
                "lossType": "mse",
                "weightDecay": 0.0,
                "lrScheduler": "constant",
                "lrWarmupSteps": 0,
                // Not implemented by the native trainer (it refuses `gradient_checkpointing`).
                "gradientCheckpointing": false,
                "sampleEvery": 250,
                "sampleSteps": 30,
                "sampleGuidanceScale": 3.0,
                "sampleCount": IRIS_PREVIEW_PROMPT_CAP,
                "qualityPreset": "balanced",
                "outputScope": "project",
                "requestedGpu": "auto",
                // The engine `model_options["iris"]` bag. Values are upstream's defaults except
                // `text_conditioning` (bit-identical to on-the-fly, but frees the 8.9 GB encoder
                // before the optimizer state is built) and `keep_last_checkpoints` (a full-model
                // checkpoint is ~48 GB, so keep-all would fill a disk).
                "iris": {
                    "init": "weights",
                    "text_conditioning": "cached",
                    "text_dropout": 0.1,
                    "gradient_clip": 0.5,
                    "ema_enabled": true,
                    "ema_decay": 0.9999,
                    "on_caption_overflow": "warn",
                    "caption_field": "caption",
                    "caption_fields": [],
                    "keep_last_checkpoints": 2,
                    "milestone_steps": [],
                    "nan_loss_tolerance": 20,
                    "betas": [0.9, 0.95],
                    "auto_lr": "none",
                    "base_batch_size": 256,
                    "preview_negative_prompt": "",
                    "upstream_validation_prompts": false,
                    "export_dtype": "fp32"
                }
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "rank": [1, 128],
            "alpha": [1, 128],
            "steps": [1, 200000],
            "batchSize": [1, 64],
            // Multiples of the backbone's patch size (16); never silently changed by the trainer.
            "resolutions": [256, 512, 1024],
            // The trainer renders at most four preview prompts per cadence.
            "sampleCount": [1, IRIS_PREVIEW_PROMPT_CAP],
            "optimizers": ["adamw", "muon"],
            "networkTypes": ["lora", "lokr", "full"],
            "lrSchedulers": ["constant", "cosine"],
            "timestepTypes": ["logit_normal", "uniform"],
            "timestepBiases": ["balanced"],
            "lossTypes": ["mse"],
            "mixedPrecisions": ["bf16", "f32"],
            "gradientCheckpointing": false,
            "microStepFields": ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
            "inits": ["weights", "random"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Iris 3B",
            "description": "Train Iris-3B generation: a LoRA or LoKr adapter, or a full fine-tune from \
                            the released weights or from random init. Pixel-space flow matching on \
                            the frozen Qwen3-VL text encoder, on Apple Silicon (MLX) and NVIDIA \
                            (Candle/CUDA).",
            "recommendedFor": ["character", "style", "concept"],
            "datasetModality": "image"
        })),
        extra,
    }
}

/// Every micro-step field of `fields` present in `config` must be a whole number of `multiple`
/// micro-steps (an optimizer-step boundary — or, for restoration, a generator + critic step).
fn validate_micro_steps(
    config: &TrainingConfig,
    multiple: u64,
    fields: &[(&str, bool)],
) -> Result<(), TrainingPlanError> {
    for (field, top_level) in fields {
        let value = if *top_level {
            Some(u64::from(match *field {
                "steps" => config.steps,
                "saveEvery" => config.save_every,
                _ => unreachable!("only steps/saveEvery are top-level micro-step fields"),
            }))
        } else {
            advanced_u64(config, field)?
        };
        if let Some(value) = value {
            if !value.is_multiple_of(multiple) {
                return Err(TrainingPlanError::TargetLimit(
                    TrainingTargetLimitError::NotMultipleOf {
                        field: (*field).to_owned(),
                        value,
                        multiple,
                    },
                ));
            }
        }
    }
    Ok(())
}

/// The per-objective dataset contract: generation items answer the caption selection; depth items
/// are aligned examples; restoration items are all pairs or all clean images.
pub fn validate_iris_dataset(
    target: &TrainingTarget,
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    match IrisObjective::from_base_model(&target.base_model) {
        Some(IrisObjective::Depth) => depth::validate_depth_dataset(dataset),
        Some(IrisObjective::Restoration) => restore::validate_restore_dataset(config, dataset),
        Some(IrisObjective::Generation) | None => validate_iris_dataset_captions(config, dataset),
    }
}

/// The store-boundary check of an item's Iris extras: named captions, a depth target, a degraded
/// input.
pub fn validate_item_iris_extras(extra: &ExtraFields) -> Result<(), String> {
    validate_item_captions(extra)?;
    depth::validate_item_depth(extra)?;
    restore::validate_item_degraded(extra)
}

fn invalid(detail: impl Into<String>) -> TrainingPlanError {
    TrainingPlanError::InvalidConfig(detail.into())
}

fn advanced_u64(config: &TrainingConfig, key: &str) -> Result<Option<u64>, TrainingPlanError> {
    match config.advanced.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| invalid(format!("{key} must be a non-negative integer."))),
    }
}

fn advanced_string<'a>(config: &'a TrainingConfig, key: &str) -> Option<&'a str> {
    config
        .advanced
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

/// The Iris options bag of a config (`advanced.iris`), or an empty map when absent.
pub fn iris_options(config: &TrainingConfig) -> Result<Map<String, Value>, TrainingPlanError> {
    match config.advanced.get(IRIS_OPTIONS_KEY) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map.clone()),
        Some(_) => Err(invalid(
            "advanced.iris must be an object of Iris training options.",
        )),
    }
}

fn one_of(key: &str, value: &Value, allowed: &[&str]) -> Result<(), TrainingPlanError> {
    let text = value
        .as_str()
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be a string.")))?;
    if allowed.contains(&text) {
        Ok(())
    } else {
        Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::UnsupportedValue {
                field: format!("iris.{key}"),
                value: text.to_owned(),
                allowed: allowed.iter().map(|value| (*value).to_owned()).collect(),
            },
        ))
    }
}

fn finite_in(
    key: &str,
    value: &Value,
    accept: impl Fn(f64) -> bool,
    expect: &str,
) -> Result<f64, TrainingPlanError> {
    let number = value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be a finite number.")))?;
    if accept(number) {
        Ok(number)
    } else {
        Err(invalid(format!("advanced.iris.{key} must be {expect}.")))
    }
}

fn whole(key: &str, value: &Value, minimum: u64) -> Result<u64, TrainingPlanError> {
    let number = value.as_u64().ok_or_else(|| {
        invalid(format!(
            "advanced.iris.{key} must be a non-negative integer."
        ))
    })?;
    if number < minimum {
        return Err(invalid(format!(
            "advanced.iris.{key} must be at least {minimum}."
        )));
    }
    Ok(number)
}

fn boolean(key: &str, value: &Value) -> Result<bool, TrainingPlanError> {
    value
        .as_bool()
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be true or false.")))
}

fn string_list(key: &str, value: &Value) -> Result<Vec<String>, TrainingPlanError> {
    value
        .as_array()
        .and_then(|values| {
            values
                .iter()
                .map(|value| value.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| invalid(format!("advanced.iris.{key} must be a list of strings.")))
}

/// Validate one `advanced.iris` value against the engine's accepted domain.
fn validate_option(key: &str, value: &Value) -> Result<(), TrainingPlanError> {
    match key {
        "init" => one_of(key, value, &["weights", "random"]),
        "prediction" => one_of(key, value, &["v", "x"]),
        "shift_law" => one_of(key, value, &["none", "sd3", "flux"]),
        "muon_adjust_lr" => one_of(key, value, &["rms_norm", "spectral_norm", "none"]),
        "auto_lr" => one_of(key, value, &["none", "sqrt", "linear"]),
        "on_caption_overflow" => one_of(key, value, &["warn", "error", "silent"]),
        "text_conditioning" => one_of(key, value, &["on_the_fly", "cached"]),
        "resume_data_policy" => one_of(key, value, &["exact", "new_phase"]),
        "preview_weights" | "export_weights" => one_of(key, value, &["raw", "ema"]),
        "export_dtype" => one_of(key, value, &["fp32", "bf16"]),
        "flow_shift" => finite_in(key, value, |x| x > 0.0, "greater than 0").map(drop),
        "logit_mean" | "logit_std" | "x_pred_sigma_min" => {
            finite_in(key, value, |_| true, "a number").map(drop)
        }
        "text_dropout" => {
            finite_in(key, value, |x| (0.0..=1.0).contains(&x), "between 0 and 1").map(drop)
        }
        "gradient_clip" => finite_in(key, value, |x| x > 0.0, "greater than 0").map(drop),
        "ema_decay" => {
            finite_in(key, value, |x| (0.0..=1.0).contains(&x), "between 0 and 1").map(drop)
        }
        "muon_momentum" => {
            finite_in(key, value, |x| (0.0..1.0).contains(&x), "in [0, 1)").map(drop)
        }
        "shift_base_tokens" | "base_batch_size" | "num_epochs" => whole(key, value, 1).map(drop),
        "nan_loss_tolerance" | "keep_last_checkpoints" => whole(key, value, 0).map(drop),
        "muon_nesterov"
        | "ema_enabled"
        | "override_lr_on_resume"
        | "upstream_validation_prompts" => boolean(key, value).map(drop),
        "caption_field" => {
            let field = value
                .as_str()
                .map(str::trim)
                .filter(|field| !field.is_empty())
                .ok_or_else(|| {
                    invalid("advanced.iris.caption_field must be a non-empty string.")
                })?;
            if field.len() > 64 {
                return Err(invalid(
                    "advanced.iris.caption_field is longer than 64 characters.",
                ));
            }
            Ok(())
        }
        "caption_fields" => {
            let fields = string_list(key, value)?;
            if fields
                .iter()
                .any(|field| field.trim().is_empty() || field.len() > 64)
            {
                return Err(invalid(
                    "advanced.iris.caption_fields entries must be non-empty names of at most 64 \
                     characters.",
                ));
            }
            Ok(())
        }
        "preview_negative_prompt" => value
            .as_str()
            .map(drop)
            .ok_or_else(|| invalid("advanced.iris.preview_negative_prompt must be a string.")),
        "milestone_steps" => value
            .as_array()
            .filter(|steps| {
                steps
                    .iter()
                    .all(|step| step.as_u64().is_some_and(|s| s > 0))
            })
            .map(drop)
            .ok_or_else(|| {
                invalid("advanced.iris.milestone_steps must be a list of positive optimizer steps.")
            }),
        "betas" => {
            let pair = value
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| invalid("advanced.iris.betas must be a pair [beta1, beta2]."))?;
            if pair.iter().all(|beta| {
                beta.as_f64()
                    .is_some_and(|beta| beta.is_finite() && (0.0..1.0).contains(&beta))
            }) {
                Ok(())
            } else {
                Err(invalid("advanced.iris.betas must both be in [0, 1)."))
            }
        }
        _ => Err(invalid(format!(
            "advanced.iris.{key} is not an Iris training option."
        ))),
    }
}

/// Submit-time validation of an Iris training config: the target-local refusals the trainer would
/// otherwise raise only after loading the weights, as typed 400s.
pub fn validate_iris_training_config(
    target: &TrainingTarget,
    config: &TrainingConfig,
) -> Result<(), TrainingPlanError> {
    match IrisObjective::from_base_model(&target.base_model) {
        Some(IrisObjective::Depth) => return depth::validate_depth_config(config),
        Some(IrisObjective::Restoration) => return restore::validate_restore_config(config),
        Some(IrisObjective::Generation) | None => {}
    }
    // ---- micro-step multiples -----------------------------------------------------------------
    validate_micro_steps(
        config,
        u64::from(config.gradient_accumulation.max(1)),
        &IRIS_MICRO_STEP_FIELDS,
    )?;

    // ---- the options bag ----------------------------------------------------------------------
    let options = iris_options(config)?;
    for (key, value) in &options {
        if IRIS_API_OWNED_OPTION_KEYS.contains(&key.as_str()) {
            return Err(invalid(format!(
                "advanced.iris.{key} is set by SceneWorks from a previous run; name the run with \
                 advanced.{} instead.",
                if key == "resume_from" {
                    RESUME_FROM_JOB_KEY
                } else {
                    LOAD_FROM_JOB_KEY
                }
            )));
        }
        if !IRIS_CLIENT_OPTION_KEYS.contains(&key.as_str()) {
            return Err(TrainingPlanError::TargetLimit(
                TrainingTargetLimitError::UnsupportedValue {
                    field: "iris".to_owned(),
                    value: key.clone(),
                    allowed: IRIS_CLIENT_OPTION_KEYS
                        .iter()
                        .map(|key| (*key).to_owned())
                        .collect(),
                },
            ));
        }
        if !value.is_null() {
            validate_option(key, value)?;
        }
    }

    let network_type = advanced_string(config, "networkType").unwrap_or("lora");
    let full = network_type.eq_ignore_ascii_case("full");
    let init = options
        .get("init")
        .and_then(Value::as_str)
        .unwrap_or("weights");
    if init == "random" && !full {
        return Err(invalid(
            "Random initialization trains a full model; an Iris adapter (LoRA/LoKr) trains on the \
             released weights. Choose the full fine-tune network type or weights init.",
        ));
    }
    if config.advanced.contains_key(LOAD_FROM_JOB_KEY) {
        if !full {
            return Err(invalid(
                "Starting from a previous run's weights (loadFromJobId) applies to a full \
                 fine-tune only.",
            ));
        }
        if init == "random" {
            return Err(invalid(
                "loadFromJobId starts from a previous run's weights; it cannot be combined with \
                 random initialization.",
            ));
        }
    }
    if config.advanced.contains_key(LOAD_FROM_JOB_KEY)
        && config.advanced.contains_key(RESUME_FROM_JOB_KEY)
    {
        return Err(invalid(
            "Resume continues a run's full state; loadFromJobId starts a fresh run from its \
             weights. Choose one.",
        ));
    }
    let ema_enabled = options
        .get("ema_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    for key in ["preview_weights", "export_weights"] {
        if options.get(key).and_then(Value::as_str) == Some("ema") && !ema_enabled {
            return Err(invalid(format!(
                "advanced.iris.{key} = ema needs the EMA enabled (advanced.iris.ema_enabled)."
            )));
        }
    }

    // ---- previews -----------------------------------------------------------------------------
    let upstream_prompts = options
        .get("upstream_validation_prompts")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let prompt_count = config
        .advanced
        .get("samplePrompts")
        .and_then(Value::as_array)
        .map_or(0, |prompts| {
            prompts
                .iter()
                .filter(|prompt| {
                    prompt
                        .as_str()
                        .is_some_and(|prompt| !prompt.trim().is_empty())
                })
                .count()
        });
    if upstream_prompts && prompt_count > 0 {
        return Err(invalid(
            "Upstream validation prompts render Iris' seven default prompts in place of your own \
             sample prompts — clear the sample prompts or turn the upstream prompts off.",
        ));
    }
    let sample_every = advanced_u64(config, "sampleEvery")?.unwrap_or(0);
    let sample_steps = advanced_u64(config, "sampleSteps")?.unwrap_or(20);
    if sample_every > 0 && (prompt_count > 0 || upstream_prompts) && sample_steps == 0 {
        return Err(invalid(
            "sampleSteps must be at least 1 when previews are on.",
        ));
    }

    // ---- the resume knobs only mean something on a resume --------------------------------------
    if !config.advanced.contains_key(RESUME_FROM_JOB_KEY) {
        for key in ["resume_data_policy", "override_lr_on_resume"] {
            if options.get(key).is_some_and(|value| !value.is_null()) {
                return Err(invalid(format!(
                    "advanced.iris.{key} applies only when resuming a previous run \
                     (advanced.{RESUME_FROM_JOB_KEY})."
                )));
            }
        }
    }
    let _ = target;
    Ok(())
}

/// Validate a dataset item's per-item named captions (`extra.captions = {field: text}`).
pub fn validate_item_captions(extra: &ExtraFields) -> Result<(), String> {
    let Some(value) = extra.get(ITEM_CAPTIONS_KEY) else {
        return Ok(());
    };
    let map = value
        .as_object()
        .ok_or_else(|| "Item captions must be an object of {field: caption text}.".to_owned())?;
    for (field, text) in map {
        let trimmed = field.trim();
        if trimmed.is_empty() || trimmed != field || field.len() > 64 {
            return Err(format!(
                "Item caption field names must be non-empty, untrimmed-whitespace-free, and at most \
                 64 characters (got {field:?})."
            ));
        }
        if !text.is_string() {
            return Err(format!("Item caption field {field:?} must be text."));
        }
    }
    Ok(())
}

fn item_has_caption_field(item: &super::TrainingDatasetItem, field: &str) -> bool {
    let named = item
        .extra
        .get(ITEM_CAPTIONS_KEY)
        .and_then(Value::as_object)
        .and_then(|captions| captions.get(field))
        .and_then(Value::as_str)
        .is_some_and(|text| !text.trim().is_empty());
    named || (field == "caption" && !item.caption.text.trim().is_empty())
}

/// Every item must be able to answer the run's caption selection, or the trainer would stop
/// mid-epoch on the first item without it (`select_caption`). With `caption_fields` set an item needs
/// one of them OR the `caption_field` fallback; otherwise it needs `caption_field` itself
/// (`"caption"` reads the item's ordinary caption).
pub fn validate_iris_dataset_captions(
    config: &TrainingConfig,
    dataset: &TrainingDataset,
) -> Result<(), TrainingPlanError> {
    for item in &dataset.items {
        validate_item_captions(&item.extra).map_err(invalid)?;
    }
    let options = iris_options(config)?;
    let caption_field = options
        .get("caption_field")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .unwrap_or("caption");
    let caption_fields: Vec<String> = options
        .get("caption_fields")
        .and_then(Value::as_array)
        .map(|fields| {
            fields
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    let missing: Vec<&str> = dataset
        .items
        .iter()
        .filter(|item| {
            let any_listed = caption_fields
                .iter()
                .any(|field| item_has_caption_field(item, field));
            !(any_listed || item_has_caption_field(item, caption_field))
        })
        .map(|item| item.display_name.as_str())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    let wanted = if caption_fields.is_empty() {
        format!("caption field '{caption_field}'")
    } else {
        format!(
            "any of caption fields {:?} (or the fallback '{caption_field}')",
            caption_fields
        )
    };
    Err(invalid(format!(
        "{} dataset item(s) have no {wanted}: {}{}. Add the caption or pick a field every item \
         carries.",
        missing.len(),
        missing
            .iter()
            .take(5)
            .copied()
            .collect::<Vec<_>>()
            .join(", "),
        if missing.len() > 5 { ", …" } else { "" }
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_client_allow_list_is_the_engine_list_minus_the_path_options() {
        // The engine's `OPTION_KEYS` (33) + `OPTION_KEYS_EXTRA` (2) = 35 keys; SceneWorks owns the
        // two path keys, the client may set the other 33 — no more, no fewer.
        assert_eq!(
            IRIS_CLIENT_OPTION_KEYS.len() + IRIS_API_OWNED_OPTION_KEYS.len(),
            35
        );
        for key in IRIS_API_OWNED_OPTION_KEYS {
            assert!(!IRIS_CLIENT_OPTION_KEYS.contains(&key));
        }
        let mut sorted = IRIS_CLIENT_OPTION_KEYS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), IRIS_CLIENT_OPTION_KEYS.len(), "no duplicates");
    }

    #[test]
    fn memory_floors_are_declared_for_every_offered_lane_artifact_and_resolution() {
        let target = iris_generation_target();
        let resolutions: Vec<u32> = target.limits["resolutions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect();
        for lane in ["mlx", "candle"] {
            for network in ["lora", "lokr", "full"] {
                for resolution in &resolutions {
                    assert!(
                        iris_training_memory_floor_gb(lane, network, *resolution).is_some(),
                        "{lane}/{network}/{resolution} has no declared floor"
                    );
                }
            }
        }
        // The evidence-backed rows, and full far above adapters.
        assert_eq!(iris_training_memory_floor_gb("mlx", "lora", 256), Some(16));
        assert_eq!(
            iris_training_memory_floor_gb("candle", "lokr", 256),
            Some(24)
        );
        assert!(
            iris_training_memory_floor_gb("mlx", "full", 256).unwrap()
                > iris_training_memory_floor_gb("mlx", "lora", 1024).unwrap()
        );
        // Between rows takes the next row up; beyond the table and unknown lanes are refusals.
        assert_eq!(iris_training_memory_floor_gb("mlx", "lora", 384), Some(24));
        assert_eq!(iris_training_memory_floor_gb("mlx", "lora", 2048), None);
        assert_eq!(iris_training_memory_floor_gb("cpu", "lora", 256), None);
        assert_eq!(
            target.extra["memoryFloors"]["lanes"]["candle"]["adapter"]["256"],
            json!(24)
        );
        assert_eq!(target.extra["memoryFloors"]["measured"], json!(false));
    }
}
