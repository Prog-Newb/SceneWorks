//! Iris-3B DEPTH training (epic sc-25678, S9 sc-25687) — the SceneWorks surface of the native
//! `iris_3b_depth` trainers (`mlx-gen-iris` / `candle-gen-iris`): the paper's supervised
//! direct-regression fine-tune (arXiv 2610.09450v1 §5.2) that produces a `depth/` export.
//!
//! The engine contract is `gen_core::iris::train::depth`: the shared `TrainingConfig` (a FULL
//! fine-tune, `loss_type = "l1"`) plus the `model_options["iris"]` bag whose unknown keys the trainer
//! refuses (`DEPTH_OPTION_KEYS`), and per-item `model_options["depth"] = {path, mask_path?, scale?,
//! min_depth?, max_depth?}`. This module owns the SceneWorks half so the Training Studio offers
//! exactly what the trainer accepts:
//!
//! * [`iris_depth_target`] — the target, defaulting to the paper recipe (AdamW lr 3e-5, weight decay
//!   0.01, constant LR, no warmup, clip 1.0, β (0.9, 0.999), raw-weights export, 10K optimizer steps
//!   at a global batch of 32). The global batch is offered as `batchSize 1 × gradientAccumulation 32`
//!   — the same 32 examples per optimizer step that fits a single machine (every micro-batch
//!   normalizes its own valid pixels; a larger `batchSize` is a submit away);
//! * [`validate_depth_config`] — the submit-time twin of `DepthTrainPlan::resolve`'s request-only
//!   refusals;
//! * [`validate_depth_dataset`] — every item is an aligned example (`extra.irisDepth`), the sidecar
//!   files live under the dataset root;
//! * the declared memory floors ([`depth_memory_floor_gb`]).
//!
//! Where the run starts (`advanced.depthInit`): `export` (the installed `depth/` export — the
//! release's own depth model, fine-tuned further; no text encoder) or `parent` (the paper's start:
//! the GENERATION backbone, input projections widened, conditioned on the empty prompt — read from
//! the installed depth export's `empty_prompt.safetensors` when present, else encoded ONCE by the
//! generation install's text encoder).

use serde_json::{json, Map, Value};

use super::super::{
    object, ContractNumber, ExtraFields, TrainingConfig, TrainingDataset, TrainingModality,
    TrainingOutputKind, TrainingPlanError, TrainingTarget, TrainingTargetLimitError,
};
use super::{
    boolean, finite_in, invalid, iris_options, one_of, validate_micro_steps, whole,
    IRIS_BASE_MODEL_REPO, IRIS_FAMILY, IRIS_OPTIONS_KEY, IRIS_TRAINING_KERNEL, LOAD_FROM_JOB_KEY,
    RESUME_FROM_JOB_KEY,
};
use crate::store_util::is_safe_relative_path;

/// The depth training target id (the same id as the engine trainer and the depth task install).
pub const IRIS_DEPTH_TARGET_ID: &str = "iris_3b_depth";
/// The catalog model (and engine trainer id) a depth run trains: the depth task install.
pub const IRIS_DEPTH_BASE_MODEL: &str = "iris_3b_depth";
/// The task a depth artifact belongs to (the engine's `irisTask` stamp).
pub const IRIS_DEPTH_TASK: &str = "depth";
/// `TrainingConfig.advanced` key choosing where a depth run starts: `export` | `parent`.
pub const DEPTH_INIT_KEY: &str = "depthInit";
pub const DEPTH_INIT_EXPORT: &str = "export";
pub const DEPTH_INIT_PARENT: &str = "parent";
/// `TrainingDatasetItem.extra` key carrying an item's aligned depth target
/// (`{path, maskPath?, scale?, minDepth?, maxDepth?}`, dataset-relative paths). The worker resolves
/// it into the engine's `model_options["depth"]`.
pub const ITEM_DEPTH_KEY: &str = "irisDepth";
/// The paper's Hypersim native training size `[height, width]`.
pub const DEFAULT_DEPTH_SIZE: [u32; 2] = [480, 640];
/// Training-size bounds (each side a multiple of the backbone's 16-pixel patch).
pub const DEPTH_SIZE_MIN: u32 = 64;
pub const DEPTH_SIZE_MAX: u32 = 2048;
pub const DEPTH_SIZE_MULTIPLE: u32 = 16;
/// Paper §5.2: 10K optimizer steps at a global batch of 32 (offered as 1 × 32 accumulation).
pub const PAPER_OPTIMIZER_STEPS: u32 = 10_000;
pub const PAPER_GLOBAL_BATCH: u32 = 32;

/// Every engine depth option a client may set: the engine's `DEPTH_OPTION_KEYS` (32) minus the two
/// PATH options SceneWorks resolves from a trusted previous job.
pub const DEPTH_CLIENT_OPTION_KEYS: [&str; 30] = [
    "depth_size",
    "resize",
    "crop",
    "hflip_prob",
    "gradient_weight",
    "low_percentile",
    "high_percentile",
    "clip_target",
    "loss_reduction",
    "invalid_examples",
    "min_valid_pixels",
    "reducer_init",
    "betas",
    "gradient_clip",
    "muon_momentum",
    "muon_nesterov",
    "muon_adjust_lr",
    "auto_lr",
    "base_batch_size",
    "ema_enabled",
    "ema_decay",
    "num_epochs",
    "nan_loss_tolerance",
    "keep_last_checkpoints",
    "milestone_steps",
    "resume_data_policy",
    "override_lr_on_resume",
    "preview_items",
    "preview_weights",
    "export_weights",
];

/// The depth objective's micro-step fields (whole `gradientAccumulation` windows).
pub const DEPTH_MICRO_STEP_FIELDS: [(&str, bool); 4] = [
    ("steps", true),
    ("saveEvery", true),
    ("sampleEvery", false),
    ("lrWarmupSteps", false),
];

/// Advanced keys of the generic Studio a depth run has no use for: the depth previews predict
/// dataset items (no prompt, no sampler), and the run is a full fine-tune (no adapter targets).
const UNREAD_ADVANCED_KEYS: [&str; 7] = [
    "samplePrompts",
    "sampleSteps",
    "sampleGuidanceScale",
    "sampleCount",
    "loraTargetModules",
    "timestepType",
    "timestepBias",
];

/// Declared (NOT measured) floors, GB, per `(lane, micro-batch pixels ≤, floor)`. A micro-batch's
/// pixels are `batchSize × height × width` of the training size.
///
/// Evidence (inference real-weight full fine-tunes of the 3B depth model, bf16 autocast over f32
/// masters, AdamW, batch 1 at 128×128): MLX peak phys_footprint 59.0 GB (nax-macos-2, run
/// 38136750391); Candle/CUDA VRAM peak 64.6 GB (RTX PRO 6000, run 38133787177). Nearly all of it is
/// the full-model state (f32 masters + grads + AdamW moments), so the 16 384-pixel rows are that peak
/// rounded up (Candle adding the 2 GB allocator reserve); larger micro-batches scale the activation
/// term conservatively. Nothing larger has been run, so a micro-batch above the last row is refused
/// rather than admitted unbounded. S6 (sc-25684) replaces these with measured values.
const DEPTH_FLOORS_GB: [(&str, u64, u32); 8] = [
    ("mlx", 16_384, 64),
    ("mlx", 307_200, 80),
    ("mlx", 589_824, 96),
    ("mlx", 1_048_576, 112),
    ("candle", 16_384, 68),
    ("candle", 307_200, 84),
    ("candle", 589_824, 100),
    ("candle", 1_048_576, 116),
];

/// The training size `[height, width]` a depth config runs at: `iris.depth_size`, else the square
/// `resolution`.
pub fn depth_training_size(config: &TrainingConfig) -> (u32, u32) {
    let size = config
        .advanced
        .get(IRIS_OPTIONS_KEY)
        .and_then(|options| options.get("depth_size"))
        .and_then(Value::as_array)
        .and_then(|pair| {
            let side = |value: &Value| value.as_u64().and_then(|value| u32::try_from(value).ok());
            match pair.as_slice() {
                [h, w] => Some((side(h)?, side(w)?)),
                _ => None,
            }
        });
    size.unwrap_or((config.resolution, config.resolution))
}

/// The pixels of one micro-batch (`batchSize × height × width`).
pub fn depth_micro_batch_pixels(config: &TrainingConfig) -> u64 {
    let (height, width) = depth_training_size(config);
    u64::from(config.batch_size.max(1)) * u64::from(height) * u64::from(width)
}

/// The declared floor (GB) of a depth run on `lane`, or `None` above the largest declared row /
/// for an unknown lane (callers refuse, never admit unbounded).
pub fn depth_memory_floor_gb(lane: &str, config: &TrainingConfig) -> Option<u32> {
    let pixels = depth_micro_batch_pixels(config);
    DEPTH_FLOORS_GB
        .iter()
        .filter(|(row_lane, row_pixels, _)| *row_lane == lane && *row_pixels >= pixels)
        .min_by_key(|(_, row_pixels, _)| *row_pixels)
        .map(|(_, _, floor)| *floor)
}

fn memory_floors_json() -> Value {
    let mut lanes = Map::new();
    for (lane, pixels, floor) in DEPTH_FLOORS_GB {
        lanes
            .entry(lane.to_owned())
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .expect("lane rows are an object")
            .insert(pixels.to_string(), json!(floor));
    }
    json!({
        "measured": false,
        "unit": "GB",
        "key": "microBatchPixels",
        "basis": "Declared floors (no measurement campaign) keyed by the pixels of one micro-batch \
                  (batch size × training height × width). Evidence: a full fine-tune of the 3B depth \
                  model at 128×128, batch 1, peaked at 59.0 GB on MLX and 64.6 GB on CUDA; larger \
                  micro-batches are conservative declarations. An MLX floor is judged against unified \
                  memory minus the macOS/app reserve. A machine under its floor is refused before the \
                  model loads, and a micro-batch above the last row is refused rather than admitted \
                  without a bound.",
        "lanes": Value::Object(lanes),
    })
}

/// The Iris-3B depth training target: a full fine-tune that writes a `depth/` export.
pub fn iris_depth_target() -> TrainingTarget {
    let mut extra = ExtraFields::new();
    extra.insert("memoryFloors".to_owned(), memory_floors_json());
    extra.insert("irisObjective".to_owned(), json!(IRIS_DEPTH_TASK));
    extra.insert(
        "depthTraining".to_owned(),
        json!({
            "sizeMin": DEPTH_SIZE_MIN,
            "sizeMax": DEPTH_SIZE_MAX,
            "sizeMultiple": DEPTH_SIZE_MULTIPLE,
            "inits": [DEPTH_INIT_EXPORT, DEPTH_INIT_PARENT],
            "depthFormats": ["npy", "png8", "png16"],
            "paperOptimizerSteps": PAPER_OPTIMIZER_STEPS,
            "paperGlobalBatch": PAPER_GLOBAL_BATCH,
        }),
    );
    let accumulation = PAPER_GLOBAL_BATCH;
    let cadence = 1_000 * accumulation;
    TrainingTarget {
        id: IRIS_DEPTH_TARGET_ID.to_owned(),
        name: "Iris 3B Depth".to_owned(),
        modality: TrainingModality::Image,
        // Always a full model (the `depth/` export) — a model-catalog entry, never an adapter.
        output_kind: TrainingOutputKind::BaseCheckpoint,
        family: IRIS_FAMILY.to_owned(),
        base_model: IRIS_DEPTH_BASE_MODEL.to_owned(),
        base_model_repo: Some(IRIS_BASE_MODEL_REPO.to_owned()),
        kernel: IRIS_TRAINING_KERNEL.to_owned(),
        defaults: TrainingConfig {
            // No adapter: rank/alpha are not read by a full fine-tune (the Studio hides them).
            rank: 1,
            alpha: 1,
            learning_rate: ContractNumber::from_f64(3e-5).expect("3e-5 is finite"),
            // MICRO-steps: the paper's 10K optimizer steps × 32 accumulation.
            steps: PAPER_OPTIMIZER_STEPS * accumulation,
            batch_size: 1,
            gradient_accumulation: accumulation,
            // The square fallback size; the run trains at `iris.depth_size`, whose larger side this
            // mirrors (validated).
            resolution: DEFAULT_DEPTH_SIZE[1],
            save_every: cadence,
            seed: 42,
            optimizer: "adamw".to_owned(),
            trigger_word: None,
            advanced: object(json!({
                "mixedPrecision": "bf16",
                "networkType": "full",
                "lossType": "l1",
                "weightDecay": 0.01,
                "lrScheduler": "constant",
                "lrWarmupSteps": 0,
                "gradientCheckpointing": false,
                "sampleEvery": cadence,
                "outputScope": "project",
                "requestedGpu": "auto",
                "depthInit": DEPTH_INIT_EXPORT,
                // The engine `model_options["iris"]` bag: the trainer's defaults (paper §5.2 where it
                // speaks) except `keep_last_checkpoints` (a full checkpoint is ~48 GB, so keep-all
                // would fill a disk).
                "iris": {
                    "depth_size": DEFAULT_DEPTH_SIZE,
                    "resize": "cover",
                    "crop": "center",
                    "hflip_prob": 0.5,
                    "gradient_weight": 5.0,
                    "low_percentile": 2.0,
                    "high_percentile": 98.0,
                    "clip_target": false,
                    "loss_reduction": "valid_pixels",
                    "invalid_examples": "reject",
                    "min_valid_pixels": 2,
                    "betas": [0.9, 0.999],
                    "gradient_clip": 1.0,
                    "ema_enabled": false,
                    "nan_loss_tolerance": 20,
                    "keep_last_checkpoints": 2,
                    "milestone_steps": [],
                    "preview_items": 4,
                    "preview_weights": "raw",
                    "export_weights": "raw"
                }
            })),
            extra: ExtraFields::new(),
        },
        limits: object(json!({
            "steps": [1, 3_200_000],
            "batchSize": [1, 64],
            "optimizers": ["adamw", "muon"],
            "networkTypes": ["full"],
            "lrSchedulers": ["constant", "cosine"],
            "lossTypes": ["l1"],
            "mixedPrecisions": ["bf16", "f32"],
            "gradientCheckpointing": false,
            "microStepFields": ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
            "outputScopes": ["project", "global"]
        })),
        ui: object(json!({
            "label": "Iris 3B Depth",
            "description": "Fine-tune Iris-3B monocular depth on aligned RGB + numerical depth (+ \
                            validity mask) examples — the paper's recipe (full fine-tune, masked L1 \
                            + gradient loss, AdamW 3e-5). Writes a depth model you can pick in the \
                            Image Editor and Library depth actions. Apple Silicon (MLX) and NVIDIA \
                            (Candle/CUDA).",
            "recommendedFor": ["depth"],
            "datasetModality": "image",
            "objective": IRIS_DEPTH_TASK
        })),
        extra,
    }
}

fn depth_size_value(value: &Value) -> Result<(u32, u32), TrainingPlanError> {
    let pair = value
        .as_array()
        .filter(|pair| pair.len() == 2)
        .ok_or_else(|| invalid("advanced.iris.depth_size must be [height, width]."))?;
    let side = |value: &Value| -> Result<u32, TrainingPlanError> {
        let side = value
            .as_u64()
            .and_then(|side| u32::try_from(side).ok())
            .ok_or_else(|| invalid("advanced.iris.depth_size sides must be integers."))?;
        if !(DEPTH_SIZE_MIN..=DEPTH_SIZE_MAX).contains(&side)
            || !side.is_multiple_of(DEPTH_SIZE_MULTIPLE)
        {
            return Err(invalid(format!(
                "advanced.iris.depth_size sides must be multiples of {DEPTH_SIZE_MULTIPLE} between \
                 {DEPTH_SIZE_MIN} and {DEPTH_SIZE_MAX} (got {side})."
            )));
        }
        Ok(side)
    };
    Ok((side(&pair[0])?, side(&pair[1])?))
}

fn validate_option(key: &str, value: &Value) -> Result<(), TrainingPlanError> {
    match key {
        "depth_size" => depth_size_value(value).map(drop),
        "resize" => one_of(key, value, &["cover", "none"]),
        "crop" => one_of(key, value, &["center", "random"]),
        "loss_reduction" => one_of(key, value, &["valid_pixels", "per_image"]),
        "invalid_examples" => one_of(key, value, &["reject", "skip"]),
        "reducer_init" => one_of(key, value, &["torch_default", "channel_mean"]),
        "muon_adjust_lr" => one_of(key, value, &["rms_norm", "spectral_norm", "none"]),
        "auto_lr" => one_of(key, value, &["none", "sqrt", "linear"]),
        "resume_data_policy" => one_of(key, value, &["exact", "new_phase"]),
        "preview_weights" | "export_weights" => one_of(key, value, &["raw", "ema"]),
        "hflip_prob" | "ema_decay" => {
            finite_in(key, value, |x| (0.0..=1.0).contains(&x), "between 0 and 1").map(drop)
        }
        "gradient_weight" => finite_in(key, value, |x| x >= 0.0, "at least 0").map(drop),
        "low_percentile" | "high_percentile" => finite_in(
            key,
            value,
            |x| (0.0..=100.0).contains(&x),
            "between 0 and 100",
        )
        .map(drop),
        "gradient_clip" => finite_in(key, value, |x| x > 0.0, "greater than 0").map(drop),
        "muon_momentum" => {
            finite_in(key, value, |x| (0.0..1.0).contains(&x), "in [0, 1)").map(drop)
        }
        "min_valid_pixels" => whole(key, value, 2).map(drop),
        "base_batch_size" | "num_epochs" => whole(key, value, 1).map(drop),
        "nan_loss_tolerance" | "keep_last_checkpoints" | "preview_items" => {
            whole(key, value, 0).map(drop)
        }
        "clip_target" | "muon_nesterov" | "ema_enabled" | "override_lr_on_resume" => {
            boolean(key, value).map(drop)
        }
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
            "advanced.iris.{key} is not an Iris depth training option."
        ))),
    }
}

/// The run's start (`advanced.depthInit`), validated.
pub fn depth_init(config: &TrainingConfig) -> Result<&str, TrainingPlanError> {
    match config.advanced.get(DEPTH_INIT_KEY) {
        None | Some(Value::Null) => Ok(DEPTH_INIT_EXPORT),
        Some(Value::String(init)) if init == DEPTH_INIT_EXPORT => Ok(DEPTH_INIT_EXPORT),
        Some(Value::String(init)) if init == DEPTH_INIT_PARENT => Ok(DEPTH_INIT_PARENT),
        Some(other) => Err(TrainingPlanError::TargetLimit(
            TrainingTargetLimitError::UnsupportedValue {
                field: DEPTH_INIT_KEY.to_owned(),
                value: other
                    .as_str()
                    .map_or_else(|| other.to_string(), str::to_owned),
                allowed: vec![DEPTH_INIT_EXPORT.to_owned(), DEPTH_INIT_PARENT.to_owned()],
            },
        )),
    }
}

/// Submit-time validation of a depth config: everything `DepthTrainPlan::resolve` would refuse from
/// the request alone, as typed 400s.
pub fn validate_depth_config(config: &TrainingConfig) -> Result<(), TrainingPlanError> {
    validate_micro_steps(
        config,
        u64::from(config.gradient_accumulation.max(1)),
        &DEPTH_MICRO_STEP_FIELDS,
    )?;
    if config
        .trigger_word
        .as_deref()
        .is_some_and(|word| !word.trim().is_empty())
    {
        return Err(invalid(
            "The depth task is conditioned on the empty prompt; it takes no trigger word.",
        ));
    }
    for key in UNREAD_ADVANCED_KEYS {
        let present = match config.advanced.get(key) {
            None | Some(Value::Null) => false,
            Some(Value::Array(values)) => !values.is_empty(),
            Some(Value::String(text)) => !text.trim().is_empty(),
            Some(_) => true,
        };
        if present {
            return Err(invalid(format!(
                "advanced.{key} is not a depth training control (depth previews predict dataset \
                 items with no prompt or sampler, and the run is a full fine-tune)."
            )));
        }
    }
    let init = depth_init(config)?;
    let options = iris_options(config)?;
    for (key, value) in &options {
        if super::IRIS_API_OWNED_OPTION_KEYS.contains(&key.as_str()) {
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
        if !DEPTH_CLIENT_OPTION_KEYS.contains(&key.as_str()) {
            return Err(TrainingPlanError::TargetLimit(
                TrainingTargetLimitError::UnsupportedValue {
                    field: "iris".to_owned(),
                    value: key.clone(),
                    allowed: DEPTH_CLIENT_OPTION_KEYS
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
    // The training size: `resolution` mirrors the larger side so it never says something else.
    if let Some(size) = options.get("depth_size").filter(|value| !value.is_null()) {
        let (height, width) = depth_size_value(size)?;
        if config.resolution != height.max(width) {
            return Err(invalid(format!(
                "resolution ({}) must equal the larger side of advanced.iris.depth_size \
                 ({height}×{width}); the depth run trains at depth_size.",
                config.resolution
            )));
        }
    } else if !config.resolution.is_multiple_of(DEPTH_SIZE_MULTIPLE)
        || !(DEPTH_SIZE_MIN..=DEPTH_SIZE_MAX).contains(&config.resolution)
    {
        return Err(invalid(format!(
            "resolution must be a multiple of {DEPTH_SIZE_MULTIPLE} between {DEPTH_SIZE_MIN} and \
             {DEPTH_SIZE_MAX}."
        )));
    }
    let low = options
        .get("low_percentile")
        .and_then(Value::as_f64)
        .unwrap_or(2.0);
    let high = options
        .get("high_percentile")
        .and_then(Value::as_f64)
        .unwrap_or(98.0);
    if low >= high {
        return Err(invalid(
            "advanced.iris.low_percentile must be below advanced.iris.high_percentile.",
        ));
    }
    if options
        .get("reducer_init")
        .is_some_and(|value| !value.is_null())
        && init != DEPTH_INIT_PARENT
    {
        return Err(invalid(
            "advanced.iris.reducer_init initializes the depth head of a run that starts from the \
             generation backbone (depthInit: parent); a run from the depth export already has one.",
        ));
    }
    let ema_enabled = options
        .get("ema_enabled")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if options
        .get("ema_decay")
        .is_some_and(|value| !value.is_null())
        && !ema_enabled
    {
        return Err(invalid(
            "advanced.iris.ema_decay needs the EMA enabled (advanced.iris.ema_enabled).",
        ));
    }
    for key in ["preview_weights", "export_weights"] {
        if options.get(key).and_then(Value::as_str) == Some("ema") && !ema_enabled {
            return Err(invalid(format!(
                "advanced.iris.{key} = ema needs the EMA enabled (advanced.iris.ema_enabled)."
            )));
        }
    }
    let resuming = config.advanced.contains_key(RESUME_FROM_JOB_KEY);
    if !resuming {
        for key in ["resume_data_policy", "override_lr_on_resume"] {
            if options.get(key).is_some_and(|value| !value.is_null()) {
                return Err(invalid(format!(
                    "advanced.iris.{key} applies only when resuming a previous run \
                     (advanced.{RESUME_FROM_JOB_KEY})."
                )));
            }
        }
    }
    if config.advanced.contains_key(LOAD_FROM_JOB_KEY) {
        if resuming {
            return Err(invalid(
                "Resume continues a run's full state; loadFromJobId starts a fresh run from its \
                 weights. Choose one.",
            ));
        }
        if init != DEPTH_INIT_EXPORT {
            return Err(invalid(
                "loadFromJobId starts from a previous depth run's weights; it replaces the \
                 generation-backbone start (depthInit: parent). Use depthInit: export.",
            ));
        }
    }
    Ok(())
}

/// One item's aligned depth target, validated (`extra.irisDepth`).
pub fn validate_item_depth(extra: &ExtraFields) -> Result<(), String> {
    let Some(value) = extra.get(ITEM_DEPTH_KEY) else {
        return Ok(());
    };
    let map = value.as_object().ok_or_else(|| {
        "Item depth must be an object {path, maskPath?, scale?, minDepth?, maxDepth?}.".to_owned()
    })?;
    for key in map.keys() {
        if !matches!(
            key.as_str(),
            "path" | "maskPath" | "scale" | "minDepth" | "maxDepth"
        ) {
            return Err(format!(
                "Item depth key {key:?} is not part of the depth target."
            ));
        }
    }
    let path = |key: &str, required: bool| -> Result<(), String> {
        match map.get(key) {
            None | Some(Value::Null) if !required => Ok(()),
            Some(Value::String(path)) if is_safe_relative_path(path) => {
                let lower = path.to_ascii_lowercase();
                if lower.ends_with(".npy") || lower.ends_with(".png") {
                    Ok(())
                } else {
                    Err(format!(
                        "Item depth {key} must be a .npy or grayscale .png file (got {path:?})."
                    ))
                }
            }
            _ => Err(format!(
                "Item depth {key} must be a path relative to the dataset root."
            )),
        }
    };
    path("path", true)?;
    path("maskPath", false)?;
    let number = |key: &str| -> Result<Option<f64>, String> {
        match map.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(value) => value
                .as_f64()
                .filter(|number| number.is_finite())
                .map(Some)
                .ok_or_else(|| format!("Item depth {key} must be a finite number.")),
        }
    };
    if number("scale")?.is_some_and(|scale| scale <= 0.0) {
        return Err("Item depth scale must be greater than 0.".to_owned());
    }
    let min_depth = number("minDepth")?.unwrap_or(0.0);
    if min_depth < 0.0 {
        return Err("Item depth minDepth must be at least 0.".to_owned());
    }
    if number("maxDepth")?.is_some_and(|max| max <= min_depth) {
        return Err("Item depth maxDepth must exceed minDepth.".to_owned());
    }
    Ok(())
}

/// Every dataset item must be an aligned example: an RGB image plus its depth target. The files'
/// alignment, formats and validity are checked by the engine's own readers (the dataset check
/// endpoint and the trainer's dataset pass).
pub fn validate_depth_dataset(dataset: &TrainingDataset) -> Result<(), TrainingPlanError> {
    for item in &dataset.items {
        validate_item_depth(&item.extra).map_err(invalid)?;
    }
    let missing: Vec<&str> = dataset
        .items
        .iter()
        .filter(|item| !item.extra.contains_key(ITEM_DEPTH_KEY))
        .map(|item| item.display_name.as_str())
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(invalid(format!(
        "{} dataset item(s) have no depth target: {}{}. Attach each image's depth map (.npy or \
         8/16-bit grayscale .png) before training depth.",
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

    fn config() -> TrainingConfig {
        iris_depth_target().defaults
    }

    #[test]
    fn the_client_allow_list_is_the_engine_list_minus_the_path_options() {
        // The engine's `DEPTH_OPTION_KEYS` (32) = these 30 + `load_from` + `resume_from`.
        assert_eq!(DEPTH_CLIENT_OPTION_KEYS.len() + 2, 32);
        let mut sorted = DEPTH_CLIENT_OPTION_KEYS.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), DEPTH_CLIENT_OPTION_KEYS.len());
    }

    #[test]
    fn the_defaults_are_the_paper_recipe_and_validate() {
        let target = iris_depth_target();
        let defaults = &target.defaults;
        assert_eq!(defaults.learning_rate.as_f64(), Some(3e-5));
        assert_eq!(defaults.advanced["weightDecay"], json!(0.01));
        assert_eq!(defaults.advanced["lrScheduler"], "constant");
        assert_eq!(defaults.advanced["lrWarmupSteps"], 0);
        assert_eq!(defaults.advanced["networkType"], "full");
        assert_eq!(defaults.advanced["lossType"], "l1");
        assert_eq!(defaults.optimizer, "adamw");
        // 10K optimizer steps at a global batch of 32.
        assert_eq!(defaults.batch_size * defaults.gradient_accumulation, 32);
        assert_eq!(defaults.steps / defaults.gradient_accumulation, 10_000);
        assert_eq!(target.output_kind, TrainingOutputKind::BaseCheckpoint);
        validate_depth_config(defaults).expect("the defaults validate");
    }

    #[test]
    fn request_only_refusals_are_typed_400s() {
        let refused = |mutate: &dyn Fn(&mut TrainingConfig)| {
            let mut config = config();
            mutate(&mut config);
            validate_depth_config(&config).expect_err("refused")
        };
        // Micro-step rule.
        let error = refused(&|config| config.save_every = 33);
        assert!(matches!(
            error,
            TrainingPlanError::TargetLimit(TrainingTargetLimitError::NotMultipleOf { .. })
        ));
        // Unknown and API-owned options.
        refused(&|config| config.advanced["iris"]["prediction"] = json!("v"));
        refused(&|config| config.advanced["iris"]["load_from"] = json!("/tmp/x"));
        // Domains.
        refused(&|config| config.advanced["iris"]["invalid_examples"] = json!("ignore"));
        refused(&|config| config.advanced["iris"]["depth_size"] = json!([480, 650]));
        refused(&|config| config.advanced["iris"]["min_valid_pixels"] = json!(1));
        refused(&|config| {
            config.advanced["iris"]["low_percentile"] = json!(60.0);
            config.advanced["iris"]["high_percentile"] = json!(40.0);
        });
        // The size mirror.
        refused(&|config| config.resolution = 512);
        // Generic controls the depth objective does not read.
        refused(&|config| {
            config
                .advanced
                .insert("samplePrompts".into(), json!(["a cat"]));
        });
        refused(&|config| config.trigger_word = Some("tok".into()));
        // The reducer initializes a parent start only; EMA-dependent selections need EMA.
        refused(&|config| config.advanced["iris"]["reducer_init"] = json!("channel_mean"));
        refused(&|config| config.advanced["iris"]["export_weights"] = json!("ema"));
        refused(&|config| config.advanced["iris"]["ema_decay"] = json!(0.99));
        refused(&|config| config.advanced["iris"]["resume_data_policy"] = json!("new_phase"));
        refused(&|config| {
            config.advanced.insert("depthInit".into(), json!("parent"));
            config
                .advanced
                .insert("loadFromJobId".into(), json!("job_1"));
        });
        refused(&|config| {
            config.advanced.insert("depthInit".into(), json!("random"));
        });
        // ...and the matching accepted shapes.
        let mut parent = config();
        parent.advanced.insert("depthInit".into(), json!("parent"));
        parent.advanced["iris"]["reducer_init"] = json!("channel_mean");
        validate_depth_config(&parent).expect("a parent start may choose the reducer init");
        let mut ema = config();
        ema.advanced["iris"]["ema_enabled"] = json!(true);
        ema.advanced["iris"]["export_weights"] = json!("ema");
        validate_depth_config(&ema).expect("EMA export with EMA on");
        let mut skip = config();
        skip.advanced["iris"]["invalid_examples"] = json!("skip");
        validate_depth_config(&skip).expect("skip is the other explicit policy");
    }

    #[test]
    fn floors_follow_the_micro_batch_and_refuse_beyond_the_table() {
        let mut config = config();
        // 480×640 × 1 = 307 200 px.
        assert_eq!(depth_memory_floor_gb("mlx", &config), Some(80));
        assert_eq!(depth_memory_floor_gb("candle", &config), Some(84));
        config.advanced["iris"]["depth_size"] = json!([128, 128]);
        config.resolution = 128;
        assert_eq!(depth_memory_floor_gb("mlx", &config), Some(64));
        assert_eq!(depth_memory_floor_gb("candle", &config), Some(68));
        // The paper's batch 32 in one micro-batch is beyond every declared row.
        let mut big = iris_depth_target().defaults;
        big.batch_size = 32;
        big.gradient_accumulation = 1;
        assert_eq!(depth_memory_floor_gb("mlx", &big), None);
        assert_eq!(depth_memory_floor_gb("cpu", &config), None);
    }

    #[test]
    fn item_depth_targets_are_validated_at_the_store_boundary() {
        let item = |value: Value| {
            let mut extra = ExtraFields::new();
            extra.insert(ITEM_DEPTH_KEY.to_owned(), value);
            validate_item_depth(&extra)
        };
        item(json!({"path": "sidecars/a.depth.npy"})).unwrap();
        item(
            json!({"path": "sidecars/a.depth.png", "maskPath": "sidecars/a.mask.png",
                    "scale": 0.001, "minDepth": 0.1, "maxDepth": 80}),
        )
        .unwrap();
        assert!(item(json!({"path": "../escape.npy"})).is_err());
        assert!(item(json!({"path": "sidecars/a.exr"})).is_err());
        assert!(item(json!({"mask": "x"})).is_err());
        assert!(item(json!({"path": "a.npy", "scale": 0})).is_err());
        assert!(item(json!({"path": "a.npy", "minDepth": 5, "maxDepth": 5})).is_err());
    }
}
