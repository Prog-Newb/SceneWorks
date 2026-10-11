// Test-only fixtures for the Iris-3B training target (sc-25685/sc-25686). Mirrors the JSON the API
// serializes from `crates/sceneworks-core/src/training/iris.rs::iris_generation_target()` so the
// web tests exercise the real contract shape (camelCase, `memoryFloors` flattened to the top).

export const IRIS_CLIENT_OPTION_KEYS = Object.freeze([
  "init", "prediction", "flow_shift", "shift_law", "shift_base_tokens", "logit_mean",
  "logit_std", "x_pred_sigma_min", "text_dropout", "gradient_clip", "betas", "muon_momentum",
  "muon_nesterov", "muon_adjust_lr", "auto_lr", "base_batch_size", "ema_enabled", "ema_decay",
  "num_epochs", "nan_loss_tolerance", "caption_field", "caption_fields", "on_caption_overflow",
  "text_conditioning", "keep_last_checkpoints", "milestone_steps", "resume_data_policy",
  "override_lr_on_resume", "preview_weights", "preview_negative_prompt", "export_weights",
  "export_dtype", "upstream_validation_prompts",
]);

export function irisTarget() {
  return {
    id: "iris_3b_generation",
    name: "Iris 3B",
    modality: "image",
    outputKind: "lora",
    family: "iris",
    baseModel: "iris_3b",
    baseModelRepo: "speridlabs/iris-3b",
    kernel: "iris_train",
    defaults: {
      rank: 16,
      alpha: 16,
      learningRate: 0.0001,
      steps: 1000,
      batchSize: 1,
      gradientAccumulation: 1,
      resolution: 256,
      saveEvery: 250,
      seed: 42,
      optimizer: "adamw",
      advanced: {
        mixedPrecision: "bf16",
        networkType: "lora",
        decomposeFactor: -1,
        timestepType: "logit_normal",
        timestepBias: "balanced",
        lossType: "mse",
        weightDecay: 0.0,
        lrScheduler: "constant",
        lrWarmupSteps: 0,
        gradientCheckpointing: false,
        sampleEvery: 250,
        sampleSteps: 30,
        sampleGuidanceScale: 3.0,
        sampleCount: 4,
        qualityPreset: "balanced",
        outputScope: "project",
        requestedGpu: "auto",
        iris: {
          init: "weights",
          text_conditioning: "cached",
          text_dropout: 0.1,
          gradient_clip: 0.5,
          ema_enabled: true,
          ema_decay: 0.9999,
          on_caption_overflow: "warn",
          caption_field: "caption",
          caption_fields: [],
          keep_last_checkpoints: 2,
          milestone_steps: [],
          nan_loss_tolerance: 20,
          betas: [0.9, 0.95],
          auto_lr: "none",
          base_batch_size: 256,
          preview_negative_prompt: "",
          upstream_validation_prompts: false,
          export_dtype: "fp32",
        },
      },
    },
    limits: {
      rank: [1, 128],
      alpha: [1, 128],
      steps: [1, 200000],
      batchSize: [1, 64],
      resolutions: [256, 512, 1024],
      sampleCount: [1, 4],
      optimizers: ["adamw", "muon"],
      networkTypes: ["lora", "lokr", "full"],
      lrSchedulers: ["constant", "cosine"],
      timestepTypes: ["logit_normal", "uniform"],
      timestepBiases: ["balanced"],
      lossTypes: ["mse"],
      mixedPrecisions: ["bf16", "f32"],
      gradientCheckpointing: false,
      microStepFields: ["steps", "saveEvery", "sampleEvery", "lrWarmupSteps"],
      inits: ["weights", "random"],
      outputScopes: ["project", "global"],
    },
    ui: { label: "Iris 3B", recommendedFor: ["character", "style", "concept"], datasetModality: "image" },
    memoryFloors: {
      measured: false,
      unit: "GB",
      basis: "Declared floors (no measurement campaign).",
      lanes: {
        mlx: {
          adapter: { 256: 16, 512: 24, 1024: 40 },
          full: { 256: 80, 512: 88, 1024: 104 },
        },
        candle: {
          adapter: { 256: 24, 512: 32, 1024: 48 },
          full: { 256: 80, 512: 88, 1024: 104 },
        },
      },
    },
  };
}

// A terminal Iris lora_train job as the jobs list carries it (`payload.plan` is the TrainingPlan).
export function irisJob({
  id,
  status = "completed",
  networkType = "lora",
  outputName = `Run ${id}`,
  projectId = "p1",
  targetId = "iris_3b_generation",
  result = {},
  type = "lora_train",
  createdAt = "2026-10-01T00:00:00Z",
} = {}) {
  return {
    id,
    type,
    status,
    projectId,
    createdAt,
    payload: {
      outputName,
      plan: {
        target: { targetId, kernel: "iris_train" },
        config: { advanced: { networkType } },
      },
    },
    result,
  };
}
