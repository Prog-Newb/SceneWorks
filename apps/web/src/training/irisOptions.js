// Iris-3B generation training (sc-25685 MLX / sc-25686 Candle, epic 25678) — the web half of
// the `advanced.iris` options-bag contract owned by
// `crates/sceneworks-core/src/training/iris.rs`.
//
// The engine REFUSES unknown option keys, so this module is schema-driven: `IRIS_OPTION_SPECS`
// names exactly the 33 client-settable keys (`IRIS_CLIENT_OPTION_KEYS` in iris.rs) and the domain
// each accepts (`validate_option`). The draft holds every key as an input-friendly value; the
// snapshot emits only the keys that are filled AND in play for this run (E3: never send a knob
// the trainer ignores or refuses, never drop one it accepts). The two PATH options
// (`resume_from` / `load_from`) are never authored here — the API resolves them from
// `advanced.resumeFromJobId` / `advanced.loadFromJobId`.
//
// Pure: no React, no app state.

import { issue } from "../validation/issues.js";
import { asText, numberFromDraft, numericDraft } from "./drafts.js";
import { terminalStatuses } from "../jobTypes.js";

export const IRIS_TRAINING_KERNEL = "iris_train";
export const IRIS_TRAINING_TARGET_ID = "iris_3b_generation";

export function isIrisTarget(target) {
  return target?.kernel === IRIS_TRAINING_KERNEL;
}

// Which Iris objective a target trains (sc-25687 / sc-25688): "generation" | "depth" |
// "restoration", or null for a non-Iris target. Depth and restoration share the kernel, so the
// target's own `irisObjective` (or `ui.objective`, else its base model) tells them apart.
const TASK_BASE_MODELS = { iris_3b_depth: "depth", iris_3b_restore: "restoration" };
export function irisObjective(target) {
  if (!isIrisTarget(target)) return null;
  const objective = target.irisObjective ?? target.ui?.objective ?? TASK_BASE_MODELS[target.baseModel];
  return objective === "depth" || objective === "restoration" ? objective : "generation";
}

export function isIrisGenerationTarget(target) {
  return irisObjective(target) === "generation";
}

export function isIrisTaskTarget(target) {
  return ["depth", "restoration"].includes(irisObjective(target));
}

const between = (min, max, { maxExclusive = false } = {}) => ({
  check: (x) => x >= min && (maxExclusive ? x < max : x <= max),
  expect: maxExclusive ? `in [${min}, ${max})` : `between ${min} and ${max}`,
});
const positive = { check: (x) => x > 0, expect: "greater than 0" };

// kind: choice | number | int | bool | text | strings | ints | betas
export const IRIS_OPTION_SPECS = Object.freeze({
  init: { kind: "choice", values: ["weights", "random"], label: "Initialization" },
  prediction: { kind: "choice", values: ["v", "x"], label: "Prediction target" },
  flow_shift: { kind: "number", ...positive, label: "Flow shift" },
  shift_law: { kind: "choice", values: ["none", "sd3", "flux"], label: "Shift law" },
  shift_base_tokens: { kind: "int", min: 1, label: "Shift base tokens" },
  logit_mean: { kind: "number", label: "Logit-normal mean" },
  logit_std: { kind: "number", label: "Logit-normal std" },
  x_pred_sigma_min: { kind: "number", label: "x-prediction sigma floor" },
  text_dropout: { kind: "number", ...between(0, 1), label: "Caption dropout" },
  gradient_clip: { kind: "number", ...positive, label: "Gradient clip" },
  betas: { kind: "betas", label: "Betas" },
  muon_momentum: { kind: "number", ...between(0, 1, { maxExclusive: true }), label: "Muon momentum" },
  muon_nesterov: { kind: "bool", label: "Muon Nesterov" },
  muon_adjust_lr: { kind: "choice", values: ["rms_norm", "spectral_norm", "none"], label: "Muon LR adjustment" },
  auto_lr: { kind: "choice", values: ["none", "sqrt", "linear"], label: "Auto LR scaling" },
  base_batch_size: { kind: "int", min: 1, label: "Base batch size" },
  ema_enabled: { kind: "bool", label: "EMA" },
  ema_decay: { kind: "number", ...between(0, 1), label: "EMA decay" },
  num_epochs: { kind: "int", min: 1, label: "Epochs" },
  nan_loss_tolerance: { kind: "int", min: 0, label: "NaN-loss tolerance" },
  caption_field: { kind: "text", maxLength: 64, label: "Caption field" },
  caption_fields: { kind: "strings", label: "Caption fields" },
  on_caption_overflow: { kind: "choice", values: ["warn", "error", "silent"], label: "Caption overflow" },
  text_conditioning: { kind: "choice", values: ["cached", "on_the_fly"], label: "Text conditioning" },
  keep_last_checkpoints: { kind: "int", min: 0, label: "Keep last checkpoints" },
  milestone_steps: { kind: "ints", label: "Milestone steps" },
  resume_data_policy: { kind: "choice", values: ["exact", "new_phase"], label: "Resume data policy" },
  override_lr_on_resume: { kind: "bool", label: "Override LR on resume" },
  preview_weights: { kind: "choice", values: ["raw", "ema"], label: "Preview weights" },
  preview_negative_prompt: { kind: "text", keepWhitespace: true, label: "Preview negative prompt" },
  export_weights: { kind: "choice", values: ["raw", "ema"], label: "Export weights" },
  export_dtype: { kind: "choice", values: ["fp32", "bf16"], label: "Export precision" },
  upstream_validation_prompts: { kind: "bool", label: "Upstream validation prompts" },
});

export const IRIS_OPTION_KEYS = Object.freeze(Object.keys(IRIS_OPTION_SPECS));

// Draft value for one key: strings for inputs/selects, booleans (or "" = trainer default) for
// toggles, an array of names for caption_fields, "a, b" text for milestone steps, a pair of
// strings for betas.
export function draftValue(spec, value) {
  switch (spec.kind) {
    case "json":
      return value && typeof value === "object" ? JSON.stringify(value) : asText(value);
    case "bool":
      return typeof value === "boolean" ? value : "";
    case "strings":
      return Array.isArray(value) ? value.map(asText) : [];
    case "ints":
      return Array.isArray(value) ? value.join(", ") : asText(value);
    case "betas":
    case "size":
      return Array.isArray(value) && value.length === 2 ? value.map(numericDraft) : ["", ""];
    case "number":
    case "int":
      return numericDraft(value);
    default:
      return asText(value);
  }
}

export function irisDraftFromDefaults(defaults = {}) {
  const source = defaults && typeof defaults === "object" ? defaults : {};
  return Object.fromEntries(IRIS_OPTION_KEYS.map((key) => [key, draftValue(IRIS_OPTION_SPECS[key], source[key])]));
}

function parseIntList(text) {
  const tokens = String(text ?? "").split(/[\s,]+/).filter(Boolean);
  return tokens.map((token) => Number(token));
}

// Parse one draft value. `{ value }` when set, `{}` when empty (= trainer/backbone default),
// `{ invalid }` when the user typed something outside the key's domain.
export function parseOption(key, raw, specs = IRIS_OPTION_SPECS) {
  const spec = specs[key];
  const label = spec.label;
  switch (spec.kind) {
    case "choice": {
      const text = asText(raw).trim();
      if (!text) return {};
      return spec.values.includes(text) ? { value: text } : { invalid: `${label} must be one of ${spec.values.join(", ")}` };
    }
    case "number": {
      if (!asText(raw).trim()) return {};
      const number = numberFromDraft(raw);
      if (number === null) return { invalid: `${label} must be a number` };
      if (spec.check && !spec.check(number)) return { invalid: `${label} must be ${spec.expect}` };
      return { value: number };
    }
    case "int": {
      if (!asText(raw).trim()) return {};
      const number = numberFromDraft(raw);
      if (number === null || !Number.isInteger(number) || number < spec.min) {
        return { invalid: `${label} must be a whole number of at least ${spec.min}` };
      }
      return { value: number };
    }
    case "bool":
      return typeof raw === "boolean" ? { value: raw } : {};
    case "text": {
      const text = spec.keepWhitespace ? asText(raw) : asText(raw).trim();
      if (!text.trim()) return {};
      if (spec.maxLength && text.length > spec.maxLength) {
        return { invalid: `${label} must be at most ${spec.maxLength} characters` };
      }
      return { value: text };
    }
    case "strings": {
      const names = (Array.isArray(raw) ? raw : []).map((name) => asText(name).trim()).filter(Boolean);
      if (!names.length) return {};
      if (names.some((name) => name.length > 64)) return { invalid: `${label} must be at most 64 characters each` };
      return { value: Array.from(new Set(names)) };
    }
    case "ints": {
      const steps = parseIntList(raw);
      if (!steps.length) return {};
      if (steps.some((step) => !Number.isInteger(step) || step <= 0)) {
        return { invalid: `${label} must be positive whole optimizer steps, separated by commas` };
      }
      return { value: steps };
    }
    case "betas": {
      const pair = Array.isArray(raw) ? raw : ["", ""];
      const filled = pair.map((beta) => asText(beta).trim() !== "");
      if (!filled[0] && !filled[1]) return {};
      const values = pair.map(numberFromDraft);
      if (!filled[0] || !filled[1] || values.some((beta) => beta === null || beta < 0 || beta >= 1)) {
        return { invalid: `${label} must both be numbers in [0, 1)` };
      }
      return { value: values };
    }
    case "size": {
      // Depth training size [height, width]: multiples of 16 in [64, 2048] (depth.rs).
      const pair = (Array.isArray(raw) ? raw : ["", ""]).map(numberFromDraft);
      if (pair.some((side) => side === null || !Number.isInteger(side) || side < 64 || side > 2048 || side % 16)) {
        return { invalid: `${label} sides must be multiples of 16 between 64 and 2048` };
      }
      return { value: pair };
    }
    case "json": {
      const text = asText(raw).trim();
      if (!text) return {};
      try {
        const value = JSON.parse(text);
        if (value && typeof value === "object" && !Array.isArray(value)) return { value };
      } catch {
        // fall through
      }
      return { invalid: `${label} must be a JSON object` };
    }
    default:
      return {};
  }
}

// The run facts that decide which options are in play.
function irisRunContext(configDraft = {}) {
  const iris = configDraft.iris ?? {};
  return {
    full: asText(configDraft.networkType).trim().toLowerCase() === "full",
    muon: asText(configDraft.optimizer).trim().toLowerCase() === "muon",
    logitNormal: asText(configDraft.timestepType).trim() === "logit_normal",
    resuming: Boolean(asText(configDraft.resumeFromJobId).trim()),
    emaOff: iris.ema_enabled === false,
    autoLr: !["", "none"].includes(asText(iris.auto_lr).trim()),
  };
}

// Whether `key` applies to this run. A key that is not in play is neither rendered as a live
// control nor sent nor validated.
export function irisOptionInPlay(key, configDraft) {
  const run = irisRunContext(configDraft);
  switch (key) {
    case "init":
      return run.full;
    case "muon_momentum":
    case "muon_nesterov":
    case "muon_adjust_lr":
      return run.muon;
    case "ema_decay":
      return !run.emaOff;
    case "base_batch_size":
      return run.autoLr;
    case "resume_data_policy":
    case "override_lr_on_resume":
      return run.resuming;
    case "logit_mean":
    case "logit_std":
      return run.logitNormal;
    default:
      return true;
  }
}

// `advanced.iris` for the request: only filled, in-play, in-domain keys.
export function irisOptionsSnapshot(configDraft = {}) {
  const iris = configDraft.iris ?? {};
  const out = {};
  for (const key of IRIS_OPTION_KEYS) {
    if (!irisOptionInPlay(key, configDraft)) continue;
    const parsed = parseOption(key, iris[key]);
    if (Object.prototype.hasOwnProperty.call(parsed, "value")) {
      out[key] = parsed.value;
    }
  }
  return out;
}

// The request-level keys that ride beside `advanced.iris`.
export function irisRequestKeys(configDraft = {}) {
  const run = irisRunContext(configDraft);
  const resumeFromJobId = asText(configDraft.resumeFromJobId).trim();
  const loadFromJobId = asText(configDraft.loadFromJobId).trim();
  return {
    ...(resumeFromJobId ? { resumeFromJobId } : {}),
    ...(run.full && loadFromJobId ? { loadFromJobId } : {}),
  };
}

export function irisUpstreamPromptsOn(configDraft = {}) {
  return configDraft.iris?.upstream_validation_prompts === true;
}

// The Iris-specific rules (fields `iris.<key>` outline their own inputs).
export function irisValidation(configDraft = {}) {
  const issues = [];
  const iris = configDraft.iris ?? {};
  const run = irisRunContext(configDraft);
  for (const key of IRIS_OPTION_KEYS) {
    if (!irisOptionInPlay(key, configDraft)) continue;
    const parsed = parseOption(key, iris[key]);
    if (parsed.invalid) {
      issues.push(issue.error(`iris.${key}`, parsed.invalid));
    }
  }
  if (run.emaOff) {
    for (const key of ["preview_weights", "export_weights"]) {
      if (asText(iris[key]).trim() === "ema") {
        issues.push(issue.error(`iris.${key}`, `${IRIS_OPTION_SPECS[key].label}: EMA weights need the EMA enabled`));
      }
    }
  }
  const { loadFromJobId } = irisRequestKeys(configDraft);
  if (loadFromJobId && run.resuming) {
    issues.push(issue.error("loadFromJobId", "Resume continues a run; starting from a previous run's weights begins a new one — choose one"));
  }
  if (loadFromJobId && asText(iris.init).trim() === "random") {
    issues.push(issue.error("loadFromJobId", "Starting from a previous run's weights cannot be combined with random initialization"));
  }
  return issues;
}

// ---- Memory floors ------------------------------------------------------------------------

// The capability switches name the active training lane: a gated Mac runs MLX, an off-Mac host
// runs Candle. Neither set (capabilities not loaded, or a Mac in observe mode) is unknown.
export function trainingLaneFromCapabilities(caps) {
  if (caps?.macGatingActive) return "mlx";
  if (caps?.candleGatingActive) return "candle";
  return null;
}

// The DECLARED floor(s) for this run, `[{ lane, gb }]`. `size` is the row key the target's table
// is keyed by (`memoryFloors.key`: the resolution, or the micro-batch pixels for depth /
// restoration). A size between two declared rows takes the next row up (iris.rs
// `iris_training_memory_floor_gb`); none above the table. With an unknown lane, both lanes' floors.
// A depth table has no artifact level (always a full fine-tune).
export function irisMemoryFloors(target, networkType, size, lane) {
  const lanes = target?.memoryFloors?.lanes;
  if (!lanes || typeof lanes !== "object") return [];
  const artifact = asText(networkType).trim().toLowerCase() === "full" ? "full" : "adapter";
  const wanted = Number(size);
  const laneNames = lane ? [lane] : ["mlx", "candle"];
  return laneNames
    .map((name) => {
      const rows = lanes[name]?.[artifact] ?? lanes[name];
      if (!rows) return null;
      const match = Object.entries(rows)
        .map(([row, gb]) => [Number(row), Number(gb)])
        .filter(([row, gb]) => Number.isFinite(wanted) && Number.isFinite(gb) && row >= wanted)
        .sort((left, right) => left[0] - right[0])[0];
      return match ? { lane: name, gb: match[1] } : null;
    })
    .filter(Boolean);
}

// ---- Captions -----------------------------------------------------------------------------

// The caption fields a run can select from the active dataset: "caption" (the item's ordinary
// caption) plus every named field any item carries.
export function irisCaptionFieldOptions(dataset) {
  const names = new Set();
  for (const item of dataset?.items ?? []) {
    const captions = item?.captions;
    if (captions && typeof captions === "object" && !Array.isArray(captions)) {
      Object.keys(captions).forEach((name) => names.add(name));
    }
  }
  names.delete("caption");
  return ["caption", ...[...names].sort()];
}

// ---- Previous runs ------------------------------------------------------------------------

function jobTargetId(job) {
  return job?.payload?.plan?.target?.targetId ?? job?.payload?.targetId ?? "";
}

function jobNetworkType(job) {
  return asText(job?.payload?.plan?.config?.advanced?.networkType).trim().toLowerCase() || "lora";
}

// Terminal Iris training runs (newest first) a new run can resume or start from. The API is the
// authority on whether a run left a durable checkpoint; this only narrows the picker. `baseModel`
// selects a depth / restoration target's runs (its plan's `target.baseModel`); without it, the
// generation target's.
export function irisPreviousRuns(jobs, { projectId, fullOnly = false, baseModel } = {}) {
  return (jobs ?? [])
    .filter(
      (job) =>
        job?.type === "lora_train" &&
        (baseModel
          ? job.payload?.plan?.target?.baseModel === baseModel
          : jobTargetId(job) === IRIS_TRAINING_TARGET_ID) &&
        terminalStatuses.has(job.status) &&
        (!projectId || job.projectId === projectId) &&
        (!fullOnly || jobNetworkType(job) === "full"),
    )
    .slice()
    .sort((left, right) => String(right.createdAt ?? "").localeCompare(String(left.createdAt ?? "")));
}

export function irisRunOutcome(job) {
  const result = job?.result ?? {};
  if (job?.status === "canceled") {
    if (result.trainedArtifactExported && Number(result.stepsCompleted) > 0) {
      return `Canceled at step ${result.stepsCompleted} — checkpoint and partial export saved`;
    }
    return "Canceled";
  }
  if (job?.status === "completed") return "Completed";
  if (job?.status === "failed") return "Failed";
  if (job?.status === "interrupted") return "Interrupted";
  return asText(job?.status);
}

export function irisRunLabel(job) {
  const name = asText(job?.payload?.outputName).trim() || job?.id;
  return `${name} · ${irisRunOutcome(job)}`;
}
