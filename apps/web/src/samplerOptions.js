// Shared labels + helpers for the configurable sampler/scheduler controls. The
// curated vocabulary (epic 7114) matches the native mlx-gen / candle-gen gen-core
// Solver / Scheduler registries exactly — the names sent in the job's `advanced`
// block ARE the engine's sampler/scheduler ids. The studios source their dropdown
// contents from each model's per-backend manifest `limits.samplers` /
// `limits.schedulers` arrays (base `limits` overridden by `mlx.limits` /
// `candle.limits` for the active backend) so invalid combos are never selectable.

export const SAMPLER_LABELS = {
  default: "Model default",
  euler: "Euler",
  euler_ancestral: "Euler ancestral",
  heun: "Heun (2nd-order)",
  dpmpp_2m: "DPM++ (2M)",
  dpmpp_sde: "DPM++ SDE",
  uni_pc: "UniPC",
  lcm: "LCM",
  ddim: "DDIM",
};

export const SCHEDULER_LABELS = {
  default: "Model default",
  normal: "Normal",
  simple: "Simple (uniform)",
  karras: "Karras",
  exponential: "Exponential",
  sgm_uniform: "SGM uniform",
  beta: "Beta",
  ddim_uniform: "DDIM uniform",
};

// Classifier-free-guidance policy (the orthogonal guidance axis, epic 7434). Names
// match gen-core's GuidanceMethod registry exactly. "cfg" is the engine's standard
// guidance (the N1 no-op the studio defaults to); the rest are advertised per-model-
// per-backend only where the linked engine honors them — CFG++ (cfg_pp) is MLX-only
// today (sc-8256), so the picker only appears for the SDXL family on the MLX backend.
export const GUIDANCE_METHOD_LABELS = {
  cfg: "CFG (standard)",
  cfg_rescale: "CFG rescale",
  apg: "APG (adaptive)",
  cfg_pp: "CFG++",
};

const SAMPLER_ORDER = [
  "default",
  "euler",
  "euler_ancestral",
  "heun",
  "dpmpp_2m",
  "dpmpp_sde",
  "uni_pc",
  "lcm",
  "ddim",
];
const SCHEDULER_ORDER = [
  "default",
  "normal",
  "simple",
  "karras",
  "exponential",
  "sgm_uniform",
  "beta",
  "ddim_uniform",
];
const GUIDANCE_METHOD_ORDER = ["cfg", "cfg_rescale", "apg", "cfg_pp"];

function uniqueOrdered(values, order) {
  const seen = new Set();
  const result = [];
  for (const key of order) {
    if (values.includes(key) && !seen.has(key)) {
      seen.add(key);
      result.push(key);
    }
  }
  // Append any keys not in our canonical ordering (forward-compat).
  for (const value of values) {
    if (typeof value === "string" && !seen.has(value)) {
      seen.add(value);
      result.push(value);
    }
  }
  return result;
}

// The effective `limits` for the active backend: the per-backend `mlx.limits` /
// `candle.limits` override when present, else the base `limits` (epic 7114 P5
// per-model-per-backend gating). `backend` is the active worker's backend
// ("mlx" | "candle"); a null/unknown backend falls back to the base menu.
export function effectiveLimits(model, backend) {
  const override = backend ? model?.[backend]?.limits : null;
  return override ?? model?.limits;
}

// Pull the menu out of a model manifest entry, falling back to default-only.
// When the menu has fewer than 2 entries, the studio hides the dropdown — the
// caller can use `samplerMenu.length > 1` to gate rendering.
export function samplerOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.samplers) ? limits.samplers : ["default"];
  return uniqueOrdered(values, SAMPLER_ORDER);
}

export function schedulerOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.schedulers) ? limits.schedulers : ["default"];
  return uniqueOrdered(values, SCHEDULER_ORDER);
}

// The guidance-method menu the active backend honors for this model (epic 7434).
// Falls back to the engine's standard ["cfg"] when the model advertises none — the
// studio hides the picker when fewer than 2 methods are offered (length > 1 gate),
// so only models that actually expose an alternative (CFG++ on SDXL/MLX) show it.
export function guidanceMethodOptionsFromModel(model, backend) {
  const limits = effectiveLimits(model, backend);
  const values = Array.isArray(limits?.guidanceMethods) ? limits.guidanceMethods : ["cfg"];
  return uniqueOrdered(values, GUIDANCE_METHOD_ORDER);
}

// Per-model defaults (UI initial values). The worker never reads these — they
// only set the studio form's initial sampler/scheduler choice. Falls back to
// "default" when the manifest doesn't pin one.
export function samplerDefaultFromModel(model) {
  const value = model?.defaults?.sampler;
  return typeof value === "string" && value.length ? value : "default";
}

export function schedulerDefaultFromModel(model) {
  const value = model?.defaults?.scheduler;
  return typeof value === "string" && value.length ? value : "default";
}

// The guidance method the studio selects initially. Defaults to "cfg" (the engine's
// standard guidance, the N1 no-op) so existing recipes are unchanged; a model may
// pin a different default via `defaults.guidanceMethod`.
export function guidanceMethodDefaultFromModel(model) {
  const value = model?.defaults?.guidanceMethod;
  return typeof value === "string" && value.length ? value : "cfg";
}

export function schedulerShiftDefaultFromModel(model) {
  const value = Number(model?.defaults?.schedulerShift);
  return Number.isFinite(value) && value > 0 ? value : 3.0;
}

export function stepsDefaultFromModel(model) {
  const value = Number(model?.defaults?.steps);
  return Number.isFinite(value) && value > 0 ? value : null;
}

// `limits.steps` — the EXACT set of denoise step counts a model can render, or `null` for no menu
// (sc-19502). Distilled models bake their sigma waypoints into training, so LTX-2.3 renders at 8
// steps and nothing else; the API and worker refuse anything off this menu.
//
// Deliberately mirrors `allowed_steps` in crates/sceneworks-core/src/video_request.rs, INCLUDING
// its "unsatisfiable means no menu" fallback: a menu whose entries are all unusable would otherwise
// pin the control to nothing and make the model unusable from the UI. A UI stricter than the gate
// blocks legal requests; a UI looser than it offers a knob the backend will refuse.
//
// ⚠️ Reads the BASE `limits` directly — NOT `effectiveLimits(model, backend)`, unlike every sampler
// helper above. That is load-bearing, not an oversight. `effectiveLimits` REPLACES the whole block
// with `mlx.limits` / `candle.limits` when one exists, so a model carrying a per-backend sampler
// override would read as having no step menu here while the Rust gate — which reads top-level
// `limits.steps` lane-agnostically — still refuses off-menu counts. That is precisely the UI/gate
// desync sc-19502 fixed. The step count is a property of the distilled weights, not of the backend
// (both LTX lanes bake the identical schedule), so there is deliberately no per-lane form of this
// key; `both_ltx_lanes_agree_on_one_legal_step_count` asserts none appears.
export function stepsMenuFromModel(model) {
  const raw = model?.limits?.steps;
  if (!Array.isArray(raw)) return null;
  const menu = raw
    .map((value) => (typeof value === "number" ? value : Number.NaN))
    .filter((value) => Number.isInteger(value) && value > 0);
  return menu.length ? menu : null;
}

export function guidanceDefaultFromModel(model) {
  const value = Number(model?.defaults?.guidanceScale);
  return Number.isFinite(value) ? value : null;
}

// The Steps input's ceiling (sc-25679): the historical 80, raised to the model's own default when
// that is higher — Iris-3B ships 100 steps. Mirrors the worker's `max_requested_steps`, which clamps
// `advanced.steps` to the same bound, so the form never offers a count the worker would cut down.
export const BASE_MAX_STEPS = 80;

export function maxStepsForModel(model) {
  return Math.max(BASE_MAX_STEPS, stepsDefaultFromModel(model) ?? 0);
}

// Is the negative prompt inert for this model at this guidance (sc-25679)? A model declaring
// `image.negativePromptRequiresGuidance` treats the negative prompt as the CFG unconditional, so at
// guidance 1.0 (CFG off) its engine refuses one. `guidanceOverride` is the Studio's raw override
// ("" = use the model default). Absent declaration = never inert.
export function negativePromptInertAtGuidance(model, guidanceOverride) {
  if (model?.image?.negativePromptRequiresGuidance !== true) return false;
  return guidanceIsOff(model, guidanceOverride);
}

// Is classifier-free guidance OFF at this override (sc-25681)? Exactly 1.0 — the override, or the
// model default when the override is cleared — is CFG off (the engines' `uses_cfg` test).
export function guidanceIsOff(model, guidanceOverride) {
  const override = guidanceOverride === "" || guidanceOverride == null ? Number.NaN : Number(guidanceOverride);
  const effective = Number.isFinite(override) ? override : guidanceDefaultFromModel(model);
  return effective === 1;
}

// sc-25681: the image sampling-control declarations (manifest `image` block). Each ABSENT MEANS
// FALSE / NONE, so every model that declares nothing keeps its historical controls.
//
// A schedule-shift axis WITHOUT a scheduler menu (Iris-3B: one shifted time grid). The studio shows
// its Schedule shift control regardless of the scheduler picker and sends the value.
export function supportsSchedulerShiftAxis(model) {
  return model?.image?.supportsSchedulerShift === true;
}

export function supportsCfgIntervalAxis(model) {
  return model?.image?.supportsCfgInterval === true;
}

export const CAPTION_OVERFLOW_LABELS = {
  default: "Checkpoint default",
  warn: "Truncate and warn",
  error: "Refuse (error)",
  silent: "Truncate silently",
};

// The caption-overflow policies the model offers, in the declared order; `[]` hides the control.
export function captionOverflowOptions(model) {
  const declared = model?.image?.captionOverflowPolicies;
  if (!Array.isArray(declared)) return [];
  return declared.filter((policy) => ["warn", "error", "silent"].includes(policy));
}

// Resolve the studio's CFG-interval inputs (sc-25681). `start` / `end` are the raw input strings
// ("" = the model's own bound: 0 / 1). Returns `{ interval, error }`:
//   * `interval` — the `[start, end]` pair to send, or `null` to send nothing: the model offers no
//     interval, the full default interval is selected, or CFG is off (the engine refuses an
//     interval there, so the control is disabled and nothing is sent);
//   * `error` — a user-facing message when the inputs cannot form an interval (0 <= start < end <= 1).
export function resolveCfgInterval({ model, start, end, guidanceOverride }) {
  if (!supportsCfgIntervalAxis(model)) return { interval: null, error: null };
  if ((start === "" || start == null) && (end === "" || end == null)) {
    return { interval: null, error: null };
  }
  const lo = start === "" || start == null ? 0 : Number(start);
  const hi = end === "" || end == null ? 1 : Number(end);
  if (!(Number.isFinite(lo) && Number.isFinite(hi) && lo >= 0 && lo < hi && hi <= 1)) {
    return {
      interval: null,
      error: "Guidance interval must satisfy 0 ≤ start < end ≤ 1.",
    };
  }
  if (guidanceIsOff(model, guidanceOverride)) return { interval: null, error: null };
  if (lo === 0 && hi === 1) return { interval: null, error: null };
  return { interval: [lo, hi], error: null };
}

// The patch-grid hint (sc-25681): a model declaring `image.patchGridVisibleBelow` (Iris-3B at 1024)
// shows its patch grid below that side. Informational only — the size is never restricted.
export function patchGridHint(model, width, height) {
  const threshold = Number(model?.image?.patchGridVisibleBelow);
  if (!Number.isFinite(threshold) || threshold <= 0) return null;
  if (!(Number(width) < threshold || Number(height) < threshold)) return null;
  const patch = Number(model?.limits?.requiresDimensionsMultipleOf);
  const grid = Number.isFinite(patch) && patch > 0 ? `${patch}-px ` : "";
  return (
    `Below ${threshold}×${threshold} the released ${model?.name ?? "checkpoint"} shows a visible ` +
    `${grid}patch grid (upstream renders it too). Use ${threshold}px or larger per side for clean output.`
  );
}
