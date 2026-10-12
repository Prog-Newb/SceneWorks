import React from "react";

// Results come exclusively from the persisted provider receipt. Requested opt-in controls alone
// never imply sparse execution, including when replaying a recipe on a different device.
export function VedaOutcome({ outcome, requested, label = "Observed attention" }) {
  if (!outcome || typeof outcome !== "object") return null;
  const effective = { dense: "Dense", sparse: "Sparse", mixed: "Mixed sparse and dense", unobserved: "Unobserved" }[outcome.effective];
  if (!effective) return null;
  const fraction = outcome.totalAttentionFraction;
  const grid = (value) => Array.isArray(value) ? value.join(" × ") : null;
  return (
    <div className="veda-outcome" role="group" aria-label="Observed Veda execution">
      <strong>{label}: {effective}</strong>
      {outcome.kernel ? <div>Kernel: {outcome.kernel}</div> : null}
      {outcome.plan ? <div>Plan: {outcome.plan} ({outcome.planMatch}; trained {grid(outcome.planGrid)}, executed {grid(outcome.executionGrid)})</div> : null}
      {Number.isFinite(fraction) ? <div>Full attention computed: {(100 * fraction).toFixed(2)}%</div> : null}
      {outcome.fallbackReason ? <div>Dense fallback: {outcome.fallbackReason}</div> : null}
      {requested ? <div>Requested controls: <code>{JSON.stringify(requested)}</code></div> : null}
    </div>
  );
}

export function AssetVedaOutcome({ asset }) {
  const raw = asset?.recipe?.rawAdapterSettings;
  return <VedaOutcome outcome={raw?.vedaExecution} requested={raw?.veda} />;
}
