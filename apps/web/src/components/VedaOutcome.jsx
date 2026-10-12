import React from "react";

// Provider receipts only: requested controls never imply that sparse execution occurred.
export function VedaOutcome({ outcome, requested, label = "Observed attention" }) {
  const effective = { dense: "Dense", sparse: "Sparse", mixed: "Mixed sparse and dense", unobserved: "Unobserved" }[outcome?.effective];
  if (!effective) return null;
  const grid = (value) => Array.isArray(value) ? value.join(" × ") : null;
  const facts = [
    ["Kernel", outcome.kernel],
    ["Plan", outcome.plan && `${outcome.plan} (${outcome.planMatch}; trained ${grid(outcome.planGrid)}, executed ${grid(outcome.executionGrid)})`],
    ["Full attention computed", Number.isFinite(outcome.totalAttentionFraction) ? `${(100 * outcome.totalAttentionFraction).toFixed(2)}%` : null],
    ["Dense fallback", outcome.fallbackReason],
  ];
  return <div className="veda-outcome" role="group" aria-label="Observed Veda execution">
    <strong>{label}: {effective}</strong>
    {facts.map(([name, value]) => value ? <div key={name}>{name}: {value}</div> : null)}
    {requested ? <div>Requested controls: <code>{JSON.stringify(requested)}</code></div> : null}
  </div>;
}

export function AssetVedaOutcome({ asset }) {
  const raw = asset?.recipe?.rawAdapterSettings;
  return <VedaOutcome outcome={raw?.vedaExecution} requested={raw?.veda} />;
}
