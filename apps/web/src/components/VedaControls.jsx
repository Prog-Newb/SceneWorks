import React from "react";

export function VedaControls({ value, onChange, steps, compatible = true }) {
  return <div className="lightning-toggle">
    <label className="checkline">
      <input type="checkbox" checked={value.enabled} onChange={(e) => onChange({ ...value, enabled: e.target.checked })} />
      Veda sparse attention
    </label>
    <p className="helper-copy">Default off; unavailable sparse execution reports dense fallback.</p>
    {value.enabled && !compatible ? <p className="helper-copy">Select MiniMax-H3 or turn Veda off.</p> : null}
    {value.enabled && compatible ? <>
      {[["generated", "Generated sparsity"], ["reference", "Reference sparsity"], ["fullAttentionLayers", "Full-attention layers"], ["fullAttentionSteps", "Full-attention steps"]].map(([key, label]) => <label key={key}>
        {label}<input aria-label={label} type="text" value={value[key]} onChange={(e) => onChange({ ...value, [key]: e.target.value })} />
      </label>)}
      <p className="helper-copy">Budgets: sparsity percentage (90%, below 100%) or positive key-tile count (32). Reference 0% keeps all keys.</p>
      <p className="helper-copy">Zero-based dense lists/ranges (0, 2–4). Layers 0–49; steps 0–{steps - 1}.</p>
    </> : null}
  </div>;
}
