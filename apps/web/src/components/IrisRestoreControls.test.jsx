import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { IrisRestoreControls } from "./IrisRestoreControls.jsx";

function setValue(el, value) {
  Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
}

const settle = () => act(async () => new Promise((resolve) => setTimeout(resolve, 260)));

function planResponse(items) {
  return { ok: true, status: 200, json: async () => ({ items }) };
}

describe("IrisRestoreControls (sc-25683)", () => {
  let container;
  let root;
  beforeEach(() => {
    global.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    container.remove();
    vi.restoreAllMocks();
  });

  function Harness({ initial = { scale: 4, inputSizing: "budgeted", colorFix: true }, planBody, onStatus }) {
    const [value, setValue_] = React.useState(initial);
    return <IrisRestoreControls onChange={setValue_} onStatus={onStatus} planBody={planBody} token="t" value={value} />;
  }

  it("shows the planner's processing + output dimensions before submission and reports ready", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      planResponse([
        {
          assetId: null,
          plan: { summary: "budgeted: 683x512 input → 2732x2048 (4× of the budgeted input; source 4032x3024, processed at 2736x2048 in 15 tiles)" },
          error: null,
        },
      ]),
    );
    const onStatus = vi.fn();
    await act(async () => root.render(<Harness onStatus={onStatus} planBody={{ width: 4032, height: 3024 }} />));
    await settle();
    expect(container.querySelector(".iris-restore-preview").textContent).toContain(
      "budgeted: 683x512 input → 2732x2048",
    );
    expect(container.textContent).not.toMatch(/of the original/);
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/api\/v1\/image-restoration\/plan$/);
    expect(JSON.parse(options.body)).toEqual({ width: 4032, height: 3024, scale: 4, inputSizing: "budgeted", colorFix: true });
    expect(onStatus).toHaveBeenLastCalledWith({ ready: true, blocked: false, error: null });
  });

  it("re-plans when a control changes (1× restore-only, original input, colour fix off)", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      planResponse([{ assetId: null, plan: { summary: "1× restore: 512x384 → 512x384 (processed at 1376x1024 in 2 tiles)" }, error: null }]),
    );
    await act(async () => root.render(<Harness planBody={{ width: 512, height: 384 }} />));
    await settle();
    const oneX = [...container.querySelectorAll("button")].find((b) => b.textContent === "1×");
    await act(async () => oneX.click());
    const original = [...container.querySelectorAll("label")].find((l) => l.textContent.includes("Original input")).querySelector("input");
    await act(async () => original.click());
    const colour = [...container.querySelectorAll("label")].find((l) => l.textContent.includes("Colour correction")).querySelector("input");
    await act(async () => colour.click());
    await settle();
    const last = JSON.parse(fetchMock.mock.calls.at(-1)[1].body);
    expect(last).toEqual({ width: 512, height: 384, scale: 1, inputSizing: "original", colorFix: false });
    expect(container.querySelector(".iris-restore-preview").textContent).toContain("1× restore: 512x384 → 512x384");
  });

  it("blocks submission on a refused plan and on an invalid scale (never substitutes one)", async () => {
    const fetchMock = vi.spyOn(global, "fetch").mockResolvedValue(
      planResponse([{ assetId: null, plan: null, error: "needs a 16128x12096 output, over the 8192px-per-side limit" }]),
    );
    const onStatus = vi.fn();
    await act(async () => root.render(<Harness onStatus={onStatus} planBody={{ width: 4032, height: 3024 }} />));
    await settle();
    expect(container.querySelector(".iris-restore-refused").textContent).toContain("Cannot restore");
    expect(onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ ready: false, blocked: true }));

    const calls = fetchMock.mock.calls.length;
    await act(async () => setValue(container.querySelector("input[type=number]"), "0"));
    await settle();
    expect(fetchMock.mock.calls.length).toBe(calls);
    expect(container.textContent).toContain("Enter a scale greater than 0.");
    expect(onStatus).toHaveBeenLastCalledWith(expect.objectContaining({ ready: false, blocked: true }));
  });
});
