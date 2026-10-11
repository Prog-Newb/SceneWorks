import React, { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { DatasetEditorPanel } from "./DatasetEditorPanel.jsx";

// Build a full-but-inert prop bag; individual tests override only what they exercise.
// Empty `memberAssets` and a null Doctor report keep the caption grid and Dataset Doctor
// band in their empty states, so the panel renders without heavy fixtures.
function makeSessions(overrides = {}) {
  const onDeleteDataset = overrides.onDeleteDataset ?? vi.fn();
  const activeDataset =
    overrides.activeDataset === undefined
      ? { id: "ds-1", name: "Kelsie", version: 1, items: [] }
      : overrides.activeDataset;
  return {
    onDeleteDataset,
    props: {
      datasetSession: {
        loadingDatasets: false,
        onRefreshDatasets: vi.fn(),
        busyDatasetId: "",
        datasetThumbAsset: () => null,
        datasets: [{ id: "ds-1", name: "Kelsie", version: 1 }],
        startNewDataset: vi.fn(),
        openDataset: vi.fn(),
        activeDataset,
        selectedDatasetId: activeDataset?.id ?? "",
        datasetsError: "",
        datasetError: "",
        datasetMessage: "",
        draftName: activeDataset?.name ?? "",
        setDraftName: vi.fn(),
        dirty: false,
        discardDraft: vi.fn(),
        setAddDialogOpen: vi.fn(),
        renamePrefix: "item",
        setRenamePrefix: vi.fn(),
        renaming: false,
        memberAssets: [],
        applyOrderedNames: vi.fn(),
        setCaptionDialog: vi.fn(),
        health: { itemCount: 0, missingCaptions: 0, duplicateFilenames: 0, valid: false },
        canSave: false,
        saveValidity: { surfaced: [] },
        saveDataset: vi.fn(),
        savingDataset: false,
        unavailableAssetIds: [],
        removeUnavailableAsset: vi.fn(),
        onImportParquet: vi.fn(),
        onDeleteDataset,
        deletingDataset: overrides.deletingDataset ?? false,
      },
      captionSession: {
        captionDraftById: {},
        onPreview: vi.fn(),
        updateCaption: vi.fn(),
        captioning: false,
        addDialogOpen: false,
        selectedAssetIds: [],
        addAssets: vi.fn(),
        handleImport: vi.fn(),
        captionDialog: null,
        updateCaptionSetting: vi.fn(),
        runCaptionJob: vi.fn(),
        toggleCaptionExtraOption: vi.fn(),
        displayedCaptionPrompt: "",
        captionSettings: {},
        captionModelMissing: false,
        onDownloadCaptionModel: vi.fn(),
        captionModelSizeLabel: "",
        captionModelName: "JoyCaption",
      },
      doctorSession: {
        datasetDoctor: { report: null, loading: false },
        readinessByKey: new Map(),
        onToggleItemAck: vi.fn(),
      },
      config: {
        imageAssets: [],
        characters: [],
        associatedCharacterId: "",
        setActiveView: vi.fn(),
        importingAssets: false,
        gpuOptions: [],
      },
    },
  };
}

describe("DatasetEditorPanel delete affordance", () => {
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
  });

  function deleteButton() {
    return container.querySelector('button[aria-label="Delete dataset Kelsie"]');
  }

  it("renders a Delete button for a saved dataset and calls onDeleteDataset when clicked", async () => {
    const { onDeleteDataset, props } = makeSessions();
    await act(async () => {
      root.render(<DatasetEditorPanel {...props} />);
    });

    const button = deleteButton();
    expect(button).not.toBeNull();
    expect(button.textContent).toContain("Delete");
    expect(button.disabled).toBe(false);

    await act(async () => {
      button.dispatchEvent(new window.Event("click", { bubbles: true }));
    });
    expect(onDeleteDataset).toHaveBeenCalledTimes(1);
  });

  it("shows a busy, disabled Delete button while a delete is in flight", async () => {
    const { props } = makeSessions({ deletingDataset: true });
    await act(async () => {
      root.render(<DatasetEditorPanel {...props} />);
    });

    const button = deleteButton();
    expect(button).not.toBeNull();
    expect(button.disabled).toBe(true);
    expect(button.textContent).toContain("Deleting");
  });

  it("hides the Delete button when no saved dataset is open", async () => {
    const { props } = makeSessions({ activeDataset: null });
    await act(async () => {
      root.render(<DatasetEditorPanel {...props} />);
    });

    expect(deleteButton()).toBeNull();
    // Nothing dataset-scoped to delete, but the panel still renders its draft shell.
    expect(container.querySelector(".dataset-identity-actions")).not.toBeNull();
  });

  it("keeps Parquet-import failures visible and handled at the parent boundary", async () => {
    const onImportParquet = vi.fn(() => Promise.reject({ message: { detail: "not renderable" } }));
    const { props } = makeSessions();
    props.datasetSession.onImportParquet = onImportParquet;
    const unhandled = [];
    const onUnhandled = (event) => {
      unhandled.push(event.reason);
      event.preventDefault();
    };
    window.addEventListener("unhandledrejection", onUnhandled);

    try {
      await act(async () => root.render(<DatasetEditorPanel {...props} />));
      await act(async () => {
        [...container.querySelectorAll("button")].find((button) => button.textContent === "Import Parquet").click();
      });
      const input = [...document.body.querySelectorAll("label")]
        .find((label) => label.textContent.includes("Parquet file or folder"))
        .querySelector("input");
      await act(async () => {
        const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(input), "value")?.set;
        setter.call(input, "D:\\broken.parquet");
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });
      await act(async () => {
        [...document.body.querySelectorAll("button")].find((button) => button.textContent === "Start import").click();
        await Promise.resolve();
      });

      expect(onImportParquet).toHaveBeenCalledTimes(1);
      expect(container.textContent).toContain("Could not start the Parquet import.");
      expect(unhandled).toEqual([]);
    } finally {
      window.removeEventListener("unhandledrejection", onUnhandled);
    }
  });
});

// sc-25685: an item's named caption fields are visible (read-only) on its caption card, and the
// add dialog accepts `.json` caption sidecars.
describe("DatasetEditorPanel named caption fields", () => {
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
  });

  function withMembers(namedCaptionsById = {}) {
    const activeDataset = {
      id: "ds-1",
      name: "Kelsie",
      version: 1,
      items: [
        { id: "item_1", assetId: "asset-a", caption: { text: "a" }, captions: { short: "a", tags: "x" } },
        { id: "item_2", assetId: "asset-b", caption: { text: "b" } },
      ],
    };
    const { props } = makeSessions({ activeDataset });
    const members = [
      { id: "asset-a", type: "image", displayName: "a.png", file: { path: "a.png" } },
      { id: "asset-b", type: "image", displayName: "b.png", file: { path: "b.png" } },
    ];
    props.datasetSession.memberAssets = members;
    props.datasetSession.health = { itemCount: 2, missingCaptions: 0, duplicateFilenames: 0, valid: true };
    props.captionSession.selectedAssetIds = ["asset-a", "asset-b"];
    props.captionSession.namedCaptionsById = namedCaptionsById;
    props.config.imageAssets = members;
    return props;
  }

  const namedHints = () =>
    [...container.querySelectorAll(".training-caption-named-fields")].map((node) => node.textContent);

  it("lists a saved item's named caption fields", async () => {
    await act(async () => root.render(<DatasetEditorPanel {...withMembers()} />));
    expect(namedHints()).toEqual(["+2 caption fields: short, tags"]);
  });

  it("lists a freshly imported item's named caption fields", async () => {
    await act(async () => root.render(<DatasetEditorPanel {...withMembers({ "asset-b": { long: "b long" } })} />));
    expect(namedHints()).toEqual(["+2 caption fields: short, tags", "+1 caption field: long"]);
  });

  it("accepts .json caption sidecars in the add dialog", async () => {
    const props = withMembers();
    props.captionSession.addDialogOpen = true;
    await act(async () => root.render(<DatasetEditorPanel {...props} />));
    const input = [...document.body.querySelectorAll('input[type="file"]')].find((node) => node.getAttribute("accept")?.includes("image/*"));
    expect(input.getAttribute("accept")).toContain(".json");
    expect(input.getAttribute("accept")).toContain(".txt");
  });
});
