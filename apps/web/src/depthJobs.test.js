import { describe, expect, it } from "vitest";
import {
  buildDepthJobBody,
  depthAssetDetails,
  depthAvailability,
  depthProcessingSize,
  depthResolutionError,
  depthSettingsFromAsset,
} from "./depthJobs.js";

const caps = (supported) => ({ features: { imageDepth: { supported } } });
const installed = [{ id: "iris_3b_depth", installState: "installed" }];

describe("Iris depth contract (sc-25682)", () => {
  it("is available only with a native lane AND a complete install, and says why otherwise", () => {
    expect(depthAvailability(installed, caps(true))).toEqual({ available: true, reason: null });
    expect(depthAvailability(installed, caps(false)).available).toBe(false);
    expect(depthAvailability(installed, {}).available).toBe(false);
    expect(depthAvailability([], caps(true)).reason).toMatch(/not in the model catalog/);
    expect(depthAvailability([{ id: "iris_3b_depth", installState: "missing" }], caps(true)).reason).toMatch(/Install Iris 3B Depth/);
    expect(depthAvailability([{ id: "iris_3b_depth", installState: "incomplete" }], caps(true)).reason).toMatch(/incomplete/);
    // The generation install is not a depth install.
    expect(depthAvailability([{ id: "iris_3b", installState: "installed" }], caps(true)).available).toBe(false);
  });

  it("bounds the max side exactly like the API and always allows native", () => {
    expect(depthResolutionError({ mode: "capped", maxSide: 1024 })).toBeNull();
    expect(depthResolutionError({ mode: "native" })).toBeNull();
    expect(depthResolutionError({ mode: "capped", maxSide: 63 })).toMatch(/between 64 and 8192/);
    expect(depthResolutionError({ mode: "capped", maxSide: 8193 })).toMatch(/between 64 and 8192/);
    expect(depthResolutionError({ mode: "capped", maxSide: Number.NaN })).not.toBeNull();
  });

  it("shows the model input the worker will run (upstream size law, half-to-even)", () => {
    expect(depthProcessingSize(4000, 3000, { mode: "capped", maxSide: 1024 })).toEqual({ width: 1024, height: 768 });
    expect(depthProcessingSize(100, 50, { mode: "capped", maxSide: 1024 })).toEqual({ width: 96, height: 48 });
    expect(depthProcessingSize(1000, 750, { mode: "native" })).toEqual({ width: 992, height: 752 });
    expect(depthProcessingSize(24, 40, { mode: "native" })).toEqual({ width: 32, height: 32 });
    expect(depthProcessingSize(0, 10, { mode: "native" })).toBeNull();
  });

  it("builds the typed body and reopens a depth asset's recorded settings", () => {
    const body = buildDepthJobBody({ project: { id: "p", name: "P" }, requestedGpu: "auto", sourceAssetId: "a", resolution: { mode: "native" } });
    expect(body).toEqual({ projectId: "p", projectName: "P", sourceAssetId: "a", resolution: { mode: "native" }, requestedGpu: "auto" });
    expect(depthSettingsFromAsset({ recipe: { rawAdapterSettings: { depth: { resolution: { mode: "capped", maxSide: 512 } } } } })).toEqual({ mode: "capped", maxSide: 512 });
    expect(depthSettingsFromAsset({})).toEqual({ mode: "capped", maxSide: 1024 });
  });

  it("describes a reloaded depth asset from its sidecar", () => {
    const rows = Object.fromEntries(
      depthAssetDetails({
        type: "depth",
        lineage: { sourceAssetId: "asset_src" },
        depth: {
          valueConvention: "relative_log_depth",
          dtype: "float32",
          shape: [6, 4],
          npyDescr: "<f4",
          layout: "row-major",
          source: { assetId: "asset_src", width: 4, height: 6, storedWidth: 6, storedHeight: 4, exifOrientation: 6 },
          preprocessing: { resolution: { mode: "capped", maxSide: 1024 }, modelWidth: 16, modelHeight: 16, inputResample: "none", outputResample: "none" },
          model: { id: "iris_3b_depth", backend: "mlx", computeDtype: "bfloat16", weightsRepo: "speridlabs/iris-3b", weightsRevision: "7445443349bc9abe", weightsSubdir: "depth", configSha256: "ab".repeat(32) },
        },
      }),
    );
    expect(rows.Values).toMatch(/relative_log_depth.*not metric/);
    expect(rows["Raw map"]).toMatch(/float32 6×4 \(<f4, row-major\)/);
    expect(rows.Source).toBe("asset_src · 4×6");
    expect(rows["EXIF orientation"]).toMatch(/6 applied \(stored 6×4\)/);
    expect(rows.Resolution).toBe("Capped at 1024px max side");
    expect(rows.Model).toBe("iris_3b_depth (mlx, bfloat16)");
    expect(rows["Config sha256"]).toBe("ab".repeat(32));
    expect(depthAssetDetails({ type: "image" })).toEqual([]);
  });
});
