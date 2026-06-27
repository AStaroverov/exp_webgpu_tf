import type { createVoxelSystem } from "../../../renderer3d_2/src/ECS/Systems/Lighting/createVoxelSystem.ts";

// The VCT voxel-GI system handle (config + rebuild + setCellSize + …). Inferred from the
// factory so the demo/host can build a tuning GUI against it without re-declaring the surface.
export type VoxelSystem = ReturnType<typeof createVoxelSystem>;

// GPU handles + the render closure. Mirrors the unknown package's RenderDI,
// dropping the RC `lighting` field. `renderFrame` is OPTIONAL so the world can run
// headless (no canvas) — tick() calls RenderDI.renderFrame?.(delta).
type RenderDIShape = {
  enabled: boolean;
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context: GPUCanvasContext;
  renderFrame?: (delta: number) => void;
  destroy?: () => void;
  // Set by createRenderTarget; exposed so a host can build a VCT tuning GUI
  // (voxel.config + voxel.rebuild()). undefined when running headless.
  voxel?: VoxelSystem;
};

export const RenderDI: RenderDIShape = { enabled: false } as unknown as RenderDIShape;
