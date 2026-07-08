import type GUI from "lil-gui";

// One demo scene = spawn (in the factory body) + optional per-frame animation + optional
// scene-specific GUI. The factory mutates the ECS world (and SunLight) at creation; demo.ts
// wires the returned hooks into its GUI/frame loop.
export type DemoScene = {
  // Initial camera framing, pixels per world unit (demo.ts default: 14).
  cameraZoom?: number;
  // Per-frame ECS mutation, called BEFORE prepare()/voxelize so the change re-voxelizes the
  // same frame and the GI follows it live.
  animate?: (now: number) => void;
  // Scene-specific GUI folders/controls (called after the shared voxel-GI folders).
  setupGUI?: (gui: GUI) => void;
};
