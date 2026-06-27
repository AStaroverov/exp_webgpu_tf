import type { EngineWorld } from "../ECS/createEngineWorld.ts";
import type { PhysicalWorld } from "../Physics/initPhysicalWorld.ts";

// World + physics + the frame entry point. A plain module-level mutable object
// (not a class), filled by createEngine.
export type EngineApi = {
  width: number;
  height: number;
  world: EngineWorld;
  physicalWorld: PhysicalWorld;
  tick: (delta: number) => void;
  destroy: () => void;
  setRenderTarget: (canvas: HTMLCanvasElement | null | undefined) => void;
};

export const EngineDI: EngineApi = {} as EngineApi;
