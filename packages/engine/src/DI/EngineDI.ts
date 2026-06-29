import type { EngineWorld } from "../ECS/createEngineWorld.ts";

export type EngineApi = {
  width: number;
  height: number;
  world: EngineWorld;
  sceneRoot: number;
  tick: (delta: number) => void;
  destroy: () => void;
  setRenderTarget: (canvas: HTMLCanvasElement | null | undefined) => void;
};

export const EngineDI: EngineApi = {} as EngineApi;
