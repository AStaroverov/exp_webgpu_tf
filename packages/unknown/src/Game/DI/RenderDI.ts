import type { createRadianceCascadesSystem } from "../ECS/Systems/Render/Lighting/createRadianceCascadesSystem.ts";

export const RenderDI: {
  enabled: boolean;
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context: GPUCanvasContext;
  lighting?: ReturnType<typeof createRadianceCascadesSystem>;
  destroy?: () => void;
  renderFrame?: (delta: number) => void;
} = {
  enabled: false,
  canvas: null as any,
  device: null as any,
  context: null as any,
  destroy: null as any,
  lighting: undefined,
  renderFrame: null as any,
};
