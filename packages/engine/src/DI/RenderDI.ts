// GPU handles + the render closure. Mirrors the unknown package's RenderDI,
// dropping the RC `lighting` field. `renderFrame` is OPTIONAL so the world can run
// headless (no canvas) — tick() calls RenderDI.renderFrame?.(delta).
export const RenderDI: {
  enabled: boolean;
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context: GPUCanvasContext;
  renderFrame?: (delta: number) => void;
  destroy?: () => void;
} = { enabled: false } as unknown as {
  enabled: boolean;
  canvas: HTMLCanvasElement;
  device: GPUDevice;
  context: GPUCanvasContext;
  renderFrame?: (delta: number) => void;
  destroy?: () => void;
};
