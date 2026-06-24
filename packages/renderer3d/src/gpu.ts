export async function initWebGPU(
  canvas: HTMLCanvasElement,
): Promise<{ device: GPUDevice; context: GPUCanvasContext }> {
  const adapter = await navigator.gpu.requestAdapter();
  if (adapter === null) throw new Error("No adapter found");

  // The surfel gather binds 10 storage buffers in one compute stage (posr/norw +
  // 7 scene-instance buffers + radiance cache), above the default 8. Request the
  // adapter's max (commonly 10) so it's allowed; clamps to whatever the adapter has.
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    },
  });
  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  canvas.width = canvas.clientWidth * window.devicePixelRatio;
  canvas.height = canvas.clientHeight * window.devicePixelRatio;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "premultiplied",
  });

  return {
    device,
    context,
  };
}
